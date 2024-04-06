from typing import Literal, Callable
from os import PathLike, SEEK_END, SEEK_SET, remove
from tempfile import mktemp
from contextlib import suppress
from pathlib import Path
from re import match
from enum import StrEnum

import polars as pl

from .names import Study


ROW_INDEX_COL = "entry"


def participant_dataframe(p: PathLike | None = None):
    from .. import characteristics as c
    from .names import CharacteristicColumn as F
    from .schemas import PARTICIPANT_CHARACTERISTICS_SCHEMA as SCHEMA
    from .utils import get_dataset_root
    from .adjustments import get_record_offset

    p = Path(p) if p is not None else get_dataset_root()
    if p.suffix != ".csv":
        p = p / "participant_characteristics.csv"

    enums: list[tuple[F, type[StrEnum]]] = [
        (F.SETTING, c.Setting),
        (F.GENDER, c.Gender),
        (F.RACE, c.Race),
        (F.EYE_COLOR, c.EyeColor),
        (F.SKIN_COLOR, c.SkinColor),
        (F.FACIAL_HAIR, c.FacialHair),
        (F.VISION, c.Vision),
        (F.HANDEDNESS, c.Handedness),
        (F.WEATHER, c.Weather),
        (F.POINTING_DEVICE, c.PointingDevice),
    ]

    df = pl.scan_csv(
        p,
        has_header=True,
        separator=",",
        quote_char='"',
        null_values=["-", ""],
        schema={key.value: value[0] for key, value in SCHEMA.items()},
    )
    df = df.with_columns(
        # indices
        pl.col(F.PID).str.split("_").list.get(1).str.to_integer().cast(pl.UInt8),
        pl.col(F.LOG_ID).cast(pl.Datetime("ms")).alias("start_time"),
        # timestamps, dates, times, durations
        pl.col(F.DATE).str.to_date(r"%m/%d/%Y"),
        pl.col(F.DURATION)
        .str.split(":")
        .cast(pl.Array(pl.UInt16, 3))
        .apply(
            lambda arr: (arr[0] * 3600 + arr[1] * 60 + arr[2]) * 1000,
            return_dtype=pl.Int64,
        )
        .cast(pl.Duration("ms")),
        pl.col(F.TIME_OF_DAY).str.to_time(r"%H:%M"),
        pl.col(F.SCREEN_RECORDING_START_TIME_UNIX).cast(pl.Datetime("ms")),
        pl.col(F.SCREEN_RECORDING_START_TIME_UTC).str.to_datetime(
            r"%m/%d/%Y %H:%M",
            # time_zone="UTC",
        ),
        # boolean characteristics
        pl.col(F.TOUCH_TYPER)
        .cast(pl.String)
        .map_dict({"Yes": True, "No": False}, return_dtype=pl.Boolean),
        # enum characteristics
        *(
            pl.col(col).cast(pl.Enum([m.value for m in enum.__members__.values()]))
            for col, enum in enums
        ),
    )
    df = df.with_columns(
        # screen sizes
        screen=pl.struct(w=F.SCREEN_WIDTH, h=F.SCREEN_HEIGHT),
        display=pl.struct(w=F.DISPLAY_WIDTH, h=F.DISPLAY_HEIGHT),
        # screen recording start
        rec_time=pl.col("start_time")
        + pl.col(F.PID).map_elements(get_record_offset, pl.Duration("us")),
    )
    df = df.drop(
        F.SCREEN_WIDTH,
        F.SCREEN_HEIGHT,
        F.DISPLAY_WIDTH,
        F.DISPLAY_HEIGHT,
        F.DATE,
        F.DURATION,
        F.SCREEN_RECORDING_START_TIME_UTC,
        F.SCREEN_RECORDING_START_TIME_UNIX,
    )
    return df.sort(by=F.PID).rename(
        {
            key.value: value[1]
            for key, value in SCHEMA.items()
            if key.value in df.columns
        }
    )


def log_dataframe(src: Path):
    from .schemas import LOG_SCHEMA as SCHEMA
    from .names import LogField as F, Event as E, Type as T
    from .participant import pid_from_path
    from ..characteristics import Source as S

    type_map = {
        T.MOUSE_CLICK: "click",
        T.MOUSE_MOVE: "move",
        T.REC_START: "start",
        T.REC_STOP: "stop",
        T.SCROLL_EVENT: "scroll",
        T.TEXT_INPUT: "input",
        T.TEXT_SUBMIT: "text",
    }

    df = pl.read_json(
        src, schema={k: v for k, v in SCHEMA.items()}, infer_schema_length=0
    ).lazy()
    df = df.with_columns(
        pl.col(F.WEBPAGE)
        .str.strip_prefix("/study/")
        .str.strip_suffix(".htm")
        .replace("thankyou", "thank_you")
        .cast(pl.Enum(Study.values())),
        pl.col(F.TYPE).cast(pl.Categorical("lexical")),
        pl.col(F.EVENT).cast(pl.Categorical("lexical")),
        pl.col(F.SESSION_ID).str.split("_").list.get(0).cast(pl.UInt64),
        pl.col(F.SESSION_ID)
        .str.split("_")
        .list.get(1)
        .cast(pl.UInt8)
        .alias(F.SESSION_STRING),
        pl.col(F.EPOCH).cast(pl.Datetime("ms")),
        pl.col(F.TIME).cast(pl.Duration("ms")),
    )
    df = df.with_columns(
        pl.when(pl.col(F.TYPE).is_not_null())
        .then(pl.col(F.TYPE).map_dict(type_map))
        .when(pl.col(F.EVENT).is_not_null() & pl.col(F.EVENT).eq(E.VIDEO_SAVE))
        .then(pl.lit("save"))
        .otherwise(None)
        .cast(pl.Categorical("lexical"))
        .alias(F.EVENT),
        pl.when(
            pl.col(F.TYPE).is_in([T.REC_START, T.REC_STOP])
            | pl.col(F.EVENT).is_in([E.VIDEO_START, E.VIDEO_STOP, E.VIDEO_SAVE])
        )
        .then(pl.lit(S.LOG))
        .when(pl.col(F.TYPE).is_in([T.MOUSE_MOVE, T.MOUSE_CLICK]))
        .then(pl.lit(S.MOUSE))
        .when(pl.col(F.TYPE) == T.SCROLL_EVENT)
        .then(pl.lit(S.SCROLL))
        .when(pl.col(F.TYPE) == T.TEXT_INPUT)
        .then(pl.lit(S.INPUT))
        .when(pl.col(F.TYPE) == T.TEXT_SUBMIT)
        .then(pl.lit(S.TEXT))
        .otherwise(pl.lit(S.LOG))
        .cast(pl.Categorical("lexical"))
        .alias(F.TYPE),
    )
    return df.sort(F.EPOCH).select(
        pid=pl.lit(pid_from_path(src), pl.UInt8),
        record=F.SESSION_STRING,
        timestamp=pl.col(F.EPOCH),  # - pl.col(F.SESSION_ID).cast(pl.Datetime("ms")),
        study=F.WEBPAGE,
        event=F.EVENT,
        source=F.TYPE,
        trusted=F.IS_TRUSTED,
        duration=F.TIME,
        caret=F.POS,
        text=F.TEXT,
        page=pl.struct(x=F.PAGE_X, y=F.PAGE_Y),
        mouse=pl.struct(x=F.SCREEN_X, y=F.SCREEN_Y),
        scroll=pl.struct(x=F.SCROLL_X, y=F.SCROLL_Y),
        window=pl.struct(x=F.WINDOW_X, y=F.WINDOW_Y),
        inner=pl.struct(w=F.WINDOW_INNER_WIDTH, h=F.WINDOW_INNER_HEIGHT),
        outer=pl.struct(w=F.WINDOW_OUTER_WIDTH, h=F.WINDOW_OUTER_HEIGHT),
    )


def log_dataset(*, root: PathLike | None = None):
    from .participant import glob_log_files as glob_files

    return pl.concat(map(log_dataframe, glob_files(root)))


def tobii_dataframe(src: Path):
    from .schemas import TOBII_SCHEMA as SCHEMA
    from .names import TobiiField as F
    from .participant import pid_from_path

    pid = pid_from_path(src)

    # handle corrupted files
    # polars cannot ignore parsing errors: https://github.com/pola-rs/polars/issues/13768
    with open(src, "rb") as f:
        f.seek(-2, SEEK_END)
        if f.peek(2) != b"}\n":
            f.seek(0, SEEK_SET)
            src = b"".join(f.readlines()[:-1])

    if isinstance(src, Path):
        df = pl.scan_ndjson(
            src, schema=SCHEMA, infer_schema_length=1, ignore_errors=True
        )
    else:
        df = pl.read_ndjson(src, schema=SCHEMA, ignore_errors=True).lazy()

    xyz = ("x", "y", "z")
    xy = ("x", "y")

    df = df.with_columns(
        pl.col(F.TRUE_TIME).mul(1e9).cast(pl.Datetime("ns")),
        pl.col(F.DEVICE_TIME_STAMP).cast(pl.Duration("us")),
        pl.col(F.SYSTEM_TIME_STAMP).cast(pl.Datetime("us")),
        pl.col(F.LEFT_PUPIL_VALIDITY).cast(pl.Boolean),
        pl.col(F.RIGHT_PUPIL_VALIDITY).cast(pl.Boolean),
        pl.col(F.LEFT_GAZE_ORIGIN_VALIDITY).cast(pl.Boolean),
        pl.col(F.RIGHT_GAZE_ORIGIN_VALIDITY).cast(pl.Boolean),
        pl.col(F.LEFT_GAZE_POINT_VALIDITY).cast(pl.Boolean),
        pl.col(F.RIGHT_GAZE_POINT_VALIDITY).cast(pl.Boolean),
        pl.col(F.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM).arr.to_struct(xyz),
        pl.col(F.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM).arr.to_struct(xyz),
        pl.col(F.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
        pl.col(F.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
        pl.col(F.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
        pl.col(F.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
        pl.col(F.LEFT_GAZE_POINT_ON_DISPLAY_AREA).arr.to_struct(xy),
        pl.col(F.RIGHT_GAZE_POINT_ON_DISPLAY_AREA).arr.to_struct(xy),
    )
    return df.sort(F.TRUE_TIME).select(
        pid=pl.lit(pid, pl.UInt8),
        timestamp=F.TRUE_TIME,
        clock=F.SYSTEM_TIME_STAMP,
        duration=F.DEVICE_TIME_STAMP,
        # pupil=pl.struct(
        pupil_validity=pl.struct(
            left=F.LEFT_PUPIL_VALIDITY, right=F.RIGHT_PUPIL_VALIDITY
        ),
        pupil_diameter=pl.struct(
            left=F.LEFT_PUPIL_DIAMETER, right=F.RIGHT_PUPIL_DIAMETER
        ),
        # ),
        # gaze=pl.struct(
        # point=pl.struct(
        gazepoint_validity=pl.struct(
            left=F.LEFT_GAZE_POINT_VALIDITY,
            right=F.RIGHT_GAZE_POINT_VALIDITY,
        ),
        gazepoint_display=pl.struct(
            left=F.LEFT_GAZE_POINT_ON_DISPLAY_AREA,
            right=F.RIGHT_GAZE_POINT_ON_DISPLAY_AREA,
        ),
        gazepoint_ucs=pl.struct(
            left=F.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM,
            right=F.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM,
        ),
        # ),
        # origin=pl.struct(
        gazeorigin_validity=pl.struct(
            left=F.LEFT_GAZE_ORIGIN_VALIDITY,
            right=F.RIGHT_GAZE_ORIGIN_VALIDITY,
        ),
        gazeorigin_trackbox=pl.struct(
            left=F.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM,
            right=F.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM,
        ),
        gazeorigin_ucs=pl.struct(
            left=F.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM,
            right=F.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM,
        ),
        # ),
        # ),
    )


def tobii_dataset(*, root: PathLike | None = None):
    from .participant import glob_tobii_files as glob_files

    return pl.concat(map(tobii_dataframe, glob_files(root)))


def webcam_dataframe(
    root: PathLike | None = None, callback: Callable[[], None] | None = None
):
    from .utils import ffmpeg
    from .participant import parse_webcam_filename, glob_webcam_files, pid_from_name

    if not callable(callback):
        callback = lambda: None

    schema = {
        "pid": pl.UInt8,
        "log": pl.UInt64,
        "record": pl.UInt8,
        "study": pl.Categorical(),
        "aux": pl.UInt8,
        "path": pl.String,
    }

    def path_to_row(path: Path):
        pid = pid_from_name(path.parent.name)
        assert pid is not None

        log, record, study, aux = parse_webcam_filename(path.stem)
        study = Study.from_id(study)

        return pid, log, record, study, aux, str(path)

    data = map(path_to_row, glob_webcam_files(root))
    df = pl.LazyFrame(data, schema, orient="row")

    def path_to_video(src: str) -> bytes:
        try:
            dst = mktemp(".mp4", f"eott_{Path(src).stem}_")
            ffmpeg(src, dst, fps_mode="passthrough")
            with open(dst, "rb") as fp:
                result = fp.read()
        finally:
            with suppress(FileNotFoundError):
                remove(dst)
        callback()
        return result

    df = df.with_columns(file=pl.col("path").map_elements(path_to_video, pl.Binary))
    return df.drop("path")


def screen_dataframe(
    root: PathLike | None = None, callback: Callable[[], None] | None = None
):
    from .utils import ffmpeg
    from .participant import glob_screen_files, pid_from_name

    if not callable(callback):
        callback = lambda: None

    schema = {
        "pid": pl.UInt8,
        "path": pl.String,
    }

    def path_to_row(path: Path):
        pid = pid_from_name(path.parent.name)
        assert pid is not None
        return pid, str(path)

    data = map(path_to_row, glob_screen_files(root))
    df = pl.LazyFrame(data, schema, orient="row")

    def path_to_video(src: str) -> bytes:
        p = Path(src)

        if p.suffix == ".flv":
            params = dict(fps_mode="passthrough")
        elif p.suffix == ".mov":
            params = {"filter:v": "fps=25,scale=iw/2:ih/2"}
        else:
            params = {}

        try:
            dst = mktemp(".mp4", f"eott_screen_{p.stem}_")
            ffmpeg(src, dst, **params)
            with open(dst, "rb") as fp:
                result = fp.read()
        finally:
            with suppress(FileNotFoundError):
                remove(dst)
        callback()
        return result

    df = df.with_columns(file=pl.col("path").map_elements(path_to_video, pl.Binary))
    return df.drop("path")


def dot_dataframe(src: Path):
    from .schemas import DOTTEST_SCHEMA as SCHEMA
    from .names import DotColumn as C
    from .participant import pid_from_path

    df = pl.scan_csv(
        src,
        has_header=True,
        separator="\t",
        schema={key.value: value for key, value in SCHEMA.items()},
        ignore_errors=True,
        infer_schema_length=0,
    )
    return df.sort(C.EPOCH).select(
        pid=pl.lit(pid_from_path(src), pl.UInt8),
        timestamp=pl.col(C.EPOCH).cast(pl.Datetime("ms")),
        dot=pl.struct(x=C.DOT_X, y=C.DOT_Y),
    )


def dot_dataset(root: PathLike | None = None):
    from .participant import glob_dot_files

    return pl.concat(map(dot_dataframe, glob_dot_files(root)))


def calibration_dataframe(src: Path):
    from .schemas import SPEC_SCHEMA as SCHEMA
    from .names import SpecColumn as C
    from .participant import pid_from_path

    def xy(name: str):
        return pl.struct(**{c: f"{name}_{c}" for c in ("x", "y")})

    def lr(name: str):
        return pl.struct(**{c: f"{name}_{c}" for c in ("left", "right")})

    def pred(name: str):
        return pl.struct(**{c: f"prediction_{c}_{name}" for c in ("x", "y")})

    df = pl.scan_csv(
        src,
        has_header=True,
        separator="\t",
        schema={key.value: value for key, value in SCHEMA.items()},
        ignore_errors=True,
    )
    df = df.drop_nulls()
    df = df.with_columns(pl.col(C.VALIDITY_LEFT) > 0, pl.col(C.VALIDITY_RIGHT) > 0)
    return df.select(
        pid=pl.lit(pid_from_path(src), pl.UInt8),
        point=xy("point"),
        validity=lr("validity"),
        left=pred("left"),
        right=pred("right"),
    )


def calibration_dataset(root: PathLike | None = None):
    from .participant import glob_specs_files

    return pl.concat(map(calibration_dataframe, glob_specs_files(root)))


def read_tobii_specs(p: PathLike):
    tracking_box: dict[
        tuple[
            Literal["back", "front"],
            Literal["lower", "upper"],
            Literal["left", "right"],
        ],
        tuple[float, float, float],
    ] = {}
    ilumination_mode: str | None = None
    frequency: float | None = None

    with open(p, "r") as file:
        for line in file:
            m = match(r"(Back|Front) (Lower|Upper) (Left|Right):", line)
            if m is not None:
                index = (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())

                values = line[m.end() :].strip(" ()\n\t").split(", ")
                x, y, z = [float(v) for v in values]

                tracking_box[index] = (x, y, z)

            m = match(r"Illumination mode:", line)
            if m is not None:
                ilumination_mode = line[m.end() :].strip()

            m = match(r"Initial gaze output frequency: (\d+\.?\d+)", line)
            if m is not None:
                frequency = float(m.group(1))

    return tracking_box, ilumination_mode, frequency


def trackbox_dataframe(src: Path):
    from .participant import pid_from_path

    schema = {
        "position": pl.Struct(
            {"z": pl.Categorical(), "y": pl.Categorical(), "x": pl.Categorical()}
        ),
        "point": pl.Array(pl.Float64, 3),
    }
    data = read_tobii_specs(src)[0]
    data = [
        [{"z": az, "y": ay, "x": ax}, (px, py, pz)]
        for (az, ay, ax), (px, py, pz) in data.items()
    ]

    df = pl.LazyFrame(data, schema, orient="row")
    return df.with_columns(pid=pl.lit(pid_from_path(src), pl.UInt8))


def trackbox_dataset(root: PathLike | None = None):
    from .participant import glob_specs_files

    with pl.StringCache():
        return pl.concat(map(trackbox_dataframe, glob_specs_files(root)))
