from typing import Literal, Callable
from io import IOBase
from os import PathLike, SEEK_END, SEEK_SET
from pathlib import Path
from re import match
from io import IOBase
from enum import StrEnum

import polars as pl

from .study import Study

import polars as pl

from .study import Study
from .data import *


ROW_INDEX_COL = "entry"


def participant_dataframe(p: PathLike | None = None):
    from . import characteristics as c
    from .names import CharacteristicColumn as F
    from .schemas import PARTICIPANT_CHARACTERISTICS_SCHEMA as SCHEMA
    from .utils import get_dataset_root

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
        pl.col(F.PID.value).str.split("_").list.get(1).str.to_integer().cast(pl.UInt8),
        pl.col(F.LOG_ID).cast(pl.Datetime("ms")).alias("start_time"),
        # timestamps, dates, times, durations
        pl.col(F.DATE.value).str.to_date(r"%m/%d/%Y"),
        pl.col(F.DURATION.value)
        .str.split(":")
        .cast(pl.Array(pl.UInt16, 3))
        .apply(
            lambda arr: (arr[0] * 3600 + arr[1] * 60 + arr[2]) * 1000,
            return_dtype=pl.Int64,
        )
        .cast(pl.Duration("ms")),
        pl.col(F.TIME_OF_DAY.value).str.to_time(r"%H:%M"),
        pl.col(F.SCREEN_RECORDING_START_TIME_UNIX.value).cast(pl.Datetime("ms")),
        pl.col(F.SCREEN_RECORDING_START_TIME_UTC.value).str.to_datetime(
            r"%m/%d/%Y %H:%M",
            # time_zone="UTC",
        ),
        # boolean characteristics
        pl.col(F.TOUCH_TYPER.value)
        .cast(pl.String)
        .map_dict({"Yes": True, "No": False}, return_dtype=pl.Boolean),
        # enum characteristics
        *(
            pl.col(col).cast(pl.Enum([m.value for m in enum.__members__.values()]))
            for col, enum in enums
        ),
        # screen sizes
        screen=pl.struct(w=F.SCREEN_WIDTH, h=F.SCREEN_HEIGHT),
        display=pl.struct(w=F.DISPLAY_WIDTH, h=F.DISPLAY_HEIGHT),
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
    df = df.sort(by=F.PID)
    df = df.rename(
        {
            key.value: value[1]
            for key, value in SCHEMA.items()
            if key.value in df.columns
        }
    )
    return df


def with_pid_and_frame_column(df: pl.LazyFrame, value: int):
    df = df.with_row_index(name=ROW_INDEX_COL)
    return df.with_columns(pid=pl.lit(value, pl.UInt8))


def _log_df(df: pl.LazyFrame):
    return df.select("pid", ROW_INDEX_COL, "index", "study", "event", "epoch", "time")


def _scroll_df(df: pl.LazyFrame):
    df = df.filter(event="scroll")
    return df.select("pid", ROW_INDEX_COL, scroll=pl.struct(x="scroll_x", y="scroll_y"))


def _mouse_df(df: pl.LazyFrame):
    df = df.filter(event="mouse")
    return df.select(
        "pid",
        ROW_INDEX_COL,
        screen=pl.struct(x="screen_x", y="screen_y"),
        client=pl.struct(x="client_x", y="client_y"),
        window=pl.struct(x="window_x", y="window_y"),
        page=pl.struct(x="page_x", y="page_y"),
        inner=pl.struct(w="inner_width", h="inner_height"),
        outer=pl.struct(w="outer_width", h="outer_height"),
    )


def event_dataframes(src: Path, pid: int):
    from .schemas import LOG_SCHEMA as SCHEMA
    from .names import LogField as F

    event_map = {
        "recording start": "start",
        "mousemove": "mouse",
        "scrollEvent": "scroll",
        "mouseclick": "click",
        "recording stop": "stop",
        "textInput": "text",
        "text_input": "text",
    }

    df = pl.read_json(src)

    with pl.StringCache():
        df = df.with_columns(
            (
                pl.col(key).cast(value[0])
                if key.value in df.columns
                else pl.lit(None, value[0]).alias(key)
            )
            for key, value in SCHEMA.items()
        )
        df = df.with_columns(
            pl.col(F.SESSION_ID)
            .str.split("/")
            .list.get(-1)
            .replace("thankyou", "thank_you")
            .cast(pl.Categorical)
            .alias("study"),
            pl.col(F.TYPE).map_dict(event_map, return_dtype=pl.Categorical),
            pl.col(F.SESSION_ID)
            .str.split("_")
            .list.get(0)
            .cast(pl.UInt64)
            .alias("log_id"),
            pl.col(F.SESSION_ID)
            .str.split("_")
            .list.get(1)
            .cast(pl.UInt8)
            .alias("index"),
            pl.col(F.EPOCH).cast(pl.Datetime("ms")),
            pl.col(F.TIME).cast(pl.Duration("ms")),
        )

    df = df.drop(
        key.value
        for key, value in SCHEMA.items()
        if key.value in df.columns and value[1] is None
    )
    df = df.rename(
        {
            key.value: value[1]
            for key, value in SCHEMA.items()
            if key.value in df.columns and value[1] is not None
        }
    )
    df = df.lazy().sort(by="epoch")
    df = with_pid_and_frame_column(df, pid)
    return _log_df(df).collect(), _scroll_df(df).collect(), _mouse_df(df).collect()


def tobii_dataframe(src: Path, pid: int):
    from .schemas import TOBII_SCHEMA as SCHEMA
    from .names import TobiiField as F

    def get_columns():
        sides = ("left", "right")
        names = ("pupil", "gaze_origin", "gaze_point")
        suffixes = (
            "on_display_area",
            "in_user_coordinate_system",
            "validity",
            "diameter",
        )

        suffix_rename = {
            "on_display_area": "display",
            "in_user_coordinate_system": "ucs",
        }

        def get_key(name: str, suffix: str):
            name = name.replace("_", "")
            suffix = suffix_rename.get(suffix, suffix)
            return f"{name}_{suffix}"

        def get_value(name: str, suffix: str):
            return pl.struct(**{side: f"{side}_{name}_{suffix}" for side in sides})

        def check(name: str, suffix: str):
            match (name, suffix):
                case ("pupil", s) if "_" in s:
                    return False
                case (n, "diameter") if n.startswith("gaze"):
                    return False
                case (n, s) if "origin" in n and "display" in s:
                    return False
                case (n, s) if "point" in n and "system" in s:
                    return False
                case _:
                    return True

        return {
            get_key(name, suffix): get_value(name, suffix)
            for name in names
            for suffix in suffixes
            if check(name, suffix)
        }

    # handle corrupted files
    # polars cannot ignore parsing errors: https://github.com/pola-rs/polars/issues/13768
    with open(src, "rb") as f:
        f.seek(-2, SEEK_END)
        if f.peek(2) != b"}\n":
            f.seek(0, SEEK_SET)
            src = b"".join(f.readlines()[:-1])

    if isinstance(src, Path):
        df = pl.scan_ndjson(src, infer_schema_length=1)
    else:
        df = pl.read_ndjson(src).lazy()

    df = df.with_columns(pl.col(key).cast(value) for key, value in SCHEMA.items())
    df = df.with_columns((pl.col(F.TRUE_TIME) * 1e9).cast(pl.Datetime("ns")))
    df = with_pid_and_frame_column(df, pid)
    df = df.rename(
        {
            f"{name}_time_stamp": name.replace("e_s", "es")
            for name in ("device", "system")
        }
    )
    df = df.with_columns(
        *(
            pl.col(key).arr.to_struct(
                ["x", "y", "z"] if value.width == 3 else ["x", "y"]
            )
            for key, value in df.schema.items()
            if value.base_type() is pl.Array
        )
    )
    return df.select("pid", ROW_INDEX_COL, **get_columns())


def webcam_dataframe(root: PathLike | None = None, callback: Callable[[], None] | None = None):
    from .utils import fix_webcam_video
    from .participant import parse_webcam_filename, glob_webcam_files, pid_from_name
    schema = {
        "pid": pl.UInt8,
        "log": pl.UInt64,
        "index": pl.UInt8,
        "study": pl.Categorical(),
        "aux": pl.UInt8,
        "path": pl.String,
    }

    if not callable(callback):
        callback = lambda: None

    def path_to_row(path: Path):
        pid = pid_from_name(path.parent.name)
        assert pid is not None

        log, index, study, aux = parse_webcam_filename(path.stem)
        study = Study.from_position(study)

        return pid, log, index, study, aux, str(path.expanduser().absolute())

    data = map(path_to_row, glob_webcam_files(root))
    df = pl.LazyFrame(data, schema, orient="row")

    def udf(value: str):
        result = fix_webcam_video(value)
        callback()
        return result

    df = df.with_columns(video=pl.col("path").map_elements(udf, pl.Binary))
    return df.drop("path")


def dot_dataframe(src: Path, pid: int):
    from .schemas import DOTTEST_SCHEMA as SCHEMA
    from .names import DotColumn as C

    df = pl.scan_csv(
        src,
        has_header=True,
        separator="\t",
        schema={key.value: value for key, value in SCHEMA.items()},
        ignore_errors=True,
    )
    df = df.select(
        epoch=pl.col(C.EPOCH).cast(pl.Datetime("ms")),
        dot=pl.struct(x=C.DOT_X, y=C.DOT_Y),
    )
    return with_pid_and_frame_column(df, pid)


def calibration_dataframe(src: Path, pid: int):
    from .schemas import SPEC_SCHEMA as SCHEMA
    from .names import SpecColumn as C

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
    df = with_pid_and_frame_column(df, pid)
    return df.select(
        "pid",
        ROW_INDEX_COL,
        point=xy("point"),
        validity=lr("validity"),
        left=pred("left"),
        right=pred("right"),
    )


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


def trackbox_dataframe(src: PathLike | bytes | IOBase, pid: int):
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

    with pl.StringCache():
        df = pl.LazyFrame(data, schema, orient="row")

    return with_pid_and_frame_column(df, pid)
