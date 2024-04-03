from typing import TYPE_CHECKING, Optional, Callable, Union
from io import IOBase
from typing import Literal
from os import PathLike
from pathlib import Path
from re import match
from datetime import datetime

import polars as pl

from .study import Study
from .utils import get_dataset_root, count_video_frames, get_video_fps

if TYPE_CHECKING:
    from .participant import Participant


__all__ = [
    "read_participant_characteristics",
    "read_user_interaction_logs",
    "read_tobii_gaze_predictions",
    "read_tobii_calibration_points",
    "read_tobii_specs",
    "read_dottest_locations",
    "get_timeline",
    "Source",
]


EVENTMAP = {
    "recording start": "start",
    "mousemove": "mouse",
    "scrollEvent": "scroll",
    "mouseclick": "click",
    "recording stop": "stop",
    "textInput": "text",
    "text_input": "text",
}


def read_participant_characteristics(p: PathLike | None = None) -> pl.DataFrame:
    from .names import CharacteristicColumn as F
    from .schemas import PARTICIPANT_CHARACTERISTICS_SCHEMA as SCHEMA

    p = Path(p) if p is not None else get_dataset_root()

    if p.suffix != ".csv":
        p = p / "participant_characteristics.csv"

    df = pl.read_csv(
        p,
        has_header=True,
        columns=[name.value for name in F.__members__.values()],
        separator=",",
        quote_char='"',
        null_values=["-", ""],
        n_threads=1,
        schema={key.value: value[0] for key, value in SCHEMA.items()},
    )

    df = df.with_columns(
        pl.col(F.PID.value).str.split("_").list.get(1).str.to_integer().cast(pl.UInt8),
        pl.col(F.LOG_ID).cast(pl.Datetime("ms")).alias("start_time"),
        pl.col(F.DATE.value).str.to_date(r"%m/%d/%Y"),
        pl.col(F.SCREEN_RECORDING_START_TIME_UNIX.value).cast(pl.Datetime("ms")),
        pl.col(F.SCREEN_RECORDING_START_TIME_UTC.value).str.to_datetime(
            r"%m/%d/%Y %H:%M",
            # time_zone="UTC",
        ),
        pl.col(F.TOUCH_TYPER.value)
        .cast(pl.String)
        .map_dict({"Yes": True, "No": False}, return_dtype=pl.Boolean),
        pl.col(F.TIME_OF_DAY.value).str.to_time(r"%H:%M"),
        pl.col(F.DURATION.value)
        .str.split(":")
        .cast(pl.Array(pl.UInt16, 3))
        .apply(
            lambda arr: (arr[0] * 3600 + arr[1] * 60 + arr[2]) * 1000,
            return_dtype=pl.Int64,
        )
        .cast(pl.Duration("ms")),
    )
    df = df.rename({key.value: value[1] for key, value in SCHEMA.items()})

    return df.sort(by="pid")


def read_user_interaction_logs(p: PathLike | bytes | IOBase):
    from .schemas import LOG_SCHEMA as SCHEMA
    from .names import LogField as F

    df = pl.read_json(p)

    with pl.StringCache():
        df = df.with_columns(
            pl.col(key).cast(value[0])
            for key, value in SCHEMA.items()
            if key.value in df.columns
        )
        df = df.with_columns(
            pl.col(F.SESSION_ID)
            .str.split("/")
            .list.get(-1)
            .replace("thankyou", "thank_you")
            .cast(pl.Categorical)
            .alias("study"),
            pl.col(F.TYPE).map_dict(EVENTMAP, return_dtype=pl.Categorical),
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

    return df.sort(by="epoch")


def read_tobii_gaze_predictions(p: PathLike):
    from .schemas import TOBII_SCHEMA as SCHEMA
    from .names import TobiiField as F

    try:
        df = pl.read_ndjson(p)
    except RuntimeError:
        # handle corrupted files
        # polars cannot ignore parsing errors: https://github.com/pola-rs/polars/issues/13768
        with open(p, "rb") as fp:
            lines = fp.readlines()
            df = pl.read_ndjson(b"".join(lines[:-1]))
        
    df = df.with_columns(pl.col(key).cast(value) for key, value in SCHEMA.items())
    return df.with_columns((pl.col(F.TRUE_TIME) * 1e9).cast(pl.Datetime("ns")))


def read_tobii_calibration_points(p: PathLike):
    from .schemas import SPEC_SCHEMA as SCHEMA
    from .names import SpecColumn as C

    df = pl.read_csv(
        p,
        has_header=True,
        columns=[name.value for name in C.__members__.values()],
        separator="\t",
        n_threads=1,
        schema={key.value: value for key, value in SCHEMA.items()},
        ignore_errors=True,
    )
    df = df.drop_nulls()
    return df.with_columns(pl.col(C.VALIDITY_LEFT) > 0, pl.col(C.VALIDITY_RIGHT) > 0)


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


def read_dottest_locations(p: PathLike):
    from .schemas import DOTTEST_SCHEMA as SCHEMA
    from .names import DotColumn as C

    df = pl.read_csv(
        p,
        has_header=True,
        columns=[name.value for name in C.__members__.values()],
        separator="\t",
        n_threads=1,
        schema={key.value: value for key, value in SCHEMA.items()},
        ignore_errors=True,
    )

    return df.with_columns(pl.col(C.EPOCH).cast(pl.Datetime("ms")))


Source = Literal["log", "webcam", "tobii", "screen", "dot"]
SourceType = pl.Enum(["log", "webcam", "tobii", "screen", "dot"])
StudyType = pl.Enum([study.value for study in Study.__members__.values()])


def _get_tobii_timeline(p: "Participant") -> pl.DataFrame:
    df = p.tobii_gaze_predictions.select("true_time")
    df = df.with_columns(
        index=pl.lit(None, pl.UInt8),
        study=pl.lit(None, StudyType),
        source=pl.lit("tobii", SourceType),
    )
    df = df.with_columns(offset=(pl.col("true_time") - p.start_time))
    df = df.with_columns(pl.col("offset").cast(pl.Duration("us")))
    return df.drop("true_time").with_row_index("frame")


def _get_log_timeline(p: "Participant") -> pl.DataFrame:
    df = p.user_interaction_logs.select("epoch", "index", "study")
    df = df.with_columns(pl.col("study").cast(StudyType))
    df = df.with_columns(source=pl.lit("log", SourceType))
    df = df.with_columns(offset=pl.col("epoch") - p.start_time)
    df = df.with_columns(pl.col("offset").cast(pl.Duration("us")))
    return df.drop("epoch").with_row_index("frame")


def _get_dot_timeline(p: "Participant") -> Optional[pl.DataFrame]:
    df = p.dottest_locations

    if df is None:
        return None

    df = df.select("Epoch").with_columns(
        index=pl.lit(None, pl.UInt8),
        study=pl.lit(None, StudyType),
        source=pl.lit("dot", SourceType),
    )
    df = df.with_columns(offset=(pl.col("Epoch") - p.start_time))
    df = df.with_columns(pl.col("offset").cast(pl.Duration("us")))
    return df.drop("Epoch").with_row_index("frame")


def _get_screen_timeline(p: "Participant") -> Optional[pl.DataFrame]:
    if not p.screen_recording_path.exists():
        return None

    offset = p.screen_offset
    frames = count_video_frames(p.screen_recording_path, verify=False)
    period = 1_000_000 / get_video_fps(p.screen_recording_path)

    df = pl.DataFrame({"frame": pl.int_range(0, frames, dtype=pl.UInt32, eager=True)})
    df = df.with_columns(
        index=pl.lit(None, pl.UInt8),
        study=pl.lit(None, StudyType),
        source=pl.lit("screen", SourceType),
    )
    df = df.with_columns(offset=pl.col("frame") * period)
    df = df.with_columns(pl.col("offset").cast(pl.Duration("us")) + offset)
    return df


def _get_webcam_timeline(p: "Participant") -> pl.DataFrame:
    from .study import Study

    df = p.user_interaction_logs.select("epoch", "index", "event", "study")
    df = df.filter(pl.col("event").is_in(["start", "stop"]))
    df = df.group_by("index", maintain_order=True).first().drop("event")

    dfs: list[pl.DataFrame] = []

    index: int
    timestamp: datetime
    for index, timestamp, study in df.iter_rows():
        study = Study(study)
        offset = timestamp - p.start_time

        try:
            path = p.get_webcam_video_paths(index=index, study=study)[0]
        except IndexError:
            continue

        frame_count = count_video_frames(path, verify=True)
        frame_time = 1_000_000 / get_video_fps(path)

        df = pl.int_range(0, frame_count, dtype=pl.UInt32, eager=True)
        df = pl.DataFrame({"frame": df})
        df = df.with_columns(
            index=pl.lit(index, pl.UInt8),
            study=pl.lit(study, StudyType),
            source=pl.lit("webcam", SourceType),
        )
        df = df.with_columns(offset=pl.col("frame") * frame_time)
        df = df.with_columns(pl.col("offset").cast(pl.Duration("us")) + offset)

        dfs.append(df)

    return pl.concat(dfs, how="vertical")


def get_timeline(
    p: "Participant", sources: Union[set[Source], Source, "ellipsis", None] = ...
) -> pl.DataFrame:
    fns: dict[Source, Callable[[Path], Optional[pl.DataFrame]]] = {
        "log": _get_log_timeline,
        "webcam": _get_webcam_timeline,
        "tobii": _get_tobii_timeline,
        "screen": _get_screen_timeline,
        "dot": _get_dot_timeline,
    }

    if sources is Ellipsis:
        sources = set(SourceType.categories)
    elif isinstance(sources, str):
        sources = set([sources])
    elif sources is None or len(sources) == 0:
        sources = {"log"}
    else:
        sources = set(sources)

    with pl.StringCache():
        dfs = [fns[source](p) for source in sources]
        df = pl.concat([df for df in dfs if df is not None], how="vertical")

    return df.sort("offset").fill_null(strategy="forward")
