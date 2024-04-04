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
    if not p.screen_path.exists():
        return None

    offset = p.screen_offset
    frames = count_video_frames(p.screen_path, verify=False)
    period = 1_000_000 / get_video_fps(p.screen_path)

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
            path = p.get_webcam_paths(index=index, study=study)[0]
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


def timeline_dataframe(src: PathLike | bytes | IOBase, pid: int):
    df = get_timeline(src).with_columns(pid=pl.lit(pid, pl.UInt8))
    return df.select("pid", "frame", "index", "study", "source", "offset")
