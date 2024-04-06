from typing import Callable
from io import BytesIO

import polars as pl
from decord import VideoReader

from .characteristics import Study, Source


def get_frame_timestamps(file: bytes):
    vr = VideoReader(BytesIO(file))

    def get_timestamp(index: int) -> float:
        return vr.get_frame_timestamp(index)[0] * 1_000

    ts = [get_timestamp(i) for i in range(len(vr))]
    del vr, file
    return ts


def generate_frame_timestamps(file: bytes):
    vr = VideoReader(BytesIO(file))
    fps = 1000 / 30
    ts = [i * fps for i in range(len(vr))]
    del vr, file
    return ts


def with_video_to_timestamps(
    df: pl.LazyFrame,
    fn: Callable[[bytes], list[float]],
    name: str = "file",
):
    df = df.with_columns(pl.col(name).map_elements(fn, pl.List(pl.Float64)))
    return df.explode(name).with_columns(pl.col(name).cast(pl.Duration("ms")))


def get_source_timeline(df: pl.LazyFrame, form: pl.LazyFrame, source: Source):
    """
    ### Warning!
    For `screen` and `webcam` sources use `get_screen_timeline` and `get_webcam_timeline`.
    """
    has_records = "record" in df.columns and "study" in df.columns
    df = df.select(*(("pid", "record", "study") if has_records else ("pid",)), "timestamp")
    df = df.join(form.select("pid", "start_time"), "pid", "left")
    df = df.select(
        "pid",
        record="record" if has_records else pl.lit(None, pl.UInt8),
        study="study" if has_records else pl.lit(None, pl.Enum(Study.values())),
        source=pl.lit(source, pl.Enum(Source.values())),
        index=pl.col("pid").cum_count(),
        offset=pl.col("timestamp") - pl.col("start_time"),
    )
    return df


def get_screen_timeline(screen: pl.LazyFrame, form: pl.LazyFrame):
    df = form.select("pid", "start_time", "rec_time").join(screen, "pid")
    df = df.with_columns(offset=pl.col("rec_time") - pl.col("start_time"))
    df = df.drop("rec_time", "start_time")
    df = with_video_to_timestamps(df, get_frame_timestamps, "file")
    df = df.select(
        "pid",
        record=pl.lit(None, pl.UInt8),
        study=pl.lit(None, pl.Enum(Study.values())),
        source=pl.lit("screen", pl.Enum(Source.values())),
        index=pl.col("pid").cum_count(), offset=pl.col("offset") + pl.col("file")
    )
    return df


def get_webcam_timeline(webcam: pl.LazyFrame, log: pl.LazyFrame):
    from .characteristics import Study

    df = log.drop("trusted", "duration").filter(event="start").drop("event")
    df = df.join(
        webcam.drop("aux").with_columns(
            pl.col("log").cast(pl.Datetime("ms")),
            pl.col("study").cast(pl.Enum(Study.values())),
        ),
        ["pid", "record", "study"],
    )
    df = with_video_to_timestamps(df, generate_frame_timestamps)
    df = df.select(
        "pid",
        "record",
        "study",
        source=pl.lit("webcam", pl.Enum(Source.values())),
        index=pl.col("pid").cum_count().over("record"),
        offset=pl.col("timestamp") + pl.col("file") - pl.col("log"),
    )
    return df
