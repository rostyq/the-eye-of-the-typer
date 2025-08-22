from typing import TYPE_CHECKING

from numpy import timedelta64, datetime64
from numpy.typing import NDArray
from polars import (
    LazyFrame,
    DataFrame,
    col,
    lit,
    when,
    concat_arr,
    concat,
    Array,
    UInt8,
)
from rerun import (
    RecordingStream,
    TimeColumn,
    TextLog,
    TextLogLevel,
    AssetVideo,
    VideoFrameReference,
    AnyValues,
    Points2D,
    send_columns,
    log,
)
from rerun.components import Color

if TYPE_CHECKING:
    from . import FormEntry, DirDataset


__all__ = [
    "log_events",
    "log_tobii",
    "log_screen_video",
    "log_webcam_video",
    "rerun_video_indexes",
    "with_timelines",
]

TCOLS = ["timestamp", "rec_time", "webcam_time", "screen_time"]
COLOR_DTYPE = Array(UInt8, 3)


def with_timelines(
    lf: LazyFrame, /, f: "FormEntry", *, tc: str = "timestamp", screen: bool = False
) -> LazyFrame:
    return (
        lf.filter(pid=f.pid)
        .drop("pid")
        .with_columns(rec_time=(col(tc) - lit(f.init_start)))
        .with_columns(webcam_time=col("rec_time") - lit(f.webcam_delay))
        .with_columns(
            screen_time=col("rec_time") - lit(f.screen_delay) if screen else None
        )
    )


def log_events(
    lf: LazyFrame,
    /,
    *,
    recording: RecordingStream | None = None,
):
    df = (
        lf.select("event", "record", "study", *TCOLS)
        .filter(col("event").is_in(["start", "stop", "click", "save", "response"]))
        .collect()
    )

    send_columns(
        "event",
        indexes=rerun_dataframe_indexes(df, tc="timestamp"),
        columns=TextLog.columns(
            text=(text := df["event"].to_list()),
            level=[TextLogLevel.INFO] * len(text),
        ),
        recording=recording,
    )

    df = (
        lf.filter(col("event").is_in(["start", "stop"]))
        .select(
            "record",
            "event",
            "study",
            *TCOLS,
        )
        .with_columns(
            study=when(c := col("event").eq("stop"))
            .then(lit("gap"))
            .otherwise(col("study").cast(str)),
            record=when(c).then(lit(0)).otherwise(col("record")),
        )
        .collect()
    )
    send_columns(
        "webcam",
        indexes=rerun_dataframe_indexes(df, tc="timestamp"),
        columns=AnyValues.columns(
            record=df["record"].to_numpy(), study=df["study"].to_list()
        ),
    )

    df = (
        lf.filter(source="mouse")
        .drop("source")
        .select(*TCOLS, "event", col("mouse").struct.unnest())
        .with_columns(
            mouse=concat_arr(["x", "y"]),
            color=when(col("event") == "click").then(
                lit((255, 140, 0), dtype=COLOR_DTYPE)
            ).otherwise(
                lit((255, 255, 0), dtype=COLOR_DTYPE)
            ),
        )
        .drop("x", "y")
        .collect()
    )
    send_columns(
        "mouse",
        indexes=rerun_dataframe_indexes(df),
        columns=Points2D.columns(
            positions=df["mouse"].to_numpy(),
            colors=[Color(arr) for arr in df["color"].to_list()],
        ),
    )


def log_tobii(lf: LazyFrame, /, form: "FormEntry"):
    lf = lf.select(*TCOLS, "gazepoint_validity", "gazepoint_display")

    LV, RV = [col("gazepoint_validity").struct[s] for s in ["left", "right"]]
    LPOG, RPOG = [col("gazepoint_display").struct[s] for s in ["left", "right"]]

    df = (
        concat(
            [
                lf.filter(LV & RV)
                .drop("gazepoint_validity")
                .with_columns(
                    gazepoint_display=(LPOG + RPOG) / 2.0,
                    color=lit((0, 255, 255), dtype=COLOR_DTYPE),
                ),
                lf.filter(LV & RV.not_())
                .drop("gazepoint_validity")
                .with_columns(
                    gazepoint_display=LPOG,
                    color=lit((0, 0, 255), dtype=COLOR_DTYPE),
                ),
                lf.filter(LV.not_() & RV)
                .drop("gazepoint_validity")
                .with_columns(
                    gazepoint_display=RPOG,
                    color=lit((0, 255, 0), dtype=COLOR_DTYPE),
                ),
            ]
        )
        .with_columns(col("gazepoint_display").struct.unnest())
        .with_columns(
            x=col("x") * form.display_resolution.w,
            y=col("y") * form.display_resolution.h,
        )
        .drop("gazepoint_display")
        .with_columns(gazepoint_display=concat_arr(["x", "y"]))
        .drop("x", "y")
    ).collect()

    send_columns(
        "screen",
        indexes=rerun_dataframe_indexes(df),
        columns=Points2D.columns(
            positions=df["gazepoint_display"].to_numpy(),
            colors=[Color(arr) for arr in df["color"].to_list()],
        ),
    )


def log_screen_video(f: "FormEntry", ds: "DirDataset") -> bool:
    if f.screen_start is None or not (path := ds.screen_path(f.pid)).exists():
        return False

    log(name := "screen", asset := AssetVideo(path=path), static=True)
    ts = asset.read_frame_timestamps_nanos().astype("timedelta64[ns]", casting="safe")
    send_columns(
        name,
        indexes=rerun_video_indexes(
            name=name,
            ts=ts,
            start=datetime64(f.screen_start),
            delay=timedelta64(f.screen_delay),
            other=("webcam", -timedelta64(f.video_delay)),
        ),
        columns=VideoFrameReference.columns_nanos(ts),
    )
    return True


def log_webcam_video(
    f: "FormEntry",
    ds: "DirDataset",
    screen: bool,
    recording: RecordingStream | None = None,
):
    log(
        name := "webcam",
        asset := AssetVideo(path=ds.webcam_path(f.pid)),
        static=True,
        recording=recording,
    )
    ts = asset.read_frame_timestamps_nanos().astype("timedelta64[ns]", casting="safe")
    send_columns(
        name,
        indexes=rerun_video_indexes(
            name=name,
            ts=ts,
            start=datetime64(f.webcam_start),
            delay=timedelta64(f.webcam_delay),
            other=("screen", timedelta64(f.video_delay)) if screen else None,
        ),
        columns=VideoFrameReference.columns_nanos(ts),
        recording=recording,
    )


def rerun_video_indexes(
    *,
    name: str,
    ts: NDArray[timedelta64],
    start: datetime64,
    delay: timedelta64,
    other: tuple[str, timedelta64] | None = None,
):
    return [
        *filter(
            None,
            [
                TimeColumn("log_time", timestamp=ts + start),
                TimeColumn("rec_time", duration=ts + delay),
                TimeColumn(f"{name}_time", duration=ts),
                (
                    TimeColumn(f"{other[0]}_time", duration=ts + other[1])
                    if other is not None
                    else None
                ),
            ],
        )
    ]


def rerun_dataframe_indexes(df: DataFrame, /, tc: str = "timestamp"):
    return [
        *filter(
            None,
            [
                TimeColumn("log_time", timestamp=df["timestamp"].to_numpy()),
                TimeColumn("rec_time", duration=df["rec_time"].to_numpy()),
                TimeColumn("webcam_time", duration=df["webcam_time"].to_numpy()),
                (
                    TimeColumn("screen_time", duration=df["screen_time"].to_numpy())
                    if "screen_time" in df.columns
                    else None
                ),
            ],
        )
    ]


# def rerun_log_tobii(entry: TobiiEntry, *, screen: tuple[int, int]):
#     sw, sh = screen
#     for side in ("left", "right"):
#         if entry["gazepoint_validity"][side]:
#             point = entry["gazepoint_display"][side]
#             positions = [[point["x"] * sw, point["y"] * sh]]
#             entity = rr.Points2D(positions, colors=[(0, 0, 255)], radii=[5])
#         else:
#             entity = rr.Clear(recursive=True)
#         rr.log(["screen", "gaze", side], entity)

#         if entry["pupil_diameter"][side] > 0:
#             entity = rr.Scalars(entry["pupil_diameter"][side])
#         else:
#             entity = rr.Clear(recursive=True)
#         rr.log(["pupil", side], entity)


# def rerun_log_mouse(entry: MouseEntry):
#     point = entry["mouse"]
#     positions = [(point["x"], point["y"])]
#     color = (255, 0, 0) if entry["event"] == "click" else (255, 255, 0)
#     entity = rr.Points2D(positions, colors=[color], radii=[5])
#     rr.log(["screen", "mouse"], entity)
