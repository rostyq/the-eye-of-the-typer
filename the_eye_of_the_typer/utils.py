from typing import TYPE_CHECKING, Optional
from io import IOBase
from typing import Literal
from os import PathLike, environ
from pathlib import Path
from subprocess import check_output
from re import match, search

import polars as pl


if TYPE_CHECKING:
    from .participant import Participant


__all__ = [
    "get_dataset_root",
    "count_video_frames",
    "get_video_fps",
    "lookup_webcam_video_paths",
    "read_participant_characteristics",
    "read_user_interaction_logs",
    "read_tobii_gaze_predictions",
    "read_tobii_calibration_points",
    "read_tobii_specs",
    "get_timeline",
]


def get_dataset_root():
    return (
        Path(environ.get("THE_EYE_OF_THE_TYPER_DATASET_PATH", Path.cwd()))
        .expanduser()
        .resolve()
    )


def count_video_frames(p: "PathLike", verify: bool = False):
    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames" if verify else "-count_packets",
        "-show_entries",
        "stream=nb_read_frames" if verify else "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        str(Path(p).expanduser().absolute()),
    ]
    result = check_output(args, shell=True).decode("utf-8").strip()

    return int(result)


def get_video_fps(p: PathLike):
    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(Path(p).expanduser().absolute()),
    ]
    result = check_output(args, shell=True).decode("utf-8").strip()

    if "/" in result:
        num, denum = result.split("/")
        return int(num) / int(denum)
    else:
        return float(result)


def lookup_webcam_video_paths(root: Optional["PathLike"] = None):
    paths = [
        path
        for path in Path(root or get_dataset_root()).glob("**/*.webm")
        if path.is_file()
    ]

    def parse_indices(p: Path):
        pid = p.parent.name.split("_")[1]
        name = p.name
        log_id = int(match(r"(\d+)_", name).group(1))
        index = int(search(r"_(\d+)_", name).group(1))

        aux = search(r"\((\d+)\)", name)
        aux = int(aux.group(1)) if aux is not None else 0
        return pid, log_id, index, aux

    paths.sort(key=parse_indices)

    return paths


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

    event_map = {
        "recording start": "start",
        "mousemove": "mouse",
        "scrollEvent": "scroll",
        "mouseclick": "click",
        "recording stop": "stop",
        "textInput": "text",
        "text_input": "text",
    }

    df = pl.read_json(p)

    with pl.StringCache():
        df = df.with_columns(
            pl.col(key).cast(value[0]) for key, value in SCHEMA.items()
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

    df = df.drop(key.value for key, value in SCHEMA.items() if value[1] is None)
    df = df.rename(
        {key.value: value[1] for key, value in SCHEMA.items() if value[1] is not None}
    )

    return df.sort(by="epoch")


def read_tobii_gaze_predictions(p: PathLike | bytes | IOBase):
    from .schemas import TOBII_SCHEMA as SCHEMA
    from .names import TobiiField as F

    df = pl.read_ndjson(p)
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
    return df.with_columns(pl.col(C.VALIDITY_LEFT) > 1, pl.col(C.VALIDITY_RIGHT) > 1)


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


def get_timeline(p: "Participant") -> pl.DataFrame:
    from .study import Study

    tobii_df = p.tobii_gaze_predictions.select("true_time")
    tobii_df = tobii_df.with_columns(
        offset=(pl.col("true_time") - p.start_time).cast(pl.Duration("us"))
    )
    tobii_df = tobii_df.drop("true_time").with_row_index("frame")

    log_df = p.user_interaction_logs.select("epoch", "study", "index")
    log_df = log_df.select("epoch").with_columns(
        offset=(pl.col("epoch") - p.start_time).cast(pl.Duration("us")),
    )
    log_df = log_df.drop("epoch").with_row_index("frame")

    dfs = {"tobii": tobii_df, "log": log_df}

    if p.screen_recording is not None:
        frame_count = count_video_frames(p.screen_recording_path, verify=False)
        frame_time = 1_000_000 / get_video_fps(p.screen_recording_path)

        screen_df = pl.int_range(0, frame_count, dtype=pl.UInt32, eager=True)
        screen_df = pl.DataFrame({"frame": screen_df})
        screen_df = screen_df.with_columns(
            (pl.col("frame") * frame_time).cast(pl.Duration("us")).alias("offset")
            + p.screen_offset,
        )

        dfs["screen"] = screen_df

    with pl.StringCache():
        dfs = {
            key: df.with_columns(pl.lit(key, pl.Categorical).alias("source"))
            for key, df in dfs.items()
        }
        sync_df = pl.concat([df for df in dfs.values()], how="vertical")

    return sync_df.sort("offset", "frame")
