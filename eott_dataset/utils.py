from typing import Optional, TYPE_CHECKING
from os import PathLike, environ
from pathlib import Path
from subprocess import check_output, run
from re import match, search
from functools import lru_cache, cache
from datetime import timedelta


if TYPE_CHECKING:
    from .participant import Participant


__all__ = [
    "get_dataset_root",
    "count_video_frames",
    "get_video_fps",
    "lookup_webcam_video_paths",
    "clear_count_video_frames_cache",
    "get_start_recording_time_offsets"
]


def get_dataset_root():
    return Path(environ.get("EOTT_DATASET_PATH", Path.cwd())).expanduser().resolve()


@lru_cache(maxsize=128)
def _count_video_frames(path: str, verify: bool):
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
        path,
    ]
    result = check_output(args, shell=True).decode("utf-8").strip()

    return int(result)


def clear_count_video_frames_cache():
    _count_video_frames.cache_clear()


def count_video_frames(p: "PathLike", verify: bool = False):
    return _count_video_frames(str(Path(p).expanduser().absolute()), verify)


def get_video_fps(p: PathLike):
    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
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
    from .study import Study

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
        study = search(r"-study-([a-z_]+)[\. ]", name).group(1)
        study = Study(study).position

        aux = search(r"\((\d+)\)", name)
        aux = int(aux.group(1)) if aux is not None else 0
        return pid, log_id, index, study, aux

    paths.sort(key=parse_indices)

    return paths


def play_screen_recordings_with_mpv(participants: list["Participant"]):
    args = ["mpv", "--osd-fractions", "--osd-level=2"]
    for p in participants:
        path = str(p.screen_recording_path.absolute())
        run([*args, path], shell=True)


def get_start_recording_time_offsets() -> dict[int, timedelta]:
    import polars as pl

    from .data import read_participant_characteristics
    from .participant import Participant

    pdf = read_participant_characteristics()

    ps = [Participant.from_dict(**row) for row in pdf.iter_rows(named=True)]
    ps = [p for p in ps if p.screen_recording_path.exists()]

    schema = {"pid": pl.UInt8, "screen_time": pl.Datetime("us")}
    df = pl.DataFrame([[p.pid, p.user_interaction_logs["epoch"][0]] for p in ps], schema)
    df = pdf.join(df.with_columns(pl.col("screen_time").cast(pl.Datetime("ms"))), on="pid")
    df = df.with_columns(screen_offset=pl.col("screen_time") - pl.col("start_time"))
    df = df.select("pid", "screen_offset")

    return {pid: offset for pid, offset in df.iter_rows()}
