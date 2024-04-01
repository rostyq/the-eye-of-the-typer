from typing import Optional
from os import PathLike, environ
from pathlib import Path
from subprocess import check_output
from re import match, search
from functools import lru_cache


__all__ = [
    "get_dataset_root",
    "count_video_frames",
    "get_video_fps",
    "lookup_webcam_video_paths",
    "clear_count_video_frames_cache",
]


def get_dataset_root():
    return (
        Path(environ.get("EOTT_DATASET_PATH", Path.cwd()))
        .expanduser()
        .resolve()
    )


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

