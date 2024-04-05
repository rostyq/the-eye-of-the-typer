from pathlib import Path
from re import match
from os import PathLike

from .utils import get_dataset_root
from .characteristics import *
from .entries import *


__all__ = [
    "parse_webcam_filename",
    "pid_from_name",
    "pid_from_path",
    "glob_webcam_files",
    "glob_specs_files",
    "glob_screen_files",
    "glob_log_files",
    "glob_dot_files",
    "glob_tobii_files",
]


def parse_webcam_filename(s: str):
    from .names import Study

    r = match(r"([0-9]+)_([0-9]+)_-study-([a-z_]+)( \(([0-9]+)\))?", s)
    assert r is not None

    log, record, study, _, aux = r.groups()

    log = int(log)
    record = int(record)
    study = Study(study).id
    aux = int(aux) if aux is not None else 0

    return log, record, study, aux


def pid_from_name(s: str):
    m = match(r"P_(\d{2})$", s.strip())
    if m is not None:
        return int(m.group(1))


def pid_from_path(p: Path):
    return pid_from_name(p.parent.name)


def _glob_dataset_files(pattern: str, root: PathLike | None = None):
    root = Path(root or get_dataset_root())
    yield from root.glob(pattern, case_sensitive=True)


def _glob_participants_dir(
    pattern: str,
    root: PathLike | None = None,
):
    yield from _glob_dataset_files(f"P_[0-9][0-9]/{pattern}", root)


def glob_webcam_files(
    root: PathLike | None = None,
    *,
    pid: int | None = None,
    log: int | None = None,
):
    if pid is not None and log is not None:
        pattern = f"P_{pid:0>2}/{log}_*"
    elif pid is None and log is not None:
        pattern = f"P_[0-9][0-9]/{log}_*"
    elif pid is not None and log is None:
        pattern = f"P_{pid:0>2}/*"
    else:
        pattern = "P_[0-9][0-9]/*"

    yield from _glob_dataset_files(f"{pattern}.webm", root)


def glob_specs_files(root: PathLike | None = None):
    yield from _glob_participants_dir(f"specs.txt", root)


def glob_log_files(root: PathLike | None = None):
    yield from _glob_participants_dir(f"*.json", root)


def glob_tobii_files(root: PathLike | None = None):
    yield from _glob_participants_dir(f"P_[0-9][0-9].txt", root)


def glob_screen_files(root: PathLike | None = None):
    yield from filter(
        lambda p: p.suffix in (".mov", ".flv"),
        _glob_participants_dir(f"P_[0-9][0-9].*", root),
    )


def glob_dot_files(root: PathLike | None = None):
    yield from _glob_participants_dir("final_dot_test_locations.tsv", root)
