from typing import (
    Self,
    IO,
    Mapping,
    overload,
    Literal,
    Callable,
    cast,
    Iterable,
    Mapping,
    DefaultDict,
    TypeAlias,
)
from pathlib import Path
from datetime import datetime
from abc import abstractmethod, ABCMeta
from os import PathLike, SEEK_END, SEEK_SET, remove as remove_file
from io import TextIOBase
from contextlib import suppress
from functools import cached_property, cache
from enum import auto, StrEnum, EnumMeta
from re import compile, match, search, findall
from dataclasses import dataclass
from zipfile import ZipFile, ZipInfo
from datetime import timedelta
from collections import defaultdict
from shutil import move as move_file

from polars import (
    String,
    Categorical,
    UInt64,
    UInt16,
    Float64,
    UInt8,
    Int32,
    Int64,
    Struct,
    Boolean,
    Array,
    UInt32,
    Int8,
    Enum,
    Datetime,
    Duration,
    from_epoch,
    read_csv,
    scan_csv,
    scan_ndjson,
    read_json,
    lit,
    when,
    struct,
    col,
    concat,
    coalesce,
    LazyFrame,
)
from polars._typing import PolarsDataType
from decord import VideoReader

import ffmpeg as ff


__all__ = [
    "ZipDataset",
    "Characteristic",
    "Webcam",
    "Log",
    "Tobii",
    "Calib",
    "Dot",
    "Spec",
    "Event",
    "Setting",
    "Gender",
    "Race",
    "SkinColor",
    "EyeColor",
    "FacialHair",
    "Vision",
    "Handedness",
    "Weather",
    "PointingDevice",
    "Study",
    "StudyName",
    "Source",
    "print_schema",
    "play_with_mpv",
    "ffmpeg_args",
    "pid_from_name",
    "parse_timedelta",
    "extract_transform_load",
]


class NameEnum(StrEnum):
    @classmethod
    def from_id(cls, i: int):
        return cls.__members__[cls._member_names_[i]]

    @classmethod
    def values(cls):
        return [item.value for item in cls.__members__.values()]

    @cached_property
    def id(self):
        return self.__class__._member_names_.index(self.name)


class Study(NameEnum):
    DOT_TEST_INSTRUCTIONS = auto()
    DOT_TEST = auto()
    FITTS_LAW_INSTRUCTIONS = auto()
    FITTS_LAW = auto()
    SERP_INSTRUCTIONS = auto()
    BENEFITS_OF_RUNNING_INSTRUCTIONS = auto()
    BENEFITS_OF_RUNNING = auto()
    BENEFITS_OF_RUNNING_WRITING = auto()
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES_INSTRUCTIONS = auto()
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES = auto()
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES_WRITING = auto()
    WHERE_TO_FIND_MOREL_MUSHROOMS_INSTRUCTIONS = auto()
    WHERE_TO_FIND_MOREL_MUSHROOMS = auto()
    WHERE_TO_FIND_MOREL_MUSHROOMS_WRITING = auto()
    TOOTH_ABSCESS_INSTRUCTIONS = auto()
    TOOTH_ABSCESS = auto()
    TOOTH_ABSCESS_WRITING = auto()
    DOT_TEST_FINAL_INSTRUCTIONS = auto()
    DOT_TEST_FINAL = auto()
    THANK_YOU = auto()


def extract_transform_load(
    zip: ZipFile,
    dstdir: Path,
    tmpdir: Path,
    *,
    form: bool = True,
    log: bool = False,
    tobii: bool = False,
    dot: bool = False,
    calib: bool = False,
    screen: bool = False,
    webcam: bool = False,
    concat: bool = False,
    sync: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
):
    _prefix = (
        Path(zip.filename).stem
        if zip.filename
        else zip.filelist[0].filename.split("/")[0]
    )
    dstdir.mkdir(parents=True, exist_ok=True)
    ds = ZipDataset(zip)

    print("Scanning participants and log files...", end=" ", flush=True)
    plf, llf = ds.scan_participants(sync)
    print("Done.", flush=True)

    if form:
        if not dry_run:
            print(
                "Saving participants and log files to parquet...", end=" ", flush=True
            )
            if not (dst := dstdir / "form.parquet").exists() or overwrite:
                plf.sink_parquet(dst, compression="uncompressed")
                print("Done.", flush=True)
            else:
                print("Skipped. (already exists)", flush=True)
            del dst
        else:
            print("Skipped. (dry run)", flush=True)

    if log:
        print("Saving log file to parquet...", end=" ", flush=True)
        if not dry_run:
            if not (dst := dstdir / "log.parquet").exists() or overwrite:
                llf.sink_parquet(dst, compression="uncompressed")
                print("Done.", flush=True)
            else:
                print("Skipped. (already exists)", flush=True)
            del dst
        else:
            print("Skipped. (dry run)", flush=True)

    if tobii:
        print("Processing Tobii data...", end=" ", flush=True)
        if not dry_run:
            if not (dst := dstdir / "tobii.parquet").exists() or overwrite:
                Tobii.scan_zip(zip).sink_parquet(dst, compression="lz4")
                print("Done.", flush=True)
            else:
                print("Skipped. (already exists)", flush=True)
            del dst
        else:
            print("Skipped. (dry run)", flush=True)

    if dot:
        print("Processing Dot data...", end=" ", flush=True)
        if not dry_run:
            if not (dst := dstdir / "dot.parquet").exists() or overwrite:
                Dot.scan_zip(zip).sink_parquet(dst, compression="uncompressed")
                print("Done.", flush=True)
            else:
                print("Skipped. (already exists)", flush=True)
            del dst
        else:
            print("Skipped. (dry run)", flush=True)

    if calib:
        if not dry_run:
            print("Processing calibration data...", end=" ", flush=True)
            if not (dst := dstdir / "calib.parquet").exists() or overwrite:
                Calib.scan_zip(zip).sink_parquet(dst, compression="uncompressed")
                print("Done.", flush=True)
            else:
                print("Skipped. (already exists)", flush=True)
            del dst
        else:
            print("Skipped. (dry run)", flush=True)

    (dstdir / "screen").mkdir(exist_ok=True)
    for file in ds.screen_files() if screen else []:
        dst = dstdir / "screen" / f"P_{pid_from_name(file.filename, error=True):02}.mp4"
        print(
            f"Extracting screen video from {file.filename.removeprefix(_prefix)} into {dst.relative_to(dstdir)}...",
            end=" " if not dry_run else "\n",
            flush=True,
        )

        input_path = None
        if dst.exists() and not overwrite:
            print("Skipped. (already exists)", flush=True)
            continue

        dst = str(dst)

        match file.filename.split(".")[-1].strip().lower():
            case "flv":
                f = ff.input("pipe:", f=file.filename.split(".")[-1])
                f = ff.output(
                    f.audio,
                    ff.filter(f.video, "scale", "iw/2", "ih/2"),
                    dst,
                    format="mp4",
                    fps_mode="passthrough",
                )

            case "mov":
                input_path = tmpdir / file.filename.split("/")[-1]

                if not dry_run:
                    with zip.open(file) as zf, open(input_path, "wb") as f:
                        f.write(zf.read())

                f = ff.input(input_path)
                f = ff.output(
                    f.audio,
                    ff.filter(ff.filter(f.video, "fps", 25), "scale", "iw/2", "ih/2"),
                    dst,
                    format="mp4",
                )

        if dry_run:
            print(
                f"{file.filename.removeprefix(_prefix)} -> {' '.join(ff.get_args(f))}"
            )
            continue

        try:
            _ = ff.run(
                f,
                input=None if input_path else zip.read(file),
                capture_stdout=True,
                capture_stderr=True,
                overwrite_output=True,
            )
            print("Done.", flush=True)
            del f

        except ff.Error as e:
            with suppress(OSError):
                remove_file(dst)

            raise RuntimeError(e.stderr.decode()) from e

        except KeyboardInterrupt:
            with suppress(OSError):
                remove_file(dst)
                break

        finally:
            if input_path and not dry_run:
                with suppress(OSError):
                    remove_file(input_path)
        del file, dst, input_path

    need_concat: DefaultDict[tuple[int, int], list[Path]] = defaultdict(list)
    (dstdir / "webcam").mkdir(exist_ok=True)
    for file in ds.webcam_files() if webcam else []:
        w = cast(Webcam, Webcam.parse_raw(file.filename))
        pid = pid_from_name(file.filename, error=True)

        dst = (
            dstdir / "webcam" / f"P_{pid:02}" / f"{w.record:02}-{w.study.value}"
        ).with_suffix(".mp4")

        if w.aux is not None:
            if dst.exists():
                print("Skipped. (already exists)", flush=True)
                continue
            dst = dst.with_stem(f"{dst.stem}-{w.aux}")
            need_concat[(pid, w.record)].append(dst)

        print(
            f"Extracting webcam video from {file.filename.removeprefix(_prefix)} into {dst.relative_to(dstdir)} ...",
            end=" ",
            flush=True,
        )

        if dst.exists() and not overwrite:
            print("Skipped. (already exists)", flush=True)
            continue

        dst.parent.mkdir(exist_ok=True)
        dst = str(dst)

        f = ff.input("pipe:", f="webm")
        # v = ff.filter(f.video, "fps", 30)
        # f = ff.output(v, f.audio, dst, format="mp4", reset_timestamps="1")
        f = ff.output(f, dst, format="mp4", fps_mode="passthrough")

        if dry_run:
            print(" ".join(ff.get_args(f)))
            continue

        try:
            _ = ff.run(
                f,
                input=zip.read(file),
                capture_stdout=True,
                capture_stderr=True,
                overwrite_output=overwrite,
            )
            print("Done.", flush=True)
            del f

        except ff.Error as e:
            with suppress(OSError):
                remove_file(dst)
            raise RuntimeError(e.stderr.decode()) from e

        except KeyboardInterrupt:
            with suppress(OSError):
                remove_file(dst)
            break
        del file, w, pid, dst

    for (pid, record), paths in need_concat.items():
        # sort by aux number
        paths.sort(key=lambda p: int(p.stem[-1]))
        # get path for file with no aux number
        path = (path := paths[0]).with_stem(path.stem.removesuffix("-1"))
        first_path = path.with_stem(path.stem + "-0")

        if not dry_run:
            # path will be concatenated video
            move_file(str(path), str(first_path))
        else:
            print(
                f"Would move {path.relative_to(dstdir)} to {first_path.relative_to(dstdir)} for concatenation"
            )

        paths.insert(0, first_path)
        f = ff.concat(*[ff.input(str(p)) for p in paths])
        f = ff.output(f, str(path))

        print(
            "Concatenating aux webcam videos for participant",
            pid,
            "record",
            record,
            "...",
            end=" ",
            flush=True,
        )
        if not dry_run:
            try:
                _ = ff.run(
                    f,
                    capture_stdout=True,
                    capture_stderr=True,
                    overwrite_output=overwrite,
                )
                print("Done.", flush=True)

                for p in map(str, paths):
                    with suppress(OSError):
                        remove_file(p)

                del f, p

            except ff.Error as e:
                with suppress(OSError):
                    remove_file(str(path))
                    move_file(str(first_path), str(path))
                raise RuntimeError(e.stderr.decode()) from e

            except KeyboardInterrupt:
                with suppress(OSError):
                    remove_file(str(path))
                    move_file(str(first_path), str(path))
                break
        else:
            print(" ".join(ff.get_args(f)), flush=True)

        del paths, pid, record, path, first_path
    del need_concat

    if webcam and concat:
        interrupted = False
        pid: int
        webcam_first_start: datetime
        for pid, webcam_first_start in (
            plf.select("pid", "webcam_start").collect().iter_rows()
        ):
            if interrupted:
                break

            previous_end = timedelta()
            record: int
            webcam_part_start: datetime
            # print(pid, webcam_first_start)
            for record, webcam_part_start, study in (
                llf.filter(pid=pid, event="start")
                .select("record", "timestamp", "study")
                .sort("record")
                .collect()
                .iter_rows()
            ):
                study = Study(study)
                path = (
                    dstdir / "webcam" / f"P_{pid:02}" / f"{record:02}-{study.value}"
                ).with_suffix(".mp4")
                start_offset = webcam_part_start - webcam_first_start

                if not path.exists():
                    print(
                        f"Webcam video for {path} does not exist, skipping...",
                        flush=True,
                    )
                    continue

                vr = VideoReader(str(path))
                dur = timedelta(seconds=float(vr.get_frame_timestamp(-1)[1]))
                gap = start_offset - previous_end
                # print(previous_end, "|", start_offset, "|", dur, "|", gap, flush=True)
                previous_end = start_offset + dur

                if not gap:
                    continue

                if not dry_run:
                    path = path.with_stem(path.stem + "-gap")
                    print(
                        f"Creating blank video for {path.relative_to(dstdir)} with duration {gap.total_seconds()} seconds...",
                        end=" ",
                        flush=True,
                    )

                    if path.exists() and not overwrite:
                        print(f"Skipped. (already exists)", flush=True)
                        continue

                    try:
                        _ = ff.run(
                            ffmpeg_blank(
                                str(path),
                                gap,
                            ),
                            capture_stdout=True,
                            capture_stderr=True,
                            overwrite_output=overwrite,
                        )
                        print("Done.", flush=True)
                    except ff.Error as e:
                        print(f"Error running ffmpeg: {e.stderr.decode('utf-8')}")
                    except KeyboardInterrupt:
                        interrupted = True
                        break
                else:
                    print(
                        f"Would create blank video for {path.relative_to(dstdir)} with duration {gap.total_seconds()} seconds",
                        flush=True,
                    )
                del vr, dur, gap
            del previous_end, record, webcam_first_start, study, start_offset, path

            input_dir = dstdir / "webcam" / f"P_{pid:02}"
            output_path = input_dir.with_suffix(".mp4")

            # for p in sorted(input_dir.glob("*.mp4"), key=lambda p: p.stem):
            #     print(p)

            if not dry_run:
                print(
                    f"Concatenating webcam videos for participant {pid} ...",
                    end=" ",
                    flush=True,
                )

                if output_path.exists() and not overwrite:
                    print("Skipped. (already exists)", flush=True)
                    continue

                inputs = [
                    ff.input(str(p))
                    for p in sorted(input_dir.glob("*.mp4"), key=lambda p: p.stem)
                ]

                try:
                    _ = ff.run(
                        ff.output(
                            ff.concat(*inputs), str(output_path), fps_mode="passthrough"
                        ),
                        capture_stdout=True,
                        capture_stderr=True,
                        overwrite_output=overwrite,
                    )
                except ff.Error as e:
                    print(f"Error running ffmpeg: {e.stderr.decode('utf-8')}")

                del inputs
            else:
                print(
                    f"Would concatenate webcam videos for participant {pid} into {output_path.relative_to(dstdir)}",
                    flush=True,
                )


def webcam_durations_by_log(llf: LazyFrame) -> dict[tuple[int, int, Study], timedelta]:
    """Calculate durations of webcam videos using log data."""
    on = ("pid", "record")
    cols = [*on, "study"]
    ts = col("timestamp")
    llf = (
        llf.filter(event="start")
        .filter(col("study").ne(Study.THANK_YOU))
        .select(*cols, ts.alias("start"))
        .join(
            llf.filter(event="stop").select(*on, ts.alias("stop")),
            on,
            "left",
        )
        .join(
            llf.filter(event="start")
            .select(
                "pid",
                (col("record") - lit(1)).alias("record"),
                ts.alias("next"),
            )
            .filter(col("record") > 1)
            .join(llf.filter(event="stop"), on, "anti"),
            on,
            "left",
        )
        .with_columns(coalesce("stop", "next").alias("stop"))
        .select(*cols, (col("stop") - col("start")).alias("duration"))
    )
    return {
        (pid, record, study): duration
        for (pid, record, study, duration) in llf.collect().iter_rows()
    }


@overload
def parse_timedelta(s: str) -> timedelta: ...
@overload
def parse_timedelta(s: str, ignore_error: Literal[False]) -> timedelta: ...
@overload
def parse_timedelta(s: str, ignore_error: Literal[True]) -> timedelta | None: ...
def parse_timedelta(s: str, ignore_error: bool = False) -> timedelta | None:
    """
    Parse a time string into a timedelta object.

    Supports all timedelta units:
    - w: weeks
    - d: days
    - h: hours
    - m: minutes
    - s: seconds
    - ms: milliseconds
    - us: microseconds

    Examples:
    - "1s" -> 1 second
    - "1m" -> 1 minute
    - "1h" -> 1 hour
    - "1d" -> 1 day
    - "1w" -> 1 week
    - "1ms" -> 1 millisecond
    - "1us" -> 1 microsecond
    - "1w2d3h4m5s6ms7us" -> combined time units
    """
    if not s.strip():
        if ignore_error:
            return None
        else:
            raise ValueError("Empty time string")

    s = "".join(c for c in s if not c.isspace())

    # Pattern to match number followed by unit
    # Order matters: longer units must come before shorter ones (ms before m, us before s)
    pattern = r"(\d+(?:\.\d+)?)(us|ms|[wdhms])"
    matches = findall(pattern, s.lower())

    if not matches:
        if ignore_error:
            return None
        else:
            raise ValueError(f"Invalid time format: {s}")

    # Check if entire string was consumed
    consumed_length = sum(len(match[0]) + len(match[1]) for match in matches)
    if consumed_length != len(s.replace(" ", "")):
        if ignore_error:
            return None
        else:
            raise ValueError(f"Invalid time format: {s}")

    weeks, days, hours, minutes, seconds, milliseconds, microseconds = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    for digits, unit in cast(
        list[tuple[str, Literal["us", "ms", "s", "m", "h", "d", "w"]]], matches
    ):
        value = float(digits)
        match unit:
            case "us":
                microseconds = value
            case "ms":
                milliseconds = value
            case "s":
                seconds = value
            case "m":
                minutes = value
            case "h":
                hours = value
            case "d":
                days = value
            case "w":
                weeks = value

    return timedelta(
        days=days,
        seconds=seconds,
        microseconds=microseconds,
        milliseconds=milliseconds,
        minutes=minutes,
        hours=hours,
        weeks=weeks,
    )


def print_schema(
    schema: Mapping[str, "PolarsDataType"],
    name: str = "",
    /,
    out: TextIOBase | None = None,
    *,
    indent: str = "\t",
    depth: int = 0,
):
    assert indent.isspace() or len(indent) == 0
    assert depth >= 0

    if depth == 0 and name:
        print(name, end=":\n", file=out)
    else:
        print(end="\n", file=out)

    for field, data_type in schema.items():
        print(f"{indent * (depth + 1 if name else depth)}{field}:", end="", file=out)
        if isinstance(data_type, Struct):
            print_schema(
                data_type.to_schema(), field, out=out, indent=indent, depth=depth + 1
            )
        else:
            print(f" {data_type}", file=out)


def ffmpeg_args(
    i: str = "pipe:0",
    o: str = "pipe:0",
    *,
    exe: str = "ffmpeg",
    fmt: str | None = None,
    cmds: Iterable[str] = [],
    params: Mapping[str, str] = {},
):
    res = [exe, "-i", i]

    if fmt is not None:
        res.insert(1, f"-f")
        res.insert(2, fmt)

    for cmd in cmds:
        res.append(f"-{cmd}")

    for key, value in params.items():
        res.append(f"-{key}")
        res.append(value)

    res.append(o)

    return res


def play_with_mpv(src: PathLike):
    from subprocess import run

    return run(["mpv", "--osd-fractions", "--osd-level=2", src], shell=True)


@overload
def pid_from_name(s: str, error: Literal[False] = False) -> int | None: ...
@overload
def pid_from_name(s: str, error: Literal[True] = True) -> int: ...
def pid_from_name(s: str, error: bool = False) -> int | None:
    m = search(r"P_(\d{2})", s.strip())
    if m is not None:
        return int(m.group(1))
    elif error:
        raise ValueError(f"Invalid participant name: {s!r}")


def concat_scan_zip(
    ref: ZipFile,
    scan_fn: Callable[[Path | bytes | IO[bytes], int], LazyFrame],
    filter_fn: Callable[[str], bool],
):
    return concat(
        [
            scan_fn(ref.open(file), pid_from_name(file.filename, error=True))
            for file in ref.filelist
            if filter_fn(file.filename)
        ]
    )


class ZipDataset:
    """
    A class to represent a dataset stored in a zip file.
    This is the raw dataset straight from the Eye of the Typer website.
    """

    def __init__(self, ref: ZipFile):
        self.ref = ref

    @property
    def files(self) -> list[ZipInfo]:
        return self.ref.filelist

    def screen_files(self):
        return [
            *sorted(
                [f for f in self.files if f.filename.endswith((".flv", ".mov"))],
                key=lambda f: f.filename,
            )
        ]

    def webcam_files(self):
        return [
            *sorted(
                [f for f in self.files if f.filename.endswith(".webm")],
                key=lambda f: f.filename,
            )
        ]

    def scan_participants(self, sync: bool = True):
        """
        Scan participant file and add additional information parsed from the log and spec files.
        Returns a lazy frame for participant data and log source lazy frame.

        The `first_task_frames` parameter dictionary is used to override default
        adjustments for screen recording start times.
        """
        pdf = Characteristic.read_zip(self.ref)
        llf = Log.scan_zip(self.ref)
        slf = Spec.scan_zip(self.ref)

        def screen_clock_sync_diff(s: str, step: int = 1) -> timedelta:
            assert step in [-1, 0, 1], "Step must be -1, 0, or 1"
            t = datetime.fromisoformat(s)
            return t.replace(minute=t.minute + step, second=0, microsecond=0) - t

        plf = (
            pdf.lazy()
            .join(
                llf.select("pid", col("timestamp").alias("start_timestamp"), "event")
                .filter(col("event") == "start")
                .group_by("pid")
                .first()
                .drop("event"),
                on="pid",
            )
            .join(
                LazyFrame(
                    list(
                        {
                            int(k.removeprefix("P_")): (
                                parse_timedelta(ftf) + screen_clock_sync_diff(scs, step)
                                if scs and sync
                                else parse_timedelta(ftf)
                            )
                            # first task frame in screen recording
                            # and timestamp when screen clock change minute
                            for k, (ftf, scs, step) in {
                                "P_01": ("11s 733ms", "2017-04-05 20:19:59.593Z", 1),
                                "P_02": ("3s 350ms", "2017-04-06 14:11:59.597Z", 1),
                                "P_06": ("36s 828ms", "2017-04-07 14:08:00.378041Z", 0),
                                "P_07": ("5s 160ms", "2017-04-07 15:13:59.574Z", 1),
                                "P_08": ("32s 32ms", "2017-04-07 19:10:00.067500Z", 1),
                                "P_10": (
                                    "2m 34s 29ms",
                                    "2017-04-07 20:44:00.647208Z",
                                    0,
                                ),
                                "P_12": ("34s 451ms", "2017-04-11 14:06:00.158666Z", 1),
                                "P_13": (
                                    "1m 3s 689ms",
                                    "2017-04-11 15:13:00.208708Z",
                                    0,
                                ),
                                "P_16": ("29s 863ms", "2017-04-11 19:10:00.332458Z", 0),
                                "P_17": ("21s 939ms", "2017-04-11 20:25:00.549375Z", 0),
                                "P_18": ("24s 441ms", "2017-04-12 14:10:00.814958Z", 0),
                                "P_19": ("25s 275ms", "2017-04-12 15:29:00.765083Z", 0),
                                "P_20": ("27s 27ms", "2017-04-12 18:19:00.157750Z", 0),
                                "P_23": ("24s 733ms", "2017-04-13 15:38:00.081583Z", 0),
                                "P_25": ("29s 655ms", "2017-04-13 19:18:00.497041Z", 0),
                                "P_27": ("25s 651ms", "2017-04-14 14:09:00.819375Z", 0),
                                "P_31": (
                                    "3m 51s 273ms",
                                    "2017-04-14 18:08:59.993833Z",
                                    1,
                                ),
                                "P_35": ("28s 362ms", "2017-04-18 18:10:00.042083Z", 0),
                                "P_40": ("21s 230ms", "2017-04-20 13:18:00.743083Z", 0),
                                "P_41": ("20s 395ms", "2017-04-20 14:11:00.169041Z", 0),
                                "P_42": ("18s 852ms", "2017-04-20 16:10:00.376500Z", 0),
                                "P_44": ("23s 398ms", "2017-04-20 18:04:00.172916Z", 0),
                                "P_45": ("30s 697ms", "2017-04-20 20:17:00.500750Z", 0),
                                "P_46": (
                                    "1m 1s 645ms",
                                    "2017-04-21 13:17:59.994500Z",
                                    1,
                                ),
                                "P_55": ("13s 805ms", "2017-04-25 16:07:00.840416Z", 0),
                                "P_56": ("22s 856ms", "2017-04-25 17:16:00.122333Z", 0),
                                "P_59": ("21s 813ms", "2017-04-26 14:12:00.214041Z", 0),
                            }.items()
                        }.items()
                    ),
                    schema={"pid": UInt8, "start_time": Duration("us")},
                    orient="row",
                ),
                on="pid",
                how="left",
            )
            .with_columns(
                (col("start_timestamp") - col("start_time")).alias("screen_start")
            )
            .drop("start_timestamp", "start_time")
            .join(slf, on="pid")
            .join(
                llf.filter(event="start")
                .select("pid", "record", col("timestamp").alias("webcam_start"))
                .sort("pid", "record")
                .group_by("pid")
                .first()
                .drop("record"),
                on="pid",
            )
        )
        return plf, llf


class Setting(StrEnum):
    LAPTOP = "Laptop"
    PC = "PC"


class Gender(StrEnum):
    MALE = "Male"
    FEMALE = "Female"


class Race(StrEnum):
    WHITE = "White"
    BLACK = "Black"
    ASIAN = "Asian"
    OTHER = "Other"


class SkinColor(StrEnum):
    C1 = "1"
    C2 = "2"
    C3 = "3"
    C4 = "4"
    C5 = "5"
    C6 = "6"


class EyeColor(StrEnum):
    DARK_BROWN_TO_BROWN = "Dark Brown to Brown"
    GRAY_TO_BLUE_OR_PINK = "Gray to Blue or Pink"
    GREEN_HAZEL_TO_BLUE_HAZEL = "Green-Hazel to Blue-Hazel"
    GREEN_HAZEL_TO_GREEN = "Green-Hazel to Green"
    AMBER = "Amber"


class FacialHair(StrEnum):
    BEARD = "Beard"
    LITTLE = "Little"
    NONE = "None"


class Vision(StrEnum):
    NORMAL = "Normal"
    GLASSES = "Glasses"
    CONTACTS = "Contacts"


class Handedness(StrEnum):
    LEFT = "Left"
    RIGHT = "Right"


class Weather(StrEnum):
    CLOUDY = "Cloudy"
    INDOORS = "Indoors"
    SUNNY = "Sunny"


class PointingDevice(StrEnum):
    TRACKPAD = "Trackpad"
    MOUSE = "Mouse"


StudyName: TypeAlias = Literal[
    "dot_test_instructions",
    "dot_test",
    "fitts_law_instructions",
    "fitts_law",
    "serp_instructions",
    "benefits_of_running_instructions",
    "benefits_of_running",
    "benefits_of_running_writing",
    "educational_advantages_of_social_networking_sites_instructions",
    "educational_advantages_of_social_networking_sites",
    "educational_advantages_of_social_networking_sites_writing",
    "where_to_find_morel_mushrooms_instructions",
    "where_to_find_morel_mushrooms",
    "where_to_find_morel_mushrooms_writing",
    "tooth_abscess_instructions",
    "tooth_abscess",
    "tooth_abscess_writing",
    "dot_test_final_instructions",
    "dot_test_final",
    "thank_you",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Webcam:
    log: int
    """Participant log ID."""
    record: int
    """Participant record ID."""
    study: Study
    """Study name."""
    aux: int | None = None
    """Auxiliary number, used for multiple recordings in the same log and record."""

    @cache
    @staticmethod
    def webcam_filename_pattern():
        return compile(r"([0-9]+)_([0-9]+)_-study-([a-z_]+)( \(([0-9]+)\))?\.webm$")

    @classmethod
    def parse_raw(cls, s: str):
        if (r := cls.webcam_filename_pattern().search(s)) is None:
            return

        log, record, study, _, aux = r.groups()

        log = int(log)
        record = int(record)
        study = Study(study)
        aux = int(aux) if aux is not None else None

        return cls(log=log, record=record, study=study, aux=aux)


class Source(NameEnum):
    """
    Enumeration class representing different sources of input.
    """

    LOG = auto()
    """log events or unknown entries"""
    MOUSE = auto()
    """mouse events"""
    SCROLL = auto()
    """scroll events"""
    INPUT = auto()
    """text input events"""
    TEXT = auto()
    """text submit events"""
    TOBII = auto()
    """tobii recording"""
    DOT = auto()
    """dot test entries"""
    CALIB = auto()
    """tobii calibration data"""


class SourceEnumClass(metaclass=EnumMeta):
    @classmethod
    @abstractmethod
    def schema(cls) -> dict[str, PolarsDataType]: ...

    @classmethod
    @abstractmethod
    def source_filename(cls, value: str) -> bool: ...

    @classmethod
    @abstractmethod
    def scan_raw(cls, source: Path | bytes | IO[bytes]) -> LazyFrame: ...

    @classmethod
    def scan(cls, source: Path | bytes | IO[bytes], pid: int) -> LazyFrame:
        return cls.scan_raw(source).with_columns(pid=lit(pid, UInt8))

    @classmethod
    def scan_zip(cls, ref: ZipFile) -> LazyFrame:
        return concat_scan_zip(ref, scan_fn=cls.scan, filter_fn=cls.source_filename)


class SourceClass(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def schema(cls) -> dict[str, PolarsDataType]: ...

    @classmethod
    def source_filename(cls, value: str) -> bool: ...

    @classmethod
    def scan_raw(cls, source: Path | bytes | IO[bytes]) -> LazyFrame: ...

    @classmethod
    def scan(cls, source: Path | bytes | IO[bytes], pid: int) -> LazyFrame:
        return cls.scan_raw(source).with_columns(pid=lit(pid, UInt8))

    @classmethod
    def scan_zip(cls, ref: ZipFile) -> LazyFrame:
        return concat_scan_zip(ref, scan_fn=cls.scan, filter_fn=cls.source_filename)


class Event(StrEnum):
    SCROLL_EVENT = "scrollEvent"
    MOUSE_MOVE = "mousemove"
    MOUSE_CLICK = "mouseclick"
    TEXT_INPUT = "textInput"
    TEXT_SUBMIT = "text_input"
    REC_START = "recording start"
    REC_STOP = "recording stop"

    VIDEO_START = "video started"
    VIDEO_STOP = "video stop"
    VIDEO_SAVE = "video saved"

    @classmethod
    def video(cls):
        return {cls.VIDEO_START.value, cls.VIDEO_STOP.value, cls.VIDEO_SAVE.value}

    @classmethod
    def recording(cls):
        return {cls.REC_START.value, cls.REC_STOP.value}

    @classmethod
    def mouse(cls):
        return {cls.MOUSE_MOVE.value, cls.MOUSE_CLICK.value}

    @classmethod
    def type_aliases(cls):
        return {
            cls.MOUSE_CLICK: "click",
            cls.MOUSE_MOVE: "move",
            cls.REC_START: "start",
            cls.REC_STOP: "stop",
            cls.SCROLL_EVENT: "scroll",
            cls.TEXT_INPUT: "input",
            cls.TEXT_SUBMIT: "text",
        }


class Characteristic(StrEnum):
    PID = "Participant ID"
    LOG_ID = "Participant Log ID"
    DATE = "Date"
    SETTING = "Setting"
    DISPLAY_WIDTH = "Display Width (pixels)"
    DISPLAY_HEIGHT = "Display Height (pixels)"
    SCREEN_WIDTH = "Screen Width (cm)"
    SCREEN_HEIGHT = "Screen Height (cm)"
    DISTANCE_FROM_SCREEN = "Distance From Screen (cm)"
    SCREEN_RECORDING_START_TIME_UNIX = "Screen Recording Start Time (Unix milliseconds)"
    SCREEN_RECORDING_START_TIME_UTC = "Screen Recording Start Time (Wall Clock)"
    GENDER = "Gender"
    AGE = "Age"
    RACE = "Self-Reported Race"
    SKIN_COLOR = "Self-Reported Skin Color"
    EYE_COLOR = "Self-Reported Eye Color"
    FACIAL_HAIR = "Facial Hair"
    VISION = "Self-Reported Vision"
    TOUCH_TYPER = "Touch Typer"
    HANDEDNESS = "Self-Reported Handedness"
    WEATHER = "Weather"
    POINTING_DEVICE = "Pointing Device"
    NOTES = "Notes"
    TIME_OF_DAY = "Time of day"
    DURATION = "Duration"

    @classmethod
    def schema(cls):
        return {
            cls.PID: String,
            cls.LOG_ID: UInt64,
            cls.DATE: String,
            cls.SETTING: Categorical,
            cls.DISPLAY_WIDTH: UInt16,
            cls.DISPLAY_HEIGHT: UInt16,
            cls.SCREEN_WIDTH: Float64,
            cls.SCREEN_HEIGHT: Float64,
            cls.DISTANCE_FROM_SCREEN: Float64,
            cls.SCREEN_RECORDING_START_TIME_UNIX: UInt64,
            cls.SCREEN_RECORDING_START_TIME_UTC: String,
            cls.GENDER: Categorical,
            cls.AGE: UInt8,
            cls.RACE: Categorical,
            cls.SKIN_COLOR: Categorical,
            cls.EYE_COLOR: Categorical,
            cls.FACIAL_HAIR: Categorical,
            cls.VISION: Categorical,
            cls.TOUCH_TYPER: Categorical,
            cls.HANDEDNESS: Categorical,
            cls.WEATHER: Categorical,
            cls.POINTING_DEVICE: Categorical,
            cls.NOTES: String,
            cls.TIME_OF_DAY: String,
            cls.DURATION: String,
        }

    @classmethod
    def aliases(cls):
        return {
            cls.PID: "pid",
            cls.LOG_ID: "log",
            cls.DATE: "date",
            cls.SETTING: "setting",
            cls.DISPLAY_WIDTH: "display_width",
            cls.DISPLAY_HEIGHT: "display_height",
            cls.SCREEN_WIDTH: "screen_width",
            cls.SCREEN_HEIGHT: "screen_height",
            cls.DISTANCE_FROM_SCREEN: "screen_distance",
            cls.SCREEN_RECORDING_START_TIME_UNIX: "screen_start",
            cls.SCREEN_RECORDING_START_TIME_UTC: "wall_clock",
            cls.GENDER: "gender",
            cls.AGE: "age",
            cls.RACE: "race",
            cls.SKIN_COLOR: "skin_color",
            cls.EYE_COLOR: "eye_color",
            cls.FACIAL_HAIR: "facial_hair",
            cls.VISION: "vision",
            cls.TOUCH_TYPER: "touch_typer",
            cls.HANDEDNESS: "handedness",
            cls.WEATHER: "weather",
            cls.POINTING_DEVICE: "pointing_device",
            cls.NOTES: "notes",
            cls.TIME_OF_DAY: "time_of_day",
            cls.DURATION: "duration",
        }

    @classmethod
    def enums(cls) -> dict[Self, type[StrEnum]]:
        return {
            cls.SETTING: Setting,
            cls.GENDER: Gender,
            cls.RACE: Race,
            cls.EYE_COLOR: EyeColor,
            cls.SKIN_COLOR: SkinColor,
            cls.FACIAL_HAIR: FacialHair,
            cls.VISION: Vision,
            cls.HANDEDNESS: Handedness,
            cls.WEATHER: Weather,
            cls.POINTING_DEVICE: PointingDevice,
        }  # type: ignore

    @classmethod
    def read_raw(cls, source: Path | bytes | IO[bytes]):
        return read_csv(
            source,
            has_header=True,
            separator=",",
            quote_char='"',
            null_values=["-", ""],
            schema={k.value: v for k, v in cls.schema().items()},
        )

    @classmethod
    def read(cls, source: Path | bytes | IO[bytes]):
        df = cls.read_raw(source)
        df = df.with_columns(
            # indices
            col(cls.PID).str.split("_").list.get(1).str.to_integer().cast(UInt8),
            from_epoch(cls.LOG_ID, "ms").alias("init_start"),
            # timestamps, dates, times, durations
            col(cls.DATE).str.to_date(r"%m/%d/%Y"),
            col(cls.DURATION)
            .str.split(":")
            .cast(Array(UInt16, 3))
            .map_elements(
                lambda arr: (arr[0] * 3600 + arr[1] * 60 + arr[2]) * 1000,
                return_dtype=Int64,
            )
            .cast(Duration("ms")),
            col(cls.TIME_OF_DAY).str.to_time(r"%H:%M"),
            from_epoch(cls.SCREEN_RECORDING_START_TIME_UNIX, "ms"),
            col(cls.SCREEN_RECORDING_START_TIME_UTC).str.to_datetime(
                r"%m/%d/%Y %H:%M", time_zone="UTC", time_unit="ms"
            ),
            # boolean characteristics
            col(cls.TOUCH_TYPER).eq("Yes").cast(Boolean),
            # .replace({"Yes": True, "No": False}, return_dtype=Boolean),
            # enum characteristics
            *(
                col(c).cast(Enum([m.value for m in e.__members__.values()]))
                for c, e in cls.enums().items()
            ),
        )
        df = df.with_columns(
            # screen sizes
            screen_size=struct(w=cls.SCREEN_WIDTH, h=cls.SCREEN_HEIGHT),
            display_resolution=struct(w=cls.DISPLAY_WIDTH, h=cls.DISPLAY_HEIGHT),
            # screen recording start
            # rec_time=col("start_time"),
            # + col(cls.PID).map_elements(get_record_offset, Duration("us")),
        )
        df = df.drop(
            cls.SCREEN_WIDTH,
            cls.SCREEN_HEIGHT,
            cls.DISPLAY_WIDTH,
            cls.DISPLAY_HEIGHT,
            cls.DATE,
            cls.DURATION,
            # cls.SCREEN_RECORDING_START_TIME_UTC,
            # cls.SCREEN_RECORDING_START_TIME_UNIX,
        )
        return df.sort(by=cls.PID).rename(
            {k.value: v for k, v in cls.aliases().items() if k.value in df.columns}
        )

    @classmethod
    def read_zip(cls, ref: ZipFile):
        file = next(
            filter(
                lambda f: "participant_characteristics.csv" in f.filename, ref.filelist
            )
        )
        return cls.read(ref.open(file))


class Log(SourceEnumClass, StrEnum):
    SESSION_ID = "sessionId"
    WEBPAGE = "webpage"
    SESSION_STRING = "sessionString"
    EPOCH = "epoch"
    TIME = "time"
    TYPE = "type"
    EVENT = "event"
    POS = "pos"
    TEXT = "text"
    SCREEN_X = "screenX"
    SCREEN_Y = "screenY"
    CLIENT_X = "clientX"
    CLIENT_Y = "clientY"
    PAGE_X = "pageX"
    PAGE_Y = "pageY"
    SCROLL_X = "scrollX"
    SCROLL_Y = "scrollY"
    WINDOW_X = "windowX"
    WINDOW_Y = "windowY"
    WINDOW_INNER_WIDTH = "windowInnerWidth"
    WINDOW_INNER_HEIGHT = "windowInnerHeight"
    WINDOW_OUTER_WIDTH = "windowOuterWidth"
    WINDOW_OUTER_HEIGHT = "windowOuterHeight"
    IS_TRUSTED = "isTrusted"

    @classmethod
    def schema(cls):
        return {
            cls.IS_TRUSTED: Boolean,
            cls.SESSION_ID: String,
            cls.WEBPAGE: String,
            cls.SESSION_STRING: String,
            cls.EPOCH: UInt64,
            cls.TIME: Float64,
            cls.TYPE: String,
            cls.EVENT: String,
            cls.TEXT: String,
            cls.POS: Struct({"top": UInt32, "left": UInt32}),
            cls.SCREEN_X: Int32,
            cls.SCREEN_Y: Int32,
            cls.CLIENT_X: UInt16,
            cls.CLIENT_Y: UInt16,
            cls.PAGE_X: UInt32,
            cls.PAGE_Y: UInt32,
            cls.SCROLL_X: UInt32,
            cls.SCROLL_Y: UInt32,
            cls.WINDOW_X: Int32,
            cls.WINDOW_Y: Int32,
            cls.WINDOW_INNER_WIDTH: UInt16,
            cls.WINDOW_INNER_HEIGHT: UInt16,
            cls.WINDOW_OUTER_WIDTH: UInt16,
            cls.WINDOW_OUTER_HEIGHT: UInt16,
        }

    @classmethod
    def source_filename(cls, value: str) -> bool:
        return value.endswith(".json")

    @classmethod
    def scan_raw(cls, source: Path | bytes):
        return read_json(
            source,
            schema={k.value: v for k, v in cls.schema().items()},
            infer_schema_length=1,
        ).lazy()

    @classmethod
    def scan(cls, source: Path | bytes, pid: int):
        lf = cls.scan_raw(source)
        lf = lf.with_columns(
            col(cls.WEBPAGE)
            .str.strip_prefix("/study/")
            .str.strip_suffix(".htm")
            .replace("thankyou", "thank_you")
            .cast(Enum(Study.values())),
            col(cls.TYPE).cast(Categorical("lexical")),
            col(cls.EVENT).cast(Categorical("lexical")),
            col(cls.SESSION_ID).str.split("_").list.get(0).cast(UInt64),
            col(cls.SESSION_ID)
            .str.split("_")
            .list.get(1)
            .cast(UInt8)
            .alias(cls.SESSION_STRING),
            from_epoch(cls.EPOCH, "ms"),
            col(cls.TIME).cast(Duration("ms")),
        )
        lf = lf.with_columns(
            when(col(cls.TYPE).is_not_null())
            .then(
                col(cls.TYPE).replace(
                    {k.value: v for k, v in Event.type_aliases().items()}
                )
            )
            .when(col(cls.EVENT).is_not_null() & col(cls.EVENT).eq(Event.VIDEO_SAVE))
            .then(lit("save"))
            .otherwise(None)
            .cast(Categorical("lexical"))
            .alias(cls.EVENT),
            when(
                col(cls.TYPE).is_in(Event.recording())
                | col(cls.EVENT).is_in(Event.video())
            )
            .then(lit(Source.LOG))
            .when(col(cls.TYPE).is_in(Event.mouse()))
            .then(lit(Source.MOUSE))
            .when(col(cls.TYPE) == Event.SCROLL_EVENT)
            .then(lit(Source.SCROLL))
            .when(col(cls.TYPE) == Event.TEXT_INPUT)
            .then(lit(Source.INPUT))
            .when(col(cls.TYPE) == Event.TEXT_SUBMIT)
            .then(lit(Source.TEXT))
            .otherwise(lit(Source.LOG))
            .cast(Categorical("lexical"))
            .alias(cls.TYPE),
        )
        return lf.sort(cls.EPOCH).select(
            pid=lit(pid, UInt8),
            record=cls.SESSION_STRING,
            timestamp=col(cls.EPOCH),  # - col(cls.SESSION_ID).cast(Datetime("ms")),
            study=cls.WEBPAGE,
            event=cls.EVENT,
            source=cls.TYPE,
            trusted=cls.IS_TRUSTED,
            duration=cls.TIME,
            caret=cls.POS,
            text=cls.TEXT,
            page=struct(x=cls.PAGE_X, y=cls.PAGE_Y),
            mouse=struct(x=cls.SCREEN_X, y=cls.SCREEN_Y),
            scroll=struct(x=cls.SCROLL_X, y=cls.SCROLL_Y),
            window=struct(x=cls.WINDOW_X, y=cls.WINDOW_Y),
            inner=struct(w=cls.WINDOW_INNER_WIDTH, h=cls.WINDOW_INNER_HEIGHT),
            outer=struct(w=cls.WINDOW_OUTER_WIDTH, h=cls.WINDOW_OUTER_HEIGHT),
        )


class Tobii(SourceEnumClass, StrEnum):
    RIGHT_PUPIL_VALIDITY = auto()
    RIGHT_GAZE_POINT_ON_DISPLAY_AREA = auto()
    LEFT_GAZE_ORIGIN_VALIDITY = auto()
    SYSTEM_TIME_STAMP = auto()
    RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM = auto()
    LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM = auto()
    LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM = auto()
    LEFT_PUPIL_VALIDITY = auto()
    RIGHT_PUPIL_DIAMETER = auto()
    TRUE_TIME = auto()
    LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM = auto()
    RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM = auto()
    LEFT_PUPIL_DIAMETER = auto()
    RIGHT_GAZE_ORIGIN_VALIDITY = auto()
    LEFT_GAZE_POINT_VALIDITY = auto()
    RIGHT_GAZE_POINT_VALIDITY = auto()
    LEFT_GAZE_POINT_ON_DISPLAY_AREA = auto()
    RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM = auto()
    DEVICE_TIME_STAMP = auto()

    @classmethod
    def schema(cls):
        return {
            cls.DEVICE_TIME_STAMP: UInt64,
            cls.SYSTEM_TIME_STAMP: UInt64,
            cls.TRUE_TIME: Float64,
            cls.LEFT_PUPIL_VALIDITY: Int32,
            cls.RIGHT_PUPIL_VALIDITY: Int32,
            cls.LEFT_GAZE_ORIGIN_VALIDITY: Int32,
            cls.RIGHT_GAZE_ORIGIN_VALIDITY: Int32,
            cls.LEFT_GAZE_POINT_VALIDITY: Int32,
            cls.RIGHT_GAZE_POINT_VALIDITY: Int32,
            cls.LEFT_PUPIL_DIAMETER: Float64,
            cls.RIGHT_PUPIL_DIAMETER: Float64,
            cls.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
            cls.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
            cls.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM: Array(Float64, 3),
            cls.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM: Array(Float64, 3),
            cls.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
            cls.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
            cls.LEFT_GAZE_POINT_ON_DISPLAY_AREA: Array(Float64, 2),
            cls.RIGHT_GAZE_POINT_ON_DISPLAY_AREA: Array(Float64, 2),
        }

    @classmethod
    def source_filename(cls, value: str) -> bool:
        return search(r"P_[0-9][0-9]\.txt$", value) is not None

    @classmethod
    def scan_raw(cls, source: Path | bytes | IO[bytes]):
        # handle corrupted files
        # polars cannot ignore parsing errors: https://github.com/pola-rs/polars/issues/13768
        test_end = b"}\n"
        # is_file = False
        if isinstance(source, (bytes, bytearray, memoryview)):
            if source[-3:-1] != test_end:
                if isinstance(source, memoryview):
                    source = source.tobytes()
                source = b"".join(source.splitlines()[:-1])
        else:
            if isinstance(source, Path):
                source = source.open("rb")
                # is_file = True
            source.seek(-2, SEEK_END)
            if source.read(2) != test_end:
                source.seek(0, SEEK_SET)
                source = b"".join(source.readlines()[:-1])
            else:
                source.seek(0, SEEK_SET)

        return scan_ndjson(
            source,
            schema={k.value: v for k, v in cls.schema().items()},
            infer_schema_length=1,
            low_memory=True,
        )

    @classmethod
    def scan(cls, source: Path | bytes | IO[bytes], pid: int):
        xyz = ("x", "y", "z")
        xy = ("x", "y")
        lf = cls.scan_raw(source)
        # print(pid, lf.count().collect())
        lf = lf.with_columns(
            from_epoch(col(cls.TRUE_TIME).mul(1e9), "ns"),
            from_epoch(cls.DEVICE_TIME_STAMP, "us"),
            from_epoch(cls.SYSTEM_TIME_STAMP, "us"),
            col(cls.LEFT_PUPIL_VALIDITY).cast(Boolean),
            col(cls.RIGHT_PUPIL_VALIDITY).cast(Boolean),
            col(cls.LEFT_GAZE_ORIGIN_VALIDITY).cast(Boolean),
            col(cls.RIGHT_GAZE_ORIGIN_VALIDITY).cast(Boolean),
            col(cls.LEFT_GAZE_POINT_VALIDITY).cast(Boolean),
            col(cls.RIGHT_GAZE_POINT_VALIDITY).cast(Boolean),
            col(cls.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.LEFT_GAZE_POINT_ON_DISPLAY_AREA).arr.to_struct(xy),
            col(cls.RIGHT_GAZE_POINT_ON_DISPLAY_AREA).arr.to_struct(xy),
        )
        return lf.sort(cls.TRUE_TIME).select(
            pid=lit(pid, UInt8),
            timestamp=cls.TRUE_TIME,
            clock=cls.SYSTEM_TIME_STAMP,
            duration=cls.DEVICE_TIME_STAMP,
            # pupil=struct(
            pupil_validity=struct(
                left=cls.LEFT_PUPIL_VALIDITY, right=cls.RIGHT_PUPIL_VALIDITY
            ),
            pupil_diameter=struct(
                left=cls.LEFT_PUPIL_DIAMETER, right=cls.RIGHT_PUPIL_DIAMETER
            ),
            # ),
            # gaze=struct(
            # point=struct(
            gazepoint_validity=struct(
                left=cls.LEFT_GAZE_POINT_VALIDITY,
                right=cls.RIGHT_GAZE_POINT_VALIDITY,
            ),
            gazepoint_display=struct(
                left=cls.LEFT_GAZE_POINT_ON_DISPLAY_AREA,
                right=cls.RIGHT_GAZE_POINT_ON_DISPLAY_AREA,
            ),
            gazepoint_ucs=struct(
                left=cls.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM,
                right=cls.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM,
            ),
            # ),
            # origin=struct(
            gazeorigin_validity=struct(
                left=cls.LEFT_GAZE_ORIGIN_VALIDITY,
                right=cls.RIGHT_GAZE_ORIGIN_VALIDITY,
            ),
            gazeorigin_trackbox=struct(
                left=cls.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM,
                right=cls.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM,
            ),
            gazeorigin_ucs=struct(
                left=cls.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM,
                right=cls.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM,
            ),
            # ),
            # ),
        )


Corner = tuple[
    Literal["back", "front"],
    Literal["lower", "upper"],
    Literal["left", "right"],
]
Point3 = tuple[float, float, float]


class Spec(SourceClass):
    def __init__(
        self,
        tracking_box: dict[Corner, Point3],
        ilumination_mode: str | None = None,
        frequency: float | None = None,
    ):
        self.tracking_box = tracking_box
        self.ilumination_mode = ilumination_mode
        self.frequency = frequency

    @classmethod
    def from_lines(cls, lines: Iterable[bytes]) -> Self:
        tracking_box: dict[Corner, Point3] = {}
        ilumination_mode: str | None = None
        frequency: float | None = None

        for line in lines:
            line = line.decode("utf-8").strip()
            m = match(r"(Back|Front) (Lower|Upper) (Left|Right):", line)
            if m is not None:
                index = cast(
                    Corner, (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())
                )

                values = line[m.end() :].strip(" ()\n\t").split(", ")
                x, y, z = [float(v) for v in values]

                tracking_box[index] = (x, y, z)

            m = match(r"Illumination mode:", line)
            if m is not None:
                ilumination_mode = line[m.end() :].strip()

            m = match(r"Initial gaze output frequency: (\d+\.?\d+)", line)
            if m is not None:
                frequency = float(m.group(1))

        return cls(
            tracking_box=tracking_box,
            ilumination_mode=ilumination_mode,
            frequency=frequency,
        )

    @classmethod
    def schema(cls):
        point = Array(Float64, 3)
        return {
            "tracking_box": Struct(
                {
                    "b_lo_l": point,
                    "b_lo_r": point,
                    "b_up_l": point,
                    "b_up_r": point,
                    "f_lo_l": point,
                    "f_lo_r": point,
                    "f_up_l": point,
                    "f_up_r": point,
                }
            ),
            "illumination_mode": Categorical(),
            "frequency": Float64,
        }

    def to_lf(self, pid: int | None = None) -> LazyFrame:
        lf = LazyFrame(
            {
                "tracking_box": [
                    {
                        f"{k[0][0]}_{k[1][:2]}_{k[2][0]}": [v[0], v[1], v[2]]
                        for k, v in self.tracking_box.items()
                    }
                ],
                "illumination_mode": [self.ilumination_mode],
                "frequency": [self.frequency],
            },
            self.schema(),
        )
        if pid is not None:
            lf = lf.with_columns(pid=lit(pid, UInt8))
        return lf

    @classmethod
    def scan_raw(cls, source: Path | bytes | IO[bytes]):
        if isinstance(source, Path):
            with source.open("rb") as src:
                payload = src.readlines()
        elif isinstance(source, (bytes, bytearray)):
            payload = source.splitlines()
        elif isinstance(source, memoryview):
            payload = source.tobytes().splitlines()
        else:
            payload = cast(IO[bytes], source)

        return cls.from_lines(payload).to_lf()

    @classmethod
    def source_filename(cls, value: str) -> bool:
        return value.endswith("specs.txt")


class Calib(SourceEnumClass, StrEnum):
    VALIDITY_LEFT = auto()
    VALIDITY_RIGHT = auto()
    POINT_X = auto()
    POINT_Y = auto()
    PREDICTION_X_LEFT = auto()
    PREDICTION_Y_LEFT = auto()
    PREDICTION_X_RIGHT = auto()
    PREDICTION_Y_RIGHT = auto()

    @classmethod
    def schema(cls):
        return {
            cls.POINT_X: Float64,
            cls.POINT_Y: Float64,
            cls.PREDICTION_X_LEFT: Float64,
            cls.PREDICTION_Y_LEFT: Float64,
            cls.VALIDITY_LEFT: Int8,
            cls.PREDICTION_X_RIGHT: Float64,
            cls.PREDICTION_Y_RIGHT: Float64,
            cls.VALIDITY_RIGHT: Int8,
        }

    @classmethod
    def scan_raw(cls, source: Path | bytes | IO[bytes]):
        return scan_csv(
            source,
            has_header=True,
            separator="\t",
            schema={key.value: value for key, value in cls.schema().items()},
            ignore_errors=True,
            raise_if_empty=True,
        )

    @classmethod
    def scan(cls, source: Path | bytes | IO[bytes], pid: int):
        def xy(name: str):
            return struct(x=col(f"{name}_x"), y=col(f"{name}_y"))

        def lr(name: str):
            return struct(left=col(f"{name}_left"), right=col(f"{name}_right"))

        def pred(name: str):
            return struct(x=col(f"prediction_x_{name}"), y=col(f"prediction_y_{name}"))

        lf = cls.scan_raw(source).drop_nulls()
        lf = lf.with_columns(col(cls.VALIDITY_LEFT) > 0, col(cls.VALIDITY_RIGHT) > 0)
        return lf.select(
            pid=lit(pid, UInt8),
            point=xy("point"),
            validity=lr("validity"),
            left=pred("left"),
            right=pred("right"),
        )

    @classmethod
    def source_filename(cls, value: str) -> bool:
        return value.endswith("specs.txt")


class Dot(SourceEnumClass, StrEnum):
    X = "Dot_X"
    Y = "Dot_Y"
    EPOCH = "Epoch"

    @classmethod
    def schema(cls):
        return {
            cls.X: UInt16,
            cls.Y: UInt16,
            cls.EPOCH: Float64,
        }

    @classmethod
    def scan_raw(cls, source: Path | bytes | IO[bytes]):
        return scan_csv(
            source,
            has_header=True,
            separator="\t",
            schema={key.value: value for key, value in cls.schema().items()},
            ignore_errors=True,
            infer_schema_length=0,
            raise_if_empty=True,
        )

    @classmethod
    def scan(cls, source: Path | bytes | IO[bytes], pid: int):
        return (
            cls.scan_raw(source)
            .sort(cls.EPOCH)
            .select(
                pid=lit(pid, UInt8),
                timestamp=col(cls.EPOCH).cast(Datetime("ms")),
                dot=struct(x=cls.X, y=cls.Y),
            )
        )

    @classmethod
    def source_filename(cls, value: str) -> bool:
        return value.endswith("final_dot_test_locations.tsv")


def ffmpeg_blank(output: str, duration: timedelta, *, width=640, height=480):
    dur = round(duration.total_seconds(), 3)
    video = ff.input(
        f"color=black:size={width}x{height}:duration={dur:.3f}",
        f="lavfi",
    )
    audio = ff.input(
        f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={dur:.3f}",
        f="lavfi",
    )
    return ff.output(
        video, audio, output, format="mp4"  # , movflags="frag_keyframe+empty_moov",
    )
