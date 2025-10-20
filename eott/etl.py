from typing import cast, Any
from pathlib import Path
from datetime import datetime
from os import remove as remove_file
from contextlib import suppress
from zipfile import ZipFile
from datetime import timedelta
from collections import defaultdict
from shutil import move as move_file
from functools import partial

from polars import LazyFrame, PartitionByKey, col, struct
from decord import VideoReader

import ffmpeg as ff

from .raw import (
    ZipDataset,
    Webcam,
    Study,
    DataType,
    Alignment,
    Tobii,
    Dot,
    Calib,
    pid_from_name,
)
from .util import printf, println, ffmpeg_blank


__all__ = ["extract_transform_load", "merge_webcam_videos"]


def _sink_data(
    lf: LazyFrame, dst: Path, params: dict[str, Any], *, overwrite: bool, dry_run: bool
):
    printf(f"Saving {dst.stem} to {dst} ... ")
    if not dry_run:
        if not dst.exists() or overwrite:
            lf.sink_parquet(dst, **params)
            println("Done.")
        else:
            println("Skipped. (already exists)")
    else:
        println("Skipped. (dry run)")


def _process_screen_videos(
    ds: "ZipDataset", dstdir: Path, tmpdir: Path, *, dry_run: bool, overwrite: bool
):
    dstdir.mkdir(exist_ok=True)

    for file in ds.screen_files():
        dst = dstdir / f"P_{pid_from_name(file.filename, error=True):02}.mp4"
        printf(
            f"Extracting screen video from {file.filename.removeprefix(ds.prefix)} into {dst.relative_to(dstdir)}..."
            + (" " if not dry_run else "\n"),
        )

        input_path = None
        if dst.exists() and not overwrite:
            println("Skipped. (already exists)")
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
                    with ds.ref.open(file) as zf, open(input_path, "wb") as f:
                        _ = f.write(zf.read())

                f = ff.input(input_path)
                f = ff.output(
                    f.audio,
                    ff.filter(ff.filter(f.video, "fps", 25), "scale", "iw/2", "ih/2"),
                    dst,
                    format="mp4",
                )
            case _:
                raise ValueError(f"Unsupported screen video format: {file.filename}")

        if dry_run:
            print(
                f"{file.filename.removeprefix(ds.prefix)} -> {' '.join(ff.get_args(f))}"
            )
            continue

        try:
            _ = ff.run(
                f,
                input=None if input_path else ds.ref.read(file),
                capture_stdout=True,
                capture_stderr=True,
                overwrite_output=True,
            )
            println("Done.")
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


def _process_webcam_videos(
    ds: "ZipDataset", dstdir: Path, *, dry_run: bool, overwrite: bool
):
    dstdir.mkdir(exist_ok=True)

    fragmented = defaultdict[tuple[int, int], list[Path]](list)
    for file in ds.webcam_files():
        w = cast(Webcam, Webcam.parse_raw(file.filename))
        pid = pid_from_name(file.filename, error=True)

        printf(
            f"Extracting webcam video from {file.filename.removeprefix(ds.prefix)} ",
        )

        dst = (dstdir / f"P_{pid:02}" / f"{w.record:02}-{w.study.value}").with_suffix(
            ".mp4"
        )

        if w.aux is not None:
            if dst.exists():
                println("Skipped. (already exists)")
                continue
            dst = dst.with_stem(f"{dst.stem}-{w.aux}")
            fragmented[(pid, w.record)].append(dst)

        printf(
            f"into {dst.relative_to(dstdir)} ... ",
        )

        if dst.exists() and not overwrite:
            println("Skipped. (already exists)")
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
                input=ds.ref.read(file),
                capture_stdout=True,
                capture_stderr=True,
                overwrite_output=overwrite,
            )
            println("Done.")
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

    return fragmented


def _concat_fragmented_webcam_videos(
    pid: int,
    record: int,
    paths: list[Path],
    dstdir: Path,
    *,
    dry_run: bool,
    overwrite: bool,
):
    # sort by aux number
    paths.sort(key=lambda p: int(p.stem[-1]))
    # get path for file with no aux number
    path = (path := paths[0]).with_stem(path.stem.removesuffix("-1"))
    first_path = path.with_stem(path.stem + "-0")

    if not dry_run:
        # path will be concatenated video
        _ = move_file(str(path), str(first_path))
    else:
        print(
            f"Would move {path.relative_to(dstdir)} to {first_path.relative_to(dstdir)} for concatenation"
        )

    paths.insert(0, first_path)
    f = ff.concat(*[ff.input(str(p)) for p in paths])
    f = ff.output(f, str(path))

    printf(
        "Concatenating aux webcam videos for participant",
        pid,
        "record",
        record,
        "... ",
    )
    if dry_run:
        println(" ".join(ff.get_args(f)))
        return

    try:
        _ = ff.run(
            f,
            capture_stdout=True,
            capture_stderr=True,
            overwrite_output=overwrite,
        )
        println(" Done.")

        for p in map(str, paths):
            with suppress(OSError):
                remove_file(p)
            del p

        del f

    except ff.Error as e:
        with suppress(OSError):
            remove_file(str(path))
            _ = move_file(str(first_path), str(path))
        raise RuntimeError(e.stderr.decode()) from e

    except KeyboardInterrupt as e:
        with suppress(OSError):
            remove_file(str(path))
            _ = move_file(str(first_path), str(path))
        raise e


def merge_webcam_videos(
    plf: LazyFrame, llf: LazyFrame, dstdir: Path, *, dry_run: bool, overwrite: bool
):
    interrupted = False
    pid: int
    webcam_first_start: datetime
    for pid, webcam_first_start in (
        plf.select("pid", "webcam_start").collect().iter_rows()
    ):
        if interrupted:
            break

        df = (
            llf.filter(pid=pid, event="start")
            .select("record", "timestamp", "study")
            .sort("timestamp")
            .collect()
        )

        previous_end = timedelta()
        record: int
        webcam_part_start: datetime
        previous_path: Path | None = None
        # print(pid, webcam_first_start)
        for record, webcam_part_start, study in df.iter_rows():
            study = Study(study)
            path = (dstdir / f"P_{pid:02}" / f"{record:02}-{study.value}").with_suffix(
                ".mp4"
            )
            start_offset = webcam_part_start - webcam_first_start

            if not path.exists():
                println(
                    f"Webcam video for {path} does not exist, skipping...",
                )
                continue

            vr = VideoReader(str(path))
            dur = timedelta(seconds=float(vr.get_frame_timestamp(-1)[1]))
            gap = start_offset - previous_end
            # println(record, study, previous_end, "|", start_offset, "|", dur, "|", gap)
            if gap < timedelta():
                del vr
                remove_file(str(path))
                continue

            previous_end = start_offset + dur

            if previous_path is None or gap < timedelta(milliseconds=30):
                previous_path = path
                continue

            blank_path = path.with_stem(previous_path.stem + "-gap")

            if dry_run:
                println(
                    f"Would create blank video for {blank_path.relative_to(dstdir)} with duration {gap.total_seconds()} seconds",
                )
                previous_path = path
                continue

            printf(
                f"Creating blank video for {blank_path.relative_to(dstdir)} with duration {gap.total_seconds()} seconds... ",
            )

            if blank_path.exists() and not overwrite:
                println("Skipped. (already exists)")
                previous_path = path
                continue

            try:
                _ = ff.run(
                    ffmpeg_blank(
                        str(blank_path),
                        gap,
                    ),
                    capture_stdout=True,
                    capture_stderr=True,
                    overwrite_output=overwrite,
                )
                println("Done.")
            except ff.Error as e:
                println(f"Error running ffmpeg: {e.stderr.decode('utf-8')}")
            except KeyboardInterrupt:
                interrupted = True
                break
            finally:
                previous_path = path

            del vr, dur, gap, record, study, start_offset, path
        del previous_end, webcam_first_start

        input_dir = dstdir / f"P_{pid:02}"
        output_path = input_dir.with_suffix(".mp4")

        # for p in sorted(input_dir.glob("*.mp4"), key=lambda p: p.stem):
        #     print(p)

        if not dry_run:
            printf(f"Concatenating webcam videos for participant {pid} ... ")

            if output_path.exists() and not overwrite:
                println("Skipped. (already exists)")
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
                println(f"Error running ffmpeg: {e.stderr.decode('utf-8')}")

            del inputs
        else:
            println(
                f"Would concatenate webcam videos for participant {pid} into {output_path.relative_to(dstdir)}",
            )


def extract_transform_load(
    zip: ZipFile,
    process: set[DataType],
    dstdir: Path,
    tmpdir: Path,
    *,
    aligns_path: Path | None = None,
    sync_video: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
):
    ds = ZipDataset(zip)

    printf("Scanning participants and log files... ")
    plf, llf = ds.scan_participants(
        sync_video, alf=Alignment.scan(aligns_path) if aligns_path else None
    )
    println("Done.")

    _sink = partial(_sink_data, overwrite=overwrite, dry_run=dry_run)
    sinkparams = dict(compression="uncompressed")

    if DataType.FORM in process:
        _sink(plf.drop("screen_viewport_y"), dstdir / "form.parquet", sinkparams)

    if DataType.LOG in process:
        _sink(llf, dstdir / "log.parquet", sinkparams)

    if DataType.TOBII in process:
        base_path = dstdir / "tobii"
        tlf = Tobii.scan_zip(zip)
        printf(f"Saving tobii to {base_path} ... ")
        if dry_run:
            println("Skipped. (dry run)")
        else:
            tlf.sink_parquet(
                PartitionByKey(
                    base_path,
                    file_path=lambda ctx: f"P_{ctx.keys[0].raw_value:02d}.parquet",
                    by="pid",
                    per_partition_sort_by="timestamp",
                    include_key=True,
                ),
                compression="lz4",
                mkdir=True,
            )
            println("Done.")

    if DataType.DOT in process:
        lf = Dot.scan_zip(zip).with_columns(col("dot").struct.unnest()).drop("dot")
        lf = lf.join(plf.select("pid", oy="screen_viewport_y"), "pid", "left")
        lf = lf.with_columns(y=col("y") + col("oy"))
        lf = lf.with_columns(dot=struct("x", "y")).drop("x", "y", "oy")
        _sink(lf, dstdir / "dot.parquet", sinkparams)

    if DataType.CALIB in process:
        _sink(Calib.scan_zip(zip), dstdir / "calib.parquet", sinkparams)

    if DataType.SCREEN in process:
        _process_screen_videos(
            ds, dstdir / "screen", tmpdir, dry_run=dry_run, overwrite=overwrite
        )

    if DataType.WEBCAM in process:
        fragmented = _process_webcam_videos(
            ds,
            webcam_dstdir := dstdir / "raw",
            dry_run=dry_run,
            overwrite=overwrite,
        )

        for (pid, record), paths in fragmented.items():
            _concat_fragmented_webcam_videos(
                pid, record, paths, webcam_dstdir, dry_run=dry_run, overwrite=overwrite
            )
            del pid, record, paths
        del fragmented
