from typing import Annotated
from enum import StrEnum
from pathlib import Path
from zipfile import ZipFile
from time import time
from datetime import timedelta

from typer import Typer, Argument, Option, BadParameter, Context, CallbackParam

from eott import DataType


def validate_output_path(
    ctx: Context, param: CallbackParam, value: Path | None
) -> Path | None:
    if ctx.resilient_parsing:
        return
    if value is not None and not value.is_dir():
        raise BadParameter("Output path must be a directory", ctx=ctx, param=param)
    return value


app = Typer()


class Prepare(StrEnum):
    WEBCAM = "webcam"
    CALIBRATION = "calibration"


PROCESSES = list(DataType)
PREPARE = list(Prepare)

DryRun = Annotated[
    bool, Option(help="Run without making changes, just show what would be done")
]
Overwrite = Annotated[
    bool, Option(help="Force overwrite existing files in the output directory")
]
OutputPath = Annotated[
    Path | None,
    Option(
        help="Output directory for transformed files",
        callback=validate_output_path,
    ),
]


@app.command(name="etl")
def _(
    dataset_path: Annotated[Path, Argument(help="Path to the dataset zip file")],
    alignment_path: Annotated[
        Path | None,
        Option(
            help="Path to the alignment CSV file to fix webcam recording start timestamps",
        ),
    ] = None,
    process: Annotated[
        list[DataType],
        Option(
            help="Data types to process. Defaults to all types",
            case_sensitive=False,
        ),
    ] = PROCESSES,
    sync_video: Annotated[
        bool,
        Option(
            help="Synchronize timestamps between screen and webcam data",
        ),
    ] = True,
    output_path: OutputPath = None,
    dry_run: DryRun = False,
    overwrite: Overwrite = False,
):
    from tempfile import TemporaryDirectory
    from eott.util import println
    from eott.etl import extract_transform_load

    if output_path is None:
        output_path = dataset_path.parent / "eott"

    println(f"Start extracting dataset from {dataset_path} to {output_path}")
    with (
        ZipFile(dataset_path, "r") as zip,
        TemporaryDirectory(prefix="eott-", ignore_cleanup_errors=True) as tmpdir,
    ):
        start_time = time()

        output_path.mkdir(parents=True, exist_ok=True)

        extract_transform_load(
            zip,
            set(process),
            output_path,
            Path(tmpdir),
            aligns_path=alignment_path,
            dry_run=dry_run,
            sync_video=sync_video,
            overwrite=overwrite,
        )
        elapsed_time = timedelta(seconds=time() - start_time)
        println(
            f"ETL process completed in {elapsed_time}. Files saved to {output_path}",
        )


@app.command("prepare")
def _(
    input_path: Annotated[
        Path, Argument(help="Path to the extracted dataset directory")
    ],
    process: Annotated[
        list[Prepare],
        Option(
            help="Processes to run. Defaults to all",
            case_sensitive=False,
        ),
    ] = PREPARE,
    output_path: OutputPath = None,
    dry_run: DryRun = False,
    overwrite: Overwrite = False,
):
    from eott import DirDataset
    from eott.etl import merge_webcam_videos
    from eott.util import println

    from polars import col, PartitionByKey, lit, Float64

    ds = DirDataset(input_path)
    output_path = input_path / "webcam" if output_path is None else output_path

    start_time = time()

    plf, llf = ds.lazyframe("form"), ds.lazyframe("log")

    if "webcam" in process:
        merge_webcam_videos(plf, llf, output_path, dry_run=dry_run, overwrite=overwrite)
    if "calibration" in process:
        lf = llf.select(
            "pid", "study", "timestamp", col("mouse").struct.unnest(), "event"
        )
        lf = lf.filter(col("study").is_in(["dot_test", "fitts_law"]))
        lf = lf.drop_nulls(["x", "y"])
        lf = lf.join(plf.select("pid", "webcam_start"), on="pid", how="left")
        lf = lf.with_columns(
            timestamp=col("timestamp").sub("webcam_start").dt.total_milliseconds(),
            weight=col("event").replace_strict(
                {"move": 0.5, "click": 1.0}, default=lit(0.0), return_dtype=Float64
            ),
        )
        lf = lf.rename({"x": "screen_x", "y": "screen_y"})
        # lf = lf.drop("study", "event", "webcam_start")
        lf.sink_csv(
            PartitionByKey(
                output_path,
                file_path=lambda ctx: f"P_{ctx.keys[0].raw_value:02d}.csv",
                by="pid",
                per_partition_sort_by="timestamp",
                include_key=False,
            )
        )

    elapsed_time = timedelta(seconds=time() - start_time)

    println(
        f"Preparing process completed in {elapsed_time}. Files saved to {output_path}",
    )


app()
