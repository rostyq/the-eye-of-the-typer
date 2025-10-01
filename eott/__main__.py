from typing import Optional, Annotated
from pathlib import Path
from zipfile import ZipFile
from time import time
from datetime import timedelta

from typer import Typer, Argument, Option, BadParameter, Context, CallbackParam

from eott import DataType


def validate_output_path(
    ctx: Context, param: CallbackParam, value: Optional[Path]
) -> Optional[Path]:
    if ctx.resilient_parsing:
        return
    if value is not None and not value.is_dir():
        raise BadParameter("Output path must be a directory", ctx=ctx, param=param)
    return value


app = Typer()


@app.command(name="etl")
def _(
    dataset_path: Annotated[Path, Argument(help="Path to the dataset zip file")],
    output_path: Annotated[
        Optional[Path],
        Option(
            help="Output directory for transformed files", callback=validate_output_path
        ),
    ] = None,
    alignment_path: Annotated[
        Optional[Path],
        Option(
            help="Path to the alignment CSV file to fix webcam recording start timestamps",
        ),
    ] = None,
    process: Annotated[
        list[DataType],
        Option(
            help="Data types to process. Defaults to all types", case_sensitive=False
        ),
    ] = list(DataType.__members__.values()),
    dry_run: Annotated[
        bool, Option(help="Run without making changes, just show what would be done")
    ] = False,
    concat: Annotated[
        bool,
        Option(
            help="Concatenate webcam video files into a single file for each participant",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        Option(
            help="Force overwrite existing files in the output directory",
        ),
    ] = False,
    sync: Annotated[
        bool,
        Option(
            help="Synchronize timestamps between screen and webcam data",
        ),
    ] = True,
):
    from tempfile import TemporaryDirectory
    from eott.raw import extract_transform_load

    if output_path is None:
        output_path = dataset_path.parent / "eott"

    print(f"Start extracting dataset from {dataset_path} to {output_path}", flush=True)
    with ZipFile(dataset_path, "r") as zip, TemporaryDirectory(
        prefix="eott-", ignore_cleanup_errors=True
    ) as tmpdir:
        start_time = time()
        extract_transform_load(
            zip,
            output_path,
            Path(tmpdir),
            aligns_path=alignment_path,
            form=DataType.FORM in process,
            log=DataType.LOG in process,
            tobii=DataType.TOBII in process,
            dot=DataType.DOT in process,
            calib=DataType.CALIB in process,
            screen=DataType.SCREEN in process,
            webcam=DataType.WEBCAM in process,
            dry_run=dry_run,
            concat_webcam=concat,
            sync=sync,
            overwrite=overwrite,
        )
        elapsed_time = timedelta(seconds=time() - start_time)
        print(
            f"ETL process completed in {elapsed_time}. Files saved to {output_path}",
            flush=True,
        )


app()
