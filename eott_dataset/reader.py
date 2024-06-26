from typing import TypeVar, Union, Optional
from os import PathLike
from pathlib import Path
from functools import cache

import polars as pl

from .utils import get_dataset_root
from .types import SourceType
from .characteristics import *
from .participant import Participant


__all__ = ["Reader", "get_participant"]


EXTENSION = "parquet"
SUFFIX = f".{EXTENSION}"


F = TypeVar("F", pl.LazyFrame, pl.DataFrame)


def get_participant(df: F, *, value: int) -> F:
    return df.filter(pid=pl.lit(value, pl.UInt8)).drop("pid")


class Reader:
    root: Path

    def __init__(self, root: PathLike | None = None):
        self.root = Path(root) if root is not None else get_dataset_root()

    @cache
    def filename(self, source: Source | SourceType):
        return self.root.joinpath(source).with_suffix(SUFFIX)

    def describe(self, *sources: Source | SourceType):
        it = self.root.glob(f"*.{EXTENSION}")
        it = filter(lambda p: p.stem in Source.values(), it)
        it = map(lambda p: (Source(p.stem), p), it)
        if len(sources) > 0:
            it = filter(lambda x: x[0] in sources, it)
        return {s: pl.read_parquet_schema(p) for s, p in it}

    def read(
        self,
        source: Source,
        /,
        *,
        columns: Optional[Union[list[int], list[str]]] = None,
        **kwargs,
    ):
        return pl.read_parquet(self.filename(source), columns=columns, **kwargs)

    def scan(self, source: Source | SourceType, /, *, cache: bool = True, **kwargs):
        return pl.scan_parquet(self.filename(source), cache=cache, **kwargs)

    def timeline(self):
        from . import timeline as tl

        form, log = self.scan(Source.FORM), self.read(Source.LOG)

        lf = pl.concat(
            [
                tl.get_source_timeline(log.lazy(), form, Source.LOG),
                tl.get_source_timeline(self.scan(Source.TOBII), form, Source.TOBII),
                tl.get_source_timeline(self.scan(Source.MOUSE), form, Source.MOUSE),
                tl.get_source_timeline(self.scan(Source.SCROLL), form, Source.SCROLL),
                tl.get_source_timeline(self.scan(Source.TEXT), form, Source.TEXT),
                tl.get_source_timeline(self.scan(Source.INPUT), form, Source.INPUT),
                tl.get_source_timeline(self.scan(Source.DOT), form, Source.DOT),
                tl.get_screen_timeline(self.scan(Source.SCREEN), form),
                tl.get_webcam_timeline(self.scan(Source.WEBCAM), log.lazy()),
            ]
        )

        lf = lf.sort("pid", "offset")
        lf = lf.with_columns(pl.col("record", "study").forward_fill().over("pid"))

        lf = lf.with_columns(
            frame=pl.when(source="webcam").then(pl.col("index")).otherwise(None)
        )
        lf = lf.with_columns(pl.col("frame").forward_fill().over("pid", "record"))

        return lf

    def load(self, pid: int) -> dict[Source, pl.DataFrame]:
        return {
            s: self.scan(s).filter(pid=pid).drop("pid").collect()
            for s in Source.__members__.values()
        }

    def participant(self, pid: int) -> Participant:
        data = self.load(pid)

        form = data.pop(Source.FORM).row(0, named=True)
        form.update(pid=pid)

        screen = data.pop(Source.SCREEN)
        screen = screen["file"][0] if len(screen) >= 1 else None

        return Participant(form=form, screen=screen, data=data)

    @staticmethod
    def _extract_videofile(lf: pl.LazyFrame, dst: Path):
        payload = lf.first().collect()["file"][0]

        with Path(dst).open("wb") as f:
            f.write(payload)

    def extract_screen_recording(self, dst: Path, /, pid: int):
        lf = pl.scan_parquet(
            self.filename(Source.SCREEN),
            use_statistics=False,
            cache=False,
            low_memory=True,
        ).filter(pid=pid)
        Reader._extract_videofile(lf, dst)

    def extract_webcam_recording(
        self, dst: Path, /, pid: int, record: int, aux: int = 0
    ):
        lf = pl.scan_parquet(
            self.filename(Source.WEBCAM),
            use_statistics=False,
            cache=False,
            low_memory=True,
        )
        lf = lf.filter(pid=pid, record=record, aux=aux)
        Reader._extract_videofile(lf, dst)
