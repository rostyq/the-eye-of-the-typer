from typing import TypeVar, Union, Optional
from os import PathLike
from pathlib import Path
from functools import cache

import polars as pl

from .utils import get_dataset_root
from .characteristics import *


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
    def filename(self, source: Source):
        return self.root.joinpath(source).with_suffix(SUFFIX)

    def describe(self, *sources: Source):
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

    def scan(self, source: Source, /, *, cache: bool = True, **kwargs):
        return pl.scan_parquet(self.filename(source), cache=cache, **kwargs)

    def timeline(self):
        from . import timeline as tl

        form, log = self.scan(Source.FORM), self.scan(Source.LOG)

        mouse = tl.get_source_timeline(self.scan(Source.MOUSE), form, Source.MOUSE)
        scroll = tl.get_source_timeline(self.scan(Source.SCROLL), form, Source.SCROLL)
        text = tl.get_source_timeline(self.scan(Source.TEXT), form, Source.TEXT)
        input_ = tl.get_source_timeline(self.scan(Source.INPUT), form, Source.INPUT)

        dot = tl.get_source_timeline(self.scan(Source.DOT), form, Source.DOT)

        screen = tl.get_screen_timeline(self.scan(Source.SCREEN), form)
        webcam = tl.get_webcam_timeline(self.scan(Source.WEBCAM), log)

        log = tl.get_source_timeline(self.scan(Source.MOUSE), form, Source.LOG)

        df = pl.concat([log, mouse, scroll, text, input_, dot, screen, webcam]).sort(
            "pid"
        )
        return (
            df.sort("pid", "offset")
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
        )

    def load(self, pid: int) -> dict[Source, pl.DataFrame]:
        return {s: self.scan(s).filter(pid=pid).collect() for s in Source.__members__.values()}
