from typing import TypeVar, Union, Optional
from os import PathLike
from pathlib import Path
from .utils import get_dataset_root
from .characteristics import *
from functools import cache

import polars as pl


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
        names = "record", "study"

        mouse = tl.get_source_timeline(self.scan(Source.MOUSE), form, *names)
        scroll = tl.get_source_timeline(self.scan(Source.SCROLL), form, *names)
        text = tl.get_source_timeline(self.scan(Source.TEXT), form, *names)
        input_ = tl.get_source_timeline(self.scan(Source.INPUT), form, *names)

        dot = tl.get_source_timeline(self.scan(Source.DOT), form)

        screen = self.scan(Source.SCREEN)
        screen = tl.get_screen_timeline(screen, form)

        webcam = self.scan(Source.WEBCAM)
        webcam = tl.get_webcam_timeline(webcam, log)

        log = tl.get_source_timeline(self.scan(Source.MOUSE), form, *names)

        return pl.concat([log, mouse, scroll, text, input_, dot, screen, webcam])
