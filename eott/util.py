from typing import cast, overload, Literal, Mapping, Iterable
from os import PathLike
from re import findall
from io import TextIOBase
from subprocess import run
from enum import StrEnum
from datetime import timedelta
from functools import cached_property, partial

from polars import Struct
from polars._typing import PolarsDataType

import ffmpeg as ff


__all__ = [
    "NameEnum",
    "ffmpeg_blank",
    "ffmpeg_args",
    "play_with_mpv",
    "print_schema",
    "parse_timedelta",
    "printf",
    "println",
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
    schema: Mapping[str, PolarsDataType],
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
    return run(["mpv", "--osd-fractions", "--osd-level=2", src], shell=True)


def ffmpeg_blank(
    output: str, duration: timedelta, *, width=640, height=480, fps: int = 30
):
    dur = round(duration.total_seconds(), 3)
    video = ff.input(
        f"color=white:size={width}x{height}:rate={fps}:duration={dur:.3f}",
        f="lavfi",
    )
    audio = ff.input(
        f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={dur:.3f}",
        f="lavfi",
    )
    return ff.output(
        video, audio, output, format="mp4"  # , movflags="frag_keyframe+empty_moov",
    )


printf = partial(print, flush=True, end="")
println = partial(print, flush=True)
