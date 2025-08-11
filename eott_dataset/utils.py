from typing import TYPE_CHECKING, Mapping
from enum import StrEnum
from pathlib import Path
from os import environ
from functools import cached_property
from io import TextIOBase

if TYPE_CHECKING:
    from polars import PolarsDataType


__all__ = ["Name", "get_dataset_root", "print_schema"]


def get_dataset_root():
    return Path(environ.get("EOTT_DATASET_PATH") or Path.cwd()).expanduser().resolve()


def print_schema(
    name: str,
    schema: Mapping[str, "PolarsDataType"],
    /,
    out: TextIOBase | None = None,
    *,
    indent: str = "\t",
    depth: int = 0,
):
    from polars import Struct

    assert indent.isspace() or len(indent) == 0
    assert depth >= 0

    if depth == 0:
        print(name, end=":\n", file=out)
    else:
        print(end="\n", file=out)

    for field, data_type in schema.items():
        print(f"{indent * (depth + 1)}{field}:", end="", file=out)
        if isinstance(data_type, Struct):
            print_schema(
                field, data_type.to_schema(), out=out, indent=indent, depth=depth + 1
            )
        else:
            print(f" {data_type}", file=out)


class Name(StrEnum):
    @classmethod
    def from_id(cls, i: int):
        return cls.__members__[cls._member_names_[i]]

    @classmethod
    def values(cls):
        return [item.value for item in cls.__members__.values()]

    @cached_property
    def id(self):
        return self.__class__._member_names_.index(self.name)
