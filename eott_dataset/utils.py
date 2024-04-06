from enum import StrEnum
from pathlib import Path
from os import environ
from functools import cached_property


__all__ = ["Name"]


def get_dataset_root():
    return Path(environ.get("EOTT_DATASET_PATH") or Path.cwd()).expanduser().resolve()


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
