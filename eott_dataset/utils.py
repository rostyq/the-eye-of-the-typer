from enum import StrEnum
from functools import cached_property


__all__ = ["Name"]


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
