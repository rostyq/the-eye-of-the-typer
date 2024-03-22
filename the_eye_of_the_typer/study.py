from enum import StrEnum as _StrEnum, auto as _auto
from functools import cached_property as _cached_property


class Study(_StrEnum):
    DOT_TEST_INSTRUCTIONS = _auto()
    DOT_TEST = _auto()
    FITTS_LAW_INSTRUCTIONS = _auto()
    FITTS_LAW = _auto()
    SERP_INSTRUCTIONS = _auto()
    BENEFITS_OF_RUNNING_INSTRUCTIONS = _auto()
    BENEFITS_OF_RUNNING = _auto()
    BENEFITS_OF_RUNNING_WRITING = _auto()
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES_INSTRUCTIONS = _auto()
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES = _auto()
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES_WRITING = _auto()
    WHERE_TO_FIND_MOREL_MUSHROOMS_INSTRUCTIONS = _auto()
    WHERE_TO_FIND_MOREL_MUSHROOMS = _auto()
    WHERE_TO_FIND_MOREL_MUSHROOMS_WRITING = _auto()
    TOOTH_ABSCESS_INSTRUCTIONS = _auto()
    TOOTH_ABSCESS = _auto()
    TOOTH_ABSCESS_WRITING = _auto()
    DOT_TEST_FINAL_INSTRUCTIONS = _auto()
    DOT_TEST_FINAL = _auto()
    THANK_YOU = _auto()

    @classmethod
    def from_position(cls, position: int):
        return cls.__members__[cls._member_names_[position]]

    @_cached_property
    def position(self):
        return self.__class__._member_names_.index(self.name)
