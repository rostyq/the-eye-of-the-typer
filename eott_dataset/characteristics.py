from enum import StrEnum as _StrEnum, auto as _auto

from .utils import Name as _Name


class Setting(_StrEnum):
    LAPTOP = "Laptop"
    PC = "PC"


class Gender(_StrEnum):
    MALE = "Male"
    FEMALE = "Female"


class Race(_StrEnum):
    WHITE = "White"
    BLACK = "Black"
    ASIAN = "Asian"
    OTHER = "Other"


class SkinColor(_StrEnum):
    C1 = "1"
    C2 = "2"
    C3 = "3"
    C4 = "4"
    C5 = "5"


class EyeColor(_StrEnum):
    DARK_BROWN_TO_BROWN = "Dark Brown to Brown"
    GRAY_TO_BLUE_OR_PINK = "Gray to Blue or Pink"
    GREEN_HAZEL_TO_BLUE_HAZEL = "Green-Hazel to Blue-Hazel"
    GREEN_HAZEL_TO_GREEN = "Green-Hazel to Green"
    AMBER = "Amber"


class FacialHair(_StrEnum):
    BEARD = "Beard"
    LITTLE = "Little"
    NONE = "None"


class Vision(_StrEnum):
    NORMAL = "Normal"
    GLASSES = "Glasses"
    CONTACTS = "Contacts"


class Handedness(_StrEnum):
    LEFT = "Left"
    RIGHT = "Right"


class Weather(_StrEnum):
    CLOUDY = "Cloudy"
    INDOORS = "Indoors"
    SUNNY = "Sunny"


class PointingDevice(_StrEnum):
    TRACKPAD = "Trackpad"
    MOUSE = "Mouse"


class Source(_Name):
    """
    Enumeration class representing different sources of input.

    LOG: log events or unknown entries
    MOUSE: mouse events
    SCROLL: scroll events
    INPUT: text input events
    TEXT: text submit events
    TOBII: tobii recording
    WEBCAM: webcam recording
    SCREEN: screen recording
    DOT: dot test entries
    CALIB: tobii calibration data
    TRACKBOX: tobii tracking box data
    FORM: participant info and characteristics
    """
    LOG = _auto()
    MOUSE = _auto()
    SCROLL = _auto()
    INPUT = _auto()
    TEXT = _auto()
    TOBII = _auto()
    WEBCAM = _auto()
    SCREEN = _auto()
    DOT = _auto()
    CALIB = _auto()
    TRACKBOX = _auto()
    FORM = _auto()


CHARACTERISTICS = {
    "setting": Setting,
    "gender": Gender,
    "race": Race,
    "skin_color": SkinColor,
    "eye_color": EyeColor,
    "facial_hair": FacialHair,
    "vision": Vision,
    "handedness": Handedness,
    "weather": Weather,
    "pointing_device": PointingDevice,
}
