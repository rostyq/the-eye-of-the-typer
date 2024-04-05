from enum import StrEnum as _StrEnum, auto as _auto
from functools import cached_property as _cached_property


class _Name(_StrEnum):
    @classmethod
    def from_id(cls, i: int):
        return cls.__members__[cls._member_names_[i]]

    @classmethod
    def values(cls):
        return [item.value for item in cls.__members__.values()]

    @_cached_property
    def id(self):
        return self.__class__._member_names_.index(self.name)


class CharacteristicColumn(_Name):
    PID = "Participant ID"
    LOG_ID = "Participant Log ID"
    DATE = "Date"
    SETTING = "Setting"
    DISPLAY_WIDTH = "Display Width (pixels)"
    DISPLAY_HEIGHT = "Display Height (pixels)"
    SCREEN_WIDTH = "Screen Width (cm)"
    SCREEN_HEIGHT = "Screen Height (cm)"
    DISTANCE_FROM_SCREEN = "Distance From Screen (cm)"
    SCREEN_RECORDING_START_TIME_UNIX = "Screen Recording Start Time (Unix milliseconds)"
    SCREEN_RECORDING_START_TIME_UTC = "Screen Recording Start Time (Wall Clock)"
    GENDER = "Gender"
    AGE = "Age"
    RACE = "Self-Reported Race"
    SKIN_COLOR = "Self-Reported Skin Color"
    EYE_COLOR = "Self-Reported Eye Color"
    FACIAL_HAIR = "Facial Hair"
    VISION = "Self-Reported Vision"
    TOUCH_TYPER = "Touch Typer"
    HANDEDNESS = "Self-Reported Handedness"
    WEATHER = "Weather"
    POINTING_DEVICE = "Pointing Device"
    NOTES = "Notes"
    TIME_OF_DAY = "Time of day"
    DURATION = "Duration"


class LogField(_Name):
    SESSION_ID = "sessionId"
    WEBPAGE = "webpage"
    SESSION_STRING = "sessionString"
    EPOCH = "epoch"
    TIME = "time"
    TYPE = "type"
    EVENT = "event"
    POS = "pos"
    TEXT = "text"
    SCREEN_X = "screenX"
    SCREEN_Y = "screenY"
    CLIENT_X = "clientX"
    CLIENT_Y = "clientY"
    PAGE_X = "pageX"
    PAGE_Y = "pageY"
    SCROLL_X = "scrollX"
    SCROLL_Y = "scrollY"
    WINDOW_X = "windowX"
    WINDOW_Y = "windowY"
    WINDOW_INNER_WIDTH = "windowInnerWidth"
    WINDOW_INNER_HEIGHT = "windowInnerHeight"
    WINDOW_OUTER_WIDTH = "windowOuterWidth"
    WINDOW_OUTER_HEIGHT = "windowOuterHeight"
    IS_TRUSTED = "isTrusted"


class TobiiField(_Name):
    RIGHT_PUPIL_VALIDITY = _auto()
    RIGHT_GAZE_POINT_ON_DISPLAY_AREA = _auto()
    LEFT_GAZE_ORIGIN_VALIDITY = _auto()
    SYSTEM_TIME_STAMP = _auto()
    RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM = _auto()
    LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM = _auto()
    LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM = _auto()
    LEFT_PUPIL_VALIDITY = _auto()
    RIGHT_PUPIL_DIAMETER = _auto()
    TRUE_TIME = _auto()
    LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM = _auto()
    RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM = _auto()
    LEFT_PUPIL_DIAMETER = _auto()
    RIGHT_GAZE_ORIGIN_VALIDITY = _auto()
    LEFT_GAZE_POINT_VALIDITY = _auto()
    RIGHT_GAZE_POINT_VALIDITY = _auto()
    LEFT_GAZE_POINT_ON_DISPLAY_AREA = _auto()
    RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM = _auto()
    DEVICE_TIME_STAMP = _auto()


class SpecColumn(_Name):
    VALIDITY_LEFT = _auto()
    VALIDITY_RIGHT = _auto()
    POINT_X = _auto()
    POINT_Y = _auto()
    PREDICTION_X_LEFT = _auto()
    PREDICTION_Y_LEFT = _auto()
    PREDICTION_X_RIGHT = _auto()
    PREDICTION_Y_RIGHT = _auto()


class DotColumn(_Name):
    DOT_X = "Dot_X"
    DOT_Y = "Dot_Y"
    EPOCH = "Epoch"


class Study(_Name):
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


class Event(_Name):
    VIDEO_START = "video started"
    VIDEO_STOP = "video stop"
    VIDEO_SAVE = "video saved"


class Type(_Name):
    SCROLL_EVENT = "scrollEvent"
    MOUSE_MOVE = "mousemove"
    MOUSE_CLICK = "mouseclick"
    TEXT_INPUT = "textInput"
    TEXT_SUBMIT = "text_input"
    REC_START = "recording start"
    REC_STOP = "recording stop"


class Source(_Name):
    LOG = _auto()
    MOUSE = _auto()
    SCROLL = _auto()
    INPUT = _auto()
    TEXT = _auto()
    TOBII = _auto()
    WEBCAM = _auto()
    SCREEN = _auto()