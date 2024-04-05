from enum import StrEnum as _StrEnum, auto as _auto


class CharacteristicColumn(_StrEnum):
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


class LogField(_StrEnum):
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


class TobiiField(_StrEnum):
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


class SpecColumn(_StrEnum):
    VALIDITY_LEFT = _auto()
    VALIDITY_RIGHT = _auto()
    POINT_X = _auto()
    POINT_Y = _auto()
    PREDICTION_X_LEFT = _auto()
    PREDICTION_Y_LEFT = _auto()
    PREDICTION_X_RIGHT = _auto()
    PREDICTION_Y_RIGHT = _auto()


class DotColumn(_StrEnum):
    DOT_X = "Dot_X"
    DOT_Y = "Dot_Y"
    EPOCH = "Epoch"
