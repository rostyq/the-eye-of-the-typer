import polars as _pl

from .names import (
    CharacteristicColumn as _Char,
    LogField as _Log,
    TobiiField as _Tobii,
    SpecColumn as _Spec,
    DotColumn as _Dot,
)


PARTICIPANT_CHARACTERISTICS_SCHEMA: dict[_Char, tuple[type, str]] = {
    _Char.PID: (_pl.String, "pid"),
    _Char.LOG_ID: (_pl.UInt64, "log"),
    _Char.DATE: (_pl.String, "date"),
    _Char.SETTING: (_pl.Categorical, "setting"),
    _Char.DISPLAY_WIDTH: (_pl.UInt16, "display_width"),
    _Char.DISPLAY_HEIGHT: (_pl.UInt16, "display_height"),
    _Char.SCREEN_WIDTH: (_pl.Float64, "screen_width"),
    _Char.SCREEN_HEIGHT: (_pl.Float64, "screen_height"),
    _Char.DISTANCE_FROM_SCREEN: (_pl.Float64, "screen_distance"),
    _Char.SCREEN_RECORDING_START_TIME_UNIX: (_pl.UInt64, "screen_recording"),
    _Char.SCREEN_RECORDING_START_TIME_UTC: (_pl.String, "wall_clock"),
    _Char.GENDER: (_pl.Categorical, "gender"),
    _Char.AGE: (_pl.UInt8, "age"),
    _Char.RACE: (_pl.Categorical, "race"),
    _Char.SKIN_COLOR: (_pl.Categorical, "skin_color"),
    _Char.EYE_COLOR: (_pl.Categorical, "eye_color"),
    _Char.FACIAL_HAIR: (_pl.Categorical, "facial_hair"),
    _Char.VISION: (_pl.Categorical, "vision"),
    _Char.TOUCH_TYPER: (_pl.Categorical, "touch_typer"),
    _Char.HANDEDNESS: (_pl.Categorical, "handedness"),
    _Char.WEATHER: (_pl.Categorical, "weather"),
    _Char.POINTING_DEVICE: (_pl.Categorical, "pointing_device"),
    _Char.NOTES: (_pl.String, "notes"),
    _Char.TIME_OF_DAY: (_pl.String, "time_of_day"),
    _Char.DURATION: (_pl.String, "duration"),
}

LOG_SCHEMA: dict[_Log, type[_pl.PolarsDataType]] = {
    _Log.IS_TRUSTED: _pl.Boolean,
    _Log.SESSION_ID: _pl.String,
    _Log.WEBPAGE: _pl.String,
    _Log.SESSION_STRING: _pl.String,
    _Log.EPOCH: _pl.UInt64,
    _Log.TIME: _pl.Float64,
    _Log.TYPE: _pl.String,
    _Log.EVENT: _pl.String,
    _Log.TEXT: _pl.String,
    _Log.POS: _pl.Struct({"top": _pl.UInt32, "left": _pl.UInt32}),
    _Log.SCREEN_X: _pl.Int32,
    _Log.SCREEN_Y: _pl.Int32,
    _Log.CLIENT_X: _pl.UInt16,
    _Log.CLIENT_Y: _pl.UInt16,
    _Log.PAGE_X: _pl.UInt32,
    _Log.PAGE_Y: _pl.UInt32,
    _Log.SCROLL_X: _pl.UInt32,
    _Log.SCROLL_Y: _pl.UInt32,
    _Log.WINDOW_X: _pl.Int32,
    _Log.WINDOW_Y: _pl.Int32,
    _Log.WINDOW_INNER_WIDTH: _pl.UInt16,
    _Log.WINDOW_INNER_HEIGHT: _pl.UInt16,
    _Log.WINDOW_OUTER_WIDTH: _pl.UInt16,
    _Log.WINDOW_OUTER_HEIGHT: _pl.UInt16,
}

TOBII_SCHEMA: dict[_Tobii, type[_pl.PolarsDataType]] = {
    _Tobii.DEVICE_TIME_STAMP: _pl.UInt64,
    _Tobii.SYSTEM_TIME_STAMP: _pl.UInt64,
    _Tobii.TRUE_TIME: _pl.Float64,
    _Tobii.LEFT_PUPIL_VALIDITY: _pl.Int32,
    _Tobii.RIGHT_PUPIL_VALIDITY: _pl.Int32,
    _Tobii.LEFT_GAZE_ORIGIN_VALIDITY: _pl.Int32,
    _Tobii.RIGHT_GAZE_ORIGIN_VALIDITY: _pl.Int32,
    _Tobii.LEFT_GAZE_POINT_VALIDITY: _pl.Int32,
    _Tobii.RIGHT_GAZE_POINT_VALIDITY: _pl.Int32,
    _Tobii.LEFT_PUPIL_DIAMETER: _pl.Float64,
    _Tobii.RIGHT_PUPIL_DIAMETER: _pl.Float64,
    _Tobii.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM: _pl.Array(_pl.Float64, 3),
    _Tobii.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM: _pl.Array(_pl.Float64, 3),
    _Tobii.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM: _pl.Array(_pl.Float64, 3),
    _Tobii.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM: _pl.Array(_pl.Float64, 3),
    _Tobii.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM: _pl.Array(_pl.Float64, 3),
    _Tobii.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM: _pl.Array(_pl.Float64, 3),
    _Tobii.LEFT_GAZE_POINT_ON_DISPLAY_AREA: _pl.Array(_pl.Float64, 2),
    _Tobii.RIGHT_GAZE_POINT_ON_DISPLAY_AREA: _pl.Array(_pl.Float64, 2),
}

SPEC_SCHEMA: dict[_Spec, type] = {
    _Spec.POINT_X: _pl.Float64,
    _Spec.POINT_Y: _pl.Float64,
    _Spec.PREDICTION_X_LEFT: _pl.Float64,
    _Spec.PREDICTION_Y_LEFT: _pl.Float64,
    _Spec.VALIDITY_LEFT: _pl.Int8,
    _Spec.PREDICTION_X_RIGHT: _pl.Float64,
    _Spec.PREDICTION_Y_RIGHT: _pl.Float64,
    _Spec.VALIDITY_RIGHT: _pl.Int8,
}

DOTTEST_SCHEMA: dict[_Dot, type] = {
    _Dot.DOT_X: _pl.UInt16,
    _Dot.DOT_Y: _pl.UInt16,
    _Dot.EPOCH: _pl.Float64,
}
