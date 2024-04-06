from typing import TypedDict as _TypedDict
from datetime import datetime as _datetime, timedelta as _timedelta


class TobiiEntry(_TypedDict):
    right_pupil_validity: bool
    right_gaze_point_on_display_area: list[float]
    left_gaze_origin_validity: bool
    system_time_stamp: _datetime
    right_gaze_origin_in_user_coordinate_system: list[float]
    left_gaze_point_in_user_coordinate_system: list[float]
    left_gaze_origin_in_user_coordinate_system: list[float]
    left_pupil_validity: bool
    right_pupil_diameter: float
    true_time: _datetime
    left_gaze_origin_in_trackbox_coordinate_system: list[float]
    right_gaze_point_in_user_coordinate_system: list[float]
    left_pupil_diameter: float
    right_gaze_origin_validity: bool
    left_gaze_point_validity: bool
    right_gaze_point_validity: bool
    left_gaze_point_on_display_area: list[float]
    right_gaze_origin_in_trackbox_coordinate_system: list[float]
    device_time_stamp: _timedelta


class LogEntry(_TypedDict):
    client_x: int | None
    client_y: int | None
    window_y: int | None
    window_x: int | None
    inner_width: int | None
    time: _timedelta
    epoch: _datetime
    outer_width: int | None
    inner_height: int | None
    page_x: int | None
    page_y: int | None
    outer_height: int | None
    screen_y: int | None
    screen_x: int | None
    event: str
    scroll_x: int | None
    scroll_y: int | None
    study: str
    log_id: int
    index: int


class DotEntry(_TypedDict):
    dot_x: int
    dot_y: int
    epoch: _datetime
