from typing import TYPE_CHECKING, Optional
from pathlib import Path as _Path
from dataclasses import dataclass as _dataclass, asdict as _asdict
from re import match as _match
from functools import cached_property as _cached_property

import datetime as _datetime

from . import characteristics as _c, utils as _utils

if TYPE_CHECKING:
    from os import PathLike


# The following are the time adjustments for the start of the screen recording
# for each participant in the dataset. It is determined manually by visually
# inspecting the screen recording for notable events such as mouse clicks.
# The adjustments are in seconds.
RECORDING_START_TIME_ADJUSTMENTS: dict[int, float] = {
    1: 5.45 + 0.875
}

@_dataclass(frozen=True, kw_only=True)
class Participant:
    root: _Path
    pid: int
    log_id: int
    date: _datetime.date
    setting: _c.Setting
    display_width: int
    display_height: int
    screen_width: float
    screen_height: float
    gender: _c.Gender
    age: int
    race: _c.Race
    skin_color: _c.SkinColor
    eye_color: _c.EyeColor
    facial_hair: _c.FacialHair
    vision: _c.Vision
    touch_typer: bool
    handedness: _c.Handedness
    weather: _c.Weather
    pointing_device: _c.PointingDevice
    notes: str
    time_of_day: _datetime.time
    duration: int
    start_time: _datetime.datetime

    distance_from_screen: float | None = None
    screen_recording: _datetime.datetime | None = None
    wall_clock: _datetime.datetime | None = None

    def __post_init__(self):
        assert self.root, f"{self.root} is not a directory"
        assert isinstance(self.pid, int), f"{self.pid} is not an integer"
        assert _match(
            r"P_\d{2}", self.root.name
        ), f"{self.root.name} is not a valid participant directory"
        assert isinstance(self.log_id, int), f"{self.log_id} is not an integer"
        assert isinstance(self.date, _datetime.date), f"{self.date} is not a date"
        assert isinstance(
            self.setting, _c.Setting
        ), f"{self.setting} is not a valid setting"
        assert isinstance(
            self.display_width, int
        ), f"{self.display_width} is not an integer"
        assert isinstance(
            self.display_height, int
        ), f"{self.display_height} is not an integer"
        assert isinstance(
            self.screen_width, float
        ), f"{self.screen_width} is not a float"
        assert isinstance(
            self.screen_height, float
        ), f"{self.screen_height} is not a float"
        if self.distance_from_screen is not None:
            assert isinstance(
                self.distance_from_screen, float
            ), f"{self.distance_from_screen} is not a float"
        if self.screen_recording is not None:
            assert isinstance(
                self.screen_recording, _datetime.datetime
            ), f"{self.screen_recording} is not a datetime"
        if self.wall_clock is not None:
            assert isinstance(
                self.wall_clock, _datetime.datetime
            ), f"{self.wall_clock} is not a datetime"

    @staticmethod
    def get_root(dataset: "PathLike", pid: int):
        return _Path(dataset).expanduser().resolve() / f"P_{pid:02}"

    @classmethod
    def create(cls, dataset: Optional["PathLike"] = None, **kwargs):
        return cls(
            root=cls.get_root(dataset or _utils.get_dataset_root(), kwargs["pid"]),
            **{
                key: value(kwargs.pop(key)) for key, value in _c.CHARACTERISTICS.items()
            },
            **{key: value for key, value in kwargs.items()},
        )

    @_cached_property
    def participant_id(self):
        return self.root.name

    @_cached_property
    def screen_offset(self):
        if self.screen_recording is not None:
            return (
                self.screen_recording
                - self.start_time
                - _datetime.timedelta(
                    seconds=RECORDING_START_TIME_ADJUSTMENTS.get(self.pid, 0)
                )
            )

    @_cached_property
    def tobii_offset(self) -> _datetime.timedelta | None:
        return self.tobii_gaze_predictions["true_time"][0] - self.start_time

    @_cached_property
    def log_offset(self) -> _datetime.timedelta | None:
        return self.user_interaction_logs["timestamp"][0] - self.start_time

    @_cached_property
    def tobii_specs_path(self):
        return self.root / "specs.txt"

    @_cached_property
    def screen_recording_path(self):
        return self.root / f"{self.participant_id}.mov"

    @_cached_property
    def user_interaction_logs_path(self):
        return self.root / f"{self.log_id}.json"

    @_cached_property
    def tobii_gaze_predictions_path(self):
        return self.root / f"{self.participant_id}.txt"

    @_cached_property
    def webcam_video_paths(self):
        return _utils.lookup_webcam_video_paths(self.root)

    @_cached_property
    def user_interaction_logs(self):
        return _utils.read_user_interaction_logs(self.user_interaction_logs_path)

    @_cached_property
    def tobii_gaze_predictions(self):
        return _utils.read_tobii_gaze_predictions(self.tobii_gaze_predictions_path)

    @_cached_property
    def tobii_calibration_points(self):
        return _utils.read_tobii_calibration_points(self.tobii_gaze_predictions_path)

    @_cached_property
    def tobii_specs(self):
        return _utils.read_tobii_specs(self.tobii_specs_path)

    @_cached_property
    def tobii_ilumination_mode(self):
        return self.tobii_specs[1]

    @_cached_property
    def tobii_frequency(self):
        return self.tobii_specs[2]

    def to_dict(self):
        return _asdict(self)
