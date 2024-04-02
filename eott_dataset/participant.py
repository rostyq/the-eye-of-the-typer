from typing import TYPE_CHECKING, Optional
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass, asdict
from datetime import timedelta, date, datetime, time
from re import match, search

from . import utils as util, data as ds
from .study import Study
from .characteristics import *
from .adjustments import get_record_offset

if TYPE_CHECKING:
    from os import PathLike


__all__ = ["Participant"]


@dataclass(frozen=True, kw_only=True)
class Participant:
    root: Path
    pid: int
    log_id: int
    date: date
    setting: Setting
    display_width: int
    display_height: int
    screen_width: float
    screen_height: float
    gender: Gender
    age: int
    race: Race
    skin_color: SkinColor
    eye_color: EyeColor
    facial_hair: FacialHair
    vision: Vision
    touch_typer: bool
    handedness: Handedness
    weather: Weather
    pointing_device: PointingDevice
    notes: str
    time_of_day: time
    duration: int
    start_time: datetime

    distance_from_screen: float | None = None
    screen_recording: datetime | None = None
    wall_clock: datetime | None = None

    def __post_init__(self):
        assert self.root, f"{self.root} is not a directory"
        assert isinstance(self.pid, int), f"{self.pid} is not an integer"
        assert match(
            r"P_\d{2}", self.root.name
        ), f"{self.root.name} is not a valid participant directory"
        assert isinstance(self.log_id, int), f"{self.log_id} is not an integer"
        assert isinstance(self.date, date), f"{self.date} is not a date"
        assert isinstance(
            self.setting, Setting
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
                self.screen_recording, datetime
            ), f"{self.screen_recording} is not a datetime"
        if self.wall_clock is not None:
            assert isinstance(
                self.wall_clock, datetime
            ), f"{self.wall_clock} is not a datetime"

    @staticmethod
    def get_root(dataset: "PathLike", pid: int):
        return Path(dataset).expanduser().resolve() / f"P_{pid:02}"

    @classmethod
    def from_dict(cls, dataset: Optional["PathLike"] = None, **kwargs):
        return cls(
            root=cls.get_root(dataset or util.get_dataset_root(), kwargs["pid"]),
            **{
                key: fn(kwargs.pop(key)) for key, fn in CHARACTERISTICS.items()
            },
            **{key: value for key, value in kwargs.items()},
        )

    @cached_property
    def participant_id(self):
        return self.root.name

    @cached_property
    def screen_offset(self):
        # screen recording value in participant.csv just makes no sense
        return get_record_offset(self.pid)

    @cached_property
    def tobii_offset(self) -> timedelta | None:
        return self.tobii_gaze_predictions["true_time"][0] - self.start_time

    @cached_property
    def log_offset(self) -> timedelta | None:
        return self.user_interaction_logs["epoch"][0] - self.start_time

    @cached_property
    def tobii_specs_path(self):
        return self.root / "specs.txt"

    @cached_property
    def dottest_locations_path(self):
        return self.root / "final_dot_test_locations.tsv"

    @cached_property
    def screen_recording_extension(self):
        match self.setting:
            case Setting.LAPTOP:
                return "mov"
            case Setting.PC:
                return "flv"
            case _:
                return None

    @cached_property
    def screen_recording_path(self):
        ext = self.screen_recording_extension
        assert ext is not None
        return self.root / f"{self.participant_id}.{ext}"

    @cached_property
    def user_interaction_logs_path(self):
        return self.root / f"{self.log_id}.json"

    @cached_property
    def tobii_gaze_predictions_path(self):
        return self.root / f"{self.participant_id}.txt"

    @cached_property
    def webcam_video_paths(self):
        return util.lookup_webcam_video_paths(self.root)

    @cached_property
    def user_interaction_logs(self):
        return ds.read_user_interaction_logs(self.user_interaction_logs_path)

    @cached_property
    def tobii_gaze_predictions(self):
        return ds.read_tobii_gaze_predictions(self.tobii_gaze_predictions_path)

    @cached_property
    def tobii_calibration_points(self):
        return ds.read_tobii_calibration_points(self.tobii_gaze_predictions_path)

    @cached_property
    def dottest_locations(self):
        return ds.read_dottest_locations(self.dottest_locations_path)

    @cached_property
    def tobii_specs(self):
        return ds.read_tobii_specs(self.tobii_specs_path)

    @cached_property
    def tobii_ilumination_mode(self):
        return self.tobii_specs[1]

    @cached_property
    def tobii_frequency(self):
        return self.tobii_specs[2]

    def get_webcam_video_paths(
        self, *, study: Optional[Study] = None, index: int | None = None
    ) -> list[Path]:
        pattern: str | None = None
        paths = self.webcam_video_paths

        match (study, index):
            case (s, i) if i is not None and s is not None:
                pattern = r"_%s_-study-%s[\. ]" % (i, s)
            case (None, i) if i is not None:
                pattern = r"_%s_-study-" % i
            case (s, None) if s is not None:
                pattern = r"-study-%s[\. ]" % s
            case (None, None):
                return paths

        return [p for p in paths if search(pattern, p.name)]

    def to_dict(self):
        return asdict(self)
