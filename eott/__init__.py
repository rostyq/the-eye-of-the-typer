from typing import TypeAlias, Literal, cast, Any
from abc import ABC, abstractmethod
from enum import StrEnum, auto
from os import environ, PathLike
from pathlib import Path
from polars import read_parquet, scan_parquet
from dataclasses import dataclass, asdict
from datetime import time, datetime
from functools import singledispatch, cached_property
from json import dumps as json_dumps

from .raw import (
    Setting,
    Gender,
    Race,
    SkinColor,
    EyeColor,
    FacialHair,
    Vision,
    Handedness,
    Weather,
    PointingDevice,
    Study,
    StudyName,
    print_schema,
)

__all__ = [
    "Setting",
    "Gender",
    "Race",
    "SkinColor",
    "EyeColor",
    "FacialHair",
    "Vision",
    "Handedness",
    "Weather",
    "PointingDevice",
    "Study",
    "DirDataset",
    "print_schema",
    "DataType",
    "DataName",
    "Size",
    "Offset",
    "Point3",
    "Point2",
    "Pair",
    "TrackingBox",
    "FormEntry",
]

VIDEO_SUFFIX = ".mp4"
DATA_SUFFIX = ".parquet"


DataName: TypeAlias = Literal[
    "form", "screen", "webcam", "log", "tobii", "dot", "calib"
]


class DataType(StrEnum):
    FORM = auto()
    SCREEN = auto()
    WEBCAM = auto()
    LOG = auto()
    TOBII = auto()
    DOT = auto()
    CALIB = auto()


class IlluminatingMode(StrEnum):
    DEFAULT = "Default"


@dataclass(frozen=True, slots=True, kw_only=True)
class Size[T: float]:
    w: T
    h: T


@dataclass(frozen=True, slots=True, kw_only=True)
class Offset:
    top: int
    left: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Point3:
    x: float
    y: float
    z: float

    @classmethod
    def from_list(cls, value: list[float]):
        return cls(x=value[0], y=value[1], z=value[2])


@dataclass(frozen=True, slots=True, kw_only=True)
class Point2[T: float]:
    x: T
    y: T

    @classmethod
    def from_list(cls, value: list[T]):
        return cls(x=value[0], y=value[1])


@dataclass(frozen=True, slots=True, kw_only=True)
class Pair[T]:
    left: T
    right: T


@dataclass(slots=True, kw_only=True)
class TrackingBox:
    b_lo_r: Point3
    b_lo_l: Point3
    b_up_l: Point3
    b_up_r: Point3
    f_lo_l: Point3
    f_lo_r: Point3
    f_up_l: Point3
    f_up_r: Point3

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, list):
                value = Point3.from_list(value)
            elif isinstance(value, tuple):
                x, y, z = value
                value = Point3(x=x, y=y, z=z)
            elif isinstance(value, dict):
                value = Point3(**value)
            elif isinstance(value, Point3):
                continue
            else:
                raise TypeError(f"Invalid type for {key}: {type(value)}")

            setattr(self, key, value)


@dataclass(kw_only=True)
class FormEntry:
    pid: int
    log: int
    setting: Setting
    screen_distance: float
    screen_start: datetime | None = None
    webcam_start: datetime
    wall_clock: datetime | None = None
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
    notes: str = ""
    time_of_day: time
    init_start: datetime
    screen_size: Size[float]
    display_resolution: Size[int]
    tracking_box: TrackingBox
    illumination_mode: IlluminatingMode
    frequency: float

    def __post_init__(self):
        self.setting = Setting(self.setting)
        self.gender = Gender(self.gender)
        self.race = Race(self.race)
        self.skin_color = SkinColor(self.skin_color)
        self.eye_color = EyeColor(self.eye_color)
        self.facial_hair = FacialHair(self.facial_hair)
        self.vision = Vision(self.vision)
        self.handedness = Handedness(self.handedness)
        self.weather = Weather(self.weather)
        self.pointing_device = PointingDevice(self.pointing_device)
        self.illumination_mode = IlluminatingMode(self.illumination_mode)

        if not isinstance(cast(Any, self.screen_size), Size):
            assert isinstance(self.screen_size, dict)
            self.screen_size = Size(**self.screen_size)

        if not isinstance(cast(Any, self.display_resolution), Size):
            assert isinstance(self.display_resolution, dict)
            self.display_resolution = Size(**self.display_resolution)

        if not isinstance(cast(Any, self.tracking_box), TrackingBox):
            assert isinstance(self.tracking_box, dict)
            self.tracking_box = TrackingBox(**self.tracking_box)

    @cached_property
    def screen_delay(self):
        return self.screen_start - self.init_start if self.screen_start else None

    @cached_property
    def webcam_delay(self):
        return self.webcam_start - self.init_start

    @cached_property
    def video_delay(self):
        return self.webcam_start - self.screen_start if self.screen_start else None

    def asdict(self):
        return asdict(self)
    
    def json(self, indent: int | str | None = None):
        return json_dumps(
            self.asdict(),
            indent=indent,
            separators=(",", ": " if indent else ":"),
            default=serialize,
        )

    def txt(self):
        """Generate a human-readable text representation of the form entry."""
        lines = [
            "=" * 60,
            f"PARTICIPANT {self.pid:02d} - LOG {self.log:02d}",
            "=" * 60,
            "",
            "STUDY INFORMATION:",
            f"  Setting:              {self.setting.value}",
            f"  Screen Distance:      {self.screen_distance:.1f} cm",
            f"  Screen Start:         {self.screen_start.strftime('%Y-%m-%d %H:%M:%S') if self.screen_start else "N/A"}",
            f"  Webcam Start:         {self.webcam_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Wall Clock:           {self.wall_clock.strftime('%Y-%m-%d %H:%M:%S') if self.wall_clock else 'N/A'}",
            f"  Init Start:           {self.init_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Time of Day:          {self.time_of_day.strftime('%H:%M:%S')}",
            "",
            "PARTICIPANT DEMOGRAPHICS:",
            f"  Gender:               {self.gender.value}",
            f"  Age:                  {self.age} years",
            f"  Race:                 {self.race.value}",
            f"  Skin Color:           {self.skin_color.value}",
            f"  Eye Color:            {self.eye_color.value}",
            f"  Facial Hair:          {self.facial_hair.value}",
            "",
            "ABILITIES & PREFERENCES:",
            f"  Vision:               {self.vision.value}",
            f"  Touch Typer:          {'Yes' if self.touch_typer else 'No'}",
            f"  Handedness:           {self.handedness.value}",
            f"  Pointing Device:      {self.pointing_device.value}",
            "",
            "ENVIRONMENTAL CONDITIONS:",
            f"  Weather:              {self.weather.value}",
            f"  Illumination Mode:    {self.illumination_mode.value}",
            "",
            "TECHNICAL SETUP:",
            f"  Screen Size:          {self.screen_size.w:.1f} × {self.screen_size.h:.1f} cm",
            f"  Display Resolution:   {self.display_resolution.w} × {self.display_resolution.h} pixels",
            f"  Frequency:            {self.frequency:.1f} Hz",
            "",
            "TRACKING BOX COORDINATES:",
            f"  Back Lower Right:     ({self.tracking_box.b_lo_r.x:.2f}, {self.tracking_box.b_lo_r.y:.2f}, {self.tracking_box.b_lo_r.z:.2f})",
            f"  Back Lower Left:      ({self.tracking_box.b_lo_l.x:.2f}, {self.tracking_box.b_lo_l.y:.2f}, {self.tracking_box.b_lo_l.z:.2f})",
            f"  Back Upper Left:      ({self.tracking_box.b_up_l.x:.2f}, {self.tracking_box.b_up_l.y:.2f}, {self.tracking_box.b_up_l.z:.2f})",
            f"  Back Upper Right:     ({self.tracking_box.b_up_r.x:.2f}, {self.tracking_box.b_up_r.y:.2f}, {self.tracking_box.b_up_r.z:.2f})",
            f"  Front Lower Left:     ({self.tracking_box.f_lo_l.x:.2f}, {self.tracking_box.f_lo_l.y:.2f}, {self.tracking_box.f_lo_l.z:.2f})",
            f"  Front Lower Right:    ({self.tracking_box.f_lo_r.x:.2f}, {self.tracking_box.f_lo_r.y:.2f}, {self.tracking_box.f_lo_r.z:.2f})",
            f"  Front Upper Left:     ({self.tracking_box.f_up_l.x:.2f}, {self.tracking_box.f_up_l.y:.2f}, {self.tracking_box.f_up_l.z:.2f})",
            f"  Front Upper Right:    ({self.tracking_box.f_up_r.x:.2f}, {self.tracking_box.f_up_r.y:.2f}, {self.tracking_box.f_up_r.z:.2f})",
        ]

        if self.notes.strip():
            lines.extend(
                [
                    "",
                    "NOTES:",
                    f"  {self.notes}",
                ]
            )

        lines.extend(["", "=" * 60, ""])

        return "\n".join(lines)


@singledispatch
def serialize(o: Any) -> Any:
    raise TypeError(f"Unsupported type for JSON serialization: {type(o)}")


@serialize.register
def _(o: datetime):
    return o.isoformat()


@serialize.register
def _(o: time):
    return o.isoformat()


@serialize.register
def _(o: Point2):
    return [o.x, o.y]


@serialize.register
def _(o: Point3):
    return [o.x, o.y, o.z]


@serialize.register
def _(o: Size):
    return {"width": o.w, "height": o.h}


@serialize.register
def _(o: Offset):
    return {"top": o.top, "left": o.left}


@serialize.register
def _(o: Pair):
    return {"left": o.left, "right": o.right}


@serialize.register
def _(o: TrackingBox):
    return {
        "b_lo_r": serialize(o.b_lo_r),
        "b_lo_l": serialize(o.b_lo_l),
        "b_up_l": serialize(o.b_up_l),
        "b_up_r": serialize(o.b_up_r),
        "f_lo_l": serialize(o.f_lo_l),
        "f_lo_r": serialize(o.f_lo_r),
        "f_up_l": serialize(o.f_up_l),
        "f_up_r": serialize(o.f_up_r),
    }


class DatasetPaths(ABC):
    @property
    @abstractmethod
    def root(self) -> Path: ...

    @property
    def screen_dir(self):
        return self.root / "screen"

    @property
    def webcam_dir(self):
        return self.root / "webcam"

    def data_path(self, name: DataType | DataName, suffix=DATA_SUFFIX):
        return (self.root / DataType(name)).with_suffix(suffix)

    def screen_path(self, pid: int, suffix=VIDEO_SUFFIX):
        return (self.screen_dir / f"P_{pid:02}").with_suffix(suffix)

    def webcam_path(self, pid: int, suffix=VIDEO_SUFFIX):
        return (self.webcam_dir / f"P_{pid:02}").with_suffix(suffix)

    def webcam_parts_dir(self, pid: int):
        return self.root.joinpath("webcam", f"P_{pid:02}")

    def webcam_part_path(
        self, pid: int, record: int, study: Study | StudyName, suffix=VIDEO_SUFFIX
    ):
        return (self.webcam_parts_dir(pid) / f"{record:02}-{Study(study)}").with_suffix(
            suffix
        )

    def webcam_parts_glob(self, pid: int, suffix=VIDEO_SUFFIX, gaps: bool = False):
        return filter(
            lambda p: True if gaps else not p.stem.endswith("-gap"),
            self.webcam_parts_dir(pid).glob(f"*{suffix}"),
        )

    def webcam_parts_paths(self, pid: int, suffix=VIDEO_SUFFIX, gaps: bool = False):
        return sorted(
            self.webcam_parts_glob(pid, suffix, gaps=gaps), key=lambda p: p.stem
        )

    def webcam_parsed_paths(self, pid: int, suffix=VIDEO_SUFFIX):
        def parse_path(p: Path):
            record, study = p.stem.split("-")
            return int(record), Study(study), p

        return sorted(
            map(parse_path, self.webcam_parts_glob(pid, suffix, False)),
            key=lambda i: i[0],
        )


class DirDataset(DatasetPaths):
    """
    A class to represent a dataset stored in a directory.
    This is the transformed dataset ready for analysis.
    """

    def __init__(self, path: PathLike[str] | str = "", /):
        if not path:
            path = Path(environ["EOTT_DATASET_PATH"]).expanduser().resolve()
        if not (path := Path(path)).is_dir():
            raise NotADirectoryError(
                f'Dataset directory "{path}" does not exist or is not a directory.'
            )
        self._path = path

    def __repr__(self):
        return f"DirDataset(Path('{self._path}'))"

    @property
    def root(self):
        return self._path

    def schema(self, name: DataType | DataName, /):
        return self.lazyframe(name).collect_schema()

    def dataframe(self, name: DataType | DataName, /, **kwargs):
        return read_parquet(self.data_path(name), **kwargs)

    def lazyframe(self, name: DataType | DataName, /, **kwargs):
        return scan_parquet(self.data_path(name), **kwargs)
