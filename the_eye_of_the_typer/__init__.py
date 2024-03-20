from typing import Literal
from dataclasses import dataclass, asdict
from enum import StrEnum, IntEnum, auto
from os import PathLike, environ
from pathlib import Path
from re import match, search
from functools import cached_property
from datetime import datetime, date, time

from polars import (
    col,
    DataFrame,
    read_csv,
    read_json,
    read_ndjson,
    String,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    Struct,
    UInt64,
    Float64,
    Categorical,
    Boolean,
    Datetime,
    Array,
    Duration,
)


__all__ = [
    "PARTICIPANT_CHARACTERISTICS_FILENAME",
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
    "Characteristic",
    "Tobii",
    "Participant",
    "read_participant_characteristics",
    "get_dataset_root",
    "read_user_interaction_logs",
    "read_tobii_gaze_predictions",
    "read_tobii_calibration_points",
    "read_tobii_specs",
    "lookup_webcam_video_paths",
]


class Setting(StrEnum):
    LAPTOP = "Laptop"
    PC = "PC"


class Gender(StrEnum):
    MALE = "Male"
    FEMALE = "Female"


class Race(StrEnum):
    WHITE = "White"
    BLACK = "Black"
    ASIAN = "Asian"
    OTHER = "Other"


class SkinColor(IntEnum):
    C1 = 1
    C2 = 2
    C3 = 3
    C4 = 4
    C5 = 5


class EyeColor(StrEnum):
    DARK_BROWN_TO_BROWN = "Dark Brown to Brown"
    GRAY_TO_BLUE_OR_PINK = "Gray to Blue or Pink"
    GREEN_HAZEL_TO_BLUE_HAZEL = "Green-Hazel to Blue-Hazel"
    GREEN_HAZEL_TO_GREEN = "Green-Hazel to Green"
    AMBER = "Amber"


class FacialHair(StrEnum):
    BEARD = "Beard"
    LITTLE = "Little"
    NONE = "None"


class Vision(StrEnum):
    NORMAL = "Normal"
    GLASSES = "Glasses"
    CONTACTS = "Contacts"


class Handedness(StrEnum):
    LEFT = "Left"
    RIGHT = "Right"


class Weather(StrEnum):
    CLOUDY = "Cloudy"
    INDOORS = "Indoors"
    SUNNY = "Sunny"


class PointingDevice(StrEnum):
    TRACKPAD = "Trackpad"
    MOUSE = "Mouse"


class Study(StrEnum):
    DOT_TEST_INSTRUCTIONS = auto()
    DOT_TEST = auto()
    FITTS_LAW_INSTRUCTIONS = auto()
    FITTS_LAW = auto()
    SERP_INSTRUCTIONS = auto()
    BENEFITS_OF_RUNNING_INSTRUCTIONS = auto()
    BENEFITS_OF_RUNNING = auto()
    BENEFITS_OF_RUNNING_WRITING = auto()
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES_INSTRUCTIONS = auto()
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES = auto()
    BEDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES_WRITING = auto()
    WHERE_TO_FIND_MOREL_MUSHROOMS_INSTRUCTIONS = auto()
    WHERE_TO_FIND_MOREL_MUSHROOMS = auto()
    WHERE_TO_FIND_MOREL_MUSHROOMS_WRITING = auto()
    TOOTH_ABSCESS_INSTRUCTIONS = auto()
    TOOTH_ABSCESS = auto()
    TOOTH_ABSCESS_WRITING = auto()
    DOT_TEST_FINAL_INSTRUCTIONS = auto()
    DOT_TEST_FINAL = auto()
    THANK_YOU = auto()


class Characteristic(StrEnum):
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


class Log(StrEnum):
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


class Tobii(StrEnum):
    RIGHT_PUPIL_VALIDITY = auto()
    RIGHT_GAZE_POINT_ON_DISPLAY_AREA = auto()
    LEFT_GAZE_ORIGIN_VALIDITY = auto()
    SYSTEM_TIME_STAMP = auto()
    RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM = auto()
    LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM = auto()
    LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM = auto()
    LEFT_PUPIL_VALIDITY = auto()
    RIGHT_PUPIL_DIAMETER = auto()
    TRUE_TIME = auto()
    LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM = auto()
    RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM = auto()
    LEFT_PUPIL_DIAMETER = auto()
    RIGHT_GAZE_ORIGIN_VALIDITY = auto()
    LEFT_GAZE_POINT_VALIDITY = auto()
    RIGHT_GAZE_POINT_VALIDITY = auto()
    LEFT_GAZE_POINT_ON_DISPLAY_AREA = auto()
    RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM = auto()
    DEVICE_TIME_STAMP = auto()


class Spec(StrEnum):
    VALIDITY_LEFT = auto()
    VALIDITY_RIGHT = auto()
    POINT_X = auto()
    POINT_Y = auto()
    PREDICTION_X_LEFT = auto()
    PREDICTION_Y_LEFT = auto()
    PREDICTION_X_RIGHT = auto()
    PREDICTION_Y_RIGHT = auto()


MAP_ENUMS = {
    "setting": Setting,
    "gender": Gender,
    "race": Race,
    "skin_color": lambda x: SkinColor(int(x)),
    "eye_color": EyeColor,
    "facial_hair": FacialHair,
    "vision": Vision,
    "handedness": Handedness,
    "weather": Weather,
    "pointing_device": PointingDevice,
}

PARTICIPANT_CHARACTERISTICS_FILENAME = "participant_characteristics.csv"
PARTICIPANT_CHARACTERISTICS_SCHEMA = {
    Characteristic.PID: (String, "pid"),
    Characteristic.LOG_ID: (UInt64, "log_id"),
    Characteristic.DATE: (String, "date"),
    Characteristic.SETTING: (Categorical, "setting"),
    Characteristic.DISPLAY_WIDTH: (UInt16, "display_width"),
    Characteristic.DISPLAY_HEIGHT: (UInt16, "display_height"),
    Characteristic.SCREEN_WIDTH: (Float64, "screen_width"),
    Characteristic.SCREEN_HEIGHT: (Float64, "screen_height"),
    Characteristic.DISTANCE_FROM_SCREEN: (Float64, "distance_from_screen"),
    Characteristic.SCREEN_RECORDING_START_TIME_UNIX: (UInt64, "screen_recording"),
    Characteristic.SCREEN_RECORDING_START_TIME_UTC: (String, "wall_clock"),
    Characteristic.GENDER: (Categorical, "gender"),
    Characteristic.AGE: (UInt8, "age"),
    Characteristic.RACE: (Categorical, "race"),
    Characteristic.SKIN_COLOR: (Categorical, "skin_color"),
    Characteristic.EYE_COLOR: (Categorical, "eye_color"),
    Characteristic.FACIAL_HAIR: (Categorical, "facial_hair"),
    Characteristic.VISION: (Categorical, "vision"),
    Characteristic.TOUCH_TYPER: (Categorical, "touch_typer"),
    Characteristic.HANDEDNESS: (Categorical, "handedness"),
    Characteristic.WEATHER: (Categorical, "weather"),
    Characteristic.POINTING_DEVICE: (Categorical, "pointing_device"),
    Characteristic.NOTES: (String, "notes"),
    Characteristic.TIME_OF_DAY: (String, "time_of_day"),
    Characteristic.DURATION: (String, "duration"),
}

LOG_SCHEMA = {
    Log.IS_TRUSTED: (Boolean, None),
    Log.SESSION_ID: (String, None),
    Log.WEBPAGE: (Categorical, None),
    Log.SESSION_STRING: (String, None),
    Log.EPOCH: (UInt64, "timestamp"),
    Log.TIME: (Float64, "duration"),
    Log.TYPE: (Categorical, "event"),
    Log.EVENT: (Categorical, None),
    # Log.TEXT: Categorical,
    # Log.POS: Struct({"top": UInt32, "left": UInt32}),
    Log.SCREEN_X: (Int32, "screen_x"),
    Log.SCREEN_Y: (Int32, "screen_y"),
    Log.CLIENT_X: (UInt16, "client_x"),
    Log.CLIENT_Y: (UInt16, "client_y"),
    Log.PAGE_X: (UInt32, "page_x"),
    Log.PAGE_Y: (UInt32, "page_y"),
    Log.SCROLL_X: (UInt32, "scroll_x"),
    Log.SCROLL_Y: (UInt32, "scroll_y"),
    Log.WINDOW_X: (Int32, "window_x"),
    Log.WINDOW_Y: (Int32, "window_y"),
    Log.WINDOW_INNER_WIDTH: (UInt16, "inner_width"),
    Log.WINDOW_INNER_HEIGHT: (UInt16, "inner_height"),
    Log.WINDOW_OUTER_WIDTH: (UInt16, "outer_width"),
    Log.WINDOW_OUTER_HEIGHT: (UInt16, "outer_height"),
}

TOBII_SCHEMA = {
    Tobii.DEVICE_TIME_STAMP: Duration("us"),
    Tobii.SYSTEM_TIME_STAMP: Datetime("us"),
    Tobii.TRUE_TIME: Float64,
    Tobii.LEFT_PUPIL_VALIDITY: Boolean,
    Tobii.RIGHT_PUPIL_VALIDITY: Boolean,
    Tobii.LEFT_GAZE_ORIGIN_VALIDITY: Boolean,
    Tobii.RIGHT_GAZE_ORIGIN_VALIDITY: Boolean,
    Tobii.LEFT_GAZE_POINT_VALIDITY: Boolean,
    Tobii.RIGHT_GAZE_POINT_VALIDITY: Boolean,
    Tobii.LEFT_PUPIL_DIAMETER: Float64,
    Tobii.RIGHT_PUPIL_DIAMETER: Float64,
    Tobii.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
    Tobii.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
    Tobii.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM: Array(Float64, 3),
    Tobii.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM: Array(Float64, 3),
    Tobii.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
    Tobii.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
    Tobii.LEFT_GAZE_POINT_ON_DISPLAY_AREA: Array(Float64, 2),
    Tobii.RIGHT_GAZE_POINT_ON_DISPLAY_AREA: Array(Float64, 2),
}

SPEC_SCHEMA = {
    Spec.POINT_X: Float64,
    Spec.POINT_Y: Float64,
    Spec.PREDICTION_X_LEFT: Float64,
    Spec.PREDICTION_Y_LEFT: Float64,
    Spec.VALIDITY_LEFT: Int8,
    Spec.PREDICTION_X_RIGHT: Float64,
    Spec.PREDICTION_Y_RIGHT: Float64,
    Spec.VALIDITY_RIGHT: Int8,
}


def get_dataset_root() -> Path:
    return (
        Path(environ.get("THE_EYE_OF_THE_TYPER_DATASET_PATH", Path.cwd()))
        .expanduser()
        .resolve()
    )


def read_participant_characteristics(root: PathLike | None = None) -> DataFrame:
    source = Path(root or get_dataset_root()) / PARTICIPANT_CHARACTERISTICS_FILENAME

    return (
        read_csv(
            source,
            has_header=True,
            columns=[name.value for name in Characteristic.__members__.values()],
            separator=",",
            quote_char='"',
            null_values=["-", ""],
            n_threads=1,
            schema={
                key.value: value[0]
                for key, value in PARTICIPANT_CHARACTERISTICS_SCHEMA.items()
            },
        )
        .with_columns(
            col(Characteristic.PID.value)
            .str.split("_")
            .list.get(1)
            .str.to_integer()
            .cast(UInt8),
            col(Characteristic.LOG_ID).cast(Datetime("ms")).alias("start_time"),
            col(Characteristic.DATE.value).str.to_date(r"%m/%d/%Y"),
            col(Characteristic.SCREEN_RECORDING_START_TIME_UNIX.value).cast(
                Datetime("ms")
            ),
            col(Characteristic.SCREEN_RECORDING_START_TIME_UTC.value).str.to_datetime(
                r"%m/%d/%Y %H:%M",
                # time_zone="UTC",
            ),
            col(Characteristic.TOUCH_TYPER.value)
            .cast(String)
            .map_dict({"Yes": True, "No": False}, return_dtype=Boolean),
            col(Characteristic.TIME_OF_DAY.value).str.to_time(r"%H:%M"),
            col(Characteristic.DURATION.value)
            .str.split(":")
            .cast(Array(UInt16, 3))
            .apply(
                lambda arr: (arr[0] * 3600 + arr[1] * 60 + arr[2]) * 1000,
                return_dtype=Int64,
            )
            .cast(Duration("ms")),
        )
        .rename(
            {
                key.value: value[1]
                for key, value in PARTICIPANT_CHARACTERISTICS_SCHEMA.items()
            }
        )
        .sort(by="pid")
    )


def lookup_webcam_video_paths(root: PathLike | None = None):
    paths = [
        path
        for path in Path(root or get_dataset_root()).glob("**/*.webm")
        if path.is_file()
    ]

    def parse_indices(p: Path):
        pid = p.parent.name.split("_")[1]
        name = p.name
        log_id = int(match(r"(\d+)_", name).group(1))
        index = int(search(r"_(\d+)_", name).group(1))

        aux = search(r"\((\d+)\)", name)
        aux = int(aux.group(1)) if aux is not None else 0
        return pid, log_id, index, aux

    paths.sort(key=parse_indices)

    return paths


def read_user_interaction_logs(p: PathLike):
    return (
        read_json(p)
        .with_columns(col(key).cast(value[0]) for key, value in LOG_SCHEMA.items())
        .with_columns(
            col(Log.SESSION_ID)
            .str.split("/")
            .list.get(-1)
            .cast(Categorical)
            .alias("study"),
            col(Log.SESSION_ID).str.split("_").list.get(0).cast(UInt64).alias("log_id"),
            col(Log.SESSION_ID).str.split("_").list.get(1).cast(UInt8).alias("index"),
            col(Log.EPOCH).cast(Datetime("ms")),
            col(Log.TIME).cast(Duration("ms")),
        )
        .drop(key.value for key, value in LOG_SCHEMA.items() if value[1] is None)
        .rename(
            {
                key.value: value[1]
                for key, value in LOG_SCHEMA.items()
                if value[1] is not None
            }
        )
        .sort(by="timestamp")
    )


def read_tobii_gaze_predictions(p: PathLike):
    return (
        read_ndjson(p)
        .with_columns(col(key).cast(value) for key, value in TOBII_SCHEMA.items())
        .with_columns(
            (col(Tobii.TRUE_TIME) * 1e9).cast(Datetime("ns")),
        )
    )


def read_tobii_calibration_points(p: PathLike):
    return (
        read_csv(
            p,
            has_header=True,
            columns=[name.value for name in Spec.__members__.values()],
            separator="\t",
            n_threads=1,
            schema={key.value: value for key, value in SPEC_SCHEMA.items()},
            ignore_errors=True,
        )
        .drop_nulls()
        .with_columns(col(Spec.VALIDITY_LEFT) > 1, col(Spec.VALIDITY_RIGHT) > 1)
    )


def read_tobii_specs(p: PathLike):
    tracking_box: dict[
        tuple[
            Literal["back", "front"],
            Literal["lower", "upper"],
            Literal["left", "right"],
        ],
        tuple[float, float, float],
    ] = {}
    ilumination_mode: str | None = None
    frequency: float | None = None

    with open(p, "r") as file:
        for line in file:
            m = match(r"(Back|Front) (Lower|Upper) (Left|Right):", line)
            if m is not None:
                index = (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())

                values = line[m.end() :].strip(" ()\n\t").split(", ")
                x, y, z = [float(v) for v in values]

                tracking_box[index] = (x, y, z)

            m = match(r"Illumination mode:", line)
            if m is not None:
                ilumination_mode = line[m.end() :].strip()

            m = match(r"Initial gaze output frequency: (\d+\.?\d+)", line)
            if m is not None:
                frequency = float(m.group(1))

    return tracking_box, ilumination_mode, frequency


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
    def get_root(dataset: PathLike, pid: int):
        return Path(dataset).expanduser().resolve() / f"P_{pid:02}"

    @classmethod
    def create(cls, dataset: PathLike | None = None, **kwargs):
        return cls(
            root=cls.get_root(dataset or get_dataset_root(), kwargs["pid"]),
            **{key: value(kwargs.pop(key)) for key, value in MAP_ENUMS.items()},
            **{key: value for key, value in kwargs.items()},
        )

    @cached_property
    def participant_id(self):
        return self.root.name

    @cached_property
    def tobii_specs_path(self):
        return self.root / "specs.txt"

    @cached_property
    def screen_recording_path(self):
        return self.root / f"{self.participant_id}.mov"

    @cached_property
    def user_interaction_logs_path(self):
        return self.root / f"{self.log_id}.json"

    @cached_property
    def tobii_gaze_predictions_path(self):
        return self.root / f"{self.participant_id}.txt"

    @cached_property
    def webcam_video_paths(self):
        return lookup_webcam_video_paths(self.root)

    @cached_property
    def user_interaction_logs(self):
        return read_user_interaction_logs(self.user_interaction_logs_path)

    @cached_property
    def tobii_gaze_predictions(self):
        return read_tobii_gaze_predictions(self.tobii_gaze_predictions_path)

    @cached_property
    def tobii_calibration_points(self):
        return read_tobii_calibration_points(self.tobii_gaze_predictions_path)

    @cached_property
    def tobii_specs(self):
        return read_tobii_specs(self.tobii_specs_path)

    @cached_property
    def tobii_ilumination_mode(self):
        return self.tobii_specs[1]

    @cached_property
    def tobii_frequency(self):
        return self.tobii_specs[2]

    def _rerun_set_time(self, value: float):
        from rerun import set_time_seconds

        set_time_seconds("global_time", value)
        set_time_seconds("test_time", value - self.start_time.timestamp())

        if self.screen_recording is not None:
            set_time_seconds("screen_time", value - self.screen_recording.timestamp())

    def _rerun_log_screen_recording(self):
        from cv2 import (
            VideoCapture,
            CAP_PROP_POS_MSEC,
            resize,
            cvtColor,
            COLOR_BGR2GRAY,
        )
        from rerun import log, Image, set_time_sequence

        scale_factor = 2
        cap = VideoCapture(str(self.screen_recording_path))
        size = (self.display_width // scale_factor, self.display_height // scale_factor)
        timestamp_offset = self.screen_recording.timestamp()

        i = 0
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            if i % 2 != 0:
                continue

            timestamp = timestamp_offset + cap.get(CAP_PROP_POS_MSEC) * 1e-3
            set_time_sequence("screen_frame", i)
            self._rerun_set_time(timestamp)

            log("screen", Image(cvtColor(resize(frame, size), COLOR_BGR2GRAY)))

            i += 1

    def _rerun_log_user_interactions(self):
        from rerun import log, TextLog, TextLogLevel, set_time_sequence

        levels = {
            "mousemove": TextLogLevel.TRACE,
            "scrollEvent": TextLogLevel.TRACE,
            "mouseclick": TextLogLevel.DEBUG,
            "text_input": TextLogLevel.DEBUG,
            "textInput": TextLogLevel.DEBUG,
            "recording start": TextLogLevel.WARN,
            "recording stop": TextLogLevel.ERROR,
        }

        dt: datetime
        event_type: str | None
        for i, (dt, event_type) in enumerate(
            self.user_interaction_logs.select("timestamp", "event").iter_rows()
        ):
            set_time_sequence("event_index", i)

            self._rerun_set_time(dt.timestamp())
            level = levels.get(event_type)

            if level is not None:
                log("event", TextLog(event_type, level=level))

    def rerun(self):
        from rerun import init, log, TextDocument

        init("EOTT", recording_id=self.participant_id, spawn=True)
        log(
            "participant",
            TextDocument(
                "\n".join(f"{key}: {value}" for key, value in asdict(self).items())
            ),
            timeless=True,
        )

        self._rerun_log_user_interactions()

        if self.screen_recording is not None:
            self._rerun_log_screen_recording()
