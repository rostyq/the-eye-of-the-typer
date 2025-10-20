import operator
from typing import (
    Any,
    Self,
    IO,
    overload,
    Literal,
    Callable,
    cast,
    override,
    TypeAlias,
)
from collections.abc import Iterable
from pathlib import Path
from abc import abstractmethod, ABCMeta
from os import SEEK_END, SEEK_SET
from functools import cached_property, cache
from enum import auto, StrEnum, EnumMeta
from re import compile, match, search
from dataclasses import dataclass
from zipfile import ZipFile, ZipInfo
from datetime import timedelta

from polars import (
    String,
    Categorical,
    UInt64,
    UInt16,
    Float64,
    UInt8,
    Int32,
    Int64,
    Struct,
    Boolean,
    Array,
    UInt32,
    Int8,
    Enum,
    Datetime,
    Duration,
    duration,
    from_epoch,
    read_csv,
    scan_csv,
    scan_ndjson,
    read_json,
    lit,
    when,
    struct,
    col,
    concat,
    coalesce,
    LazyFrame,
)
from polars._typing import PolarsDataType

from .util import parse_timedelta, NameEnum


__all__ = [
    "ZipDataset",
    "DataType",
    "Alignment",
    "Characteristic",
    "Webcam",
    "Log",
    "Tobii",
    "Calib",
    "Dot",
    "Spec",
    "Event",
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
    "StudyName",
    "Source",
    "pid_from_name",
]

InputType = Path | bytes | IO[bytes]


class DataType(StrEnum):
    FORM = auto()
    """Participant characteristics form"""
    SCREEN = auto()
    """Screen recording video"""
    WEBCAM = auto()
    """Webcam recording video"""
    LOG = auto()
    """Web interaction log"""
    TOBII = auto()
    """Tobii eye-tracking data"""
    DOT = auto()
    """Dot test final data"""
    CALIB = auto()
    """Tobii calibration data"""

    def path(self):
        return Path(self if self != "tobii" else f"{self}/*")


class Study(NameEnum):
    DOT_TEST_INSTRUCTIONS = auto()
    """Dot test instructions"""
    DOT_TEST = auto()
    """Dot test"""
    FITTS_LAW_INSTRUCTIONS = auto()
    """Fitts' law instructions"""
    FITTS_LAW = auto()
    """Fitts' law"""
    SERP_INSTRUCTIONS = auto()
    """SERP instructions"""
    BENEFITS_OF_RUNNING_INSTRUCTIONS = auto()
    """Benefits of running instructions"""
    BENEFITS_OF_RUNNING = auto()
    """Benefits of running"""
    BENEFITS_OF_RUNNING_WRITING = auto()
    """Benefits of running writing"""
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES_INSTRUCTIONS = auto()
    """Educational advantages of social networking sites instructions"""
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES = auto()
    """Educational advantages of social networking sites"""
    EDUCATIONAL_ADVANTAGES_OF_SOCIAL_NETWORKING_SITES_WRITING = auto()
    """Educational advantages of social networking sites writing"""
    WHERE_TO_FIND_MOREL_MUSHROOMS_INSTRUCTIONS = auto()
    """Where to find morel mushrooms instructions"""
    WHERE_TO_FIND_MOREL_MUSHROOMS = auto()
    """Where to find morel mushrooms"""
    WHERE_TO_FIND_MOREL_MUSHROOMS_WRITING = auto()
    """Where to find morel mushrooms writing"""
    TOOTH_ABSCESS_INSTRUCTIONS = auto()
    """Tooth abscess instructions"""
    TOOTH_ABSCESS = auto()
    """Tooth abscess"""
    TOOTH_ABSCESS_WRITING = auto()
    """Tooth abscess writing"""
    DOT_TEST_FINAL_INSTRUCTIONS = auto()
    """Dot test final instructions"""
    DOT_TEST_FINAL = auto()
    """Dot test final"""
    THANK_YOU = auto()
    """Thank you"""

    def _cmp(self, other: Self | str, /, op: Callable[[int, int], bool]):
        return op(self.id, (other if isinstance(other, Study) else Study(other)).id)

    @override
    def __ge__(self, other: str) -> bool:
        return self._cmp(other, operator.ge)

    @override
    def __gt__(self, other: str) -> bool:
        return self._cmp(other, operator.gt)

    @override
    def __le__(self, other: str) -> bool:
        return self._cmp(other, operator.le)

    @override
    def __lt__(self, other: str) -> bool:
        return self._cmp(other, operator.lt)


def webcam_durations_by_log(llf: LazyFrame) -> dict[tuple[int, int, Study], timedelta]:
    """Calculate durations of webcam videos using log data."""
    on = ("pid", "record")
    cols = [*on, "study"]
    ts = col("timestamp")
    llf = (
        llf.filter(event="start")
        .filter(col("study").ne(Study.THANK_YOU))
        .select(*cols, ts.alias("start"))
        .join(
            llf.filter(event="stop").select(*on, ts.alias("stop")),
            on,
            "left",
        )
        .join(
            llf.filter(event="start")
            .select(
                "pid",
                (col("record") - lit(1)).alias("record"),
                ts.alias("next"),
            )
            .filter(col("record") > 1)
            .join(llf.filter(event="stop"), on, "anti"),
            on,
            "left",
        )
        .with_columns(coalesce("stop", "next").alias("stop"))
        .select(*cols, (col("stop") - col("start")).alias("duration"))
    )
    return {
        (pid, record, study): duration
        for (pid, record, study, duration) in llf.collect().iter_rows()
    }


@overload
def pid_from_name(s: str, error: Literal[False]) -> int | None: ...
@overload
def pid_from_name(s: str, error: Literal[True]) -> int: ...
def pid_from_name(s: str, error: bool = False) -> int | None:
    m = search(r"P_(\d{2})", s.strip())
    if m is not None:
        return int(m.group(1))
    elif error:
        raise ValueError(f"Invalid participant name: {s!r}")


def concat_scan_zip(
    ref: ZipFile,
    scan_fn: Callable[[InputType, int], LazyFrame],
    filter_fn: Callable[[str], bool],
):
    return concat(
        [
            scan_fn(ref.open(file), pid_from_name(file.filename, error=True))
            for file in ref.filelist
            if filter_fn(file.filename)
        ]
    )


class ZipDataset:
    """
    A class to represent a dataset stored in a zip file.
    This is the raw dataset straight from the Eye of the Typer website.
    """

    ref: ZipFile

    def __init__(self, ref: ZipFile):
        self.ref = ref

    @property
    def files(self) -> list[ZipInfo]:
        return self.ref.filelist

    @cached_property
    def prefix(self) -> str:
        if (ref := self.ref).filename:
            return Path(ref.filename).stem
        else:
            return ref.filelist[0].filename.split("/")[0]

    def screen_files(self):
        return [
            *sorted(
                [f for f in self.files if f.filename.endswith((".flv", ".mov"))],
                key=lambda f: f.filename,
            )
        ]

    def webcam_files(self):
        return [
            *sorted(
                [f for f in self.files if f.filename.endswith(".webm")],
                key=lambda f: f.filename,
            )
        ]

    def scan_participants(self, sync: bool = True, alf: LazyFrame | None = None):
        """
        Scan participant file and add additional information parsed from the log and spec files.
        Returns a lazy frame for participant data and log source lazy frame.

        The `first_task_frames` parameter dictionary is used to override default
        adjustments for screen recording start times.
        """
        pdf = Characteristic.read_zip(self.ref)
        llf = Log.scan_zip(self.ref)
        slf = Spec.scan_zip(self.ref)

        # first task frame in screen recording and timestamp
        # when screen clock change minute
        adjustments_table = {
            "P_01": ("11s 733ms", 407),
            "P_02": ("3s 350ms", 403),
            "P_06": ("36s 828ms", -378),
            "P_07": ("5s 160ms", 436),
            "P_08": ("32s 32ms", -68),
            "P_10": ("2m 34s 29ms", -647),
            "P_12": ("34s 451ms", -159),
            "P_13": ("1m 3s 689ms", -209),
            "P_16": ("29s 863ms", -332),
            "P_17": ("21s 939ms", -549),
            "P_18": ("24s 441ms", -815),
            "P_19": ("25s 275ms", -765),
            "P_20": ("27s 27ms", -158),
            "P_23": ("24s 733ms", -82),
            "P_25": ("29s 655ms", -497),
            "P_27": ("25s 651ms", -819),
            "P_31": ("3m 51s 273ms", 6),
            "P_35": ("28s 362ms", -42),
            "P_40": ("21s 230ms", -743),
            "P_41": ("20s 395ms", -169),
            "P_42": ("18s 852ms", -376),
            "P_44": ("23s 398ms", -173),
            "P_45": ("30s 697ms", -501),
            "P_46": ("1m 1s 645ms", 6),
            "P_55": ("13s 805ms", -840),
            "P_56": ("22s 856ms", -122),
            "P_59": ("21s 813ms", -214),
        }

        if sync:
            adjustments_rows = [
                (
                    int(k.removeprefix("P_")),
                    parse_timedelta(ftf) - timedelta(milliseconds=scs),
                )
                for k, (ftf, scs) in adjustments_table.items()
            ]
        else:
            adjustments_rows = []

        adf = LazyFrame(
            adjustments_rows,
            schema={"pid": UInt8, "start_time": Duration("us")},
            orient="row",
        )
        odf = LazyFrame(
            [("Laptop", 97), ("PC", 66)],
            schema={"setting": Enum(Setting), "screen_viewport_y": Int32},
            orient="row",
        )

        plf = pdf.lazy().join(adf, on="pid", how="left")
        plf = plf.join(odf, on="setting", how="left")
        plf = plf.join(slf, on="pid", how="left")
        plf = plf.drop("screen_start").join(
            llf.select(
                "pid", "event", webcam_start="timestamp", screen_start="timestamp"
            )
            .filter(event="start")
            .drop("event")
            .sort("pid", "webcam_start")
            .group_by("pid")
            .first(),
            on="pid",
            how="left",
        )
        plf = plf.with_columns(screen_start=col("screen_start") - col("start_time"))
        plf = plf.drop("start_time")

        if alf is not None:
            llf = llf.join(alf.drop("log"), ["pid", "record", "study"], "left")
            llf = llf.with_columns(
                timestamp=col("timestamp")
                - col("frameshift").fill_null(duration(milliseconds=0, time_unit="ms")),
                # aligned=pl.col("frameshift").is_not_null(),
            ).drop("frameshift")

        return plf.sort("pid"), llf.sort("pid", "timestamp")


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


class SkinColor(StrEnum):
    C1 = "1"
    C2 = "2"
    C3 = "3"
    C4 = "4"
    C5 = "5"
    C6 = "6"


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


StudyName: TypeAlias = Literal[
    "dot_test_instructions",
    "dot_test",
    "fitts_law_instructions",
    "fitts_law",
    "serp_instructions",
    "benefits_of_running_instructions",
    "benefits_of_running",
    "benefits_of_running_writing",
    "educational_advantages_of_social_networking_sites_instructions",
    "educational_advantages_of_social_networking_sites",
    "educational_advantages_of_social_networking_sites_writing",
    "where_to_find_morel_mushrooms_instructions",
    "where_to_find_morel_mushrooms",
    "where_to_find_morel_mushrooms_writing",
    "tooth_abscess_instructions",
    "tooth_abscess",
    "tooth_abscess_writing",
    "dot_test_final_instructions",
    "dot_test_final",
    "thank_you",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Webcam:
    log: int
    """Participant log ID."""
    record: int
    """Participant record ID."""
    study: Study
    """Study name."""
    aux: int | None = None
    """Auxiliary number, used for multiple recordings in the same log and record."""

    @cache
    @staticmethod
    def webcam_raw_filename_pattern():
        return compile(r"([0-9]+)_([0-9]+)_-study-([a-z_]+)( \(([0-9]+)\))?\.webm$")

    @cache
    @staticmethod
    def webcam_filename_pattern():
        return compile(r"([0-9]+)_([0-9]+)_-study-([a-z_]+)( \(([0-9]+)\))?\.webm$")

    @classmethod
    def parse_raw(cls, s: str):
        if (r := cls.webcam_raw_filename_pattern().search(s)) is None:
            return

        log, record, study, _, aux = r.groups()

        log = int(log)
        record = int(record)
        study = Study(study)
        aux = int(aux) if aux is not None else None

        return cls(log=log, record=record, study=study, aux=aux)

    def _index(self):
        return (self.log, self.record, self.study.id, self.aux or 0)

    def _cmp(
        self,
        other: Self,
        op: Callable[[tuple[int, int, int, int], tuple[int, int, int, int]], bool],
    ) -> bool:
        return op(self._index(), other._index())

    def __ge__(self, other: Self) -> bool:
        return self._cmp(other, operator.ge)

    def __gt__(self, other: Self) -> bool:
        return self._cmp(other, operator.gt)

    def __le__(self, other: Self) -> bool:
        return self._cmp(other, operator.le)

    def __lt__(self, other: Self) -> bool:
        return self._cmp(other, operator.lt)


class Source(NameEnum):
    """
    Enumeration class representing different sources of input.
    """

    LOG = auto()
    """log events or unknown entries"""
    MOUSE = auto()
    """mouse events"""
    SCROLL = auto()
    """scroll events"""
    INPUT = auto()
    """text input events"""
    TEXT = auto()
    """text submit events"""
    TOBII = auto()
    """tobii recording"""
    DOT = auto()
    """dot test entries"""
    CALIB = auto()
    """tobii calibration data"""


class SourceEnumClass(metaclass=EnumMeta):
    @classmethod
    def schema(cls) -> dict[Any, PolarsDataType]:
        raise NotImplementedError()

    @classmethod
    def source_filename(cls, value: str) -> bool:  # pyright: ignore[reportUnusedParameter]
        raise NotImplementedError()

    @classmethod
    def scan_raw(cls, source: InputType) -> LazyFrame:  # pyright: ignore[reportUnusedParameter]
        raise NotImplementedError()

    @classmethod
    def scan(cls, source: InputType, pid: int) -> LazyFrame:
        return cls.scan_raw(source).with_columns(pid=lit(pid, UInt8))

    @classmethod
    def scan_zip(cls, ref: ZipFile) -> LazyFrame:
        return concat_scan_zip(ref, scan_fn=cls.scan, filter_fn=cls.source_filename)


class SourceClass(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def schema(cls) -> dict[str, PolarsDataType]: ...

    @classmethod
    @abstractmethod
    def source_filename(cls, value: str) -> bool: ...

    @classmethod
    @abstractmethod
    def scan_raw(cls, source: InputType) -> LazyFrame: ...

    @classmethod
    def scan(cls, source: InputType, pid: int) -> LazyFrame:
        return cls.scan_raw(source).with_columns(pid=lit(pid, UInt8))

    @classmethod
    def scan_zip(cls, ref: ZipFile) -> LazyFrame:
        return concat_scan_zip(ref, scan_fn=cls.scan, filter_fn=cls.source_filename)


class Event(StrEnum):
    SCROLL_EVENT = "scrollEvent"
    MOUSE_MOVE = "mousemove"
    MOUSE_CLICK = "mouseclick"
    TEXT_INPUT = "textInput"
    TEXT_SUBMIT = "text_input"
    REC_START = "recording start"
    REC_STOP = "recording stop"

    VIDEO_START = "video started"
    VIDEO_STOP = "video stop"
    VIDEO_SAVE = "video saved"

    @classmethod
    def video(cls):
        return {cls.VIDEO_START.value, cls.VIDEO_STOP.value, cls.VIDEO_SAVE.value}

    @classmethod
    def recording(cls):
        return {cls.REC_START.value, cls.REC_STOP.value}

    @classmethod
    def mouse(cls):
        return {cls.MOUSE_MOVE.value, cls.MOUSE_CLICK.value}

    @classmethod
    def type_aliases(cls):
        return {
            cls.MOUSE_CLICK: "click",
            cls.MOUSE_MOVE: "move",
            cls.REC_START: "start",
            cls.REC_STOP: "stop",
            cls.SCROLL_EVENT: "scroll",
            cls.TEXT_INPUT: "input",
            cls.TEXT_SUBMIT: "text",
        }


class Alignment(StrEnum):
    VIDEO = "video"
    FRAMESHIFT = "frameshift"
    CORR_OPTIMAL = "corrOptimal"
    CORR_ORIG = "corrOrig"

    @classmethod
    def schema(cls):
        return {
            cls.VIDEO: String,
            cls.FRAMESHIFT: Float64,
            cls.CORR_OPTIMAL: Float64,
            cls.CORR_ORIG: Float64,
        }

    @classmethod
    def scan_raw(cls, source: Path | bytes | IO[bytes]):
        return scan_csv(
            source,
            has_header=True,
            separator=",",
            quote_char='"',
            null_values=["-", ""],
            schema={k.value: v for k, v in cls.schema().items()},
        )

    @classmethod
    def scan(cls, source: Path | bytes | IO[bytes]):
        from dataclasses import asdict

        def parse_filename_col(v: str):
            w = Webcam.parse_raw(v)
            assert w is not None
            return asdict(w)

        lf = cls.scan_raw(source)
        lf = (
            lf.select(
                col(cls.VIDEO)
                .str.split_exact("/", 1)
                .struct.rename_fields(["parent", "filename"])
                .struct.unnest(),
                (col(cls.FRAMESHIFT) * 1e3).cast(Duration("ms")),
            )
            .select(
                col("parent")
                .str.split_exact("_", 1)
                .struct.rename_fields(["_", "pid"])
                .struct.field("pid")
                .cast(UInt8),
                col("filename")
                .map_elements(
                    parse_filename_col,
                    return_dtype=Struct(
                        {
                            "log": Int64,
                            "record": Int64,
                            "study": String,
                            "aux": Int64,
                        }
                    ),
                )
                .struct.unnest(),
                cls.FRAMESHIFT,
            )
            .drop("aux")
            .with_columns(
                col("pid").cast(UInt8),
                col("log").cast(UInt64),
                col("record").cast(UInt8),
                col("study").cast(Enum(Study.values())),
            )
            .sort("pid", "record")
        )

        return lf


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

    @classmethod
    def schema(cls):
        return {
            cls.PID: String,
            cls.LOG_ID: UInt64,
            cls.DATE: String,
            cls.SETTING: Categorical,
            cls.DISPLAY_WIDTH: UInt16,
            cls.DISPLAY_HEIGHT: UInt16,
            cls.SCREEN_WIDTH: Float64,
            cls.SCREEN_HEIGHT: Float64,
            cls.DISTANCE_FROM_SCREEN: Float64,
            cls.SCREEN_RECORDING_START_TIME_UNIX: UInt64,
            cls.SCREEN_RECORDING_START_TIME_UTC: String,
            cls.GENDER: Categorical,
            cls.AGE: UInt8,
            cls.RACE: Categorical,
            cls.SKIN_COLOR: Categorical,
            cls.EYE_COLOR: Categorical,
            cls.FACIAL_HAIR: Categorical,
            cls.VISION: Categorical,
            cls.TOUCH_TYPER: Categorical,
            cls.HANDEDNESS: Categorical,
            cls.WEATHER: Categorical,
            cls.POINTING_DEVICE: Categorical,
            cls.NOTES: String,
            cls.TIME_OF_DAY: String,
            cls.DURATION: String,
        }

    @classmethod
    def aliases(cls):
        return {
            cls.PID: "pid",
            cls.LOG_ID: "log",
            cls.DATE: "date",
            cls.SETTING: "setting",
            cls.DISPLAY_WIDTH: "display_width",
            cls.DISPLAY_HEIGHT: "display_height",
            cls.SCREEN_WIDTH: "screen_width",
            cls.SCREEN_HEIGHT: "screen_height",
            cls.DISTANCE_FROM_SCREEN: "screen_distance",
            cls.SCREEN_RECORDING_START_TIME_UNIX: "screen_start",
            cls.SCREEN_RECORDING_START_TIME_UTC: "wall_clock",
            cls.GENDER: "gender",
            cls.AGE: "age",
            cls.RACE: "race",
            cls.SKIN_COLOR: "skin_color",
            cls.EYE_COLOR: "eye_color",
            cls.FACIAL_HAIR: "facial_hair",
            cls.VISION: "vision",
            cls.TOUCH_TYPER: "touch_typer",
            cls.HANDEDNESS: "handedness",
            cls.WEATHER: "weather",
            cls.POINTING_DEVICE: "pointing_device",
            cls.NOTES: "notes",
            cls.TIME_OF_DAY: "time_of_day",
            cls.DURATION: "duration",
        }

    @classmethod
    def enums(cls) -> dict[Self, type[StrEnum]]:
        return {  # pyright: ignore[reportReturnType]
            cls.SETTING: Setting,
            cls.GENDER: Gender,
            cls.RACE: Race,
            cls.EYE_COLOR: EyeColor,
            cls.SKIN_COLOR: SkinColor,
            cls.FACIAL_HAIR: FacialHair,
            cls.VISION: Vision,
            cls.HANDEDNESS: Handedness,
            cls.WEATHER: Weather,
            cls.POINTING_DEVICE: PointingDevice,
        }

    @classmethod
    def read_raw(cls, source: Path | bytes | IO[bytes]):
        return read_csv(
            source,
            has_header=True,
            separator=",",
            quote_char='"',
            null_values=["-", ""],
            schema={k.value: v for k, v in cls.schema().items()},
        )

    @classmethod
    def read(cls, source: Path | bytes | IO[bytes]):
        df = cls.read_raw(source)
        df = df.with_columns(
            # indices
            col(cls.PID).str.split("_").list.get(1).str.to_integer().cast(UInt8),
            from_epoch(cls.LOG_ID, "ms").alias("init_start"),
            # timestamps, dates, times, durations
            col(cls.DATE).str.to_date(r"%m/%d/%Y"),
            col(cls.DURATION)
            .str.split(":")
            .cast(Array(UInt16, 3))
            .map_elements(
                lambda arr: (arr[0] * 3600 + arr[1] * 60 + arr[2]) * 1000,
                return_dtype=Int64,
            )
            .cast(Duration("ms")),
            col(cls.TIME_OF_DAY).str.to_time(r"%H:%M"),
            from_epoch(cls.SCREEN_RECORDING_START_TIME_UNIX, "ms"),
            col(cls.SCREEN_RECORDING_START_TIME_UTC).str.to_datetime(
                r"%m/%d/%Y %H:%M", time_zone="UTC", time_unit="ms"
            ),
            # boolean characteristics
            col(cls.TOUCH_TYPER).eq("Yes").cast(Boolean),
            # .replace({"Yes": True, "No": False}, return_dtype=Boolean),
            # enum characteristics
            *(
                col(c).cast(Enum([m.value for m in e.__members__.values()]))
                for c, e in cls.enums().items()
            ),
        )
        df = df.with_columns(
            # screen sizes
            screen_size=struct(w=cls.SCREEN_WIDTH, h=cls.SCREEN_HEIGHT),
            display_resolution=struct(w=cls.DISPLAY_WIDTH, h=cls.DISPLAY_HEIGHT),
            # screen recording start
            # rec_time=col("start_time"),
            # + col(cls.PID).map_elements(get_record_offset, Duration("us")),
        )
        df = df.drop(
            cls.SCREEN_WIDTH,
            cls.SCREEN_HEIGHT,
            cls.DISPLAY_WIDTH,
            cls.DISPLAY_HEIGHT,
            cls.DATE,
            cls.DURATION,
            # cls.SCREEN_RECORDING_START_TIME_UTC,
            # cls.SCREEN_RECORDING_START_TIME_UNIX,
        )
        return df.sort(by=cls.PID).rename(
            {k.value: v for k, v in cls.aliases().items() if k.value in df.columns}
        )

    @classmethod
    def read_zip(cls, ref: ZipFile):
        file = next(
            filter(
                lambda f: "participant_characteristics.csv" in f.filename, ref.filelist
            )
        )
        return cls.read(ref.open(file))


class Log(SourceEnumClass, StrEnum):
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

    @classmethod
    @override
    def schema(cls):
        return cast(
            dict[Log, PolarsDataType],
            {
                cls.IS_TRUSTED: Boolean,
                cls.SESSION_ID: String,
                cls.WEBPAGE: String,
                cls.SESSION_STRING: String,
                cls.EPOCH: UInt64,
                cls.TIME: Float64,
                cls.TYPE: String,
                cls.EVENT: String,
                cls.TEXT: String,
                cls.POS: Struct({"top": UInt32, "left": UInt32}),
                cls.SCREEN_X: Int32,
                cls.SCREEN_Y: Int32,
                cls.CLIENT_X: UInt16,
                cls.CLIENT_Y: UInt16,
                cls.PAGE_X: UInt32,
                cls.PAGE_Y: UInt32,
                cls.SCROLL_X: UInt32,
                cls.SCROLL_Y: UInt32,
                cls.WINDOW_X: Int32,
                cls.WINDOW_Y: Int32,
                cls.WINDOW_INNER_WIDTH: UInt16,
                cls.WINDOW_INNER_HEIGHT: UInt16,
                cls.WINDOW_OUTER_WIDTH: UInt16,
                cls.WINDOW_OUTER_HEIGHT: UInt16,
            },
        )

    @classmethod
    @override
    def source_filename(cls, value: str) -> bool:
        return value.endswith(".json")

    @classmethod
    @override
    def scan_raw(cls, source: Path | bytes):  # pyright: ignore[reportIncompatibleMethodOverride]
        return read_json(
            source,
            schema={k.value: v for k, v in cls.schema().items()},
            infer_schema_length=1,
        ).lazy()

    @classmethod
    @override
    def scan(cls, source: Path | bytes, pid: int):  # pyright: ignore[reportIncompatibleMethodOverride]
        lf = cls.scan_raw(source)
        lf = lf.with_columns(
            col(cls.WEBPAGE)
            .str.strip_prefix("/study/")
            .str.strip_suffix(".htm")
            .replace("thankyou", "thank_you")
            .cast(Enum(Study.values())),
            col(cls.TYPE).cast(Categorical("lexical")),
            col(cls.EVENT).cast(Categorical("lexical")),
            col(cls.SESSION_ID).str.split("_").list.get(0).cast(UInt64),
            col(cls.SESSION_ID)
            .str.split("_")
            .list.get(1)
            .cast(UInt8)
            .alias(cls.SESSION_STRING),
            from_epoch(cls.EPOCH, "ms"),
            col(cls.TIME).cast(Duration("ms")),
        )
        lf = lf.with_columns(
            when(col(cls.TYPE).is_not_null())
            .then(
                col(cls.TYPE).replace(
                    {k.value: v for k, v in Event.type_aliases().items()}
                )
            )
            .when(col(cls.EVENT).is_not_null() & col(cls.EVENT).eq(Event.VIDEO_SAVE))
            .then(lit("save"))
            .otherwise(None)
            .cast(Categorical("lexical"))
            .alias(cls.EVENT),
            when(
                col(cls.TYPE).is_in(Event.recording())
                | col(cls.EVENT).is_in(Event.video())
            )
            .then(lit(Source.LOG))
            .when(col(cls.TYPE).is_in(Event.mouse()))
            .then(lit(Source.MOUSE))
            .when(col(cls.TYPE) == Event.SCROLL_EVENT)
            .then(lit(Source.SCROLL))
            .when(col(cls.TYPE) == Event.TEXT_INPUT)
            .then(lit(Source.INPUT))
            .when(col(cls.TYPE) == Event.TEXT_SUBMIT)
            .then(lit(Source.TEXT))
            .otherwise(lit(Source.LOG))
            .cast(Categorical("lexical"))
            .alias(cls.TYPE),
        )
        return lf.sort(cls.EPOCH).select(
            pid=lit(pid, UInt8),
            record=cls.SESSION_STRING,
            timestamp=col(cls.EPOCH),  # - col(cls.SESSION_ID).cast(Datetime("ms")),
            study=cls.WEBPAGE,
            event=cls.EVENT,
            source=cls.TYPE,
            trusted=cls.IS_TRUSTED,
            duration=cls.TIME,
            caret=cls.POS,
            text=cls.TEXT,
            page=struct(x=cls.PAGE_X, y=cls.PAGE_Y),
            mouse=struct(x=cls.SCREEN_X, y=cls.SCREEN_Y),
            scroll=struct(x=cls.SCROLL_X, y=cls.SCROLL_Y),
            window=struct(x=cls.WINDOW_X, y=cls.WINDOW_Y),
            inner=struct(w=cls.WINDOW_INNER_WIDTH, h=cls.WINDOW_INNER_HEIGHT),
            outer=struct(w=cls.WINDOW_OUTER_WIDTH, h=cls.WINDOW_OUTER_HEIGHT),
        )


class Tobii(SourceEnumClass, StrEnum):
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

    @classmethod
    @override
    def schema(cls):
        return cast(
            dict[Self, PolarsDataType],
            {
                cls.DEVICE_TIME_STAMP: UInt64,
                cls.SYSTEM_TIME_STAMP: UInt64,
                cls.TRUE_TIME: Float64,
                cls.LEFT_PUPIL_VALIDITY: Int32,
                cls.RIGHT_PUPIL_VALIDITY: Int32,
                cls.LEFT_GAZE_ORIGIN_VALIDITY: Int32,
                cls.RIGHT_GAZE_ORIGIN_VALIDITY: Int32,
                cls.LEFT_GAZE_POINT_VALIDITY: Int32,
                cls.RIGHT_GAZE_POINT_VALIDITY: Int32,
                cls.LEFT_PUPIL_DIAMETER: Float64,
                cls.RIGHT_PUPIL_DIAMETER: Float64,
                cls.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
                cls.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
                cls.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM: Array(Float64, 3),
                cls.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM: Array(Float64, 3),
                cls.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
                cls.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM: Array(Float64, 3),
                cls.LEFT_GAZE_POINT_ON_DISPLAY_AREA: Array(Float64, 2),
                cls.RIGHT_GAZE_POINT_ON_DISPLAY_AREA: Array(Float64, 2),
            },
        )

    @classmethod
    @override
    def source_filename(cls, value: str) -> bool:
        return search(r"P_[0-9][0-9]\.txt$", value) is not None

    @classmethod
    @override
    def scan_raw(cls, source):
        # handle corrupted files
        # polars cannot ignore parsing errors: https://github.com/pola-rs/polars/issues/13768
        test_end = b"}\n"
        # is_file = False
        if isinstance(source, (bytes, bytearray, memoryview)):
            if source[-3:-1] != test_end:
                if isinstance(source, memoryview):  # pyright: ignore[reportUnnecessaryIsInstance]
                    source = source.tobytes()
                source = b"".join(source.splitlines()[:-1])
        else:
            if isinstance(source, Path):
                source = source.open("rb")
                # is_file = True
            _ = source.seek(-2, SEEK_END)
            if source.read(2) != test_end:
                _ = source.seek(0, SEEK_SET)
                source = b"".join(source.readlines()[:-1])
            else:
                _ = source.seek(0, SEEK_SET)

        return scan_ndjson(
            source,
            schema={k.value: v for k, v in cls.schema().items()},
            infer_schema_length=1,
            low_memory=True,
        )

    @classmethod
    @override
    def scan(cls, source: InputType, pid: int):
        xyz = ("x", "y", "z")
        xy = ("x", "y")
        lf = cls.scan_raw(source)
        # print(pid, lf.count().collect())
        lf = lf.with_columns(
            from_epoch(col(cls.TRUE_TIME).mul(1e9), "ns"),
            from_epoch(cls.DEVICE_TIME_STAMP, "us"),
            from_epoch(cls.SYSTEM_TIME_STAMP, "us"),
            col(cls.LEFT_PUPIL_VALIDITY).cast(Boolean),
            col(cls.RIGHT_PUPIL_VALIDITY).cast(Boolean),
            col(cls.LEFT_GAZE_ORIGIN_VALIDITY).cast(Boolean),
            col(cls.RIGHT_GAZE_ORIGIN_VALIDITY).cast(Boolean),
            col(cls.LEFT_GAZE_POINT_VALIDITY).cast(Boolean),
            col(cls.RIGHT_GAZE_POINT_VALIDITY).cast(Boolean),
            col(cls.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM).arr.to_struct(xyz),
            col(cls.LEFT_GAZE_POINT_ON_DISPLAY_AREA).arr.to_struct(xy),
            col(cls.RIGHT_GAZE_POINT_ON_DISPLAY_AREA).arr.to_struct(xy),
        )
        return lf.sort(cls.TRUE_TIME).select(
            pid=lit(pid, UInt8),
            timestamp=cls.TRUE_TIME,
            clock=cls.SYSTEM_TIME_STAMP,
            duration=cls.DEVICE_TIME_STAMP,
            # pupil=struct(
            pupil_validity=struct(
                left=cls.LEFT_PUPIL_VALIDITY, right=cls.RIGHT_PUPIL_VALIDITY
            ),
            pupil_diameter=struct(
                left=cls.LEFT_PUPIL_DIAMETER, right=cls.RIGHT_PUPIL_DIAMETER
            ),
            # ),
            # gaze=struct(
            # point=struct(
            gazepoint_validity=struct(
                left=cls.LEFT_GAZE_POINT_VALIDITY,
                right=cls.RIGHT_GAZE_POINT_VALIDITY,
            ),
            gazepoint_display=struct(
                left=cls.LEFT_GAZE_POINT_ON_DISPLAY_AREA,
                right=cls.RIGHT_GAZE_POINT_ON_DISPLAY_AREA,
            ),
            gazepoint_ucs=struct(
                left=cls.LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM,
                right=cls.RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM,
            ),
            # ),
            # origin=struct(
            gazeorigin_validity=struct(
                left=cls.LEFT_GAZE_ORIGIN_VALIDITY,
                right=cls.RIGHT_GAZE_ORIGIN_VALIDITY,
            ),
            gazeorigin_trackbox=struct(
                left=cls.LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM,
                right=cls.RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM,
            ),
            gazeorigin_ucs=struct(
                left=cls.LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM,
                right=cls.RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM,
            ),
            # ),
            # ),
        )


Corner = tuple[
    Literal["back", "front"],
    Literal["lower", "upper"],
    Literal["left", "right"],
]
Point3 = tuple[float, float, float]


class Spec(SourceClass):
    tracking_box: dict[Corner, Point3]
    ilumination_mode: str | None
    frequency: float | None

    def __init__(
        self,
        tracking_box: dict[Corner, Point3],
        ilumination_mode: str | None = None,
        frequency: float | None = None,
    ):
        self.tracking_box = tracking_box
        self.ilumination_mode = ilumination_mode
        self.frequency = frequency

    @classmethod
    def from_lines(cls, lines: Iterable[bytes]) -> Self:
        tracking_box: dict[Corner, Point3] = {}
        ilumination_mode: str | None = None
        frequency: float | None = None

        for line in lines:
            line = line.decode("utf-8").strip()
            m = match(r"(Back|Front) (Lower|Upper) (Left|Right):", line)
            if m is not None:
                index = cast(
                    Corner, (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())
                )

                values = line[m.end() :].strip(" ()\n\t").split(", ")
                x, y, z = [float(v) for v in values]

                tracking_box[index] = (x, y, z)

            m = match(r"Illumination mode:", line)
            if m is not None:
                ilumination_mode = line[m.end() :].strip()

            m = match(r"Initial gaze output frequency: (\d+\.?\d+)", line)
            if m is not None:
                frequency = float(m.group(1))

        return cls(
            tracking_box=tracking_box,
            ilumination_mode=ilumination_mode,
            frequency=frequency,
        )

    @classmethod
    @override
    def schema(cls):
        point = Array(Float64, 3)
        return cast(
            dict[str, PolarsDataType],
            {
                "tracking_box": Struct(
                    {
                        "b_lo_l": point,
                        "b_lo_r": point,
                        "b_up_l": point,
                        "b_up_r": point,
                        "f_lo_l": point,
                        "f_lo_r": point,
                        "f_up_l": point,
                        "f_up_r": point,
                    }
                ),
                "illumination_mode": Categorical(),
                "frequency": Float64,
            },
        )

    def to_lf(self, pid: int | None = None) -> LazyFrame:
        lf = LazyFrame(
            {
                "tracking_box": [
                    {
                        f"{k[0][0]}_{k[1][:2]}_{k[2][0]}": [v[0], v[1], v[2]]
                        for k, v in self.tracking_box.items()
                    }
                ],
                "illumination_mode": [self.ilumination_mode],
                "frequency": [self.frequency],
            },
            self.schema(),
        )
        if pid is not None:
            lf = lf.with_columns(pid=lit(pid, UInt8))
        return lf

    @classmethod
    @override
    def scan_raw(cls, source):
        if isinstance(source, Path):
            with source.open("rb") as src:
                payload = src.readlines()
        elif isinstance(source, (bytes, bytearray)):
            payload = source.splitlines()
        elif isinstance(source, memoryview):  # pyright: ignore[reportUnnecessaryIsInstance]
            payload = source.tobytes().splitlines()
        else:
            payload = cast(IO[bytes], source)  # pyright: ignore[reportUnnecessaryCast]

        return cls.from_lines(payload).to_lf()

    @classmethod
    @override
    def source_filename(cls, value: str) -> bool:
        return value.endswith("specs.txt")


class Calib(SourceEnumClass, StrEnum):
    VALIDITY_LEFT = auto()
    VALIDITY_RIGHT = auto()
    POINT_X = auto()
    POINT_Y = auto()
    PREDICTION_X_LEFT = auto()
    PREDICTION_Y_LEFT = auto()
    PREDICTION_X_RIGHT = auto()
    PREDICTION_Y_RIGHT = auto()

    @classmethod
    @override
    def schema(cls):
        return cast(
            dict[Self, PolarsDataType],
            {
                cls.POINT_X: Float64,
                cls.POINT_Y: Float64,
                cls.PREDICTION_X_LEFT: Float64,
                cls.PREDICTION_Y_LEFT: Float64,
                cls.VALIDITY_LEFT: Int8,
                cls.PREDICTION_X_RIGHT: Float64,
                cls.PREDICTION_Y_RIGHT: Float64,
                cls.VALIDITY_RIGHT: Int8,
            },
        )

    @classmethod
    @override
    def scan_raw(cls, source):
        return scan_csv(
            source,
            has_header=True,
            separator="\t",
            schema={key.value: value for key, value in cls.schema().items()},
            ignore_errors=True,
            raise_if_empty=True,
        )

    @classmethod
    @override
    def scan(cls, source, pid: int):
        def xy(name: str):
            return struct(x=col(f"{name}_x"), y=col(f"{name}_y"))

        def lr(name: str):
            return struct(left=col(f"{name}_left"), right=col(f"{name}_right"))

        def pred(name: str):
            return struct(x=col(f"prediction_x_{name}"), y=col(f"prediction_y_{name}"))

        lf = cls.scan_raw(source).drop_nulls()
        lf = lf.with_columns(col(cls.VALIDITY_LEFT) > 0, col(cls.VALIDITY_RIGHT) > 0)
        return lf.select(
            pid=lit(pid, UInt8),
            point=xy("point"),
            validity=lr("validity"),
            left=pred("left"),
            right=pred("right"),
        )

    @classmethod
    @override
    def source_filename(cls, value: str) -> bool:
        return value.endswith("specs.txt")


class Dot(SourceEnumClass, StrEnum):
    X = "Dot_X"
    Y = "Dot_Y"
    EPOCH = "Epoch"

    @classmethod
    @override
    def schema(cls):
        return cast(
            dict[Self, PolarsDataType],
            {
                cls.X: Int32,
                cls.Y: Int32,
                cls.EPOCH: Float64,
            },
        )

    @classmethod
    @override
    def scan_raw(cls, source):
        return scan_csv(
            source,
            has_header=True,
            separator="\t",
            schema={key.value: value for key, value in cls.schema().items()},
            ignore_errors=True,
            infer_schema_length=0,
            raise_if_empty=True,
        )

    @classmethod
    @override
    def scan(cls, source, pid: int):
        return (
            cls.scan_raw(source)
            .sort(cls.EPOCH)
            .select(
                pid=lit(pid, UInt8),
                timestamp=col(cls.EPOCH).cast(Datetime("ms")),
                dot=struct(x=cls.X, y=cls.Y),
            )
        )

    @classmethod
    @override
    def source_filename(cls, value: str) -> bool:
        return value.endswith("final_dot_test_locations.tsv")
