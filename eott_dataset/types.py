from typing import TypedDict, TypeVar, Generic, Optional, Literal, Union
from datetime import datetime, timedelta
from numpy.typing import NDArray as NPArray


T = TypeVar("T")


StudyType = Literal[
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
EventType = Literal["click", "move", "start", "stop", "scroll", "input", "text"]
SourceType = Literal[
    "log",
    "mouse",
    "scroll",
    "input",
    "text",
    "tobii",
    "webcam",
    "screen",
    "dot",
    "calib",
    "trackbox",
    "form",
    "timeline",
]


class Point3(TypedDict):
    x: float
    y: float
    z: float


class Point2(TypedDict):
    x: float
    y: float


class Pair(Generic[T], TypedDict):
    left: T
    right: T


class Offset(TypedDict):
    top: int
    left: int


class Position(TypedDict):
    x: int
    y: int


class Size(TypedDict):
    w: int
    h: int


class TobiiEntry(TypedDict):
    timestamp: datetime
    clock: datetime
    duration: timedelta
    pupil_validity: Pair[bool]
    pupil_diameter: Pair[float]
    gazepoint_validity: Pair[bool]
    gazepoint_display: Pair[Point2]
    gazepoint_ucs: Pair[Point3]
    gazeorigin_validity: Pair[bool]
    gazeorigin_trackbox: Pair[Point3]
    gazeorigin_ucs: Pair[Point3]


class DotEntry(TypedDict):
    timestamp: datetime
    dot: Point2


class CalibEntry(Pair[Point2]):
    point: Point2
    validity: Pair[bool]


class InputEntry(TypedDict):
    record: int
    timestamp: datetime
    study: str
    duration: timedelta
    trusted: Optional[bool]
    event: Literal["input"]
    caret: Offset
    text: str


class LogEntry(TypedDict):
    record: int
    timestamp: datetime
    study: StudyType
    duration: timedelta
    trusted: Optional[bool]
    event: Literal["start", "stop", "save"]


class MouseEntry(TypedDict):
    record: int
    timestamp: datetime
    study: StudyType
    duration: timedelta
    trusted: Optional[bool]
    event: Literal["click", "move"]
    page: Position
    mouse: Position
    window: Position
    inner: Size
    outer: Size


class ScrollEntry(TypedDict):
    record: int
    timestamp: datetime
    study: StudyType
    duration: timedelta
    trusted: Optional[bool]
    event: Literal["scroll"]
    scroll: Position


class TextEntry(TypedDict):
    record: int
    timestamp: datetime
    study: StudyType
    duration: timedelta
    trusted: Optional[bool]
    text: str


class TimelineEntry(TypedDict):
    record: int
    study: StudyType
    source: SourceType
    index: int
    offset: timedelta
    frame: int


class BoundingPart(TypedDict):
    z: Literal["front", "back"]
    y: Literal["lower", "upper"]
    x: Literal["left", "right"]


class TrackboxEntry(TypedDict):
    point: list[float]
    position: BoundingPart


class FrameEntry(TypedDict):
    frame: NPArray
    time: float


class WebcamEntry(TypedDict):
    pid: int
    record: int
    study: StudyType
    aux: int
    file: bytes


class TimelineItem(TimelineEntry):
    data: Union[
        FrameEntry,
        TextEntry,
        ScrollEntry,
        MouseEntry,
        LogEntry,
        InputEntry,
        DotEntry,
    ]
