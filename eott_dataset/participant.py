from typing import Generator, cast, Any
from io import BytesIO
from datetime import datetime, time

from decord import VideoReader, cpu
from decord.ndarray import NDArray
from polars import DataFrame

from .characteristics import *
from .types import *


__all__ = ["Participant"]


class Participant:
    pid: int
    log: int
    age: int

    settings: Setting
    gender: Gender
    race: Race
    skin_color: SkinColor
    eye_color: EyeColor
    facial_hair: FacialHair
    vision: Vision
    touch_typer: bool
    handedness: Handedness
    weather: Weather
    pointing_device: PointingDevice

    time_of_day: time
    start_time: datetime
    rec_time: datetime

    screen_distance: float

    screen_dim: tuple[float, float]
    screen_res: tuple[int, int]

    notes: str

    screen_rec: bytes | None

    data: dict[Source, DataFrame]

    def __init__(
        self,
        *,
        form: dict[str, Any],
        screen: bytes | None = None,
        data: dict[Source, DataFrame] = {},
    ):
        self.pid = int(form.pop("pid"))
        self.log = int(form.pop("log"))
        self.age = int(form.pop("age"))

        self.setting = Setting(form.pop("setting"))
        self.gender = Gender(form.pop("gender"))
        self.race = Race(form.pop("race"))
        self.skin_color = SkinColor(form.pop("skin_color"))
        self.eye_color = EyeColor(form.pop("eye_color"))
        self.facial_hair = FacialHair(form.pop("facial_hair"))
        self.vision = Vision(form.pop("vision"))
        self.touch_typer = bool(form.pop("touch_typer"))
        self.handedness = Handedness(form.pop("handedness"))
        self.weather = Weather(form.pop("weather"))
        self.pointing_device = PointingDevice(form.pop("pointing_device"))

        self.time_of_day = form.pop("time_of_day")
        self.start_time = form.pop("start_time")
        self.rec_time = form.pop("rec_time")

        self.screen_distance = float(form.pop("screen_distance"))

        screen_dim: dict[Literal["w", "h"], Any] = form.pop("screen")
        self.screen_dim = float(screen_dim["w"]), float(screen_dim["h"])
        screen_res: dict[Literal["w", "h"], Any] = form.pop("display")
        self.screen_res = int(screen_res["w"]), int(screen_res["h"])
        self.screen_rec = screen

        self.data = data

    def get_entry(self, source: Source, index: int) -> dict:
        return self.data[source].row(index, named=True)

    def iter_timeline(
        self, *predicates, **constraints
    ) -> Generator[TimelineItem, None, None]:
        sb: BytesIO | None = None
        sr: VideoReader | None = None

        wb: BytesIO | None = None
        wr: VideoReader | None = None

        try:
            ctx = cpu(0)

            if self.screen_rec is not None:
                sb = BytesIO(self.screen_rec)
                sr = VideoReader(sb, ctx=ctx, num_threads=1)

            timeline = self.data[Source.TIMELINE].filter(*predicates, **constraints)

            for entry in map(lambda x: cast(TimelineEntry, x), timeline.iter_rows(named=True)):
                source, record, index = entry["source"], entry["record"], entry["index"]
                match source:
                    case (
                        "mouse" | "scroll" | "text" | "input" | "dot" | "log" | "tobii"
                    ):
                        data = self.get_entry(Source(source), index)

                    case "screen" if sr is not None:
                        frame_array: NDArray = sr[index]
                        frame_timestamp = float(sr.get_frame_timestamp(index)[0])
                        frame_timestamp = timedelta(seconds=frame_timestamp)

                        data = dict(frame=frame_array.asnumpy(), time=frame_timestamp)

                        del frame_array

                    case "webcam":
                        if wr is None:
                            payload = self.data[Source.WEBCAM]
                            payload = payload.filter(record=record, aux=0)
                            wb = BytesIO(payload["file"][0])
                            wr = VideoReader(wb, ctx=ctx, num_threads=1)

                        frame_array: NDArray = wr[index]
                        frame_timestamp = timedelta(seconds=index * wr.get_avg_fps())

                        data = dict(frame=frame_array.asnumpy(), time=frame_timestamp)

                        del frame_array

                        if (index + 1) >= len(wr):
                            del wr
                            wr = None

                    case _:
                        raise ValueError(f"Unexpected source: {source}")

                yield cast(TimelineItem, {**entry, "data": data})

        finally:
            del sr, wr, ctx

            if sb is not None:
                sb.close()
            if wb is not None:
                wb.close()

    def __getitem__(self, args: tuple[Source, int]):
        return self.get_entry(*args)
