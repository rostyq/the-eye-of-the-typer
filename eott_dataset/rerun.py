from string import Template

import rerun as rr

from .types import *
from .characteristics import *
from .participant import Participant


RECORDING = Template(
    """
Participant $pid
$start_time
recording started: $rec_time
study: $record - $study
"""
)

CHARACTERISTICS = Template(
    """
age: $age
gender: $gender
race: $race
skin color: $skin_color
eye color: $eye_color
facial hair: $facial_hair
vision: $vision
touch typer: $touch_typer
handedness: $handedness
"""
)

SETUP = Template(
    """
time of day: $time_of_day
weather: $weather
setting: $setting
pointing device: $pointing_device
screen distance: $screen_distance cm
screen dimensions: $screen_dim cm
display resolution: $screen_res
"""
)


def rerun_log_participant(p: Participant, record: int):
    rr.log(
        f"screen",
        rr.Boxes2D(
            sizes=[p.screen_res], centers=[tuple(map(lambda v: 0.5 * v, p.screen_res))]
        ),
        timeless=True,
    )

    rr.log(
        ["pupil", "left"],
        rr.SeriesLine(name="left pupil diameter", color=(255, 255, 0), width=1),
        timeless=True,
    )
    rr.log(
        ["pupil", "right"],
        rr.SeriesLine(name="right pupil diameter", color=(255, 0, 255), width=1),
        timeless=True,
    )

    txt = RECORDING.substitute(
        pid=p.pid,
        start_time=str(p.start_time),
        rec_time=str(p.rec_time),
        record=record,
        study=p.data[Source.WEBCAM].filter(record=record, aux=0)["study"][0]
    )
    rr.log("recording", rr.TextDocument(txt), timeless=True)

    txt = CHARACTERISTICS.substitute(
        age=p.age,
        gender=p.gender,
        race=p.race,
        skin_color=p.skin_color,
        eye_color=p.eye_color,
        facial_hair=p.facial_hair,
        vision=p.vision,
        touch_typer=p.touch_typer,
        handedness=p.handedness,
    )
    rr.log("characteristics", rr.TextDocument(txt), timeless=True)

    txt = SETUP.substitute(
        time_of_day=str(p.time_of_day),
        weather=p.weather,
        setting=p.setting,
        pointing_device=p.pointing_device,
        screen_distance="%.1f" % p.screen_distance,
        screen_dim="%.1f x %.1f" % p.screen_dim,
        screen_res="%d x %d" % p.screen_res,
    )
    rr.log("setup", rr.TextDocument(txt), timeless=True)


def rerun_log_tobii(entry: TobiiEntry, *, screen: tuple[int, int]):
    sw, sh = screen
    for side in ("left", "right"):
        if entry["gazepoint_validity"][side]:
            point = entry["gazepoint_display"][side]
            positions = [[point["x"] * sw, point["y"] * sh]]
            entity = rr.Points2D(positions, colors=[(0, 0, 255)], radii=[5])
        else:
            entity = rr.Clear(recursive=True)
        rr.log(["screen", "gaze", side], entity)

        if entry["pupil_diameter"][side] > 0:
            entity = rr.Scalar(entry["pupil_diameter"][side])
        else:
            entity = rr.Clear(recursive=True)
        rr.log(["pupil", side], entity)


def rerun_log_mouse(entry: MouseEntry):
    point = entry["mouse"]
    positions = [(point["x"], point["y"])]
    color = (255, 0, 0) if entry["event"] == "click" else (255, 255, 0)
    entity = rr.Points2D(positions, colors=[color], radii=[5])
    rr.log(["screen", "mouse"], entity)
