import cv2 as cv
import rerun as rr

from .entries import *


def rerun_log_tobii(entry: TobiiEntry, *, screen: tuple[int, int]):
    sw, sh = screen
    for eye in ("left", "right"):
        if entry[f"{eye}_gaze_point_validity"]:
            x, y = entry[f"{eye}_gaze_point_on_display_area"]
            rr.log(
                f"screen/gazepoint/{eye}/tobii",
                rr.Points2D(
                    [[x * sw, y * sh]],
                    colors=[[0, 0, 255]],
                    radii=[1],
                ),
            )
        else:
            rr.log(
                f"screen/gazepoint/{eye}/tobii",
                rr.Clear(recursive=True),
            )

        if entry[f"{eye}_pupil_validity"] and entry[f"{eye}_pupil_diameter"] > 0:
            rr.log(
                f"participant/pupil/{eye}/tobii",
                rr.Scalar(entry[f"{eye}_pupil_diameter"]),
            )
        else:
            rr.log(f"participant/pupil/{eye}/tobii", rr.Clear(recursive=True))


def rerun_log_screen(
    cap: cv.VideoCapture,
    *,
    position: int | None = None,
    size: tuple[int, int] | None = None,
    convert: int = cv.COLOR_BGR2RGB,
):
    if position is not None and position != cap.get(cv.CAP_PROP_POS_FRAMES):
        cap.set(cv.CAP_PROP_POS_FRAMES, position)
        rr.log(
            "log/event",
            rr.TextLog(f"screen frame skip", level=rr.TextLogLevel.WARN),
        )

    rr.set_time_sequence("screen_index", int(cap.get(cv.CAP_PROP_POS_FRAMES)))
    rr.set_time_seconds("screen_time", cap.get(cv.CAP_PROP_POS_MSEC) / 1_000)

    success, image = cap.read()

    if not success:
        return cap.release()

    if position % 10 != 0:
        return cap

    if size is not None:
        image = cv.resize(image, size)

    image = cv.cvtColor(image, convert)

    rr.log("screen", rr.Image(image))
    return cap


def rerun_log_webcam(
    cap: cv.VideoCapture,
    *,
    name: str | None = None,
    scale: float = 1.0,
    convert: int = cv.COLOR_BGR2RGB,
):
    if name is not None:
        time = cap.get(cv.CAP_PROP_POS_MSEC) / 1_000
        rr.set_time_seconds(f"{name}_time", time)

    success, image = cap.read()

    if not success:
        return cap.release()

    h, w, _ = image.shape
    image = cv.resize(image, (int(w * scale), int(h * scale)))
    image = cv.cvtColor(image, convert)
    rr.log("webcam", rr.Image(image))
    return cap


def rerun_log_user(entry: LogEntry, scale: float = 1.0, index: int | None = None):
    event_type = entry["event"]
    study_name = entry["study"]

    match event_type:
        case "start" | "stop":
            rr.log(
                "log/event",
                rr.TextLog(
                    f"recording {event_type} {study_name} ({index})",
                    level=rr.TextLogLevel.INFO,
                ),
            )

        case "click":
            rr.log(
                "log/event",
                rr.TextLog("mouse click", level=rr.TextLogLevel.DEBUG),
            )

        case "scroll":
            # rr.log(
            #     "log/event",
            #     rr.TextLog("mouse scroll", level=rr.TextLogLevel.TRACE),
            # )
            pass

        case "mouse":
            screen_x, screen_y = entry["screen_x"], entry["screen_y"]
            # rr.log(
            #     "log/event",
            #     rr.TextLog("mouse move", level=rr.TextLogLevel.TRACE),
            # )
            rr.log(
                "screen/mouse",
                rr.Points2D(
                    [[screen_x * scale, screen_y * scale]],
                    colors=[(255, 255, 0)],
                    radii=[1],
                ),
            )
