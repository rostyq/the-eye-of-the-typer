from datetime import timedelta as dt
from functools import cache
from contextlib import suppress


__all__ = ["REC_FIRSTTASK", "REC_ADJUST", "get_record_offset"]


REC_FIRSTTASK: dict[int, dt] = {
    1: dt(seconds=11, milliseconds=733),
    2: dt(seconds=3, milliseconds=350),
    6: dt(seconds=36, milliseconds=828),
    7: dt(seconds=5, milliseconds=150),
    8: dt(seconds=32, milliseconds=32),
    10: dt(minutes=2, seconds=34, milliseconds=29),
    12: dt(seconds=34, milliseconds=451),
    13: dt(minutes=1, seconds=3, milliseconds=689),
    16: dt(seconds=29, milliseconds=863),
    17: dt(seconds=21, milliseconds=939),
    18: dt(seconds=24, milliseconds=441),
    19: dt(seconds=25, milliseconds=275),
    20: dt(seconds=27, milliseconds=27),
    23: dt(seconds=24, milliseconds=733),
    25: dt(seconds=29, milliseconds=655),
    27: dt(seconds=25, milliseconds=651),
    31: dt(minutes=3, seconds=51, milliseconds=273),
    35: dt(seconds=28, milliseconds=362),
    40: dt(seconds=21, milliseconds=230),
    41: dt(seconds=20, milliseconds=395),
    42: dt(seconds=18, milliseconds=852),
    44: dt(seconds=23, milliseconds=398),
    45: dt(seconds=30, milliseconds=697),
    46: dt(minutes=1, seconds=1, milliseconds=645),
    55: dt(seconds=13, milliseconds=805),
    56: dt(seconds=22, milliseconds=856),
    59: dt(seconds=21, milliseconds=813),
}
REC_ADJUST: dict[int, dt] = {
    1: dt(seconds=0, milliseconds=0),
    2: dt(seconds=0, milliseconds=0),
    6: dt(seconds=0, milliseconds=0),
    7: dt(seconds=0, milliseconds=0),
    8: dt(seconds=0, milliseconds=0),
    10: dt(seconds=0, milliseconds=0),
    12: dt(seconds=0, milliseconds=0),
    13: dt(seconds=0, milliseconds=0),
    16: dt(seconds=0, milliseconds=0),
    17: dt(seconds=0, milliseconds=0),
    18: dt(seconds=0, milliseconds=0),
    19: dt(seconds=0, milliseconds=0),
    20: dt(seconds=0, milliseconds=0),
    23: dt(seconds=0, milliseconds=0),
    25: dt(seconds=0, milliseconds=0),
    27: dt(seconds=0, milliseconds=0),
    31: dt(seconds=0, milliseconds=0),
    35: dt(seconds=0, milliseconds=0),
    40: dt(seconds=0, milliseconds=0),
    41: dt(seconds=0, milliseconds=0),
    42: dt(seconds=0, milliseconds=0),
    44: dt(seconds=0, milliseconds=0),
    45: dt(seconds=0, milliseconds=0),
    46: dt(seconds=0, milliseconds=0),
    55: dt(seconds=0, milliseconds=0),
    56: dt(seconds=0, milliseconds=0),
    59: dt(seconds=0, milliseconds=0),
}
REC_STARTS: dict[int, dt] = {
    1: dt(seconds=340, microseconds=162000),
    2: dt(seconds=178, microseconds=817000),
    6: dt(seconds=244, microseconds=251000),
    7: dt(seconds=181, microseconds=772000),
    8: dt(seconds=246, microseconds=995000),
    10: dt(seconds=363, microseconds=631000),
    12: dt(seconds=318, microseconds=857000),
    13: dt(seconds=180, microseconds=595000),
    16: dt(seconds=138, microseconds=951000),
    17: dt(seconds=222, microseconds=18000),
    18: dt(seconds=144, microseconds=371000),
    19: dt(seconds=336, microseconds=429000),
    20: dt(seconds=181, microseconds=560000),
    23: dt(seconds=194, microseconds=254000),
    25: dt(seconds=279, microseconds=237000),
    27: dt(seconds=251, microseconds=102000),
    31: dt(seconds=330, microseconds=914000),
    35: dt(seconds=215, microseconds=130000),
    40: dt(seconds=412, microseconds=771000),
    41: dt(seconds=191, microseconds=220000),
    42: dt(seconds=223, microseconds=360000),
    44: dt(seconds=151, microseconds=130000),
    45: dt(seconds=213, microseconds=719000),
    46: dt(seconds=170, microseconds=982000),
    55: dt(seconds=103, microseconds=587000),
    56: dt(seconds=137, microseconds=607000),
    59: dt(seconds=192, microseconds=981000),
}


@cache
def get_record_offset(pid: int):
    """
    Get screen recording time offset by
    ```
    offset = start - (frame + adjust)
    ```
    where:
    - `frame` is a time of the first task frame in screen recording,
    - `start` timestamp of the first `start` event in user log,
    - `adjust` manual adjustment is found manually using `rerun`.
    """
    with suppress(KeyError):
        return REC_STARTS[pid] - (REC_FIRSTTASK[pid] + REC_ADJUST[pid])
