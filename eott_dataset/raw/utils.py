from os import PathLike


__all__ = [
    "get_dataset_root",
    "get_output_path",
    "mpv_play",
    "ffmpeg",
]


def ffmpeg(src: PathLike, dst: PathLike | str, **kwargs: str):
    from subprocess import check_output
    from itertools import chain

    args = chain.from_iterable([(f"-{key}", value) for key, value in kwargs.items()])
    args = ["ffmpeg", "-i", str(src), *args, str(dst)]
    return check_output(args, shell=True)


def mpv_play(src: PathLike):
    from subprocess import run

    return run(["mpv", "--osd-fractions", "--osd-level=2", src], shell=True)


# def _get_start_recording_time_offsets() -> dict[int, timedelta]:
#     raise NotImplementedError()

#     import polars as pl
#     from .data import read_participant_characteristics
#     from .participant import Participant

#     pdf = read_participant_characteristics()

#     ps = [Participant.from_dict(**row) for row in pdf.iter_rows(named=True)]
#     ps = [p for p in ps if p.screen_path.exists()]

#     schema = {"pid": pl.UInt8, "screen_time": pl.Datetime("us")}
#     df = pl.DataFrame(
#         [[p.pid, p.user_interaction_logs["epoch"][0]] for p in ps], schema
#     )
#     df = pdf.join(
#         df.with_columns(pl.col("screen_time").cast(pl.Datetime("ms"))), on="pid"
#     )
#     df = df.with_columns(screen_offset=pl.col("screen_time") - pl.col("start_time"))
#     df = df.select("pid", "screen_offset")

#     return {pid: offset for pid, offset in df.iter_rows()}
