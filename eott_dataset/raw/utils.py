from os import PathLike


__all__ = [
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
