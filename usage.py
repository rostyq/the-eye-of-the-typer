# pyright: basic
# ruff: noqa

# %%
# %load_ext dotenv
# %dotenv
# %env EOTT_DATASET_PATH

# %%
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import ffmpeg as ff
import numpy as np
import polars as pl
import rerun as rr
import rerun.blueprint as rrb
from decord import VideoReader

from eott import *
from eott.rerun import *

# %%
pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_rows(25)
pl.Config.set_tbl_hide_column_data_types(True)


# %%
ds = DirDataset()
# %%
print_schema(ds.schema("dot"))

# %%
print_schema(ds.schema("log"))
# %%
rrdir = ds.root / "rerun"
rrdir.mkdir(exist_ok=True)

form = FormEntry(
    **next(
        ds.lazyframe("form", cache=False)
        .filter(pid=1)
        .collect()
        .iter_rows(named=True, buffer_size=1)
    )
)
# if form.screen_start:
#     print(form.screen_start, end=" -> ")
#     form.screen_start -= timedelta(milliseconds=378)
#     print(form.screen_start)

# form.webcam_start += timedelta(milliseconds=200)

bb = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial2DView(origin="webcam"),
            rrb.Spatial2DView(origin="screen", contents="+screen +mouse +tobii"),
        ),
        rrb.TextLogView(origin="event"),
    )
)

with rr.RecordingStream("EOTT", recording_id=f"{form.pid:02d}") as rrd:
    # rrd.save((rrdir / rrd.get_recording_id()).with_suffix(".rrd"))
    rrd.connect_grpc()
    rrd.send_blueprint(bb)

    screen_available = log_screen_video(form, ds)
    log_webcam_video(form, ds, screen=screen_available)

    llf = with_timelines(
        ds.lazyframe("log", cache=False), form, screen=screen_available
    )
    tlf = with_timelines(ds.lazyframe("tobii"), form, screen=screen_available)

    log_events(llf)
    log_tobii(tlf, form)

    rrd.flush()
