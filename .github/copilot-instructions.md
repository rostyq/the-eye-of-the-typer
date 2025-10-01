# The Eye of the Typer Dataset - AI Coding Instructions

## Project Overview
This is an **ETL (Extract-Transform-Load) framework** for processing "The Eye of the Typer" eye-tracking dataset. The project transforms raw experimental data from a ZIP archive into structured Parquet files and MP4 videos for analysis with Rerun visualization.

**Dataset source**: https://webgazer.cs.brown.edu/data/

## Architecture & Data Flow

### Core Pipeline (ETL Process)
The ETL pipeline is the heart of this project, transforming raw dataset archives into analysis-ready formats:

1. **Input**: ZIP archive containing raw experimental data (forms, logs, videos, eye-tracking data)
2. **Processing**: `eott.raw.extract_transform_load()` orchestrates all transformations
3. **Output**: Structured directory with Parquet files and processed videos

```
python -m eott etl <dataset.zip> --output-path <output-dir>
```

### Module Structure
- **`eott/__init__.py`**: Core data models (`FormEntry`, `Point2/3`, `Size`, `Offset`, `TrackingBox`, `DataType` enum)
- **`eott/raw.py`**: ETL implementation, data sources (`Log`, `Tobii`, `Calib`, `Dot`, `Webcam`), and parsing logic
- **`eott/rerun.py`**: Visualization helpers for Rerun SDK integration
- **`eott/__main__.py`**: CLI interface using Typer

### Key Abstractions

**Data Sources** (`eott/raw.py`):
- Each source (Tobii, Log, Calib, Dot) implements `scan_raw()` and `scan_zip()` methods
- Sources are `StrEnum` subclasses that also handle data transformation
- Pattern: `<Source>.scan_zip(zip_file)` → `LazyFrame` (lazy Polars DataFrame)

**Dataset Access**:
- `ZipDataset`: Reads raw data from ZIP archives (input phase)
- `DirDataset`: Accesses transformed data directory (analysis phase)
- Both inherit from `DatasetPaths` for consistent path management

## Python Environment & Dependencies

This project uses **uv** for dependency management and requires **Python ≥3.13**.

### Setup Commands
```powershell
# Install dependencies (use uv, not pip directly)
uv sync

# Run ETL pipeline
uv run python -m eott etl path/to/dataset.zip

# With specific data types
uv run python -m eott etl dataset.zip --process FORM --process WEBCAM --process SCREEN
```

### Critical Dependencies
- **polars**: All data processing uses lazy DataFrames (`LazyFrame`) for memory efficiency
- **rerun-sdk**: Visualization in notebooks (see `rerun.ipynb`)
- **ffmpeg-python**: Video processing wrapper (requires system `ffmpeg` installed)
- **decord**: Video frame extraction
- **typer**: CLI framework

## Data Processing Patterns

### Polars LazyFrame Pattern
All data transformations use **lazy evaluation** with Polars:

```python
# Typical pattern in raw.py
lf = scan_csv(source, schema=SCHEMA)
lf = lf.filter(col("event") == "start")
lf = lf.with_columns(timestamp=col("timestamp").cast(Datetime))
# Only executes when .collect() or .sink_parquet() is called
lf.sink_parquet(output_path)
```

Never call `.collect()` until absolutely necessary - maintain lazy evaluation for memory efficiency.

### Video Processing
Videos are processed using `ffmpeg-python`:
- Screen recordings: Downscaled 2x (`scale=iw/2:ih/2`) and re-encoded to MP4
- Webcam recordings: Converted from WebM to MP4, concatenated across gaps if `--concat` flag is used
- Blank videos are generated for temporal gaps between recordings

### Timestamp Synchronization
Critical aspect: Multiple timestamp sources need alignment:
- `screen_start`, `webcam_start`, `init_start` (from form data)
- Optional alignment CSV (`--alignment-path`) adjusts webcam timestamps
- `eott/rerun.py:with_timelines()` creates unified timelines: `rec_time`, `webcam_time`, `screen_time`

## Development Workflows

### Running ETL with Options
```powershell
# Dry run (see what would happen)
uv run python -m eott etl dataset.zip --dry-run

# Process only specific data types
uv run python -m eott etl dataset.zip --process FORM --process LOG

# Concatenate webcam videos
uv run python -m eott etl dataset.zip --concat

# Synchronize timestamps with alignment file
uv run python -m eott etl dataset.zip --alignment-path alignments.csv

# Force overwrite existing files
uv run python -m eott etl dataset.zip --overwrite
```

### Jupyter Notebook Usage
Notebooks (`rerun.ipynb`, `gazefilter.ipynb`) require `.env` file with:
```
EOTT_DATASET_PATH=path/to/transformed/dataset
```

Load environment in notebooks:
```python
%load_ext dotenv
%dotenv
```

Then access via:
```python
from eott import DirDataset
ds = DirDataset()  # Automatically loads from EOTT_DATASET_PATH
```

### Rerun Visualization
`rerun.ipynb` demonstrates the visualization workflow:
1. Load form entry for participant
2. Set up Rerun recording stream with blueprint
3. Log videos (screen, webcam) with `log_screen_video()`, `log_webcam_video()`
4. Log event timelines with `log_events()`, `log_tobii()`

```python
with rr.RecordingStream("EOTT", recording_id=f"{pid:02d}") as rrd:
    rrd.connect_grpc()  # Or rrd.save("file.rrd")
    log_screen_video(form, ds)
    log_webcam_video(form, ds)
    log_events(llf)
    log_tobii(tlf, form)
```

## Project-Specific Conventions

### Naming Patterns
- Participant IDs: `pid` (int), formatted as `P_{pid:02}` in paths
- Log/record numbers: `log`, `record` (int)
- Data types: Use `DataType` enum members (`DataType.FORM`, not strings)
- Studies: Use `Study` enum with snake_case names (e.g., `Study.DOT_TEST`)

### File Paths
The `DatasetPaths` mixin provides consistent path construction:
```python
ds.data_path(DataType.FORM)  # → root/form.parquet
ds.screen_path(pid=7)        # → root/screen/P_07.mp4
ds.webcam_path(pid=7)        # → root/webcam/P_07.mp4
ds.webcam_part_path(7, 3, Study.DOT_TEST)  # → root/webcam/P_07/03-dot_test.mp4
```

### Dataclass Usage
All data structures are frozen dataclasses with `slots=True, kw_only=True`:
```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Point2[T: float]:
    x: T
    y: T
```

`FormEntry` is mutable (`frozen=False`) and performs validation in `__post_init__`.

### StrEnum Pattern
Enumerations extend `StrEnum` for string serialization. The `NameEnum` base adds `id` property and `from_id()` classmethod for indexed access.

## Common Tasks

**Add new data source**:
1. Create enum class inheriting from `SourceEnumClass` and `StrEnum`
2. Implement `scan_raw()` method returning `LazyFrame`
3. Define Polars schema as class attribute
4. Add to `extract_transform_load()` processing flow

**Modify data schema**:
- Schemas are defined inline with `scan_csv()` calls in each source class
- Use Polars dtypes: `UInt64`, `Float64`, `Datetime("ms")`, `Struct`, etc.
- Update transformations in `with_columns()` chains

**Debug ETL issues**:
- Use `--dry-run` to see planned operations
- Check `print_schema()` output to verify transformed schemas
- Enable verbose output (already present) shows progress

## Anti-Patterns to Avoid
- ❌ Don't use pandas - this project is **Polars-only**
- ❌ Don't call `.collect()` early - maintain lazy evaluation
- ❌ Don't hardcode paths - use `DatasetPaths` methods
- ❌ Don't modify data without updating schemas
- ❌ Don't run `pip install` - use `uv` for all package management
