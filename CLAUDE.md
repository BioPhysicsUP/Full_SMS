# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full SMS is a DearPyGui-based GUI application for single-molecule spectroscopy (SMS) data analysis, developed by the Biophysics Group at the University of Pretoria. It analyzes fluorescence measurements from HDF5 files, performing change point analysis, hierarchical clustering, lifetime fitting, and correlation functions.

**Online docs:** https://up-biophysics-sms.readthedocs.io/en/latest/index.html

**Status:** The application is fully functional. The PyQt5-to-DearPyGui rewrite (42 tasks) and UI improvements (25 tasks) have been completed.

## Tooling & Commands

- **mise** - Tool version management (`.mise.toml`)
- **uv** - Python package management (`pyproject.toml`, `uv.lock`)
- **Python 3.14** - Required version

```bash
# Install dependencies
uv sync

# Run the application
uv run python -m full_sms.app

# Run tests
uv run pytest

# Build distributable
uv run pyinstaller build.spec
```

## Directory Structure

```
src/full_sms/
├── app.py                  # Main application entry point & orchestrator
├── config.py               # Settings management & persistence
├── analysis/               # Scientific algorithms (Numba-JIT accelerated)
│   ├── change_point.py     # Watkins & Yang change point detection
│   ├── clustering.py       # AHCA hierarchical clustering
│   ├── lifetime.py         # Fluorescence decay fitting
│   ├── correlation.py      # Auto/cross-correlation (g2)
│   └── histograms.py       # Binning utilities
├── models/                 # Immutable data models (frozen dataclasses)
│   ├── measurement.py      # MeasurementData, ChannelData, SpectraData, RasterScanData
│   ├── level.py            # LevelData (change point result)
│   ├── group.py            # GroupData, ClusteringResult, ClusteringStep
│   ├── fit.py              # FitResult, FitResultData, IRFData
│   └── session.py          # SessionState, FileMetadata, UIState
├── io/                     # File I/O operations
│   ├── hdf5_reader.py      # Load HDF5 files → MeasurementData
│   ├── session.py          # Save/load analysis sessions (JSON)
│   ├── exporters.py        # CSV/HDF5 data export
│   └── plot_exporters.py   # PNG/PDF plot export
├── ui/                     # User interface components
│   ├── layout.py           # MainLayout: 2-column layout manager
│   ├── theme.py            # Colors, fonts, DearPyGui theme
│   ├── keyboard.py         # Cross-platform keyboard shortcuts
│   ├── dialogs/            # File, fitting, settings dialogs
│   ├── views/              # Tab views (7 tabs)
│   ├── widgets/            # MeasurementTree, StatusBar
│   └── plots/              # Plot implementations
├── workers/                # Parallel processing
│   ├── pool.py             # AnalysisPool (ProcessPoolExecutor)
│   └── tasks.py            # Picklable task functions
├── utils/
│   └── platform.py         # Cross-platform utilities (DPI, GPU, shortcuts)
└── resources/
    ├── icons/              # Application icons
    └── data/tau_data/      # Precomputed threshold tables
```

## Architecture

### UI Layout

```
┌─────────────────────────────────────────┐
│  Menu Bar (File, Edit, View, Help)      │
├──────────────┬──────────────────────────┤
│  Measurement │  Tab Bar (7 tabs)        │
│  Tree        │  ┌────────────────────┐  │
│  Sidebar     │  │ Content Area       │  │
│  (180px)     │  │ (Plot + Controls)  │  │
├──────────────┴──────────────────────────┤
│  Status Bar (progress + file info)      │
└─────────────────────────────────────────┘
```

**7 Tab Views**: Intensity, Lifetime, Grouping, Correlation, Spectra, Raster, Export

### Core Data Flow

```
HDF5 File
    ↓
[hdf5_reader.py] → MeasurementData + ChannelData
    ↓
    ├→ [change_point.py] → LevelData[] (async worker)
    │   ↓
    │   [clustering.py] → ClusteringResult (async worker)
    │   ↓
    │   [lifetime.py] → FitResult (async worker)
    │   ↓
    │   [correlation.py] → CorrelationResult (async worker)
    │
    └→ UI renders plots based on selection
    ↓
[session.py] → SessionState (JSON)
    ↓
[exporters.py] → CSV/HDF5/PNG/PDF
```

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `Application` | app.py | Main orchestrator, event handling |
| `MeasurementData` | models/measurement.py | Single molecule measurement data |
| `ChannelData` | models/measurement.py | TCSPC channel (abstimes, microtimes) |
| `LevelData` | models/level.py | Brightness level from change point analysis |
| `ClusteringResult` | models/group.py | Hierarchical clustering with BIC optimization |
| `FitResult` | models/fit.py | Lifetime decay fit (1-3 exponentials) |
| `FitResultData` | models/fit.py | Serializable fit parameters for persistence |
| `IRFData` | models/fit.py | Instrument response function (simulated or loaded) |
| `SessionState` | models/session.py | Complete application state |
| `AnalysisPool` | workers/pool.py | ProcessPoolExecutor wrapper |
| `MainLayout` | ui/layout.py | 2-column layout manager |

### Threading Model

- **GUI Thread**: Single-threaded DearPyGui event loop
- **Worker Pool**: `ProcessPoolExecutor` with spawn context
  - Isolated processes (no GIL contention)
  - Cross-platform compatible (macOS/Windows/Linux)
  - `max_workers = cpu_count - 1`

**Task Flow**:
1. User triggers analysis from tab
2. `pool.submit(task_func, params)` → `Future[TaskResult]`
3. Main loop polls `_pending_futures`
4. On completion: update `SessionState` → trigger UI redraw

### Analysis Algorithms

| Module | Algorithm | Key Function |
|--------|-----------|--------------|
| `change_point.py` | Watkins & Yang weighted likelihood | `find_change_points()` |
| `clustering.py` | AHCA with BIC optimization | `cluster_levels()` |
| `lifetime.py` | Multi-exponential IRF convolution | `fit_decay()`, `compute_convolved_fit_curve()` |
| `correlation.py` | Auto/cross-correlation g(2) | `calculate_g2()` |
| `histograms.py` | Photon binning utilities | `bin_photons()`, `build_decay_histogram()` |

### Settings Persistence

Platform-specific JSON storage via `config.py`:
- macOS: `~/Library/Application Support/Full SMS/settings.json`
- Windows: `%APPDATA%\Full SMS\settings.json`
- Linux: `~/.config/full_sms/settings.json`

### Session Save/Load

Sessions are saved as `.smsa` files (JSON format) containing:
- File metadata and measurement data references
- All analysis results (levels, groups, fits, correlations)
- UI state (selections, tab, zoom levels)
- Export settings and IRF data

## Key Dependencies

| Package | Purpose |
|---------|---------|
| dearpygui | GUI framework |
| numpy | Numerical arrays |
| scipy | Scientific algorithms |
| h5py | HDF5 file I/O |
| numba | JIT compilation for algorithms |
| matplotlib | Plot export (PNG, PDF) |

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures (make_clustering_result helper)
├── test_analysis/           # Algorithm tests (change point, clustering, lifetime, correlation, histograms)
├── test_models/             # Data model tests (measurement, level, group, fit, session)
├── test_io/                 # I/O tests
│   ├── test_hdf5_reader.py  # HDF5 file loading
│   ├── test_session.py      # Session save/load round-trip
│   ├── test_exporters.py    # Data export (CSV, Parquet, Excel, JSON)
│   └── test_plot_exporters.py # Plot export (PNG, PDF, SVG)
├── test_workers/            # Worker pool tests
│   ├── test_pool.py         # AnalysisPool (submit, map, error handling, concurrency)
│   └── test_tasks.py        # Picklable task functions
├── test_ui/                 # UI tests (decay_plot, lifetime_tab)
├── test_config.py           # Settings persistence
└── test_utils/              # Platform utilities
```

**Test count:** 577 tests

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_io/test_exporters.py

# Run specific test class
uv run pytest tests/test_workers/test_pool.py::TestAnalysisPoolConcurrency

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=full_sms --cov-report=html
```

### Test Coverage Notes

**Well-tested modules:**
- `analysis/` - Excellent coverage with synthetic data generation
- `models/` - Complete coverage including immutability validation
- `io/session.py` - Excellent round-trip testing
- `io/exporters.py` - All formats (CSV, Parquet, Excel, JSON)
- `io/plot_exporters.py` - All plot types and formats
- `workers/pool.py` - Error handling, concurrency, memory isolation

**Optional dependencies for tests:**
- `pyarrow` - Required for Parquet export tests
- `openpyxl` - Required for Excel export tests
- Tests skip gracefully if these are not installed

## Reference Materials

- `old/` - Original PyQt5 implementation (archived, do not modify)
- `old/full_context.md` - Documentation of the old PyQt5 architecture
- `rewrite_python_dearpygui_plan.md` - Completed rewrite plan (42 tasks)
- `ui_improvements_plan.md` - Completed UI improvements plan (25 tasks)
- `rewrite_options/` - Architecture decision documents
