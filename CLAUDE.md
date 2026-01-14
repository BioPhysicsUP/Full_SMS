# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active Work: UI Improvements

The improvement plan is tracked in `ui_improvements_plan.md`. When asked to continue:

1. Read `ui_improvements_plan.md`
2. Find the task marked `[NEXT]`
3. Plan the task, summarize the changes, and **ask the user for the go-ahead**
4. Complete that task
   - Change `[NEXT]` to `[DONE]` with today's date
   - Change the following `[TODO]` to `[NEXT]`
   - Update the Progress Summary table
5. **Ask the user to test the changes** and provide feedback
6. Update the plan document and commit your changes with a message referencing the task

### User Testing Protocol

After completing each task:
- Ask the user to run the application (`uv run python -m full_sms.app`)
- Request they test the specific functionality that was changed
- Wait for feedback before proceeding to the next task
- If feedback indicates issues, address them before marking the task complete

---

## Project Overview

Full SMS is a DearPyGui-based GUI application for single-molecule spectroscopy (SMS) data analysis, developed by the Biophysics Group at the University of Pretoria. It analyzes fluorescence measurements from HDF5 files, performing change point analysis, hierarchical clustering, lifetime fitting, and correlation functions.

Online docs: https://up-biophysics-sms.readthedocs.io/en/latest/index.html

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
│   ├── particle.py         # ParticleData, ChannelData, SpectraData, RasterScanData
│   ├── level.py            # LevelData (change point result)
│   ├── group.py            # GroupData, ClusteringResult, ClusteringStep
│   ├── fit.py              # FitResult (lifetime fitting result)
│   └── session.py          # SessionState, FileMetadata, UIState
├── io/                     # File I/O operations
│   ├── hdf5_reader.py      # Load HDF5 files → ParticleData
│   ├── session.py          # Save/load analysis sessions (JSON)
│   ├── exporters.py        # CSV/HDF5 data export
│   └── plot_exporters.py   # PNG/PDF plot export
├── ui/                     # User interface components
│   ├── layout.py           # MainLayout: 2-column layout manager
│   ├── theme.py            # Colors, fonts, DearPyGui theme
│   ├── keyboard.py         # Cross-platform keyboard shortcuts
│   ├── dialogs/            # File, fitting, settings dialogs
│   ├── views/              # Tab views (7 tabs)
│   ├── widgets/            # ParticleTree, StatusBar
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
│  Particle    │  Tab Bar (7 tabs)        │
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
[hdf5_reader.py] → ParticleData + ChannelData
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
| `ParticleData` | models/particle.py | Single molecule measurement data |
| `ChannelData` | models/particle.py | TCSPC channel (abstimes, microtimes) |
| `LevelData` | models/level.py | Brightness level from change point analysis |
| `ClusteringResult` | models/group.py | Hierarchical clustering with BIC optimization |
| `FitResult` | models/fit.py | Lifetime decay fit (1-3 exponentials) |
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
| `lifetime.py` | Multi-exponential IRF convolution | `fit_decay()` |
| `correlation.py` | Auto/cross-correlation g(2) | `calculate_g2()` |
| `histograms.py` | Photon binning utilities | `bin_photons()`, `build_decay_histogram()` |

### Settings Persistence

Platform-specific JSON storage via `config.py`:
- macOS: `~/Library/Application Support/Full SMS/settings.json`
- Windows: `%APPDATA%\Full SMS\settings.json`
- Linux: `~/.config/full_sms/settings.json`

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
├── conftest.py              # Pytest fixtures
├── test_analysis/           # Algorithm tests
├── test_models/             # Data model tests
├── test_io/                 # I/O tests
├── test_workers/            # Worker pool tests
└── test_ui/                 # UI tests
```

## Reference Materials

- `old/` - Original PyQt5 implementation (archived, do not modify)
- `rewrite_python_dearpygui_plan.md` - Completed rewrite plan (42 tasks)
- `rewrite_options/` - Architecture decision documents
