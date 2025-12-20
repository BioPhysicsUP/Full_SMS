# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full SMS is a PyQt5-based GUI application for single-molecule spectroscopy (SMS) data analysis, developed by the Biophysics Group at the University of Pretoria. It analyzes fluorescence measurements from HDF5 files, performing change point analysis, hierarchical clustering, lifetime fitting, and correlation functions.

## Build and Run Commands

```bash
# Install dependencies with pipenv
pipenv install

# Run the application
pipenv run python src/main.py

# Run in debug mode (single process, no freeze_support)
pipenv run python src/main.py --debug

# Build distributable with PyInstaller
pipenv run pyinstaller --noconfirm --clean build.spec
```

## Architecture

### Core Data Flow

1. **File Loading** (`smsh5.py`, `smsh5_file_reader.py`): HDF5 files are loaded via `H5dataset` class which creates `Particle` objects for each measurement
2. **Change Point Analysis** (`change_point.py`): `ChangePoints` class detects intensity changes and creates `Level` objects representing brightness states
3. **Hierarchical Clustering** (`grouping.py`): `AHCA` class groups similar levels using agglomerative clustering with BIC optimization
4. **Lifetime Fitting** (`tcspcfit.py`): `FluoFit` class performs fluorescence decay fitting with IRF convolution

### GUI Architecture

- **MainWindow** (`main.py`): Central hub coordinating all functionality
- **Controllers** (`controllers.py`): Domain-specific controllers (IntController, LifetimeController, GroupingController, etc.) manage their respective UI tabs and analysis workflows
- **Threading** (`threads.py`, `processes.py`): `ProcessThread` wraps multiprocessing for CPU-intensive analysis tasks
- **Signals** (`signals.py`): PyQt signals for thread-safe UI updates

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `H5dataset` | smsh5.py | Container for entire HDF5 measurement file |
| `Particle` | smsh5.py | Single molecule measurement with intensity/lifetime data |
| `ChangePoints` | change_point.py | Change point detection and level creation |
| `Level` | change_point.py | Single brightness level with timing and photon data |
| `AHCA` | grouping.py | Agglomerative hierarchical clustering algorithm |
| `Group` | grouping.py | Collection of levels grouped by similarity |
| `FluoFit` | tcspcfit.py | Fluorescence decay fitting engine |

### File Organization

- `src/resources/ui/`: Qt Designer `.ui` files defining GUI layouts
- `src/resources/icons/`: Application icons
- `src/resources/data/`: Runtime data files (e.g., precomputed sums)
- `file_manager.py`: Path resolution utility supporting both development and PyInstaller bundled execution

### Threading Model

The application uses `QThreadPool` with custom `ProcessThread` workers that spawn Python `multiprocessing.Process` for CPU-bound analysis. Signal queues pass progress updates back to the main GUI thread.

## Key Patterns

- UI files are loaded dynamically via `uic.loadUiType()` rather than pre-compiled
- `file_manager.path()` resolves resource paths for both dev and frozen (PyInstaller) execution
- Analysis results are stored on `Particle` objects (e.g., `particle.cpts.levels`, `particle.ahca.groups`)
- Export uses both matplotlib (for publication figures) and pyqtgraph exporters

## Documentation

Online docs: https://up-biophysics-sms.readthedocs.io/en/latest/index.html

Build docs locally:
```bash
cd docs
make html
```
