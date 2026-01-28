# Full SMS

Single-molecule spectroscopy data analysis application developed by the Biophysics Group at the University of Pretoria.

Full SMS analyzes fluorescence measurements from HDF5 files, performing change point analysis, hierarchical clustering, lifetime fitting, and correlation functions.

**Online Documentation:** https://up-biophysics-sms.readthedocs.io/en/latest/index.html

## Features

- Load and visualize fluorescence intensity traces from HDF5 files
- Detect intensity change points (brightness states) using Watkins & Yang algorithm
- Hierarchically cluster similar brightness levels into groups (AHCA with BIC optimization)
- Fit fluorescence decay lifetimes (1-3 exponential models with IRF convolution)
- Perform second-order photon correlation (g2 antibunching analysis)
- Export results to CSV, HDF5, PNG, and PDF formats
- Save and restore analysis sessions

## Installation

This project uses [mise](https://mise.jdx.dev/) for tool management and [uv](https://docs.astral.sh/uv/) for Python package management.

```bash
# Install mise (if not already installed)
curl https://mise.run | sh

# Initialize the project (installs Python 3.14 and dependencies)
mise trust && mise install
uv sync
```

## Usage

```bash
# Run the application
uv run python -m full_sms.app

# Run tests
uv run pytest

# Build distributable
uv run pyinstaller build.spec
```

## Manual Setup

If you prefer not to use mise:

```bash
# Ensure Python 3.14 is installed
python --version

# Install uv
pip install uv

# Install dependencies
uv sync

# Run the application
uv run python -m full_sms.app
```

## Technology Stack

- **GUI Framework:** DearPyGui with ImPlot
- **Scientific Computing:** NumPy, SciPy, Numba (JIT acceleration)
- **File I/O:** h5py (HDF5), matplotlib (plot export)
- **Build:** PyInstaller for cross-platform distribution
