# Full SMS

Single-molecule spectroscopy (SMS) data analysis application for time-resolved fluorescence measurements. Built with DearPyGui by the Biophysics Group at the University of Pretoria.

**Online docs:** https://up-biophysics-sms.readthedocs.io/en/latest/index.html

## Publication

Botha, J.L., van Heerden, B., and Krüger, T.P.J. — *Advanced analysis of single-molecule spectroscopic data*, Biophysical Reports 4(3), September 2024.  
https://doi.org/10.1016/j.bpr.2024.100173

---

## Features

Full SMS processes HDF5 files from TCSPC (Time-Correlated Single Photon Counting) acquisition systems and exposes the full analysis pipeline through a 7-tab GUI.

| Tab | Analysis |
|-----|----------|
| **Intensity** | Binned photon count traces with change-point level overlay |
| **Grouping** | AHCA hierarchical clustering with BIC optimization |
| **Lifetime** | Multi-exponential decay fitting with IRF convolution |
| **Correlation** | Second-order photon correlation g(2), antibunching |
| **Spectra** | Wavelength-resolved spectral evolution heatmap |
| **Raster** | 2D spatial intensity scan visualization |
| **Export** | Batch data and plot export |

**Core algorithms:**
- **Change point analysis** — Watkins & Yang weighted likelihood ratio for detecting brightness state transitions; confidence levels: 69%, 90%, 95%, 99%
- **Clustering** — Agglomerative hierarchical clustering algorithm (AHCA) with BIC optimization
- **Lifetime fitting** — 1–3 exponential model fitting with IRF convolution (simulated Gaussian or loaded from file)
- **Correlation** — Auto/cross-correlation g(2) with configurable windows and timing offset correction

---

## Requirements

- Python 3.14+
- [mise](https://mise.jdx.dev/) (recommended) or manual `uv` setup

---

## Installation

### With mise (recommended)

```bash
git clone <repo-url>
cd Full_SMS
mise trust && mise install
uv sync
```

### Without mise

```bash
# Ensure Python 3.14+ is active
pip install uv
uv sync
```

---

## Usage

```bash
# Run the application
uv run python -m full_sms.app

# Run tests
uv run pytest

# Build standalone executable
uv run pyinstaller build.spec
```

**Distributable output:**
- macOS: `dist/Full_SMS.app`
- Windows: `dist/Full_SMS/Full_SMS.exe`
- Linux: `dist/Full_SMS/Full_SMS`

---

## File Formats

**Input:** HDF5 (`.h5`) files from UP Biophysics SMS acquisition software. Supports multi-measurement files, dual TCSPC channels, raster scans, and spectral data.

**Export:**
- Data: CSV, Parquet, Excel (`.xlsx`), JSON
- Plots: PNG, PDF, SVG (publication-quality via matplotlib)
- Sessions: `.smsa` (JSON snapshot of complete analysis state for reproducibility)

---

## Platform Support

| Platform | Backend | Config location |
|----------|---------|----------------|
| macOS | Metal | `~/Library/Application Support/Full SMS/settings.json` |
| Windows | DirectX 11 | `%APPDATA%\Full SMS\settings.json` |
| Linux | OpenGL | `~/.config/full_sms/settings.json` |

Retina/high-DPI displays supported on all platforms. Keyboard shortcuts use Cmd on macOS, Ctrl on Windows/Linux.

---

## Architecture

```
HDF5 File
    ↓
[hdf5_reader] → MeasurementData
    ↓
    ├─ [change_point]  → LevelData[]          (worker process)
    │       ↓
    │   [clustering]   → ClusteringResult     (worker process)
    │       ↓
    │   [lifetime]     → FitResult            (worker process)
    │       ↓
    │   [correlation]  → CorrelationResult    (worker process)
    │
    └─ UI renders based on selection
    ↓
[session]    → SessionState  (.smsa)
[exporters]  → CSV / PNG / PDF / ...
```

Long-running analyses run in a `ProcessPoolExecutor` worker pool (spawn context, `cpu_count - 1` workers). The DearPyGui event loop polls pending futures each frame and updates `SessionState` on completion. Performance-critical code in `analysis/` is JIT-compiled via Numba.

---

## Development

```bash
# Run all 577 tests
uv run pytest

# With coverage report
uv run pytest --cov=full_sms --cov-report=html

# Specific module
uv run pytest tests/test_analysis/
```

---

## License

MIT
