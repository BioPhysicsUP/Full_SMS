# Full SMS Application - Complete Context Document

This document provides comprehensive documentation of the Full SMS (Single-Molecule Spectroscopy) application for the purpose of informing a future rewrite with a modern web-based stack.

---

## Table of Contents

1. [Overview](#overview)
2. [HDF5 File Format (The Data Contract)](#hdf5-file-format-the-data-contract)
3. [Data Model](#data-model)
4. [User Journey](#user-journey)
5. [Analysis Pipeline](#analysis-pipeline)
6. [Controller Layer](#controller-layer)
7. [Threading and Multiprocessing Architecture](#threading-and-multiprocessing-architecture)
8. [Settings and Configuration](#settings-and-configuration)
9. [Save/Load System](#saveload-system)
10. [Export System](#export-system)
11. [UI Structure](#ui-structure)
12. [Key Algorithms Summary](#key-algorithms-summary)
13. [Dependencies](#dependencies)

---

## Overview

Full SMS is a desktop application for analyzing single-molecule fluorescence spectroscopy data. It was developed by the Biophysics Group at the University of Pretoria. The application processes time-correlated single-photon counting (TCSPC) data stored in HDF5 files.

**Core Capabilities:**
- Load and visualize fluorescence intensity traces
- Detect intensity change points (brightness states)
- Hierarchically cluster similar brightness levels into groups
- Fit fluorescence decay lifetimes
- Perform second-order photon correlation (antibunching)
- Export results in various formats

---

## HDF5 File Format (The Data Contract)

The HDF5 file format is the primary data input and the one constant that will persist across any rewrite. Understanding this structure is critical.

### File Structure

```
root/
├── file.attrs['date'] (str)
├── file.attrs['measuring time (h)'] (float)
├── file.attrs['t_min (s)'] (float)
├── file.attrs['t_unit'] (str)
├── file.attrs['bin_size (ms)'] (float)
├── file.attrs['has_spectra'] (bool)
├── file.attrs['has_raster_scan'] (bool)
├── file.attrs['has_irf'] (bool)
├── file.attrs['num_tcspc_channel'] (int) - usually 1 or 2
│
├── IRF/ (optional)
│   ├── decay (ndarray)
│   └── t (ndarray) - time axis
│
├── Spectra/ (optional)
│   ├── wavelengths (ndarray)
│   └── spectra (ndarray)
│
├── Raster Scan/ (optional)
│   └── image (ndarray)
│
└── Particle #/ (one per measurement, numbered from 1)
    ├── particle.attrs['Particle #']
    ├── particle.attrs['Description']
    ├── particle.attrs['tcspc_card'] (str)
    ├── abstimes (ndarray) - absolute photon arrival times in nanoseconds
    ├── microtimes (ndarray) - micro times (within laser pulse cycle)
    └── [secondary channel data if num_tcspc_channel > 1]
```

### Key Data Arrays Per Particle

| Array | Type | Description |
|-------|------|-------------|
| `abstimes` | int64 ndarray | Absolute arrival time of each photon in nanoseconds |
| `microtimes` | float64 ndarray | Time within the laser pulse cycle (for lifetime analysis) |

### Derived Data (calculated, not stored in HDF5)

From `abstimes`:
- **Intensity trace**: Binned photon counts over time (histogram of abstimes)
- **Total measurement time**: `abstimes[-1] - abstimes[0]`

From `microtimes`:
- **Decay histogram**: Distribution of microtimes (for lifetime fitting)

### Time Units

- `abstimes`: nanoseconds (ns), stored as integers
- `microtimes`: nanoseconds (ns), stored as floats
- `bin_size`: milliseconds (ms) for intensity binning
- `t_min`: seconds (s), time resolution
- `channelwidth`: nanoseconds per TCSPC channel (derived from t_min)

---

## Data Model

### Class Hierarchy

```
H5dataset
├── particles: List[Particle]
├── global_particle: GlobalParticle (for cross-particle analysis)
├── irf: ndarray (Instrument Response Function)
├── irf_t: ndarray (IRF time axis)
├── spectra: Spectra
└── raster_scan: RasterScan

Particle
├── abstimes: ndarray (raw photon arrival times)
├── microtimes: Microtimes
├── histogram: Histogram (decay histogram)
├── cpts: ChangePoints (change point analysis results)
├── ahca: AHCA (hierarchical clustering results)
├── ab_analysis: AntibunchingAnalysis
├── sec_part: Particle (secondary TCSPC channel, if present)
└── Various computed properties

ChangePoints
├── levels: List[Level]
├── num_levels: int
└── Analysis metadata

Level
├── times_s/times_ns: time boundaries
├── int_p_s: intensity (photons per second)
├── num_photons: photon count
├── dwell_time_s: duration
├── microtimes: Microtimes (subset for this level)
├── histogram: Histogram
└── Lifetime fit results

AHCA (Agglomerative Hierarchical Clustering Algorithm)
├── steps: List[AHCAaliStep] (clustering history)
├── best_step: AHCAliStep (optimal clustering)
├── groups: List[Group]
└── BIC optimization data

Group
├── lvls: List[Level] (member levels)
├── int_p_s: average intensity
├── dwell_time_s: total dwell time
├── histogram: Histogram (combined)
└── Lifetime fit results

GlobalParticle (for multi-particle analysis)
├── levels: List[GlobalLevel]
├── ahca: AHCA
└── Aggregated analysis across particles
```

### Histogram Class

```python
class Histogram:
    decay: ndarray      # Photon counts per time bin
    t: ndarray          # Time axis (ns)
    fitted: bool        # Whether lifetime has been fit
    fit: FluoFit        # Fit results (tau, amp, chi-squared, etc.)
    avtau: float        # Amplitude-weighted average lifetime
```

### Microtimes Class

```python
class Microtimes:
    all: ndarray        # All microtime values
    indices: ndarray    # Indices into particle's microtimes array

    # Methods for slicing by time range
    def sub_microtimes(start_ind, end_ind) -> ndarray
```

---

## User Journey

### 1. File Loading

**Action:** User opens an HDF5 file (`.h5`)

**Process:**
1. `MainWindow.load_data()` triggered
2. `WorkerOpenFile` spawns in thread pool
3. `H5dataset` created, wrapping h5pickle file handle
4. For each particle in file:
   - `Particle` object created with lazy-loaded data
   - Tree node added to UI particle list
5. Default bin size applied, intensity traces computed
6. First particle displayed in intensity plot

**Key Files:** `main.py:load_data()`, `smsh5.py:H5dataset`, `smsh5_file_reader.py`

### 2. Intensity Trace Visualization

**What the user sees:** Photon count vs. time plot

**Process:**
1. `abstimes` binned into histogram with configurable bin size
2. `Particle.intdata` property returns binned counts
3. Plotted via pyqtgraph in `pgIntTrace` widget
4. User can adjust bin size (slider), zoom, pan

**Key properties:**
- `Particle.intdata` - binned intensity array
- `Particle.intt` - time axis for intensity
- `Particle.bin_size` - current bin size in ms

### 3. Change Point Analysis (Level Detection)

**Purpose:** Detect discrete intensity states ("levels") in the trace

**Action:** User clicks "Resolve" button (current/selected/all particles)

**Process:**
1. `IntController.resolve_clicked()` triggered
2. `WorkerResolveLevels` or `ProcessThread` spawned
3. For each particle:
   - `ChangePoints.run_cpa()` executes
   - Uses modified version of Watkins & Yang algorithm
   - Confidence level configurable (typically 95%)
   - Recursively splits trace at significant change points
4. Results stored in `particle.cpts.levels`
5. Levels drawn as colored rectangles on intensity plot

**Algorithm inputs:**
- `abstimes` (photon arrival times)
- `confidence` (statistical threshold, 0-100%)
- `min_photons` (minimum photons per level)
- `min_boundary_offset` (minimum change point separation)

**Algorithm outputs (per Level):**
- Start/end times
- Intensity (photons/second)
- Photon count
- Microtimes subset
- Decay histogram

### 4. Hierarchical Clustering (Grouping)

**Purpose:** Group similar brightness levels together based on intensity and optionally lifetime

**Action:** User clicks "Group" button

**Process:**
1. `GroupingController.group_clicked()` triggered
2. `WorkerGrouping` or `ProcessThread` spawned
3. For each particle with levels:
   - `AHCA.run_grouping()` executes
   - Agglomerative clustering with BIC optimization
   - Merges levels pairwise based on similarity
4. Optimal number of groups determined by BIC minimum
5. Groups stored in `particle.ahca.groups`

**Grouping modes:**
- **Per-particle**: Each particle grouped independently
- **Global**: All particles' levels grouped together

**Algorithm:**
1. Start with each level as its own cluster
2. Compute distance matrix (based on intensity, optionally lifetime)
3. Iteratively merge closest clusters
4. At each step, compute BIC (Bayesian Information Criterion)
5. Select step with minimum BIC as optimal grouping

### 5. Lifetime Fitting

**Purpose:** Extract fluorescence decay lifetimes from photon timing data

**Action:** User opens Fitting Dialog, configures parameters, clicks Fit

**Process:**
1. `LifetimeController.fit_lifetimes_clicked()` triggered
2. `WorkerFitLifetimes` or `ProcessThread` spawned
3. For each entity (particle/level/group):
   - Build decay histogram from microtimes
   - Load IRF (Instrument Response Function)
   - `FluoFit` (OneExp/TwoExp/ThreeExp) performs fit
   - Uses scipy.optimize.curve_fit or custom ML fitting
4. Results stored on histogram object

**Fitting models:**
- `OneExp`: Single exponential decay
- `TwoExp`: Bi-exponential decay
- `ThreeExp`: Tri-exponential decay

**Fitting parameters (from FittingDialog):**
- Number of exponential components
- Initial tau/amplitude guesses
- Fit boundaries (start/end time)
- IRF shift parameter
- Background estimation settings

**Fit outputs:**
- `tau`: Lifetime(s) in nanoseconds
- `amp`: Amplitude(s)
- `avtau`: Amplitude-weighted average lifetime
- `chisq`: Chi-squared goodness of fit
- `residuals`: Fit residuals
- `dw`: Durbin-Watson parameter

### 6. Antibunching / Correlation

**Purpose:** Calculate second-order photon correlation (requires dual TCSPC channels)

**Action:** User configures correlation parameters, clicks Correlate

**Process:**
1. `AntibunchingController` handles UI
2. `ProcessThread` with `correlate_particle` task
3. For particles with secondary channel:
   - Merge abstimes from both channels
   - Count coincidences within time window
   - Build correlation histogram
4. Results displayed in correlation plot

**Parameters:**
- `difftime`: Time offset between channels
- `window`: Correlation window (ns)
- `binsize`: Histogram bin size (ns)

### 7. Export

**Purpose:** Save analysis results in various formats

**Process:**
1. User selects export options in Export tab
2. `ExportWorker` runs in thread
3. `Exporter` class handles format-specific output

**Export formats:**
- **Data:** CSV, Parquet, Feather, Excel, HDF5
- **Plots:** PNG/PDF via matplotlib
- **DataFrames:** Pandas DataFrames with level/group data

**Exportable data:**
- Intensity traces
- Level information (times, intensities, lifetimes)
- Group information
- Decay histograms
- Fit results
- Correlation histograms
- Spectra and raster scan images

### 8. Save/Load Analysis

**Purpose:** Persist analysis state for later continuation

**Save format:** `.smsa` file (pickled `H5dataset` object)

**What's saved:**
- All particle analysis results (levels, groups, fits)
- UI state (selected particles, view settings)
- Application settings
- Filter settings

**Load process:**
1. Unpickle `.smsa` file
2. Reconnect to original `.h5` file (must be present)
3. Restore UI state and selections

---

## Analysis Pipeline

### Pipeline Flow

```
HDF5 File
    │
    ▼
┌─────────────────┐
│   Load Data     │ → H5dataset, Particle objects created
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Bin Data      │ → Intensity traces computed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Change Point   │ → Levels detected (ChangePoints, Level objects)
│    Analysis     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Grouping      │ → Levels clustered (AHCA, Group objects)
│    (AHCA)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Lifetime      │ → Decay fits performed (FluoFit results)
│    Fitting      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Correlation   │ → Antibunching analysis (optional)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Export      │ → Results saved to files
└─────────────────┘
```

### Dependencies Between Stages

| Stage | Requires |
|-------|----------|
| Bin Data | Loaded particle |
| Change Point Analysis | Binned data |
| Grouping | Levels (from CPA) |
| Lifetime Fitting | Histogram (particle, level, or group) + IRF |
| Correlation | Dual TCSPC channels |

---

## Controller Layer

The application uses a controller pattern to separate UI logic from core analysis. Each major feature area has its own controller.

### IntController (Intensity Analysis)

**Responsibility:** Change point analysis and level visualization

**Key methods:**
- `resolve_clicked()` - Trigger CPA for current/selected/all
- `resolve_levels()` - Execute CPA on particle(s)
- `plot_levels()` - Draw level rectangles on intensity plot
- `update_level_plot_items()` - Refresh level display

**Signals handled:**
- `resolve_finished` - CPA complete
- `level_resolved` - Single level done

### LifetimeController

**Responsibility:** Decay fitting and lifetime visualization

**Key methods:**
- `fit_lifetimes_clicked()` - Trigger fitting
- `fit_lifetimes()` - Execute fits
- `apply_all_fits()` - Batch apply fit to multiple entities
- `update_lifetime_plot()` - Refresh decay display

**Owns:**
- `FittingDialog` - Parameter configuration UI
- `FittingParameters` - Current fit parameters

### GroupingController

**Responsibility:** Hierarchical clustering and group visualization

**Key methods:**
- `group_clicked()` - Trigger grouping
- `run_grouping()` - Execute AHCA
- `plot_groups()` - Visualize group boundaries
- `update_bic_plot()` - Show BIC optimization curve

**Grouping modes:**
- Per-particle
- Global (across all selected particles)

### SpectraController

**Responsibility:** Spectral data visualization (if present in file)

### RasterScanController

**Responsibility:** 2D scan image visualization (if present in file)

### AntibunchingController

**Responsibility:** Photon correlation analysis

**Key methods:**
- `correlate_clicked()` - Trigger correlation
- `update_corr_plot()` - Display correlation histogram

### FilteringController

**Responsibility:** Filter particles based on criteria

**Filter criteria:**
- Intensity thresholds
- Number of levels
- Lifetime values
- Chi-squared values

---

## Threading and Multiprocessing Architecture

The application uses a hybrid threading model to keep the UI responsive during long-running analysis operations.

### Architecture Overview

```
Main Thread (UI)
    │
    ▼
QThreadPool
    │
    ├── QRunnable Workers (light tasks)
    │   ├── WorkerOpenFile
    │   ├── WorkerBinAll
    │   ├── SaveAnalysisWorker
    │   └── LoadAnalysisWorker
    │
    └── ProcessThread (heavy tasks)
            │
            ├── Spawns multiprocessing.Process workers
            │   └── SingleProcess (N instances, one per CPU core)
            │
            ├── Task Queue (input)
            ├── Result Queue (output)
            └── Feedback Queue (progress updates)
```

### ProcessThread

The main mechanism for CPU-intensive parallel work.

**Components:**
- `ProcessThread(QRunnable)`: Orchestrates multiprocessing
- `SingleProcess(mp.Process)`: Worker process that executes tasks
- `ProcessTask`: Encapsulates a method call to execute
- `ProcessTaskResult`: Returned result with updated object

**Flow:**
1. Main thread creates `ProcessThread` with list of `ProcessTask` objects
2. `ProcessThread.run()` spawns N `SingleProcess` workers
3. Tasks distributed via `task_queue`
4. Workers execute tasks, return results via `result_queue`
5. Progress updates sent via `feedback_queue`
6. Results emitted to main thread via Qt signals

### Signal System

**WorkerSignals:** Communication from workers to main thread
- `progress`, `start_progress`, `end_progress`: Progress bar updates
- `status_message`: Status bar updates
- `error`: Exception propagation
- `resolve_finished`, `fitting_finished`, `grouping_finished`: Task completion

**ProcessThreadSignals:** ProcessThread to main thread
- `results`: Task results
- `finished`: All tasks complete

### Task Types

| Task | Mechanism | Parallelism |
|------|-----------|-------------|
| Open File | QRunnable | Single |
| Bin All | QRunnable | Single |
| Resolve Levels | ProcessThread | Per-particle |
| Grouping | ProcessThread | Per-particle |
| Lifetime Fitting | ProcessThread | Per-histogram |
| Correlation | ProcessThread | Per-particle |
| Save/Load | QRunnable | Single |
| Export | QRunnable | Single |

---

## Settings and Configuration

### Settings Storage

**Location:** `settings.json` in project root

### Settings Categories

**Change Point Analysis:**
```json
{
  "min_num_photons": 20,      // Minimum photons per level
  "min_boundary_offset": 7     // Minimum separation between change points
}
```

**Photon Bursts:**
```json
{
  "min_level_dwell_time": 0.001,    // Minimum level duration (s)
  "use_sigma_int_thresh": true,     // Use statistical threshold
  "sigma_int_thresh": 3.0,          // Number of std devs
  "defined_int_thresh": 5000        // Fixed intensity threshold
}
```

**Lifetimes:**
```json
{
  "use_moving_avg": true,           // Smooth decay before boundary detection
  "moving_avg_window": 10,          // Window size (channels)
  "start_percent": 80,              // Start fit at 80% of peak
  "end_multiple": 20,               // End at 20x background
  "end_percent": 1,                 // Or at 1% of peak
  "minimum_decay_window": 0.5,      // Minimum fit range (ns)
  "bg_percent": 5                   // Max background as % of peak
}
```

**General:**
```json
{
  "auto_resolve_levels": true  // Auto-run CPA after loading
}
```

### Settings Dialog

`SettingsDialog` provides UI for all settings with:
- Spinboxes for numeric values
- Checkboxes for boolean flags
- Reset to defaults button
- Save to file on accept

---

## Save/Load System

### Save Format

- **Extension:** `.smsa` (Single-Molecule Spectroscopy Analysis)
- **Format:** Python pickle (optionally LZMA compressed)
- **Version:** Tracked via `SAVING_VERSION` constant (currently "1.07")

### What Gets Saved

```python
# Core data
dataset.save_version          # Format version for compatibility
dataset.particles             # All particle objects with analysis
dataset.irf, dataset.irf_t    # IRF data

# UI state
dataset.save_selected         # List of selected particle checkboxes
dataset.global_settings       # Global grouping mode, etc.
dataset.filter_settings       # Current filter configuration
dataset.settings              # Application settings
```

### Save Process

1. Unload HDF5 file handle (can't pickle file objects)
2. Pickle entire `H5dataset` object
3. Reopen HDF5 file handle

### Load Process

1. Unpickle `.smsa` file → `H5dataset` object
2. Open corresponding `.h5` file (must be same location with `.h5` extension)
3. Reconnect file handle to dataset
4. Handle version migrations if needed
5. Restore UI state

### Version Compatibility

The loader handles older save formats by checking `save_version` and applying migrations:
- v1.05 → v1.06: Added `is_secondary_part`, `tcspc_card`, `sec_part`
- v1.06 → v1.07: Added `ab_analysis`

---

## Export System

### Export Options

The `ExporterOptions` class reads checkbox states from the Export tab to determine what to export.

### Data Exports

| Export | File Type | Content |
|--------|-----------|---------|
| Traces | CSV/DataFrame | Time, intensity columns |
| Levels | CSV/DataFrame | Start, end, intensity, photons per level |
| Grouped Levels | CSV/DataFrame | Level data with group assignments |
| Grouping Info | CSV/DataFrame | Group statistics |
| Lifetimes | CSV/DataFrame | Tau, amplitude, chi-squared per entity |
| Histograms | CSV | Decay time, counts |
| Correlation | CSV | Correlation bins, counts |

### DataFrame Formats

Configurable output format:
- Parquet (recommended for large data)
- Feather
- Pickle
- HDF5
- Excel
- CSV

### Plot Exports

Matplotlib-based exports:
- Intensity traces (with/without levels/groups)
- Decay histograms (with/without fits)
- Residuals
- BIC plots
- Correlation histograms
- Spectra
- Raster scans

---

## UI Structure

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Menu Bar                                                       │
├───────────────┬─────────────────────────────────────────────────┤
│               │  Tab Widget (Analysis Tabs)                     │
│   Particle    │  ┌───────────────────────────────────────────┐  │
│   Tree        │  │ Intensity │ Lifetime │ Grouping │ Export │  │
│               │  ├───────────────────────────────────────────┤  │
│   (Selection  │  │                                           │  │
│    list)      │  │   Plots and Controls                      │  │
│               │  │                                           │  │
│               │  │   (pyqtgraph widgets)                     │  │
│               │  │                                           │  │
├───────────────┴──┴───────────────────────────────────────────┤  │
│  Status Bar                                        Progress Bar │
└─────────────────────────────────────────────────────────────────┘
```

### Key UI Files (Qt Designer .ui files)

Located in `src/resources/ui/`:

| File | Purpose |
|------|---------|
| `mainwindow.ui` | Main application window |
| `fitting_dialog.ui` | Lifetime fitting parameters |
| `settings_dialog.ui` | Application settings |

### PyQtGraph Widgets

Main plotting is done with pyqtgraph for performance:

- `pgIntTrace` - Intensity trace plot
- `pgDecay` - Decay histogram plot
- `pgResiduals` - Fit residuals plot
- `pgGroupingBIC` - BIC optimization plot
- `pgCorr` - Correlation plot
- `pgSpectra` - Spectra plot
- `pgRaster_Scan_Image_View` - Raster scan image

---

## Key Algorithms Summary

### Change Point Analysis (Watkins & Yang variant)

**Purpose:** Detect statistically significant intensity changes

**Algorithm:**
1. Start with entire trace as one segment
2. Compute log-likelihood ratio test for each possible split point
3. If best split exceeds confidence threshold, split segment
4. Recursively apply to each new segment
5. Stop when no significant splits found

**Key parameters:**
- Confidence level (typically 95%)
- Minimum photons per level
- Minimum boundary separation

### Agglomerative Hierarchical Clustering (AHCA)

**Purpose:** Group similar brightness levels

**Algorithm:**
1. Initialize: each level is its own cluster
2. Compute pairwise distances (Gaussian-based)
3. Merge closest pair
4. Recompute distances
5. Repeat until one cluster remains
6. Track BIC at each step
7. Select step with minimum BIC

**Distance metric:** Based on overlap of Gaussian distributions (intensity ± uncertainty)

### Lifetime Fitting

**Purpose:** Extract decay time constants

**Algorithm:**
1. Build decay histogram from microtimes
2. Estimate background from pre-rise region
3. Convolve IRF with multi-exponential model
4. Fit model to data using least-squares or maximum likelihood
5. Compute chi-squared, Durbin-Watson for fit quality

**Models:**
- Single exponential: `I(t) = A * exp(-t/tau)`
- Double exponential: `I(t) = A1 * exp(-t/tau1) + A2 * exp(-t/tau2)`
- Triple exponential: Similar with three components

---

## Dependencies

### Python Version

Python 3.8+ (based on typing features used)

### Core Dependencies

| Package | Purpose |
|---------|---------|
| PyQt5 | GUI framework |
| pyqtgraph | Fast plotting |
| numpy | Numerical operations |
| scipy | Optimization, signal processing |
| h5py / h5pickle | HDF5 file handling |
| pandas | DataFrames for export |
| matplotlib | Publication-quality plots |
| dill / pickle | Object serialization |
| lzma | Compression for save files |

### Analysis-Specific

| Package | Purpose |
|---------|---------|
| numdifftools | Hessian calculation for ML fitting |
| pyarrow / feather | Fast DataFrame formats |

### Full requirements in `requirements.txt` and `Pipfile`

---

## Notes for Rewrite

### What Must Stay
1. **HDF5 file format** - This is the data contract with existing measurements

### What Can Change
1. **All analysis code** - Can be rewritten in any language
2. **UI framework** - Moving to web-based (React/Vue)
3. **Communication** - REST API, WebSocket, or similar between frontend and backend
4. **Persistence** - Could use database instead of pickle files

### Considerations for Web Architecture

**Backend needs:**
- Load/parse HDF5 files efficiently
- Run CPU-intensive analysis (consider worker processes, Celery, etc.)
- Stream progress updates (WebSocket)
- Serve analysis results as JSON
- Handle file uploads/downloads

**Frontend needs:**
- Interactive plots (consider Plotly, D3, or similar)
- Real-time progress updates
- Particle selection/filtering
- Parameter configuration dialogs
- Export triggering

**Data flow:**
1. User uploads HDF5 file → stored server-side
2. Backend parses, creates session state
3. Frontend requests analysis via API
4. Backend runs analysis, streams progress
5. Results returned as JSON, rendered client-side
6. Export generates files, served for download
