# Full SMS Rewrite Plan: Python + DearPyGui

This document tracks the incremental rewrite of Full SMS from PyQt5 to DearPyGui. Each task is designed to be completable in a single Claude Code session.

---

## How to Use This Document

1. Find the task marked `[NEXT]`
2. Tell Claude: "Continue with the next task in the rewrite plan"
3. Claude will complete the task and update this document:
   - Change `[NEXT]` to `[DONE]` with completion date
   - Mark the following task as `[NEXT]`
4. Repeat daily until complete

---

## Status Legend

- `[DONE]` - Task completed (date in parentheses)
- `[NEXT]` - Current task to work on
- `[TODO]` - Pending task
- `[SKIP]` - Task skipped (reason noted)
- `[BLOCKED]` - Waiting on something

---

## Phase 1: Project Setup

### Task 1.1: Move existing code to old/ directory `[DONE]` (2025-01-05)
**Objective**: Preserve existing code for reference while creating clean workspace

**Actions**:
- Create `old/` directory at project root
- Move all existing source files (`src/`, `Pipfile`, `Pipfile.lock`, `build.spec`, etc.) to `old/`
- Keep `rewrite_options/`, `full_context.md`, `CLAUDE.md`, and this plan file in place
- Keep `.git` intact (do not move)
- Keep any test HDF5 data files accessible

**Verification**: Project root is clean except for docs and git

---

### Task 1.2: Initialize project with mise and uv `[DONE]` (2025-01-05)
**Objective**: Set up modern Python tooling

**Actions**:
- Create `.mise.toml` with Python 3.12 configuration
- Run `mise install` to ensure Python is available
- Run `uv init` or create `pyproject.toml` manually
- Create `uv.lock` by running `uv sync`
- Create `.python-version` file

**Verification**: `uv run python --version` works

---

### Task 1.3: Create project structure `[DONE]` (2025-01-05)
**Objective**: Set up the new source tree

**Actions**:
- Create directory structure:
  ```
  src/
  └── full_sms/
      ├── __init__.py
      ├── app.py              # Entry point
      ├── config.py           # Settings
      ├── models/             # Data models
      ├── analysis/           # Algorithms
      ├── io/                 # File I/O
      ├── ui/                 # DearPyGui components
      │   ├── views/          # Tab views
      │   ├── plots/          # ImPlot wrappers
      │   ├── widgets/        # Reusable widgets
      │   └── dialogs/        # Modal dialogs
      ├── workers/            # Background processing
      └── utils/              # Helpers
  tests/
  └── conftest.py
  ```
- Add DearPyGui and core dependencies to pyproject.toml
- Create minimal `app.py` that opens an empty DearPyGui window

**Verification**: `uv run python -m full_sms.app` opens a window

---

## Phase 2: Data Models

### Task 2.1: Create core data models `[DONE]` (2026-01-05)
**Objective**: Define immutable data structures for particles and channels

**Actions**:
- Create `models/particle.py`:
  - `ParticleData` dataclass (id, description, tcspc_card, channelwidth)
  - `ChannelData` dataclass (abstimes, microtimes arrays)
  - Properties for num_photons, measurement_time_s
- Create `models/__init__.py` with exports
- Write unit tests in `tests/test_models/test_particle.py`

**Verification**: Tests pass

---

### Task 2.2: Create analysis result models `[DONE]` (2026-01-05)
**Objective**: Define structures for analysis outputs

**Actions**:
- Create `models/level.py`:
  - `LevelData` dataclass (indices, times, photon count, optional fit results)
  - Properties for dwell_time_s, intensity_cps
- Create `models/group.py`:
  - `GroupData` dataclass (level_ids, aggregate statistics, fit results)
- Create `models/fit.py`:
  - `FitResult` dataclass (tau, amplitude, chi_squared, residuals, etc.)
- Write unit tests

**Verification**: Tests pass

---

### Task 2.3: Create session state model `[DONE]` (2026-01-05)
**Objective**: Define the application state container

**Actions**:
- Create `models/session.py`:
  - `SessionState` dataclass containing:
    - File path and metadata
    - List of ParticleData
    - Current particle/channel selection
    - Analysis results (levels dict, groups dict)
    - UI state (bin_size, confidence, active_tab)
    - Processing state (is_busy, progress, message)
  - Methods for get_particle(), get_levels(), get_groups()
- This will be the single source of truth for the app

**Verification**: Can instantiate and manipulate SessionState

---

## Phase 3: File I/O

### Task 3.1: Implement HDF5 file reader `[DONE]` (2026-01-05)
**Objective**: Load HDF5 files into the new data model

**Actions**:
- Create `io/hdf5_reader.py`:
  - `load_h5_file(path) -> tuple[FileMetadata, list[ParticleData]]`
  - Handle single and dual TCSPC channels
  - Load IRF, spectra, raster scan if present
  - Reference `old/src/smsh5_file_reader.py` for format details
- Write tests with a sample HDF5 file (or mock)

**Verification**: Can load an HDF5 file and access particle abstimes

---

### Task 3.2: Implement session save/load `[DONE]` (2026-01-05)
**Objective**: Persist and restore analysis sessions

**Actions**:
- Create `io/session.py`:
  - `save_session(state: SessionState, path: Path)` - JSON format
  - `load_session(path: Path) -> dict` - Returns data to restore state
  - Handle version compatibility
- Use JSON (not pickle) for transparency and safety
- Store references to HDF5 file path (not the raw data)

**Verification**: Round-trip save/load preserves analysis results

---

## Phase 4: Analysis Core

### Task 4.1: Port histogram utilities `[DONE]` (2026-01-05)
**Objective**: Implement binning and histogram functions

**Actions**:
- Create `analysis/histograms.py`:
  - `bin_photons(abstimes, bin_size_ns) -> (times, counts)`
  - `build_decay_histogram(microtimes, channelwidth) -> (t, counts)`
  - Ensure NumPy vectorized implementations
- Reference `old/src/smsh5.py` for current implementation

**Verification**: Histogram output matches expected shape and values

---

### Task 4.2: Port change point analysis `[DONE]` (2026-01-05)
**Objective**: Implement the Watkins & Yang CPA algorithm

**Actions**:
- Create `analysis/change_point.py`:
  - `find_change_points(abstimes, confidence, min_photons, min_offset) -> list[LevelData]`
  - Implement log-likelihood ratio calculation
  - Implement recursive splitting
  - Use chi-squared threshold from scipy.stats
- Reference `old/src/change_point.py`
- Focus on correctness first, optimization later

**Verification**: Detects expected levels in test data

---

### Task 4.3: Port AHCA clustering `[DONE]` (2026-01-06)
**Objective**: Implement agglomerative hierarchical clustering with BIC, optimized with Numba JIT

**Actions**:
- Add `numba` dependency to pyproject.toml
- Create `analysis/clustering.py`:
  - `cluster_levels(levels, use_lifetime) -> ClusteringResult`
  - `ClusteringResult` contains steps, BIC values, optimal grouping
  - Implement Gaussian-based distance metric
  - Track BIC at each merge step
  - Use Numba `@jit(nopython=True)` for merge merit calculation (O(n²) inner loop)
- Reference `old/src/grouping.py`

**Verification**: Produces expected number of groups, BIC curve is sensible

---

### Task 4.3.1: JIT-optimize change point analysis `[DONE]` (2026-01-06)
**Objective**: Apply Numba JIT to CPA computational hotspots

**Actions**:
- Refactor `_weighted_likelihood_ratio()` inner loop as Numba-compiled function
- Vectorize `_compute_sums()` using cumulative operations (or JIT compile)
- Benchmark before/after optimization
- Ensure all CPA tests still pass

**Verification**: Analysis completes faster, tests pass, results unchanged

---

### Task 4.4: Port lifetime fitting - single exponential `[DONE]` (2026-01-06)
**Objective**: Implement single-exponential decay fitting

**Actions**:
- Create `analysis/lifetime.py`:
  - `fit_decay(t, counts, irf, num_exp, **params) -> FitResult`
  - Start with single exponential model
  - Implement IRF convolution
  - Use scipy.optimize.curve_fit
  - Calculate chi-squared, Durbin-Watson
- Reference `old/src/tcspcfit.py`

**Verification**: Fit converges on synthetic data with known tau

---

### Task 4.5: Port lifetime fitting - multi-exponential `[DONE]` (2026-01-06)
**Objective**: Extend fitting to 2 and 3 exponential components

**Actions**:
- Extend `analysis/lifetime.py`:
  - Add TwoExp, ThreeExp models
  - Implement amplitude-weighted average lifetime
  - Handle parameter bounds and initial guesses
- Ensure backwards compatibility with single-exp interface

**Verification**: Bi-exponential fit recovers two known taus

---

### Task 4.6: Port correlation analysis `[DONE]` (2026-01-06)
**Objective**: Implement second-order photon correlation (antibunching)

**Actions**:
- Create `analysis/correlation.py`:
  - `calculate_g2(times1, times2, window, binsize) -> (tau, g2)`
  - Cross-correlation between two SPAD channels
- Reference `old/src/antibunching.py`

**Verification**: g2 curve shows expected antibunching dip

---

## Phase 5: Worker Infrastructure

### Task 5.1: Implement parallel processing infrastructure `[DONE]` (2026-01-06)
**Objective**: Create reusable parallel processing infrastructure with modern Python

**Actions**:
- **Upgrade Python version**:
  - Update `.mise.toml` and `pyproject.toml` to the latest Python version supported by all dependencies (3.14 if Numba, DearPyGui, NumPy, SciPy, h5py all support it)
  - Run `uv sync` to verify all dependencies resolve
  - Run tests to confirm nothing breaks
- **Evaluate parallelism approach** - consider trade-offs:
  - **Multiprocessing** (ProcessPoolExecutor): Proven, stable, works well for CPU-bound scientific computing. May actually be faster for some workloads due to memory isolation.
  - **Free-threading** (no-GIL build): True thread parallelism, lower overhead than processes, but experimental. Numba's free-threading support is still experimental.
  - **Multiple interpreters** (`concurrent.interpreters`): New in 3.14, actor-model concurrency, less overhead than multiprocessing.
  - Document the decision rationale in code comments or a brief note
- **If free-threading is chosen**: Start with the standard Python 3.14 build first, not the free-threaded build. Get the infrastructure working, then optionally test with free-threaded build later.
- Create `workers/pool.py`:
  - `AnalysisPool` class with persistent executor
  - `submit()` for single tasks
  - `map_with_progress()` for batch tasks with callback
  - Proper cleanup on shutdown
- Use `multiprocessing.get_context('spawn')` for GUI compatibility (if using multiprocessing)

**Verification**: Python version upgraded, dependencies work, can run a simple parallel task

---

### Task 5.2: Implement analysis task functions `[DONE]` (2026-01-06)
**Objective**: Create picklable task functions for workers

**Actions**:
- Create `workers/tasks.py`:
  - `run_cpa_task(params) -> dict` - Change point analysis
  - `run_clustering_task(params) -> dict` - AHCA
  - `run_fit_task(params) -> dict` - Lifetime fitting
  - `run_correlation_task(params) -> dict` - g2 calculation
- These must be module-level functions (not methods)

**Verification**: Tasks can be submitted to pool and return results

---

## Phase 6: UI Foundation

### Task 6.1: Create main application window `[DONE]` (2026-01-06)
**Objective**: Set up the core DearPyGui application structure

**Actions**:
- Update `app.py`:
  - Initialize DearPyGui context and viewport
  - Create primary window with menu bar
  - Set up theme (dark, scientific-appropriate)
  - Create frame callback for main render loop
  - Handle graceful shutdown
- Create `ui/theme.py` for consistent styling

**Verification**: App launches with menu bar and themed window

---

### Task 6.2: Create main layout with sidebar and tabs `[DONE]` (2026-01-06)
**Objective**: Implement the two-column layout with tab navigation

**Actions**:
- Create `ui/layout.py`:
  - Left sidebar (particle tree area)
  - Right content area with tab bar
  - Status bar at bottom
- Use DearPyGui's child_window and group for layout
- Tab bar with: Intensity, Lifetime, Grouping, Spectra, Raster, Correlation, Export

**Verification**: Layout renders correctly, tabs switch

---

### Task 6.3: Create particle tree component `[DONE]` (2026-01-07)
**Objective**: Implement hierarchical particle/channel selection

**Actions**:
- Create `ui/widgets/particle_tree.py`:
  - Tree nodes for each particle
  - Nested selectables for SPAD channels
  - Checkbox for batch selection
  - Visual indicator for current selection
  - Select All / Clear buttons
- Wire up to SessionState for selection tracking

**Verification**: Can expand particles, select channels, batch select

---

### Task 6.4: Create status bar and progress indicator `[DONE]` (2026-01-07)
**Objective**: Show status messages and analysis progress

**Actions**:
- Create `ui/widgets/status_bar.py`:
  - Status message text
  - Progress bar (visible during processing)
  - File info display
- Wire up to SessionState.is_busy, progress, message

**Verification**: Progress bar animates during mock processing

---

## Phase 7: Intensity Tab

### Task 7.1: Implement intensity trace plot `[NEXT]`
**Objective**: Render binned photon counts over time with ImPlot

**Actions**:
- Create `ui/plots/intensity_plot.py`:
  - Line plot of counts vs time
  - Configurable bin size
  - Zoom/pan enabled
  - Axis labels and formatting
- Create `ui/views/intensity_tab.py`:
  - Plot area
  - Bin size slider
  - Basic controls

**Verification**: Intensity trace renders from loaded HDF5 data

---

### Task 7.2: Implement level overlay rendering `[TODO]`
**Objective**: Display detected levels as colored regions on intensity plot

**Actions**:
- Extend `ui/plots/intensity_plot.py`:
  - Render levels as shaded rectangles
  - Color coding by level index (or group)
  - Toggle visibility
- Handle performance with many levels (50+)

**Verification**: Levels appear as colored bands on trace

---

### Task 7.3: Implement intensity histogram sidebar `[TODO]`
**Objective**: Show distribution of intensity values

**Actions**:
- Create `ui/plots/intensity_histogram.py`:
  - Vertical histogram of bin counts
  - Updates with bin size changes
- Add to intensity tab layout (right side)
- Toggle visibility checkbox

**Verification**: Histogram shows distribution of counts

---

### Task 7.4: Implement resolve controls `[TODO]`
**Objective**: Trigger change point analysis from UI

**Actions**:
- Extend `ui/views/intensity_tab.py`:
  - Confidence level selector (69%, 90%, 95%, 99%)
  - Resolve Current / Resolve Selected / Resolve All buttons
  - Wire up to worker pool
  - Update SessionState.levels on completion
  - Show progress during analysis

**Verification**: Clicking Resolve detects levels and displays them

---

## Phase 8: Lifetime Tab

### Task 8.1: Implement decay histogram plot `[TODO]`
**Objective**: Display fluorescence decay curve

**Actions**:
- Create `ui/plots/decay_plot.py`:
  - Log-scale Y axis (toggleable)
  - Time in nanoseconds on X
  - Data as line/scatter
- Create `ui/views/lifetime_tab.py`:
  - Plot area
  - Log scale toggle

**Verification**: Decay histogram renders from particle microtimes

---

### Task 8.2: Implement fit curve overlay `[TODO]`
**Objective**: Show fitted curve on decay plot

**Actions**:
- Extend `ui/plots/decay_plot.py`:
  - Overlay fit curve in different color
  - IRF display (optional, dashed)
  - Legend for data/fit/IRF
- Display fit results (tau, chi-squared) as text

**Verification**: Fit curve overlays data after fitting

---

### Task 8.3: Implement residuals plot `[TODO]`
**Objective**: Show weighted residuals below decay plot

**Actions**:
- Create `ui/plots/residuals_plot.py`:
  - Weighted residuals vs time
  - Zero line reference
  - Linked X axis with decay plot
- Add below decay plot in lifetime tab

**Verification**: Residuals display after fitting

---

### Task 8.4: Implement fitting dialog and controls `[TODO]`
**Objective**: Configure and trigger lifetime fitting

**Actions**:
- Create `ui/dialogs/fitting_dialog.py`:
  - Number of exponentials (1/2/3)
  - IRF settings (use IRF, shift)
  - Fit range (auto or manual)
  - Initial guesses for tau
  - Background correction settings
  - Fit / Cancel buttons
- Create modal using DearPyGui's popup_modal
- Wire up to fit_decay() and worker pool

**Verification**: Dialog opens, parameters apply, fit runs

---

## Phase 9: Grouping Tab

### Task 9.1: Implement BIC plot `[TODO]`
**Objective**: Display BIC optimization curve for clustering

**Actions**:
- Create `ui/plots/bic_plot.py`:
  - BIC value vs number of groups
  - Highlight optimal (minimum) point
  - Allow clicking to select different group count
- Create `ui/views/grouping_tab.py`:
  - BIC plot
  - Current group count display

**Verification**: BIC curve displays after clustering

---

### Task 9.2: Implement group visualization `[TODO]`
**Objective**: Show groups on intensity plot and in list

**Actions**:
- Extend level overlay to show group coloring
- Add group list/table showing:
  - Group ID
  - Intensity
  - Dwell time
  - Number of levels
  - Fit results (if fitted)
- Selection highlights group on plot

**Verification**: Groups visible on plot with distinct colors

---

### Task 9.3: Implement grouping controls `[TODO]`
**Objective**: Trigger clustering and adjust group count

**Actions**:
- Extend `ui/views/grouping_tab.py`:
  - Group Current / Selected / All buttons
  - Use lifetime in clustering toggle
  - Manual group count override slider
  - Global grouping mode toggle
- Wire up to clustering worker

**Verification**: Clustering runs, groups appear, can adjust count

---

## Phase 10: Additional Tabs

### Task 10.1: Implement spectra tab `[TODO]`
**Objective**: Display spectral data if present in file

**Actions**:
- Create `ui/views/spectra_tab.py`:
  - Check if file has spectra
  - Line plot of intensity vs wavelength
  - Handle case where spectra not present

**Verification**: Spectra displays if available, graceful fallback if not

---

### Task 10.2: Implement raster scan tab `[TODO]`
**Objective**: Display 2D raster scan image if present

**Actions**:
- Create `ui/views/raster_tab.py`:
  - Check if file has raster scan
  - 2D image/heatmap display
  - Color scale controls
  - Handle case where not present

**Verification**: Raster image displays if available

---

### Task 10.3: Implement correlation tab `[TODO]`
**Objective**: Display antibunching / g2 correlation

**Actions**:
- Create `ui/plots/correlation_plot.py`:
  - g2 vs tau plot
  - Symmetric around zero
- Create `ui/views/correlation_tab.py`:
  - Only enabled for dual-channel particles
  - Window and bin size controls
  - Correlate button
- Wire up to correlation worker

**Verification**: g2 curve displays for dual-channel data

---

## Phase 11: Export

### Task 11.1: Implement data export `[TODO]`
**Objective**: Export analysis results to files

**Actions**:
- Create `io/exporters.py`:
  - Export intensity trace to CSV
  - Export levels to CSV/DataFrame
  - Export groups to CSV/DataFrame
  - Export fit results
  - Support multiple formats (CSV, Parquet, Excel)
- Create `ui/views/export_tab.py`:
  - Checkboxes for what to export
  - Format selection
  - Output directory picker
  - Export button

**Verification**: Export produces valid files with expected data

---

### Task 11.2: Implement plot export `[TODO]`
**Objective**: Export plots as images

**Actions**:
- Extend `io/exporters.py`:
  - Export intensity plot (PNG/PDF via matplotlib recreation)
  - Export decay plot
  - Export BIC plot
- DearPyGui/ImPlot doesn't have direct export, so recreate with matplotlib for publication quality

**Verification**: Exported images look correct

---

## Phase 12: Polish & Packaging

### Task 12.1: Implement settings dialog `[TODO]`
**Objective**: Configure application preferences

**Actions**:
- Create `ui/dialogs/settings_dialog.py`:
  - Change point settings (min_photons, min_boundary_offset)
  - Lifetime settings (moving average, fit boundaries)
  - Default bin size
  - Save/load from config file
- Create `config.py` for settings persistence

**Verification**: Settings save and persist across restarts

---

### Task 12.2: Implement keyboard shortcuts `[TODO]`
**Objective**: Add productivity shortcuts

**Actions**:
- Implement in `app.py`:
  - Cmd/Ctrl+O: Open file
  - Cmd/Ctrl+S: Save analysis
  - Cmd/Ctrl+E: Export
  - Cmd/Ctrl+R: Resolve current
  - Tab/Shift+Tab: Navigate tabs
- Use DearPyGui's keyboard handler

**Verification**: Shortcuts work as expected

---

### Task 12.3: Implement file dialogs `[TODO]`
**Objective**: Native file open/save dialogs

**Actions**:
- Use DearPyGui's file_dialog for:
  - Open HDF5 file
  - Save analysis (.smsa)
  - Load analysis
  - Export directory selection
- Filter by appropriate extensions

**Verification**: Dialogs open, filter correctly, return paths

---

### Task 12.4: Create PyInstaller build configuration `[TODO]`
**Objective**: Package application for distribution

**Actions**:
- Create `build.spec` for PyInstaller
- Handle DearPyGui resources
- Include hidden imports (scipy, h5py, numpy)
- Configure for single executable
- Add application icon
- Test on current platform

**Verification**: Built executable runs correctly

---

### Task 12.5: Cross-platform testing and fixes `[TODO]`
**Objective**: Ensure app works on macOS, Windows, Linux

**Actions**:
- Test on each platform (or document testing needed)
- Fix any platform-specific issues:
  - File path handling
  - DPI scaling
  - GPU rendering backend selection
- Document any platform-specific notes

**Verification**: App functions on all target platforms

---

## Progress Summary

| Phase | Tasks | Completed | Remaining |
|-------|-------|-----------|-----------|
| 1. Setup | 3 | 3 | 0 |
| 2. Data Models | 3 | 3 | 0 |
| 3. File I/O | 2 | 2 | 0 |
| 4. Analysis Core | 7 | 7 | 0 |
| 5. Workers | 2 | 2 | 0 |
| 6. UI Foundation | 4 | 4 | 0 |
| 7. Intensity Tab | 4 | 0 | 4 |
| 8. Lifetime Tab | 4 | 0 | 4 |
| 9. Grouping Tab | 3 | 0 | 3 |
| 10. Additional Tabs | 3 | 0 | 3 |
| 11. Export | 2 | 0 | 2 |
| 12. Polish | 5 | 0 | 5 |
| **Total** | **42** | **21** | **21** |

---

## Notes

- Tasks can be split further if they prove too large
- Some tasks may be combined if they're trivial
- Reference `old/` directory for implementation details
- Deviation from old implementation is acceptable if it improves the result
- Test data files should remain accessible during development

---

*Created: January 2025*
*Last Updated: 2026-01-07 (Task 6.4 completed - status bar and progress indicator)*
