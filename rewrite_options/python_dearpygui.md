# Full SMS Rewrite: Python + DearPyGui

## Executive Summary

This document outlines a rewrite strategy for the Full SMS application using DearPyGui, a fast GPU-accelerated Python GUI framework based on Dear ImGui. The solution prioritises:

- **Performance**: True GPU rendering via OpenGL/DirectX/Metal with built-in ImPlot for scientific plotting
- **Simplicity**: Single language (Python), single executable, no web technologies
- **Native Experience**: Direct OS integration without browser engine overhead
- **Small Footprint**: ~30-50MB distribution (smallest of all options)
- **Immediate Mode**: Simpler state management paradigm than retained-mode GUIs

This approach is recommended when native performance and minimal complexity are priorities over UI polish.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack Details](#tech-stack-details)
3. [Project Structure](#project-structure)
4. [Core Design](#core-design)
5. [DearPyGui Component Design](#dearpygui-component-design)
6. [Desktop Packaging](#desktop-packaging)
7. [Parallel Processing Strategy](#parallel-processing-strategy)
8. [Data Flow](#data-flow)
9. [State Management](#state-management)
10. [Performance Optimisation](#performance-optimisation)
11. [Development Workflow](#development-workflow)
12. [Risks and Mitigations](#risks-and-mitigations)
13. [Comparison with Other Options](#comparison-with-other-options)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              DearPyGui Application (Single Process)             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Native Viewport (OpenGL/DX/Metal)            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  ImPlot Charts │ ImGui Widgets │ File Dialogs      │  │  │
│  │  └────────────────────────┬────────────────────────────┘  │  │
│  │                           │                                │  │
│  │  ┌────────────────────────▼────────────────────────────┐  │  │
│  │  │               Application State                      │  │  │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────────┐   │  │  │
│  │  │  │ Session    │ │ Particles  │ │   Analysis     │   │  │  │
│  │  │  │ State      │ │ Data       │ │   Results      │   │  │  │
│  │  │  └────────────┘ └────────────┘ └────────────────┘   │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                           │                                │  │
│  │  ┌────────────────────────▼────────────────────────────┐  │  │
│  │  │              Analysis Core (Reused)                  │  │  │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────────────┐  │  │  │
│  │  │  │ChangePoint│ │   AHCA    │ │    FluoFit        │  │  │  │
│  │  │  │ Analysis  │ │ Clustering│ │    Lifetime       │  │  │  │
│  │  │  └───────────┘ └───────────┘ └───────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                           │                                │  │
│  │  ┌────────────────────────▼────────────────────────────┐  │  │
│  │  │      ProcessPoolExecutor (Background Workers)       │  │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │  │  │
│  │  │  │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker N │   │  │  │
│  │  │  │ (NumPy) │ │ (NumPy) │ │ (NumPy) │ │ (NumPy) │   │  │  │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Immediate Mode GUI**: UI is redrawn every frame; no widget state to synchronise
2. **GPU Rendering**: All rendering via OpenGL/DirectX/Metal - true hardware acceleration
3. **Single Process**: UI, analysis, and state in one process (except worker pool)
4. **ImPlot Integration**: Purpose-built scientific plotting with real-time capability
5. **No Web Stack**: No HTML, CSS, JavaScript, WebView, or browser engine

---

## Tech Stack Details

### Core Framework

| Component | Technology | Justification |
|-----------|------------|---------------|
| **GUI Framework** | **DearPyGui 2.x** | GPU-accelerated, ImGui-based, scientific focus |
| **Plotting** | **ImPlot (built-in)** | Real-time capable, GPU-rendered, scientific |
| **Rendering** | **OpenGL/DirectX/Metal** | Native GPU via Dear ImGui backend |
| **Language** | **Python 3.11+** | Team expertise, scientific ecosystem |

### Scientific Computing

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Arrays** | **NumPy** | Foundation for all numerical work |
| **Optimisation** | **SciPy** | curve_fit, signal processing, statistics |
| **HDF5** | **h5py** | Direct HDF5 access (existing format) |
| **DataFrames** | **Pandas** | Export and data manipulation |
| **Parallelism** | **ProcessPoolExecutor** | True parallelism, standard library |

### Packaging

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Bundler** | **PyInstaller** | Mature, cross-platform, single executable |
| **Environment** | **uv** | Fast dependency resolution |

---

## Project Structure

```
full_sms/
├── pyproject.toml
├── src/
│   └── full_sms/
│       ├── __init__.py
│       ├── app.py                  # Main DearPyGui application entry
│       ├── config.py               # Settings management
│       │
│       ├── state/                  # Application state
│       │   ├── __init__.py
│       │   ├── session.py          # Session state container
│       │   ├── particles.py        # Particle data state
│       │   └── analysis.py         # Analysis results state
│       │
│       ├── views/                  # UI view functions (immediate mode)
│       │   ├── __init__.py
│       │   ├── main_window.py      # Main window layout
│       │   ├── particle_list.py    # Particle tree/list view
│       │   ├── intensity_tab.py    # Intensity analysis tab
│       │   ├── lifetime_tab.py     # Lifetime fitting tab
│       │   ├── grouping_tab.py     # Grouping analysis tab
│       │   ├── spectra_tab.py      # Spectra view tab
│       │   ├── raster_tab.py       # Raster scan tab
│       │   ├── antibunching_tab.py # Correlation tab
│       │   ├── export_tab.py       # Export options tab
│       │   └── dialogs/
│       │       ├── fitting_dialog.py
│       │       ├── settings_dialog.py
│       │       └── file_dialogs.py
│       │
│       ├── plots/                  # ImPlot plotting functions
│       │   ├── __init__.py
│       │   ├── intensity_plot.py
│       │   ├── decay_plot.py
│       │   ├── bic_plot.py
│       │   ├── correlation_plot.py
│       │   └── raster_plot.py
│       │
│       ├── analysis/               # Core algorithms (reused from current)
│       │   ├── __init__.py
│       │   ├── change_point.py     # ChangePointAnalysis, Level
│       │   ├── clustering.py       # AHCA, ClusteringStep, Group
│       │   ├── lifetime.py         # FluoFit, OneExp, TwoExp, ThreeExp
│       │   ├── correlation.py      # Antibunching analysis
│       │   └── histograms.py       # Histogram, binning utilities
│       │
│       ├── io/                     # File I/O
│       │   ├── __init__.py
│       │   ├── hdf5_reader.py      # HDF5 file parsing
│       │   ├── session.py          # Save/load .smsa files
│       │   └── exporters.py        # CSV, DataFrame, plot exports
│       │
│       ├── models/                 # Data models
│       │   ├── __init__.py
│       │   ├── particle.py         # Particle, H5dataset
│       │   ├── level.py            # Level data model
│       │   ├── group.py            # Group data model
│       │   └── settings.py         # Application settings
│       │
│       ├── workers/                # Background processing
│       │   ├── __init__.py
│       │   ├── pool.py             # ProcessPoolExecutor wrapper
│       │   └── tasks.py            # Task definitions
│       │
│       └── utils/
│           ├── __init__.py
│           ├── colors.py           # Level/group color schemes
│           └── formatters.py       # Number/time formatting
│
├── tests/
│   ├── conftest.py
│   ├── test_analysis/
│   ├── test_views/
│   └── test_io/
│
└── scripts/
    ├── build.py                    # PyInstaller build script
    └── dev.py                      # Development launcher
```

---

## Core Design

### Understanding Immediate Mode

DearPyGui uses an **immediate mode** paradigm, fundamentally different from PyQt's retained mode:

**Retained Mode (PyQt):**
```python
# Create widgets once, update them when state changes
self.label = QLabel("Count: 0")
layout.addWidget(self.label)
# Later...
self.label.setText(f"Count: {self.count}")  # Must track and update widget
```

**Immediate Mode (DearPyGui):**
```python
# "Recreate" UI every frame based on current state
def render():
    dpg.add_text(f"Count: {state.count}")  # Just read state, no widget tracking
```

This simplifies state management - you don't maintain two copies of state (data + widgets).

### Data Models

```python
# models/particle.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ParticleData:
    """Immutable particle data loaded from HDF5."""
    id: int
    description: str
    abstimes: np.ndarray          # Absolute photon arrival times (ns)
    microtimes: np.ndarray        # TCSPC microtimes (ns)
    channelwidth: float           # ns per TCSPC channel
    tcspc_card: str

    @property
    def num_photons(self) -> int:
        return len(self.abstimes)

    @property
    def measurement_time_s(self) -> float:
        if len(self.abstimes) < 2:
            return 0.0
        return (self.abstimes[-1] - self.abstimes[0]) / 1e9


@dataclass
class LevelData:
    """Single brightness level."""
    id: int
    particle_id: int
    start_idx: int
    end_idx: int
    start_time_ns: int
    end_time_ns: int
    num_photons: int
    group_id: Optional[int] = None

    # Fit results
    tau: Optional[list[float]] = None
    amplitude: Optional[list[float]] = None
    chi_squared: Optional[float] = None

    @property
    def dwell_time_s(self) -> float:
        return (self.end_time_ns - self.start_time_ns) / 1e9

    @property
    def intensity_cps(self) -> float:
        return self.num_photons / self.dwell_time_s


@dataclass
class GroupData:
    """Grouped brightness levels."""
    id: int
    particle_id: int
    level_ids: list[int]
    intensity_cps: float
    total_dwell_time_s: float
    num_photons: int

    # Fit results
    tau: Optional[list[float]] = None
    amplitude: Optional[list[float]] = None
    avtau: Optional[float] = None
    chi_squared: Optional[float] = None
```

### Application State

```python
# state/session.py
from dataclasses import dataclass, field
from typing import Optional
from ..models.particle import ParticleData, LevelData, GroupData


@dataclass
class SessionState:
    """
    Central application state.

    In immediate mode, this is the single source of truth.
    The UI reads from this every frame.
    """
    # File state
    file_path: str = ""
    file_loaded: bool = False

    # Particle data
    particles: list[ParticleData] = field(default_factory=list)
    current_particle_id: int = -1
    selected_particle_ids: set[int] = field(default_factory=set)

    # Analysis results
    levels: dict[int, list[LevelData]] = field(default_factory=dict)
    groups: dict[int, list[GroupData]] = field(default_factory=dict)

    # UI state
    bin_size_ms: float = 10.0
    confidence_level: float = 0.95
    active_tab: str = "intensity"

    # Processing state
    is_processing: bool = False
    progress: float = 0.0
    status_message: str = "Ready"

    # Dialog state
    show_fitting_dialog: bool = False
    show_settings_dialog: bool = False

    @property
    def current_particle(self) -> Optional[ParticleData]:
        if self.current_particle_id < 0:
            return None
        for p in self.particles:
            if p.id == self.current_particle_id:
                return p
        return None

    def get_levels(self, particle_id: int) -> list[LevelData]:
        return self.levels.get(particle_id, [])

    def get_groups(self, particle_id: int) -> list[GroupData]:
        return self.groups.get(particle_id, [])


# Global state instance
state = SessionState()
```

---

## DearPyGui Component Design

### Main Application

```python
# app.py
import dearpygui.dearpygui as dpg
from .state.session import state
from .views.main_window import render_main_window
from .views.dialogs.fitting_dialog import render_fitting_dialog
from .views.dialogs.settings_dialog import render_settings_dialog
from .io.hdf5_reader import load_hdf5_file
from .workers.pool import get_pool


def create_app():
    """Initialize and run the DearPyGui application."""
    dpg.create_context()

    # Configure viewport
    dpg.create_viewport(
        title="Full SMS - Single-Molecule Spectroscopy",
        width=1400,
        height=900,
        min_width=1024,
        min_height=768
    )

    # Set up theme
    _setup_theme()

    # Create main window
    with dpg.window(tag="main_window", no_title_bar=True, no_move=True,
                    no_resize=True, no_collapse=True):
        pass  # Content rendered in frame callback

    # Register frame callback
    dpg.set_frame_callback(1, _first_frame)

    # Setup and run
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    # Main loop with custom render
    while dpg.is_dearpygui_running():
        _render_frame()
        dpg.render_dearpygui_frame()

    # Cleanup
    get_pool().shutdown()
    dpg.destroy_context()


def _first_frame():
    """Called on first frame to set up file dialogs."""
    _setup_file_dialogs()


def _render_frame():
    """Called every frame to update UI."""
    # Clear and rebuild main window content
    dpg.delete_item("main_window", children_only=True)

    with dpg.group(parent="main_window"):
        render_main_window(state)

    # Render modal dialogs if open
    if state.show_fitting_dialog:
        render_fitting_dialog(state)
    if state.show_settings_dialog:
        render_settings_dialog(state)


def _setup_theme():
    """Configure application theme."""
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Dark theme colors
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 41, 59))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (51, 65, 85))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (59, 130, 246))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (96, 165, 250))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (59, 130, 246, 100))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)

    dpg.bind_theme(global_theme)


def _setup_file_dialogs():
    """Create reusable file dialogs."""
    # Open HDF5 dialog
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_on_file_selected,
        tag="file_dialog_open",
        width=700,
        height=400
    ):
        dpg.add_file_extension(".h5", color=(0, 255, 0))
        dpg.add_file_extension(".hdf5", color=(0, 255, 0))

    # Save analysis dialog
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_on_save_selected,
        tag="file_dialog_save",
        width=700,
        height=400
    ):
        dpg.add_file_extension(".smsa", color=(255, 255, 0))


def _on_file_selected(sender, app_data):
    """Handle file selection from dialog."""
    file_path = app_data['file_path_name']
    if file_path:
        _load_file(file_path)


def _on_save_selected(sender, app_data):
    """Handle save location selection."""
    file_path = app_data['file_path_name']
    if file_path:
        _save_session(file_path)


def _load_file(path: str):
    """Load HDF5 file in background."""
    import threading

    state.is_processing = True
    state.status_message = "Loading file..."

    def load_task():
        try:
            particles = load_hdf5_file(path)
            state.particles = particles
            state.file_path = path
            state.file_loaded = True
            if particles:
                state.current_particle_id = particles[0].id
            state.status_message = f"Loaded {len(particles)} particles"
        except Exception as e:
            state.status_message = f"Error: {e}"
        finally:
            state.is_processing = False

    thread = threading.Thread(target=load_task)
    thread.start()


def _save_session(path: str):
    """Save current session."""
    from .io.session import save_session
    save_session(state, path)
    state.status_message = f"Saved to {path}"


if __name__ == "__main__":
    create_app()
```

### Main Window Layout

```python
# views/main_window.py
import dearpygui.dearpygui as dpg
from ..state.session import SessionState
from .particle_list import render_particle_list
from .intensity_tab import render_intensity_tab
from .lifetime_tab import render_lifetime_tab
from .grouping_tab import render_grouping_tab
from .spectra_tab import render_spectra_tab
from .raster_tab import render_raster_tab
from .antibunching_tab import render_antibunching_tab
from .export_tab import render_export_tab


def render_main_window(state: SessionState):
    """Render the main application window."""

    # Menu bar
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(
                label="Open HDF5...",
                callback=lambda: dpg.show_item("file_dialog_open")
            )
            dpg.add_menu_item(
                label="Save Analysis...",
                callback=lambda: dpg.show_item("file_dialog_save"),
                enabled=state.file_loaded
            )
            dpg.add_separator()
            dpg.add_menu_item(
                label="Exit",
                callback=lambda: dpg.stop_dearpygui()
            )

        with dpg.menu(label="Settings"):
            dpg.add_menu_item(
                label="Preferences...",
                callback=lambda: setattr(state, 'show_settings_dialog', True)
            )

    # Main content area
    with dpg.group(horizontal=True):
        # Left sidebar - particle list
        with dpg.child_window(width=280, border=True):
            render_particle_list(state)

        # Right content - tabs
        with dpg.child_window(border=True):
            with dpg.tab_bar(callback=_on_tab_change, tag="main_tabs"):
                with dpg.tab(label="Intensity"):
                    render_intensity_tab(state)

                with dpg.tab(label="Lifetime"):
                    render_lifetime_tab(state)

                with dpg.tab(label="Grouping"):
                    render_grouping_tab(state)

                with dpg.tab(label="Spectra"):
                    render_spectra_tab(state)

                with dpg.tab(label="Raster Scan"):
                    render_raster_tab(state)

                with dpg.tab(label="Antibunching"):
                    render_antibunching_tab(state)

                with dpg.tab(label="Export"):
                    render_export_tab(state)

    # Status bar
    with dpg.group(horizontal=True):
        dpg.add_text(state.status_message)
        dpg.add_spacer(width=-150)
        if state.is_processing:
            dpg.add_progress_bar(
                default_value=state.progress / 100,
                width=140
            )


def _on_tab_change(sender, app_data):
    """Track active tab."""
    from ..state.session import state
    tab_labels = ["intensity", "lifetime", "grouping", "spectra",
                  "raster", "antibunching", "export"]
    if app_data < len(tab_labels):
        state.active_tab = tab_labels[app_data]
```

### Particle List View

```python
# views/particle_list.py
import dearpygui.dearpygui as dpg
from ..state.session import SessionState


def render_particle_list(state: SessionState):
    """Render the particle selection list."""

    dpg.add_text("Particles", color=(150, 200, 255))
    dpg.add_separator()

    if not state.file_loaded:
        dpg.add_text("No file loaded", color=(150, 150, 150))
        dpg.add_button(
            label="Open HDF5 File",
            callback=lambda: dpg.show_item("file_dialog_open")
        )
        return

    # Selection controls
    with dpg.group(horizontal=True):
        dpg.add_button(
            label="Select All",
            callback=lambda: _select_all(state),
            width=80
        )
        dpg.add_button(
            label="Clear",
            callback=lambda: _clear_selection(state),
            width=80
        )

    dpg.add_separator()

    # Particle list with collapsible headers
    with dpg.child_window(height=-60, border=False):
        for particle in state.particles:
            is_selected = particle.id in state.selected_particle_ids
            is_current = particle.id == state.current_particle_id
            has_levels = particle.id in state.levels

            # Collapsible header for each particle
            header_label = f"Particle {particle.id}"
            if has_levels:
                header_label += f" ({len(state.get_levels(particle.id))} levels)"

            # Highlight current particle
            if is_current:
                dpg.add_text(">", color=(59, 130, 246))
                dpg.add_same_line()

            with dpg.collapsing_header(
                label=header_label,
                default_open=(particle.id == state.current_particle_id)
            ):
                # Checkbox for batch selection
                dpg.add_checkbox(
                    label="Selected",
                    default_value=is_selected,
                    callback=lambda s, a, p=particle.id: _toggle_selection(state, p)
                )

                # Info
                dpg.add_text(f"  Photons: {particle.num_photons:,}",
                           color=(180, 180, 180))
                dpg.add_text(f"  Time: {particle.measurement_time_s:.1f} s",
                           color=(180, 180, 180))
                dpg.add_text(f"  {particle.description}",
                           color=(150, 150, 150))

                # Make current button
                if not is_current:
                    dpg.add_button(
                        label="View",
                        callback=lambda s, a, p=particle.id: _set_current(state, p),
                        width=60
                    )

                # Show levels if present
                levels = state.get_levels(particle.id)
                if levels:
                    dpg.add_separator()
                    dpg.add_text("Levels:", color=(150, 200, 150))
                    for i, level in enumerate(levels[:5]):  # Show first 5
                        dpg.add_text(
                            f"  L{i+1}: {level.intensity_cps:.0f} cps, "
                            f"{level.dwell_time_s*1000:.1f} ms",
                            color=(140, 140, 140)
                        )
                    if len(levels) > 5:
                        dpg.add_text(f"  ... and {len(levels)-5} more",
                                   color=(120, 120, 120))

    # Summary
    dpg.add_separator()
    selected_count = len(state.selected_particle_ids)
    dpg.add_text(f"Selected: {selected_count} / {len(state.particles)}")


def _toggle_selection(state: SessionState, particle_id: int):
    if particle_id in state.selected_particle_ids:
        state.selected_particle_ids.remove(particle_id)
    else:
        state.selected_particle_ids.add(particle_id)


def _set_current(state: SessionState, particle_id: int):
    state.current_particle_id = particle_id


def _select_all(state: SessionState):
    state.selected_particle_ids = {p.id for p in state.particles}


def _clear_selection(state: SessionState):
    state.selected_particle_ids.clear()
```

### Intensity Tab with ImPlot

```python
# views/intensity_tab.py
import dearpygui.dearpygui as dpg
import numpy as np
from ..state.session import SessionState
from ..plots.intensity_plot import plot_intensity_trace
from ..workers.pool import get_pool
from ..workers.tasks import run_change_point_analysis


def render_intensity_tab(state: SessionState):
    """Render the intensity analysis tab."""

    particle = state.current_particle
    if particle is None:
        dpg.add_text("Select a particle to view intensity trace")
        return

    # Controls row
    with dpg.group(horizontal=True):
        # Bin size slider
        dpg.add_text("Bin Size (ms):")
        dpg.add_slider_float(
            default_value=state.bin_size_ms,
            min_value=0.1,
            max_value=100.0,
            callback=lambda s, a: setattr(state, 'bin_size_ms', a),
            width=150
        )

        dpg.add_spacer(width=20)

        # Confidence selector
        dpg.add_text("Confidence:")
        dpg.add_combo(
            items=["69%", "90%", "95%", "99%"],
            default_value=f"{int(state.confidence_level * 100)}%",
            callback=_on_confidence_change,
            width=80
        )

        dpg.add_spacer(width=20)

        # Analysis buttons
        dpg.add_button(
            label="Resolve Current",
            callback=lambda: _resolve_levels(state, [state.current_particle_id])
        )
        dpg.add_button(
            label="Resolve Selected",
            callback=lambda: _resolve_levels(state, list(state.selected_particle_ids)),
            enabled=len(state.selected_particle_ids) > 0
        )
        dpg.add_button(
            label="Resolve All",
            callback=lambda: _resolve_levels(state, [p.id for p in state.particles])
        )

    dpg.add_separator()

    # Display options
    with dpg.group(horizontal=True):
        dpg.add_checkbox(label="Show Levels", default_value=True, tag="show_levels")
        dpg.add_checkbox(label="Show Groups", default_value=False, tag="show_groups")
        dpg.add_checkbox(label="Show Histogram", default_value=True, tag="show_hist")

    dpg.add_separator()

    # Main plot area
    with dpg.group(horizontal=True):
        # Intensity trace plot
        with dpg.child_window(width=-250, height=-30):
            plot_intensity_trace(
                particle=particle,
                bin_size_ms=state.bin_size_ms,
                levels=state.get_levels(particle.id) if dpg.get_value("show_levels") else [],
                groups=state.get_groups(particle.id) if dpg.get_value("show_groups") else []
            )

        # Histogram (if enabled)
        if dpg.get_value("show_hist"):
            with dpg.child_window(width=240, height=-30):
                _render_intensity_histogram(particle, state.bin_size_ms)

    # Level info
    levels = state.get_levels(particle.id)
    if levels:
        dpg.add_text(f"Detected {len(levels)} levels")


def _on_confidence_change(sender, app_data):
    from ..state.session import state
    confidence_map = {"69%": 0.69, "90%": 0.90, "95%": 0.95, "99%": 0.99}
    state.confidence_level = confidence_map.get(app_data, 0.95)


def _resolve_levels(state: SessionState, particle_ids: list[int]):
    """Run change point analysis on specified particles."""
    import threading

    if not particle_ids:
        return

    state.is_processing = True
    state.status_message = "Detecting levels..."
    state.progress = 0

    def analysis_task():
        pool = get_pool()
        total = len(particle_ids)

        for i, pid in enumerate(particle_ids):
            particle = next(p for p in state.particles if p.id == pid)

            # Run CPA
            result = run_change_point_analysis(
                abstimes=particle.abstimes,
                confidence=state.confidence_level,
                min_photons=20,
                min_boundary_offset=7
            )

            # Update state with results
            from ..models.particle import LevelData
            levels = [
                LevelData(
                    id=j,
                    particle_id=pid,
                    **level_dict
                )
                for j, level_dict in enumerate(result)
            ]
            state.levels[pid] = levels

            # Update progress
            state.progress = ((i + 1) / total) * 100
            state.status_message = f"Resolved {i + 1}/{total} particles"

        state.is_processing = False
        state.status_message = f"Resolved {total} particles"

    thread = threading.Thread(target=analysis_task)
    thread.start()


def _render_intensity_histogram(particle, bin_size_ms: float):
    """Render intensity distribution histogram."""
    # Compute binned trace
    bin_size_ns = bin_size_ms * 1e6
    bins = np.arange(
        particle.abstimes[0],
        particle.abstimes[-1] + bin_size_ns,
        bin_size_ns
    )
    counts, _ = np.histogram(particle.abstimes, bins=bins)

    # Compute histogram of counts
    hist_counts, hist_edges = np.histogram(counts, bins=30)

    with dpg.plot(label="Intensity Distribution", height=-1, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, label="Counts")
        with dpg.plot_axis(dpg.mvYAxis, label="Frequency"):
            dpg.add_bar_series(
                hist_edges[:-1].tolist(),
                hist_counts.tolist(),
                weight=hist_edges[1] - hist_edges[0]
            )
```

### ImPlot Intensity Plot

```python
# plots/intensity_plot.py
import dearpygui.dearpygui as dpg
import numpy as np
from ..models.particle import ParticleData, LevelData, GroupData


def plot_intensity_trace(
    particle: ParticleData,
    bin_size_ms: float,
    levels: list[LevelData] = None,
    groups: list[GroupData] = None
):
    """
    Render intensity trace using ImPlot.

    ImPlot is GPU-accelerated and handles millions of points efficiently.
    """
    # Compute binned trace
    bin_size_ns = bin_size_ms * 1e6
    bins = np.arange(
        particle.abstimes[0],
        particle.abstimes[-1] + bin_size_ns,
        bin_size_ns
    )
    counts, edges = np.histogram(particle.abstimes, bins=bins)
    time_s = ((edges[:-1] + edges[1:]) / 2) / 1e9  # Convert to seconds

    # Create plot
    with dpg.plot(
        label=f"Particle {particle.id} - Intensity Trace",
        height=-1,
        width=-1,
        anti_aliased=True
    ):
        dpg.add_plot_legend()

        # X axis - time
        dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis")

        # Y axis - counts
        with dpg.plot_axis(dpg.mvYAxis, label="Counts", tag="y_axis"):
            # Main intensity line
            dpg.add_line_series(
                time_s.tolist(),
                counts.tolist(),
                label="Intensity"
            )

            # Level overlays
            if levels:
                _add_level_overlays(levels, bin_size_ms)

            # Group overlays
            if groups:
                _add_group_overlays(groups, levels)


def _add_level_overlays(levels: list[LevelData], bin_size_ms: float):
    """Add colored rectangles for detected levels."""
    for i, level in enumerate(levels):
        # Generate color using golden angle for good distribution
        hue = (i * 137) % 360
        color = _hsv_to_rgb(hue, 0.7, 0.8)

        start_s = level.start_time_ns / 1e9
        end_s = level.end_time_ns / 1e9
        intensity = level.intensity_cps * bin_size_ms / 1000  # Convert to counts

        # Draw level as filled region
        dpg.add_shade_series(
            [start_s, end_s],
            [intensity, intensity],
            y2=[0, 0],
            label=f"L{i+1}" if i < 10 else None
        )


def _add_group_overlays(groups: list[GroupData], levels: list[LevelData]):
    """Add group indicators."""
    level_lookup = {lv.id: lv for lv in levels}

    for i, group in enumerate(groups):
        hue = (i * 137 + 60) % 360  # Offset from level colors
        color = _hsv_to_rgb(hue, 0.9, 0.9)

        # Draw horizontal line at group intensity
        for lid in group.level_ids:
            level = level_lookup.get(lid)
            if level:
                dpg.add_line_series(
                    [level.start_time_ns / 1e9, level.end_time_ns / 1e9],
                    [group.intensity_cps, group.intensity_cps],
                    label=f"G{i+1}" if lid == group.level_ids[0] else None
                )


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    """Convert HSV to RGB tuple (0-255)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return (int(r * 255), int(g * 255), int(b * 255), 180)
```

### Lifetime Tab

```python
# views/lifetime_tab.py
import dearpygui.dearpygui as dpg
import numpy as np
from ..state.session import SessionState
from ..plots.decay_plot import plot_decay_histogram


def render_lifetime_tab(state: SessionState):
    """Render the lifetime fitting tab."""

    particle = state.current_particle
    if particle is None:
        dpg.add_text("Select a particle to view decay histogram")
        return

    # Controls
    with dpg.group(horizontal=True):
        dpg.add_text("Exponentials:")
        dpg.add_radio_button(
            items=["1", "2", "3"],
            default_value="1",
            horizontal=True,
            tag="num_exp"
        )

        dpg.add_spacer(width=20)

        dpg.add_checkbox(label="Use IRF", default_value=True, tag="use_irf")
        dpg.add_checkbox(label="Log Scale", default_value=True, tag="log_scale")

        dpg.add_spacer(width=20)

        dpg.add_button(
            label="Fit Parameters...",
            callback=lambda: setattr(state, 'show_fitting_dialog', True)
        )
        dpg.add_button(
            label="Fit Current",
            callback=lambda: _fit_lifetime(state, particle.id)
        )

    dpg.add_separator()

    # Display options
    with dpg.group(horizontal=True):
        dpg.add_checkbox(label="Show Fit", default_value=True, tag="show_fit")
        dpg.add_checkbox(label="Show Residuals", default_value=True, tag="show_residuals")
        dpg.add_checkbox(label="Show IRF", default_value=False, tag="show_irf")

    dpg.add_separator()

    # Plot area
    plot_height = -150 if dpg.get_value("show_residuals") else -50

    with dpg.child_window(height=plot_height):
        plot_decay_histogram(
            particle=particle,
            log_scale=dpg.get_value("log_scale"),
            show_fit=dpg.get_value("show_fit"),
            show_irf=dpg.get_value("show_irf")
        )

    # Residuals plot (if enabled)
    if dpg.get_value("show_residuals"):
        with dpg.child_window(height=100):
            _render_residuals_plot()

    # Fit results display
    dpg.add_separator()
    _render_fit_results(state, particle.id)


def _fit_lifetime(state: SessionState, particle_id: int):
    """Run lifetime fitting."""
    # TODO: Implement fitting
    state.status_message = "Fitting not yet implemented"


def _render_residuals_plot():
    """Render fit residuals."""
    with dpg.plot(label="Residuals", height=-1, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time (ns)")
        with dpg.plot_axis(dpg.mvYAxis, label="Weighted Residuals"):
            # Placeholder - would show actual residuals after fitting
            dpg.add_line_series([0, 100], [0, 0], label="Zero")


def _render_fit_results(state: SessionState, particle_id: int):
    """Display fit results."""
    with dpg.group(horizontal=True):
        with dpg.child_window(width=200, height=80, border=True):
            dpg.add_text("Fit Results", color=(150, 200, 255))
            dpg.add_text("Tau: -- ns")
            dpg.add_text("Chi-sq: --")

        with dpg.child_window(width=200, height=80, border=True):
            dpg.add_text("Amplitudes", color=(150, 200, 255))
            dpg.add_text("A1: --")
            dpg.add_text("A2: --")
```

### Modal Dialogs

```python
# views/dialogs/fitting_dialog.py
import dearpygui.dearpygui as dpg
from ...state.session import SessionState


def render_fitting_dialog(state: SessionState):
    """Render the fitting parameters dialog as a modal."""

    # Check if dialog window exists
    if not dpg.does_item_exist("fitting_dialog"):
        with dpg.window(
            label="Fitting Parameters",
            modal=True,
            show=True,
            tag="fitting_dialog",
            no_title_bar=False,
            width=400,
            height=500,
            pos=(500, 200)
        ):
            _render_dialog_content(state)
    else:
        dpg.configure_item("fitting_dialog", show=True)


def _render_dialog_content(state: SessionState):
    """Render dialog content."""

    dpg.add_text("Model Configuration", color=(150, 200, 255))
    dpg.add_separator()

    # Number of exponentials
    dpg.add_text("Number of Exponentials:")
    dpg.add_radio_button(
        items=["Single", "Double", "Triple"],
        default_value="Single",
        tag="fit_num_exp"
    )

    dpg.add_separator()
    dpg.add_text("IRF Settings", color=(150, 200, 255))
    dpg.add_separator()

    dpg.add_checkbox(label="Use IRF Convolution", default_value=True, tag="fit_use_irf")
    dpg.add_slider_float(
        label="IRF Shift",
        default_value=0.0,
        min_value=-10.0,
        max_value=10.0,
        tag="fit_irf_shift"
    )

    dpg.add_separator()
    dpg.add_text("Fit Range", color=(150, 200, 255))
    dpg.add_separator()

    dpg.add_checkbox(label="Auto-detect Range", default_value=True, tag="fit_auto_range")
    dpg.add_input_int(label="Start Channel", default_value=0, tag="fit_start")
    dpg.add_input_int(label="End Channel", default_value=1000, tag="fit_end")

    dpg.add_separator()
    dpg.add_text("Initial Guesses", color=(150, 200, 255))
    dpg.add_separator()

    dpg.add_input_float(label="Tau 1 (ns)", default_value=2.0, tag="fit_tau1")
    dpg.add_input_float(label="Tau 2 (ns)", default_value=5.0, tag="fit_tau2")
    dpg.add_input_float(label="Tau 3 (ns)", default_value=10.0, tag="fit_tau3")

    dpg.add_separator()
    dpg.add_text("Background", color=(150, 200, 255))
    dpg.add_separator()

    dpg.add_checkbox(label="Background Correction", default_value=True, tag="fit_bg")
    dpg.add_input_int(label="BG Channels", default_value=50, tag="fit_bg_channels")

    # Buttons
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_button(
            label="Cancel",
            width=100,
            callback=lambda: _close_dialog(state)
        )
        dpg.add_button(
            label="Apply",
            width=100,
            callback=lambda: _apply_settings(state)
        )
        dpg.add_button(
            label="Fit",
            width=100,
            callback=lambda: _fit_and_close(state)
        )


def _close_dialog(state: SessionState):
    state.show_fitting_dialog = False
    dpg.configure_item("fitting_dialog", show=False)


def _apply_settings(state: SessionState):
    # TODO: Apply settings to state
    pass


def _fit_and_close(state: SessionState):
    _apply_settings(state)
    # TODO: Trigger fitting
    _close_dialog(state)
```

---

## Desktop Packaging

### PyInstaller Configuration

DearPyGui packaging is simpler than web-based approaches - no sidecar, no WebView:

```python
# build.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Collect DearPyGui resources
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = []
datas += collect_data_files('dearpygui')

binaries = []
binaries += collect_dynamic_libs('dearpygui')

a = Analysis(
    ['src/full_sms/app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'scipy.optimize',
        'scipy.signal',
        'h5py',
        'numpy',
        'pandas',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Full SMS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.icns' if sys.platform == 'darwin' else 'resources/icon.ico'
)

# macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='Full SMS.app',
        icon='resources/icon.icns',
        bundle_identifier='za.ac.up.fullsms',
        info_plist={
            'CFBundleShortVersionString': '2.0.0',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.15'
        }
    )
```

### Build Script

```python
# scripts/build.py
import subprocess
import sys
import shutil
from pathlib import Path


def clean():
    """Remove build artifacts."""
    for pattern in ['build', 'dist', '*.egg-info']:
        for path in Path('.').glob(pattern):
            shutil.rmtree(path, ignore_errors=True)


def build():
    """Build the application."""
    clean()

    subprocess.run([
        sys.executable, '-m', 'PyInstaller',
        '--noconfirm',
        '--clean',
        'build.spec'
    ], check=True)

    print("\nBuild complete!")
    print(f"Output: {Path('dist').absolute()}")


if __name__ == "__main__":
    build()
```

---

## Parallel Processing Strategy

### ProcessPoolExecutor Wrapper

```python
# workers/pool.py
import asyncio
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Callable, TypeVar
import multiprocessing as mp
import os

T = TypeVar('T')


class AnalysisPool:
    """
    Manages a persistent process pool for CPU-intensive analysis.

    DearPyGui runs in a single thread, so we use a background thread
    to manage the executor and update state when results are ready.
    """

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or max(1, os.cpu_count() - 1)
        self._executor: ProcessPoolExecutor = None

    def _ensure_pool(self):
        """Lazily create the process pool."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context('spawn')
            )

    def submit(self, func: Callable[..., T], *args, **kwargs) -> Future:
        """Submit a task to the pool."""
        self._ensure_pool()
        return self._executor.submit(func, *args, **kwargs)

    def map(
        self,
        func: Callable[..., T],
        items: list,
        progress_callback: Callable[[int, int], None] = None
    ) -> list[T]:
        """
        Map function over items in parallel.

        Blocks until all complete (call from background thread).
        """
        self._ensure_pool()

        if not items:
            return []

        futures = [self._executor.submit(func, item) for item in items]
        results = []

        for i, future in enumerate(futures):
            results.append(future.result())
            if progress_callback:
                progress_callback(i + 1, len(items))

        return results

    def shutdown(self, wait: bool = True):
        """Shutdown the process pool."""
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None


# Global pool instance
_pool: AnalysisPool = None


def get_pool() -> AnalysisPool:
    """Get or create the global analysis pool."""
    global _pool
    if _pool is None:
        _pool = AnalysisPool()
    return _pool
```

### Task Definitions

```python
# workers/tasks.py
import numpy as np


def run_change_point_analysis(
    abstimes: np.ndarray,
    confidence: float,
    min_photons: int,
    min_boundary_offset: int
) -> list[dict]:
    """
    Run change point analysis on photon arrival times.

    This runs in a worker process.
    """
    from ..analysis.change_point import ChangePointAnalysis

    cpa = ChangePointAnalysis()
    levels = cpa.find_change_points(
        abstimes=abstimes,
        confidence=confidence,
        min_photons=min_photons,
        min_boundary_offset=min_boundary_offset
    )

    return [
        {
            'start_idx': lv.start_idx,
            'end_idx': lv.end_idx,
            'start_time_ns': int(lv.start_time_ns),
            'end_time_ns': int(lv.end_time_ns),
            'num_photons': lv.num_photons
        }
        for lv in levels
    ]


def run_lifetime_fit(
    microtimes: np.ndarray,
    channelwidth: float,
    irf: np.ndarray,
    num_exponentials: int,
    **fit_params
) -> dict:
    """
    Fit fluorescence decay with multi-exponential model.
    """
    from ..analysis.lifetime import OneExp, TwoExp, ThreeExp

    # Build histogram
    max_time = microtimes.max()
    bins = np.arange(0, max_time + channelwidth, channelwidth)
    counts, edges = np.histogram(microtimes, bins=bins)
    t = (edges[:-1] + edges[1:]) / 2

    fitters = {1: OneExp, 2: TwoExp, 3: ThreeExp}
    Fitter = fitters[num_exponentials]

    fitter = Fitter(
        t=t,
        decay=counts,
        channelwidth=channelwidth,
        irf=irf,
        **fit_params
    )

    result = fitter.fit()

    return {
        'tau': result['tau'],
        'amplitude': result['amplitude'],
        'chi_squared': result['chi_squared'],
        'avtau': result.get('avtau'),
        'fit_curve': fitter.model(t, *result['params']).tolist(),
        't': t.tolist()
    }
```

---

## Data Flow

### File Open Sequence

```
┌─────────────────┐
│ User clicks     │
│ "Open File"     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DearPyGui       │
│ file_dialog     │
│ opens           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│ User selects    │      │ Callback fires  │
│ .h5 file        │─────>│ with path       │
└─────────────────┘      └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Background      │
                         │ thread spawned  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ h5py reads:     │
                         │ - particles     │
                         │ - abstimes      │
                         │ - microtimes    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ SessionState    │
                         │ updated         │
                         └────────┬────────┘
                                  │
                                  ▼ (next frame)
                         ┌─────────────────┐
                         │ UI re-renders   │
                         │ with new data   │
                         └─────────────────┘
```

### Analysis Sequence

```
┌─────────────────┐
│ User clicks     │
│ "Resolve All"   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│ Callback sets   │─────>│ state.          │
│ processing flag │      │ is_processing   │
└────────┬────────┘      │ = True          │
         │               └─────────────────┘
         ▼
┌─────────────────┐
│ Background      │
│ thread spawned  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│ ProcessPool     │      │ Worker Process  │
│ .submit()       │─────>│ run_cpa_task    │
│                 │      │ (NumPy)         │
└────────┬────────┘      └────────┬────────┘
         │                        │
         │◄───────────────────────┘
         │  result
         ▼
┌─────────────────┐      ┌─────────────────┐
│ Update          │─────>│ state.levels    │
│ state           │      │ updated         │
└────────┬────────┘      └─────────────────┘
         │
         ▼ (next frame)
┌─────────────────┐
│ ImPlot re-draws │
│ with level      │
│ overlays        │
└─────────────────┘
```

---

## State Management

### Immediate Mode State

In immediate mode, state management is simpler than retained mode:

```python
# Traditional retained mode (PyQt)
class MyWidget(QWidget):
    def __init__(self):
        self.count = 0
        self.label = QLabel("0")
        self.button = QPushButton("Increment")
        self.button.clicked.connect(self.increment)

    def increment(self):
        self.count += 1
        self.label.setText(str(self.count))  # Must manually sync

# Immediate mode (DearPyGui)
state = {"count": 0}

def render():
    dpg.add_text(f"Count: {state['count']}")  # Just read state
    if dpg.add_button(label="Increment"):
        state['count'] += 1  # Modify state, UI updates next frame
```

### State Structure

```python
# Single global state object
state = SessionState()

# UI just reads from state each frame
def render_particle_info():
    particle = state.current_particle
    if particle:
        dpg.add_text(f"Photons: {particle.num_photons}")
        dpg.add_text(f"Levels: {len(state.get_levels(particle.id))}")

# Actions modify state, UI updates automatically
def on_resolve_clicked():
    state.is_processing = True
    # ... run analysis ...
    state.levels[particle.id] = new_levels
    state.is_processing = False
    # No explicit UI update needed - next frame reads new state
```

### Threading Considerations

```python
# DearPyGui is single-threaded but thread-safe for state updates
import threading

def background_task():
    """Runs in background thread."""
    result = expensive_computation()

    # Safe to update state from any thread
    state.levels[particle_id] = result
    state.is_processing = False
    # UI thread will see changes on next frame

# Launch background work
thread = threading.Thread(target=background_task)
thread.start()
```

---

## Performance Optimisation

### GPU Rendering

DearPyGui uses GPU for all rendering:

| Feature | Benefit |
|---------|---------|
| **ImPlot GPU backend** | Millions of points at 60fps |
| **Immediate mode** | No retained widget overhead |
| **Native rendering** | No DOM, no CSS, no JavaScript |
| **Direct GPU access** | OpenGL/DirectX/Metal |

### Plotting Performance

```python
# ImPlot handles large datasets efficiently
def plot_large_dataset():
    # 1 million points - no problem
    x = np.linspace(0, 100, 1_000_000)
    y = np.sin(x)

    with dpg.plot(label="Large Dataset"):
        dpg.add_plot_axis(dpg.mvXAxis, label="X")
        with dpg.plot_axis(dpg.mvYAxis, label="Y"):
            # GPU-rendered line series
            dpg.add_line_series(x.tolist(), y.tolist())
```

### Memory Efficiency

| Technique | Application |
|-----------|-------------|
| **Lazy loading** | Load microtimes only when needed |
| **NumPy views** | Avoid copying large arrays |
| **ProcessPoolExecutor** | Parallel analysis without GIL |
| **No widget state** | Immediate mode = less memory overhead |

---

## Development Workflow

### Prerequisites

```bash
# Python 3.11+
brew install python@3.11

# uv for fast package management
pip install uv
```

### Project Setup

```bash
git clone <repo>
cd full-sms

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Running in Development

```bash
# Run application
python src/full_sms/app.py

# With debug output
python src/full_sms/app.py --debug
```

### Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=full_sms --cov-report=html

# Type checking
mypy src/full_sms
```

### pyproject.toml

```toml
[project]
name = "full-sms"
version = "2.0.0"
description = "Single-Molecule Spectroscopy Analysis"
requires-python = ">=3.11"
dependencies = [
    "dearpygui>=2.0.0",
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "h5py>=3.10.0",
    "pandas>=2.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.7.0",
    "ruff>=0.1.6",
    "pyinstaller>=6.3.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **ImGui aesthetic** | High | Medium | Custom theming; accept functional over pretty |
| **Immediate mode learning curve** | Medium | Medium | Different paradigm but simpler once understood |
| **Smaller community** | Medium | Low | Good documentation; active Discord community |
| **Limited layout system** | Medium | Medium | Manual positioning; use tables for grids |
| **Cross-platform rendering** | Low | High | Test early; OpenGL widely supported |
| **Large dataset scrolling** | Low | Medium | Virtualisation via clipper API |

### Alternative Paths

If DearPyGui proves insufficient:

1. **PyQt6**: Return to retained mode, familiar territory
2. **Panel + PyWebView**: Web-based alternative
3. **Dear ImGui C++ directly**: More control, less Python convenience

---

## Comparison with Other Options

| Criterion | DearPyGui | Panel + PyWebView | FastAPI + React + Tauri |
|-----------|:---------:|:-----------------:|:-----------------------:|
| **Languages** | 1 (Python) | 1 (Python) | 3 (Python, TS, Rust) |
| **Rendering** | GPU (native) | WebGL (browser) | WebGL (browser) |
| **Bundle size** | ~30-50MB | ~50-80MB | ~65MB |
| **IPC** | None | None | HTTP + WebSocket |
| **State management** | Simple (immediate) | Medium (Param) | Complex (dual) |
| **UI polish** | Functional | Dashboard | App-quality |
| **Development time** | 3-4 months | 3-4 months | 6-9 months |
| **Tree views** | Collapsing headers | Limited (Tabulator) | Full (react-arborist) |
| **Modal dialogs** | Native | Workarounds | Native |
| **Plotting performance** | Excellent (ImPlot) | Good (Bokeh WebGL) | Good (Plotly WebGL) |
| **Community size** | Small | Medium | Large |
| **Learning curve** | Medium (new paradigm) | Low (Python) | High (multi-stack) |

### When to Choose DearPyGui

**Choose DearPyGui if:**
- Performance and responsiveness are critical
- Functional UI is acceptable (over polished)
- Single developer maintaining long-term
- Want smallest possible distribution size
- Interested in trying immediate mode paradigm
- Scientific plotting is primary use case

**Avoid DearPyGui if:**
- UI polish is a hard requirement
- Need web deployment option
- Team prefers web technologies
- Extensive custom widget needs
- Drag-and-drop is critical feature

---

## Summary

The DearPyGui architecture provides:

- **True GPU Performance**: Native rendering via Dear ImGui, not browser-based
- **Simplest Stack**: Single Python process, no web technologies
- **Immediate Mode**: Simpler state management paradigm
- **Smallest Bundle**: ~30-50MB without browser engine overhead
- **Built-in Scientific Plotting**: ImPlot purpose-built for real-time data
- **Native File Dialogs**: Proper OS integration without workarounds
- **Fast Development**: Estimated 3-4 months for feature parity

This approach is recommended when native performance and minimal complexity are priorities, and when a functional (rather than polished) UI is acceptable. The immediate mode paradigm is different from PyQt but often simpler once understood.

---

*Document created: December 2024*
*Related: python_panel_pywebview.md, python_fastapi_react_tauri.md*
