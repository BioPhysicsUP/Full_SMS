# Full SMS Rewrite: Python + Panel + PyWebView

## Executive Summary

This document outlines a rewrite strategy for the Full SMS application using the HoloViz ecosystem (Panel + Bokeh) for the UI, packaged as a native desktop application via PyWebView. The solution prioritises:

- **Simplicity**: Single language (Python) throughout the entire stack
- **Maintainability**: No IPC boundaries, no multi-language debugging
- **Performance**: WebGL-accelerated plots via Bokeh, ProcessPoolExecutor for parallelism
- **Offline-first**: Fully local execution with no cloud dependencies
- **Familiarity**: Leverages existing Python expertise and scientific ecosystem

This approach is recommended for single-developer maintenance in an academic/scientific context.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack Details](#tech-stack-details)
3. [Project Structure](#project-structure)
4. [Core Design](#core-design)
5. [Panel Component Design](#panel-component-design)
6. [Desktop Packaging](#desktop-packaging)
7. [Parallel Processing Strategy](#parallel-processing-strategy)
8. [Data Flow](#data-flow)
9. [State Management](#state-management)
10. [Performance Optimisation](#performance-optimisation)
11. [Development Workflow](#development-workflow)
12. [Risks and Mitigations](#risks-and-mitigations)
13. [Comparison with FastAPI + React + Tauri](#comparison-with-fastapi--react--tauri)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│               PyWebView Shell (Native Window)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Panel Application (Bokeh Server)              │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Bokeh Plots (WebGL) │ Panel Widgets │ Param State │  │  │
│  │  └────────────────────────────┬────────────────────────┘  │  │
│  │                               │                            │  │
│  │  ┌────────────────────────────▼────────────────────────┐  │  │
│  │  │           Parameterized Controllers                  │  │  │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────────┐   │  │  │
│  │  │  │ Intensity  │ │ Lifetime   │ │   Grouping     │   │  │  │
│  │  │  │ Controller │ │ Controller │ │   Controller   │   │  │  │
│  │  │  └────────────┘ └────────────┘ └────────────────┘   │  │  │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────────┐   │  │  │
│  │  │  │ Spectra    │ │ Raster     │ │ Antibunching   │   │  │  │
│  │  │  │ Controller │ │ Controller │ │ Controller     │   │  │  │
│  │  │  └────────────┘ └────────────┘ └────────────────┘   │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                               │                            │  │
│  │  ┌────────────────────────────▼────────────────────────┐  │  │
│  │  │              Analysis Core (Reused)                  │  │  │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────────────┐  │  │  │
│  │  │  │ChangePoint│ │   AHCA    │ │    FluoFit        │  │  │  │
│  │  │  │ Analysis  │ │ Clustering│ │    Lifetime       │  │  │  │
│  │  │  └───────────┘ └───────────┘ └───────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                               │                            │  │
│  │  ┌────────────────────────────▼────────────────────────┐  │  │
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

1. **Single Process**: UI and analysis run in the same Python process (except worker pool)
2. **No IPC**: No HTTP/WebSocket between frontend and backend
3. **Param-based State**: All reactive state uses the `param` library
4. **WebGL Plots**: Bokeh's WebGL backend for millions of data points
5. **ProcessPoolExecutor**: Standard library multiprocessing for CPU-intensive work

---

## Tech Stack Details

### Core Framework

| Component | Technology | Justification |
|-----------|------------|---------------|
| **UI Framework** | **Panel 1.4+** | Python-native, HoloViz ecosystem, scientific focus |
| **Plotting** | **Bokeh 3.x** | WebGL acceleration, rich interactivity, streaming |
| **High-level Plots** | **hvPlot / HoloViews** | Declarative API, automatic linking |
| **State Management** | **Param 2.x** | Reactive parameters, type validation |
| **Desktop Shell** | **PyWebView 5.x** | Native window wrapper, single executable |

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
│       ├── app.py                  # Main Panel application entry
│       ├── config.py               # Settings management
│       │
│       ├── controllers/            # Domain controllers (Parameterized)
│       │   ├── __init__.py
│       │   ├── base.py             # BaseController class
│       │   ├── intensity.py        # IntensityController
│       │   ├── lifetime.py         # LifetimeController
│       │   ├── grouping.py         # GroupingController
│       │   ├── spectra.py          # SpectraController
│       │   ├── raster.py           # RasterController
│       │   ├── antibunching.py     # AntibunchingController
│       │   └── filtering.py        # FilteringController
│       │
│       ├── components/             # Reusable Panel components
│       │   ├── __init__.py
│       │   ├── plots/
│       │   │   ├── intensity_plot.py
│       │   │   ├── decay_plot.py
│       │   │   ├── bic_plot.py
│       │   │   ├── correlation_plot.py
│       │   │   └── raster_plot.py
│       │   ├── widgets/
│       │   │   ├── particle_tree.py
│       │   │   ├── file_picker.py
│       │   │   ├── progress_indicator.py
│       │   │   └── fitting_dialog.py
│       │   └── layout/
│       │       ├── main_layout.py
│       │       ├── sidebar.py
│       │       └── tabs.py
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
│   ├── test_controllers/
│   └── test_io/
│
└── scripts/
    ├── build.py                    # PyInstaller build script
    └── dev.py                      # Development server launcher
```

---

## Core Design

### Data Models (Pydantic-style, Param-based)

```python
# models/particle.py
import param
import numpy as np
from typing import Optional
from dataclasses import dataclass, field


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

    # Computed on demand
    _intensity_cps: Optional[float] = field(default=None, repr=False)
    _microtimes: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def dwell_time_s(self) -> float:
        return (self.end_time_ns - self.start_time_ns) / 1e9

    @property
    def intensity_cps(self) -> float:
        if self._intensity_cps is None:
            self._intensity_cps = self.num_photons / self.dwell_time_s
        return self._intensity_cps


@dataclass
class GroupData:
    """Grouped brightness levels."""
    id: int
    particle_id: int
    level_ids: list[int]
    intensity_cps: float
    total_dwell_time_s: float
    num_photons: int

    # Fit results (if fitted)
    tau: Optional[list[float]] = None
    amplitude: Optional[list[float]] = None
    avtau: Optional[float] = None
    chi_squared: Optional[float] = None


class SessionState(param.Parameterized):
    """Global application state using Param for reactivity."""

    # File state
    file_path = param.String(default="", doc="Path to loaded HDF5 file")
    file_loaded = param.Boolean(default=False)

    # Particle data
    particles = param.List(default=[], item_type=ParticleData)
    current_particle_id = param.Integer(default=-1)
    selected_particle_ids = param.List(default=[], item_type=int)

    # Analysis state
    levels = param.Dict(default={}, doc="particle_id -> list[LevelData]")
    groups = param.Dict(default={}, doc="particle_id -> list[GroupData]")

    # UI state
    bin_size_ms = param.Number(default=10.0, bounds=(0.1, 1000))
    confidence_level = param.Selector(
        default=0.95,
        objects=[0.69, 0.90, 0.95, 0.99]
    )

    # Processing state
    is_processing = param.Boolean(default=False)
    progress = param.Number(default=0, bounds=(0, 100))
    status_message = param.String(default="Ready")

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
```

### Base Controller Pattern

```python
# controllers/base.py
import param
import panel as pn
from abc import abstractmethod
from ..models.particle import SessionState


class BaseController(param.Parameterized):
    """
    Base class for domain controllers.

    Each controller manages one analysis domain (intensity, lifetime, etc.)
    and owns its associated Panel components.
    """

    session = param.ClassSelector(class_=SessionState)

    def __init__(self, session: SessionState, **params):
        super().__init__(session=session, **params)
        self._setup_watchers()

    def _setup_watchers(self):
        """Set up param watchers for reactive updates."""
        self.session.param.watch(
            self._on_particle_changed,
            ['current_particle_id']
        )

    def _on_particle_changed(self, event):
        """Called when current particle changes. Override in subclass."""
        pass

    @abstractmethod
    def panel(self) -> pn.viewable.Viewable:
        """Return the Panel component for this controller."""
        pass

    @abstractmethod
    def controls(self) -> pn.viewable.Viewable:
        """Return the control widgets for this controller."""
        pass
```

### Intensity Controller

```python
# controllers/intensity.py
import param
import panel as pn
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, BoxAnnotation
from concurrent.futures import ProcessPoolExecutor

from .base import BaseController
from ..analysis.change_point import ChangePointAnalysis
from ..models.particle import LevelData


class IntensityController(BaseController):
    """
    Controller for intensity trace analysis and change point detection.
    """

    # Display options
    show_levels = param.Boolean(default=True, doc="Show detected levels")
    show_groups = param.Boolean(default=False, doc="Show grouped levels")
    show_histogram = param.Boolean(default=True, doc="Show intensity histogram")

    # Analysis parameters
    min_photons = param.Integer(default=20, bounds=(5, 1000))
    min_boundary_offset = param.Integer(default=7, bounds=(2, 100))

    def __init__(self, session, **params):
        super().__init__(session=session, **params)

        # Bokeh data sources for efficient updates
        self._trace_source = ColumnDataSource(data={'t': [], 'counts': []})
        self._hist_source = ColumnDataSource(data={'counts': [], 'edges': []})
        self._level_annotations = []

        # Create plots
        self._intensity_plot = self._create_intensity_plot()
        self._histogram_plot = self._create_histogram_plot()

    def _create_intensity_plot(self):
        """Create the main intensity trace plot."""
        p = figure(
            title="Intensity Trace",
            x_axis_label="Time (s)",
            y_axis_label="Counts",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            sizing_mode="stretch_both",
            output_backend="webgl"  # WebGL for performance
        )

        p.line(
            't', 'counts',
            source=self._trace_source,
            line_width=1,
            color="#3b82f6"
        )

        return p

    def _create_histogram_plot(self):
        """Create the intensity histogram plot."""
        p = figure(
            title="Intensity Distribution",
            x_axis_label="Counts",
            y_axis_label="Frequency",
            tools="pan,wheel_zoom,reset",
            sizing_mode="stretch_both",
            output_backend="webgl"
        )

        p.quad(
            top='counts',
            bottom=0,
            left='left',
            right='right',
            source=self._hist_source,
            fill_color="#3b82f6",
            fill_alpha=0.6,
            line_color="white"
        )

        return p

    def _on_particle_changed(self, event):
        """Update plots when particle selection changes."""
        self._update_trace()
        self._update_levels()

    @param.depends('session.bin_size_ms', watch=True)
    def _update_trace(self):
        """Recompute and update intensity trace."""
        particle = self.session.current_particle
        if particle is None:
            self._trace_source.data = {'t': [], 'counts': []}
            return

        # Compute binned trace
        bin_size_ns = self.session.bin_size_ms * 1e6
        bins = np.arange(
            particle.abstimes[0],
            particle.abstimes[-1] + bin_size_ns,
            bin_size_ns
        )
        counts, edges = np.histogram(particle.abstimes, bins=bins)
        t = (edges[:-1] + edges[1:]) / 2 / 1e9  # Convert to seconds

        # Update data source (triggers plot refresh)
        self._trace_source.data = {'t': t.tolist(), 'counts': counts.tolist()}

        # Update histogram
        hist_counts, hist_edges = np.histogram(counts, bins=50)
        self._hist_source.data = {
            'counts': hist_counts.tolist(),
            'left': hist_edges[:-1].tolist(),
            'right': hist_edges[1:].tolist()
        }

    @param.depends('show_levels', 'show_groups', watch=True)
    def _update_levels(self):
        """Update level overlay annotations."""
        # Clear existing annotations
        for ann in self._level_annotations:
            if ann in self._intensity_plot.renderers:
                self._intensity_plot.renderers.remove(ann)
        self._level_annotations.clear()

        if not self.show_levels:
            return

        particle = self.session.current_particle
        if particle is None:
            return

        levels = self.session.get_levels(particle.id)

        for i, level in enumerate(levels):
            color = f"hsl({(i * 137) % 360}, 70%, 60%)"

            ann = BoxAnnotation(
                left=level.start_time_ns / 1e9,
                right=level.end_time_ns / 1e9,
                top=level.intensity_cps * self.session.bin_size_ms / 1000,
                bottom=0,
                fill_color=color,
                fill_alpha=0.3,
                line_color=None
            )
            self._intensity_plot.add_layout(ann)
            self._level_annotations.append(ann)

    async def resolve_levels(self, particle_ids: list[int]):
        """
        Run change point analysis on specified particles.
        Uses ProcessPoolExecutor for parallel execution.
        """
        self.session.is_processing = True
        self.session.status_message = "Detecting levels..."
        self.session.progress = 0

        try:
            # Prepare tasks
            tasks = []
            for pid in particle_ids:
                particle = next(p for p in self.session.particles if p.id == pid)
                tasks.append({
                    'particle_id': pid,
                    'abstimes': particle.abstimes,
                    'confidence': self.session.confidence_level,
                    'min_photons': self.min_photons,
                    'min_boundary_offset': self.min_boundary_offset
                })

            # Run in process pool
            loop = pn.state.curdoc.session_context.server_context.io_loop
            with ProcessPoolExecutor() as executor:
                futures = [
                    loop.run_in_executor(executor, _run_cpa_task, task)
                    for task in tasks
                ]

                completed = 0
                for future in futures:
                    result = await future
                    completed += 1

                    # Update state with results
                    levels = [LevelData(**ld) for ld in result['levels']]
                    self.session.levels = {
                        **self.session.levels,
                        result['particle_id']: levels
                    }

                    # Update progress
                    self.session.progress = (completed / len(tasks)) * 100
                    self.session.status_message = f"Resolved {completed}/{len(tasks)} particles"

            self._update_levels()

        finally:
            self.session.is_processing = False
            self.session.status_message = "Ready"
            self.session.progress = 0

    def panel(self) -> pn.viewable.Viewable:
        """Return the main panel layout."""
        return pn.Column(
            pn.pane.Bokeh(self._intensity_plot, sizing_mode="stretch_both"),
            pn.Row(
                pn.pane.Bokeh(self._histogram_plot, sizing_mode="stretch_both"),
                visible=self.param.show_histogram,
                sizing_mode="stretch_width",
                height=200
            ),
            sizing_mode="stretch_both"
        )

    def controls(self) -> pn.viewable.Viewable:
        """Return control widgets."""
        return pn.Column(
            pn.pane.Markdown("### Intensity Analysis"),
            pn.widgets.FloatSlider.from_param(
                self.session.param.bin_size_ms,
                name="Bin Size (ms)",
                step=0.1
            ),
            pn.widgets.Select.from_param(
                self.session.param.confidence_level,
                name="Confidence Level"
            ),
            pn.widgets.Checkbox.from_param(self.param.show_levels, name="Show Levels"),
            pn.widgets.Checkbox.from_param(self.param.show_groups, name="Show Groups"),
            pn.widgets.Checkbox.from_param(self.param.show_histogram, name="Show Histogram"),
            pn.layout.Divider(),
            pn.widgets.Button(
                name="Resolve Current",
                button_type="primary",
                on_click=lambda e: self._resolve_current()
            ),
            pn.widgets.Button(
                name="Resolve Selected",
                button_type="default",
                on_click=lambda e: self._resolve_selected()
            ),
            pn.widgets.Button(
                name="Resolve All",
                button_type="default",
                on_click=lambda e: self._resolve_all()
            ),
            sizing_mode="stretch_width"
        )

    def _resolve_current(self):
        if self.session.current_particle_id >= 0:
            pn.state.execute(
                lambda: self.resolve_levels([self.session.current_particle_id])
            )

    def _resolve_selected(self):
        if self.session.selected_particle_ids:
            pn.state.execute(
                lambda: self.resolve_levels(self.session.selected_particle_ids)
            )

    def _resolve_all(self):
        all_ids = [p.id for p in self.session.particles]
        if all_ids:
            pn.state.execute(lambda: self.resolve_levels(all_ids))


def _run_cpa_task(task: dict) -> dict:
    """
    Worker function for change point analysis.
    Runs in separate process via ProcessPoolExecutor.
    """
    from ..analysis.change_point import ChangePointAnalysis

    cpa = ChangePointAnalysis()
    levels = cpa.find_change_points(
        abstimes=task['abstimes'],
        confidence=task['confidence'],
        min_photons=task['min_photons'],
        min_boundary_offset=task['min_boundary_offset']
    )

    return {
        'particle_id': task['particle_id'],
        'levels': [
            {
                'id': i,
                'particle_id': task['particle_id'],
                'start_idx': lv.start_idx,
                'end_idx': lv.end_idx,
                'start_time_ns': lv.start_time_ns,
                'end_time_ns': lv.end_time_ns,
                'num_photons': lv.num_photons
            }
            for i, lv in enumerate(levels)
        ]
    }
```

### Lifetime Controller

```python
# controllers/lifetime.py
import param
import panel as pn
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

from .base import BaseController
from ..analysis.lifetime import FluoFit, OneExp, TwoExp, ThreeExp


class LifetimeController(BaseController):
    """
    Controller for fluorescence decay and lifetime fitting.
    """

    # Display options
    show_fit = param.Boolean(default=True, doc="Show fitted curve")
    show_residuals = param.Boolean(default=True, doc="Show fit residuals")
    show_irf = param.Boolean(default=False, doc="Show IRF overlay")
    log_scale = param.Boolean(default=True, doc="Logarithmic Y axis")

    # Fit parameters
    num_exponentials = param.Selector(
        default=1,
        objects=[1, 2, 3],
        doc="Number of exponential components"
    )
    use_irf = param.Boolean(default=True, doc="Use IRF in fitting")

    # Fit results (read-only display)
    tau_values = param.List(default=[], doc="Fitted lifetimes (ns)")
    amplitude_values = param.List(default=[], doc="Fitted amplitudes")
    chi_squared = param.Number(default=0.0, doc="Chi-squared value")
    avg_lifetime = param.Number(default=0.0, doc="Amplitude-weighted average")

    def __init__(self, session, **params):
        super().__init__(session=session, **params)

        # Data sources
        self._decay_source = ColumnDataSource(data={'t': [], 'counts': []})
        self._fit_source = ColumnDataSource(data={'t': [], 'counts': []})
        self._residuals_source = ColumnDataSource(data={'t': [], 'residuals': []})
        self._irf_source = ColumnDataSource(data={'t': [], 'counts': []})

        # Create plots
        self._decay_plot = self._create_decay_plot()
        self._residuals_plot = self._create_residuals_plot()

    def _create_decay_plot(self):
        """Create the decay histogram plot."""
        p = figure(
            title="Fluorescence Decay",
            x_axis_label="Time (ns)",
            y_axis_label="Counts",
            y_axis_type="log" if self.log_scale else "linear",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            sizing_mode="stretch_both",
            output_backend="webgl"
        )

        # Data histogram
        p.line(
            't', 'counts',
            source=self._decay_source,
            line_width=1,
            color="#3b82f6",
            legend_label="Data"
        )

        # Fit curve
        p.line(
            't', 'counts',
            source=self._fit_source,
            line_width=2,
            color="#ef4444",
            legend_label="Fit"
        )

        # IRF
        p.line(
            't', 'counts',
            source=self._irf_source,
            line_width=1,
            color="#10b981",
            line_dash="dashed",
            legend_label="IRF",
            visible=self.show_irf
        )

        p.legend.click_policy = "hide"

        return p

    def _create_residuals_plot(self):
        """Create the residuals plot."""
        p = figure(
            title="Residuals",
            x_axis_label="Time (ns)",
            y_axis_label="Weighted Residuals",
            tools="pan,wheel_zoom,reset",
            sizing_mode="stretch_both",
            height=150
        )

        p.line(
            't', 'residuals',
            source=self._residuals_source,
            line_width=1,
            color="#6b7280"
        )

        # Zero line
        p.line([0, 100], [0, 0], line_dash="dashed", color="#9ca3af")

        return p

    def _on_particle_changed(self, event):
        """Update decay plot when particle changes."""
        self._update_decay()

    def _update_decay(self):
        """Update the decay histogram display."""
        particle = self.session.current_particle
        if particle is None:
            self._decay_source.data = {'t': [], 'counts': []}
            return

        # Build histogram from microtimes
        channelwidth = particle.channelwidth
        max_time = particle.microtimes.max()
        bins = np.arange(0, max_time + channelwidth, channelwidth)
        counts, edges = np.histogram(particle.microtimes, bins=bins)
        t = (edges[:-1] + edges[1:]) / 2

        # Filter out zero counts for log scale
        mask = counts > 0
        self._decay_source.data = {
            't': t[mask].tolist(),
            'counts': counts[mask].tolist()
        }

    @param.depends('log_scale', watch=True)
    def _update_log_scale(self):
        """Toggle log/linear scale."""
        self._decay_plot.y_axis_type = "log" if self.log_scale else "linear"

    async def fit_decay(
        self,
        target_type: str = "particle",
        target_ids: list[int] = None
    ):
        """
        Fit fluorescence decay for specified targets.

        Args:
            target_type: "particle", "level", or "group"
            target_ids: List of IDs to fit
        """
        self.session.is_processing = True
        self.session.status_message = "Fitting lifetimes..."

        try:
            # Select fitting class based on number of exponentials
            fit_classes = {1: OneExp, 2: TwoExp, 3: ThreeExp}
            FitClass = fit_classes[self.num_exponentials]

            # Get histogram data
            particle = self.session.current_particle
            if particle is None:
                return

            # Build histogram
            channelwidth = particle.channelwidth
            max_time = particle.microtimes.max()
            bins = np.arange(0, max_time + channelwidth, channelwidth)
            counts, edges = np.histogram(particle.microtimes, bins=bins)
            t = (edges[:-1] + edges[1:]) / 2

            # Create fitter
            fitter = FitClass(
                t=t,
                decay=counts,
                channelwidth=channelwidth,
                irf=None,  # TODO: load from session
                use_irf=self.use_irf
            )

            # Run fit
            result = fitter.fit()

            # Update display
            self.tau_values = result['tau']
            self.amplitude_values = result['amplitude']
            self.chi_squared = result['chi_squared']
            self.avg_lifetime = result.get('avtau', self.tau_values[0])

            # Update fit curve source
            fit_curve = fitter.model(t, *result['params'])
            self._fit_source.data = {
                't': t.tolist(),
                'counts': fit_curve.tolist()
            }

            # Update residuals
            residuals = (counts - fit_curve) / np.sqrt(np.maximum(counts, 1))
            self._residuals_source.data = {
                't': t.tolist(),
                'residuals': residuals.tolist()
            }

        finally:
            self.session.is_processing = False
            self.session.status_message = "Ready"

    def panel(self) -> pn.viewable.Viewable:
        """Return the main panel layout."""
        return pn.Column(
            pn.pane.Bokeh(self._decay_plot, sizing_mode="stretch_both"),
            pn.Row(
                pn.pane.Bokeh(self._residuals_plot, sizing_mode="stretch_both"),
                visible=self.param.show_residuals,
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_both"
        )

    def controls(self) -> pn.viewable.Viewable:
        """Return control widgets."""
        return pn.Column(
            pn.pane.Markdown("### Lifetime Fitting"),
            pn.widgets.Select.from_param(
                self.param.num_exponentials,
                name="Exponentials"
            ),
            pn.widgets.Checkbox.from_param(self.param.use_irf, name="Use IRF"),
            pn.widgets.Checkbox.from_param(self.param.log_scale, name="Log Scale"),
            pn.widgets.Checkbox.from_param(self.param.show_fit, name="Show Fit"),
            pn.widgets.Checkbox.from_param(self.param.show_residuals, name="Show Residuals"),
            pn.widgets.Checkbox.from_param(self.param.show_irf, name="Show IRF"),
            pn.layout.Divider(),
            pn.pane.Markdown("### Fit Results"),
            pn.widgets.StaticText(
                value=self.param.tau_values,
                name="Tau (ns)"
            ),
            pn.widgets.StaticText(
                value=self.param.chi_squared,
                name="Chi-squared"
            ),
            pn.widgets.StaticText(
                value=self.param.avg_lifetime,
                name="Avg Lifetime (ns)"
            ),
            pn.layout.Divider(),
            pn.widgets.Button(
                name="Fit Current",
                button_type="primary",
                on_click=lambda e: pn.state.execute(self.fit_decay)
            ),
            sizing_mode="stretch_width"
        )
```

---

## Panel Component Design

### Main Application Layout

```python
# app.py
import panel as pn
import param
from pathlib import Path

from .models.particle import SessionState
from .controllers.intensity import IntensityController
from .controllers.lifetime import LifetimeController
from .controllers.grouping import GroupingController
from .controllers.spectra import SpectraController
from .controllers.raster import RasterController
from .controllers.antibunching import AntibunchingController
from .controllers.filtering import FilteringController
from .components.widgets.particle_tree import ParticleTree
from .components.widgets.progress_indicator import ProgressIndicator
from .io.hdf5_reader import load_hdf5_file


pn.extension(sizing_mode="stretch_width")


class FullSMSApp(param.Parameterized):
    """
    Main Full SMS application container.
    """

    def __init__(self, **params):
        super().__init__(**params)

        # Create global session state
        self.session = SessionState()

        # Create controllers
        self.intensity_ctrl = IntensityController(session=self.session)
        self.lifetime_ctrl = LifetimeController(session=self.session)
        self.grouping_ctrl = GroupingController(session=self.session)
        self.spectra_ctrl = SpectraController(session=self.session)
        self.raster_ctrl = RasterController(session=self.session)
        self.antibunching_ctrl = AntibunchingController(session=self.session)
        self.filtering_ctrl = FilteringController(session=self.session)

        # Create shared widgets
        self.particle_tree = ParticleTree(session=self.session)
        self.progress = ProgressIndicator(session=self.session)

        # File picker
        self.file_picker = pn.widgets.FileInput(
            accept=".h5,.hdf5",
            multiple=False
        )
        self.file_picker.param.watch(self._on_file_selected, 'value')

    def _on_file_selected(self, event):
        """Handle file selection."""
        if event.new is None:
            return

        # Save to temp and load
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as f:
            f.write(event.new)
            temp_path = f.name

        pn.state.execute(lambda: self._load_file(temp_path))

    async def _load_file(self, path: str):
        """Load HDF5 file asynchronously."""
        self.session.is_processing = True
        self.session.status_message = "Loading file..."

        try:
            particles = await load_hdf5_file(path)
            self.session.particles = particles
            self.session.file_path = path
            self.session.file_loaded = True

            if particles:
                self.session.current_particle_id = particles[0].id

            self.session.status_message = f"Loaded {len(particles)} particles"

        except Exception as e:
            self.session.status_message = f"Error: {e}"
            raise
        finally:
            self.session.is_processing = False

    def sidebar(self) -> pn.viewable.Viewable:
        """Build the sidebar layout."""
        return pn.Column(
            pn.pane.Markdown("## Full SMS"),
            self.file_picker,
            pn.layout.Divider(),
            pn.pane.Markdown("### Particles"),
            self.particle_tree.panel(),
            pn.layout.Divider(),
            # Show controls for active tab
            self._active_controls,
            sizing_mode="stretch_both",
            width=300
        )

    @param.depends('_active_tab')
    def _active_controls(self):
        """Return controls for currently active tab."""
        controllers = {
            0: self.intensity_ctrl,
            1: self.lifetime_ctrl,
            2: self.grouping_ctrl,
            3: self.spectra_ctrl,
            4: self.raster_ctrl,
            5: self.antibunching_ctrl,
            6: self.filtering_ctrl
        }
        ctrl = controllers.get(self._active_tab, self.intensity_ctrl)
        return ctrl.controls()

    _active_tab = param.Integer(default=0)

    def main_content(self) -> pn.viewable.Viewable:
        """Build the main content area with tabs."""
        tabs = pn.Tabs(
            ("Intensity", self.intensity_ctrl.panel()),
            ("Lifetime", self.lifetime_ctrl.panel()),
            ("Grouping", self.grouping_ctrl.panel()),
            ("Spectra", self.spectra_ctrl.panel()),
            ("Raster Scan", self.raster_ctrl.panel()),
            ("Antibunching", self.antibunching_ctrl.panel()),
            ("Filtering", self.filtering_ctrl.panel()),
            sizing_mode="stretch_both",
            dynamic=True  # Only render active tab
        )

        # Track active tab for controls
        tabs.param.watch(
            lambda e: setattr(self, '_active_tab', e.new),
            'active'
        )

        return tabs

    def status_bar(self) -> pn.viewable.Viewable:
        """Build the status bar."""
        return pn.Row(
            pn.pane.Str(
                object=self.session.param.status_message,
                sizing_mode="stretch_width"
            ),
            self.progress.panel(),
            sizing_mode="stretch_width",
            height=30
        )

    def view(self) -> pn.viewable.Viewable:
        """Build the complete application view."""
        return pn.template.FastListTemplate(
            title="Full SMS",
            sidebar=[self.sidebar()],
            main=[
                pn.Column(
                    self.main_content(),
                    self.status_bar(),
                    sizing_mode="stretch_both"
                )
            ],
            sidebar_width=320,
            accent_base_color="#3b82f6",
            header_background="#1e293b"
        )


def create_app():
    """Factory function for creating the Panel app."""
    app = FullSMSApp()
    return app.view()


# For Panel serve
if __name__.startswith("bokeh"):
    create_app().servable()
```

### Particle Tree Widget

```python
# components/widgets/particle_tree.py
import param
import panel as pn
from ...models.particle import SessionState


class ParticleTree(param.Parameterized):
    """
    Hierarchical particle selection widget.

    Uses Panel's Tabulator for efficient virtualized list rendering.
    """

    session = param.ClassSelector(class_=SessionState)

    # Selection state
    select_all = param.Boolean(default=False)

    def __init__(self, session: SessionState, **params):
        super().__init__(session=session, **params)
        self._table = None

    @param.depends('session.particles')
    def _particle_table_data(self):
        """Generate table data from particles."""
        import pandas as pd

        if not self.session.particles:
            return pd.DataFrame(columns=['id', 'description', 'photons', 'selected'])

        data = []
        for p in self.session.particles:
            data.append({
                'id': p.id,
                'description': p.description,
                'photons': f"{p.num_photons:,}",
                'time_s': f"{p.measurement_time_s:.1f}",
                'has_levels': p.id in self.session.levels,
                'selected': p.id in self.session.selected_particle_ids
            })

        return pd.DataFrame(data)

    def panel(self) -> pn.viewable.Viewable:
        """Return the particle tree panel."""
        df = self._particle_table_data()

        self._table = pn.widgets.Tabulator(
            df,
            show_index=False,
            selectable='checkbox',
            selection=[],
            pagination='local',
            page_size=50,
            sizing_mode="stretch_both",
            height=400,
            configuration={
                'columns': [
                    {'field': 'id', 'title': '#', 'width': 40},
                    {'field': 'description', 'title': 'Description'},
                    {'field': 'photons', 'title': 'Photons', 'width': 80},
                    {'field': 'has_levels', 'title': 'Lvl', 'width': 40,
                     'formatter': 'tickCross'}
                ]
            }
        )

        # Handle selection changes
        self._table.param.watch(self._on_selection, 'selection')

        # Handle row click for current particle
        self._table.on_click(self._on_row_click)

        return pn.Column(
            pn.Row(
                pn.widgets.Checkbox.from_param(
                    self.param.select_all,
                    name="Select All"
                ),
                pn.widgets.Button(
                    name="Clear",
                    button_type="light",
                    on_click=lambda e: self._clear_selection()
                ),
                sizing_mode="stretch_width"
            ),
            self._table,
            sizing_mode="stretch_both"
        )

    def _on_selection(self, event):
        """Handle multi-selection via checkboxes."""
        if self._table is None:
            return

        df = self._table.value
        selected_ids = df.iloc[event.new]['id'].tolist()
        self.session.selected_particle_ids = selected_ids

    def _on_row_click(self, event):
        """Handle single row click for current particle."""
        if event.row is not None:
            particle_id = self._table.value.iloc[event.row]['id']
            self.session.current_particle_id = particle_id

    @param.depends('select_all', watch=True)
    def _toggle_select_all(self):
        """Select/deselect all particles."""
        if self._table is None:
            return

        if self.select_all:
            self._table.selection = list(range(len(self._table.value)))
        else:
            self._table.selection = []

    def _clear_selection(self):
        """Clear all selections."""
        self.select_all = False
        self.session.selected_particle_ids = []
        if self._table:
            self._table.selection = []
```

### Progress Indicator

```python
# components/widgets/progress_indicator.py
import param
import panel as pn
from ...models.particle import SessionState


class ProgressIndicator(param.Parameterized):
    """
    Progress bar and status indicator.
    """

    session = param.ClassSelector(class_=SessionState)

    def __init__(self, session: SessionState, **params):
        super().__init__(session=session, **params)

    def panel(self) -> pn.viewable.Viewable:
        """Return the progress indicator panel."""
        progress_bar = pn.indicators.Progress(
            value=int(self.session.progress),
            max=100,
            bar_color='primary',
            sizing_mode='stretch_width',
            visible=self.session.param.is_processing
        )

        # Bind progress value
        self.session.param.watch(
            lambda e: setattr(progress_bar, 'value', int(e.new)),
            'progress'
        )

        return pn.Row(
            progress_bar,
            width=200
        )
```

### Fitting Dialog

```python
# components/widgets/fitting_dialog.py
import param
import panel as pn


class FittingDialog(param.Parameterized):
    """
    Modal dialog for configuring lifetime fitting parameters.
    """

    # Fitting parameters
    num_exponentials = param.Selector(default=1, objects=[1, 2, 3])
    use_irf = param.Boolean(default=True)
    irf_shift = param.Number(default=0.0, bounds=(-10, 10))

    # Fitting range
    auto_range = param.Boolean(default=True)
    start_channel = param.Integer(default=0, bounds=(0, 10000))
    end_channel = param.Integer(default=1000, bounds=(1, 10000))

    # Initial guesses (for multi-exponential)
    tau1_guess = param.Number(default=2.0, bounds=(0.01, 100))
    tau2_guess = param.Number(default=5.0, bounds=(0.01, 100))
    tau3_guess = param.Number(default=10.0, bounds=(0.01, 100))

    # Background
    bg_correction = param.Boolean(default=True)
    bg_channels = param.Integer(default=50, bounds=(10, 500))

    # Result callback
    on_fit = param.Callable(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        self._dialog = None

    def panel(self) -> pn.viewable.Viewable:
        """Build the dialog content."""
        # Initial guesses section (conditional on num_exponentials)
        guesses = pn.Column(
            pn.widgets.FloatInput.from_param(self.param.tau1_guess, name="Tau 1 (ns)"),
            pn.widgets.FloatInput.from_param(
                self.param.tau2_guess,
                name="Tau 2 (ns)",
                visible=self.num_exponentials >= 2
            ),
            pn.widgets.FloatInput.from_param(
                self.param.tau3_guess,
                name="Tau 3 (ns)",
                visible=self.num_exponentials >= 3
            ),
        )

        return pn.Column(
            pn.pane.Markdown("## Fitting Parameters"),
            pn.layout.Divider(),

            pn.pane.Markdown("### Model"),
            pn.widgets.RadioButtonGroup.from_param(
                self.param.num_exponentials,
                name="Exponentials"
            ),

            pn.pane.Markdown("### IRF"),
            pn.widgets.Checkbox.from_param(self.param.use_irf, name="Use IRF"),
            pn.widgets.FloatSlider.from_param(
                self.param.irf_shift,
                name="IRF Shift",
                step=0.1
            ),

            pn.pane.Markdown("### Fit Range"),
            pn.widgets.Checkbox.from_param(self.param.auto_range, name="Auto Range"),
            pn.widgets.IntInput.from_param(
                self.param.start_channel,
                name="Start Channel",
                disabled=self.param.auto_range
            ),
            pn.widgets.IntInput.from_param(
                self.param.end_channel,
                name="End Channel",
                disabled=self.param.auto_range
            ),

            pn.pane.Markdown("### Initial Guesses"),
            guesses,

            pn.pane.Markdown("### Background"),
            pn.widgets.Checkbox.from_param(self.param.bg_correction, name="Correction"),
            pn.widgets.IntSlider.from_param(
                self.param.bg_channels,
                name="Channels for BG"
            ),

            pn.layout.Divider(),
            pn.Row(
                pn.widgets.Button(
                    name="Cancel",
                    button_type="light",
                    on_click=lambda e: self.close()
                ),
                pn.widgets.Button(
                    name="Fit",
                    button_type="primary",
                    on_click=lambda e: self._do_fit()
                )
            ),

            width=400
        )

    def show(self):
        """Display the dialog as a modal."""
        self._dialog = pn.Column(
            self.panel(),
            css_classes=['fitting-dialog'],
            sizing_mode="fixed"
        )
        # In a real implementation, use Panel's modal/overlay system
        return self._dialog

    def close(self):
        """Close the dialog."""
        if self._dialog:
            self._dialog.visible = False

    def _do_fit(self):
        """Execute fit and close dialog."""
        if self.on_fit:
            self.on_fit(self.get_parameters())
        self.close()

    def get_parameters(self) -> dict:
        """Get all fitting parameters as a dictionary."""
        return {
            'num_exponentials': self.num_exponentials,
            'use_irf': self.use_irf,
            'irf_shift': self.irf_shift,
            'auto_range': self.auto_range,
            'start_channel': self.start_channel,
            'end_channel': self.end_channel,
            'tau_guesses': [self.tau1_guess, self.tau2_guess, self.tau3_guess][:self.num_exponentials],
            'bg_correction': self.bg_correction,
            'bg_channels': self.bg_channels
        }
```

---

## Desktop Packaging

### PyWebView Integration

```python
# scripts/desktop.py
"""
Desktop application entry point using PyWebView.
"""
import webview
import threading
import sys
from pathlib import Path


def start_panel_server():
    """Start the Panel server in a background thread."""
    import panel as pn
    from full_sms.app import create_app

    # Find an available port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    # Start Panel server
    pn.serve(
        create_app,
        port=port,
        show=False,
        threaded=True,
        title="Full SMS"
    )

    return port


def main():
    """Main entry point for desktop application."""
    # Start Panel server
    port = start_panel_server()

    # Wait for server to be ready
    import time
    import urllib.request

    for _ in range(30):  # 3 second timeout
        try:
            urllib.request.urlopen(f"http://localhost:{port}")
            break
        except:
            time.sleep(0.1)

    # Create native window
    window = webview.create_window(
        title="Full SMS - Single-Molecule Spectroscopy",
        url=f"http://localhost:{port}",
        width=1400,
        height=900,
        min_size=(1024, 768),
        background_color="#1e293b"
    )

    # Enable file dialogs via pywebview API
    def open_file_dialog():
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=('HDF5 Files (*.h5;*.hdf5)', 'All files (*.*)')
        )
        return result[0] if result else None

    def save_file_dialog(default_name="analysis.smsa"):
        result = window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=default_name,
            file_types=('SMS Analysis (*.smsa)', 'All files (*.*)')
        )
        return result if result else None

    # Expose to JavaScript
    window.expose(open_file_dialog)
    window.expose(save_file_dialog)

    # Start the application
    webview.start(debug=('--debug' in sys.argv))


if __name__ == "__main__":
    main()
```

### PyInstaller Build Configuration

```python
# build.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Collect all package data
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []
datas += collect_data_files('panel')
datas += collect_data_files('bokeh')
datas += collect_data_files('full_sms')

hiddenimports = []
hiddenimports += collect_submodules('panel')
hiddenimports += collect_submodules('bokeh')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('h5py')

a = Analysis(
    ['scripts/desktop.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    console=False,  # No console window
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
"""
Cross-platform build script for Full SMS.
"""
import subprocess
import sys
import shutil
from pathlib import Path


def clean():
    """Remove build artifacts."""
    dirs_to_remove = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_remove:
        for path in Path('.').glob(pattern):
            shutil.rmtree(path, ignore_errors=True)


def build():
    """Build the application."""
    clean()

    # Run PyInstaller
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
"""
Background processing infrastructure using ProcessPoolExecutor.
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Callable, TypeVar, Any
from dataclasses import dataclass
from uuid import UUID, uuid4
import multiprocessing as mp
import os


T = TypeVar('T')


@dataclass
class TaskResult:
    """Result from a background task."""
    task_id: UUID
    success: bool
    result: Any = None
    error: str = None


class AnalysisPool:
    """
    Manages a persistent process pool for CPU-intensive analysis.

    Designed for Panel's async context - tasks are submitted and
    results are awaited without blocking the event loop.
    """

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or max(1, os.cpu_count() - 1)
        self._executor: ProcessPoolExecutor = None
        self._active_tasks: dict[UUID, Future] = {}

    def _ensure_pool(self):
        """Lazily create the process pool."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context('spawn')  # Safer for GUI apps
            )

    async def submit(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Submit a single task and await its result.

        Args:
            func: Function to execute (must be picklable)
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result
        """
        self._ensure_pool()

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self._executor,
            func,
            *args
        )

        return await future

    async def map(
        self,
        func: Callable[..., T],
        items: list,
        progress_callback: Callable[[int, int, T], None] = None
    ) -> list[T]:
        """
        Map function over items in parallel with progress tracking.

        Args:
            func: Function to execute on each item
            items: List of items to process
            progress_callback: Optional callback(completed, total, result)

        Returns:
            List of results in same order as input
        """
        self._ensure_pool()

        if not items:
            return []

        loop = asyncio.get_event_loop()
        total = len(items)

        # Submit all tasks
        futures = [
            (i, loop.run_in_executor(self._executor, func, item))
            for i, item in enumerate(items)
        ]

        # Collect results maintaining order
        results = [None] * total
        completed = 0

        for idx, future in futures:
            result = await future
            results[idx] = result
            completed += 1

            if progress_callback:
                progress_callback(completed, total, result)

        return results

    async def map_unordered(
        self,
        func: Callable[..., T],
        items: list,
        progress_callback: Callable[[int, int, T], None] = None
    ) -> list[T]:
        """
        Map function over items, returning results as they complete.

        More efficient when order doesn't matter.
        """
        self._ensure_pool()

        if not items:
            return []

        loop = asyncio.get_event_loop()
        total = len(items)

        # Submit all tasks
        futures = [
            loop.run_in_executor(self._executor, func, item)
            for item in items
        ]

        # Collect results as they complete
        results = []
        completed = 0

        for coro in asyncio.as_completed(futures):
            result = await coro
            results.append(result)
            completed += 1

            if progress_callback:
                progress_callback(completed, total, result)

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
"""
Picklable task functions for background processing.

These functions run in separate processes via ProcessPoolExecutor.
They must be module-level functions (not methods) to be picklable.
"""
import numpy as np
from typing import Any


def run_change_point_analysis(
    abstimes: np.ndarray,
    confidence: float,
    min_photons: int,
    min_boundary_offset: int
) -> list[dict]:
    """
    Run change point analysis on photon arrival times.

    Args:
        abstimes: Absolute photon arrival times in nanoseconds
        confidence: Confidence level (e.g., 0.95)
        min_photons: Minimum photons per level
        min_boundary_offset: Minimum separation between change points

    Returns:
        List of level dictionaries with start/end indices and times
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


def run_ahca_clustering(
    levels_data: list[dict],
    use_lifetime: bool = False
) -> dict:
    """
    Run agglomerative hierarchical clustering on levels.

    Args:
        levels_data: List of level dictionaries with intensities
        use_lifetime: Whether to include lifetime in distance metric

    Returns:
        Clustering result with groups and BIC values
    """
    from ..analysis.clustering import AHCA

    ahca = AHCA()
    result = ahca.cluster(levels_data, use_lifetime=use_lifetime)

    return {
        'groups': result.groups,
        'bic_values': result.bic_values,
        'best_num_groups': result.best_num_groups,
        'level_group_assignments': result.assignments
    }


def run_lifetime_fit(
    microtimes: np.ndarray,
    channelwidth: float,
    irf: np.ndarray,
    num_exponentials: int,
    **fit_params
) -> dict:
    """
    Fit fluorescence decay with multi-exponential model.

    Args:
        microtimes: TCSPC microtimes in nanoseconds
        channelwidth: Channel width in nanoseconds
        irf: Instrument response function array
        num_exponentials: Number of exponential components (1-3)
        **fit_params: Additional fitting parameters

    Returns:
        Fit results including tau, amplitude, chi-squared
    """
    from ..analysis.lifetime import OneExp, TwoExp, ThreeExp

    # Build histogram
    max_time = microtimes.max()
    bins = np.arange(0, max_time + channelwidth, channelwidth)
    counts, edges = np.histogram(microtimes, bins=bins)
    t = (edges[:-1] + edges[1:]) / 2

    # Select fitter
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


def run_correlation(
    abstimes_primary: np.ndarray,
    abstimes_secondary: np.ndarray,
    window_ns: float,
    bin_size_ns: float
) -> dict:
    """
    Calculate second-order photon correlation (antibunching).

    Args:
        abstimes_primary: Primary channel arrival times
        abstimes_secondary: Secondary channel arrival times
        window_ns: Correlation window in nanoseconds
        bin_size_ns: Histogram bin size in nanoseconds

    Returns:
        Correlation histogram with tau and g2 values
    """
    from ..analysis.correlation import calculate_g2

    tau, g2 = calculate_g2(
        abstimes_primary,
        abstimes_secondary,
        window=window_ns,
        binsize=bin_size_ns
    )

    return {
        'tau': tau.tolist(),
        'g2': g2.tolist()
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
┌─────────────────┐      ┌─────────────────┐
│ PyWebView       │      │ Panel Server    │
│ file_dialog()   │─────>│ receives path   │
└─────────────────┘      └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ SessionState    │
                         │ is_processing   │
                         │ = True          │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ load_hdf5_file  │
                         │ (async)         │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ h5py reads:     │
                         │ - particles     │
                         │ - abstimes      │
                         │ - microtimes    │
                         │ - IRF, spectra  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ SessionState    │
                         │ .particles      │
                         │ updated         │
                         └────────┬────────┘
                                  │
                                  ▼ (param triggers)
                         ┌─────────────────┐
                         │ ParticleTree    │
                         │ refreshes       │
                         └─────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ IntensityCtrl   │
                         │ plots first     │
                         │ particle        │
                         └─────────────────┘
```

### Analysis Sequence (Change Point Detection)

```
┌─────────────────┐
│ User clicks     │
│ "Resolve All"   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│ IntensityCtrl   │─────>│ SessionState    │
│ resolve_levels  │      │ is_processing   │
│ (async)         │      │ = True          │
└────────┬────────┘      └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Prepare tasks:  │
│ [(pid, abstimes,│
│   params), ...] │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│ AnalysisPool    │      │ Worker Process  │
│ .map_unordered  │─────>│ run_cpa_task    │
│                 │      │ (Numba/NumPy)   │
└────────┬────────┘      └────────┬────────┘
         │                        │
         │◄───────────────────────┘
         │  result for particle N
         ▼
┌─────────────────┐      ┌─────────────────┐
│ progress_       │─────>│ SessionState    │
│ callback        │      │ .progress       │
│                 │      │ .status_message │
└────────┬────────┘      └─────────────────┘
         │
         ▼ (for each result)
┌─────────────────┐
│ SessionState    │
│ .levels[pid]    │
│ updated         │
└────────┬────────┘
         │
         ▼ (param triggers)
┌─────────────────┐
│ IntensityCtrl   │
│ _update_levels  │
│ (Bokeh redraws) │
└─────────────────┘
```

---

## State Management

### Param-based Reactive State

The application uses the `param` library for reactive state management. This provides:

1. **Type Validation**: Parameters are typed and validated on assignment
2. **Reactive Updates**: `param.depends` decorators trigger updates when dependencies change
3. **Serialization**: Parameters can be serialized for save/load
4. **No Boilerplate**: Unlike React/Redux, no action creators or reducers needed

### State Hierarchy

```
SessionState (global)
├── file_path, file_loaded
├── particles: list[ParticleData]
├── current_particle_id
├── selected_particle_ids
├── levels: dict[int, list[LevelData]]
├── groups: dict[int, list[GroupData]]
├── bin_size_ms, confidence_level
├── is_processing, progress, status_message
│
├── IntensityController
│   ├── show_levels, show_groups, show_histogram
│   ├── min_photons, min_boundary_offset
│   └── _trace_source, _hist_source (Bokeh)
│
├── LifetimeController
│   ├── show_fit, show_residuals, show_irf
│   ├── num_exponentials, use_irf
│   ├── tau_values, chi_squared (results)
│   └── _decay_source, _fit_source (Bokeh)
│
├── GroupingController
│   ├── use_lifetime_in_clustering
│   ├── selected_num_groups
│   └── _bic_source (Bokeh)
│
└── [Other controllers...]
```

### Watcher Pattern Example

```python
class IntensityController(BaseController):

    # When bin_size_ms changes, recompute trace
    @param.depends('session.bin_size_ms', watch=True)
    def _update_trace(self):
        # This runs automatically when bin_size_ms changes
        particle = self.session.current_particle
        if particle is None:
            return

        # Recompute binned trace
        t, counts = self._compute_binned_trace(particle)

        # Update Bokeh source (triggers plot refresh)
        self._trace_source.data = {'t': t, 'counts': counts}

    # When current particle changes, update everything
    @param.depends('session.current_particle_id', watch=True)
    def _on_particle_changed(self):
        self._update_trace()
        self._update_levels()
        self._update_histogram()
```

### Session Persistence

```python
# io/session.py
import pickle
import json
from pathlib import Path
from dataclasses import asdict
from typing import Any

from ..models.particle import SessionState, ParticleData, LevelData, GroupData


def save_session(session: SessionState, path: Path):
    """
    Save session state to .smsa file.

    Format: JSON for metadata + pickled numpy arrays
    """
    data = {
        'version': '2.0.0',
        'h5_file_path': session.file_path,
        'current_particle_id': session.current_particle_id,
        'selected_particle_ids': session.selected_particle_ids,
        'bin_size_ms': session.bin_size_ms,
        'confidence_level': session.confidence_level,

        # Analysis results (without numpy arrays)
        'levels': {
            str(pid): [asdict(lv) for lv in levels]
            for pid, levels in session.levels.items()
        },
        'groups': {
            str(pid): [asdict(g) for g in groups]
            for pid, groups in session.groups.items()
        }
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_session(path: Path) -> dict:
    """
    Load session state from .smsa file.

    Returns dict that can be used to restore SessionState.
    """
    with open(path) as f:
        data = json.load(f)

    # Validate version
    version = data.get('version', '1.0')
    if version.startswith('1.'):
        # Handle legacy pickle format
        return _migrate_v1_session(path)

    # Convert levels back to LevelData
    levels = {}
    for pid_str, level_dicts in data.get('levels', {}).items():
        pid = int(pid_str)
        levels[pid] = [LevelData(**ld) for ld in level_dicts]

    # Convert groups back to GroupData
    groups = {}
    for pid_str, group_dicts in data.get('groups', {}).items():
        pid = int(pid_str)
        groups[pid] = [GroupData(**gd) for gd in group_dicts]

    return {
        'h5_file_path': data['h5_file_path'],
        'current_particle_id': data.get('current_particle_id', -1),
        'selected_particle_ids': data.get('selected_particle_ids', []),
        'bin_size_ms': data.get('bin_size_ms', 10.0),
        'confidence_level': data.get('confidence_level', 0.95),
        'levels': levels,
        'groups': groups
    }


def _migrate_v1_session(path: Path) -> dict:
    """Migrate from legacy pickle-based format."""
    # Handle backward compatibility with current .smsa files
    with open(path, 'rb') as f:
        old_session = pickle.load(f)

    # Extract data from old format
    # ... migration logic ...

    return migrated_data
```

---

## Performance Optimisation

### Backend Optimisations

| Technique | Application | Expected Benefit |
|-----------|-------------|------------------|
| **NumPy Vectorisation** | Histogram binning, intensity calculation | 10-50x vs pure Python loops |
| **ProcessPoolExecutor** | CPA, AHCA, fitting per particle | Nx parallelism (N = cores) |
| **Lazy Loading** | Microtimes loaded on demand | Reduced memory footprint |
| **Cached Properties** | Level intensity, dwell time | Avoid recomputation |
| **Chunked HDF5 Reading** | Large files (>1GB) | Streaming without full load |

### Optional Numba Acceleration

Numba can be added incrementally for specific hot loops if profiling shows need:

```python
# analysis/change_point.py (optional Numba version)
from numba import njit
import numpy as np


@njit(cache=True)
def compute_likelihood_ratio(
    times: np.ndarray,
    start: int,
    end: int,
    split: int
) -> float:
    """
    Compute log-likelihood ratio for splitting at given point.

    Only use Numba if profiling shows this is a bottleneck.
    """
    n_left = split - start
    n_right = end - split

    if n_left < 2 or n_right < 2:
        return 0.0

    t_left = times[split] - times[start]
    t_right = times[end - 1] - times[split]

    if t_left <= 0 or t_right <= 0:
        return 0.0

    lambda_left = n_left / t_left
    lambda_right = n_right / t_right
    lambda_total = (n_left + n_right) / (t_left + t_right)

    return (
        n_left * np.log(lambda_left) +
        n_right * np.log(lambda_right) -
        (n_left + n_right) * np.log(lambda_total)
    )
```

### Frontend Optimisations

| Technique | Application | Benefit |
|-----------|-------------|---------|
| **WebGL Rendering** | `output_backend="webgl"` in Bokeh | Millions of points at 60fps |
| **Dynamic Tabs** | `dynamic=True` in Panel Tabs | Only render active tab |
| **ColumnDataSource** | Bokeh data streaming | Efficient incremental updates |
| **Virtualised Lists** | Tabulator for particle list | Smooth scrolling with 1000+ items |
| **Debounced Sliders** | Bin size adjustment | Reduce recomputation frequency |

### Memory Management

```python
# Ensure large arrays are properly cleaned up
import gc

class IntensityController(BaseController):

    def _on_file_closed(self):
        """Clean up when file is closed."""
        # Clear Bokeh sources
        self._trace_source.data = {'t': [], 'counts': []}
        self._hist_source.data = {'counts': [], 'left': [], 'right': []}

        # Clear annotations
        self._level_annotations.clear()

        # Force garbage collection for large arrays
        gc.collect()
```

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
# Clone and setup
git clone <repo>
cd full-sms

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Development Server

```bash
# Run Panel dev server with auto-reload
panel serve src/full_sms/app.py --dev --show

# Or with specific port
panel serve src/full_sms/app.py --port 5006 --dev
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=full_sms --cov-report=html

# Run specific test file
pytest tests/test_analysis/test_change_point.py -v

# Run with parallel execution
pytest tests/ -n auto
```

### Type Checking

```bash
# Run mypy
mypy src/full_sms

# With strict mode
mypy src/full_sms --strict
```

### Linting

```bash
# Run ruff
ruff check src/
ruff format src/
```

### Desktop Testing

```bash
# Run as desktop app (development)
python scripts/desktop.py --debug

# Build and test packaged app
python scripts/build.py
./dist/Full\ SMS.app/Contents/MacOS/Full\ SMS  # macOS
```

### pyproject.toml

```toml
[project]
name = "full-sms"
version = "2.0.0"
description = "Single-Molecule Spectroscopy Analysis"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "panel>=1.4.0",
    "bokeh>=3.3.0",
    "holoviews>=1.18.0",
    "hvplot>=0.9.0",
    "param>=2.0.0",
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "h5py>=3.10.0",
    "pandas>=2.1.0",
    "pywebview>=5.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.5.0",
    "mypy>=1.7.0",
    "ruff>=0.1.6",
    "pyinstaller>=6.3.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Bokeh plotting limitations** | Low | Medium | Plotly.js fallback for specific charts; hvPlot abstraction |
| **PyWebView quirks** | Medium | Medium | Test on all platforms early; fallback to browser mode |
| **Panel learning curve** | Low | Low | Strong documentation; similar patterns to PyQt signals |
| **Large file performance** | Medium | Medium | Chunk-based HDF5 reading; memory mapping; lazy loading |
| **Session format changes** | Low | High | JSON-based format with versioning; migration helpers |
| **Cross-platform packaging** | Medium | Medium | GitHub Actions CI for all platforms; early testing |
| **WebGL browser support** | Low | Low | Fallback to Canvas renderer; target modern browsers only |

### Alternative Paths

If Panel proves insufficient:

1. **Streamlit**: Simpler but less interactive; good for quick prototypes
2. **Dash (Plotly)**: More mature, React-based; more JavaScript needed
3. **NiceGUI**: Python-native, Vue.js based; newer but promising
4. **Gradio**: Even simpler; limited customisation

If PyWebView proves problematic:

1. **Browser-only**: Run as local web app; use system browser
2. **Electron + Python**: Heavier but more control; established pattern
3. **CEF Python**: Chromium Embedded Framework; more complex setup

---

## Comparison with FastAPI + React + Tauri

| Criterion | FastAPI + React + Tauri | Panel + PyWebView |
|-----------|:-----------------------:|:-----------------:|
| **Languages** | 3 (Python, TypeScript, Rust) | 1 (Python) |
| **IPC Complexity** | HTTP + WebSocket | None (same process) |
| **State Management** | Dual (Zustand + Python) | Single (Param) |
| **Build Artifacts** | 3 (sidecar + shell + frontend) | 1 (single executable) |
| **Bundle Size** | ~65MB total | ~50-80MB |
| **Development Time** | 6-9 months | 3-4 months |
| **Debugging** | Multi-process, multi-language | Single process Python |
| **UI Polish Potential** | Excellent | Good |
| **Team Learning Curve** | Steep (React, TypeScript) | Gentle (Python ecosystem) |
| **Long-term Maintainability** | Medium | High |
| **Community/Ecosystem** | Large (React) | Medium (HoloViz) |
| **Reuse of Existing Code** | Analysis core only | Analysis + patterns |

### When to Choose Each

**Choose Panel + PyWebView if:**
- Single developer or small Python-focused team
- Scientific/academic environment
- Time-to-market is important
- Long-term maintainability is priority
- UI needs are functional rather than highly polished

**Choose FastAPI + React + Tauri if:**
- Frontend specialists available
- Commercial product with UI polish requirements
- Plan to deploy as web app in future
- Budget allows 6-9 months development
- Team wants to learn modern web stack

---

## Summary

The Panel + PyWebView architecture provides:

- **Single Language**: Python throughout - analysis, UI, and packaging
- **No IPC**: UI and analysis in same process (except worker pool)
- **Reactive State**: Param-based state with automatic UI updates
- **WebGL Performance**: Bokeh's WebGL backend for large datasets
- **Simple Packaging**: Single PyInstaller executable via PyWebView
- **Fast Development**: Estimated 3-4 months for feature parity
- **Easy Maintenance**: Single developer can manage entire stack

This approach is recommended for the Full SMS rewrite given the scientific context, single-developer maintenance requirement, and strong existing Python expertise.

---

*Document created: December 2024*
*Related: python_fastapi_react_tauri.md, critique_fastapi_react_tauri.md*