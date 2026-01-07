# Critique: Python + FastAPI + React + Tauri Proposal

This document provides a critical analysis of the proposed rewrite architecture and presents an alternative approach.

---

## Table of Contents

1. [Strengths of the Proposal](#strengths-of-the-proposal)
2. [Concerns and Risks](#concerns-and-risks)
3. [Alternative: Python + Panel + PyWebView](#alternative-python--panel--pywebview)
4. [Comparison Summary](#comparison-summary)
5. [Recommendation](#recommendation)

---

## Strengths of the Proposal

The FastAPI + React + Tauri architecture has genuine merits:

1. **Preserves Python expertise** - Scientific computing ecosystem stays accessible; NumPy, SciPy, h5py all remain available

2. **Modern UX potential** - React + Plotly.js can deliver excellent interactive visualisations with smooth animations and responsive design

3. **WebSocket progress streaming** - Clean real-time feedback mechanism for long-running analysis tasks

4. **Well-structured API design** - Pydantic v2 contracts provide type safety and automatic validation

5. **Tauri's small footprint** - ~15MB vs Electron's ~150MB is compelling for distribution

6. **Clear separation of concerns** - Frontend/backend split enables independent development and testing

---

## Concerns and Risks

### 1. Architecture Complexity (High Risk)

The proposal requires managing **three languages** (Python, TypeScript, Rust) and **two runtimes** (Node/browser, Python). For a single developer maintaining this long-term:

- **Debugging complexity**: Issues can span IPC boundaries (HTTP/WebSocket between processes), making root cause analysis difficult
- **Build tooling**: Involves PyInstaller + npm + Cargo, each with their own dependency management and versioning
- **Deployment artifacts**: Python sidecar binary + Tauri shell + bundled frontend assets

**Context**: The current PyQt5 app is ~38,000 lines of Python in a single runtime. This proposal fragments that across ecosystems, increasing cognitive load for maintenance.

### 2. Sidecar Management (Medium Risk)

The Python-as-sidecar pattern with Tauri is relatively uncommon in production. Specific risks include:

| Issue | Impact |
|-------|--------|
| Port conflicts | What if port 8000 is already in use? Need dynamic port allocation and discovery |
| Health monitoring | Frontend needs to detect backend crashes and restart or notify user |
| Process lifecycle | Risk of zombie Python processes on app quit, especially on crash |
| macOS App Sandbox | Restrictions on subprocess spawning may complicate notarisation |
| Windows security | Defender/antivirus may flag unsigned sidecar binaries |

The proposal's Rust startup code shows basic sidecar spawning but doesn't address these edge cases.

### 3. Numba Performance Claims Are Optimistic (Medium Risk)

The proposal claims "10-100x speedup" from Numba JIT compilation, but:

- **No baseline**: Current code doesn't use Numba, so there's no measured comparison
- **Algorithm structure**: Looking at `change_point.py`, the hot loop (`_find_all_cpts`) is recursive Python with NumPy operations. Numba's `@njit` has constraints:
  - No Python objects (the `Level` class instantiation inside loops won't work)
  - Limited support for recursion depth
  - No dynamic list appending in nopython mode
- **Cold start overhead**: First invocation incurs ~1-2s JIT compilation delay
- **Parallelisation constraints**: `@njit(parallel=True)` with `prange` requires loop-independent iterations, which the recursive CPA doesn't have

**More realistic assessment**: Numba may help specific inner loops (e.g., the likelihood ratio calculation), but the algorithm structure would need significant refactoring to benefit fully. A 2-5x improvement on specific functions is more realistic than 10-100x overall.

### 4. State Synchronisation Complexity

Session state is split across boundaries:

**Backend (Python)**:
```python
@dataclass
class Session:
    particles: dict[int, ParticleState]
    settings: dict
    # ... serialised to pickle
```

**Frontend (TypeScript/Zustand)**:
```typescript
const useSessionStore = create<SessionState>()(...);
const useParticleStore = create<ParticleState>()(...);
const useAnalysisStore = create<AnalysisState>()(...);
```

This creates **dual sources of truth**. Scenarios requiring careful handling:

1. User drags bin size slider → frontend state updates → API call → backend recomputes → response updates frontend
2. Analysis completes → backend state changes → WebSocket notification → frontend must invalidate/refetch
3. User rapidly clicks multiple particles → race conditions between requests
4. Network/IPC hiccup during analysis → state divergence

These are solvable but add development and testing burden.

### 5. Development Timeline Underestimate

The proposal estimates 14-16 weeks for a single developer. Required work:

| Task | Estimated Effort |
|------|------------------|
| Port ~38k lines of analysis code to new structure | 4-6 weeks |
| Build complete React frontend with Plotly integration | 4-6 weeks |
| Implement WebSocket progress streaming | 1-2 weeks |
| Tauri + PyInstaller packaging and testing | 2-3 weeks |
| Save/load compatibility with existing .smsa files | 1-2 weeks |
| Export functionality (CSV, plots, etc.) | 1-2 weeks |
| Cross-platform testing (macOS, Windows, Linux) | 2-3 weeks |
| Bug fixes and polish | 2-4 weeks |

**Realistic estimate**: 6-9 months for production quality, assuming no major architectural pivots.

### 6. Hidden Costs

- **Plotly.js bundle size**: ~3MB minified, partially offsetting Tauri's small footprint advantage
- **React + TypeScript learning curve**: If the team is Python-focused, this is non-trivial
- **No offline-first data sync**: Session state is just pickle with no versioning or conflict resolution
- **Testing complexity**: Need both Python pytest and JavaScript/TypeScript test suites

---

## Alternative: Python + Panel + PyWebView

A simpler architecture that maintains Python as the single language while still delivering a modern web-based UI.

### Tech Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| **UI Framework** | **Panel (HoloViz)** | Python-native, designed for scientific apps, WebGL via Bokeh |
| **Desktop Shell** | **PyWebView** | Pure Python webview wrapper, single executable |
| **Plotting** | **Bokeh / hvPlot** | WebGL-accelerated, integrates with Panel |
| **Parallelism** | **ProcessPoolExecutor** | Same as proposal, proven pattern |
| **Performance** | **NumPy vectorisation** | Start without Numba, add if profiling shows need |
| **Packaging** | **PyInstaller** | Single artifact, no sidecar complexity |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               PyWebView Shell (Native Window)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Panel Application (WebView)                   │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Bokeh Plots (WebGL) │ Panel Widgets │ Param State │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                          │                                 │  │
│  │  ┌───────────────────────▼─────────────────────────────┐  │  │
│  │  │          Python Core (Same Process)                  │  │  │
│  │  │  ┌─────────────────────────────────────────────────┐│  │  │
│  │  │  │  Analysis Engine │ H5 Reader │ Session State   ││  │  │
│  │  │  └─────────────────────────────────────────────────┘│  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                          │                                 │  │
│  │  ┌───────────────────────▼─────────────────────────────┐  │  │
│  │  │      ProcessPoolExecutor (Background Workers)       │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Differences from FastAPI + React + Tauri

| Aspect | FastAPI + React + Tauri | Panel + PyWebView |
|--------|------------------------|-------------------|
| **Languages** | 3 (Python, TypeScript, Rust) | 1 (Python) |
| **IPC** | HTTP + WebSocket | In-process (no IPC) |
| **State sync** | Dual (frontend + backend) | Single source of truth |
| **Build artifacts** | Sidecar + Shell + Frontend | Single executable |
| **Learning curve** | React, TypeScript, Tauri APIs | Panel, Param (Python) |
| **Bundle size** | ~15MB (Tauri) + ~50MB (sidecar) | ~50-80MB (PyInstaller) |
| **Debugging** | Multi-process, multi-language | Single process Python |

### Sample Implementation

```python
import panel as pn
import param
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

pn.extension()

class IntensityPlot(param.Parameterized):
    """Interactive intensity trace with level overlays."""

    particle_id = param.Integer(default=1)
    bin_size_ms = param.Number(default=10.0, bounds=(1, 1000))
    show_levels = param.Boolean(default=True)

    def __init__(self, session, **params):
        super().__init__(**params)
        self.session = session
        self.source = ColumnDataSource(data={'t': [], 'counts': []})

    @param.depends('particle_id', 'bin_size_ms')
    def intensity_plot(self):
        particle = self.session.get_particle(self.particle_id)
        t, counts = particle.get_trace(bin_size_ms=self.bin_size_ms)

        self.source.data = {'t': t, 'counts': counts}

        p = figure(
            title=f"Particle {self.particle_id}",
            x_axis_label="Time (s)",
            y_axis_label="Counts",
            tools="pan,wheel_zoom,box_zoom,reset",
            active_scroll="wheel_zoom",
            output_backend="webgl"  # WebGL for performance
        )
        p.line('t', 'counts', source=self.source, line_width=1)

        if self.show_levels and particle.has_levels:
            self._add_level_rectangles(p, particle.levels)

        return p

    def _add_level_rectangles(self, p, levels):
        for i, level in enumerate(levels):
            color = f"hsl({(i * 137) % 360}, 70%, 60%)"
            p.quad(
                left=level.start_time_s, right=level.end_time_s,
                top=level.intensity_cps, bottom=0,
                fill_alpha=0.3, fill_color=color, line_width=0
            )


class FullSMSApp(param.Parameterized):
    """Main application container."""

    def __init__(self, **params):
        super().__init__(**params)
        self.session = Session()
        self.intensity = IntensityPlot(self.session)

    def view(self):
        sidebar = pn.Column(
            pn.pane.Markdown("## Particles"),
            self._particle_list(),
            pn.pane.Markdown("## Controls"),
            pn.Param(self.intensity, parameters=['bin_size_ms', 'show_levels'])
        )

        main = pn.Tabs(
            ("Intensity", self.intensity.intensity_plot),
            ("Lifetime", self._lifetime_tab),
            ("Grouping", self._grouping_tab),
            ("Export", self._export_tab),
        )

        return pn.template.MaterialTemplate(
            title="Full SMS",
            sidebar=[sidebar],
            main=[main]
        )


# Desktop packaging with PyWebView
if __name__ == "__main__":
    import webview

    app = FullSMSApp()
    server = pn.serve(app.view, show=False, port=0)  # Random available port

    webview.create_window(
        "Full SMS",
        f"http://localhost:{server.port}",
        width=1400, height=900,
        min_size=(1024, 768)
    )
    webview.start()
```

### Advantages of Panel Approach

1. **Single Language**: Entire codebase remains Python - matches team expertise
2. **No IPC Overhead**: UI and analysis in same process (except background workers)
3. **Single State**: Param-based reactive state is the source of truth
4. **Simpler Debugging**: Standard Python debugging, profiling, tracing tools
5. **Incremental Migration**: Could port modules one at a time from PyQt5
6. **Mature Ecosystem**: Panel/Bokeh widely used in scientific Python community
7. **Faster Development**: Estimated 3-4 months for MVP vs 6-9 months

### Disadvantages of Panel Approach

1. **Less UI Polish**: Panel's widget set is functional but not as refined as React component libraries like shadcn/ui or Material UI
2. **Larger Bundle**: ~50-80MB vs Tauri's potential ~15MB (though no sidecar needed)
3. **Less Customisation**: Complex custom UI interactions harder than React
4. **Smaller Community**: Panel community is smaller than React ecosystem

---

## Comparison Summary

| Criterion | FastAPI + React + Tauri | Panel + PyWebView |
|-----------|:----------------------:|:-----------------:|
| Development time | 6-9 months | 3-4 months |
| Maintenance complexity | High | Low |
| UI polish potential | Excellent | Good |
| Bundle size | ~65MB total | ~50-80MB |
| Languages to maintain | 3 | 1 |
| State management | Complex | Simple |
| Debugging ease | Difficult | Easy |
| Team learning curve | Steep | Gentle |
| Long-term maintainability | Medium | High |
| Community/ecosystem | Large (React) | Medium (Panel) |

---

## Recommendation

Given the context:
- Single developer maintenance
- Scientific/academic environment
- Strong Python team expertise
- ~38,000 lines of existing Python code
- Need for long-term maintainability

### Primary Recommendation: Panel + PyWebView

For most scenarios, the **Panel + PyWebView** approach offers:
- Lower complexity and faster development
- Easier maintenance by a single developer
- Incremental migration path from current PyQt5
- Sufficient UI quality for scientific applications

### When to Choose FastAPI + React + Tauri

The more complex architecture makes sense if:
- UI polish is a hard requirement (e.g., commercial product)
- Team will grow to include frontend specialists
- Budget allows 6-9 months of development
- Long-term plan includes web deployment (not just desktop)

### Hybrid Option

A middle ground exists: **FastAPI + htmx + Tauri**
- Python backend (FastAPI)
- Server-rendered HTML with htmx for interactivity
- Plotly.js embedded in templates
- Simpler than full React SPA
- Still benefits from Tauri's small footprint

This could be explored as a third option if neither extreme fits perfectly.

---

## Next Steps

1. **Prototype Panel approach** (1-2 weeks): Build a minimal viable prototype with Panel to validate plotting performance and UX
2. **Benchmark Numba claims**: Profile actual CPA algorithm to determine realistic speedup potential
3. **User testing**: Show both approaches to end users for feedback on UI requirements
4. **Final decision**: Based on prototype results and user feedback

---

*Document created: December 2024*
*Related: python_fastapi_react_tauri.md*
