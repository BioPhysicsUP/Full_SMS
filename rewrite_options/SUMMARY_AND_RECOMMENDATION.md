# Full SMS Rewrite Options: Summary and Recommendation

This document summarizes the three rewrite options that have been evaluated and provides a recommendation based on the project context.

---

## Context Recap

Full SMS is a ~38,000-line PyQt5 desktop application for single-molecule spectroscopy analysis. Key requirements for the rewrite:

- Load and analyze HDF5 files containing photon timing data
- Change point analysis (level detection)
- Hierarchical clustering (grouping)
- Fluorescence lifetime fitting
- Interactive visualization with WebGL-capable plots
- Desktop distribution (macOS, Windows, Linux)
- Single developer maintenance
- Academic/research environment

---

## Option Comparison Matrix

| Criterion | DearPyGui | Panel + PyWebView | FastAPI + React + Tauri |
|-----------|:---------:|:-----------------:|:-----------------------:|
| **Languages** | 1 (Python) | 1 (Python) | 3 (Python, TS, Rust) |
| **Development time** | 3-5 months | 4-6 months | 6-9 months |
| **UI polish** | Functional | Dashboard | App-quality |
| **Bundle size** | ~35MB | ~100-150MB | ~65-80MB |
| **Plot performance** | Excellent | Good | Good |
| **Modal dialogs** | Native | Workaround | Native |
| **Tree views** | Basic | Limited | Full |
| **State management** | Simple | Medium | Complex |
| **Community size** | Small | Medium | Large |
| **Web deployment** | No | Limited | Yes |
| **Debugging** | Easy | Easy | Difficult |
| **IPC complexity** | None | None | HTTP + WebSocket |

---

## Option 1: Python + FastAPI + React + Tauri

### Summary

A modern web-based architecture with React frontend, FastAPI Python backend, and Tauri for desktop packaging. Uses WebSocket for progress streaming and Numba for computational acceleration.

### Architecture

```
Tauri Shell (Rust)
  └── React Frontend (WebView)
        └── HTTP + WebSocket
              └── FastAPI Backend (Python Sidecar)
                    └── ProcessPoolExecutor (Workers)
```

### Strengths

- **Polish potential**: React + Tailwind can deliver an excellent, modern UI
- **Small Tauri bundle**: ~15MB shell (but +50MB Python sidecar)
- **Web-ready**: Could serve as web app with minimal changes
- **Rich ecosystem**: Vast React component libraries (tree views, modals, charts)
- **Type safety**: TypeScript + Pydantic provide strong contracts

### Weaknesses

- **Complexity**: Three languages, two runtimes, IPC boundaries
- **Sidecar management**: Port conflicts, health monitoring, process lifecycle
- **Development time**: Realistically 6-9 months for production quality
- **Debugging**: Multi-process, multi-language debugging is painful
- **Dual state**: Must keep frontend (Zustand) and backend (Python) state synchronized

### When to Choose

- UI polish is a hard requirement
- Future web deployment is planned
- Team includes or will include frontend specialists
- Budget allows longer development timeline

---

## Option 2: Python + Panel + PyWebView

### Summary

A Python-only approach using Panel (HoloViz ecosystem) for the UI, Bokeh for WebGL plotting, and PyWebView for desktop packaging. Single process architecture with Param-based reactive state.

### Architecture

```
PyWebView Shell (Native Window)
  └── Panel Application (Bokeh Server)
        └── Python Core (Same Process)
              └── ProcessPoolExecutor (Workers)
```

### Strengths

- **Single language**: Python throughout - no context switching
- **No IPC**: UI and analysis in same process
- **Param reactivity**: Clean, declarative reactive state without boilerplate
- **Scientific integration**: Direct NumPy/SciPy/h5py access
- **Faster development**: 3-4 months realistic for MVP

### Weaknesses

- **Dashboard aesthetic**: Panel is designed for dashboards, not desktop apps
- **UI limitations**: No native modals, limited tree views, no context menus
- **Platform inconsistencies**: PyWebView uses different engines per platform
- **Bundle size**: Realistically ~100-150MB with all dependencies
- **Smaller community**: Fewer resources than React ecosystem

### When to Choose

- Single developer maintenance
- Python expertise is strong, frontend expertise is weak
- Functional UI is acceptable
- Speed to market matters more than polish

---

## Option 3: Python + DearPyGui

### Summary

A GPU-accelerated native GUI using DearPyGui (Python bindings for Dear ImGui). Immediate mode paradigm with ImPlot for scientific plotting. No web technologies involved.

### Architecture

```
DearPyGui (Single Process)
  └── Native Viewport (OpenGL/DirectX/Metal)
        └── ImPlot Charts + ImGui Widgets
              └── ProcessPoolExecutor (Workers)
```

### Strengths

- **True GPU performance**: Native rendering, not WebGL through browser
- **Simplest stack**: One language, one runtime, no web technologies
- **Smallest bundle**: ~30-50MB
- **Immediate mode**: Simpler state management paradigm
- **Native features**: File dialogs, modals, keyboard shortcuts work natively

### Weaknesses

- **ImGui aesthetic**: Functional but not polished; "game engine debugger" look
- **Paradigm shift**: Immediate mode is different from PyQt's retained mode
- **Limited layout system**: No CSS-like flexbox/grid
- **Basic tree views**: Collapsing headers, not full hierarchical tree
- **No web path**: Desktop only, no future web deployment option

### When to Choose

- Performance is paramount
- Single developer maintaining long-term
- Functional UI is acceptable
- Want to avoid web technologies entirely
- Interested in learning immediate mode paradigm

---

## Recommendation

### Primary: DearPyGui

For the Full SMS rewrite, I recommend **DearPyGui** for the following reasons:

1. **Simplest architecture**: Single Python process, no IPC, no web technologies. This minimizes debugging complexity and maintenance burden for a single developer.

2. **Best plot performance**: ImPlot is purpose-built for scientific real-time plotting with true GPU acceleration. Given that Full SMS deals with intensity traces potentially containing millions of data points and 50+ level overlays, this matters.

3. **Smallest distribution**: ~30-50MB is significantly smaller than both alternatives. Important for academic distribution where users may have limited bandwidth.

4. **Native features work**: File dialogs, modals, keyboard shortcuts all work natively without workarounds - addressing key weaknesses of the Panel approach.

5. **Fastest development**: The immediate mode paradigm, while different, is often simpler once learned. State management is straightforward (no reactive binding bugs).

6. **Matches context**: For a scientific tool used by researchers in an academic setting, functional UI is acceptable. Users prioritize analysis capability over visual polish.

### Trade-off Acceptance

The main trade-off is **UI aesthetics**. DearPyGui produces functional, professional-looking but not "modern app" styled interfaces. For a scientific analysis tool, this is acceptable.

### Runner-up: Panel + PyWebView

If the immediate mode paradigm is too unfamiliar or if there's any chance of future web deployment, **Panel + PyWebView** is a solid alternative. Key differences:

- Slightly more familiar patterns (reactive rather than immediate)
- Web deployment possible (with work)
- Larger bundle, more platform inconsistencies
- UI limitations require workarounds (modals, tree views)

### Avoid: FastAPI + React + Tauri

While this option produces the most polished result, the complexity is not justified for:
- Single developer maintenance
- Academic/research context
- ~38k lines of Python that would need significant restructuring

The 6-9 month timeline and multi-language debugging complexity are significant drawbacks that outweigh the UI polish benefits.

---

## Implementation Approach (if DearPyGui chosen)

### Phase 1: Foundation (2-3 weeks)
- Set up project structure
- Implement data models (ParticleData, LevelData, GroupData)
- HDF5 file loading
- Basic window layout with ImPlot placeholder

### Phase 2: Core Analysis (3-4 weeks)
- Port change point analysis algorithm
- Port AHCA clustering algorithm
- ProcessPoolExecutor integration
- Progress tracking

### Phase 3: Visualization (3-4 weeks)
- Intensity trace plot with level overlays
- Decay histogram plot
- BIC optimization plot
- Correlation plot

### Phase 4: Fitting (2-3 weeks)
- Port lifetime fitting code
- Fitting dialog
- Fit result display and residuals

### Phase 5: Export & Persistence (2-3 weeks)
- Save/load analysis sessions
- CSV/DataFrame export
- Plot export

### Phase 6: Polish & Testing (2-4 weeks)
- Cross-platform testing
- PyInstaller packaging
- Bug fixes
- User testing

**Total: ~14-21 weeks (3.5-5 months)**

---

## Risk Mitigation

Regardless of choice, plan for:

1. **Early prototyping**: Build a minimal prototype with the chosen framework to validate assumptions about plotting performance and UI patterns

2. **Cross-platform testing**: Test on all target platforms early, not just before release

3. **User feedback**: Get feedback from actual users mid-development to catch UX issues

4. **Performance profiling**: Profile with real data (100+ particles, 1M+ photons) to catch scaling issues

5. **Incremental migration**: Consider porting the analysis core first, then building the new UI around it

---

## Conclusion

The three options represent a spectrum from simplicity (DearPyGui) through pragmatic compromise (Panel) to maximum capability (React/Tauri). Given Full SMS's context as a scientific tool maintained by a single developer in an academic setting, **DearPyGui** offers the best balance of development speed, runtime performance, and long-term maintainability.

The immediate mode paradigm requires a mindset shift from PyQt's retained mode, but this is a worthwhile investment that will pay off in simpler state management and fewer synchronization bugs.

---

*Document created: January 2025*
*Based on analysis of: full_context.md, python_dearpygui.md, python_dearpygui_critique.md, python_fastapi_react_tauri.md, python_fastapi_react_tauri_critique.md, python_panel_pywebview.md, python_panel_pywebview_critique.md*
