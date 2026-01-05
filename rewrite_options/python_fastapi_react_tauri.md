# Full SMS Rewrite: Python + FastAPI + React + Tauri

## Executive Summary

This document outlines a complete rewrite strategy for the Full SMS application using a modern web-based architecture, packaged as a native desktop application. The solution prioritises:

- **Performance**: Near C-speed computation via Numba JIT compilation
- **Offline-first**: Fully local execution with no cloud dependencies
- **Maintainability**: Single developer-friendly Python backend
- **Modern UX**: React-based responsive UI with real-time updates
- **Small footprint**: ~10-15MB distribution via Tauri (vs ~150MB+ Electron)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack Details](#tech-stack-details)
3. [Backend Design](#backend-design)
4. [Frontend Design](#frontend-design)
5. [Desktop Packaging](#desktop-packaging)
6. [Parallel Processing Strategy](#parallel-processing-strategy)
7. [Data Flow](#data-flow)
8. [State Management](#state-management)
9. [Performance Optimisation](#performance-optimisation)
10. [Development Workflow](#development-workflow)
11. [Risks and Mitigations](#risks-and-mitigations)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Tauri Shell (Rust)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              React Frontend (WebView)                      │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Plotly.js Charts │ State (Zustand) │ WebSocket     │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │ HTTP REST + WebSocket              │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │           FastAPI Backend (Python Sidecar)                 │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  REST Endpoints │ WebSocket Manager │ Session State │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                          │                                 │  │
│  │  ┌───────────────────────▼─────────────────────────────┐  │  │
│  │  │      ProcessPoolExecutor (Persistent Pool)          │  │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │  │  │
│  │  │  │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker N │   │  │  │
│  │  │  │ (Numba) │ │ (Numba) │ │ (Numba) │ │ (Numba) │   │  │  │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                          │                                 │  │
│  │  ┌───────────────────────▼─────────────────────────────┐  │  │
│  │  │     Shared Memory Pool (HDF5 Data, Zero-Copy)       │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack Details

### Backend

| Component | Technology | Justification |
|-----------|------------|---------------|
| Language | **Python 3.12** | Team expertise, scientific ecosystem |
| Framework | **FastAPI** | Async support, WebSocket native, fast |
| Analysis | **NumPy + SciPy** | Standard scientific computing |
| JIT Compiler | **Numba** | Near-C speed for hot loops |
| HDF5 | **h5py** | Direct HDF5 access (existing format) |
| Parallelism | **ProcessPoolExecutor** | True parallelism, no GIL issues |
| Shared Memory | **multiprocessing.shared_memory** | Zero-copy large arrays |
| Serialisation | **Pydantic v2** | Type-safe API contracts |

### Frontend

| Component | Technology | Justification |
|-----------|------------|---------------|
| Framework | **React 18** | Component model, ecosystem |
| Language | **TypeScript** | Type safety, better tooling |
| Build Tool | **Vite** | Fast HMR, modern bundling |
| State | **Zustand** | Simple, performant, no boilerplate |
| Plotting | **Plotly.js** | WebGL acceleration, scientific plots |
| Styling | **Tailwind CSS** | Rapid UI development |
| Data Fetching | **TanStack Query** | Caching, background updates |

### Desktop Packaging

| Component | Technology | Justification |
|-----------|------------|---------------|
| Shell | **Tauri 2.0** | Tiny bundle, native performance |
| Python Bundling | **PyInstaller** or **Nuitka** | Single executable sidecar |
| IPC | **HTTP localhost** | Simple, debuggable |

---

## Backend Design

### Project Structure

```
backend/
├── pyproject.toml
├── src/
│   └── full_sms/
│       ├── __init__.py
│       ├── main.py              # FastAPI app entry point
│       ├── config.py            # Settings management
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes/
│       │   │   ├── files.py     # File open/close endpoints
│       │   │   ├── particles.py # Particle data endpoints
│       │   │   ├── analysis.py  # CPA, grouping, fitting
│       │   │   ├── export.py    # Export endpoints
│       │   │   └── session.py   # Save/load analysis
│       │   └── websocket.py     # WebSocket progress streaming
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py        # Pydantic models (API contracts)
│       │   ├── session.py       # Session state management
│       │   └── exceptions.py    # Custom exceptions
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── change_point.py  # CPA algorithm (Numba)
│       │   ├── clustering.py    # AHCA algorithm (Numba)
│       │   ├── lifetime.py      # Decay fitting
│       │   ├── correlation.py   # Antibunching
│       │   └── histograms.py    # Histogram computation
│       │
│       ├── io/
│       │   ├── __init__.py
│       │   ├── hdf5_reader.py   # HDF5 file parsing
│       │   ├── exporters.py     # Export implementations
│       │   └── save_load.py     # Session persistence
│       │
│       └── workers/
│           ├── __init__.py
│           ├── pool.py          # ProcessPoolExecutor management
│           └── tasks.py         # Parallel task definitions
│
└── tests/
    └── ...
```

### API Design

#### REST Endpoints

```
Files
  POST   /api/files/open              # Open HDF5 file, return metadata
  POST   /api/files/close             # Close file, cleanup session
  GET    /api/files/info              # Get current file info

Particles
  GET    /api/particles               # List all particles (summary)
  GET    /api/particles/{id}          # Get particle details
  GET    /api/particles/{id}/trace    # Get intensity trace data
  GET    /api/particles/{id}/decay    # Get decay histogram
  PATCH  /api/particles/{id}/bin-size # Update bin size

Analysis
  POST   /api/analysis/resolve        # Run CPA (body: particle IDs, params)
  POST   /api/analysis/group          # Run AHCA (body: particle IDs, params)
  POST   /api/analysis/fit            # Run lifetime fitting
  POST   /api/analysis/correlate      # Run antibunching
  DELETE /api/analysis/levels/{id}    # Clear levels for particle
  DELETE /api/analysis/groups/{id}    # Clear groups for particle

Results
  GET    /api/results/levels/{particle_id}    # Get levels
  GET    /api/results/groups/{particle_id}    # Get groups
  GET    /api/results/fits/{particle_id}      # Get fit results

Export
  POST   /api/export                  # Trigger export (body: options)
  GET    /api/export/download/{id}    # Download exported file

Session
  POST   /api/session/save            # Save analysis to .smsa
  POST   /api/session/load            # Load analysis from .smsa
  GET    /api/session/settings        # Get current settings
  PATCH  /api/session/settings        # Update settings
```

#### WebSocket Endpoint

```
WS /ws/progress

Messages (Server → Client):
{
  "type": "task_started",
  "task_id": "uuid",
  "task_type": "resolve",
  "total": 50
}

{
  "type": "progress",
  "task_id": "uuid",
  "current": 25,
  "total": 50,
  "particle_id": 42,
  "message": "Resolved particle 42: 5 levels"
}

{
  "type": "task_completed",
  "task_id": "uuid",
  "result_summary": {...}
}

{
  "type": "error",
  "task_id": "uuid",
  "message": "Analysis failed",
  "details": "..."
}
```

### Pydantic Models (API Contracts)

```python
from pydantic import BaseModel
from typing import Optional
import numpy as np

class ParticleSummary(BaseModel):
    id: int
    description: str
    num_photons: int
    measurement_time_s: float
    has_levels: bool
    num_levels: int
    has_groups: bool
    num_groups: int

class IntensityTrace(BaseModel):
    time_s: list[float]
    counts: list[int]
    bin_size_ms: float

class Level(BaseModel):
    id: int
    start_time_s: float
    end_time_s: float
    intensity_cps: float  # counts per second
    num_photons: int
    dwell_time_s: float
    group_id: Optional[int] = None
    
    # Fit results (if fitted)
    tau: Optional[list[float]] = None
    amplitude: Optional[list[float]] = None
    chi_squared: Optional[float] = None

class Group(BaseModel):
    id: int
    level_ids: list[int]
    intensity_cps: float
    total_dwell_time_s: float
    num_photons: int
    
    # Fit results
    tau: Optional[list[float]] = None
    amplitude: Optional[list[float]] = None
    avtau: Optional[float] = None
    chi_squared: Optional[float] = None

class AnalysisRequest(BaseModel):
    particle_ids: list[int]
    confidence: float = 0.95
    min_photons: int = 20
    min_boundary_offset: int = 7

class FitRequest(BaseModel):
    target_type: str  # "particle", "level", "group"
    target_ids: list[int]
    num_exponentials: int = 1
    start_channel: Optional[int] = None
    end_channel: Optional[int] = None
    use_irf: bool = True
```

---

## Frontend Design

### Project Structure

```
frontend/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── index.html
│
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   │
│   ├── api/
│   │   ├── client.ts            # Axios/fetch wrapper
│   │   ├── endpoints.ts         # API endpoint functions
│   │   └── websocket.ts         # WebSocket connection manager
│   │
│   ├── stores/
│   │   ├── sessionStore.ts      # File/session state (Zustand)
│   │   ├── particleStore.ts     # Particle selection/data
│   │   ├── analysisStore.ts     # Analysis parameters
│   │   └── uiStore.ts           # UI state (tabs, dialogs)
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── MainLayout.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── TabPanel.tsx
│   │   │
│   │   ├── particles/
│   │   │   ├── ParticleList.tsx
│   │   │   ├── ParticleItem.tsx
│   │   │   └── ParticleFilter.tsx
│   │   │
│   │   ├── plots/
│   │   │   ├── IntensityPlot.tsx
│   │   │   ├── DecayPlot.tsx
│   │   │   ├── CorrelationPlot.tsx
│   │   │   ├── BICPlot.tsx
│   │   │   └── RasterScanPlot.tsx
│   │   │
│   │   ├── analysis/
│   │   │   ├── ResolveControls.tsx
│   │   │   ├── GroupingControls.tsx
│   │   │   ├── FittingDialog.tsx
│   │   │   └── ProgressIndicator.tsx
│   │   │
│   │   └── common/
│   │       ├── Button.tsx
│   │       ├── Slider.tsx
│   │       ├── Dialog.tsx
│   │       └── FileDropzone.tsx
│   │
│   ├── hooks/
│   │   ├── useParticle.ts       # Particle data fetching
│   │   ├── useAnalysis.ts       # Analysis mutations
│   │   ├── useProgress.ts       # WebSocket progress
│   │   └── usePlotData.ts       # Plot data transformation
│   │
│   ├── types/
│   │   ├── api.ts               # API response types
│   │   ├── analysis.ts          # Analysis types
│   │   └── plots.ts             # Plot configuration types
│   │
│   └── utils/
│       ├── formatters.ts        # Number/time formatting
│       └── colors.ts            # Level/group color schemes
│
└── public/
    └── ...
```

### Key Components

#### IntensityPlot.tsx

```tsx
import Plot from 'react-plotly.js';
import { useMemo } from 'react';
import type { Level, Group } from '@/types/api';

interface Props {
  timeS: number[];
  counts: number[];
  levels?: Level[];
  groups?: Group[];
  showLevels: boolean;
  showGroups: boolean;
}

export function IntensityPlot({ timeS, counts, levels, groups, showLevels, showGroups }: Props) {
  const traces = useMemo(() => {
    const result: Plotly.Data[] = [
      {
        x: timeS,
        y: counts,
        type: 'scattergl',  // WebGL for performance
        mode: 'lines',
        name: 'Intensity',
        line: { color: '#3b82f6', width: 1 }
      }
    ];

    // Add level rectangles as shapes
    if (showLevels && levels) {
      levels.forEach((level, i) => {
        result.push({
          x: [level.start_time_s, level.end_time_s, level.end_time_s, level.start_time_s, level.start_time_s],
          y: [0, 0, level.intensity_cps, level.intensity_cps, 0],
          type: 'scatter',
          fill: 'toself',
          fillcolor: `hsla(${(i * 137) % 360}, 70%, 60%, 0.3)`,
          line: { width: 0 },
          showlegend: false,
          hoverinfo: 'skip'
        });
      });
    }

    return result;
  }, [timeS, counts, levels, showLevels]);

  return (
    <Plot
      data={traces}
      layout={{
        xaxis: { title: 'Time (s)' },
        yaxis: { title: 'Counts' },
        hovermode: 'closest',
        dragmode: 'zoom'
      }}
      config={{ responsive: true, scrollZoom: true }}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
```

#### Progress via WebSocket

```tsx
// hooks/useProgress.ts
import { useEffect } from 'react';
import { useAnalysisStore } from '@/stores/analysisStore';

export function useProgress() {
  const { setProgress, setTaskStatus } = useAnalysisStore();

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/progress');

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      
      switch (msg.type) {
        case 'task_started':
          setTaskStatus(msg.task_id, 'running');
          break;
        case 'progress':
          setProgress(msg.task_id, msg.current, msg.total);
          break;
        case 'task_completed':
          setTaskStatus(msg.task_id, 'completed');
          break;
        case 'error':
          setTaskStatus(msg.task_id, 'error', msg.message);
          break;
      }
    };

    return () => ws.close();
  }, []);
}
```

---

## Desktop Packaging

### Tauri Configuration

```json
// src-tauri/tauri.conf.json
{
  "productName": "Full SMS",
  "version": "2.0.0",
  "bundle": {
    "identifier": "za.ac.up.fullsms",
    "icon": ["icons/icon.icns", "icons/icon.ico", "icons/icon.png"]
  },
  "app": {
    "windows": [
      {
        "title": "Full SMS",
        "width": 1400,
        "height": 900,
        "minWidth": 1024,
        "minHeight": 768
      }
    ]
  },
  "plugins": {
    "shell": {
      "sidecar": true
    }
  }
}
```

### Python Sidecar

The Python backend is bundled as a sidecar executable using PyInstaller:

```python
# build_sidecar.py
import PyInstaller.__main__

PyInstaller.__main__.run([
    'src/full_sms/main.py',
    '--name=full-sms-backend',
    '--onefile',
    '--hidden-import=h5py',
    '--hidden-import=numba',
    '--hidden-import=scipy.optimize',
    '--add-data=src/full_sms/config:config',
    '--target-arch=universal2',  # macOS universal
])
```

### Startup Flow

```rust
// src-tauri/src/main.rs
use tauri::Manager;
use std::process::Command;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Spawn Python backend as sidecar
            let sidecar = app.shell()
                .sidecar("full-sms-backend")
                .args(["--port", "8000"])
                .spawn()
                .expect("Failed to start backend");
            
            // Store handle for cleanup
            app.manage(sidecar);
            
            // Wait for backend to be ready
            wait_for_backend("http://localhost:8000/health");
            
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error running app");
}
```

---

## Parallel Processing Strategy

### ProcessPoolExecutor with Shared Memory

```python
# workers/pool.py
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np
import os
from typing import Callable
import asyncio

class AnalysisPool:
    """Manages a persistent process pool for CPU-intensive analysis."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or os.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self._shared_memory_refs: dict[str, shared_memory.SharedMemory] = {}
    
    def share_array(self, name: str, array: np.ndarray) -> tuple[str, tuple, np.dtype]:
        """Put array into shared memory, return reference info."""
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        shared_arr = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        np.copyto(shared_arr, array)
        
        self._shared_memory_refs[name] = shm
        return shm.name, array.shape, array.dtype
    
    def cleanup_shared(self, name: str):
        """Release shared memory."""
        if name in self._shared_memory_refs:
            self._shared_memory_refs[name].close()
            self._shared_memory_refs[name].unlink()
            del self._shared_memory_refs[name]
    
    async def run_parallel(
        self,
        func: Callable,
        items: list,
        progress_callback: Callable[[int, int, dict], None] = None
    ) -> list:
        """Run function on items in parallel, streaming progress."""
        loop = asyncio.get_event_loop()
        total = len(items)
        results = []
        
        # Submit all tasks
        futures = {
            loop.run_in_executor(self.executor, func, item): i 
            for i, item in enumerate(items)
        }
        
        # Collect results as they complete
        completed = 0
        for coro in asyncio.as_completed(futures.keys()):
            result = await coro
            completed += 1
            results.append(result)
            
            if progress_callback:
                await progress_callback(completed, total, result)
        
        return results
    
    def shutdown(self):
        """Clean up resources."""
        for shm in self._shared_memory_refs.values():
            shm.close()
            shm.unlink()
        self.executor.shutdown(wait=True)


# Global pool instance
analysis_pool = AnalysisPool()
```

### Worker Task Example (CPA)

```python
# workers/tasks.py
from multiprocessing import shared_memory
import numpy as np
from numba import njit, prange

from ..analysis.change_point import find_change_points


def run_cpa_task(task_input: dict) -> dict:
    """
    Execute CPA on a single particle.
    Runs in worker process with access to shared memory.
    """
    # Reconstruct array from shared memory
    shm = shared_memory.SharedMemory(name=task_input['shm_name'])
    abstimes = np.ndarray(
        task_input['shape'], 
        dtype=task_input['dtype'], 
        buffer=shm.buf
    )
    
    # Run analysis
    levels = find_change_points(
        abstimes=abstimes,
        confidence=task_input['confidence'],
        min_photons=task_input['min_photons'],
        min_boundary_offset=task_input['min_boundary_offset']
    )
    
    shm.close()  # Don't unlink - main process manages lifecycle
    
    return {
        'particle_id': task_input['particle_id'],
        'levels': [level.to_dict() for level in levels],
        'num_levels': len(levels)
    }
```

### Numba-Accelerated Core Algorithm

```python
# analysis/change_point.py
import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import Optional


@njit(cache=True)
def compute_log_likelihood_ratio(
    times: np.ndarray, 
    start: int, 
    end: int, 
    split: int
) -> float:
    """
    Compute log-likelihood ratio for splitting at given point.
    Compiled to machine code by Numba.
    """
    n_left = split - start
    n_right = end - split
    n_total = end - start
    
    if n_left < 2 or n_right < 2:
        return 0.0
    
    # Time spans
    t_left = times[split] - times[start]
    t_right = times[end - 1] - times[split]
    t_total = times[end - 1] - times[start]
    
    if t_left <= 0 or t_right <= 0:
        return 0.0
    
    # Intensities
    lambda_left = n_left / t_left
    lambda_right = n_right / t_right
    lambda_total = n_total / t_total
    
    # Log-likelihood ratio
    ll_ratio = (
        n_left * np.log(lambda_left) + 
        n_right * np.log(lambda_right) - 
        n_total * np.log(lambda_total)
    )
    
    return ll_ratio


@njit(cache=True, parallel=True)
def find_best_split(times: np.ndarray, start: int, end: int, min_offset: int) -> tuple:
    """
    Find the best split point in a segment.
    Parallelised across potential split points.
    """
    n = end - start
    if n < 2 * min_offset:
        return -1, 0.0
    
    best_split = -1
    best_ratio = 0.0
    
    # Check all potential split points in parallel
    ratios = np.zeros(n)
    for i in prange(min_offset, n - min_offset):
        split = start + i
        ratios[i] = compute_log_likelihood_ratio(times, start, end, split)
    
    # Find maximum (sequential, but array is small)
    for i in range(min_offset, n - min_offset):
        if ratios[i] > best_ratio:
            best_ratio = ratios[i]
            best_split = start + i
    
    return best_split, best_ratio


def find_change_points(
    abstimes: np.ndarray,
    confidence: float = 0.95,
    min_photons: int = 20,
    min_boundary_offset: int = 7
) -> list:
    """
    Main CPA algorithm - finds all significant change points.
    
    Returns list of Level objects.
    """
    from scipy import stats
    
    # Chi-squared threshold for given confidence
    threshold = stats.chi2.ppf(confidence, df=1)
    
    # Convert to float for Numba
    times = abstimes.astype(np.float64)
    
    # Find change points recursively
    boundaries = [0]
    _find_cps_recursive(
        times, 0, len(times), 
        threshold, min_photons, min_boundary_offset, 
        boundaries
    )
    boundaries.append(len(times))
    boundaries.sort()
    
    # Build Level objects from boundaries
    levels = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        level = Level(
            start_idx=start_idx,
            end_idx=end_idx,
            start_time_ns=int(abstimes[start_idx]),
            end_time_ns=int(abstimes[end_idx - 1]),
            num_photons=end_idx - start_idx
        )
        levels.append(level)
    
    return levels


def _find_cps_recursive(
    times: np.ndarray,
    start: int,
    end: int,
    threshold: float,
    min_photons: int,
    min_offset: int,
    boundaries: list
):
    """Recursive helper for CPA."""
    n = end - start
    if n < 2 * min_photons:
        return
    
    split, ratio = find_best_split(times, start, end, min_offset)
    
    if split < 0 or 2 * ratio < threshold:
        return
    
    boundaries.append(split)
    
    # Recurse on both segments
    _find_cps_recursive(times, start, split, threshold, min_photons, min_offset, boundaries)
    _find_cps_recursive(times, split, end, threshold, min_photons, min_offset, boundaries)
```

---

## Data Flow

### File Open Sequence

```
┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
│ Frontend│       │ FastAPI │       │ HDF5    │       │  Shared │
│         │       │         │       │ Reader  │       │  Memory │
└────┬────┘       └────┬────┘       └────┬────┘       └────┬────┘
     │                 │                 │                 │
     │ POST /files/open│                 │                 │
     │ {path: "..."}   │                 │                 │
     │────────────────>│                 │                 │
     │                 │                 │                 │
     │                 │ parse_h5_file() │                 │
     │                 │────────────────>│                 │
     │                 │                 │                 │
     │                 │                 │ Load abstimes   │
     │                 │                 │────────────────>│
     │                 │                 │                 │
     │                 │ Session created │                 │
     │                 │<────────────────│                 │
     │                 │                 │                 │
     │ {file_id,       │                 │                 │
     │  particles: [...]}                │                 │
     │<────────────────│                 │                 │
     │                 │                 │                 │
```

### Analysis Sequence (with Progress)

```
┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
│ Frontend│       │WebSocket│       │ FastAPI │       │  Pool   │
└────┬────┘       └────┬────┘       └────┬────┘       └────┬────┘
     │                 │                 │                 │
     │ Connect WS      │                 │                 │
     │────────────────>│                 │                 │
     │                 │                 │                 │
     │ POST /analysis/resolve           │                 │
     │─────────────────────────────────>│                 │
     │                 │                 │                 │
     │                 │                 │ Submit tasks    │
     │                 │                 │────────────────>│
     │                 │                 │                 │
     │                 │  task_started   │                 │
     │<────────────────│<────────────────│                 │
     │                 │                 │                 │
     │                 │                 │   result[0]     │
     │                 │    progress     │<────────────────│
     │<────────────────│<────────────────│                 │
     │                 │                 │                 │
     │                 │    progress     │   result[1]     │
     │<────────────────│<────────────────│<────────────────│
     │                 │                 │                 │
     │                 │ task_completed  │                 │
     │<────────────────│<────────────────│                 │
     │                 │                 │                 │
     │ 200 OK {summary}│                 │                 │
     │<─────────────────────────────────│                 │
```

---

## State Management

### Session State (Backend)

```python
# core/session.py
from dataclasses import dataclass, field
from typing import Optional
import json
import pickle
from pathlib import Path


@dataclass
class ParticleState:
    """Analysis state for a single particle."""
    id: int
    bin_size_ms: float = 10.0
    levels: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    fits: dict = field(default_factory=dict)  # {entity_id: FitResult}


@dataclass  
class Session:
    """Complete session state - what gets saved/loaded."""
    version: str = "2.0.0"
    h5_file_path: str = ""
    particles: dict[int, ParticleState] = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    global_grouping_enabled: bool = False
    global_groups: list = field(default_factory=list)
    
    def save(self, path: Path):
        """Save session to .smsa file (JSON + binary arrays)."""
        # Use JSON for structure, separate .npy for large arrays
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> 'Session':
        """Load session from .smsa file."""
        try:
            with open(path, 'rb') as f:
                session = pickle.load(f)
            return session
        except Exception as e:
            # Return empty session if load fails (recovery)
            return cls()
    
    @classmethod
    def load_safe(cls, path: Path) -> tuple['Session', list[str]]:
        """
        Load with error recovery.
        Returns (session, list of warnings).
        """
        warnings = []
        try:
            session = cls.load(path)
            
            # Validate H5 file still exists
            if not Path(session.h5_file_path).exists():
                warnings.append(f"Original HDF5 file not found: {session.h5_file_path}")
            
            return session, warnings
        except Exception as e:
            warnings.append(f"Failed to load session: {e}")
            return cls(), warnings
```

### UI State (Frontend - Zustand)

```typescript
// stores/sessionStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SessionState {
  fileId: string | null;
  filePath: string | null;
  isLoading: boolean;
  
  // Actions
  openFile: (path: string) => Promise<void>;
  closeFile: () => Promise<void>;
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      fileId: null,
      filePath: null,
      isLoading: false,
      
      openFile: async (path) => {
        set({ isLoading: true });
        try {
          const response = await api.openFile(path);
          set({ 
            fileId: response.file_id, 
            filePath: path,
            isLoading: false 
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },
      
      closeFile: async () => {
        const { fileId } = get();
        if (fileId) {
          await api.closeFile(fileId);
        }
        set({ fileId: null, filePath: null });
      }
    }),
    { name: 'full-sms-session' }
  )
);
```

```typescript
// stores/particleStore.ts
import { create } from 'zustand';

interface ParticleState {
  particles: ParticleSummary[];
  selectedIds: Set<number>;
  currentId: number | null;
  
  // Actions
  setParticles: (particles: ParticleSummary[]) => void;
  selectParticle: (id: number) => void;
  toggleSelection: (id: number) => void;
  selectAll: () => void;
  selectNone: () => void;
}

export const useParticleStore = create<ParticleState>()((set, get) => ({
  particles: [],
  selectedIds: new Set(),
  currentId: null,
  
  setParticles: (particles) => set({ particles }),
  
  selectParticle: (id) => set({ currentId: id }),
  
  toggleSelection: (id) => set((state) => {
    const newSelected = new Set(state.selectedIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    return { selectedIds: newSelected };
  }),
  
  selectAll: () => set((state) => ({
    selectedIds: new Set(state.particles.map(p => p.id))
  })),
  
  selectNone: () => set({ selectedIds: new Set() })
}));
```

---

## Performance Optimisation

### Backend Optimisations

| Technique | Application | Expected Speedup |
|-----------|-------------|------------------|
| **Numba JIT** | CPA inner loops, AHCA distance matrix | 10-100x vs pure Python |
| **Numba parallel** | Split point search, histogram binning | Nx (N = cores) |
| **Shared memory** | Particle abstimes between processes | Eliminates copy overhead |
| **Memory mapping** | Large HDF5 files | Reduces memory footprint |
| **Caching** | Compiled Numba functions, histograms | Eliminates recomputation |

### Frontend Optimisations

| Technique | Application | Benefit |
|-----------|-------------|---------|
| **WebGL plots** | `scattergl` for intensity traces | Millions of points |
| **Virtualisation** | Particle list (if >1000) | Smooth scrolling |
| **Memoisation** | Plot data transformation | Fewer re-renders |
| **Lazy loading** | Decay histograms per particle | Faster initial load |
| **Debouncing** | Bin size slider updates | Reduced API calls |

### Data Transfer Optimisations

```python
# Compress large arrays for API transfer
import lz4.frame

def compress_array(arr: np.ndarray) -> bytes:
    """Compress numpy array for network transfer."""
    return lz4.frame.compress(arr.tobytes())

def decompress_array(data: bytes, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Decompress array on client."""
    return np.frombuffer(lz4.frame.decompress(data), dtype=dtype).reshape(shape)
```

---

## Development Workflow

### Prerequisites

```bash
# Python
python -m pip install pipx
pipx install uv  # Fast Python package manager

# Node.js
brew install node@20

# Rust (for Tauri)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Backend Setup

```bash
cd backend
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run in development
uvicorn full_sms.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install

# Run in development (connects to localhost:8000)
npm run dev
```

### Desktop Build

```bash
# Build Python sidecar
cd backend
python build_sidecar.py

# Copy to Tauri
cp dist/full-sms-backend ../src-tauri/binaries/

# Build Tauri app
cd ..
npm run tauri build
```

### Testing

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=full_sms

# Frontend tests
cd frontend
npm run test

# E2E tests
npm run test:e2e
```

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Numba compatibility issues | Medium | High | Fall back to NumPy vectorisation; test early |
| Tauri + Python sidecar complexity | Medium | Medium | Alternative: PyWebView (simpler, less performant) |
| WebGL plotting limits | Low | Medium | Downsample for display, full data for export |
| HDF5 file format edge cases | Medium | High | Comprehensive test suite with real data files |
| Session save/load corruption | Medium | High | JSON with schema validation + automatic backups |
| Cross-platform build issues | Medium | Medium | CI/CD with GitHub Actions for all platforms |

### Alternative Paths

If Tauri proves problematic:

1. **PyWebView**: Python-native webview wrapper
   - Simpler packaging (all Python)
   - Slightly larger bundle (~30MB)
   - Less native feel

2. **Electron**: Heavy but battle-tested
   - Larger bundle (~150MB)
   - More documentation
   - Node.js backend instead of Python (would need rewrite)

3. **Desktop web app**: Just run in browser
   - No packaging needed
   - Less native experience
   - No file system access without FileSystem Access API

---

## Summary

This architecture provides:

- **Performance**: Numba JIT for near-C speed, multiprocessing for parallelism
- **Modern UX**: React with WebGL-accelerated Plotly charts
- **Small footprint**: Tauri produces ~15MB installers
- **Maintainability**: Python backend leverages team expertise
- **Offline-first**: Fully local, no network required
- **Robustness**: Type-safe API contracts, session recovery

The estimated development time is 14-16 weeks for a single developer, resulting in a maintainable, performant application that addresses the pain points of the current PyQt implementation.

