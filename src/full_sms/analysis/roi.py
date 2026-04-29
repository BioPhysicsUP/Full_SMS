"""Region of Interest (ROI) utilities for photon time traces.

Provides functions to slice photon arrays by time window and to
automatically detect bleaching for ROI trimming.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from full_sms.models.level import LevelData


def get_roi_photon_indices(
    abstimes: NDArray[np.uint64],
    roi_start_s: float,
    roi_end_s: float,
) -> tuple[int, int]:
    """Get the start and end indices for photons within an ROI.

    Uses np.searchsorted for O(log n) lookup on sorted abstimes.

    Args:
        abstimes: Sorted absolute photon arrival times in nanoseconds.
        roi_start_s: ROI start time in seconds.
        roi_end_s: ROI end time in seconds.

    Returns:
        Tuple of (start_index, end_index) where end_index is exclusive.
    """
    start_ns = int(roi_start_s * 1e9)
    end_ns = int(roi_end_s * 1e9)
    start_idx = int(np.searchsorted(abstimes, start_ns, side="left"))
    end_idx = int(np.searchsorted(abstimes, end_ns, side="right"))
    return start_idx, end_idx


def slice_by_roi(
    abstimes: NDArray[np.uint64],
    microtimes: NDArray[np.float64],
    roi: Optional[tuple[float, float]],
) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
    """Slice photon arrays to only include photons within the ROI.

    Args:
        abstimes: Absolute photon arrival times in nanoseconds.
        microtimes: Microtime values for each photon.
        roi: Tuple of (start_s, end_s) or None for no filtering.

    Returns:
        Tuple of (abstimes_roi, microtimes_roi). Returns the original
        arrays unmodified if roi is None.
    """
    if roi is None:
        return abstimes, microtimes

    start_idx, end_idx = get_roi_photon_indices(abstimes, roi[0], roi[1])
    return abstimes[start_idx:end_idx], microtimes[start_idx:end_idx]


def get_default_roi(abstimes: NDArray[np.uint64]) -> tuple[float, float]:
    """Get the default ROI spanning the full trace.

    Args:
        abstimes: Absolute photon arrival times in nanoseconds.

    Returns:
        Tuple of (start_s, end_s) covering the entire trace.
        Returns (0.0, 0.0) if abstimes is empty.
    """
    if len(abstimes) == 0:
        return (0.0, 0.0)
    return (float(abstimes[0]) / 1e9, float(abstimes[-1]) / 1e9)


def filter_levels_by_roi(
    levels: Sequence[LevelData],
    roi: tuple[float, float] | None,
) -> list[LevelData]:
    """Filter levels to only those fully contained within the ROI.

    A level is excluded if the ROI boundary cuts through it, i.e. if its
    start_time_ns < roi_start_ns or its end_time_ns > roi_end_ns.

    Args:
        levels: Sequence of LevelData objects.
        roi: Tuple of (start_s, end_s) in seconds, or None for no filtering.

    Returns:
        List of LevelData objects fully within the ROI.
        Returns all levels if roi is None.
    """
    if roi is None:
        return list(levels)

    roi_start_ns = int(roi[0] * 1e9)
    roi_end_ns = int(roi[1] * 1e9)

    return [
        level for level in levels
        if level.start_time_ns >= roi_start_ns and level.end_time_ns <= roi_end_ns
    ]


def auto_trim_roi(
    levels: list,
    threshold_cps: float,
    min_duration_s: float,
) -> Optional[float]:
    """Detect bleaching and suggest a new ROI end time.

    Scans levels from end backwards. Accumulates dwell time of consecutive
    levels with intensity below threshold. If accumulated time exceeds
    min_duration, returns the boundary time as the new ROI end.

    Args:
        levels: List of LevelData objects (must have intensity_cps,
            dwell_time_s, and start_time_ns attributes).
        threshold_cps: Intensity threshold in counts per second.
            Levels below this are considered bleached.
        min_duration_s: Minimum accumulated duration of low-intensity
            levels to trigger trimming.

    Returns:
        The suggested ROI end time in seconds, or None if no
        bleaching was detected.
    """
    if not levels:
        return None

    # Sort levels by start time
    sorted_levels = sorted(levels, key=lambda lv: lv.start_time_ns)

    accumulated_s = 0.0

    # Scan backwards
    for level in reversed(sorted_levels):
        if level.intensity_cps < threshold_cps:
            accumulated_s += level.dwell_time_s
            if accumulated_s >= min_duration_s:
                # Find the boundary: first low-intensity level in the trailing run
                # Walk backwards to find where the run starts
                run_start_time_ns = level.start_time_ns
                return float(run_start_time_ns) / 1e9
        else:
            # Break: a high-intensity level interrupts the trailing run
            break

    return None
