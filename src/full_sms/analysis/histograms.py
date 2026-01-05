"""Histogram utilities for intensity binning and decay histograms.

This module provides functions for creating histograms from photon arrival times:
- bin_photons: Bin absolute arrival times for intensity traces
- build_decay_histogram: Build TCSPC decay histograms from microtimes
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def bin_photons(
    abstimes: NDArray[np.float64],
    bin_size_ms: float,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Bin photon arrival times to create an intensity trace.

    Takes absolute photon arrival times (in nanoseconds) and bins them into
    time windows of the specified size. This produces the intensity trace
    commonly used in single-molecule spectroscopy.

    Args:
        abstimes: Absolute photon arrival times in nanoseconds.
        bin_size_ms: Size of each time bin in milliseconds.

    Returns:
        Tuple of (times, counts) where:
        - times: Array of bin start times in milliseconds
        - counts: Array of photon counts per bin

    Example:
        >>> abstimes = np.array([1e6, 2e6, 2.5e6, 5e6])  # ns
        >>> times, counts = bin_photons(abstimes, bin_size_ms=1.0)
        >>> # times will be [0, 1, 2, 3, 4, 5] ms
        >>> # counts will show photon counts in each 1ms window
    """
    if len(abstimes) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    # Convert bin size from ms to ns
    bin_size_ns = bin_size_ms * 1e6

    # Calculate bin edges
    max_time = np.max(abstimes)
    num_bins = int(np.ceil(max_time / bin_size_ns))
    bin_edges = np.arange(0, (num_bins + 1) * bin_size_ns, bin_size_ns)

    # Use numpy histogram for efficient binning
    counts, _ = np.histogram(abstimes, bins=bin_edges)

    # Bin times are the start of each bin, in milliseconds
    times = bin_edges[:-1] / 1e6

    return times, counts.astype(np.int64)


def build_decay_histogram(
    microtimes: NDArray[np.float64],
    channelwidth: float,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a TCSPC decay histogram from microtime data.

    Takes microtime (TCSPC) arrival times and creates a histogram with bins
    matching the hardware channel width. This produces the decay curve used
    for fluorescence lifetime analysis.

    Args:
        microtimes: Microtime (TCSPC) arrival times in nanoseconds.
        channelwidth: Hardware TCSPC bin width in nanoseconds.
        tmin: Optional minimum time for histogram range. If None, uses
            minimum microtime value.
        tmax: Optional maximum time for histogram range. If None, uses
            maximum microtime value.

    Returns:
        Tuple of (t, counts) where:
        - t: Array of bin center times in nanoseconds
        - counts: Array of photon counts per bin

    Example:
        >>> microtimes = np.array([1.5, 2.3, 2.7, 5.1])  # ns
        >>> t, counts = build_decay_histogram(microtimes, channelwidth=0.1)
        >>> # t will be time points at 0.1ns intervals
        >>> # counts will show decay profile
    """
    if len(microtimes) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    # Determine histogram range
    if tmin is None:
        tmin = np.min(microtimes)
    if tmax is None:
        tmax = np.max(microtimes)

    # Ensure positive times
    tmin = max(0, tmin)

    # Create bin edges aligned to channel width
    bin_edges = np.arange(tmin, tmax + channelwidth, channelwidth)

    if len(bin_edges) < 2:
        # Not enough range for histogram
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    # Build histogram
    counts, edges = np.histogram(microtimes, bins=bin_edges)

    # Time values are bin starts (removing the last edge)
    t = edges[:-1]

    # Filter out negative times and corresponding counts
    positive_mask = t > 0
    t = t[positive_mask]
    counts = counts[positive_mask]

    return t, counts.astype(np.int64)


def compute_intensity_cps(
    counts: NDArray[np.int64],
    bin_size_ms: float,
) -> NDArray[np.float64]:
    """Convert binned photon counts to intensity in counts per second.

    Args:
        counts: Array of photon counts per bin.
        bin_size_ms: Size of each time bin in milliseconds.

    Returns:
        Array of intensity values in counts per second (cps).
    """
    if len(counts) == 0:
        return np.array([], dtype=np.float64)

    # Convert from counts/bin to counts/second
    bin_size_s = bin_size_ms / 1000.0
    return counts.astype(np.float64) / bin_size_s


def rebin_histogram(
    t: NDArray[np.float64],
    counts: NDArray[np.int64],
    factor: int,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Rebin a histogram by combining adjacent bins.

    This is useful for smoothing noisy decay histograms or reducing
    the number of points for faster fitting.

    Args:
        t: Array of time values.
        counts: Array of counts.
        factor: Number of bins to combine. Must be >= 1.

    Returns:
        Tuple of (new_t, new_counts) with reduced resolution.

    Raises:
        ValueError: If factor is less than 1.
    """
    if factor < 1:
        raise ValueError(f"Rebin factor must be >= 1, got {factor}")

    if factor == 1 or len(counts) == 0:
        return t.copy(), counts.copy()

    # Calculate new size (truncate incomplete bins at end)
    new_size = len(counts) // factor
    if new_size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    # Truncate to exact multiple of factor
    truncated_counts = counts[: new_size * factor]
    truncated_t = t[: new_size * factor]

    # Reshape and sum counts
    new_counts = truncated_counts.reshape(new_size, factor).sum(axis=1)

    # Take first time value of each group (bin start)
    new_t = truncated_t.reshape(new_size, factor)[:, 0]

    return new_t, new_counts.astype(np.int64)
