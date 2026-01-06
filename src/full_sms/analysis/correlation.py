"""Second-order photon correlation (antibunching) analysis.

Implements g2 correlation for dual-channel single-molecule spectroscopy data.

The correlation function g2(tau) measures the probability of detecting a photon
at time t+tau given a detection at time t. For single quantum emitters,
g2(0) < 1 indicates antibunching (sub-Poissonian statistics), proving single
photon emission.

References:
    Watkins, L.P., Yang, H. (2004). Information bounds and optimal analysis of
    dynamic single molecule measurements. Biophys J, 86(6), 4015-4024.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import jit

__all__ = [
    "calculate_g2",
    "CorrelationResult",
    "rebin_correlation",
]


@dataclass(frozen=True)
class CorrelationResult:
    """Result of second-order correlation analysis.

    Attributes:
        tau: Delay time bins in nanoseconds (bin centers).
        g2: Correlation histogram values (unnormalized counts).
        events: Raw delay times for all correlation events, useful for rebinning.
        window_ns: Correlation window in nanoseconds.
        binsize_ns: Histogram bin size in nanoseconds.
        num_photons_ch1: Number of photons in channel 1.
        num_photons_ch2: Number of photons in channel 2.
    """

    tau: np.ndarray
    g2: np.ndarray
    events: np.ndarray
    window_ns: float
    binsize_ns: float
    num_photons_ch1: int
    num_photons_ch2: int

    @property
    def num_events(self) -> int:
        """Total number of correlation events detected."""
        return len(self.events)

    def rebin(self, new_binsize_ns: float, new_window_ns: float | None = None) -> "CorrelationResult":
        """Create a new result with different binning.

        Args:
            new_binsize_ns: New bin size in nanoseconds.
            new_window_ns: New window size in nanoseconds. If None, uses original window.

        Returns:
            New CorrelationResult with rebinned histogram.
        """
        return rebin_correlation(self, new_binsize_ns, new_window_ns)


def calculate_g2(
    abstimes1: np.ndarray,
    abstimes2: np.ndarray,
    microtimes1: np.ndarray,
    microtimes2: np.ndarray,
    window_ns: float = 500.0,
    binsize_ns: float = 0.5,
    difftime_ns: float = 0.0,
) -> CorrelationResult:
    """Calculate second-order correlation function g2(tau).

    Computes the cross-correlation between photon arrival times from two
    detection channels. The correlation is symmetric around tau=0, with
    channel 1 defined as the "start" channel.

    The algorithm merges both channels into a single sorted time stream,
    then for each photon finds all cross-channel coincidences within the
    correlation window.

    Args:
        abstimes1: Absolute arrival times for channel 1 in nanoseconds.
        abstimes2: Absolute arrival times for channel 2 in nanoseconds.
        microtimes1: TCSPC micro times for channel 1 in nanoseconds.
        microtimes2: TCSPC micro times for channel 2 in nanoseconds.
        window_ns: Correlation window size in nanoseconds. Events with
            |tau| > window are excluded. Default 500 ns.
        binsize_ns: Histogram bin size in nanoseconds. Default 0.5 ns.
        difftime_ns: Time offset between channels (ch1 - ch2) in nanoseconds.
            Use to correct for cable delays between TCSPC cards. Default 0.

    Returns:
        CorrelationResult containing:
        - tau: Bin centers in nanoseconds
        - g2: Correlation histogram (counts per bin)
        - events: Raw delay times for potential rebinning

    Example:
        >>> result = calculate_g2(abs1, abs2, micro1, micro2, window_ns=1000)
        >>> plt.plot(result.tau, result.g2)
        >>> # Antibunching dip visible at tau=0 for single emitters
    """
    # Validate inputs
    abstimes1 = np.asarray(abstimes1, dtype=np.float64)
    abstimes2 = np.asarray(abstimes2, dtype=np.float64)
    microtimes1 = np.asarray(microtimes1, dtype=np.float64)
    microtimes2 = np.asarray(microtimes2, dtype=np.float64)

    if len(abstimes1) != len(microtimes1):
        raise ValueError(
            f"Channel 1 arrays must have same length: "
            f"abstimes={len(abstimes1)}, microtimes={len(microtimes1)}"
        )
    if len(abstimes2) != len(microtimes2):
        raise ValueError(
            f"Channel 2 arrays must have same length: "
            f"abstimes={len(abstimes2)}, microtimes={len(microtimes2)}"
        )

    # Handle empty inputs
    if len(abstimes1) == 0 or len(abstimes2) == 0:
        num_bins = max(1, int(2 * window_ns / binsize_ns))
        tau = np.linspace(-window_ns, window_ns, num_bins)
        return CorrelationResult(
            tau=tau,
            g2=np.zeros(num_bins, dtype=np.int64),
            events=np.array([], dtype=np.float64),
            window_ns=window_ns,
            binsize_ns=binsize_ns,
            num_photons_ch1=len(abstimes1),
            num_photons_ch2=len(abstimes2),
        )

    # Combine absolute and micro times, apply channel offset
    times1 = abstimes1 + microtimes1
    times2 = abstimes2 + microtimes2 + difftime_ns

    # Merge channels into single sorted time stream
    size1 = len(times1)
    size2 = len(times2)

    # Channel markers: 0 for channel 1, 1 for channel 2
    channels = np.concatenate([np.zeros(size1, dtype=np.int32), np.ones(size2, dtype=np.int32)])
    all_times = np.concatenate([times1, times2])

    # Sort by arrival time
    sort_idx = np.argsort(all_times)
    all_times = all_times[sort_idx]
    channels = channels[sort_idx]

    # Calculate correlation events using JIT-compiled function
    events = _compute_correlation_events(all_times, channels, window_ns)

    # Build histogram
    num_bins = int(2 * window_ns / binsize_ns)
    g2, bin_edges = np.histogram(events, bins=num_bins, range=(-window_ns, window_ns))

    # Compute bin centers
    tau = (bin_edges[:-1] + bin_edges[1:]) / 2

    return CorrelationResult(
        tau=tau,
        g2=g2,
        events=events,
        window_ns=window_ns,
        binsize_ns=binsize_ns,
        num_photons_ch1=size1,
        num_photons_ch2=size2,
    )


@jit(nopython=True, cache=True)
def _compute_correlation_events(
    all_times: np.ndarray,
    channels: np.ndarray,
    window: float,
) -> np.ndarray:
    """JIT-compiled inner loop for correlation calculation.

    For each photon, find all cross-channel coincidences within the window.
    Channel 0 (channel 1) is the start channel, so positive tau means
    channel 2 photon arrived after channel 1.

    Args:
        all_times: Sorted arrival times (all channels merged).
        channels: Channel index for each photon (0 or 1).
        window: Correlation window in nanoseconds.

    Returns:
        Array of delay times (tau) for all correlation events.
    """
    n = len(all_times)
    # Pre-allocate with estimated capacity
    # Worst case: every pair within window is a cross-channel event
    # Typical: much fewer events
    max_events = min(n * 100, 10_000_000)  # Cap memory usage
    events = np.empty(max_events, dtype=np.float64)
    event_count = 0

    for i in range(n):
        time_i = all_times[i]
        ch_i = channels[i]

        # Look forward for coincidences
        for j in range(i + 1, n):
            time_j = all_times[j]
            dt = time_j - time_i

            # Beyond window, stop searching
            if dt > window:
                break

            ch_j = channels[j]

            # Skip same-channel events
            if ch_i == ch_j:
                continue

            # Compute signed delay time
            # Channel 0 is start: positive tau = ch1 before ch2
            if ch_i == 0:
                tau = dt  # ch1 -> ch2: positive
            else:
                tau = -dt  # ch2 -> ch1: negative

            # Store event
            if event_count < max_events:
                events[event_count] = tau
                event_count += 1

    return events[:event_count].copy()


def rebin_correlation(
    result: CorrelationResult,
    new_binsize_ns: float,
    new_window_ns: float | None = None,
) -> CorrelationResult:
    """Rebin a correlation result with different parameters.

    Uses the stored raw events to create a new histogram without
    recalculating the correlation.

    Args:
        result: Original correlation result.
        new_binsize_ns: New bin size in nanoseconds.
        new_window_ns: New window size. If None, uses original window.

    Returns:
        New CorrelationResult with rebinned histogram.
    """
    if new_window_ns is None:
        new_window_ns = result.window_ns

    # Can't expand window beyond original data
    if new_window_ns > result.window_ns:
        raise ValueError(
            f"Cannot expand window from {result.window_ns} to {new_window_ns} ns. "
            "Rebinning can only narrow the window."
        )

    num_bins = int(2 * new_window_ns / new_binsize_ns)
    g2, bin_edges = np.histogram(
        result.events, bins=num_bins, range=(-new_window_ns, new_window_ns)
    )
    tau = (bin_edges[:-1] + bin_edges[1:]) / 2

    return CorrelationResult(
        tau=tau,
        g2=g2,
        events=result.events,
        window_ns=new_window_ns,
        binsize_ns=new_binsize_ns,
        num_photons_ch1=result.num_photons_ch1,
        num_photons_ch2=result.num_photons_ch2,
    )
