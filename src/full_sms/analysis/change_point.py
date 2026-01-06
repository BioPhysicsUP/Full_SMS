"""Change point detection using the Watkins & Yang algorithm.

Implements the weighted likelihood ratio method for detecting intensity change
points in single-molecule spectroscopy data.

Based on: Watkins & Yang, "Detection of Intensity Change Points in Time-Resolved
Single-Molecule Measurements", J. Phys. Chem. B 2005, 109, 617-628.
http://pubs.acs.org/doi/abs/10.1021/jp0467548

Joshua Botha, University of Pretoria (original implementation)
"""

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Tuple

import numpy as np
from numba import jit
from numpy.typing import NDArray

from full_sms.models import LevelData


class ConfidenceLevel(Enum):
    """Supported confidence levels for change point detection."""

    CONF_69 = 0.69
    CONF_90 = 0.90
    CONF_95 = 0.95
    CONF_99 = 0.99


@dataclass(frozen=True)
class CPAParams:
    """Parameters for change point analysis.

    Attributes:
        confidence: Confidence level for detection threshold.
        min_photons: Minimum photons in a segment to analyze (default 20).
        min_boundary_offset: Minimum distance from segment edges (default 7).
    """

    confidence: ConfidenceLevel
    min_photons: int = 20
    min_boundary_offset: int = 7


@dataclass
class ChangePointResult:
    """Result from change point analysis.

    Attributes:
        change_point_indices: Array of change point indices in the photon array.
        levels: List of LevelData objects for each detected level.
        num_change_points: Number of detected change points.
        confidence_regions: List of (start, end) tuples for confidence regions.
    """

    change_point_indices: NDArray[np.int64]
    levels: list[LevelData]
    num_change_points: int
    confidence_regions: list[tuple[int, int]]


class TauData:
    """Loads and provides access to tau threshold values.

    The tau_a values are detection thresholds for the weighted likelihood ratio.
    The tau_b values define confidence regions around detected change points.
    """

    _tau_files = {
        ConfidenceLevel.CONF_69: ("Ta-69.txt", "Tb-69.txt"),
        ConfidenceLevel.CONF_90: ("Ta-90.txt", "Tb-90.txt"),
        ConfidenceLevel.CONF_95: ("Ta-95.txt", "Tb-95.txt"),
        ConfidenceLevel.CONF_99: ("Ta-99.txt", "Tb-99.txt"),
    }

    def __init__(self, confidence: ConfidenceLevel):
        """Load tau data for the specified confidence level.

        Args:
            confidence: Confidence level to load thresholds for.
        """
        self.confidence = confidence
        ta_file, tb_file = self._tau_files[confidence]

        # Load tau data from package resources
        tau_data_dir = files("full_sms.data") / "tau_data"
        ta_path = tau_data_dir / ta_file
        tb_path = tau_data_dir / tb_file

        self._tau_a = np.loadtxt(str(ta_path), usecols=1)
        self._tau_b = np.loadtxt(str(tb_path), usecols=1)

    def get_tau_a(self, n: int) -> float:
        """Get detection threshold for n data points.

        Args:
            n: Number of data points in the segment.

        Returns:
            tau_a threshold value.
        """
        return self._tau_a[n]

    def get_tau_b(self, n: int) -> float:
        """Get confidence region threshold for n data points.

        Args:
            n: Number of data points in the segment.

        Returns:
            tau_b threshold value.
        """
        return self._tau_b[n]


@lru_cache(maxsize=1)
def _compute_sums(n_max: int = 1000, n_min: int = 10) -> dict:
    """Precompute sums needed for weighted likelihood ratio calculation.

    These sums are used in equations 6 and 7 of the Watkins & Yang paper.

    Args:
        n_max: Maximum segment size (default 1000).
        n_min: Minimum segment size (default 10).

    Returns:
        Dictionary containing precomputed sum arrays.
    """
    n_range = n_max - n_min

    # Precompute 1/j and 1/j^2 for efficient calculation
    j_vals = np.arange(1, n_max)
    inv_j = 1.0 / j_vals
    inv_j_sq = 1.0 / (j_vals**2)

    # Cumulative sums for efficient computation
    cum_inv_j = np.cumsum(inv_j[::-1])[::-1]  # sum from j to end
    cum_inv_j_sq = np.cumsum(inv_j_sq[::-1])[::-1]

    # sig_e values: pi^2/6 - sum(1/j^2 for j in 1..n-1)
    sums_sig_e = np.zeros(n_range)
    pi_sq_6 = (np.pi**2) / 6
    for i, n in enumerate(range(n_min + 1, n_max + 1)):
        if n > 1:
            sums_sig_e[i] = pi_sq_6 - np.sum(inv_j_sq[: n - 1])

    # u_k, u_n_k, v2_k, v2_n_k values
    sums_u_k = np.zeros((n_max, n_range))
    sums_u_n_k = np.zeros((n_max, n_range))
    sums_v2_k = np.zeros((n_max, n_range))
    sums_v2_n_k = np.zeros((n_max, n_range))

    for i, n in enumerate(range(n_min + 1, n_max + 1)):
        for k in range(1, n):
            # u_k = -sum(1/j for j in k..n-1)
            if k < n:
                sums_u_k[k - 1, i] = -np.sum(inv_j[k - 1 : n - 1])
            # u_n_k = -sum(1/j for j in n-k..n-1)
            if n - k > 0:
                sums_u_n_k[k - 1, i] = -np.sum(inv_j[n - k - 1 : n - 1])
            # v2_k = sum(1/j^2 for j in k..n-1)
            if k < n:
                sums_v2_k[k - 1, i] = np.sum(inv_j_sq[k - 1 : n - 1])
            # v2_n_k = sum(1/j^2 for j in n-k..n-1)
            if n - k > 0:
                sums_v2_n_k[k - 1, i] = np.sum(inv_j_sq[n - k - 1 : n - 1])

    return {
        "sums_u_k": sums_u_k,
        "sums_u_n_k": sums_u_n_k,
        "sums_v2_k": sums_v2_k,
        "sums_v2_n_k": sums_v2_n_k,
        "sums_sig_e": sums_sig_e,
        "n_min": n_min,
        "n_max": n_max,
    }


def _get_sums_set(sums: dict, n: int, k: int) -> dict:
    """Get all sum values for given n and k.

    Args:
        sums: Precomputed sums dictionary.
        n: Number of points in segment.
        k: Point index within segment.

    Returns:
        Dictionary with u_k, u_n_k, v2_k, v2_n_k values.
    """
    n_min = sums["n_min"]
    row = k - 1
    col = n - n_min - 1

    return {
        "u_k": sums["sums_u_k"][row, col],
        "u_n_k": sums["sums_u_n_k"][row, col],
        "v2_k": sums["sums_v2_k"][row, col],
        "v2_n_k": sums["sums_v2_n_k"][row, col],
    }


def _get_sig_e(sums: dict, n: int) -> float:
    """Get sig_e value for given n."""
    n_min = sums["n_min"]
    return sums["sums_sig_e"][n - n_min - 1]


@jit(nopython=True, cache=True)
def _compute_wlr_array(
    time_data: np.ndarray,
    sums_u_k: np.ndarray,
    sums_u_n_k: np.ndarray,
    sums_v2_k: np.ndarray,
    sums_v2_n_k: np.ndarray,
    sig_e: float,
    n: int,
    col: int,
) -> np.ndarray:
    """JIT-compiled weighted likelihood ratio computation.

    Computes WLR for each potential change point k from 2 to n-2.
    Based on equations 4-7 of Watkins & Yang 2005.

    Args:
        time_data: Segment time data (normalized: 0 to period).
        sums_u_k: Precomputed u_k sum array.
        sums_u_n_k: Precomputed u_n_k sum array.
        sums_v2_k: Precomputed v2_k sum array.
        sums_v2_n_k: Precomputed v2_n_k sum array.
        sig_e: Precomputed sig_e value for this n.
        n: Number of points in segment.
        col: Column index into sum arrays (n - n_min - 1).

    Returns:
        Array of WLR values for each k.
    """
    wlr = np.zeros(n, dtype=np.float64)
    ini_time = time_data[0]
    period = time_data[-1] - ini_time

    if period == 0:
        return wlr

    for k in range(2, n - 1):
        row = k - 1

        # V_k: normalized time position (eq. 4)
        cap_v_k = (time_data[k] - ini_time) / period

        # Avoid log(0) or log(1)
        if cap_v_k <= 0 or cap_v_k >= 1:
            continue

        u_k = sums_u_k[row, col]
        u_n_k = sums_u_n_k[row, col]
        v2_k = sums_v2_k[row, col]
        v2_n_k = sums_v2_n_k[row, col]

        # Log-likelihood ratio minus expected value (eq. 6)
        l0_minus_expec_l0 = (
            -2 * k * np.log(cap_v_k)
            + 2 * k * u_k
            - 2 * (n - k) * np.log(1 - cap_v_k)
            + 2 * (n - k) * u_n_k
        )

        # Standard deviation (eq. 7, with errata correction)
        sigma_k_sq = (
            4 * (k**2) * v2_k
            + 4 * ((n - k) ** 2) * v2_n_k
            - 8 * k * (n - k) * sig_e
        )
        if sigma_k_sq <= 0:
            continue
        sigma_k = np.sqrt(sigma_k_sq)

        # Weight factor (after eq. 6)
        w_k = 0.5 * np.log((4 * k * (n - k)) / (n**2))

        # Weighted likelihood ratio (eq. 6)
        wlr[k] = l0_minus_expec_l0 / sigma_k + w_k

    return wlr


def _weighted_likelihood_ratio(
    abstimes: NDArray[np.float64],
    seg_start: int,
    seg_end: int,
    tau_data: TauData,
    sums: dict,
    params: CPAParams,
) -> Tuple[bool, int | None, tuple[int, int] | None]:
    """Calculate weighted likelihood ratio and detect change point.

    Based on equations 4-7 of Watkins & Yang 2005.

    Args:
        abstimes: Array of absolute photon arrival times in nanoseconds.
        seg_start: Start index of segment to analyze.
        seg_end: End index of segment to analyze.
        tau_data: TauData instance with threshold values.
        sums: Precomputed sums dictionary.
        params: Analysis parameters.

    Returns:
        Tuple of (change_point_found, change_point_index, confidence_region).
        If no change point found, returns (False, None, None).
    """
    n = seg_end - seg_start

    # Check minimum segment size
    if n < params.min_photons:
        return False, None, None

    if n > 1000:
        raise ValueError(f"Segment size {n} exceeds maximum of 1000 points")

    # Get time data for this segment
    time_data = abstimes[seg_start:seg_end]

    if time_data[-1] == time_data[0]:
        return False, None, None

    # Calculate weighted likelihood ratio using JIT-compiled function
    n_min = sums["n_min"]
    col = n - n_min - 1
    sig_e = _get_sig_e(sums, n)

    wlr = _compute_wlr_array(
        time_data,
        sums["sums_u_k"],
        sums["sums_u_n_k"],
        sums["sums_v2_k"],
        sums["sums_v2_n_k"],
        sig_e,
        n,
        col,
    )

    # Find maximum WLR
    max_ind_local = int(np.argmax(wlr))

    # Check if change point is valid
    if (
        max_ind_local >= params.min_boundary_offset
        and n - max_ind_local >= params.min_boundary_offset
        and wlr[max_ind_local] >= tau_data.get_tau_a(n)
    ):
        # Change point detected
        cpt = max_ind_local + seg_start

        # Calculate confidence region using tau_b
        tau_b_inv = wlr[max_ind_local] - tau_data.get_tau_b(n)
        region_mask = wlr >= tau_b_inv
        region_indices = np.where(region_mask)[0]

        if len(region_indices) > 0:
            conf_region = (
                region_indices[0] + seg_start,
                region_indices[-1] + seg_start,
            )
        else:
            conf_region = (cpt, cpt)

        return True, cpt, conf_region

    return False, None, None


def _find_all_change_points(
    abstimes: NDArray[np.float64],
    tau_data: TauData,
    sums: dict,
    params: CPAParams,
    end_photon: int | None = None,
) -> Tuple[NDArray[np.int64], list[tuple[int, int]]]:
    """Find all change points recursively.

    Uses a sliding window approach with recursive splitting at detected
    change points.

    Args:
        abstimes: Array of absolute photon arrival times in nanoseconds.
        tau_data: TauData instance with threshold values.
        sums: Precomputed sums dictionary.
        params: Analysis parameters.
        end_photon: Optional end index (exclusive). If None, uses all photons.

    Returns:
        Tuple of (change_point_indices, confidence_regions).
    """
    num_photons = len(abstimes)
    if end_photon is None:
        end_photon = num_photons

    if num_photons < 200:
        return np.array([], dtype=np.int64), []

    cpt_indices: list[int] = []
    conf_regions: list[tuple[int, int]] = []

    def find_recursive(seg_start: int, seg_end: int, side: str | None = None):
        """Recursive helper to find change points in a segment."""
        n = seg_end - seg_start

        if n < params.min_photons:
            return

        found, cpt, conf_region = _weighted_likelihood_ratio(
            abstimes, seg_start, seg_end, tau_data, sums, params
        )

        if found and cpt is not None and conf_region is not None:
            cpt_indices.append(cpt)
            conf_regions.append(conf_region)

            # Recurse on left side of change point
            if cpt - seg_start >= params.min_photons:
                find_recursive(seg_start, cpt, side="left")

            # Recurse on right side of change point
            if seg_end - cpt >= params.min_photons:
                find_recursive(cpt, seg_end, side="right")

    # Process in windows of at most 1000 photons
    window_start = 0
    overlap = 200  # Overlap to catch change points near window boundaries

    while window_start < end_photon:
        window_end = min(window_start + 1000, end_photon)

        find_recursive(window_start, window_end)

        # Move window, but overlap with previous
        if window_end >= end_photon:
            break
        window_start = window_end - overlap

    # Sort and deduplicate change points
    if cpt_indices:
        sorted_indices = np.argsort(cpt_indices)
        cpt_array = np.array([cpt_indices[i] for i in sorted_indices], dtype=np.int64)
        sorted_regions = [conf_regions[i] for i in sorted_indices]

        # Remove duplicates
        unique_cpts, unique_indices = np.unique(cpt_array, return_index=True)
        unique_regions = [sorted_regions[i] for i in unique_indices]

        return unique_cpts, unique_regions

    return np.array([], dtype=np.int64), []


def find_change_points(
    abstimes: NDArray[np.float64],
    confidence: ConfidenceLevel | float = ConfidenceLevel.CONF_95,
    min_photons: int = 20,
    min_boundary_offset: int = 7,
    end_time_ns: float | None = None,
) -> ChangePointResult:
    """Find intensity change points in photon arrival time data.

    Implements the Watkins & Yang (2005) weighted likelihood ratio method
    for detecting intensity change points in single-molecule data.

    Args:
        abstimes: Array of absolute photon arrival times in nanoseconds.
        confidence: Confidence level for detection. Can be a ConfidenceLevel
            enum or a float (0.69, 0.90, 0.95, or 0.99).
        min_photons: Minimum photons in a segment to analyze (default 20).
        min_boundary_offset: Minimum distance from segment edges (default 7).
        end_time_ns: Optional end time in nanoseconds. If provided, only
            analyzes photons up to this time.

    Returns:
        ChangePointResult containing detected change points and levels.

    Raises:
        ValueError: If confidence level is not supported.

    Example:
        >>> abstimes = np.array([...])  # photon arrival times in ns
        >>> result = find_change_points(abstimes, confidence=0.95)
        >>> print(f"Found {result.num_change_points} change points")
        >>> for level in result.levels:
        ...     print(f"Level {level.id}: {level.intensity_cps:.0f} cps")
    """
    # Convert float confidence to enum
    if isinstance(confidence, float):
        confidence_map = {
            0.69: ConfidenceLevel.CONF_69,
            0.90: ConfidenceLevel.CONF_90,
            0.95: ConfidenceLevel.CONF_95,
            0.99: ConfidenceLevel.CONF_99,
        }
        if confidence not in confidence_map:
            raise ValueError(
                f"Unsupported confidence level: {confidence}. "
                f"Must be one of {list(confidence_map.keys())}"
            )
        confidence = confidence_map[confidence]

    # Handle empty input
    if len(abstimes) == 0:
        return ChangePointResult(
            change_point_indices=np.array([], dtype=np.int64),
            levels=[],
            num_change_points=0,
            confidence_regions=[],
        )

    # Determine end photon index
    end_photon = None
    if end_time_ns is not None:
        end_photon = np.searchsorted(abstimes, end_time_ns, side="right")
        if end_photon == 0:
            return ChangePointResult(
                change_point_indices=np.array([], dtype=np.int64),
                levels=[],
                num_change_points=0,
                confidence_regions=[],
            )

    num_photons = end_photon if end_photon is not None else len(abstimes)

    # Check minimum data requirement
    if num_photons < 200:
        # Not enough photons, return single level if any
        if num_photons >= min_photons:
            level = LevelData.from_photon_indices(
                abstimes=abstimes,
                start_index=0,
                end_index=num_photons - 1,
            )
            return ChangePointResult(
                change_point_indices=np.array([], dtype=np.int64),
                levels=[level],
                num_change_points=0,
                confidence_regions=[],
            )
        return ChangePointResult(
            change_point_indices=np.array([], dtype=np.int64),
            levels=[],
            num_change_points=0,
            confidence_regions=[],
        )

    # Load tau thresholds and precomputed sums
    tau_data = TauData(confidence)
    sums = _compute_sums()
    params = CPAParams(
        confidence=confidence,
        min_photons=min_photons,
        min_boundary_offset=min_boundary_offset,
    )

    # Find all change points
    cpt_indices, conf_regions = _find_all_change_points(
        abstimes, tau_data, sums, params, end_photon
    )

    # Create levels from change points
    levels = _create_levels_from_change_points(
        abstimes, cpt_indices, num_photons
    )

    return ChangePointResult(
        change_point_indices=cpt_indices,
        levels=levels,
        num_change_points=len(cpt_indices),
        confidence_regions=conf_regions,
    )


def _create_levels_from_change_points(
    abstimes: NDArray[np.float64],
    cpt_indices: NDArray[np.int64],
    num_photons: int,
) -> list[LevelData]:
    """Create LevelData objects from change point indices.

    Args:
        abstimes: Array of absolute photon arrival times in nanoseconds.
        cpt_indices: Array of change point indices.
        num_photons: Total number of photons (may be less than len(abstimes)).

    Returns:
        List of LevelData objects, one for each level.
    """
    if len(cpt_indices) == 0:
        # Single level spanning all photons
        if num_photons > 0:
            return [
                LevelData.from_photon_indices(
                    abstimes=abstimes,
                    start_index=0,
                    end_index=num_photons - 1,
                )
            ]
        return []

    levels = []
    num_levels = len(cpt_indices) + 1

    for i in range(num_levels):
        if i == 0:
            # First level: from start to first change point
            start_idx = 0
            end_idx = cpt_indices[0] - 1
        elif i == num_levels - 1:
            # Last level: from last change point to end
            start_idx = cpt_indices[-1]
            end_idx = num_photons - 1
        else:
            # Middle level: between two change points
            start_idx = cpt_indices[i - 1]
            end_idx = cpt_indices[i] - 1

        # Ensure valid indices
        if start_idx <= end_idx and end_idx < len(abstimes):
            level = LevelData.from_photon_indices(
                abstimes=abstimes,
                start_index=start_idx,
                end_index=end_idx,
            )
            levels.append(level)

    return levels
