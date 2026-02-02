"""Picklable task functions for parallel processing.

These functions are designed to be submitted to AnalysisPool for execution
in worker processes. They must be module-level functions (not methods or
lambdas) to be picklable by the multiprocessing machinery.

Each task function:
- Takes a parameters dictionary as input
- Returns a result dictionary
- Handles errors gracefully (returns error info in result)

Usage:
    from full_sms.workers import AnalysisPool
    from full_sms.workers.tasks import run_cpa_task

    with AnalysisPool() as pool:
        params = {
            "abstimes": abstimes_array,
            "confidence": 0.95,
        }
        future = pool.submit(run_cpa_task, params)
        result = future.result()
        if result.success:
            cpa_result = result.value
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray


def run_cpa_task(params: dict[str, Any]) -> dict[str, Any]:
    """Run change point analysis in a worker process.

    Parameters:
        params: Dictionary containing:
            - abstimes (NDArray[np.float64]): Photon arrival times in nanoseconds.
            - confidence (float): Confidence level (0.69, 0.90, 0.95, or 0.99).
            - min_photons (int, optional): Minimum photons per segment (default 20).
            - min_boundary_offset (int, optional): Minimum edge distance (default 7).
            - end_time_ns (float, optional): Analysis end time in nanoseconds.
            - measurement_id (Any, optional): Identifier to include in result.
            - channel_id (Any, optional): Channel identifier to include in result.

    Returns:
        Dictionary containing:
            - change_point_indices (list[int]): Detected change point indices.
            - levels (list[dict]): List of level data dictionaries.
            - num_change_points (int): Number of detected change points.
            - confidence_regions (list[tuple[int, int]]): Confidence regions.
            - measurement_id: Original measurement_id if provided.
            - channel_id: Original channel_id if provided.
    """
    from full_sms.analysis import find_change_points

    # Extract parameters with defaults
    abstimes = np.asarray(params["abstimes"], dtype=np.float64)
    confidence = params.get("confidence", 0.95)
    min_photons = params.get("min_photons", 20)
    min_boundary_offset = params.get("min_boundary_offset", 7)
    end_time_ns = params.get("end_time_ns")

    # Run analysis
    result = find_change_points(
        abstimes=abstimes,
        confidence=confidence,
        min_photons=min_photons,
        min_boundary_offset=min_boundary_offset,
        end_time_ns=end_time_ns,
    )

    # Convert to serializable dict
    return {
        "change_point_indices": result.change_point_indices.tolist(),
        "levels": [_level_to_dict(level) for level in result.levels],
        "num_change_points": result.num_change_points,
        "confidence_regions": result.confidence_regions,
        "measurement_id": params.get("measurement_id"),
        "channel_id": params.get("channel_id"),
    }


def run_clustering_task(params: dict[str, Any]) -> dict[str, Any]:
    """Run AHCA clustering analysis in a worker process.

    Parameters:
        params: Dictionary containing:
            - levels (list[dict]): List of level data dictionaries from CPA.
                Each dict should have: start_index, end_index, start_time_ns,
                end_time_ns, num_photons, dwell_time_s, intensity_cps.
            - use_lifetime (bool, optional): Include lifetime in clustering (default False).
            - measurement_id (Any, optional): Identifier to include in result.
            - channel_id (Any, optional): Channel identifier to include in result.

    Returns:
        Dictionary containing:
            - steps (list[dict]): List of clustering step dictionaries.
            - optimal_step_index (int): Index of optimal BIC step.
            - selected_step_index (int): Currently selected step index.
            - num_original_levels (int): Number of input levels.
            - measurement_id: Original measurement_id if provided.
            - channel_id: Original channel_id if provided.
            Returns None if clustering not possible (e.g., < 1 level).
    """
    from full_sms.analysis.clustering import cluster_levels
    from full_sms.models.level import LevelData

    # Convert level dicts back to LevelData objects
    level_dicts = params["levels"]
    levels = [_dict_to_level(d) for d in level_dicts]

    use_lifetime = params.get("use_lifetime", False)

    # Run clustering
    result = cluster_levels(levels=levels, use_lifetime=use_lifetime)

    if result is None:
        return {
            "result": None,
            "measurement_id": params.get("measurement_id"),
            "channel_id": params.get("channel_id"),
        }

    # Convert to serializable dict
    return {
        "steps": [_clustering_step_to_dict(step) for step in result.steps],
        "optimal_step_index": result.optimal_step_index,
        "selected_step_index": result.selected_step_index,
        "num_original_levels": result.num_original_levels,
        "measurement_id": params.get("measurement_id"),
        "channel_id": params.get("channel_id"),
    }


def run_fit_task(params: dict[str, Any]) -> dict[str, Any]:
    """Run lifetime fitting in a worker process.

    Parameters:
        params: Dictionary containing:
            - t (NDArray[np.float64]): Time axis in nanoseconds.
            - counts (NDArray[np.int64]): Photon counts per channel.
            - channelwidth (float): TCSPC channel width in nanoseconds.
            - irf (NDArray[np.float64], optional): Instrument response function.
            - num_exponentials (int, optional): 1, 2, or 3 (default 1).
            - tau_init (float or list[float], optional): Initial tau guess(es).
            - tau_bounds (tuple[float, float], optional): Tau bounds.
            - amp_init (list[float], optional): Initial amplitude guess(es).
            - amp_bounds (tuple[float, float], optional): Amplitude bounds.
            - shift_init (float, optional): Initial IRF shift (default 0).
            - shift_bounds (tuple[float, float], optional): Shift bounds.
            - start (int, optional): Manual start point.
            - end (int, optional): Manual end point.
            - autostart (str, optional): Autostart mode name (e.g., "Manual").
            - autoend (bool, optional): Auto-detect endpoint (default False).
            - background (float, optional): Pre-calculated background.
            - irf_background (float, optional): Pre-calculated IRF background.
            - fit_irf_fwhm (bool, optional): Fit IRF FWHM (default False).
            - irf_fwhm_init (float, optional): Initial FWHM guess in nanoseconds.
            - irf_fwhm_bounds (tuple[float, float], optional): FWHM bounds.
            - measurement_id (Any, optional): Identifier to include in result.
            - channel_id (Any, optional): Channel identifier to include in result.
            - level_id (Any, optional): Level identifier to include in result.
            - group_id (Any, optional): Group identifier to include in result.

    Returns:
        Dictionary containing fit results:
            - tau (list[float]): Fitted lifetimes in nanoseconds.
            - tau_std (list[float]): Standard errors of tau.
            - amplitude (list[float]): Relative amplitudes.
            - amplitude_std (list[float]): Standard errors of amplitudes.
            - shift (float): IRF shift in nanoseconds.
            - shift_std (float): Standard error of shift.
            - chi_squared (float): Reduced chi-squared.
            - durbin_watson (float): Durbin-Watson statistic.
            - dw_bounds (tuple[float, float] or None): DW critical bounds.
            - residuals (list[float]): Weighted residuals.
            - fitted_curve (list[float]): Fitted decay curve.
            - fit_start_index (int): Start index of fit range.
            - fit_end_index (int): End index of fit range.
            - background (float): Background used.
            - num_exponentials (int): Number of exponential components.
            - average_lifetime (float): Amplitude-weighted average lifetime.
            - fitted_irf_fwhm (float or None): Fitted IRF FWHM if fit_irf_fwhm=True.
            - fitted_irf_fwhm_std (float or None): Standard error of fitted FWHM.
            - measurement_id, channel_id, level_id, group_id: Original identifiers.
            - error (str or None): Error message if fitting failed.
    """
    from full_sms.analysis import fit_decay
    from full_sms.analysis.lifetime import StartpointMode

    # Extract required parameters
    t = np.asarray(params["t"], dtype=np.float64)
    counts = np.asarray(params["counts"], dtype=np.int64)
    channelwidth = float(params["channelwidth"])

    # Extract optional parameters
    irf = params.get("irf")
    if irf is not None:
        irf = np.asarray(irf, dtype=np.float64)

    num_exponentials = params.get("num_exponentials", 1)
    tau_init = params.get("tau_init")
    tau_bounds = params.get("tau_bounds")
    amp_init = params.get("amp_init")
    amp_bounds = params.get("amp_bounds")
    shift_init = params.get("shift_init", 0.0)
    shift_bounds = params.get("shift_bounds")
    start = params.get("start")
    end = params.get("end")
    background = params.get("background")
    irf_background = params.get("irf_background")
    autoend = params.get("autoend", False)

    # Extract FWHM fitting parameters
    fit_irf_fwhm = params.get("fit_irf_fwhm", False)
    irf_fwhm_init = params.get("irf_fwhm_init")
    irf_fwhm_bounds = params.get("irf_fwhm_bounds")

    # Convert autostart string to enum
    autostart_str = params.get("autostart", "Manual")
    autostart = StartpointMode.MANUAL
    for mode in StartpointMode:
        if mode.value == autostart_str:
            autostart = mode
            break

    # Build result dict with identifiers
    result_dict: dict[str, Any] = {
        "measurement_id": params.get("measurement_id"),
        "channel_id": params.get("channel_id"),
        "level_id": params.get("level_id"),
        "group_id": params.get("group_id"),
        "error": None,
    }

    try:
        result = fit_decay(
            t=t,
            counts=counts,
            channelwidth=channelwidth,
            irf=irf,
            num_exponentials=num_exponentials,
            tau_init=tau_init,
            tau_bounds=tau_bounds,
            amp_init=amp_init,
            amp_bounds=amp_bounds,
            shift_init=shift_init,
            shift_bounds=shift_bounds,
            start=start,
            end=end,
            autostart=autostart,
            autoend=autoend,
            background=background,
            irf_background=irf_background,
            fit_irf_fwhm=fit_irf_fwhm,
            irf_fwhm_init=irf_fwhm_init,
            irf_fwhm_bounds=irf_fwhm_bounds,
        )

        # Convert FitResult to dict
        result_dict.update(
            {
                "tau": list(result.tau),
                "tau_std": list(result.tau_std),
                "amplitude": list(result.amplitude),
                "amplitude_std": list(result.amplitude_std),
                "shift": result.shift,
                "shift_std": result.shift_std,
                "chi_squared": result.chi_squared,
                "durbin_watson": result.durbin_watson,
                "dw_bounds": result.dw_bounds,
                "residuals": result.residuals.tolist(),
                "fitted_curve": result.fitted_curve.tolist(),
                "fit_start_index": result.fit_start_index,
                "fit_end_index": result.fit_end_index,
                "background": result.background,
                "num_exponentials": result.num_exponentials,
                "average_lifetime": result.average_lifetime,
                "fitted_irf_fwhm": result.fitted_irf_fwhm,
                "fitted_irf_fwhm_std": result.fitted_irf_fwhm_std,
            }
        )

    except (ValueError, RuntimeError) as e:
        result_dict["error"] = str(e)

    return result_dict


def run_correlation_task(params: dict[str, Any]) -> dict[str, Any]:
    """Run g2 correlation analysis in a worker process.

    Parameters:
        params: Dictionary containing:
            - abstimes1 (NDArray[np.float64]): Absolute times for channel 1 (ns).
            - abstimes2 (NDArray[np.float64]): Absolute times for channel 2 (ns).
            - microtimes1 (NDArray[np.float64]): Micro times for channel 1 (ns).
            - microtimes2 (NDArray[np.float64]): Micro times for channel 2 (ns).
            - window_ns (float, optional): Correlation window (default 500 ns).
            - binsize_ns (float, optional): Histogram bin size (default 0.5 ns).
            - difftime_ns (float, optional): Channel offset (default 0).
            - measurement_id (Any, optional): Identifier to include in result.

    Returns:
        Dictionary containing:
            - tau (list[float]): Delay time bin centers in nanoseconds.
            - g2 (list[int]): Correlation histogram values.
            - events (list[float]): Raw delay times for rebinning.
            - window_ns (float): Correlation window used.
            - binsize_ns (float): Bin size used.
            - num_photons_ch1 (int): Photons in channel 1.
            - num_photons_ch2 (int): Photons in channel 2.
            - num_events (int): Total correlation events.
            - measurement_id: Original measurement_id if provided.
    """
    from full_sms.analysis import calculate_g2

    # Extract parameters
    abstimes1 = np.asarray(params["abstimes1"], dtype=np.float64)
    abstimes2 = np.asarray(params["abstimes2"], dtype=np.float64)
    microtimes1 = np.asarray(params["microtimes1"], dtype=np.float64)
    microtimes2 = np.asarray(params["microtimes2"], dtype=np.float64)
    window_ns = params.get("window_ns", 500.0)
    binsize_ns = params.get("binsize_ns", 0.5)
    difftime_ns = params.get("difftime_ns", 0.0)

    # Run correlation
    result = calculate_g2(
        abstimes1=abstimes1,
        abstimes2=abstimes2,
        microtimes1=microtimes1,
        microtimes2=microtimes2,
        window_ns=window_ns,
        binsize_ns=binsize_ns,
        difftime_ns=difftime_ns,
    )

    # Convert to serializable dict
    return {
        "tau": result.tau.tolist(),
        "g2": result.g2.tolist(),
        "events": result.events.tolist(),
        "window_ns": result.window_ns,
        "binsize_ns": result.binsize_ns,
        "num_photons_ch1": result.num_photons_ch1,
        "num_photons_ch2": result.num_photons_ch2,
        "num_events": result.num_events,
        "measurement_id": params.get("measurement_id"),
    }


# Helper functions for serialization/deserialization


def _level_to_dict(level) -> dict[str, Any]:
    """Convert a LevelData object to a dictionary.

    Args:
        level: LevelData instance.

    Returns:
        Dictionary with level attributes.
    """
    return {
        "start_index": level.start_index,
        "end_index": level.end_index,
        "start_time_ns": level.start_time_ns,
        "end_time_ns": level.end_time_ns,
        "num_photons": level.num_photons,
        "dwell_time_s": level.dwell_time_s,
        "intensity_cps": level.intensity_cps,
        "group_id": level.group_id,
    }


def _dict_to_level(d: dict[str, Any]):
    """Convert a dictionary back to a LevelData object.

    Args:
        d: Dictionary with level attributes.

    Returns:
        LevelData instance.
    """
    from full_sms.models.level import LevelData

    return LevelData(
        start_index=d["start_index"],
        end_index=d["end_index"],
        start_time_ns=int(d["start_time_ns"]),
        end_time_ns=int(d["end_time_ns"]),
        num_photons=d["num_photons"],
        intensity_cps=d["intensity_cps"],
        group_id=d.get("group_id"),
    )


def _group_to_dict(group) -> dict[str, Any]:
    """Convert a GroupData object to a dictionary.

    Args:
        group: GroupData instance.

    Returns:
        Dictionary with group attributes.
    """
    return {
        "group_id": group.group_id,
        "level_indices": list(group.level_indices),
        "total_photons": group.total_photons,
        "total_dwell_time_s": group.total_dwell_time_s,
        "intensity_cps": group.intensity_cps,
    }


def _clustering_step_to_dict(step) -> dict[str, Any]:
    """Convert a ClusteringStep to a dictionary.

    Args:
        step: ClusteringStep instance.

    Returns:
        Dictionary with step attributes.
    """
    return {
        "groups": [_group_to_dict(g) for g in step.groups],
        "level_group_assignments": list(step.level_group_assignments),
        "bic": step.bic,
        "num_groups": step.num_groups,
    }
