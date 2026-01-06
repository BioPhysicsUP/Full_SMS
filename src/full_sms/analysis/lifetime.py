"""Fluorescence lifetime fitting module.

This module implements fluorescence decay fitting using IRF convolution
and least-squares optimization. It supports single-exponential models
(with multi-exponential support planned for future tasks).

Based on the FluoFit implementation from the original Full SMS codebase,
with reference to MATLAB code by Jörg Enderlein:
https://www.uni-goettingen.de/en/513325.html
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import convolve

from full_sms.models.fit import FitResult


class FitMethod(Enum):
    """Fitting method for lifetime analysis."""

    LEAST_SQUARES = "ls"
    MAXIMUM_LIKELIHOOD = "ml"


class StartpointMode(Enum):
    """Mode for automatic startpoint determination."""

    MANUAL = "Manual"
    CLOSE_TO_MAX = "(Close to) max"
    RISE_MIDDLE = "Rise middle"
    RISE_START = "Rise start"
    SAFE_RISE_START = "Safe rise start"


@dataclass
class FitSettings:
    """Configuration settings for lifetime fitting.

    Attributes:
        use_moving_avg: Whether to use moving average smoothing for boundary detection.
        moving_avg_window: Window size for moving average in channels.
        start_percent: Percentage of max for automatic start detection.
        end_multiple: Multiple of background for automatic end detection.
        end_percent: Percentage of max for automatic end detection.
        minimum_decay_window_ns: Minimum decay window in nanoseconds.
        bg_percent: Maximum background as percentage of max.
    """

    use_moving_avg: bool = True
    moving_avg_window: int = 10
    start_percent: float = 80.0
    end_multiple: float = 20.0
    end_percent: float = 1.0
    minimum_decay_window_ns: float = 2.0
    bg_percent: float = 5.0


BACKGROUND_SECTION_LENGTH = 50
DEFAULT_SETTINGS = FitSettings()


def _moving_avg(
    vector: NDArray[np.float64], window_length: int, pad_same_size: bool = True
) -> NDArray[np.float64]:
    """Apply moving average filter.

    Used to smooth decay histogram before determining fitting endpoints.

    Args:
        vector: Data to smooth.
        window_length: Moving average window size in data points.
        pad_same_size: Whether to pad output to same size as input.

    Returns:
        Smoothed data.
    """
    if window_length <= 1:
        return vector.copy()

    vector_size = len(vector)
    left_window = int(np.floor(window_length / 2))
    right_window = int(np.ceil(window_length / 2))

    if pad_same_size:
        start_pad = np.zeros(left_window)
        end_pad = np.zeros(right_window)
        padded = np.concatenate([start_pad, vector, end_pad])
    else:
        padded = vector

    new_vector = np.array(
        [
            np.mean(padded[i : i + window_length])
            for i in range(vector_size if pad_same_size else vector_size - window_length + 1)
        ]
    )
    return new_vector


def _max_continuous_zeros(vector: NDArray[np.float64]) -> int:
    """Find maximum number of consecutive zeros in data.

    Used for background estimation to detect gaps before the decay rise.

    Args:
        vector: Input data.

    Returns:
        Maximum number of consecutive zeros.
    """
    if len(vector) == 0:
        return 0

    is_zero = vector == 0
    changes = np.concatenate(([is_zero[0]], is_zero[:-1] != is_zero[1:], [True]))
    run_lengths = np.diff(np.where(changes)[0])[::2]

    return int(np.max(run_lengths)) if len(run_lengths) > 0 else 0


def colorshift(irf: NDArray[np.float64], shift: float) -> NDArray[np.float64]:
    """Shift IRF left or right with periodic wrapping.

    A shift past the start or end results in values wrapping around.
    This implements sub-channel interpolation for non-integer shifts.

    Args:
        irf: Instrument response function.
        shift: Amount to shift in channels (can be non-integer).

    Returns:
        Shifted IRF.
    """
    irf = irf.flatten()
    irf_length = len(irf)
    t = np.arange(irf_length)

    # Calculate indices for interpolation
    new_index_left = np.fmod(
        np.fmod(t - np.floor(shift), irf_length) + irf_length, irf_length
    ).astype(int)
    new_index_right = np.fmod(
        np.fmod(t - np.ceil(shift), irf_length) + irf_length, irf_length
    ).astype(int)

    # Interpolate between integer shifts
    integer_left_shift = irf[new_index_left]
    integer_right_shift = irf[new_index_right]

    irs = (1 - shift + np.floor(shift)) * integer_left_shift + (
        shift - np.floor(shift)
    ) * integer_right_shift

    return irs


def estimate_background(
    measured: NDArray[np.float64],
    settings: Optional[FitSettings] = None,
    return_rise_index: bool = False,
) -> float | int:
    """Estimate decay background from pre-rise region.

    Finds the flat region before the decay rise and estimates background
    as the mean of that region.

    Args:
        measured: Measured decay data.
        settings: Fit settings (uses defaults if None).
        return_rise_index: If True, return index of rise start instead of background.

    Returns:
        Background estimate (float) or rise index (int) if return_rise_index is True.
    """
    if settings is None:
        settings = DEFAULT_SETTINGS

    if len(measured) == 0:
        return 0 if return_rise_index else 0.0

    maxind = np.argmax(measured)
    nonzero_indices = np.nonzero(measured)[0]

    if len(nonzero_indices) == 0:
        return 0 if return_rise_index else 0.0

    meas_real_start = nonzero_indices[0]
    bg_section = measured[meas_real_start : meas_real_start + BACKGROUND_SECTION_LENGTH]

    # Attempt to remove low "island" of counts before real start
    if (
        len(bg_section) > 0
        and _max_continuous_zeros(bg_section) / BACKGROUND_SECTION_LENGTH >= 0.5
        and len(measured[meas_real_start:]) > 2 * BACKGROUND_SECTION_LENGTH
    ):
        original_start = meas_real_start
        while _max_continuous_zeros(bg_section) / BACKGROUND_SECTION_LENGTH >= 0.5:
            zero_indices = np.where(bg_section == 0)[0]
            if len(zero_indices) == 0:
                break
            next_seg_start = meas_real_start + zero_indices[0] + 1

            if len(measured[next_seg_start:]) < 2 * BACKGROUND_SECTION_LENGTH:
                meas_real_start = original_start
                bg_section = measured[meas_real_start : meas_real_start + BACKGROUND_SECTION_LENGTH]
                break

            nonzero_after = np.nonzero(measured[next_seg_start:])[0]
            if len(nonzero_after) == 0:
                break
            meas_real_start = nonzero_after[0] + next_seg_start
            bg_section = measured[meas_real_start : meas_real_start + BACKGROUND_SECTION_LENGTH]

    bg_section_mean = np.mean(bg_section) if len(bg_section) > 0 else 0

    # Find where signal drops to background level before the peak
    bglim = meas_real_start + 50
    for i in reversed(range(meas_real_start, maxind)):
        if measured[i] <= bg_section_mean:
            bglim = i
            break

    if return_rise_index:
        return bglim

    bg_est = np.mean(measured[meas_real_start:bglim]) if bglim > meas_real_start else 0
    bg_percent = settings.bg_percent / 100
    bg_est = min(bg_est, bg_percent * measured.max())

    return 0.0 if np.isnan(bg_est) else float(bg_est)


def estimate_irf_background(irf: NDArray[np.float64]) -> float:
    """Estimate IRF background.

    Args:
        irf: Instrument response function.

    Returns:
        Background estimate.
    """
    if len(irf) == 0:
        return 0.0

    maxind = np.argmax(irf)
    bglim = None

    first_20_mean = np.mean(irf[:20]) if len(irf) >= 20 else np.mean(irf)

    for i in range(maxind):
        reverse = maxind - i
        if int(irf[reverse]) == int(first_20_mean):
            bglim = reverse
            break

    if bglim is not None and bglim > 0:
        return float(np.mean(irf[:bglim]))

    return 0.0


def calculate_boundaries(
    measured: NDArray[np.float64],
    channelwidth: float,
    start: Optional[int] = None,
    end: Optional[int] = None,
    autostart: StartpointMode = StartpointMode.MANUAL,
    autoend: bool = False,
    background: Optional[float] = None,
    settings: Optional[FitSettings] = None,
) -> Tuple[int, int]:
    """Calculate fitting boundaries.

    Determines the start and end points for decay fitting, either using
    provided values or automatic detection based on decay characteristics.

    Args:
        measured: Measured decay data.
        channelwidth: TCSPC channel width in nanoseconds.
        start: Manual start point (channel index).
        end: Manual end point (channel index).
        autostart: Mode for automatic start detection.
        autoend: Whether to automatically determine end point.
        background: Pre-calculated background (will be estimated if None).
        settings: Fit settings (uses defaults if None).

    Returns:
        Tuple of (startpoint, endpoint) as channel indices.
    """
    if settings is None:
        settings = DEFAULT_SETTINGS

    if len(measured) == 0:
        return 0, 0

    if background is None:
        background = estimate_background(measured, settings)

    # Apply moving average for boundary detection if enabled
    if settings.use_moving_avg:
        smoothed = _moving_avg(
            measured.astype(np.float64),
            window_length=min(settings.moving_avg_window, int(0.1 * len(measured))),
            pad_same_size=True,
        )
    else:
        smoothed = measured.astype(np.float64)

    # Determine startpoint
    startmax = None
    if autostart == StartpointMode.MANUAL:
        startpoint = start if start is not None else 0
    else:
        maxpoint = smoothed.max()
        close_percentage = settings.start_percent / 100
        close_to_max = np.where(smoothed > close_percentage * maxpoint)[0]
        startmax = close_to_max[0] if len(close_to_max) > 0 else 0
        startmin = estimate_background(measured, settings, return_rise_index=True)

        if autostart == StartpointMode.CLOSE_TO_MAX:
            startpoint = startmax
        elif autostart == StartpointMode.RISE_MIDDLE:
            startpoint = int(0.5 * (startmin + startmax))
        elif autostart == StartpointMode.RISE_START:
            startpoint = startmin
        elif autostart == StartpointMode.SAFE_RISE_START:
            startpoint = startmin - min((startmax - startmin), 10)
        else:
            startpoint = start if start is not None else 0

    # Ensure startpoint is non-negative
    startpoint = max(0, startpoint)

    # Determine endpoint
    if autoend:
        end_multiple = settings.end_multiple
        end_percent = settings.end_percent / 100
        min_val = max(end_multiple * background, end_percent * measured.max())

        if not np.isnan(min_val):
            greater_than_bg = np.where(smoothed[startpoint:] > min_val)[0]
            greater_than_bg = greater_than_bg + startpoint

            if len(greater_than_bg) > 0 and greater_than_bg[-1] == len(measured) - 1:
                greater_than_bg = greater_than_bg[:-1]

            if len(greater_than_bg) > 0:
                endpoint = int(greater_than_bg[-1])
            else:
                endpoint = startpoint + int(np.round(settings.minimum_decay_window_ns / channelwidth))
        else:
            endpoint = len(measured) - 1

        if settings.use_moving_avg:
            endpoint += int(np.round(settings.moving_avg_window / 2))
    elif end is not None:
        endpoint = min(end, len(measured) - 1)
    else:
        endpoint = len(measured) - 1

    # Ensure minimum decay window
    if channelwidth * (endpoint - startpoint) < settings.minimum_decay_window_ns:
        start_ref = startmax if startmax is not None else startpoint
        endpoint = start_ref + int(np.round(settings.minimum_decay_window_ns / channelwidth))
        if autostart == StartpointMode.SAFE_RISE_START:
            startpoint -= int(np.round(0.3 * settings.minimum_decay_window_ns / channelwidth))

    # Final bounds checking
    startpoint = max(0, min(startpoint, len(measured) - 1))
    endpoint = max(startpoint + 1, min(endpoint, len(measured)))

    return startpoint, endpoint


def durbin_watson_bounds(num_points: int, num_params: int) -> Tuple[float, float, float, float]:
    """Calculate Durbin-Watson lower bounds for different significance levels.

    Based on Turner 2020 https://doi.org/10.1080/13504851.2019.1706908

    Args:
        num_points: Number of data points in the fit.
        num_params: Number of fit parameters.

    Returns:
        Tuple of (dw_5pct, dw_1pct, dw_0.3pct, dw_0.1pct) critical values.
    """
    # Coefficients for different number of parameters (k)
    # Based on number of exponentials: 1 exp = 2 params, 2 exp = 4 params, 3 exp = 6 params
    if num_params <= 2:  # Single exponential
        beta1_5, beta2_5, beta3_5, beta4_5 = -3.312097, -3.332536, -3.632166, 19.31135
        beta1_1, beta2_1, beta3_1, beta4_1 = -4.642915, -4.052984, 5.966592, 14.91894
    elif num_params <= 4:  # Double exponential
        beta1_5, beta2_5, beta3_5, beta4_5 = -3.447993, -4.229294, -28.91627, 80.00972
        beta1_1, beta2_1, beta3_1, beta4_1 = -4.655069, -7.296073, -5.300441, 60.11130
    else:  # Triple exponential (using 5-param values as approximation)
        beta1_5, beta2_5, beta3_5, beta4_5 = -3.535331, -4.085190, -47.63654, 127.7127
        beta1_1, beta2_1, beta3_1, beta4_1 = -4.675041, -8.518908, -15.25711, 96.32291

    n = num_points
    sqrt_n = np.sqrt(n)

    # 5% critical bound
    dw_5 = 2 + beta1_5 / sqrt_n + beta2_5 / n + beta3_5 / (sqrt_n**3) + beta4_5 / (n**2)

    # 1% critical bound
    dw_1 = 2 + beta1_1 / sqrt_n + beta2_1 / n + beta3_1 / (sqrt_n**3) + beta4_1 / (n**2)

    # For < 1% use normal distribution approximation
    var = (4 * n**2 * (n - 2)) / ((n + 1) * (n - 1) ** 3)
    std = np.sqrt(var)
    dw_03 = 2 - 3 * std  # 0.3%
    dw_01 = 2 - 3.28 * std  # 0.1%

    return (round(dw_5, 3), round(dw_1, 3), round(dw_03, 3), round(dw_01, 3))


def fit_decay(
    t: NDArray[np.float64],
    counts: NDArray[np.int64],
    channelwidth: float,
    irf: Optional[NDArray[np.float64]] = None,
    num_exponentials: int = 1,
    tau_init: Optional[float | list[float]] = None,
    tau_bounds: Optional[Tuple[float, float]] = None,
    shift_init: float = 0.0,
    shift_bounds: Optional[Tuple[float, float]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    autostart: StartpointMode = StartpointMode.MANUAL,
    autoend: bool = False,
    background: Optional[float] = None,
    irf_background: Optional[float] = None,
    settings: Optional[FitSettings] = None,
) -> FitResult:
    """Fit fluorescence decay data with exponential models.

    Performs least-squares fitting of decay data using IRF convolution.
    Currently supports single-exponential models.

    Args:
        t: Time axis in nanoseconds.
        counts: Photon counts per channel.
        channelwidth: TCSPC channel width in nanoseconds.
        irf: Instrument response function. If None, a delta function is used.
        num_exponentials: Number of exponential components (only 1 supported currently).
        tau_init: Initial guess for lifetime(s) in nanoseconds.
        tau_bounds: Bounds for tau as (min, max) in nanoseconds.
        shift_init: Initial guess for IRF shift in channels.
        shift_bounds: Bounds for shift as (min, max) in channels.
        start: Manual start point (channel index).
        end: Manual end point (channel index).
        autostart: Mode for automatic start detection.
        autoend: Whether to automatically determine end point.
        background: Pre-calculated decay background.
        irf_background: Pre-calculated IRF background.
        settings: Fit settings.

    Returns:
        FitResult containing fitted parameters and statistics.

    Raises:
        ValueError: If fitting fails or invalid parameters provided.
    """
    if num_exponentials != 1:
        raise ValueError(f"Only single exponential fitting is currently supported, got {num_exponentials}")

    if settings is None:
        settings = DEFAULT_SETTINGS

    if len(t) == 0 or len(counts) == 0:
        raise ValueError("Empty data provided for fitting")

    if len(t) != len(counts):
        raise ValueError(f"Length mismatch: t ({len(t)}) != counts ({len(counts)})")

    measured = counts.astype(np.float64)

    # Estimate backgrounds if not provided
    if background is None:
        background = estimate_background(measured, settings)

    # Setup IRF
    if irf is not None:
        if irf_background is None:
            irf_background = estimate_irf_background(irf)
        irf_processed = irf - irf_background
        irf_processed = irf_processed / irf_processed.max()  # Normalize
    else:
        # Use delta function if no IRF provided
        irf_processed = np.zeros(len(measured))
        irf_processed[0] = 1.0

    # Calculate fitting boundaries
    startpoint, endpoint = calculate_boundaries(
        measured, channelwidth, start, end, autostart, autoend, background, settings
    )

    # Subtract background and normalize
    measured_bg_sub = measured - background
    measured_bg_sub[measured_bg_sub <= 0] = 0

    measured_bounded = measured_bg_sub[startpoint:endpoint]
    meas_sum = np.sum(measured_bounded)

    if meas_sum == 0:
        raise ValueError("No photons in fitting range after background subtraction")

    measured_norm = measured_bounded / meas_sum
    bg_norm = background / meas_sum

    # Set default parameter values
    if tau_init is None:
        tau_init = 5.0

    if tau_bounds is None:
        tau_bounds = (0.01, 100.0)

    if shift_bounds is None:
        shift_bounds = (-2000.0, 2000.0)

    # Build fit function using closure
    n_channels = len(t)

    def make_convd(shift: float, tau: float, amplitude: float = 1.0) -> NDArray[np.float64]:
        """Convolve exponential model with shifted IRF."""
        model = amplitude * np.exp(-t / tau)
        irf_shifted = colorshift(irf_processed, shift)
        # Use full convolution and take first n_channels elements
        convd = convolve(irf_shifted, model, mode="full")[:n_channels]
        # Slice to fit range and normalize
        convd = convd[startpoint:endpoint]
        convd_sum = convd.sum()
        if convd_sum > 0:
            convd = convd / convd_sum
        return convd

    def fitfunc(t_fit: NDArray, tau1: float, amp: float, shift: float) -> NDArray[np.float64]:
        """Single exponential fit function."""
        return make_convd(shift, tau1, amp)

    # Setup parameter bounds
    paramin = [tau_bounds[0], 0.0, shift_bounds[0]]
    paramax = [tau_bounds[1], 100.0, shift_bounds[1]]
    paraminit = [tau_init, 1.0, shift_init]

    # Perform fit
    try:
        param, pcov = curve_fit(
            fitfunc,
            t[startpoint:endpoint],
            measured_norm,
            bounds=(paramin, paramax),
            p0=paraminit,
            maxfev=10000,
        )
    except RuntimeError as e:
        raise ValueError(f"Fitting failed to converge: {e}")

    # Extract results
    tau = param[0]
    amplitude = param[1]
    shift = param[2]

    # Standard errors from covariance
    stds = np.sqrt(np.diag(pcov))
    tau_std = stds[0]
    amp_std = stds[1]
    shift_std = stds[2]

    # Generate fitted curve
    convd = fitfunc(t[startpoint:endpoint], tau, amplitude, shift)

    # Calculate residuals
    measured_for_resid = measured[startpoint:endpoint]
    fitted_denorm = (convd + bg_norm) * meas_sum

    residuals_raw = fitted_denorm - measured_for_resid
    weights = np.sqrt(np.abs(fitted_denorm))
    weights[weights == 0] = 1.0
    residuals = residuals_raw / weights

    # Remove infinite residuals
    valid_residuals = residuals[np.isfinite(residuals)]

    # Chi-squared (reduced)
    num_params = 3  # tau, amplitude, shift
    dof = len(valid_residuals) - num_params - 1
    if dof <= 0:
        dof = 1
    chi_squared = float(np.sum(valid_residuals**2) / dof)

    # Durbin-Watson statistic
    if len(valid_residuals) > 1:
        dw = float(np.sum(np.diff(valid_residuals) ** 2) / np.sum(valid_residuals**2))
    else:
        dw = 2.0  # Ideal value if not enough data

    # Durbin-Watson bounds
    dw_bounds = durbin_watson_bounds(len(valid_residuals), num_params)

    # Convert shift to nanoseconds
    shift_ns = shift * channelwidth

    # Create FitResult
    return FitResult.from_fit_parameters(
        tau=[tau],
        tau_std=[tau_std],
        amplitude=[1.0],  # Normalized amplitude for single exp
        amplitude_std=[amp_std],
        shift=shift_ns,
        shift_std=shift_std * channelwidth,
        chi_squared=chi_squared,
        durbin_watson=dw,
        residuals=valid_residuals,
        fitted_curve=fitted_denorm,
        fit_start_index=startpoint,
        fit_end_index=endpoint,
        background=background,
        dw_bounds=(dw_bounds[0], dw_bounds[1]),  # Use 5% and 1% bounds
    )


def simulate_irf(
    channelwidth: float,
    fwhm: float,
    measured: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a Gaussian IRF.

    Creates a Gaussian instrument response function with peak aligned
    to the peak of measured data.

    Args:
        channelwidth: TCSPC channel width in nanoseconds.
        fwhm: Full width at half maximum of Gaussian IRF in nanoseconds.
        measured: Measured decay data (for peak alignment).

    Returns:
        Tuple of (irf, t) where irf is the simulated IRF and t is the time axis.
    """
    fwhm_channels = fwhm / channelwidth
    sigma = fwhm_channels / 2.35482  # FWHM to sigma conversion

    t_channels = np.arange(len(measured))
    maxind = np.argmax(measured)

    gauss = np.exp(-((t_channels - maxind) ** 2) / (2 * sigma**2))
    irf = gauss * measured.max() / gauss.max()

    t_ns = t_channels * channelwidth

    return irf, t_ns
