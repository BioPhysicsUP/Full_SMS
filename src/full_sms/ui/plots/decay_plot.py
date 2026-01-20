"""Fluorescence decay histogram plot widget.

Renders TCSPC decay data using DearPyGui's ImPlot with optional log-scale Y axis.
Supports fit curve overlay, IRF display, and legend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve

from full_sms.analysis.histograms import build_decay_histogram
from full_sms.models.fit import FitResult
from full_sms.ui.theme import COLORS

logger = logging.getLogger(__name__)


# Colors for decay plot overlays
DECAY_COLORS = {
    "data": COLORS["series_1"],  # Blue for data
    "fit": COLORS["series_2"],  # Orange for fit curve
    "irf": COLORS["series_3"],  # Green for IRF
}


@dataclass
class DecayPlotTags:
    """Tags for decay plot elements."""

    container: str = "decay_plot_container"
    plot: str = "decay_plot"
    x_axis: str = "decay_plot_x_axis"
    y_axis: str = "decay_plot_y_axis"
    series: str = "decay_plot_series"
    fit_series: str = "decay_plot_fit_series"
    irf_series: str = "decay_plot_irf_series"
    legend: str = "decay_plot_legend"


DECAY_PLOT_TAGS = DecayPlotTags()


class DecayPlot:
    """Fluorescence decay histogram plot widget.

    Displays TCSPC decay data as a line plot with optional logarithmic Y axis.
    Supports configurable time range with zoom and pan enabled.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the decay plot.

        Args:
            parent: The parent container to build the plot in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._t: NDArray[np.float64] | None = None
        self._counts: NDArray[np.int64] | None = None
        self._channelwidth: float = 0.1  # Default 100ps channel width
        self._log_scale: bool = True  # Default to log scale for decay data

        # Fit overlay state
        self._fit_result: Optional[FitResult] = None
        self._show_fit: bool = True

        # IRF overlay state
        self._irf_t: NDArray[np.float64] | None = None
        self._irf_counts: NDArray[np.float64] | None = None
        self._show_irf: bool = False

        # Raw IRF data for convolution computation (stored separately from display IRF)
        self._raw_irf: NDArray[np.float64] | None = None

        # Generate unique tags
        self._tags = DecayPlotTags(
            container=f"{tag_prefix}decay_plot_container",
            plot=f"{tag_prefix}decay_plot",
            x_axis=f"{tag_prefix}decay_plot_x_axis",
            y_axis=f"{tag_prefix}decay_plot_y_axis",
            series=f"{tag_prefix}decay_plot_series",
            fit_series=f"{tag_prefix}decay_plot_fit_series",
            irf_series=f"{tag_prefix}decay_plot_irf_series",
            legend=f"{tag_prefix}decay_plot_legend",
        )

    @property
    def tags(self) -> DecayPlotTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def log_scale(self) -> bool:
        """Whether the Y axis is in log scale."""
        return self._log_scale

    @property
    def channelwidth(self) -> float:
        """Get the current TCSPC channel width in nanoseconds."""
        return self._channelwidth

    def build(self) -> None:
        """Build the plot UI structure."""
        if self._is_built:
            return

        # Plot container
        with dpg.group(parent=self._parent, tag=self._tags.container):
            # Create the plot
            with dpg.plot(
                tag=self._tags.plot,
                label="Fluorescence Decay",
                width=-1,
                height=-1,
                anti_aliased=True,
            ):
                # Add legend
                dpg.add_plot_legend(tag=self._tags.legend)

                # X axis (time in nanoseconds)
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="Time (ns)",
                    tag=self._tags.x_axis,
                )

                # Y axis (photon counts) - default to log scale
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label="Counts",
                    tag=self._tags.y_axis,
                    log_scale=self._log_scale,
                )

                # Add empty line series for data (will be populated when data is set)
                dpg.add_line_series(
                    [],
                    [],
                    label="Data",
                    parent=self._tags.y_axis,
                    tag=self._tags.series,
                )

                # Add empty line series for fit curve (hidden until fit is set)
                dpg.add_line_series(
                    [],
                    [],
                    label="Fit",
                    parent=self._tags.y_axis,
                    tag=self._tags.fit_series,
                    show=False,
                )

                # Add empty line series for IRF (hidden until IRF is set)
                dpg.add_line_series(
                    [],
                    [],
                    label="IRF",
                    parent=self._tags.y_axis,
                    tag=self._tags.irf_series,
                    show=False,
                )

        # Apply colors to series
        self._apply_series_colors()

        self._is_built = True
        logger.debug("Decay plot built")

    def set_data(
        self,
        microtimes: NDArray[np.float64],
        channelwidth: float,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> None:
        """Set the microtime data and update the plot.

        Args:
            microtimes: TCSPC microtime values in nanoseconds.
            channelwidth: TCSPC channel width in nanoseconds.
            tmin: Optional minimum time for histogram range.
            tmax: Optional maximum time for histogram range.
        """
        self._channelwidth = channelwidth

        if len(microtimes) == 0:
            self.clear()
            return

        # Build the decay histogram
        self._t, self._counts = build_decay_histogram(
            microtimes, channelwidth, tmin=tmin, tmax=tmax
        )

        # Update the plot
        self._update_series()
        self._fit_axes()

        logger.debug(
            f"Decay plot updated: {len(self._t)} bins at {channelwidth} ns resolution"
        )

    def set_histogram_data(
        self,
        t: NDArray[np.float64],
        counts: NDArray[np.int64],
    ) -> None:
        """Set pre-computed histogram data directly.

        Use this when the histogram has already been built (e.g., for level data).

        Args:
            t: Array of time values in nanoseconds.
            counts: Array of photon counts per bin.
        """
        if len(t) == 0 or len(counts) == 0:
            self.clear()
            return

        self._t = t
        self._counts = counts

        # Update the plot
        self._update_series()
        self._fit_axes()

        logger.debug(f"Decay plot updated with pre-computed data: {len(t)} bins")

    def _update_series(self) -> None:
        """Update the plot series with current data."""
        if not dpg.does_item_exist(self._tags.series):
            return

        if self._t is None or self._counts is None:
            dpg.configure_item(self._tags.series, x=[], y=[])
            return

        # For log scale, filter out zero counts (can't take log of zero)
        if self._log_scale:
            # Replace zeros with a small value (0.5) for visualization
            # This is a common practice in TCSPC data display
            display_counts = np.where(self._counts > 0, self._counts, 0.5)
        else:
            display_counts = self._counts.astype(np.float64)

        # Convert to lists for DearPyGui
        dpg.configure_item(
            self._tags.series,
            x=self._t.tolist(),
            y=display_counts.tolist(),
        )

    def _fit_axes(self) -> None:
        """Auto-fit the axes to show all data."""
        if not dpg.does_item_exist(self._tags.x_axis):
            return

        dpg.fit_axis_data(self._tags.x_axis)
        dpg.fit_axis_data(self._tags.y_axis)

    def clear(self) -> None:
        """Clear the plot data."""
        self._t = None
        self._counts = None
        self._raw_irf = None
        self._update_series()
        self.clear_fit()
        self.clear_irf()
        logger.debug("Decay plot cleared")

    def set_log_scale(self, log_scale: bool) -> None:
        """Set whether the Y axis uses logarithmic scale.

        Args:
            log_scale: True for log scale, False for linear.
        """
        if self._log_scale == log_scale:
            return

        self._log_scale = log_scale

        # Update the Y axis
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.configure_item(self._tags.y_axis, log_scale=log_scale)

        # Re-update series to handle zero values appropriately
        self._update_series()

        logger.debug(f"Decay plot log scale set to {log_scale}")

    def toggle_log_scale(self) -> bool:
        """Toggle the Y axis between log and linear scale.

        Returns:
            The new log scale state.
        """
        self.set_log_scale(not self._log_scale)
        return self._log_scale

    def set_axis_label_y(self, label: str) -> None:
        """Set the Y axis label.

        Args:
            label: The new axis label.
        """
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.configure_item(self._tags.y_axis, label=label)

    def set_axis_label_x(self, label: str) -> None:
        """Set the X axis label.

        Args:
            label: The new axis label.
        """
        if dpg.does_item_exist(self._tags.x_axis):
            dpg.configure_item(self._tags.x_axis, label=label)

    def get_time_range(self) -> tuple[float, float] | None:
        """Get the current time range of the data.

        Returns:
            Tuple of (min_time, max_time) in nanoseconds, or None if no data.
        """
        if self._t is None or len(self._t) == 0:
            return None
        return (float(self._t[0]), float(self._t[-1]))

    def get_count_range(self) -> tuple[int, int] | None:
        """Get the current count range of the data.

        Returns:
            Tuple of (min_counts, max_counts), or None if no data.
        """
        if self._counts is None or len(self._counts) == 0:
            return None
        return (int(np.min(self._counts)), int(np.max(self._counts)))

    def get_max_counts(self) -> int | None:
        """Get the maximum count value.

        Returns:
            Maximum count value, or None if no data.
        """
        if self._counts is None or len(self._counts) == 0:
            return None
        return int(np.max(self._counts))

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._t is not None and len(self._t) > 0

    def fit_view(self) -> None:
        """Fit the view to show all data (reset zoom/pan)."""
        self._fit_axes()

    def set_x_limits(self, xmin: float, xmax: float) -> None:
        """Set the X axis limits.

        Args:
            xmin: Minimum X value.
            xmax: Maximum X value.
        """
        if dpg.does_item_exist(self._tags.x_axis):
            dpg.set_axis_limits(self._tags.x_axis, xmin, xmax)

    def set_y_limits(self, ymin: float, ymax: float) -> None:
        """Set the Y axis limits.

        Args:
            ymin: Minimum Y value.
            ymax: Maximum Y value.
        """
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.set_axis_limits(self._tags.y_axis, ymin, ymax)

    def reset_x_limits(self) -> None:
        """Reset X axis limits to auto-fit."""
        if dpg.does_item_exist(self._tags.x_axis):
            dpg.set_axis_limits_auto(self._tags.x_axis)
            dpg.fit_axis_data(self._tags.x_axis)

    def reset_y_limits(self) -> None:
        """Reset Y axis limits to auto-fit."""
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.set_axis_limits_auto(self._tags.y_axis)
            dpg.fit_axis_data(self._tags.y_axis)

    # -------------------------------------------------------------------------
    # Series Color Methods
    # -------------------------------------------------------------------------

    def _apply_series_colors(self) -> None:
        """Apply themed colors to all series."""
        # Data series - blue
        self._apply_line_color(self._tags.series, DECAY_COLORS["data"])
        # Fit series - orange
        self._apply_line_color(self._tags.fit_series, DECAY_COLORS["fit"])
        # IRF series - green, with dashed style
        self._apply_line_color(self._tags.irf_series, DECAY_COLORS["irf"])

    def _apply_line_color(
        self, tag: str, color: tuple[int, int, int, int]
    ) -> None:
        """Apply a color to a line series via theme.

        Args:
            tag: The tag of the line series.
            color: RGBA color tuple.
        """
        if not dpg.does_item_exist(tag):
            return

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots
                )

        dpg.bind_item_theme(tag, theme)

    # -------------------------------------------------------------------------
    # Fit Curve Methods
    # -------------------------------------------------------------------------

    @property
    def has_fit(self) -> bool:
        """Whether a fit result is currently set."""
        return self._fit_result is not None

    @property
    def fit_result(self) -> Optional[FitResult]:
        """Get the current fit result."""
        return self._fit_result

    @property
    def show_fit(self) -> bool:
        """Whether the fit curve is currently visible."""
        return self._show_fit

    def set_raw_irf(self, irf: NDArray[np.float64]) -> None:
        """Store raw IRF data for computing full convolved fit curve.

        This IRF is used internally for convolution computation when set_fit
        is called. It should be the normalized IRF as used during fitting.

        Args:
            irf: Raw IRF array (same length as decay histogram).
        """
        self._raw_irf = irf

    def clear_raw_irf(self) -> None:
        """Clear the stored raw IRF data."""
        self._raw_irf = None

    def _colorshift(
        self, irf: NDArray[np.float64], shift: float
    ) -> NDArray[np.float64]:
        """Shift IRF with periodic wrapping (same as lifetime.colorshift).

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

    def set_fit(self, fit_result: FitResult) -> None:
        """Set the fit result and display the fit curve.

        If raw IRF data is available (via set_raw_irf), computes the full
        convolved fit curve including the rising edge. Otherwise, uses the
        pre-computed fitted curve and extrapolates to the right only.

        Args:
            fit_result: The FitResult containing the fitted curve data.
        """
        self._fit_result = fit_result

        if not dpg.does_item_exist(self._tags.fit_series):
            return

        if self._t is None or len(self._t) == 0:
            logger.warning("Cannot display fit without data")
            return

        # Get fit parameters
        fit_start = fit_result.fit_start_index
        fit_end = fit_result.fit_end_index
        background = fit_result.background
        avg_tau = fit_result.average_lifetime

        # Check if fit range is valid
        if fit_start >= len(self._t) or fit_end > len(self._t):
            logger.warning(
                f"Fit range [{fit_start}:{fit_end}] exceeds data length {len(self._t)}"
            )
            return

        # Try to compute full convolved curve if we have raw IRF data
        if (
            self._raw_irf is not None
            and len(self._raw_irf) == len(self._t)
            and self._counts is not None
        ):
            try:
                full_curve = self._compute_full_convolved_curve(fit_result)
                if full_curve is not None:
                    # Use full curve - it covers the entire data range
                    fit_t = self._t.copy()
                    fitted_curve = full_curve

                    # Extend to the right using exponential decay
                    # (full_curve already covers the data range, but we extend further)
                    # Actually the full_curve covers the same range as data,
                    # so no further extension is needed here

                    # For log scale, clip to minimum 0.5
                    if self._log_scale:
                        display_fit = np.maximum(fitted_curve, 0.5)
                    else:
                        display_fit = fitted_curve

                    # Update the fit series
                    dpg.configure_item(
                        self._tags.fit_series,
                        x=fit_t.tolist(),
                        y=display_fit.tolist(),
                        show=self._show_fit,
                    )

                    logger.debug(
                        f"Full convolved fit curve set: {len(fit_t)} points, "
                        f"chi2={fit_result.chi_squared:.3f}"
                    )
                    return
            except Exception as e:
                logger.warning(f"Failed to compute full convolved curve: {e}")
                # Fall through to use original fitted curve

        # Fallback: Use the pre-computed fitted curve from fit result
        fitted_curve = fit_result.fitted_curve
        fit_t = self._t[fit_start:fit_end]

        # Ensure fitted curve matches the expected length
        if len(fitted_curve) != len(fit_t):
            logger.warning(
                f"Fit curve length {len(fitted_curve)} doesn't match "
                f"time range length {len(fit_t)}"
            )
            min_len = min(len(fitted_curve), len(fit_t))
            fitted_curve = fitted_curve[:min_len]
            fit_t = fit_t[:min_len]

        # Extrapolate to the right using exponential decay
        if fit_end < len(self._t) and len(fitted_curve) > 0 and avg_tau > 0:
            extra_t_right = self._t[fit_end:]
            t0_right = fit_t[-1]
            end_value = fitted_curve[-1]
            extra_curve_right = background + (end_value - background) * np.exp(
                -(extra_t_right - t0_right) / avg_tau
            )
            fit_t = np.concatenate([fit_t, extra_t_right])
            fitted_curve = np.concatenate([fitted_curve, extra_curve_right])

        # For log scale, clip to minimum 0.5
        if self._log_scale:
            display_fit = np.maximum(fitted_curve, 0.5)
        else:
            display_fit = fitted_curve

        # Update the fit series
        dpg.configure_item(
            self._tags.fit_series,
            x=fit_t.tolist(),
            y=display_fit.tolist(),
            show=self._show_fit,
        )

        logger.debug(
            f"Fit curve set: {len(fit_t)} points (extrapolated), "
            f"chi2={fit_result.chi_squared:.3f}"
        )

    def _compute_full_convolved_curve(
        self, fit_result: FitResult
    ) -> Optional[NDArray[np.float64]]:
        """Compute the full convolved fit curve for the entire data range.

        This replicates the convolution computation from the fitting algorithm
        but produces the curve for the full time range, not just the fit range.

        Args:
            fit_result: The fit result with parameters.

        Returns:
            Full convolved curve array, or None if computation fails.
        """
        if self._raw_irf is None or self._t is None or self._counts is None:
            return None

        # Get fit parameters
        taus = list(fit_result.tau)
        amps = list(fit_result.amplitude)
        shift_ns = fit_result.shift
        background = fit_result.background
        fit_start = fit_result.fit_start_index
        fit_end = fit_result.fit_end_index

        # Convert shift from nanoseconds to channels
        shift_channels = shift_ns / self._channelwidth

        # Prepare IRF (subtract background and normalize)
        irf = self._raw_irf.copy()
        # Estimate IRF background from first 20 channels
        irf_bg = np.mean(irf[:20]) if len(irf) >= 20 else 0
        irf_processed = irf - irf_bg
        irf_max = irf_processed.max()
        if irf_max > 0:
            irf_processed = irf_processed / irf_max

        # Build time axis in channel units for model (relative to start)
        n_channels = len(self._t)
        t_channels = np.arange(n_channels) * self._channelwidth

        # Build multi-exponential model
        model = np.zeros(n_channels, dtype=np.float64)
        for tau_val, amp_val in zip(taus, amps):
            if tau_val > 0:
                model += amp_val * np.exp(-t_channels / tau_val)

        # Apply IRF shift and convolve
        irf_shifted = self._colorshift(irf_processed, shift_channels)
        convd = convolve(irf_shifted, model, mode="full")[:n_channels]

        # Normalize to match the fit range intensity
        # Use the same normalization as the fitting algorithm
        measured_in_range = self._counts[fit_start:fit_end].astype(np.float64)
        measured_bg_sub = measured_in_range - background
        measured_bg_sub[measured_bg_sub <= 0] = 0
        meas_sum = np.sum(measured_bg_sub)

        if meas_sum <= 0:
            return None

        # Normalize convolved curve within fit range
        convd_in_range = convd[fit_start:fit_end]
        convd_sum = convd_in_range.sum()
        if convd_sum > 0:
            # Scale full curve to match measured intensity
            bg_norm = background / meas_sum
            convd_normalized = convd / convd_sum
            full_curve = (convd_normalized + bg_norm) * meas_sum
        else:
            return None

        return full_curve

    def clear_fit(self) -> None:
        """Clear the fit curve overlay."""
        self._fit_result = None

        if dpg.does_item_exist(self._tags.fit_series):
            dpg.configure_item(
                self._tags.fit_series,
                x=[],
                y=[],
                show=False,
            )

        logger.debug("Fit curve cleared")

    def set_show_fit(self, show: bool) -> None:
        """Show or hide the fit curve.

        Args:
            show: Whether to show the fit curve.
        """
        self._show_fit = show

        if dpg.does_item_exist(self._tags.fit_series):
            # Only show if we have fit data
            should_show = show and self._fit_result is not None
            dpg.configure_item(self._tags.fit_series, show=should_show)

        logger.debug(f"Fit curve visibility set to {show}")

    def toggle_show_fit(self) -> bool:
        """Toggle fit curve visibility.

        Returns:
            The new visibility state.
        """
        self.set_show_fit(not self._show_fit)
        return self._show_fit

    # -------------------------------------------------------------------------
    # IRF Methods
    # -------------------------------------------------------------------------

    @property
    def has_irf(self) -> bool:
        """Whether IRF data is currently set."""
        return self._irf_t is not None and len(self._irf_t) > 0

    @property
    def show_irf(self) -> bool:
        """Whether the IRF is currently visible."""
        return self._show_irf

    def set_irf(
        self,
        t: NDArray[np.float64],
        counts: NDArray[np.float64],
        normalize: bool = True,
        shift: float = 0.0,
    ) -> None:
        """Set the IRF data and display it on the plot.

        The IRF is typically displayed normalized to match the peak of the
        decay data for easier visual comparison. Only the significant part
        of the IRF is shown (values above 1% of peak).

        Args:
            t: Time array in nanoseconds.
            counts: Count array.
            normalize: If True, normalize IRF to match data peak.
            shift: Time shift to apply to IRF in nanoseconds (from fit result).
        """
        self._irf_t = t
        self._irf_counts = counts

        if not dpg.does_item_exist(self._tags.irf_series):
            return

        if len(t) == 0 or len(counts) == 0:
            self.clear_irf()
            return

        # Apply shift to time array
        display_t = t + shift

        # Normalize IRF to match data peak if requested and data is available
        display_counts = counts.astype(np.float64)
        if normalize and self._counts is not None and len(self._counts) > 0:
            data_max = float(np.max(self._counts))
            irf_max = float(np.max(counts))
            if irf_max > 0:
                display_counts = counts * (data_max / irf_max)

        # Determine minimum display value: use 0.5 to match bar chart visual baseline
        # (bars on log scale don't extend below ~0.5 visually)
        min_display = 0.5

        # Trim IRF to only show significant part (above min_display threshold)
        # Include one extra point on each side so we can extend to exactly min_display
        significant_mask = display_counts >= min_display

        if np.any(significant_mask):
            # Find contiguous region around peak
            significant_indices = np.where(significant_mask)[0]
            start_idx = max(0, significant_indices[0] - 1)
            end_idx = min(len(display_t), significant_indices[-1] + 2)

            display_t = display_t[start_idx:end_idx]
            display_counts = display_counts[start_idx:end_idx]

        # Clip small values to min_display for clean log scale display
        display_counts = np.maximum(display_counts, min_display)

        # Update the IRF series
        dpg.configure_item(
            self._tags.irf_series,
            x=display_t.tolist(),
            y=display_counts.tolist(),
            show=self._show_irf,
        )

        logger.debug(f"IRF set: {len(display_t)} points (trimmed from {len(t)})")

    def clear_irf(self) -> None:
        """Clear the IRF overlay."""
        self._irf_t = None
        self._irf_counts = None

        if dpg.does_item_exist(self._tags.irf_series):
            dpg.configure_item(
                self._tags.irf_series,
                x=[],
                y=[],
                show=False,
            )

        logger.debug("IRF cleared")

    def set_show_irf(self, show: bool) -> None:
        """Show or hide the IRF.

        Args:
            show: Whether to show the IRF.
        """
        self._show_irf = show

        if dpg.does_item_exist(self._tags.irf_series):
            # Only show if we have IRF data
            should_show = show and self.has_irf
            dpg.configure_item(self._tags.irf_series, show=should_show)

        logger.debug(f"IRF visibility set to {show}")

    def toggle_show_irf(self) -> bool:
        """Toggle IRF visibility.

        Returns:
            The new visibility state.
        """
        self.set_show_irf(not self._show_irf)
        return self._show_irf

    def refresh_irf_scaling(self) -> None:
        """Refresh the IRF display scaling after data changes.

        Call this after updating the decay data if you want the IRF
        to be re-normalized to the new data peak.
        """
        if self._irf_t is not None and self._irf_counts is not None:
            self.set_irf(self._irf_t, self._irf_counts, normalize=True)
