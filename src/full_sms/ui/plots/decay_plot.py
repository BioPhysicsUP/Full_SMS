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

from full_sms.analysis.histograms import build_decay_histogram
from full_sms.analysis.lifetime import compute_convolved_fit_curve
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
        # Store the max of the displayed fit curve (full convolved or extrapolated)
        # Used for IRF normalization to match what's visible on screen
        self._displayed_fit_max: float | None = None

        # IRF overlay state
        self._irf_t: NDArray[np.float64] | None = None
        self._irf_counts: NDArray[np.float64] | None = None
        self._irf_shift: float = 0.0  # Time shift applied to IRF display
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

    def build(self, for_subplot: bool = False) -> None:
        """Build the plot UI structure.

        Args:
            for_subplot: If True, build directly inside a subplot context
                        (no container group, plot uses parent from subplot).
        """
        if self._is_built:
            return

        if for_subplot:
            # Build directly inside subplot - no container group needed
            self._build_plot_content()
        else:
            # Standalone mode - create container group
            with dpg.group(parent=self._parent, tag=self._tags.container):
                self._build_plot_content()

        # Apply colors to series
        self._apply_series_colors()

        self._is_built = True
        logger.debug("Decay plot built")

    def _build_plot_content(self) -> None:
        """Build the plot content (used by both standalone and subplot modes)."""
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

        # DearPyGui doesn't support changing log_scale dynamically via configure_item
        # We must rebuild the Y axis with the new scale setting
        self._rebuild_y_axis()

        logger.debug(f"Decay plot log scale set to {log_scale}")

    def _rebuild_y_axis(self) -> None:
        """Rebuild the Y axis and all its series with current log_scale setting.

        This is necessary because DearPyGui's log_scale is a creation-time parameter.
        """
        if not dpg.does_item_exist(self._tags.plot):
            return

        # Store current series data before deletion
        data_x = []
        data_y = []
        fit_x = []
        fit_y = []
        fit_show = self._show_fit and self._fit_result is not None
        irf_x = []
        irf_y = []
        irf_show = self._show_irf

        if dpg.does_item_exist(self._tags.series):
            # Note: DearPyGui stores series data in the item's value
            value = dpg.get_value(self._tags.series)
            if value and len(value) >= 2:
                data_x = list(value[0]) if value[0] else []
                data_y = list(value[1]) if value[1] else []

        if dpg.does_item_exist(self._tags.fit_series):
            value = dpg.get_value(self._tags.fit_series)
            if value and len(value) >= 2:
                fit_x = list(value[0]) if value[0] else []
                fit_y = list(value[1]) if value[1] else []

        if dpg.does_item_exist(self._tags.irf_series):
            value = dpg.get_value(self._tags.irf_series)
            if value and len(value) >= 2:
                irf_x = list(value[0]) if value[0] else []
                irf_y = list(value[1]) if value[1] else []
            irf_show = irf_show and len(irf_x) > 0

        # Delete old Y axis (this also deletes all child series)
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.delete_item(self._tags.y_axis)

        # Create new Y axis with updated log_scale
        dpg.add_plot_axis(
            dpg.mvYAxis,
            label="Counts",
            tag=self._tags.y_axis,
            log_scale=self._log_scale,
            parent=self._tags.plot,
        )

        # Recreate data series
        dpg.add_line_series(
            data_x,
            data_y,
            label="Data",
            parent=self._tags.y_axis,
            tag=self._tags.series,
        )

        # Recreate fit series
        dpg.add_line_series(
            fit_x,
            fit_y,
            label="Fit",
            parent=self._tags.y_axis,
            tag=self._tags.fit_series,
            show=fit_show,
        )

        # Recreate IRF series
        dpg.add_line_series(
            irf_x,
            irf_y,
            label="IRF",
            parent=self._tags.y_axis,
            tag=self._tags.irf_series,
            show=irf_show,
        )

        # Re-apply colors to series
        self._apply_series_colors()

        # Re-update series data to handle zero values appropriately for log scale
        self._update_series()

        # Update fit display if we have a fit result
        if self._fit_result is not None:
            self.set_fit(self._fit_result)

        # Re-render IRF if we have IRF data (to apply appropriate scaling for new mode)
        if self._irf_t is not None and self._irf_counts is not None:
            # Store current show state
            was_showing = self._show_irf
            self.set_irf(
                self._irf_t, self._irf_counts, normalize=True, shift=self._irf_shift
            )
            # Restore show state (set_irf doesn't change _show_irf)
            if dpg.does_item_exist(self._tags.irf_series):
                dpg.configure_item(self._tags.irf_series, show=was_showing)

        # Fit axes to data
        self._fit_axes()

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

                    # Store the max of displayed curve for IRF normalization
                    self._displayed_fit_max = float(np.max(fitted_curve))

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

        # Store the max of displayed curve for IRF normalization
        self._displayed_fit_max = float(np.max(fitted_curve))

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

        Uses the shared compute_convolved_fit_curve function from lifetime module
        to ensure consistency between UI display and export.

        Args:
            fit_result: The fit result with parameters.

        Returns:
            Full convolved curve array, or None if computation fails.
        """
        if self._raw_irf is None or self._t is None or self._counts is None:
            return None

        try:
            full_curve, _ = compute_convolved_fit_curve(
                t_ns=self._t,
                counts=self._counts,
                channelwidth=self._channelwidth,
                tau=fit_result.tau,
                amplitude=fit_result.amplitude,
                shift_ns=fit_result.shift,
                background=fit_result.background,
                fit_start_index=fit_result.fit_start_index,
                fit_end_index=fit_result.fit_end_index,
                irf_array=self._raw_irf,
            )
            # Check if we got a valid curve
            if np.sum(full_curve) <= 0:
                return None
            return full_curve
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to compute convolved curve: {e}")
            return None

    def clear_fit(self) -> None:
        """Clear the fit curve overlay."""
        self._fit_result = None
        self._displayed_fit_max = None

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

        The IRF is displayed normalized to match the peak of the fitted curve
        (or data peak if no fit exists) for easier visual comparison. Only
        the significant part of the IRF is shown.

        Args:
            t: Time array in nanoseconds.
            counts: Count array.
            normalize: If True, normalize IRF to match fit/data peak.
            shift: Time shift to apply to IRF in nanoseconds (from fit result).
        """
        self._irf_t = t
        self._irf_counts = counts
        self._irf_shift = shift

        if not dpg.does_item_exist(self._tags.irf_series):
            return

        if len(t) == 0 or len(counts) == 0:
            self.clear_irf()
            return

        # Apply shift to time array
        display_t = t + shift

        # Normalize IRF to match the fitted curve peak (or data peak if no fit)
        display_counts = counts.astype(np.float64)
        if normalize:
            # Use the displayed fit max (full convolved or extrapolated)
            # instead of fit_result.fitted_curve which is only the fit range
            target_max = None
            if self._displayed_fit_max is not None and self._displayed_fit_max > 0:
                target_max = self._displayed_fit_max
            elif self._counts is not None and len(self._counts) > 0:
                target_max = float(np.max(self._counts))

            if target_max is not None and target_max > 0:
                irf_max = float(np.max(counts))
                if irf_max > 0:
                    display_counts = counts * (target_max / irf_max)

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
        """Clear the IRF overlay and raw IRF data."""
        self._irf_t = None
        self._irf_counts = None
        self._irf_shift = 0.0
        self._raw_irf = None  # Also clear raw IRF to prevent stale data

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
