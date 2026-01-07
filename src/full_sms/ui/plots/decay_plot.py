"""Fluorescence decay histogram plot widget.

Renders TCSPC decay data using DearPyGui's ImPlot with optional log-scale Y axis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.analysis.histograms import build_decay_histogram
from full_sms.ui.theme import COLORS

logger = logging.getLogger(__name__)


@dataclass
class DecayPlotTags:
    """Tags for decay plot elements."""

    container: str = "decay_plot_container"
    plot: str = "decay_plot"
    x_axis: str = "decay_plot_x_axis"
    y_axis: str = "decay_plot_y_axis"
    series: str = "decay_plot_series"


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

        # Generate unique tags
        self._tags = DecayPlotTags(
            container=f"{tag_prefix}decay_plot_container",
            plot=f"{tag_prefix}decay_plot",
            x_axis=f"{tag_prefix}decay_plot_x_axis",
            y_axis=f"{tag_prefix}decay_plot_y_axis",
            series=f"{tag_prefix}decay_plot_series",
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

                # Add empty line series (will be populated when data is set)
                dpg.add_line_series(
                    [],
                    [],
                    label="Decay",
                    parent=self._tags.y_axis,
                    tag=self._tags.series,
                )

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
        self._update_series()
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
