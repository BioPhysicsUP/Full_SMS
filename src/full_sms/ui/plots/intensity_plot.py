"""Intensity trace plot widget.

Renders binned photon counts over time using DearPyGui's ImPlot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.analysis.histograms import bin_photons, compute_intensity_cps
from full_sms.ui.theme import COLORS

logger = logging.getLogger(__name__)


@dataclass
class IntensityPlotTags:
    """Tags for intensity plot elements."""

    container: str = "intensity_plot_container"
    plot: str = "intensity_plot"
    x_axis: str = "intensity_plot_x_axis"
    y_axis: str = "intensity_plot_y_axis"
    series: str = "intensity_plot_series"


INTENSITY_PLOT_TAGS = IntensityPlotTags()


class IntensityPlot:
    """Intensity trace plot widget.

    Displays binned photon counts over time as a line plot.
    Supports configurable bin size with zoom and pan enabled.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the intensity plot.

        Args:
            parent: The parent container to build the plot in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._times: NDArray[np.float64] | None = None
        self._counts: NDArray[np.int64] | None = None
        self._bin_size_ms: float = 10.0  # Default 10ms bins

        # Generate unique tags
        self._tags = IntensityPlotTags(
            container=f"{tag_prefix}intensity_plot_container",
            plot=f"{tag_prefix}intensity_plot",
            x_axis=f"{tag_prefix}intensity_plot_x_axis",
            y_axis=f"{tag_prefix}intensity_plot_y_axis",
            series=f"{tag_prefix}intensity_plot_series",
        )

    @property
    def tags(self) -> IntensityPlotTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def bin_size_ms(self) -> float:
        """Get the current bin size in milliseconds."""
        return self._bin_size_ms

    def build(self) -> None:
        """Build the plot UI structure."""
        if self._is_built:
            return

        # Plot container
        with dpg.group(parent=self._parent, tag=self._tags.container):
            # Create the plot
            with dpg.plot(
                tag=self._tags.plot,
                label="Intensity Trace",
                width=-1,
                height=-1,
                anti_aliased=True,
            ):
                # X axis (time in milliseconds or seconds)
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="Time (ms)",
                    tag=self._tags.x_axis,
                )

                # Y axis (counts per bin or counts per second)
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label="Counts/bin",
                    tag=self._tags.y_axis,
                )

                # Add empty line series (will be populated when data is set)
                dpg.add_line_series(
                    [],
                    [],
                    label="Intensity",
                    parent=self._tags.y_axis,
                    tag=self._tags.series,
                )

        self._is_built = True
        logger.debug("Intensity plot built")

    def set_data(
        self,
        abstimes: NDArray[np.uint64],
        bin_size_ms: float | None = None,
    ) -> None:
        """Set the photon arrival time data and update the plot.

        Args:
            abstimes: Absolute photon arrival times in nanoseconds.
            bin_size_ms: Optional bin size in milliseconds. If None, uses
                the current bin size.
        """
        if bin_size_ms is not None:
            self._bin_size_ms = bin_size_ms

        if len(abstimes) == 0:
            self.clear()
            return

        # Bin the photons
        self._times, self._counts = bin_photons(
            abstimes.astype(np.float64), self._bin_size_ms
        )

        # Update the plot
        self._update_series()
        self._fit_axes()

        logger.debug(
            f"Intensity plot updated: {len(self._times)} bins at {self._bin_size_ms}ms"
        )

    def update_bin_size(self, abstimes: NDArray[np.uint64], bin_size_ms: float) -> None:
        """Update the bin size and rebin the data.

        Args:
            abstimes: Absolute photon arrival times in nanoseconds.
            bin_size_ms: New bin size in milliseconds.
        """
        self.set_data(abstimes, bin_size_ms)

    def _update_series(self) -> None:
        """Update the plot series with current data."""
        if not dpg.does_item_exist(self._tags.series):
            return

        if self._times is None or self._counts is None:
            dpg.configure_item(self._tags.series, x=[], y=[])
            return

        # Convert to lists for DearPyGui
        dpg.configure_item(
            self._tags.series,
            x=self._times.tolist(),
            y=self._counts.tolist(),
        )

    def _fit_axes(self) -> None:
        """Auto-fit the axes to show all data."""
        if not dpg.does_item_exist(self._tags.x_axis):
            return

        dpg.fit_axis_data(self._tags.x_axis)
        dpg.fit_axis_data(self._tags.y_axis)

    def clear(self) -> None:
        """Clear the plot data."""
        self._times = None
        self._counts = None
        self._update_series()
        logger.debug("Intensity plot cleared")

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
            Tuple of (min_time, max_time) in milliseconds, or None if no data.
        """
        if self._times is None or len(self._times) == 0:
            return None
        return (float(self._times[0]), float(self._times[-1]))

    def get_count_range(self) -> tuple[int, int] | None:
        """Get the current count range of the data.

        Returns:
            Tuple of (min_counts, max_counts), or None if no data.
        """
        if self._counts is None or len(self._counts) == 0:
            return None
        return (int(np.min(self._counts)), int(np.max(self._counts)))

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._times is not None and len(self._times) > 0

    def fit_view(self) -> None:
        """Fit the view to show all data (reset zoom/pan)."""
        self._fit_axes()
