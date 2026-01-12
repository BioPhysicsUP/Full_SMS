"""Intensity histogram plot widget.

Displays distribution of binned photon counts as a vertical histogram sidebar.
Shows frequency (normalized) on X-axis and intensity values on Y-axis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.ui.theme import COLORS

logger = logging.getLogger(__name__)


# Default number of histogram bins
DEFAULT_NUM_BINS = 50


@dataclass
class IntensityHistogramTags:
    """Tags for intensity histogram plot elements."""

    container: str = "intensity_histogram_container"
    plot: str = "intensity_histogram_plot"
    x_axis: str = "intensity_histogram_x_axis"
    y_axis: str = "intensity_histogram_y_axis"
    series: str = "intensity_histogram_series"


INTENSITY_HISTOGRAM_TAGS = IntensityHistogramTags()


class IntensityHistogram:
    """Intensity histogram plot widget.

    Displays the distribution of intensity values as a vertical histogram.
    The histogram is drawn horizontally (frequency on X, intensity on Y)
    to align with the intensity trace plot's Y-axis.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
        width: int = 150,
    ) -> None:
        """Initialize the intensity histogram.

        Args:
            parent: The parent container to build the plot in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
            width: Width of the histogram plot in pixels.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._width = width
        self._is_built = False

        # Data state
        self._counts: NDArray[np.int64] | None = None
        self._num_bins: int = DEFAULT_NUM_BINS

        # Cached histogram data
        self._hist_freqs: NDArray[np.float64] | None = None
        self._hist_bins: NDArray[np.float64] | None = None
        self._bin_width: float = 1.0  # Width of each histogram bin

        # Generate unique tags
        self._tags = IntensityHistogramTags(
            container=f"{tag_prefix}intensity_histogram_container",
            plot=f"{tag_prefix}intensity_histogram_plot",
            x_axis=f"{tag_prefix}intensity_histogram_x_axis",
            y_axis=f"{tag_prefix}intensity_histogram_y_axis",
            series=f"{tag_prefix}intensity_histogram_series",
        )

    @property
    def tags(self) -> IntensityHistogramTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def num_bins(self) -> int:
        """Get the current number of histogram bins."""
        return self._num_bins

    def build(self, for_subplot: bool = False) -> None:
        """Build the plot UI structure.

        Args:
            for_subplot: If True, build the plot directly without a container group.
                Use this when the plot is being added inside a dpg.subplots() context.
        """
        if self._is_built:
            return

        if for_subplot:
            # Build plot directly (for use inside dpg.subplots)
            self._build_plot_content()
        else:
            # Build with container group (standalone mode)
            with dpg.group(parent=self._parent, tag=self._tags.container):
                self._build_plot_content()

        self._is_built = True
        logger.debug("Intensity histogram built")

    def _build_plot_content(self) -> None:
        """Build the plot content (plot, axes, series)."""
        # Create the plot - oriented horizontally
        # Y-axis = intensity values (to match main plot)
        # X-axis = frequency (normalized)
        with dpg.plot(
            tag=self._tags.plot,
            label="",  # No title needed
            width=self._width,
            height=-1,  # Fill available height
            no_title=True,
            no_mouse_pos=True,
            no_inputs=True,  # Disable zoom/pan - Y-axis is linked to intensity plot
        ):
            # X axis (frequency, hidden label)
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="",
                tag=self._tags.x_axis,
                no_tick_labels=True,
                no_tick_marks=True,
            )

            # Y axis (intensity values - should align with main plot)
            dpg.add_plot_axis(
                dpg.mvYAxis,
                label="",
                tag=self._tags.y_axis,
                no_tick_labels=True,  # Main plot shows the labels
            )

            # Add empty bar series (horizontal bars for histogram)
            dpg.add_bar_series(
                [],
                [],
                parent=self._tags.y_axis,
                tag=self._tags.series,
                horizontal=True,
                weight=1.0,
            )

    def set_data(self, counts: NDArray[np.int64]) -> None:
        """Set the binned count data and update the histogram.

        Args:
            counts: Array of binned photon counts.
        """
        if len(counts) == 0:
            self.clear()
            return

        self._counts = counts
        self._compute_histogram()
        self._update_series()

        logger.debug(f"Intensity histogram updated: {len(counts)} bins")

    def _compute_histogram(self) -> None:
        """Compute the histogram from current count data."""
        if self._counts is None or len(self._counts) == 0:
            self._hist_freqs = None
            self._hist_bins = None
            self._bin_width = 1.0
            return

        # Compute histogram with fixed number of bins
        # Use density=True for normalized frequency
        freqs, bin_edges = np.histogram(
            self._counts,
            bins=self._num_bins,
            density=True,
        )

        # Normalize to max = 1.0 for display
        if freqs.max() > 0:
            freqs = freqs / freqs.max()

        # Compute bin centers for bar chart
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate bin width for bar series weight
        self._bin_width = bin_edges[1] - bin_edges[0]

        self._hist_freqs = freqs
        self._hist_bins = bin_centers

    def _update_series(self) -> None:
        """Update the plot series with current histogram data."""
        if not dpg.does_item_exist(self._tags.series):
            return

        if self._hist_freqs is None or self._hist_bins is None:
            dpg.configure_item(self._tags.series, x=[], y=[])
            return

        # For horizontal bar series:
        # x = bar length (frequency values)
        # y = bar position (intensity values / bin centers)
        # weight = bar width (matches histogram bin width so bars are adjacent)
        dpg.configure_item(
            self._tags.series,
            x=self._hist_freqs.tolist(),
            y=self._hist_bins.tolist(),
            weight=self._bin_width,
        )

        # Fit X-axis
        dpg.fit_axis_data(self._tags.x_axis)

        # Set Y-axis limits: 0 to max with 20% padding above highest bar
        if self._counts is not None and len(self._counts) > 0:
            max_val = float(self._hist_freqs.max())
            dpg.set_axis_limits(self._tags.x_axis, 0, max_val * 1.20)

    def _fit_axes(self) -> None:
        """Auto-fit the axes with Y-axis padding."""
        if not dpg.does_item_exist(self._tags.x_axis):
            return

        dpg.fit_axis_data(self._tags.x_axis)

        # Set Y-axis limits: 0 to max with 20% padding
        if self._counts is not None and len(self._counts) > 0:
            max_val = float(self._counts.max())
            dpg.set_axis_limits(self._tags.y_axis, 0, max_val * 1.20)
            dpg.set_axis_limits_auto(self._tags.y_axis)

    def clear(self) -> None:
        """Clear the histogram data."""
        self._counts = None
        self._hist_freqs = None
        self._hist_bins = None
        self._bin_width = 1.0

        if dpg.does_item_exist(self._tags.series):
            dpg.configure_item(self._tags.series, x=[], y=[])

        logger.debug("Intensity histogram cleared")

    @property
    def has_data(self) -> bool:
        """Whether the histogram has data loaded."""
        return self._counts is not None and len(self._counts) > 0

    def set_num_bins(self, num_bins: int) -> None:
        """Set the number of histogram bins and recompute.

        Args:
            num_bins: Number of histogram bins (e.g., 50, 100).
        """
        if num_bins < 1:
            num_bins = 1
        elif num_bins > 500:
            num_bins = 500

        self._num_bins = num_bins

        if self._counts is not None:
            self._compute_histogram()
            self._update_series()

    def set_y_limits(self, y_min: float, y_max: float) -> None:
        """Set the Y-axis limits to match the main intensity plot.

        Args:
            y_min: Minimum Y value.
            y_max: Maximum Y value.
        """
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.set_axis_limits(self._tags.y_axis, y_min, y_max)

    def fit_y_axis(self) -> None:
        """Fit Y-axis to data."""
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.fit_axis_data(self._tags.y_axis)

    def get_y_limits(self) -> tuple[float, float] | None:
        """Get the current Y-axis limits.

        Returns:
            Tuple of (min, max) Y values, or None if not available.
        """
        if not dpg.does_item_exist(self._tags.y_axis):
            return None

        limits = dpg.get_axis_limits(self._tags.y_axis)
        return (limits[0], limits[1])
