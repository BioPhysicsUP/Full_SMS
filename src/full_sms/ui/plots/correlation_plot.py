"""Correlation (g2) plot widget.

Renders the second-order correlation function g2(tau) using DearPyGui's ImPlot.
The correlation is symmetric around tau=0, showing antibunching for single emitters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.analysis.correlation import CorrelationResult

logger = logging.getLogger(__name__)


@dataclass
class CorrelationPlotTags:
    """Tags for correlation plot elements."""

    container: str = "correlation_plot_container"
    plot: str = "correlation_plot"
    x_axis: str = "correlation_plot_x_axis"
    y_axis: str = "correlation_plot_y_axis"
    series: str = "correlation_plot_series"
    zero_line: str = "correlation_plot_zero_line"


CORRELATION_PLOT_TAGS = CorrelationPlotTags()


class CorrelationPlot:
    """Correlation (g2) plot widget.

    Displays the second-order correlation function g2(tau) as a line plot.
    The plot is symmetric around tau=0, with antibunching visible as a dip
    at zero delay for single quantum emitters.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the correlation plot.

        Args:
            parent: The parent container to build the plot in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._tau: NDArray[np.float64] | None = None
        self._g2: NDArray[np.int64] | None = None
        self._correlation_result: CorrelationResult | None = None

        # Generate unique tags
        self._tags = CorrelationPlotTags(
            container=f"{tag_prefix}correlation_plot_container",
            plot=f"{tag_prefix}correlation_plot",
            x_axis=f"{tag_prefix}correlation_plot_x_axis",
            y_axis=f"{tag_prefix}correlation_plot_y_axis",
            series=f"{tag_prefix}correlation_plot_series",
            zero_line=f"{tag_prefix}correlation_plot_zero_line",
        )

    @property
    def tags(self) -> CorrelationPlotTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._tau is not None and len(self._tau) > 0

    @property
    def correlation_result(self) -> CorrelationResult | None:
        """Get the current correlation result."""
        return self._correlation_result

    def build(self) -> None:
        """Build the plot UI structure."""
        if self._is_built:
            return

        # Plot container
        with dpg.group(parent=self._parent, tag=self._tags.container):
            # Create the plot
            with dpg.plot(
                tag=self._tags.plot,
                label="g2 Correlation",
                width=-1,
                height=-1,
                anti_aliased=True,
            ):
                # X axis (tau in nanoseconds)
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="Delay (ns)",
                    tag=self._tags.x_axis,
                )

                # Y axis (correlation counts)
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label="Counts",
                    tag=self._tags.y_axis,
                )

                # Add empty line series (will be populated when data is set)
                dpg.add_line_series(
                    [],
                    [],
                    label="g2",
                    parent=self._tags.y_axis,
                    tag=self._tags.series,
                )

                # Add vertical line at tau=0 (reference)
                dpg.add_inf_line_series(
                    [0.0],
                    label="tau=0",
                    parent=self._tags.y_axis,
                    tag=self._tags.zero_line,
                )
                # Style the zero line
                self._style_zero_line()

        self._is_built = True
        logger.debug("Correlation plot built")

    def _style_zero_line(self) -> None:
        """Apply styling to the zero reference line."""
        if not dpg.does_item_exist(self._tags.zero_line):
            return

        # Create a subtle dashed line style
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvInfLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (128, 128, 128, 100),
                    category=dpg.mvThemeCat_Plots,
                )

        dpg.bind_item_theme(self._tags.zero_line, theme)

    def set_data(self, result: CorrelationResult) -> None:
        """Set the correlation data from a CorrelationResult.

        Args:
            result: The CorrelationResult containing tau and g2 values.
        """
        self._correlation_result = result
        self._tau = result.tau
        self._g2 = result.g2

        # Update the plot
        self._update_series()
        self._fit_axes()

        logger.debug(
            f"Correlation plot updated: {len(self._tau)} bins, "
            f"window={result.window_ns}ns, binsize={result.binsize_ns}ns"
        )

    def set_raw_data(
        self,
        tau: NDArray[np.float64],
        g2: NDArray[np.int64],
    ) -> None:
        """Set raw tau and g2 arrays directly.

        Args:
            tau: Delay time values in nanoseconds.
            g2: Correlation histogram counts.
        """
        self._tau = np.asarray(tau, dtype=np.float64)
        self._g2 = np.asarray(g2, dtype=np.int64)
        self._correlation_result = None

        # Update the plot
        self._update_series()
        self._fit_axes()

        logger.debug(f"Correlation plot updated: {len(self._tau)} bins")

    def _update_series(self) -> None:
        """Update the plot series with current data."""
        if not dpg.does_item_exist(self._tags.series):
            return

        if self._tau is None or self._g2 is None:
            dpg.configure_item(self._tags.series, x=[], y=[])
            return

        # Convert to lists for DearPyGui
        dpg.configure_item(
            self._tags.series,
            x=self._tau.tolist(),
            y=self._g2.tolist(),
        )

    def _fit_axes(self) -> None:
        """Auto-fit the axes to show all data."""
        if not dpg.does_item_exist(self._tags.x_axis):
            return

        dpg.fit_axis_data(self._tags.x_axis)
        dpg.fit_axis_data(self._tags.y_axis)

    def clear(self) -> None:
        """Clear the plot data."""
        self._tau = None
        self._g2 = None
        self._correlation_result = None
        self._update_series()
        logger.debug("Correlation plot cleared")

    def fit_view(self) -> None:
        """Fit the view to show all data (reset zoom/pan)."""
        self._fit_axes()

    def get_tau_range(self) -> tuple[float, float] | None:
        """Get the current tau range of the data.

        Returns:
            Tuple of (min_tau, max_tau) in nanoseconds, or None if no data.
        """
        if self._tau is None or len(self._tau) == 0:
            return None
        return (float(self._tau[0]), float(self._tau[-1]))

    def get_g2_range(self) -> tuple[int, int] | None:
        """Get the current g2 count range.

        Returns:
            Tuple of (min_counts, max_counts), or None if no data.
        """
        if self._g2 is None or len(self._g2) == 0:
            return None
        return (int(np.min(self._g2)), int(np.max(self._g2)))

    def get_g2_at_zero(self) -> int | None:
        """Get the g2 value at tau=0 (antibunching dip).

        Returns:
            The g2 count at the bin closest to tau=0, or None if no data.
        """
        if self._tau is None or self._g2 is None:
            return None

        # Find the bin closest to tau=0
        zero_idx = np.argmin(np.abs(self._tau))
        return int(self._g2[zero_idx])
