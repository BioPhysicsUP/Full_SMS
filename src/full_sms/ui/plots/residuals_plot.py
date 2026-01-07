"""Weighted residuals plot widget.

Displays weighted residuals from lifetime fitting below the decay plot.
Shows residuals vs time with a zero reference line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.ui.theme import COLORS

logger = logging.getLogger(__name__)


# Color for residuals plot
RESIDUALS_COLOR = COLORS["series_4"]  # Pink for residuals
ZERO_LINE_COLOR = (150, 150, 150, 200)  # Gray for zero reference


@dataclass
class ResidualsPlotTags:
    """Tags for residuals plot elements."""

    container: str = "residuals_plot_container"
    plot: str = "residuals_plot"
    x_axis: str = "residuals_plot_x_axis"
    y_axis: str = "residuals_plot_y_axis"
    series: str = "residuals_plot_series"
    zero_line: str = "residuals_plot_zero_line"


class ResidualsPlot:
    """Weighted residuals plot widget.

    Displays the weighted residuals (data - fit) / sigma as a scatter or line plot
    vs time. Includes a zero reference line for easy visualization of fit quality.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
        height: int = 120,
    ) -> None:
        """Initialize the residuals plot.

        Args:
            parent: The parent container to build the plot in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
            height: Fixed height for the residuals plot in pixels.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._height = height
        self._is_built = False

        # Data state
        self._t: NDArray[np.float64] | None = None
        self._residuals: NDArray[np.float64] | None = None

        # Generate unique tags
        self._tags = ResidualsPlotTags(
            container=f"{tag_prefix}residuals_plot_container",
            plot=f"{tag_prefix}residuals_plot",
            x_axis=f"{tag_prefix}residuals_plot_x_axis",
            y_axis=f"{tag_prefix}residuals_plot_y_axis",
            series=f"{tag_prefix}residuals_plot_series",
            zero_line=f"{tag_prefix}residuals_plot_zero_line",
        )

        # Reference to linked decay plot X axis (for linked panning/zooming)
        self._linked_x_axis: str | None = None

    @property
    def tags(self) -> ResidualsPlotTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._t is not None and len(self._t) > 0

    def build(self) -> None:
        """Build the plot UI structure."""
        if self._is_built:
            return

        # Plot container
        with dpg.group(parent=self._parent, tag=self._tags.container):
            # Create the plot with fixed height
            with dpg.plot(
                tag=self._tags.plot,
                label="Residuals",
                width=-1,
                height=self._height,
                anti_aliased=True,
                no_title=True,  # Save vertical space
            ):
                # X axis (time in nanoseconds) - will be linked to decay plot
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="Time (ns)",
                    tag=self._tags.x_axis,
                    no_tick_labels=False,
                )

                # Y axis (weighted residuals)
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label="Residuals",
                    tag=self._tags.y_axis,
                )

                # Add zero reference line (horizontal line at y=0)
                dpg.add_inf_line_series(
                    x=[],  # Empty for horizontal line
                    horizontal=True,
                    parent=self._tags.y_axis,
                    tag=self._tags.zero_line,
                )

                # Add residuals series (scatter/stem plot style)
                dpg.add_stem_series(
                    [],
                    [],
                    parent=self._tags.y_axis,
                    tag=self._tags.series,
                )

        # Apply colors
        self._apply_series_colors()

        self._is_built = True
        logger.debug("Residuals plot built")

    def set_residuals(
        self,
        t: NDArray[np.float64],
        residuals: NDArray[np.float64],
    ) -> None:
        """Set the residuals data and update the plot.

        Args:
            t: Time array in nanoseconds (from the fit range).
            residuals: Weighted residuals array.
        """
        if len(t) == 0 or len(residuals) == 0:
            self.clear()
            return

        self._t = t
        self._residuals = residuals

        # Update the plot
        self._update_series()
        self._update_zero_line()
        self._fit_axes()

        logger.debug(f"Residuals plot updated: {len(t)} points")

    def _update_series(self) -> None:
        """Update the plot series with current data."""
        if not dpg.does_item_exist(self._tags.series):
            return

        if self._t is None or self._residuals is None:
            dpg.configure_item(self._tags.series, x=[], y=[])
            return

        # Convert to lists for DearPyGui
        dpg.configure_item(
            self._tags.series,
            x=self._t.tolist(),
            y=self._residuals.tolist(),
        )

    def _update_zero_line(self) -> None:
        """Update the zero reference line."""
        if not dpg.does_item_exist(self._tags.zero_line):
            return

        # Set zero line at y=0
        dpg.configure_item(self._tags.zero_line, x=[0.0])

    def _fit_axes(self) -> None:
        """Auto-fit the axes to show all data."""
        if not dpg.does_item_exist(self._tags.x_axis):
            return

        dpg.fit_axis_data(self._tags.x_axis)
        dpg.fit_axis_data(self._tags.y_axis)

        # Center Y axis around zero with some padding
        if self._residuals is not None and len(self._residuals) > 0:
            max_abs = np.max(np.abs(self._residuals))
            # Add 10% padding
            y_limit = max_abs * 1.1
            if y_limit > 0:
                dpg.set_axis_limits(self._tags.y_axis, -y_limit, y_limit)

    def clear(self) -> None:
        """Clear the plot data."""
        self._t = None
        self._residuals = None

        if dpg.does_item_exist(self._tags.series):
            dpg.configure_item(self._tags.series, x=[], y=[])

        logger.debug("Residuals plot cleared")

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

    def link_x_axis(self, decay_plot_x_axis: str) -> None:
        """Link this plot's X axis to the decay plot's X axis.

        This stores the reference for potential future use. Note that DearPyGui's
        ImPlot backend doesn't support direct axis linking between separate plots.
        Synchronized zooming would require manual callback handling.

        Args:
            decay_plot_x_axis: Tag of the decay plot's X axis.
        """
        self._linked_x_axis = decay_plot_x_axis
        logger.debug(f"Residuals X axis reference set to {decay_plot_x_axis}")

    def _apply_series_colors(self) -> None:
        """Apply themed colors to the plot elements."""
        # Residuals series color
        self._apply_stem_color(self._tags.series, RESIDUALS_COLOR)

        # Zero line color
        self._apply_line_color(self._tags.zero_line, ZERO_LINE_COLOR)

    def _apply_stem_color(
        self, tag: str, color: tuple[int, int, int, int]
    ) -> None:
        """Apply a color to a stem series via theme.

        Args:
            tag: The tag of the stem series.
            color: RGBA color tuple.
        """
        if not dpg.does_item_exist(tag):
            return

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvStemSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots
                )

        dpg.bind_item_theme(tag, theme)

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
            with dpg.theme_component(dpg.mvInfLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots
                )

        dpg.bind_item_theme(tag, theme)

    def show(self, visible: bool = True) -> None:
        """Show or hide the residuals plot.

        Args:
            visible: Whether to show the plot.
        """
        if dpg.does_item_exist(self._tags.container):
            dpg.configure_item(self._tags.container, show=visible)
