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
RESIDUALS_COLOR = COLORS["series_4"]  # Pink for residuals (in fit range)
RESIDUALS_OOR_COLOR = (128, 128, 128, 180)  # Gray for out-of-range residuals
ZERO_LINE_COLOR = (150, 150, 150, 200)  # Gray for zero reference


@dataclass
class ResidualsPlotTags:
    """Tags for residuals plot elements."""

    container: str = "residuals_plot_container"
    plot: str = "residuals_plot"
    x_axis: str = "residuals_plot_x_axis"
    y_axis: str = "residuals_plot_y_axis"
    series: str = "residuals_plot_series"  # In-range residuals (colored)
    series_out_of_range: str = "residuals_plot_series_oor"  # Out-of-range (gray)
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
            series_out_of_range=f"{tag_prefix}residuals_plot_series_oor",
            zero_line=f"{tag_prefix}residuals_plot_zero_line",
        )

        # Reference to linked decay plot X axis (for linked panning/zooming)
        self._linked_x_axis: str | None = None
        self._last_linked_limits: tuple[float, float] | None = None

        # Fit range indices for coloring
        self._fit_start_index: int = 0
        self._fit_end_index: int = 0

    @property
    def tags(self) -> ResidualsPlotTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._t is not None and len(self._t) > 0

    def build(self, for_subplot: bool = False) -> None:
        """Build the plot UI structure.

        Args:
            for_subplot: If True, build directly inside a subplot context
                        (no container group, plot uses parent from subplot).
                        X-axis linking is handled by the subplot's link_all_x.
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

        # Apply colors
        self._apply_series_colors()

        self._is_built = True
        logger.debug("Residuals plot built")

    def _build_plot_content(self) -> None:
        """Build the plot content (used by both standalone and subplot modes)."""
        # Create the plot with fixed height
        with dpg.plot(
            tag=self._tags.plot,
            label="Residuals",
            width=-1,
            height=self._height,
            anti_aliased=True,
            no_title=True,  # Save vertical space
        ):
            # X axis (time in nanoseconds) - synced to decay plot
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="Time (ns)",
                tag=self._tags.x_axis,
                no_tick_labels=False,
            )

            # Y axis (weighted residuals) - fixed limits based on data
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

            # Add out-of-range residuals series (gray, drawn first so it's behind)
            dpg.add_stem_series(
                [],
                [],
                parent=self._tags.y_axis,
                tag=self._tags.series_out_of_range,
            )

            # Add in-range residuals series (colored, drawn on top)
            dpg.add_stem_series(
                [],
                [],
                parent=self._tags.y_axis,
                tag=self._tags.series,
            )

    def set_residuals(
        self,
        t: NDArray[np.float64],
        residuals: NDArray[np.float64],
        fit_start_index: int = 0,
        fit_end_index: int | None = None,
    ) -> None:
        """Set the residuals data and update the plot.

        Args:
            t: Time array in nanoseconds (full data range).
            residuals: Weighted residuals array (full data range).
            fit_start_index: Index where fit range starts (residuals before are gray).
            fit_end_index: Index where fit range ends (residuals after are gray).
        """
        if len(t) == 0 or len(residuals) == 0:
            self.clear()
            return

        self._t = t
        self._residuals = residuals
        self._fit_start_index = fit_start_index
        self._fit_end_index = fit_end_index if fit_end_index is not None else len(t)

        # Update the plot
        self._update_series()
        self._update_zero_line()
        self._fit_y_axis()

        logger.debug(
            f"Residuals plot updated: {len(t)} points, "
            f"fit range [{fit_start_index}:{self._fit_end_index}]"
        )

    def _update_series(self) -> None:
        """Update the plot series with current data.

        Splits data into in-range (colored) and out-of-range (gray) portions.
        """
        if self._t is None or self._residuals is None:
            # Clear both series
            if dpg.does_item_exist(self._tags.series):
                dpg.configure_item(self._tags.series, x=[], y=[])
            if dpg.does_item_exist(self._tags.series_out_of_range):
                dpg.configure_item(self._tags.series_out_of_range, x=[], y=[])
            return

        # Split into in-range and out-of-range portions
        fit_start = getattr(self, "_fit_start_index", 0)
        fit_end = getattr(self, "_fit_end_index", len(self._t))

        # In-range data (colored)
        in_range_t = self._t[fit_start:fit_end]
        in_range_residuals = self._residuals[fit_start:fit_end]

        # Out-of-range data (gray) - before and after fit range
        oor_t = np.concatenate([self._t[:fit_start], self._t[fit_end:]])
        oor_residuals = np.concatenate(
            [self._residuals[:fit_start], self._residuals[fit_end:]]
        )

        # Update in-range series
        if dpg.does_item_exist(self._tags.series):
            dpg.configure_item(
                self._tags.series,
                x=in_range_t.tolist(),
                y=in_range_residuals.tolist(),
            )

        # Update out-of-range series
        if dpg.does_item_exist(self._tags.series_out_of_range):
            dpg.configure_item(
                self._tags.series_out_of_range,
                x=oor_t.tolist(),
                y=oor_residuals.tolist(),
            )

    def _update_zero_line(self) -> None:
        """Update the zero reference line."""
        if not dpg.does_item_exist(self._tags.zero_line):
            return

        # Set zero line at y=0
        dpg.configure_item(self._tags.zero_line, x=[0.0])

    def _fit_axes(self) -> None:
        """Auto-fit both axes to show all data."""
        if dpg.does_item_exist(self._tags.x_axis):
            dpg.fit_axis_data(self._tags.x_axis)
        self._fit_y_axis()

    def _fit_y_axis(self) -> None:
        """Fit only the Y axis, centered around zero."""
        if not dpg.does_item_exist(self._tags.y_axis):
            return

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
        if dpg.does_item_exist(self._tags.series_out_of_range):
            dpg.configure_item(self._tags.series_out_of_range, x=[], y=[])

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

    def sync_x_axis_from(self, source_x_axis: str) -> None:
        """Sync X axis limits from another axis.

        Args:
            source_x_axis: Tag of the source X axis to sync from.
        """
        if not dpg.does_item_exist(source_x_axis):
            return
        if not dpg.does_item_exist(self._tags.x_axis):
            return

        try:
            limits = dpg.get_axis_limits(source_x_axis)
            if limits and len(limits) == 2:
                dpg.set_axis_limits(self._tags.x_axis, limits[0], limits[1])
        except Exception:
            pass

    def link_x_axis(self, decay_plot_x_axis: str) -> None:
        """Link this plot's X axis to the decay plot's X axis.

        Sets up a handler to sync X-axis limits when the decay plot changes.

        Args:
            decay_plot_x_axis: Tag of the decay plot's X axis.
        """
        self._linked_x_axis = decay_plot_x_axis
        self._last_linked_limits: tuple[float, float] | None = None

        # Set up a handler registry for syncing
        callback_tag = f"{self._tags.plot}_x_sync_handler"
        if dpg.does_item_exist(callback_tag):
            dpg.delete_item(callback_tag)

        def sync_callback():
            """Sync X axis limits from the linked decay plot."""
            if not self._linked_x_axis or not dpg.does_item_exist(self._linked_x_axis):
                return
            if not dpg.does_item_exist(self._tags.x_axis):
                return

            try:
                limits = dpg.get_axis_limits(self._linked_x_axis)
                if limits and len(limits) == 2:
                    current = tuple(limits)
                    if self._last_linked_limits != current:
                        self._last_linked_limits = current
                        dpg.set_axis_limits(self._tags.x_axis, limits[0], limits[1])
            except Exception:
                pass

        # Register visible handler to sync on each frame when visible
        with dpg.item_handler_registry(tag=callback_tag):
            dpg.add_item_visible_handler(callback=lambda: sync_callback())

        dpg.bind_item_handler_registry(self._tags.plot, callback_tag)
        logger.debug(f"Residuals X axis linked to {decay_plot_x_axis}")

    def _apply_series_colors(self) -> None:
        """Apply themed colors to the plot elements."""
        # In-range residuals series color (colored)
        self._apply_stem_color(self._tags.series, RESIDUALS_COLOR)

        # Out-of-range residuals series color (gray)
        self._apply_stem_color(self._tags.series_out_of_range, RESIDUALS_OOR_COLOR)

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
