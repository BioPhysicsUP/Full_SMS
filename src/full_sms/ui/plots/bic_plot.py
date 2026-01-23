"""BIC (Bayesian Information Criterion) plot widget for clustering visualization.

Renders the BIC optimization curve to help users select the optimal number of groups
from hierarchical clustering analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.models.group import ClusteringResult
from full_sms.ui.theme import COLORS

logger = logging.getLogger(__name__)


# Colors for BIC plot elements
BIC_COLORS = {
    "curve": COLORS["series_1"],  # Blue for BIC curve
    "optimal": COLORS["series_3"],  # Green for optimal point
    "selected": COLORS["series_2"],  # Orange for selected point
}


@dataclass
class BICPlotTags:
    """Tags for BIC plot elements."""

    container: str = "bic_plot_container"
    plot: str = "bic_plot"
    x_axis: str = "bic_plot_x_axis"
    y_axis: str = "bic_plot_y_axis"
    series: str = "bic_plot_series"
    all_points_scatter: str = "bic_plot_all_points_scatter"
    optimal_scatter: str = "bic_plot_optimal_scatter"
    selected_scatter: str = "bic_plot_selected_scatter"
    legend: str = "bic_plot_legend"


class BICPlot:
    """BIC optimization curve plot widget.

    Displays BIC values vs number of groups from hierarchical clustering.
    Features:
    - BIC curve as line plot
    - Highlighted optimal point (maximum BIC)
    - Highlighted selected point (may differ from optimal)
    - Click to select different group count
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
        height: int = -1,
    ) -> None:
        """Initialize the BIC plot.

        Args:
            parent: The parent container to build the plot in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
            height: Plot height in pixels, -1 for auto.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._height = height
        self._is_built = False

        # Data state
        self._num_groups: NDArray[np.int64] | None = None
        self._bic_values: NDArray[np.float64] | None = None
        self._optimal_index: int = 0
        self._selected_index: int = 0
        self._clustering_result: Optional[ClusteringResult] = None

        # Callback for when user clicks to select a group count
        self._on_group_selected: Callable[[int], None] | None = None

        # Generate unique tags
        self._tags = BICPlotTags(
            container=f"{tag_prefix}bic_plot_container",
            plot=f"{tag_prefix}bic_plot",
            x_axis=f"{tag_prefix}bic_plot_x_axis",
            y_axis=f"{tag_prefix}bic_plot_y_axis",
            series=f"{tag_prefix}bic_plot_series",
            all_points_scatter=f"{tag_prefix}bic_plot_all_points_scatter",
            optimal_scatter=f"{tag_prefix}bic_plot_optimal_scatter",
            selected_scatter=f"{tag_prefix}bic_plot_selected_scatter",
            legend=f"{tag_prefix}bic_plot_legend",
        )

    @property
    def tags(self) -> BICPlotTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def selected_num_groups(self) -> int:
        """Get the currently selected number of groups."""
        if self._num_groups is None or len(self._num_groups) == 0:
            return 0
        return int(self._num_groups[self._selected_index])

    @property
    def optimal_num_groups(self) -> int:
        """Get the optimal number of groups (maximum BIC)."""
        if self._num_groups is None or len(self._num_groups) == 0:
            return 0
        return int(self._num_groups[self._optimal_index])

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._num_groups is not None and len(self._num_groups) > 0

    def build(self) -> None:
        """Build the plot UI structure."""
        if self._is_built:
            return

        # Plot container
        with dpg.group(parent=self._parent, tag=self._tags.container):
            # Create the plot
            with dpg.plot(
                tag=self._tags.plot,
                label="BIC Optimization",
                width=-1,
                height=self._height,
                anti_aliased=True,
                callback=self._on_plot_clicked,
            ):
                # Add legend
                dpg.add_plot_legend(tag=self._tags.legend, location=dpg.mvPlot_Location_SouthEast)

                # X axis (number of groups)
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="Number of Groups",
                    tag=self._tags.x_axis,
                )

                # Y axis (BIC value)
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label="BIC",
                    tag=self._tags.y_axis,
                )

                # Add empty line series for BIC curve
                dpg.add_line_series(
                    [],
                    [],
                    label="BIC",
                    parent=self._tags.y_axis,
                    tag=self._tags.series,
                )

                # Add scatter series for all points (blue, same as line)
                dpg.add_scatter_series(
                    [],
                    [],
                    parent=self._tags.y_axis,
                    tag=self._tags.all_points_scatter,
                )

                # Add scatter series for optimal point (green)
                dpg.add_scatter_series(
                    [],
                    [],
                    label="Optimal",
                    parent=self._tags.y_axis,
                    tag=self._tags.optimal_scatter,
                )

                # Add scatter series for selected point (orange)
                dpg.add_scatter_series(
                    [],
                    [],
                    label="Selected",
                    parent=self._tags.y_axis,
                    tag=self._tags.selected_scatter,
                )

        # Apply colors to series
        self._apply_series_colors()

        self._is_built = True
        logger.debug("BIC plot built")

    def set_clustering_result(self, result: ClusteringResult) -> None:
        """Set clustering result and update the plot.

        Args:
            result: The ClusteringResult from hierarchical clustering.
        """
        self._clustering_result = result

        # Extract BIC values and number of groups for each step
        # Steps go from N-1 groups down to 1 group
        bic_values = []
        num_groups = []

        for step in result.steps:
            num_groups.append(step.num_groups)
            bic_values.append(step.bic)

        # Convert to arrays and sort by num_groups (ascending)
        indices = np.argsort(num_groups)
        self._num_groups = np.array(num_groups)[indices]
        self._bic_values = np.array(bic_values)[indices]

        # Find optimal and selected indices in sorted arrays
        optimal_step = result.steps[result.optimal_step_index]
        selected_step = result.steps[result.selected_step_index]

        self._optimal_index = int(np.where(self._num_groups == optimal_step.num_groups)[0][0])
        self._selected_index = int(np.where(self._num_groups == selected_step.num_groups)[0][0])

        # Update the plot
        self._update_series()
        self._fit_axes()

        logger.debug(
            f"BIC plot updated: {len(result.steps)} steps, "
            f"optimal={optimal_step.num_groups} groups, "
            f"selected={selected_step.num_groups} groups"
        )

    def set_data(
        self,
        num_groups: NDArray[np.int64],
        bic_values: NDArray[np.float64],
        optimal_index: int,
        selected_index: int | None = None,
    ) -> None:
        """Set BIC data directly.

        Args:
            num_groups: Array of group counts for each step.
            bic_values: Array of BIC values for each step.
            optimal_index: Index of optimal (maximum BIC) point.
            selected_index: Index of selected point (defaults to optimal).
        """
        if len(num_groups) == 0 or len(bic_values) == 0:
            self.clear()
            return

        # Sort by num_groups
        indices = np.argsort(num_groups)
        self._num_groups = num_groups[indices].astype(np.int64)
        self._bic_values = bic_values[indices].astype(np.float64)

        self._optimal_index = optimal_index
        self._selected_index = selected_index if selected_index is not None else optimal_index

        # Update the plot
        self._update_series()
        self._fit_axes()

        logger.debug(
            f"BIC plot data set: {len(num_groups)} points, "
            f"optimal index={optimal_index}"
        )

    def set_selected_index(self, index: int) -> None:
        """Set the selected point index.

        Args:
            index: Index of the selected point in the sorted arrays.
        """
        if self._num_groups is None:
            return

        if 0 <= index < len(self._num_groups):
            self._selected_index = index
            self._update_selected_scatter()
            logger.debug(f"BIC plot selection changed: {self._num_groups[index]} groups")

    def set_selected_num_groups(self, num_groups: int) -> None:
        """Set the selection by number of groups.

        Args:
            num_groups: The number of groups to select.
        """
        if self._num_groups is None:
            return

        matches = np.where(self._num_groups == num_groups)[0]
        if len(matches) > 0:
            self.set_selected_index(int(matches[0]))

    def _update_series(self) -> None:
        """Update all plot series with current data."""
        if not dpg.does_item_exist(self._tags.series):
            return

        if self._num_groups is None or self._bic_values is None:
            dpg.configure_item(self._tags.series, x=[], y=[])
            dpg.configure_item(self._tags.all_points_scatter, x=[], y=[])
            dpg.configure_item(self._tags.optimal_scatter, x=[], y=[])
            dpg.configure_item(self._tags.selected_scatter, x=[], y=[])
            return

        # Update main BIC curve
        dpg.configure_item(
            self._tags.series,
            x=self._num_groups.tolist(),
            y=self._bic_values.tolist(),
        )

        # Update all points scatter (blue markers)
        dpg.configure_item(
            self._tags.all_points_scatter,
            x=self._num_groups.tolist(),
            y=self._bic_values.tolist(),
        )

        # Update optimal point scatter
        self._update_optimal_scatter()

        # Update selected point scatter
        self._update_selected_scatter()

    def _update_optimal_scatter(self) -> None:
        """Update the optimal point scatter marker."""
        if not dpg.does_item_exist(self._tags.optimal_scatter):
            return

        if self._num_groups is None or self._bic_values is None:
            dpg.configure_item(self._tags.optimal_scatter, x=[], y=[])
            return

        opt_x = [float(self._num_groups[self._optimal_index])]
        opt_y = [float(self._bic_values[self._optimal_index])]

        dpg.configure_item(
            self._tags.optimal_scatter,
            x=opt_x,
            y=opt_y,
        )

    def _update_selected_scatter(self) -> None:
        """Update the selected point scatter marker."""
        if not dpg.does_item_exist(self._tags.selected_scatter):
            return

        if self._num_groups is None or self._bic_values is None:
            dpg.configure_item(self._tags.selected_scatter, x=[], y=[])
            return

        # Only show selected marker if different from optimal
        if self._selected_index != self._optimal_index:
            sel_x = [float(self._num_groups[self._selected_index])]
            sel_y = [float(self._bic_values[self._selected_index])]
        else:
            sel_x = []
            sel_y = []

        dpg.configure_item(
            self._tags.selected_scatter,
            x=sel_x,
            y=sel_y,
        )

    def _fit_axes(self) -> None:
        """Auto-fit the axes to show all data with padding."""
        if not dpg.does_item_exist(self._tags.x_axis):
            return

        dpg.fit_axis_data(self._tags.x_axis)

        # Fit Y-axis with padding at the top so max value is visible
        if self._bic_values is not None and len(self._bic_values) > 0:
            y_min = float(np.min(self._bic_values))
            y_max = float(np.max(self._bic_values))
            y_range = y_max - y_min
            # Add 10% padding at top and bottom
            padding = y_range * 0.1 if y_range > 0 else abs(y_max) * 0.1
            dpg.set_axis_limits(self._tags.y_axis, y_min - padding, y_max + padding)
        else:
            dpg.fit_axis_data(self._tags.y_axis)

    def clear(self) -> None:
        """Clear the plot data."""
        self._num_groups = None
        self._bic_values = None
        self._optimal_index = 0
        self._selected_index = 0
        self._clustering_result = None
        self._update_series()
        logger.debug("BIC plot cleared")

    def _apply_series_colors(self) -> None:
        """Apply themed colors to all series."""
        # BIC curve - blue line
        self._apply_line_color(self._tags.series, BIC_COLORS["curve"])

        # All points - blue scatter (same as line)
        self._apply_scatter_color(self._tags.all_points_scatter, BIC_COLORS["curve"])

        # Optimal point - green scatter
        self._apply_scatter_color(self._tags.optimal_scatter, BIC_COLORS["optimal"])

        # Selected point - orange scatter
        self._apply_scatter_color(self._tags.selected_scatter, BIC_COLORS["selected"])

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
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 2.0, category=dpg.mvThemeCat_Plots
                )

        dpg.bind_item_theme(tag, theme)

    def _apply_scatter_color(
        self, tag: str, color: tuple[int, int, int, int]
    ) -> None:
        """Apply a color to a scatter series via theme.

        Args:
            tag: The tag of the scatter series.
            color: RGBA color tuple.
        """
        if not dpg.does_item_exist(tag):
            return

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_MarkerSize, 8.0, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots
                )

        dpg.bind_item_theme(tag, theme)

    def _on_plot_clicked(self, sender: int, app_data: tuple) -> None:
        """Handle plot click to select a group count.

        Args:
            sender: The plot widget.
            app_data: Click data containing (mouse_x, mouse_y).
        """
        if self._num_groups is None or len(self._num_groups) == 0:
            return

        if not dpg.does_item_exist(self._tags.plot):
            return

        # Get the mouse position in plot coordinates
        if not dpg.is_plot_queried(self._tags.plot):
            # Get mouse position in plot data coordinates
            mouse_x, mouse_y = app_data

            # Find the closest point on the curve
            closest_idx = self._find_closest_point(mouse_x)

            if closest_idx is not None and closest_idx != self._selected_index:
                self._selected_index = closest_idx
                self._update_selected_scatter()

                # Call callback if set
                if self._on_group_selected:
                    self._on_group_selected(int(self._num_groups[closest_idx]))

                logger.debug(
                    f"User selected {self._num_groups[closest_idx]} groups via click"
                )

    def _find_closest_point(self, x_value: float) -> int | None:
        """Find the closest data point to the given X value.

        Args:
            x_value: X coordinate to find closest point for.

        Returns:
            Index of closest point, or None if no data.
        """
        if self._num_groups is None or len(self._num_groups) == 0:
            return None

        # Round to nearest integer since group counts are integers
        rounded_x = round(x_value)

        # Find the closest match
        distances = np.abs(self._num_groups - rounded_x)
        return int(np.argmin(distances))

    def set_on_group_selected(
        self, callback: Callable[[int], None]
    ) -> None:
        """Set callback for when user selects a group count.

        Args:
            callback: Function called when user clicks to select a group count.
                Receives the number of groups as argument.
        """
        self._on_group_selected = callback

    def fit_view(self) -> None:
        """Fit the view to show all data (reset zoom/pan)."""
        self._fit_axes()

    def get_bic_at_groups(self, num_groups: int) -> float | None:
        """Get the BIC value for a specific number of groups.

        Args:
            num_groups: Number of groups to get BIC for.

        Returns:
            BIC value, or None if not found.
        """
        if self._num_groups is None or self._bic_values is None:
            return None

        matches = np.where(self._num_groups == num_groups)[0]
        if len(matches) > 0:
            return float(self._bic_values[matches[0]])
        return None
