"""Intensity trace plot widget.

Renders binned photon counts over time using DearPyGui's ImPlot.
Supports level overlay rendering for change point analysis results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.analysis.histograms import bin_photons, compute_intensity_cps
from full_sms.models.level import LevelData
from full_sms.ui.theme import COLORS, get_series_colors

logger = logging.getLogger(__name__)


# Extended color palette for levels (more colors than standard series)
# Semi-transparent for overlays (alpha = 80-120)
LEVEL_COLORS = [
    (100, 180, 255, 100),  # Blue
    (255, 150, 100, 100),  # Orange
    (100, 220, 150, 100),  # Green
    (255, 100, 150, 100),  # Pink
    (180, 150, 255, 100),  # Purple
    (255, 220, 100, 100),  # Yellow
    (100, 220, 220, 100),  # Cyan
    (220, 150, 180, 100),  # Rose
    (150, 200, 100, 100),  # Lime
    (200, 180, 150, 100),  # Tan
]

# Highlighted versions (higher alpha, brighter)
LEVEL_COLORS_HIGHLIGHTED = [
    (100, 180, 255, 180),  # Blue
    (255, 150, 100, 180),  # Orange
    (100, 220, 150, 180),  # Green
    (255, 100, 150, 180),  # Pink
    (180, 150, 255, 180),  # Purple
    (255, 220, 100, 180),  # Yellow
    (100, 220, 220, 180),  # Cyan
    (220, 150, 180, 180),  # Rose
    (150, 200, 100, 180),  # Lime
    (200, 180, 150, 180),  # Tan
]

# Dimmed versions (lower alpha for non-highlighted groups)
LEVEL_COLORS_DIMMED = [
    (100, 180, 255, 40),  # Blue
    (255, 150, 100, 40),  # Orange
    (100, 220, 150, 40),  # Green
    (255, 100, 150, 40),  # Pink
    (180, 150, 255, 40),  # Purple
    (255, 220, 100, 40),  # Yellow
    (100, 220, 220, 40),  # Cyan
    (220, 150, 180, 40),  # Rose
    (150, 200, 100, 40),  # Lime
    (200, 180, 150, 40),  # Tan
]

# Maximum number of individual level overlays before batching
MAX_INDIVIDUAL_LEVELS = 100


@dataclass
class IntensityPlotTags:
    """Tags for intensity plot elements."""

    container: str = "intensity_plot_container"
    plot: str = "intensity_plot"
    x_axis: str = "intensity_plot_x_axis"
    y_axis: str = "intensity_plot_y_axis"
    series: str = "intensity_plot_series"
    level_overlay_group: str = "intensity_plot_level_overlays"
    level_step_line: str = "intensity_plot_level_step_line"
    selected_level_band: str = "intensity_plot_selected_level_band"


INTENSITY_PLOT_TAGS = IntensityPlotTags()


class IntensityPlot:
    """Intensity trace plot widget.

    Displays binned photon counts over time as a line plot.
    Supports configurable bin size with zoom and pan enabled.
    Supports level overlay rendering for change point analysis results.
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

        # Level overlay state
        self._levels: list[LevelData] | None = None
        self._level_series_tags: list[str] = []
        self._levels_visible: bool = True
        self._color_by_group: bool = False
        self._highlighted_group_id: int | None = None  # Group ID to highlight

        # Generate unique tags
        self._tags = IntensityPlotTags(
            container=f"{tag_prefix}intensity_plot_container",
            plot=f"{tag_prefix}intensity_plot",
            x_axis=f"{tag_prefix}intensity_plot_x_axis",
            y_axis=f"{tag_prefix}intensity_plot_y_axis",
            series=f"{tag_prefix}intensity_plot_series",
            level_overlay_group=f"{tag_prefix}intensity_plot_level_overlays",
            level_step_line=f"{tag_prefix}intensity_plot_level_step_line",
            selected_level_band=f"{tag_prefix}intensity_plot_selected_level_band",
        )

        # Display options
        self._show_legend: bool = True

    @property
    def tags(self) -> IntensityPlotTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def bin_size_ms(self) -> float:
        """Get the current bin size in milliseconds."""
        return self._bin_size_ms

    def build(self, for_subplot: bool = False, show_legend: bool = True) -> None:
        """Build the plot UI structure.

        Args:
            for_subplot: If True, build the plot directly without a container group.
                Use this when the plot is being added inside a dpg.subplots() context.
            show_legend: If True, show the plot legend. Set to False for compact plots.
        """
        if self._is_built:
            return

        self._show_legend = show_legend

        if for_subplot:
            # Build plot directly (for use inside dpg.subplots)
            self._build_plot_content()
        else:
            # Build with container group (standalone mode)
            with dpg.group(parent=self._parent, tag=self._tags.container):
                self._build_plot_content()

        self._is_built = True
        logger.debug("Intensity plot built")

    def _build_plot_content(self) -> None:
        """Build the plot content (plot, axes, series)."""
        # Create the plot
        with dpg.plot(
            tag=self._tags.plot,
            label="Intensity Trace",
            width=-1,
            height=-1,
            anti_aliased=True,
        ):
            # Add legend - allows user to toggle series visibility by clicking
            if self._show_legend:
                dpg.add_plot_legend()

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
        """Auto-fit the axes to show all data (allows zoom/pan)."""
        if not dpg.does_item_exist(self._tags.x_axis):
            return

        # Auto-fit both axes - this allows user zoom/pan
        dpg.fit_axis_data(self._tags.x_axis)
        dpg.fit_axis_data(self._tags.y_axis)

    def clear(self) -> None:
        """Clear the plot data."""
        self._times = None
        self._counts = None
        self._update_series()
        self.clear_levels()
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

    def get_y_axis_limits(self) -> tuple[float, float] | None:
        """Get the current Y-axis visible limits.

        Returns:
            Tuple of (min, max) Y-axis limits, or None if not available.
        """
        if not dpg.does_item_exist(self._tags.y_axis):
            return None
        limits = dpg.get_axis_limits(self._tags.y_axis)
        return (limits[0], limits[1])

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._times is not None and len(self._times) > 0

    def fit_view(self) -> None:
        """Fit the view to show all data (reset zoom/pan)."""
        self._fit_axes()

    def set_y_axis_limits(self, y_min: float, y_max: float) -> None:
        """Set fixed Y-axis limits.

        Args:
            y_min: Minimum Y value.
            y_max: Maximum Y value.
        """
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.set_axis_limits(self._tags.y_axis, y_min, y_max)

    def fix_y_axis_to_data(self) -> None:
        """Set Y-axis limits from 0 to max data value (with padding)."""
        count_range = self.get_count_range()
        if count_range:
            y_max = count_range[1] * 1.1  # 10% padding
            self.set_y_axis_limits(0, y_max)

    # -------------------------------------------------------------------------
    # Level Overlay Methods
    # -------------------------------------------------------------------------

    @property
    def levels_visible(self) -> bool:
        """Whether level overlays are currently visible."""
        return self._levels_visible

    @property
    def color_by_group(self) -> bool:
        """Whether levels are colored by group ID (True) or level index (False)."""
        return self._color_by_group

    @property
    def has_levels(self) -> bool:
        """Whether any levels are set."""
        return self._levels is not None and len(self._levels) > 0

    @property
    def num_levels(self) -> int:
        """Number of levels currently set."""
        return len(self._levels) if self._levels else 0

    def set_levels(
        self,
        levels: Sequence[LevelData],
        color_by_group: bool = False,
    ) -> None:
        """Set the level data to overlay on the plot.

        Levels are rendered as shaded horizontal bands at their intensity,
        spanning their time range. Colors cycle through the level palette.

        Args:
            levels: Sequence of LevelData objects to display.
            color_by_group: If True, color by group_id; if False, by level index.
        """
        self._levels = list(levels)
        self._color_by_group = color_by_group

        # Remove existing level overlays
        self._remove_level_series()

        # Clear any selected level highlighting (from previous measurement)
        self.clear_selected_level()

        if not self._levels:
            logger.debug("No levels to display")
            return

        # Render the level overlays
        self._render_level_overlays()

        logger.debug(
            f"Set {len(self._levels)} level overlays (color_by_group={color_by_group})"
        )

    def clear_levels(self) -> None:
        """Clear all level overlays from the plot."""
        self._levels = None
        self._remove_level_series()
        logger.debug("Level overlays cleared")

    def set_levels_visible(self, visible: bool) -> None:
        """Show or hide level overlays.

        Args:
            visible: Whether to show level overlays.
        """
        self._levels_visible = visible

        # Toggle step line visibility
        if dpg.does_item_exist(self._tags.level_step_line):
            dpg.configure_item(self._tags.level_step_line, show=visible)

        # Also handle any old shade series (for backward compatibility)
        for tag in self._level_series_tags:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=visible)

        logger.debug(f"Level overlays visibility set to {visible}")

    def toggle_levels_visible(self) -> bool:
        """Toggle level overlay visibility.

        Returns:
            The new visibility state.
        """
        self.set_levels_visible(not self._levels_visible)
        return self._levels_visible

    def set_color_by_group(self, by_group: bool) -> None:
        """Change the coloring scheme for levels.

        Note: With the step line display, this currently has no visual effect.
        Group coloring will be implemented with horizontal band overlays.

        Args:
            by_group: If True, color by group_id; if False, by level index.
        """
        self._color_by_group = by_group
        logger.debug(f"Level coloring changed to by_group={by_group}")

    def _render_level_overlays(self) -> None:
        """Render all level overlays as a step line."""
        if not self._levels or not dpg.does_item_exist(self._tags.y_axis):
            return

        # Sort levels by start time
        sorted_levels = sorted(self._levels, key=lambda lv: lv.start_time_ns)

        # Build step line points with true vertical steps between levels
        x_points = []
        y_points = []
        prev_intensity = None

        for i, level in enumerate(sorted_levels):
            # Convert times from nanoseconds to milliseconds
            start_ms = level.start_time_ns / 1e6

            # For the end, use the next level's start time if available
            # This makes the step line visually continuous
            if i + 1 < len(sorted_levels):
                end_ms = sorted_levels[i + 1].start_time_ns / 1e6
            else:
                end_ms = level.end_time_ns / 1e6

            # Convert cps to counts per bin for display consistency
            if self._bin_size_ms > 0:
                intensity_per_bin = level.intensity_cps * (self._bin_size_ms / 1000.0)
            else:
                intensity_per_bin = level.intensity_cps / 100.0  # Fallback

            # Add vertical step at start (continue previous intensity to this x)
            if prev_intensity is not None:
                x_points.append(start_ms)
                y_points.append(prev_intensity)

            # Horizontal line at this level's intensity
            x_points.append(start_ms)
            y_points.append(intensity_per_bin)
            x_points.append(end_ms)
            y_points.append(intensity_per_bin)

            prev_intensity = intensity_per_bin

        # Create the step line series with label for legend
        if x_points:
            dpg.add_line_series(
                x_points,
                y_points,
                label="Levels",
                parent=self._tags.y_axis,
                tag=self._tags.level_step_line,
            )

            # Apply a theme for the level step line (a distinct color)
            self._apply_step_line_theme()

    def _apply_step_line_theme(self) -> None:
        """Apply a theme to the level step line."""
        if not dpg.does_item_exist(self._tags.level_step_line):
            return

        # Use a red/orange color for the step line to stand out
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (255, 100, 100, 255),  # Red-orange, fully opaque
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight,
                    2.0,
                    category=dpg.mvThemeCat_Plots,
                )

        dpg.bind_item_theme(self._tags.level_step_line, theme)

    def _remove_level_series(self) -> None:
        """Remove existing level step line series."""
        # Remove the step line if it exists
        if dpg.does_item_exist(self._tags.level_step_line):
            dpg.delete_item(self._tags.level_step_line)

        # Also clean up any old shade series (for backward compatibility)
        for tag in self._level_series_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

        self._level_series_tags.clear()

    def update_levels_for_bin_size(self) -> None:
        """Re-render level overlays after bin size change.

        Call this after changing bin size to update level display heights.
        """
        if self._levels:
            self._remove_level_series()
            self._render_level_overlays()

    def get_level_at_time(self, time_ms: float) -> LevelData | None:
        """Get the level at a specific time.

        Args:
            time_ms: Time in milliseconds.

        Returns:
            The LevelData at that time, or None if no level found.
        """
        if not self._levels:
            return None

        time_ns = int(time_ms * 1e6)
        for level in self._levels:
            if level.start_time_ns <= time_ns <= level.end_time_ns:
                return level

        return None

    @property
    def highlighted_group_id(self) -> int | None:
        """Get the currently highlighted group ID, or None if no highlighting."""
        return self._highlighted_group_id

    def set_highlighted_group(self, group_id: int | None) -> None:
        """Highlight a specific group, dimming all others.

        Note: With the step line display, this currently has no visual effect.
        Group highlighting will be implemented with horizontal band overlays.

        Args:
            group_id: The group ID to highlight, or None to clear highlighting.
        """
        self._highlighted_group_id = group_id
        logger.debug(f"Highlighted group set to {group_id}")

    def clear_highlighted_group(self) -> None:
        """Clear any group highlighting."""
        self.set_highlighted_group(None)

    # -------------------------------------------------------------------------
    # Selected Level Highlighting
    # -------------------------------------------------------------------------

    def set_selected_level(self, level: LevelData | None) -> None:
        """Highlight a selected level with a vertical band.

        Args:
            level: The level to highlight, or None to clear highlighting.
        """
        # Remove existing highlight band
        if dpg.does_item_exist(self._tags.selected_level_band):
            dpg.delete_item(self._tags.selected_level_band)

        if level is None or not dpg.does_item_exist(self._tags.y_axis):
            return

        # Convert level times to milliseconds
        start_ms = level.start_time_ns / 1e6

        # Find the end time - use next level's start if available (to match step line)
        end_ms = level.end_time_ns / 1e6
        if self._levels:
            sorted_levels = sorted(self._levels, key=lambda lv: lv.start_time_ns)
            for i, lv in enumerate(sorted_levels):
                if lv.start_time_ns == level.start_time_ns:
                    # Found the level, check if there's a next one
                    if i + 1 < len(sorted_levels):
                        end_ms = sorted_levels[i + 1].start_time_ns / 1e6
                    break

        # Create X coordinates for the band (left edge, right edge)
        x_coords = [start_ms, end_ms]
        # Y coordinates: Use 0 for bottom and a very large value for top
        # This ensures the band always extends to the full plot height
        # regardless of current axis limits or auto-fit timing
        y1_coords = [0, 0]
        y2_coords = [1e10, 1e10]  # Very large value to always reach plot top

        # Add shade series for the vertical band
        dpg.add_shade_series(
            x_coords,
            y1_coords,
            y2=y2_coords,
            parent=self._tags.y_axis,
            tag=self._tags.selected_level_band,
        )

        # Apply light green theme to the band
        self._apply_selected_level_theme()

    def _apply_selected_level_theme(self) -> None:
        """Apply a light green theme to the selected level band."""
        if not dpg.does_item_exist(self._tags.selected_level_band):
            return

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvShadeSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Fill,
                    (100, 200, 100, 60),  # Light green, semi-transparent
                    category=dpg.mvThemeCat_Plots,
                )

        dpg.bind_item_theme(self._tags.selected_level_band, theme)

    def clear_selected_level(self) -> None:
        """Clear any selected level highlighting."""
        if dpg.does_item_exist(self._tags.selected_level_band):
            dpg.delete_item(self._tags.selected_level_band)
