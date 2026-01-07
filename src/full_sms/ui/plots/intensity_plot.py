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

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._times is not None and len(self._times) > 0

    def fit_view(self) -> None:
        """Fit the view to show all data (reset zoom/pan)."""
        self._fit_axes()

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

        Args:
            by_group: If True, color by group_id; if False, by level index.
        """
        if self._color_by_group == by_group:
            return

        self._color_by_group = by_group

        # Re-render with new colors if we have levels
        if self._levels:
            self._remove_level_series()
            self._render_level_overlays()

        logger.debug(f"Level coloring changed to by_group={by_group}")

    def _render_level_overlays(self) -> None:
        """Render all level overlays as shade series."""
        if not self._levels or not dpg.does_item_exist(self._tags.y_axis):
            return

        # For performance with many levels, we create individual shade series
        # but could batch them if needed (DearPyGui handles this reasonably well)
        for i, level in enumerate(self._levels):
            self._add_level_shade(level, i)

    def _add_level_shade(self, level: LevelData, index: int) -> None:
        """Add a shade series for a single level.

        Args:
            level: The level data.
            index: The index of this level (for coloring).
        """
        # Determine color index
        if self._color_by_group and level.group_id is not None:
            color_idx = level.group_id % len(LEVEL_COLORS)
        else:
            color_idx = index % len(LEVEL_COLORS)

        # Determine color palette based on highlighting
        if self._highlighted_group_id is not None and self._color_by_group:
            # Highlighting is active - use highlighted or dimmed colors
            if level.group_id == self._highlighted_group_id:
                color = LEVEL_COLORS_HIGHLIGHTED[color_idx]
            else:
                color = LEVEL_COLORS_DIMMED[color_idx]
        else:
            # No highlighting - use normal colors
            color = LEVEL_COLORS[color_idx]

        # Convert times from nanoseconds to milliseconds
        start_ms = level.start_time_ns / 1e6
        end_ms = level.end_time_ns / 1e6

        # Create x values (start and end of the level)
        x_vals = [start_ms, end_ms]

        # Create y values (the intensity level as a horizontal band)
        # We use the level's intensity in counts/sec, converted to counts/bin
        # for consistency with the binned data display
        if self._bin_size_ms > 0:
            # Convert cps to counts per bin for display consistency
            intensity_per_bin = level.intensity_cps * (self._bin_size_ms / 1000.0)
        else:
            intensity_per_bin = level.intensity_cps / 100.0  # Fallback

        # Shade from 0 to the intensity level
        y1_vals = [0.0, 0.0]
        y2_vals = [intensity_per_bin, intensity_per_bin]

        # Generate unique tag for this level
        tag = f"{self._tag_prefix}level_shade_{index}"
        self._level_series_tags.append(tag)

        # Add the shade series
        dpg.add_shade_series(
            x_vals,
            y1_vals,
            y2=y2_vals,
            parent=self._tags.y_axis,
            tag=tag,
            show=self._levels_visible,
        )

        # Apply color theme to the shade series
        self._apply_shade_color(tag, color)

    def _apply_shade_color(self, tag: str, color: tuple[int, int, int, int]) -> None:
        """Apply a color to a shade series via theme.

        Args:
            tag: The tag of the shade series.
            color: RGBA color tuple.
        """
        # Create a theme for this specific shade series
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvShadeSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Fill, color, category=dpg.mvThemeCat_Plots
                )

        # Apply the theme
        dpg.bind_item_theme(tag, theme)

    def _remove_level_series(self) -> None:
        """Remove all existing level shade series."""
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

        When a group is highlighted:
        - The highlighted group's levels are shown with brighter colors
        - All other groups' levels are dimmed
        - Only works when color_by_group is True

        Args:
            group_id: The group ID to highlight, or None to clear highlighting.
        """
        if self._highlighted_group_id == group_id:
            return

        self._highlighted_group_id = group_id

        # Re-render levels with new highlighting
        if self._levels:
            self._remove_level_series()
            self._render_level_overlays()

        logger.debug(f"Highlighted group set to {group_id}")

    def clear_highlighted_group(self) -> None:
        """Clear any group highlighting."""
        self.set_highlighted_group(None)
