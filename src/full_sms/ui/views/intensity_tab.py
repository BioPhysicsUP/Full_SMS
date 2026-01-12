"""Intensity analysis tab view.

Provides the intensity trace visualization with:
- Intensity plot showing binned photon counts over time
- Bin size control slider
- Level overlay controls
- Basic plot controls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.models.level import LevelData
from full_sms.models.session import ConfidenceLevel
from full_sms.ui.plots.intensity_histogram import IntensityHistogram
from full_sms.ui.plots.intensity_plot import IntensityPlot

# Confidence level options for the combo box
CONFIDENCE_OPTIONS = ["69%", "90%", "95%", "99%"]
CONFIDENCE_MAP = {
    "69%": ConfidenceLevel.CONF_69,
    "90%": ConfidenceLevel.CONF_90,
    "95%": ConfidenceLevel.CONF_95,
    "99%": ConfidenceLevel.CONF_99,
}
DEFAULT_CONFIDENCE = "95%"

logger = logging.getLogger(__name__)


@dataclass
class IntensityTabTags:
    """Tags for intensity tab elements."""

    container: str = "intensity_tab_view_container"
    controls_group: str = "intensity_tab_controls"
    bin_size_slider: str = "intensity_tab_bin_size"
    bin_size_label: str = "intensity_tab_bin_size_label"
    fit_view_button: str = "intensity_tab_fit_view"
    show_levels_checkbox: str = "intensity_tab_show_levels"
    color_by_group_checkbox: str = "intensity_tab_color_by_group"
    show_histogram_checkbox: str = "intensity_tab_show_histogram"
    level_info_text: str = "intensity_tab_level_info"
    plot_container: str = "intensity_tab_plot_container"
    plot_area: str = "intensity_tab_plot_area"
    subplots: str = "intensity_tab_subplots"
    histogram_container: str = "intensity_tab_histogram_container"
    info_text: str = "intensity_tab_info"
    no_data_text: str = "intensity_tab_no_data"
    # Resolve controls
    resolve_group: str = "intensity_tab_resolve_group"
    confidence_combo: str = "intensity_tab_confidence"
    resolve_current_btn: str = "intensity_tab_resolve_current"
    resolve_selected_btn: str = "intensity_tab_resolve_selected"
    resolve_all_btn: str = "intensity_tab_resolve_all"


INTENSITY_TAB_TAGS = IntensityTabTags()


# Default bin size options in milliseconds
DEFAULT_BIN_SIZE_MS = 10.0
MIN_BIN_SIZE_MS = 0.1
MAX_BIN_SIZE_MS = 1000.0


class IntensityTab:
    """Intensity analysis tab view.

    Contains the intensity trace plot and controls for bin size adjustment.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the intensity tab.

        Args:
            parent: The parent container to build the tab in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._abstimes: NDArray[np.uint64] | None = None
        self._bin_size_ms: float = DEFAULT_BIN_SIZE_MS
        self._levels: list[LevelData] | None = None

        # Histogram visibility state
        self._histogram_visible: bool = True

        # Callbacks
        self._on_bin_size_changed: Callable[[float], None] | None = None
        self._on_resolve: Callable[[str, ConfidenceLevel], None] | None = None

        # UI components
        self._intensity_plot: IntensityPlot | None = None
        self._intensity_histogram: IntensityHistogram | None = None

        # Generate unique tags
        self._tags = IntensityTabTags(
            container=f"{tag_prefix}intensity_tab_view_container",
            controls_group=f"{tag_prefix}intensity_tab_controls",
            bin_size_slider=f"{tag_prefix}intensity_tab_bin_size",
            bin_size_label=f"{tag_prefix}intensity_tab_bin_size_label",
            fit_view_button=f"{tag_prefix}intensity_tab_fit_view",
            show_levels_checkbox=f"{tag_prefix}intensity_tab_show_levels",
            color_by_group_checkbox=f"{tag_prefix}intensity_tab_color_by_group",
            show_histogram_checkbox=f"{tag_prefix}intensity_tab_show_histogram",
            level_info_text=f"{tag_prefix}intensity_tab_level_info",
            plot_container=f"{tag_prefix}intensity_tab_plot_container",
            plot_area=f"{tag_prefix}intensity_tab_plot_area",
            subplots=f"{tag_prefix}intensity_tab_subplots",
            histogram_container=f"{tag_prefix}intensity_tab_histogram_container",
            info_text=f"{tag_prefix}intensity_tab_info",
            no_data_text=f"{tag_prefix}intensity_tab_no_data",
            resolve_group=f"{tag_prefix}intensity_tab_resolve_group",
            confidence_combo=f"{tag_prefix}intensity_tab_confidence",
            resolve_current_btn=f"{tag_prefix}intensity_tab_resolve_current",
            resolve_selected_btn=f"{tag_prefix}intensity_tab_resolve_selected",
            resolve_all_btn=f"{tag_prefix}intensity_tab_resolve_all",
        )

    @property
    def tags(self) -> IntensityTabTags:
        """Get the tags for this tab instance."""
        return self._tags

    @property
    def bin_size_ms(self) -> float:
        """Get the current bin size in milliseconds."""
        return self._bin_size_ms

    @property
    def intensity_plot(self) -> IntensityPlot | None:
        """Get the intensity plot widget."""
        return self._intensity_plot

    @property
    def intensity_histogram(self) -> IntensityHistogram | None:
        """Get the intensity histogram widget."""
        return self._intensity_histogram

    @property
    def histogram_visible(self) -> bool:
        """Whether the histogram sidebar is visible."""
        return self._histogram_visible

    def build(self) -> None:
        """Build the tab UI structure."""
        if self._is_built:
            return

        # Main container
        with dpg.group(parent=self._parent, tag=self._tags.container):
            # Controls bar at top
            self._build_controls()

            # Separator
            dpg.add_separator()

            # Plot area (takes remaining space)
            with dpg.child_window(
                tag=self._tags.plot_container,
                border=False,
                autosize_x=True,
                autosize_y=True,
            ):
                # No data placeholder (shown when no data loaded)
                dpg.add_text(
                    "Load an HDF5 file and select a particle to view intensity trace.",
                    tag=self._tags.no_data_text,
                    color=(128, 128, 128),
                )

                # Subplots layout for plot + histogram with linked Y-axes
                # Using subplots with link_all_y=True ensures the histogram Y-axis
                # stays in sync with the intensity plot when zooming/panning
                with dpg.group(
                    tag=self._tags.plot_area,
                    show=False,  # Hidden until data loaded
                ):
                    with dpg.subplots(
                        rows=1,
                        columns=2,
                        tag=self._tags.subplots,
                        width=-1,
                        height=-1,
                        link_all_y=True,  # Link Y-axes between plots
                        column_ratios=[5.0, 1.0],  # Intensity plot gets 5x the width
                        no_title=True,
                    ):
                        # Main intensity plot (left, takes most space)
                        self._intensity_plot = IntensityPlot(
                            parent=self._tags.subplots,
                            tag_prefix=f"{self._tag_prefix}main_",
                        )
                        self._intensity_plot.build(for_subplot=True)

                        # Histogram sidebar (right, narrow)
                        self._intensity_histogram = IntensityHistogram(
                            parent=self._tags.subplots,
                            tag_prefix=f"{self._tag_prefix}hist_",
                            width=-1,  # Let subplot control width
                        )
                        self._intensity_histogram.build(for_subplot=True)

        self._is_built = True
        logger.debug("Intensity tab built")

    def _build_controls(self) -> None:
        """Build the controls bar at the top of the tab."""
        # First row: Bin size, display options, and fit view
        with dpg.group(horizontal=True, tag=self._tags.controls_group):
            # Bin size control
            dpg.add_text("Bin Size:")
            dpg.add_slider_float(
                tag=self._tags.bin_size_slider,
                default_value=DEFAULT_BIN_SIZE_MS,
                min_value=MIN_BIN_SIZE_MS,
                max_value=MAX_BIN_SIZE_MS,
                width=200,
                format="%.1f ms",
                callback=self._on_bin_size_slider_changed,
                clamped=True,
            )

            # Current value label
            dpg.add_text(
                f"{DEFAULT_BIN_SIZE_MS:.1f} ms",
                tag=self._tags.bin_size_label,
                color=(180, 180, 180),
            )

            # Spacer
            dpg.add_spacer(width=20)

            # Histogram visibility toggle
            dpg.add_checkbox(
                label="Show Histogram",
                tag=self._tags.show_histogram_checkbox,
                default_value=True,
                callback=self._on_show_histogram_changed,
                enabled=False,
            )

            dpg.add_spacer(width=10)

            # Level overlay controls
            dpg.add_checkbox(
                label="Show Levels",
                tag=self._tags.show_levels_checkbox,
                default_value=True,
                callback=self._on_show_levels_changed,
                enabled=False,
            )

            dpg.add_spacer(width=10)

            dpg.add_checkbox(
                label="Colour by Group",
                tag=self._tags.color_by_group_checkbox,
                default_value=False,
                callback=self._on_color_by_group_changed,
                enabled=False,
            )

            dpg.add_spacer(width=20)

            # Fit view button
            dpg.add_button(
                label="Fit View",
                tag=self._tags.fit_view_button,
                callback=self._on_fit_view_clicked,
                enabled=False,
            )

        # Second row: Resolve controls
        with dpg.group(horizontal=True, tag=self._tags.resolve_group):
            dpg.add_text("Confidence:")
            dpg.add_combo(
                items=CONFIDENCE_OPTIONS,
                default_value=DEFAULT_CONFIDENCE,
                tag=self._tags.confidence_combo,
                width=80,
            )

            dpg.add_spacer(width=15)

            dpg.add_button(
                label="Resolve Current",
                tag=self._tags.resolve_current_btn,
                callback=self._on_resolve_current_clicked,
                enabled=False,
            )

            dpg.add_spacer(width=5)

            dpg.add_button(
                label="Resolve Selected",
                tag=self._tags.resolve_selected_btn,
                callback=self._on_resolve_selected_clicked,
                enabled=False,
            )

            dpg.add_spacer(width=5)

            dpg.add_button(
                label="Resolve All",
                tag=self._tags.resolve_all_btn,
                callback=self._on_resolve_all_clicked,
                enabled=False,
            )

        # Third row: Info text
        with dpg.group(horizontal=True):
            # Level info text
            dpg.add_text(
                "",
                tag=self._tags.level_info_text,
                color=(150, 200, 150),
            )

            dpg.add_spacer(width=20)

            # Info text (shows photon count, time range)
            dpg.add_text(
                "",
                tag=self._tags.info_text,
                color=(128, 128, 128),
            )

    def _on_bin_size_slider_changed(
        self, sender: int, app_data: float
    ) -> None:
        """Handle bin size slider changes.

        Args:
            sender: The slider widget.
            app_data: The new bin size value in milliseconds.
        """
        self._bin_size_ms = app_data

        # Update label
        if dpg.does_item_exist(self._tags.bin_size_label):
            dpg.set_value(self._tags.bin_size_label, f"{app_data:.1f} ms")

        # Rebin data if we have it
        if self._abstimes is not None and self._intensity_plot is not None:
            self._intensity_plot.update_bin_size(self._abstimes, app_data)
            # Also update level overlays for the new bin size
            self._intensity_plot.update_levels_for_bin_size()
            # Update the histogram with the new binned counts
            self._update_histogram()

        # Call callback if set
        if self._on_bin_size_changed:
            self._on_bin_size_changed(app_data)

        logger.debug(f"Bin size changed to {app_data:.1f} ms")

    def _on_fit_view_clicked(self) -> None:
        """Handle fit view button click."""
        if self._intensity_plot:
            self._intensity_plot.fit_view()

    def _on_show_levels_changed(self, sender: int, app_data: bool) -> None:
        """Handle show levels checkbox changes.

        Args:
            sender: The checkbox widget.
            app_data: Whether levels should be visible.
        """
        if self._intensity_plot:
            self._intensity_plot.set_levels_visible(app_data)
        logger.debug(f"Show levels changed to {app_data}")

    def _on_color_by_group_changed(self, sender: int, app_data: bool) -> None:
        """Handle color by group checkbox changes.

        Args:
            sender: The checkbox widget.
            app_data: Whether to color by group.
        """
        if self._intensity_plot:
            self._intensity_plot.set_color_by_group(app_data)
        logger.debug(f"Color by group changed to {app_data}")

    def _on_show_histogram_changed(self, sender: int, app_data: bool) -> None:
        """Handle show histogram checkbox changes.

        Args:
            sender: The checkbox widget.
            app_data: Whether to show the histogram.
        """
        self._histogram_visible = app_data

        # Hide the histogram plot directly within the subplot
        if self._intensity_histogram is not None:
            hist_plot_tag = self._intensity_histogram.tags.plot
            if dpg.does_item_exist(hist_plot_tag):
                dpg.configure_item(hist_plot_tag, show=app_data)

        logger.debug(f"Show histogram changed to {app_data}")

    # -------------------------------------------------------------------------
    # Resolve Control Callbacks
    # -------------------------------------------------------------------------

    def _get_selected_confidence(self) -> ConfidenceLevel:
        """Get the currently selected confidence level."""
        if dpg.does_item_exist(self._tags.confidence_combo):
            value = dpg.get_value(self._tags.confidence_combo)
            return CONFIDENCE_MAP.get(value, ConfidenceLevel.CONF_95)
        return ConfidenceLevel.CONF_95

    def _on_resolve_current_clicked(self) -> None:
        """Handle Resolve Current button click."""
        confidence = self._get_selected_confidence()
        logger.info(f"Resolve Current clicked with confidence {confidence.value}")
        if self._on_resolve:
            self._on_resolve("current", confidence)

    def _on_resolve_selected_clicked(self) -> None:
        """Handle Resolve Selected button click."""
        confidence = self._get_selected_confidence()
        logger.info(f"Resolve Selected clicked with confidence {confidence.value}")
        if self._on_resolve:
            self._on_resolve("selected", confidence)

    def _on_resolve_all_clicked(self) -> None:
        """Handle Resolve All button click."""
        confidence = self._get_selected_confidence()
        logger.info(f"Resolve All clicked with confidence {confidence.value}")
        if self._on_resolve:
            self._on_resolve("all", confidence)

    def set_data(self, abstimes: NDArray[np.uint64]) -> None:
        """Set the photon arrival time data.

        Args:
            abstimes: Absolute photon arrival times in nanoseconds.
        """
        self._abstimes = abstimes

        if len(abstimes) == 0:
            self.clear()
            return

        # Update plot
        if self._intensity_plot:
            self._intensity_plot.set_data(abstimes, self._bin_size_ms)

        # Update histogram with the binned counts from the plot
        self._update_histogram()

        # Show plot, hide placeholder
        self._show_plot(True)

        # Enable controls
        if dpg.does_item_exist(self._tags.fit_view_button):
            dpg.configure_item(self._tags.fit_view_button, enabled=True)

        # Enable histogram checkbox
        if dpg.does_item_exist(self._tags.show_histogram_checkbox):
            dpg.configure_item(self._tags.show_histogram_checkbox, enabled=True)

        # Enable resolve current button (data is loaded)
        if dpg.does_item_exist(self._tags.resolve_current_btn):
            dpg.configure_item(self._tags.resolve_current_btn, enabled=True)

        # Update info text
        self._update_info_text()

        logger.debug(f"Intensity tab data set: {len(abstimes)} photons")

    def clear(self) -> None:
        """Clear the tab data."""
        self._abstimes = None
        self._levels = None

        if self._intensity_plot:
            self._intensity_plot.clear()

        if self._intensity_histogram:
            self._intensity_histogram.clear()

        # Hide plot, show placeholder
        self._show_plot(False)

        # Disable controls
        if dpg.does_item_exist(self._tags.fit_view_button):
            dpg.configure_item(self._tags.fit_view_button, enabled=False)

        # Disable level controls
        if dpg.does_item_exist(self._tags.show_levels_checkbox):
            dpg.configure_item(self._tags.show_levels_checkbox, enabled=False)
        if dpg.does_item_exist(self._tags.color_by_group_checkbox):
            dpg.configure_item(self._tags.color_by_group_checkbox, enabled=False)

        # Disable histogram checkbox
        if dpg.does_item_exist(self._tags.show_histogram_checkbox):
            dpg.configure_item(self._tags.show_histogram_checkbox, enabled=False)

        # Disable resolve current button
        if dpg.does_item_exist(self._tags.resolve_current_btn):
            dpg.configure_item(self._tags.resolve_current_btn, enabled=False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        # Clear level info text
        if dpg.does_item_exist(self._tags.level_info_text):
            dpg.set_value(self._tags.level_info_text, "")

        logger.debug("Intensity tab cleared")

    def _show_plot(self, show: bool) -> None:
        """Show or hide the plot area.

        Args:
            show: Whether to show the plot area (True) or placeholder (False).
        """
        # Show/hide the plot area (contains the subplots with plot and histogram)
        if dpg.does_item_exist(self._tags.plot_area):
            dpg.configure_item(self._tags.plot_area, show=show)

        # Show/hide the no data placeholder
        if dpg.does_item_exist(self._tags.no_data_text):
            dpg.configure_item(self._tags.no_data_text, show=not show)

        # Respect histogram visibility preference when showing
        if show and self._intensity_histogram is not None:
            hist_plot_tag = self._intensity_histogram.tags.plot
            if dpg.does_item_exist(hist_plot_tag):
                dpg.configure_item(hist_plot_tag, show=self._histogram_visible)

    def _update_info_text(self) -> None:
        """Update the info text with current data stats."""
        if not dpg.does_item_exist(self._tags.info_text):
            return

        if self._abstimes is None or len(self._abstimes) == 0:
            dpg.set_value(self._tags.info_text, "")
            return

        # Calculate stats
        num_photons = len(self._abstimes)
        time_range = self._intensity_plot.get_time_range() if self._intensity_plot else None

        if time_range:
            duration_s = (time_range[1] - time_range[0]) / 1000.0
            avg_rate = num_photons / duration_s if duration_s > 0 else 0
            info = f"{num_photons:,} photons | {duration_s:.1f} s | {avg_rate:.0f} cps avg"
        else:
            info = f"{num_photons:,} photons"

        dpg.set_value(self._tags.info_text, info)

    def _update_histogram(self) -> None:
        """Update the histogram with the current binned counts from the plot."""
        if self._intensity_histogram is None:
            return

        # Get the counts from the intensity plot
        if (
            self._intensity_plot is not None
            and self._intensity_plot._counts is not None
        ):
            self._intensity_histogram.set_data(self._intensity_plot._counts)
        else:
            self._intensity_histogram.clear()

    def set_on_bin_size_changed(self, callback: Callable[[float], None]) -> None:
        """Set callback for bin size changes.

        Args:
            callback: Function called when bin size changes, receives new bin size.
        """
        self._on_bin_size_changed = callback

    def set_on_resolve(
        self, callback: Callable[[str, ConfidenceLevel], None]
    ) -> None:
        """Set callback for resolve button clicks.

        Args:
            callback: Function called when resolve is triggered.
                First arg is mode ("current", "selected", or "all").
                Second arg is the confidence level.
        """
        self._on_resolve = callback

    def set_resolve_buttons_state(
        self,
        has_current: bool = False,
        has_selected: bool = False,
        has_any: bool = False,
    ) -> None:
        """Enable/disable resolve buttons based on application state.

        Args:
            has_current: Whether there is a currently viewed particle.
            has_selected: Whether there are batch-selected particles.
            has_any: Whether there are any particles loaded.
        """
        if dpg.does_item_exist(self._tags.resolve_current_btn):
            dpg.configure_item(self._tags.resolve_current_btn, enabled=has_current)
        if dpg.does_item_exist(self._tags.resolve_selected_btn):
            dpg.configure_item(self._tags.resolve_selected_btn, enabled=has_selected)
        if dpg.does_item_exist(self._tags.resolve_all_btn):
            dpg.configure_item(self._tags.resolve_all_btn, enabled=has_any)

    def set_resolving(self, is_resolving: bool) -> None:
        """Set the resolving state, disabling buttons while analysis is running.

        Args:
            is_resolving: Whether change point analysis is currently running.
        """
        # Disable all resolve buttons during analysis
        for tag in [
            self._tags.resolve_current_btn,
            self._tags.resolve_selected_btn,
            self._tags.resolve_all_btn,
        ]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=not is_resolving)

    def set_bin_size(self, bin_size_ms: float) -> None:
        """Programmatically set the bin size.

        Args:
            bin_size_ms: The new bin size in milliseconds.
        """
        # Clamp to valid range
        bin_size_ms = max(MIN_BIN_SIZE_MS, min(MAX_BIN_SIZE_MS, bin_size_ms))
        self._bin_size_ms = bin_size_ms

        # Update slider
        if dpg.does_item_exist(self._tags.bin_size_slider):
            dpg.set_value(self._tags.bin_size_slider, bin_size_ms)

        # Update label
        if dpg.does_item_exist(self._tags.bin_size_label):
            dpg.set_value(self._tags.bin_size_label, f"{bin_size_ms:.1f} ms")

        # Rebin data if we have it
        if self._abstimes is not None and self._intensity_plot is not None:
            self._intensity_plot.update_bin_size(self._abstimes, bin_size_ms)

    @property
    def has_data(self) -> bool:
        """Whether the tab has data loaded."""
        return self._abstimes is not None and len(self._abstimes) > 0

    # -------------------------------------------------------------------------
    # Level Overlay Methods
    # -------------------------------------------------------------------------

    @property
    def has_levels(self) -> bool:
        """Whether the tab has levels loaded."""
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
        """Set the level data to overlay on the intensity plot.

        Args:
            levels: Sequence of LevelData objects to display.
            color_by_group: If True, color by group_id; if False, by level index.
        """
        self._levels = list(levels)

        if self._intensity_plot:
            self._intensity_plot.set_levels(levels, color_by_group)

        # Enable level controls if we have levels
        has_levels = len(levels) > 0
        if dpg.does_item_exist(self._tags.show_levels_checkbox):
            dpg.configure_item(self._tags.show_levels_checkbox, enabled=has_levels)
        if dpg.does_item_exist(self._tags.color_by_group_checkbox):
            dpg.configure_item(self._tags.color_by_group_checkbox, enabled=has_levels)

        # Update checkbox state
        if dpg.does_item_exist(self._tags.color_by_group_checkbox):
            dpg.set_value(self._tags.color_by_group_checkbox, color_by_group)

        # Update level info text
        self._update_level_info_text()

        logger.debug(f"Intensity tab levels set: {len(levels)} levels")

    def clear_levels(self) -> None:
        """Clear all level overlays."""
        self._levels = None

        if self._intensity_plot:
            self._intensity_plot.clear_levels()

        # Disable level controls
        if dpg.does_item_exist(self._tags.show_levels_checkbox):
            dpg.configure_item(self._tags.show_levels_checkbox, enabled=False)
        if dpg.does_item_exist(self._tags.color_by_group_checkbox):
            dpg.configure_item(self._tags.color_by_group_checkbox, enabled=False)

        # Clear level info text
        if dpg.does_item_exist(self._tags.level_info_text):
            dpg.set_value(self._tags.level_info_text, "")

        logger.debug("Intensity tab levels cleared")

    def _update_level_info_text(self) -> None:
        """Update the level info text."""
        if not dpg.does_item_exist(self._tags.level_info_text):
            return

        if not self._levels or len(self._levels) == 0:
            dpg.set_value(self._tags.level_info_text, "")
            return

        num_levels = len(self._levels)

        # Count unique groups if any levels have group assignments
        groups = {level.group_id for level in self._levels if level.group_id is not None}
        if groups:
            dpg.set_value(
                self._tags.level_info_text,
                f"{num_levels} levels | {len(groups)} groups",
            )
        else:
            dpg.set_value(self._tags.level_info_text, f"{num_levels} levels")

    def set_levels_visible(self, visible: bool) -> None:
        """Show or hide level overlays.

        Args:
            visible: Whether to show level overlays.
        """
        if self._intensity_plot:
            self._intensity_plot.set_levels_visible(visible)

        # Update checkbox
        if dpg.does_item_exist(self._tags.show_levels_checkbox):
            dpg.set_value(self._tags.show_levels_checkbox, visible)

    def set_color_by_group(self, by_group: bool) -> None:
        """Change the coloring scheme for levels.

        Args:
            by_group: If True, color by group_id; if False, by level index.
        """
        if self._intensity_plot:
            self._intensity_plot.set_color_by_group(by_group)

        # Update checkbox
        if dpg.does_item_exist(self._tags.color_by_group_checkbox):
            dpg.set_value(self._tags.color_by_group_checkbox, by_group)

    # -------------------------------------------------------------------------
    # Histogram Methods
    # -------------------------------------------------------------------------

    def set_histogram_visible(self, visible: bool) -> None:
        """Show or hide the histogram sidebar.

        Args:
            visible: Whether to show the histogram.
        """
        self._histogram_visible = visible

        # Hide the histogram plot directly within the subplot
        if self._intensity_histogram is not None:
            hist_plot_tag = self._intensity_histogram.tags.plot
            if dpg.does_item_exist(hist_plot_tag):
                dpg.configure_item(hist_plot_tag, show=visible)

        # Update checkbox
        if dpg.does_item_exist(self._tags.show_histogram_checkbox):
            dpg.set_value(self._tags.show_histogram_checkbox, visible)

    # -------------------------------------------------------------------------
    # Group Highlighting Methods
    # -------------------------------------------------------------------------

    def set_highlighted_group(self, group_id: int | None) -> None:
        """Highlight a specific group on the intensity plot.

        When a group is highlighted:
        - The highlighted group's levels are shown with brighter colors
        - All other groups' levels are dimmed

        This requires color_by_group to be enabled to have any visual effect.

        Args:
            group_id: The group ID to highlight (0-indexed), or None to clear.
        """
        if self._intensity_plot:
            self._intensity_plot.set_highlighted_group(group_id)

        # If highlighting is enabled, ensure color_by_group is also enabled
        if group_id is not None:
            self.set_color_by_group(True)

    def clear_highlighted_group(self) -> None:
        """Clear any group highlighting."""
        if self._intensity_plot:
            self._intensity_plot.clear_highlighted_group()

    @property
    def highlighted_group_id(self) -> int | None:
        """Get the currently highlighted group ID."""
        if self._intensity_plot:
            return self._intensity_plot.highlighted_group_id
        return None
