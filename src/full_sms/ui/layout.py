"""Main application layout for Full SMS.

Provides the two-column layout with particle tree sidebar, tab navigation,
and status bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import dearpygui.dearpygui as dpg

from full_sms.analysis.correlation import CorrelationResult
from full_sms.models.group import ClusteringResult
from full_sms.models.particle import ParticleData, RasterScanData, SpectraData
from full_sms.models.session import ActiveTab, ChannelSelection, ProcessingState
from full_sms.ui.views.correlation_tab import CorrelationTab
from full_sms.ui.views.grouping_tab import GroupingTab
from full_sms.ui.views.intensity_tab import IntensityTab
from full_sms.ui.views.lifetime_tab import LifetimeTab
from full_sms.ui.views.raster_tab import RasterTab
from full_sms.ui.views.spectra_tab import SpectraTab
from full_sms.ui.widgets.particle_tree import ParticleTree
from full_sms.ui.widgets.status_bar import StatusBar


# Layout configuration
SIDEBAR_WIDTH = 280
STATUS_BAR_HEIGHT = 28


@dataclass
class LayoutTags:
    """Tags for layout elements."""

    # Main containers
    main_container: str = "main_layout_container"
    sidebar: str = "sidebar_container"
    content_area: str = "content_area_container"
    tab_bar: str = "main_tab_bar"
    status_bar: str = "status_bar_container"

    # Sidebar elements
    particle_tree_placeholder: str = "particle_tree_placeholder"
    sidebar_controls: str = "sidebar_controls"

    # Tab contents (each tab has its own container)
    tab_intensity: str = "tab_content_intensity"
    tab_lifetime: str = "tab_content_lifetime"
    tab_grouping: str = "tab_content_grouping"
    tab_spectra: str = "tab_content_spectra"
    tab_raster: str = "tab_content_raster"
    tab_correlation: str = "tab_content_correlation"
    tab_export: str = "tab_content_export"

    # Status bar elements
    status_text: str = "status_bar_text"
    progress_bar: str = "status_progress_bar"
    file_info: str = "status_file_info"


LAYOUT_TAGS = LayoutTags()


# Map ActiveTab enum to tab tags
TAB_TAG_MAP: dict[ActiveTab, str] = {
    ActiveTab.INTENSITY: LAYOUT_TAGS.tab_intensity,
    ActiveTab.LIFETIME: LAYOUT_TAGS.tab_lifetime,
    ActiveTab.GROUPING: LAYOUT_TAGS.tab_grouping,
    ActiveTab.SPECTRA: LAYOUT_TAGS.tab_spectra,
    ActiveTab.RASTER: LAYOUT_TAGS.tab_raster,
    ActiveTab.CORRELATION: LAYOUT_TAGS.tab_correlation,
    ActiveTab.EXPORT: LAYOUT_TAGS.tab_export,
}


class MainLayout:
    """Main application layout manager.

    Creates and manages the two-column layout with:
    - Left sidebar for particle tree and controls
    - Right content area with tab navigation
    - Bottom status bar with progress indicator
    """

    def __init__(self, parent: int | str) -> None:
        """Initialize the layout.

        Args:
            parent: The parent window/container to build the layout in.
        """
        self._parent = parent
        self._active_tab: ActiveTab = ActiveTab.INTENSITY
        self._on_tab_change: Callable[[ActiveTab], None] | None = None
        self._on_selection_change: Callable[[ChannelSelection | None], None] | None = None
        self._on_batch_change: Callable[[list[ChannelSelection]], None] | None = None
        self._particle_tree: ParticleTree | None = None
        self._status_bar: StatusBar | None = None
        self._intensity_tab: IntensityTab | None = None
        self._lifetime_tab: LifetimeTab | None = None
        self._grouping_tab: GroupingTab | None = None
        self._spectra_tab: SpectraTab | None = None
        self._raster_tab: RasterTab | None = None
        self._correlation_tab: CorrelationTab | None = None
        self._is_built = False

    def build(self) -> None:
        """Build the complete layout structure."""
        if self._is_built:
            return

        with dpg.group(parent=self._parent, horizontal=True, tag=LAYOUT_TAGS.main_container):
            # Left sidebar
            self._build_sidebar()

            # Right content area with tabs
            self._build_content_area()

        # Status bar at the bottom (outside the horizontal group)
        self._build_status_bar()

        self._is_built = True

    def _build_sidebar(self) -> None:
        """Build the left sidebar with particle tree."""
        with dpg.child_window(
            tag=LAYOUT_TAGS.sidebar,
            width=SIDEBAR_WIDTH,
            border=True,
            autosize_y=True,
        ):
            # Header
            dpg.add_text("Particles", color=(180, 180, 180))
            dpg.add_separator()

            # Particle tree container
            with dpg.child_window(
                tag=LAYOUT_TAGS.particle_tree_placeholder,
                autosize_x=True,
                height=-60,  # Leave room for controls at bottom
                border=False,
            ):
                # Build the particle tree widget
                self._particle_tree = ParticleTree(
                    parent=LAYOUT_TAGS.particle_tree_placeholder,
                    on_selection_changed=self._on_particle_selection_changed,
                    on_batch_changed=self._on_particle_batch_changed,
                )
                self._particle_tree.build()

            # Bottom controls
            dpg.add_separator()
            with dpg.group(tag=LAYOUT_TAGS.sidebar_controls):
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Select All",
                        width=85,
                        enabled=False,
                        tag="sidebar_select_all",
                        callback=self._on_select_all_clicked,
                    )
                    dpg.add_button(
                        label="Clear",
                        width=85,
                        enabled=False,
                        tag="sidebar_clear_selection",
                        callback=self._on_clear_selection_clicked,
                    )
                # Selection info
                dpg.add_text(
                    "0 particles selected",
                    color=(128, 128, 128),
                    tag="sidebar_selection_info",
                )

    def _build_content_area(self) -> None:
        """Build the right content area with tab bar."""
        with dpg.child_window(
            tag=LAYOUT_TAGS.content_area,
            border=False,
            autosize_x=True,
            autosize_y=True,
        ):
            # Tab bar
            with dpg.tab_bar(
                tag=LAYOUT_TAGS.tab_bar,
                callback=self._on_tab_selected,
            ):
                # Intensity tab
                with dpg.tab(label="Intensity", tag="tab_button_intensity"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_intensity,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        # Build the actual intensity tab view
                        self._intensity_tab = IntensityTab(
                            parent=LAYOUT_TAGS.tab_intensity,
                        )
                        self._intensity_tab.build()

                # Lifetime tab
                with dpg.tab(label="Lifetime", tag="tab_button_lifetime"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_lifetime,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        # Build the actual lifetime tab view
                        self._lifetime_tab = LifetimeTab(
                            parent=LAYOUT_TAGS.tab_lifetime,
                        )
                        self._lifetime_tab.build()

                # Grouping tab
                with dpg.tab(label="Grouping", tag="tab_button_grouping"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_grouping,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        # Build the actual grouping tab view
                        self._grouping_tab = GroupingTab(
                            parent=LAYOUT_TAGS.tab_grouping,
                        )
                        self._grouping_tab.build()

                        # Wire up group selection to highlight on intensity plot
                        self._grouping_tab.set_on_group_selected(
                            self._on_group_selected_in_grouping_tab
                        )

                # Spectra tab
                with dpg.tab(label="Spectra", tag="tab_button_spectra"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_spectra,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        # Build the actual spectra tab view
                        self._spectra_tab = SpectraTab(
                            parent=LAYOUT_TAGS.tab_spectra,
                        )
                        self._spectra_tab.build()

                # Raster tab
                with dpg.tab(label="Raster", tag="tab_button_raster"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_raster,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        # Build the actual raster tab view
                        self._raster_tab = RasterTab(
                            parent=LAYOUT_TAGS.tab_raster,
                        )
                        self._raster_tab.build()

                # Correlation tab
                with dpg.tab(label="Correlation", tag="tab_button_correlation"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_correlation,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        # Build the actual correlation tab view
                        self._correlation_tab = CorrelationTab(
                            parent=LAYOUT_TAGS.tab_correlation,
                        )
                        self._correlation_tab.build()

                # Export tab
                with dpg.tab(label="Export", tag="tab_button_export"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_export,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        self._build_placeholder_tab("Export", ActiveTab.EXPORT)

    def _build_placeholder_tab(self, name: str, tab_type: ActiveTab) -> None:
        """Build a placeholder tab content.

        Args:
            name: Display name for the tab.
            tab_type: The ActiveTab enum value.
        """
        dpg.add_text(f"{name} Analysis", color=(100, 180, 255))
        dpg.add_separator()
        dpg.add_spacer(height=20)
        dpg.add_text(
            f"The {name.lower()} analysis view will be implemented here.",
            color=(180, 180, 180),
        )
        dpg.add_spacer(height=10)
        dpg.add_text(
            "Load an HDF5 file to begin analysis.",
            color=(128, 128, 128),
        )

    def _build_status_bar(self) -> None:
        """Build the status bar at the bottom using the StatusBar widget."""
        self._status_bar = StatusBar(parent=self._parent)
        self._status_bar.build()

    def _on_tab_selected(self, sender: int, app_data: int) -> None:
        """Handle tab selection.

        Args:
            sender: The tab bar that sent the callback.
            app_data: The selected tab button ID.
        """
        # Map tab button tags to ActiveTab
        tab_alias = dpg.get_item_alias(app_data)
        tab_map = {
            "tab_button_intensity": ActiveTab.INTENSITY,
            "tab_button_lifetime": ActiveTab.LIFETIME,
            "tab_button_grouping": ActiveTab.GROUPING,
            "tab_button_spectra": ActiveTab.SPECTRA,
            "tab_button_raster": ActiveTab.RASTER,
            "tab_button_correlation": ActiveTab.CORRELATION,
            "tab_button_export": ActiveTab.EXPORT,
        }

        if tab_alias in tab_map:
            self._active_tab = tab_map[tab_alias]
            if self._on_tab_change:
                self._on_tab_change(self._active_tab)

    def set_on_tab_change(self, callback: Callable[[ActiveTab], None]) -> None:
        """Set callback for tab change events.

        Args:
            callback: Function to call when tab changes, receives the new ActiveTab.
        """
        self._on_tab_change = callback

    @property
    def active_tab(self) -> ActiveTab:
        """Get the currently active tab."""
        return self._active_tab

    def set_active_tab(self, tab: ActiveTab) -> None:
        """Programmatically set the active tab.

        Args:
            tab: The tab to switch to.
        """
        tab_button_map = {
            ActiveTab.INTENSITY: "tab_button_intensity",
            ActiveTab.LIFETIME: "tab_button_lifetime",
            ActiveTab.GROUPING: "tab_button_grouping",
            ActiveTab.SPECTRA: "tab_button_spectra",
            ActiveTab.RASTER: "tab_button_raster",
            ActiveTab.CORRELATION: "tab_button_correlation",
            ActiveTab.EXPORT: "tab_button_export",
        }
        if tab in tab_button_map and dpg.does_item_exist(tab_button_map[tab]):
            dpg.set_value(LAYOUT_TAGS.tab_bar, tab_button_map[tab])
            self._active_tab = tab

    # Status bar methods

    def set_status(self, message: str) -> None:
        """Update the status bar message.

        Args:
            message: The status message to display.
        """
        if self._status_bar:
            self._status_bar.set_status(message)

    def set_file_info(self, info: str) -> None:
        """Update the file info display.

        Args:
            info: The file info text to display.
        """
        if self._status_bar:
            self._status_bar.set_file_info(info)

    def show_progress(self, value: float = 0.0, task: str = "") -> None:
        """Show the progress bar.

        Args:
            value: Initial progress value (0.0 to 1.0).
            task: Optional task description to display.
        """
        if self._status_bar:
            self._status_bar.show_progress(value, task)

    def update_progress(self, value: float, message: str = "") -> None:
        """Update the progress bar value.

        Args:
            value: Progress value (0.0 to 1.0).
            message: Optional status message to display.
        """
        if self._status_bar:
            self._status_bar.update_progress(value, message)

    def hide_progress(self) -> None:
        """Hide the progress bar."""
        if self._status_bar:
            self._status_bar.hide_progress()

    def sync_status_with_state(self, state: ProcessingState) -> None:
        """Synchronize the status bar with a ProcessingState.

        Args:
            state: The ProcessingState to sync with.
        """
        if self._status_bar:
            self._status_bar.sync_with_state(state)

    def show_error(self, message: str) -> None:
        """Display an error message in the status bar.

        Args:
            message: The error message to display.
        """
        if self._status_bar:
            self._status_bar.show_error(message)

    def show_success(self, message: str) -> None:
        """Display a success message in the status bar.

        Args:
            message: The success message to display.
        """
        if self._status_bar:
            self._status_bar.show_success(message)

    @property
    def status_bar(self) -> StatusBar | None:
        """Get the status bar widget instance."""
        return self._status_bar

    # Sidebar methods

    def update_selection_info(self, count: int) -> None:
        """Update the selection info text.

        Args:
            count: Number of selected items (particle/channel combinations).
        """
        if dpg.does_item_exist("sidebar_selection_info"):
            text = f"{count} item{'s' if count != 1 else ''} selected"
            dpg.set_value("sidebar_selection_info", text)

    def enable_sidebar_controls(self, enabled: bool = True) -> None:
        """Enable or disable sidebar control buttons.

        Args:
            enabled: Whether to enable the controls.
        """
        for tag in ["sidebar_select_all", "sidebar_clear_selection"]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=enabled)

    def get_tab_container(self, tab: ActiveTab) -> str:
        """Get the container tag for a specific tab.

        This allows other modules to add content to specific tabs.

        Args:
            tab: The tab to get the container for.

        Returns:
            The tag of the tab's content container.
        """
        return TAB_TAG_MAP.get(tab, LAYOUT_TAGS.tab_intensity)

    # Particle tree methods

    def set_particles(self, particles: list[ParticleData]) -> None:
        """Set particles to display in the tree.

        Args:
            particles: List of particles to display.
        """
        if self._particle_tree:
            self._particle_tree.set_particles(particles)
            # Enable controls if there are particles
            self.enable_sidebar_controls(len(particles) > 0)

    def set_on_selection_change(
        self, callback: Callable[[ChannelSelection | None], None]
    ) -> None:
        """Set callback for current selection changes.

        Args:
            callback: Function called when current selection changes.
        """
        self._on_selection_change = callback

    def set_on_batch_change(
        self, callback: Callable[[list[ChannelSelection]], None]
    ) -> None:
        """Set callback for batch selection changes.

        Args:
            callback: Function called when batch selection changes.
        """
        self._on_batch_change = callback

    def _on_particle_selection_changed(
        self, selection: ChannelSelection | None
    ) -> None:
        """Internal callback for particle tree selection changes.

        Args:
            selection: The new current selection, or None.
        """
        if self._on_selection_change:
            self._on_selection_change(selection)

    def _on_particle_batch_changed(
        self, selections: list[ChannelSelection]
    ) -> None:
        """Internal callback for particle tree batch selection changes.

        Args:
            selections: List of all selected items.
        """
        self.update_selection_info(len(selections))
        if self._on_batch_change:
            self._on_batch_change(selections)

    def _on_select_all_clicked(self) -> None:
        """Handle Select All button click."""
        if self._particle_tree:
            self._particle_tree.select_all()

    def _on_clear_selection_clicked(self) -> None:
        """Handle Clear button click."""
        if self._particle_tree:
            self._particle_tree.clear_selection()

    def select_particle(self, particle_id: int, channel: int = 1) -> None:
        """Programmatically select a particle/channel.

        Args:
            particle_id: The particle ID.
            channel: The channel number (default 1).
        """
        if self._particle_tree:
            self._particle_tree.select(particle_id, channel)

    def clear_selection(self) -> None:
        """Clear all particle selections."""
        if self._particle_tree:
            self._particle_tree.clear_selection()

    def select_all(self) -> None:
        """Select all particles for batch operations."""
        if self._particle_tree:
            self._particle_tree.select_all()

    @property
    def current_selection(self) -> ChannelSelection | None:
        """Get the current selection."""
        if self._particle_tree:
            return self._particle_tree.current_selection
        return None

    @property
    def batch_selection(self) -> list[ChannelSelection]:
        """Get all selected items for batch operations."""
        if self._particle_tree:
            return self._particle_tree.batch_selection
        return []

    @property
    def particle_tree(self) -> ParticleTree | None:
        """Get the particle tree widget instance."""
        return self._particle_tree

    # Intensity tab methods

    @property
    def intensity_tab(self) -> IntensityTab | None:
        """Get the intensity tab widget instance."""
        return self._intensity_tab

    def set_intensity_data(self, abstimes) -> None:
        """Set intensity trace data for the current particle.

        Args:
            abstimes: Absolute photon arrival times in nanoseconds.
        """
        if self._intensity_tab:
            self._intensity_tab.set_data(abstimes)

    def clear_intensity_data(self) -> None:
        """Clear the intensity tab data."""
        if self._intensity_tab:
            self._intensity_tab.clear()

    def set_intensity_bin_size(self, bin_size_ms: float) -> None:
        """Set the intensity plot bin size.

        Args:
            bin_size_ms: Bin size in milliseconds.
        """
        if self._intensity_tab:
            self._intensity_tab.set_bin_size(bin_size_ms)

    # Lifetime tab methods

    @property
    def lifetime_tab(self) -> LifetimeTab | None:
        """Get the lifetime tab widget instance."""
        return self._lifetime_tab

    def set_lifetime_data(self, microtimes, channelwidth: float) -> None:
        """Set decay histogram data for the current particle.

        Args:
            microtimes: TCSPC microtime values in nanoseconds.
            channelwidth: TCSPC channel width in nanoseconds.
        """
        if self._lifetime_tab:
            self._lifetime_tab.set_data(microtimes, channelwidth)

    def clear_lifetime_data(self) -> None:
        """Clear the lifetime tab data."""
        if self._lifetime_tab:
            self._lifetime_tab.clear()

    def set_lifetime_log_scale(self, log_scale: bool) -> None:
        """Set the lifetime plot log scale.

        Args:
            log_scale: Whether to use log scale.
        """
        if self._lifetime_tab:
            self._lifetime_tab.set_log_scale(log_scale)

    # Grouping tab methods

    @property
    def grouping_tab(self) -> GroupingTab | None:
        """Get the grouping tab widget instance."""
        return self._grouping_tab

    def set_clustering_result(self, result: ClusteringResult) -> None:
        """Set clustering result for the grouping tab.

        Args:
            result: The ClusteringResult from hierarchical clustering.
        """
        if self._grouping_tab:
            self._grouping_tab.set_clustering_result(result)

    def clear_grouping_data(self) -> None:
        """Clear the grouping tab data."""
        if self._grouping_tab:
            self._grouping_tab.clear()

    def enable_grouping_button(self, enabled: bool = True) -> None:
        """Enable or disable the Group button in the grouping tab.

        Args:
            enabled: Whether to enable the button.
        """
        if self._grouping_tab:
            self._grouping_tab.enable_group_button(enabled)

    def _on_group_selected_in_grouping_tab(self, group_id: int | None) -> None:
        """Handle group selection in the grouping tab.

        Updates the intensity plot to highlight the selected group.

        Args:
            group_id: The selected group ID (0-indexed), or None if deselected.
        """
        if self._intensity_tab:
            self._intensity_tab.set_highlighted_group(group_id)

    def set_highlighted_group(self, group_id: int | None) -> None:
        """Set the highlighted group on the intensity plot.

        Args:
            group_id: The group ID to highlight (0-indexed), or None to clear.
        """
        if self._intensity_tab:
            self._intensity_tab.set_highlighted_group(group_id)

        # Also update the grouping tab selection
        if self._grouping_tab:
            self._grouping_tab.set_selected_group(group_id)

    def clear_highlighted_group(self) -> None:
        """Clear any group highlighting on the intensity plot."""
        if self._intensity_tab:
            self._intensity_tab.clear_highlighted_group()

        if self._grouping_tab:
            self._grouping_tab.clear_selected_group()

    # Spectra tab methods

    @property
    def spectra_tab(self) -> SpectraTab | None:
        """Get the spectra tab widget instance."""
        return self._spectra_tab

    def set_spectra_data(self, spectra: SpectraData) -> None:
        """Set spectra data for the current particle.

        Args:
            spectra: The SpectraData object containing the spectral time series.
        """
        if self._spectra_tab:
            self._spectra_tab.set_data(spectra)

    def set_spectra_unavailable(self) -> None:
        """Indicate that the current particle has no spectra data."""
        if self._spectra_tab:
            self._spectra_tab.set_no_spectra()

    def clear_spectra_data(self) -> None:
        """Clear the spectra tab data."""
        if self._spectra_tab:
            self._spectra_tab.clear()

    def set_file_has_spectra(self, has_spectra: bool) -> None:
        """Set whether the loaded file has any spectra data.

        Args:
            has_spectra: Whether any particles in the file have spectra.
        """
        if self._spectra_tab:
            self._spectra_tab.set_file_has_spectra(has_spectra)

    # Raster tab methods

    @property
    def raster_tab(self) -> RasterTab | None:
        """Get the raster tab widget instance."""
        return self._raster_tab

    def set_raster_data(self, raster: RasterScanData) -> None:
        """Set raster scan data for the current particle.

        Args:
            raster: The RasterScanData object containing the 2D scan image.
        """
        if self._raster_tab:
            self._raster_tab.set_data(raster)

    def set_raster_unavailable(self) -> None:
        """Indicate that the current particle has no raster scan data."""
        if self._raster_tab:
            self._raster_tab.set_no_raster()

    def clear_raster_data(self) -> None:
        """Clear the raster tab data."""
        if self._raster_tab:
            self._raster_tab.clear()

    def set_file_has_raster(self, has_raster: bool) -> None:
        """Set whether the loaded file has any raster scan data.

        Args:
            has_raster: Whether any particles in the file have raster scans.
        """
        if self._raster_tab:
            self._raster_tab.set_file_has_raster(has_raster)

    # Correlation tab methods

    @property
    def correlation_tab(self) -> CorrelationTab | None:
        """Get the correlation tab widget instance."""
        return self._correlation_tab

    def set_correlation_data(
        self,
        abstimes1,
        abstimes2,
        microtimes1,
        microtimes2,
    ) -> None:
        """Set dual-channel data for correlation analysis.

        Args:
            abstimes1: Absolute times for channel 1 in nanoseconds.
            abstimes2: Absolute times for channel 2 in nanoseconds.
            microtimes1: Micro times for channel 1 in nanoseconds.
            microtimes2: Micro times for channel 2 in nanoseconds.
        """
        if self._correlation_tab:
            self._correlation_tab.set_dual_channel_data(
                abstimes1, abstimes2, microtimes1, microtimes2
            )

    def set_correlation_single_channel(self) -> None:
        """Indicate that the current particle has only one TCSPC channel."""
        if self._correlation_tab:
            self._correlation_tab.set_single_channel()

    def set_correlation_result(self, result: CorrelationResult) -> None:
        """Set the correlation analysis result.

        Args:
            result: The CorrelationResult from calculate_g2.
        """
        if self._correlation_tab:
            self._correlation_tab.set_correlation_result(result)

    def clear_correlation_data(self) -> None:
        """Clear the correlation tab data."""
        if self._correlation_tab:
            self._correlation_tab.clear()

    def set_on_correlate(self, callback) -> None:
        """Set callback for the correlate button.

        Args:
            callback: Function called with (window_ns, binsize_ns, difftime_ns).
        """
        if self._correlation_tab:
            self._correlation_tab.set_on_correlate(callback)
