"""Main application layout for Full SMS.

Provides the two-column layout with particle tree sidebar, tab navigation,
and status bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import dearpygui.dearpygui as dpg

from full_sms.models.particle import ParticleData
from full_sms.models.session import ActiveTab, ChannelSelection
from full_sms.ui.widgets.particle_tree import ParticleTree


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
                        self._build_placeholder_tab("Intensity", ActiveTab.INTENSITY)

                # Lifetime tab
                with dpg.tab(label="Lifetime", tag="tab_button_lifetime"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_lifetime,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        self._build_placeholder_tab("Lifetime", ActiveTab.LIFETIME)

                # Grouping tab
                with dpg.tab(label="Grouping", tag="tab_button_grouping"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_grouping,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        self._build_placeholder_tab("Grouping", ActiveTab.GROUPING)

                # Spectra tab
                with dpg.tab(label="Spectra", tag="tab_button_spectra"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_spectra,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        self._build_placeholder_tab("Spectra", ActiveTab.SPECTRA)

                # Raster tab
                with dpg.tab(label="Raster", tag="tab_button_raster"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_raster,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        self._build_placeholder_tab("Raster", ActiveTab.RASTER)

                # Correlation tab
                with dpg.tab(label="Correlation", tag="tab_button_correlation"):
                    with dpg.child_window(
                        tag=LAYOUT_TAGS.tab_correlation,
                        border=False,
                        autosize_x=True,
                        autosize_y=True,
                    ):
                        self._build_placeholder_tab("Correlation", ActiveTab.CORRELATION)

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
        """Build the status bar at the bottom."""
        dpg.add_separator(parent=self._parent)
        with dpg.group(
            parent=self._parent,
            horizontal=True,
            tag=LAYOUT_TAGS.status_bar,
        ):
            # Status message
            dpg.add_text(
                "Ready",
                tag=LAYOUT_TAGS.status_text,
            )

            # Spacer to push file info to the right
            dpg.add_spacer(width=-1)

            # Progress bar (initially hidden via width=0)
            dpg.add_progress_bar(
                tag=LAYOUT_TAGS.progress_bar,
                default_value=0.0,
                width=150,
                show=False,
            )

            # File info (right-aligned)
            dpg.add_text(
                "",
                tag=LAYOUT_TAGS.file_info,
                color=(128, 128, 128),
            )

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
        if dpg.does_item_exist(LAYOUT_TAGS.status_text):
            dpg.set_value(LAYOUT_TAGS.status_text, message)

    def set_file_info(self, info: str) -> None:
        """Update the file info display.

        Args:
            info: The file info text to display.
        """
        if dpg.does_item_exist(LAYOUT_TAGS.file_info):
            dpg.set_value(LAYOUT_TAGS.file_info, info)

    def show_progress(self, value: float = 0.0) -> None:
        """Show the progress bar.

        Args:
            value: Initial progress value (0.0 to 1.0).
        """
        if dpg.does_item_exist(LAYOUT_TAGS.progress_bar):
            dpg.configure_item(LAYOUT_TAGS.progress_bar, show=True)
            dpg.set_value(LAYOUT_TAGS.progress_bar, max(0.0, min(1.0, value)))

    def update_progress(self, value: float) -> None:
        """Update the progress bar value.

        Args:
            value: Progress value (0.0 to 1.0).
        """
        if dpg.does_item_exist(LAYOUT_TAGS.progress_bar):
            dpg.set_value(LAYOUT_TAGS.progress_bar, max(0.0, min(1.0, value)))

    def hide_progress(self) -> None:
        """Hide the progress bar."""
        if dpg.does_item_exist(LAYOUT_TAGS.progress_bar):
            dpg.configure_item(LAYOUT_TAGS.progress_bar, show=False)

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
