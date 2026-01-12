"""Raster scan analysis tab view.

Provides the raster scan image visualization with:
- 2D heatmap showing intensity at each X,Y position
- Colormap selection
- Basic plot controls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg

from full_sms.models.particle import RasterScanData
from full_sms.ui.plots.raster_plot import RasterPlot

logger = logging.getLogger(__name__)


# Available colormaps for raster display
COLORMAP_OPTIONS = {
    "Plasma": dpg.mvPlotColormap_Plasma,
    "Viridis": dpg.mvPlotColormap_Viridis,
    "Inferno": dpg.mvPlotColormap_Jet,
    "Hot": dpg.mvPlotColormap_Hot,
    "Cool": dpg.mvPlotColormap_Cool,
    "Twilight": dpg.mvPlotColormap_Twilight,
}
DEFAULT_COLORMAP = "Plasma"


@dataclass
class RasterTabTags:
    """Tags for raster tab elements."""

    container: str = "raster_tab_view_container"
    controls_group: str = "raster_tab_controls"
    colormap_combo: str = "raster_tab_colormap"
    fit_view_button: str = "raster_tab_fit_view"
    info_text: str = "raster_tab_info"
    plot_container: str = "raster_tab_plot_container"
    plot_area: str = "raster_tab_plot_area"
    no_data_text: str = "raster_tab_no_data"
    no_raster_text: str = "raster_tab_no_raster"


RASTER_TAB_TAGS = RasterTabTags()


class RasterTab:
    """Raster scan analysis tab view.

    Contains the raster scan heatmap plot and controls for visualization.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the raster tab.

        Args:
            parent: The parent container to build the tab in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._raster_data: RasterScanData | None = None
        self._has_raster_in_file: bool = False

        # UI components
        self._raster_plot: RasterPlot | None = None

        # Generate unique tags
        self._tags = RasterTabTags(
            container=f"{tag_prefix}raster_tab_view_container",
            controls_group=f"{tag_prefix}raster_tab_controls",
            colormap_combo=f"{tag_prefix}raster_tab_colormap",
            fit_view_button=f"{tag_prefix}raster_tab_fit_view",
            info_text=f"{tag_prefix}raster_tab_info",
            plot_container=f"{tag_prefix}raster_tab_plot_container",
            plot_area=f"{tag_prefix}raster_tab_plot_area",
            no_data_text=f"{tag_prefix}raster_tab_no_data",
            no_raster_text=f"{tag_prefix}raster_tab_no_raster",
        )

    @property
    def tags(self) -> RasterTabTags:
        """Get the tags for this tab instance."""
        return self._tags

    @property
    def raster_plot(self) -> RasterPlot | None:
        """Get the raster plot widget."""
        return self._raster_plot

    @property
    def has_data(self) -> bool:
        """Whether the tab has data loaded."""
        return self._raster_data is not None

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
                # No data placeholder (shown when no file loaded)
                dpg.add_text(
                    "Load an HDF5 file and select a particle to view raster scan.",
                    tag=self._tags.no_data_text,
                    color=(128, 128, 128),
                )

                # No raster scan placeholder (shown when particle has no raster)
                dpg.add_text(
                    "This particle does not have raster scan data.",
                    tag=self._tags.no_raster_text,
                    color=(180, 140, 100),
                    show=False,
                )

                # Plot area (hidden until data loaded)
                with dpg.group(
                    tag=self._tags.plot_area,
                    show=False,
                ):
                    # Main raster plot
                    self._raster_plot = RasterPlot(
                        parent=self._tags.plot_area,
                        tag_prefix=f"{self._tag_prefix}main_",
                    )
                    self._raster_plot.build()

        self._is_built = True
        logger.debug("Raster tab built")

    def _build_controls(self) -> None:
        """Build the controls bar at the top of the tab."""
        with dpg.group(horizontal=True, tag=self._tags.controls_group):
            # Colormap selection
            dpg.add_text("Colormap:")
            dpg.add_combo(
                items=list(COLORMAP_OPTIONS.keys()),
                default_value=DEFAULT_COLORMAP,
                tag=self._tags.colormap_combo,
                width=120,
                callback=self._on_colormap_changed,
                enabled=False,
            )

            # Spacer
            dpg.add_spacer(width=20)

            # Fit view button
            dpg.add_button(
                label="Fit View",
                tag=self._tags.fit_view_button,
                callback=self._on_fit_view_clicked,
                enabled=False,
            )

            # Spacer
            dpg.add_spacer(width=30)

            # Info text (shows raster stats)
            dpg.add_text(
                "",
                tag=self._tags.info_text,
                color=(128, 128, 128),
            )

    def _on_colormap_changed(self, sender: int, app_data: str) -> None:
        """Handle colormap combo change.

        Args:
            sender: The combo widget.
            app_data: The selected colormap name.
        """
        if self._raster_plot and app_data in COLORMAP_OPTIONS:
            self._raster_plot.set_colormap(COLORMAP_OPTIONS[app_data])
        logger.debug(f"Colormap changed to {app_data}")

    def _on_fit_view_clicked(self) -> None:
        """Handle fit view button click."""
        if self._raster_plot:
            self._raster_plot.fit_view()

    def set_data(self, raster: RasterScanData) -> None:
        """Set the raster scan data.

        Args:
            raster: The RasterScanData object containing the 2D scan image.
        """
        self._raster_data = raster

        # Update plot
        if self._raster_plot:
            self._raster_plot.set_data(raster)

        # Show plot, hide placeholders
        self._show_plot(True)

        # Enable controls
        self._enable_controls(True)

        # Update info text
        self._update_info_text()

        logger.debug(
            f"Raster tab data set: {raster.num_pixels_y}x{raster.num_pixels_x} pixels"
        )

    def set_no_raster(self) -> None:
        """Set the tab to show that the current particle has no raster scan."""
        self._raster_data = None

        if self._raster_plot:
            self._raster_plot.clear()

        # Show "no raster" message
        self._show_no_raster()

        # Disable controls
        self._enable_controls(False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        logger.debug("Raster tab showing no raster message")

    def clear(self) -> None:
        """Clear the tab data."""
        self._raster_data = None

        if self._raster_plot:
            self._raster_plot.clear()

        # Hide plot, show placeholder
        self._show_plot(False)

        # Disable controls
        self._enable_controls(False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        logger.debug("Raster tab cleared")

    def _show_plot(self, show: bool) -> None:
        """Show or hide the plot area.

        Args:
            show: Whether to show the plot area (True) or placeholder (False).
        """
        # Show/hide the plot area
        if dpg.does_item_exist(self._tags.plot_area):
            dpg.configure_item(self._tags.plot_area, show=show)

        # Show/hide the no data placeholder
        if dpg.does_item_exist(self._tags.no_data_text):
            dpg.configure_item(self._tags.no_data_text, show=not show)

        # Hide the no raster message when showing plot
        if show and dpg.does_item_exist(self._tags.no_raster_text):
            dpg.configure_item(self._tags.no_raster_text, show=False)

    def _show_no_raster(self) -> None:
        """Show the 'no raster available' message."""
        if dpg.does_item_exist(self._tags.plot_area):
            dpg.configure_item(self._tags.plot_area, show=False)

        if dpg.does_item_exist(self._tags.no_data_text):
            dpg.configure_item(self._tags.no_data_text, show=False)

        if dpg.does_item_exist(self._tags.no_raster_text):
            dpg.configure_item(self._tags.no_raster_text, show=True)

    def _enable_controls(self, enable: bool) -> None:
        """Enable or disable control widgets.

        Args:
            enable: Whether to enable the controls.
        """
        for tag in [
            self._tags.colormap_combo,
            self._tags.fit_view_button,
        ]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=enable)

    def _update_info_text(self) -> None:
        """Update the info text with current data stats."""
        if not dpg.does_item_exist(self._tags.info_text):
            return

        if self._raster_data is None:
            dpg.set_value(self._tags.info_text, "")
            return

        raster = self._raster_data

        # Format info string
        size = f"{raster.num_pixels_y}x{raster.num_pixels_x} pixels"
        scan_range = f"{raster.scan_range:.1f} um range"
        position = f"({raster.x_start:.1f}, {raster.y_start:.1f}) um start"
        pixel_size = f"{raster.pixel_size:.3f} um/pixel"
        intensity = f"{raster.intensity_min:.0f}-{raster.intensity_max:.0f} counts"

        info = f"{size} | {scan_range} | {position} | {pixel_size} | {intensity}"

        dpg.set_value(self._tags.info_text, info)

    def set_file_has_raster(self, has_raster: bool) -> None:
        """Set whether the loaded file has any raster scan data.

        This is used to determine whether to show the no-raster message
        vs. the no-data-loaded message.

        Args:
            has_raster: Whether any particles in the file have raster scans.
        """
        self._has_raster_in_file = has_raster

    def set_particle_marker(self, x: float, y: float) -> None:
        """Set the particle position marker on the raster scan.

        Args:
            x: X coordinate in micrometers.
            y: Y coordinate in micrometers.
        """
        if self._raster_plot:
            self._raster_plot.set_particle_marker(x, y)

    def clear_particle_marker(self) -> None:
        """Clear the particle position marker."""
        if self._raster_plot:
            self._raster_plot.clear_particle_marker()
