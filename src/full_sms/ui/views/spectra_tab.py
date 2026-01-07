"""Spectra analysis tab view.

Provides the spectral time series visualization with:
- 2D heatmap showing intensity vs wavelength over time
- Colormap selection
- Basic plot controls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg

from full_sms.models.particle import SpectraData
from full_sms.ui.plots.spectra_plot import SpectraPlot

logger = logging.getLogger(__name__)


# Available colormaps for spectra display
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
class SpectraTabTags:
    """Tags for spectra tab elements."""

    container: str = "spectra_tab_view_container"
    controls_group: str = "spectra_tab_controls"
    colormap_combo: str = "spectra_tab_colormap"
    fit_view_button: str = "spectra_tab_fit_view"
    info_text: str = "spectra_tab_info"
    plot_container: str = "spectra_tab_plot_container"
    plot_area: str = "spectra_tab_plot_area"
    no_data_text: str = "spectra_tab_no_data"
    no_spectra_text: str = "spectra_tab_no_spectra"


SPECTRA_TAB_TAGS = SpectraTabTags()


class SpectraTab:
    """Spectra analysis tab view.

    Contains the spectral heatmap plot and controls for visualization.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the spectra tab.

        Args:
            parent: The parent container to build the tab in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._spectra_data: SpectraData | None = None
        self._has_spectra_in_file: bool = False

        # UI components
        self._spectra_plot: SpectraPlot | None = None

        # Generate unique tags
        self._tags = SpectraTabTags(
            container=f"{tag_prefix}spectra_tab_view_container",
            controls_group=f"{tag_prefix}spectra_tab_controls",
            colormap_combo=f"{tag_prefix}spectra_tab_colormap",
            fit_view_button=f"{tag_prefix}spectra_tab_fit_view",
            info_text=f"{tag_prefix}spectra_tab_info",
            plot_container=f"{tag_prefix}spectra_tab_plot_container",
            plot_area=f"{tag_prefix}spectra_tab_plot_area",
            no_data_text=f"{tag_prefix}spectra_tab_no_data",
            no_spectra_text=f"{tag_prefix}spectra_tab_no_spectra",
        )

    @property
    def tags(self) -> SpectraTabTags:
        """Get the tags for this tab instance."""
        return self._tags

    @property
    def spectra_plot(self) -> SpectraPlot | None:
        """Get the spectra plot widget."""
        return self._spectra_plot

    @property
    def has_data(self) -> bool:
        """Whether the tab has data loaded."""
        return self._spectra_data is not None

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
                    "Load an HDF5 file and select a particle to view spectra.",
                    tag=self._tags.no_data_text,
                    color=(128, 128, 128),
                )

                # No spectra placeholder (shown when particle has no spectra)
                dpg.add_text(
                    "This particle does not have spectral data.",
                    tag=self._tags.no_spectra_text,
                    color=(180, 140, 100),
                    show=False,
                )

                # Plot area (hidden until data loaded)
                with dpg.group(
                    tag=self._tags.plot_area,
                    show=False,
                ):
                    # Main spectra plot
                    self._spectra_plot = SpectraPlot(
                        parent=self._tags.plot_area,
                        tag_prefix=f"{self._tag_prefix}main_",
                    )
                    self._spectra_plot.build()

        self._is_built = True
        logger.debug("Spectra tab built")

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

            # Info text (shows spectra stats)
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
        if self._spectra_plot and app_data in COLORMAP_OPTIONS:
            self._spectra_plot.set_colormap(COLORMAP_OPTIONS[app_data])
        logger.debug(f"Colormap changed to {app_data}")

    def _on_fit_view_clicked(self) -> None:
        """Handle fit view button click."""
        if self._spectra_plot:
            self._spectra_plot.fit_view()

    def set_data(self, spectra: SpectraData) -> None:
        """Set the spectra data.

        Args:
            spectra: The SpectraData object containing the spectral time series.
        """
        self._spectra_data = spectra

        # Update plot
        if self._spectra_plot:
            self._spectra_plot.set_data(spectra)

        # Show plot, hide placeholders
        self._show_plot(True)

        # Enable controls
        self._enable_controls(True)

        # Update info text
        self._update_info_text()

        logger.debug(
            f"Spectra tab data set: {spectra.num_spectra} spectra × "
            f"{spectra.num_wavelengths} wavelengths"
        )

    def set_no_spectra(self) -> None:
        """Set the tab to show that the current particle has no spectra."""
        self._spectra_data = None

        if self._spectra_plot:
            self._spectra_plot.clear()

        # Show "no spectra" message
        self._show_no_spectra()

        # Disable controls
        self._enable_controls(False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        logger.debug("Spectra tab showing no spectra message")

    def clear(self) -> None:
        """Clear the tab data."""
        self._spectra_data = None

        if self._spectra_plot:
            self._spectra_plot.clear()

        # Hide plot, show placeholder
        self._show_plot(False)

        # Disable controls
        self._enable_controls(False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        logger.debug("Spectra tab cleared")

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

        # Hide the no spectra message when showing plot
        if show and dpg.does_item_exist(self._tags.no_spectra_text):
            dpg.configure_item(self._tags.no_spectra_text, show=False)

    def _show_no_spectra(self) -> None:
        """Show the 'no spectra available' message."""
        if dpg.does_item_exist(self._tags.plot_area):
            dpg.configure_item(self._tags.plot_area, show=False)

        if dpg.does_item_exist(self._tags.no_data_text):
            dpg.configure_item(self._tags.no_data_text, show=False)

        if dpg.does_item_exist(self._tags.no_spectra_text):
            dpg.configure_item(self._tags.no_spectra_text, show=True)

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

        if self._spectra_data is None:
            dpg.set_value(self._tags.info_text, "")
            return

        spectra = self._spectra_data

        # Format info string
        wl_range = f"{spectra.wavelength_min:.1f}-{spectra.wavelength_max:.1f} nm"
        time_range = f"{spectra.time_min:.1f}-{spectra.time_max:.1f} s"
        info = (
            f"{spectra.num_spectra} spectra | "
            f"{wl_range} | "
            f"{time_range} | "
            f"{spectra.exposure_time:.3f}s exposure"
        )

        dpg.set_value(self._tags.info_text, info)

    def set_file_has_spectra(self, has_spectra: bool) -> None:
        """Set whether the loaded file has any spectra data.

        This is used to determine whether to show the no-spectra message
        vs. the no-data-loaded message.

        Args:
            has_spectra: Whether any particles in the file have spectra.
        """
        self._has_spectra_in_file = has_spectra
