"""Intensity analysis tab view.

Provides the intensity trace visualization with:
- Intensity plot showing binned photon counts over time
- Bin size control slider
- Basic plot controls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.ui.plots.intensity_plot import IntensityPlot

logger = logging.getLogger(__name__)


@dataclass
class IntensityTabTags:
    """Tags for intensity tab elements."""

    container: str = "intensity_tab_view_container"
    controls_group: str = "intensity_tab_controls"
    bin_size_slider: str = "intensity_tab_bin_size"
    bin_size_label: str = "intensity_tab_bin_size_label"
    fit_view_button: str = "intensity_tab_fit_view"
    plot_container: str = "intensity_tab_plot_container"
    info_text: str = "intensity_tab_info"
    no_data_text: str = "intensity_tab_no_data"


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

        # Callbacks
        self._on_bin_size_changed: Callable[[float], None] | None = None

        # UI components
        self._intensity_plot: IntensityPlot | None = None

        # Generate unique tags
        self._tags = IntensityTabTags(
            container=f"{tag_prefix}intensity_tab_view_container",
            controls_group=f"{tag_prefix}intensity_tab_controls",
            bin_size_slider=f"{tag_prefix}intensity_tab_bin_size",
            bin_size_label=f"{tag_prefix}intensity_tab_bin_size_label",
            fit_view_button=f"{tag_prefix}intensity_tab_fit_view",
            plot_container=f"{tag_prefix}intensity_tab_plot_container",
            info_text=f"{tag_prefix}intensity_tab_info",
            no_data_text=f"{tag_prefix}intensity_tab_no_data",
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

                # Intensity plot (hidden until data loaded)
                self._intensity_plot = IntensityPlot(
                    parent=self._tags.plot_container,
                    tag_prefix=f"{self._tag_prefix}main_",
                )
                self._intensity_plot.build()

                # Hide plot container initially
                dpg.configure_item(
                    self._intensity_plot.tags.container,
                    show=False,
                )

        self._is_built = True
        logger.debug("Intensity tab built")

    def _build_controls(self) -> None:
        """Build the controls bar at the top of the tab."""
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

            # Fit view button
            dpg.add_button(
                label="Fit View",
                tag=self._tags.fit_view_button,
                callback=self._on_fit_view_clicked,
                enabled=False,
            )

            # Spacer
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

        # Call callback if set
        if self._on_bin_size_changed:
            self._on_bin_size_changed(app_data)

        logger.debug(f"Bin size changed to {app_data:.1f} ms")

    def _on_fit_view_clicked(self) -> None:
        """Handle fit view button click."""
        if self._intensity_plot:
            self._intensity_plot.fit_view()

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

        # Show plot, hide placeholder
        self._show_plot(True)

        # Enable controls
        if dpg.does_item_exist(self._tags.fit_view_button):
            dpg.configure_item(self._tags.fit_view_button, enabled=True)

        # Update info text
        self._update_info_text()

        logger.debug(f"Intensity tab data set: {len(abstimes)} photons")

    def clear(self) -> None:
        """Clear the tab data."""
        self._abstimes = None

        if self._intensity_plot:
            self._intensity_plot.clear()

        # Hide plot, show placeholder
        self._show_plot(False)

        # Disable controls
        if dpg.does_item_exist(self._tags.fit_view_button):
            dpg.configure_item(self._tags.fit_view_button, enabled=False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        logger.debug("Intensity tab cleared")

    def _show_plot(self, show: bool) -> None:
        """Show or hide the plot.

        Args:
            show: Whether to show the plot (True) or placeholder (False).
        """
        if self._intensity_plot and dpg.does_item_exist(
            self._intensity_plot.tags.container
        ):
            dpg.configure_item(self._intensity_plot.tags.container, show=show)

        if dpg.does_item_exist(self._tags.no_data_text):
            dpg.configure_item(self._tags.no_data_text, show=not show)

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

    def set_on_bin_size_changed(self, callback: Callable[[float], None]) -> None:
        """Set callback for bin size changes.

        Args:
            callback: Function called when bin size changes, receives new bin size.
        """
        self._on_bin_size_changed = callback

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
