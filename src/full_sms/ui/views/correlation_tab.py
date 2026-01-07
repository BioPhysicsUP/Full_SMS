"""Correlation analysis tab view.

Provides the g2 correlation function analysis with:
- Correlation plot showing g2(tau)
- Window and bin size controls
- Channel offset control for timing corrections
- Statistics display
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.analysis.correlation import CorrelationResult
from full_sms.ui.plots.correlation_plot import CorrelationPlot

logger = logging.getLogger(__name__)


# Default correlation parameters
DEFAULT_WINDOW_NS = 500.0
DEFAULT_BINSIZE_NS = 1.0
DEFAULT_DIFFTIME_NS = 0.0


@dataclass
class CorrelationTabTags:
    """Tags for correlation tab elements."""

    container: str = "correlation_tab_view_container"
    controls_group: str = "correlation_tab_controls"
    window_input: str = "correlation_tab_window"
    binsize_input: str = "correlation_tab_binsize"
    difftime_input: str = "correlation_tab_difftime"
    correlate_button: str = "correlation_tab_correlate"
    rebin_button: str = "correlation_tab_rebin"
    fit_view_button: str = "correlation_tab_fit_view"
    info_text: str = "correlation_tab_info"
    plot_container: str = "correlation_tab_plot_container"
    plot_area: str = "correlation_tab_plot_area"
    no_data_text: str = "correlation_tab_no_data"
    single_channel_text: str = "correlation_tab_single_channel"


CORRELATION_TAB_TAGS = CorrelationTabTags()


class CorrelationTab:
    """Correlation analysis tab view.

    Contains the g2 correlation plot and controls for analysis parameters.
    Only enabled for particles with dual TCSPC channels.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the correlation tab.

        Args:
            parent: The parent container to build the tab in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._abstimes1: NDArray[np.uint64] | None = None
        self._abstimes2: NDArray[np.uint64] | None = None
        self._microtimes1: NDArray[np.float64] | None = None
        self._microtimes2: NDArray[np.float64] | None = None
        self._is_dual_channel: bool = False
        self._correlation_result: CorrelationResult | None = None

        # Parameters
        self._window_ns: float = DEFAULT_WINDOW_NS
        self._binsize_ns: float = DEFAULT_BINSIZE_NS
        self._difftime_ns: float = DEFAULT_DIFFTIME_NS

        # UI components
        self._correlation_plot: CorrelationPlot | None = None

        # Callbacks
        self._on_correlate: Callable[
            [float, float, float], None
        ] | None = None

        # Generate unique tags
        self._tags = CorrelationTabTags(
            container=f"{tag_prefix}correlation_tab_view_container",
            controls_group=f"{tag_prefix}correlation_tab_controls",
            window_input=f"{tag_prefix}correlation_tab_window",
            binsize_input=f"{tag_prefix}correlation_tab_binsize",
            difftime_input=f"{tag_prefix}correlation_tab_difftime",
            correlate_button=f"{tag_prefix}correlation_tab_correlate",
            rebin_button=f"{tag_prefix}correlation_tab_rebin",
            fit_view_button=f"{tag_prefix}correlation_tab_fit_view",
            info_text=f"{tag_prefix}correlation_tab_info",
            plot_container=f"{tag_prefix}correlation_tab_plot_container",
            plot_area=f"{tag_prefix}correlation_tab_plot_area",
            no_data_text=f"{tag_prefix}correlation_tab_no_data",
            single_channel_text=f"{tag_prefix}correlation_tab_single_channel",
        )

    @property
    def tags(self) -> CorrelationTabTags:
        """Get the tags for this tab instance."""
        return self._tags

    @property
    def correlation_plot(self) -> CorrelationPlot | None:
        """Get the correlation plot widget."""
        return self._correlation_plot

    @property
    def has_data(self) -> bool:
        """Whether the tab has correlation result data."""
        return self._correlation_result is not None

    @property
    def is_dual_channel(self) -> bool:
        """Whether current particle has dual channels."""
        return self._is_dual_channel

    @property
    def window_ns(self) -> float:
        """Current correlation window in nanoseconds."""
        return self._window_ns

    @property
    def binsize_ns(self) -> float:
        """Current bin size in nanoseconds."""
        return self._binsize_ns

    @property
    def difftime_ns(self) -> float:
        """Current channel time offset in nanoseconds."""
        return self._difftime_ns

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
                    "Load an HDF5 file and select a particle to view correlation.",
                    tag=self._tags.no_data_text,
                    color=(128, 128, 128),
                )

                # Single channel placeholder (shown when particle has only one channel)
                dpg.add_text(
                    "This particle has only one TCSPC channel.\n"
                    "Correlation analysis requires dual-channel data.",
                    tag=self._tags.single_channel_text,
                    color=(180, 140, 100),
                    show=False,
                )

                # Plot area (hidden until data loaded)
                with dpg.group(
                    tag=self._tags.plot_area,
                    show=False,
                ):
                    # Main correlation plot
                    self._correlation_plot = CorrelationPlot(
                        parent=self._tags.plot_area,
                        tag_prefix=f"{self._tag_prefix}main_",
                    )
                    self._correlation_plot.build()

        self._is_built = True
        logger.debug("Correlation tab built")

    def _build_controls(self) -> None:
        """Build the controls bar at the top of the tab."""
        with dpg.group(horizontal=True, tag=self._tags.controls_group):
            # Window input
            dpg.add_text("Window:")
            dpg.add_input_float(
                default_value=DEFAULT_WINDOW_NS,
                tag=self._tags.window_input,
                width=80,
                min_value=10.0,
                max_value=10000.0,
                min_clamped=True,
                max_clamped=True,
                step=50.0,
                callback=self._on_window_changed,
                enabled=False,
            )
            dpg.add_text("ns")

            dpg.add_spacer(width=15)

            # Bin size input
            dpg.add_text("Bin:")
            dpg.add_input_float(
                default_value=DEFAULT_BINSIZE_NS,
                tag=self._tags.binsize_input,
                width=60,
                min_value=0.1,
                max_value=100.0,
                min_clamped=True,
                max_clamped=True,
                step=0.5,
                callback=self._on_binsize_changed,
                enabled=False,
            )
            dpg.add_text("ns")

            dpg.add_spacer(width=15)

            # Channel offset input
            dpg.add_text("Offset:")
            dpg.add_input_float(
                default_value=DEFAULT_DIFFTIME_NS,
                tag=self._tags.difftime_input,
                width=60,
                min_value=-100.0,
                max_value=100.0,
                step=0.1,
                callback=self._on_difftime_changed,
                enabled=False,
            )
            dpg.add_text("ns")

            dpg.add_spacer(width=20)

            # Correlate button
            dpg.add_button(
                label="Correlate",
                tag=self._tags.correlate_button,
                callback=self._on_correlate_clicked,
                enabled=False,
            )

            dpg.add_spacer(width=10)

            # Rebin button (rebins from existing events)
            dpg.add_button(
                label="Rebin",
                tag=self._tags.rebin_button,
                callback=self._on_rebin_clicked,
                enabled=False,
            )

            dpg.add_spacer(width=10)

            # Fit view button
            dpg.add_button(
                label="Fit View",
                tag=self._tags.fit_view_button,
                callback=self._on_fit_view_clicked,
                enabled=False,
            )

            # Spacer
            dpg.add_spacer(width=30)

            # Info text (shows correlation stats)
            dpg.add_text(
                "",
                tag=self._tags.info_text,
                color=(128, 128, 128),
            )

    def _on_window_changed(self, sender: int, app_data: float) -> None:
        """Handle window input change.

        Args:
            sender: The input widget.
            app_data: The new window value.
        """
        self._window_ns = app_data
        logger.debug(f"Correlation window changed to {app_data}ns")

    def _on_binsize_changed(self, sender: int, app_data: float) -> None:
        """Handle bin size input change.

        Args:
            sender: The input widget.
            app_data: The new bin size value.
        """
        self._binsize_ns = app_data
        logger.debug(f"Correlation bin size changed to {app_data}ns")

    def _on_difftime_changed(self, sender: int, app_data: float) -> None:
        """Handle channel offset input change.

        Args:
            sender: The input widget.
            app_data: The new offset value.
        """
        self._difftime_ns = app_data
        logger.debug(f"Correlation channel offset changed to {app_data}ns")

    def _on_correlate_clicked(self) -> None:
        """Handle correlate button click."""
        if self._on_correlate:
            self._on_correlate(self._window_ns, self._binsize_ns, self._difftime_ns)
        logger.debug("Correlate button clicked")

    def _on_rebin_clicked(self) -> None:
        """Handle rebin button click."""
        if self._correlation_result is not None:
            # Rebin using the stored events
            try:
                new_result = self._correlation_result.rebin(
                    self._binsize_ns, self._window_ns
                )
                self.set_correlation_result(new_result)
                logger.debug(
                    f"Rebinned correlation: window={self._window_ns}ns, "
                    f"binsize={self._binsize_ns}ns"
                )
            except ValueError as e:
                logger.warning(f"Rebin failed: {e}")

    def _on_fit_view_clicked(self) -> None:
        """Handle fit view button click."""
        if self._correlation_plot:
            self._correlation_plot.fit_view()

    def set_on_correlate(
        self,
        callback: Callable[[float, float, float], None],
    ) -> None:
        """Set callback for correlate button.

        Args:
            callback: Function called with (window_ns, binsize_ns, difftime_ns).
        """
        self._on_correlate = callback

    def set_dual_channel_data(
        self,
        abstimes1: NDArray[np.uint64],
        abstimes2: NDArray[np.uint64],
        microtimes1: NDArray[np.float64],
        microtimes2: NDArray[np.float64],
    ) -> None:
        """Set dual-channel photon data for correlation analysis.

        Args:
            abstimes1: Absolute times for channel 1 in nanoseconds.
            abstimes2: Absolute times for channel 2 in nanoseconds.
            microtimes1: Micro times for channel 1 in nanoseconds.
            microtimes2: Micro times for channel 2 in nanoseconds.
        """
        self._abstimes1 = abstimes1
        self._abstimes2 = abstimes2
        self._microtimes1 = microtimes1
        self._microtimes2 = microtimes2
        self._is_dual_channel = True

        # Clear previous correlation result
        self._correlation_result = None
        if self._correlation_plot:
            self._correlation_plot.clear()

        # Show controls, enable correlate button
        self._show_plot_area(True)
        self._enable_controls(True)

        # Update info text
        self._update_info_text()

        logger.debug(
            f"Correlation tab data set: {len(abstimes1)} + {len(abstimes2)} photons"
        )

    def set_single_channel(self) -> None:
        """Indicate that the current particle has only one channel."""
        self._abstimes1 = None
        self._abstimes2 = None
        self._microtimes1 = None
        self._microtimes2 = None
        self._is_dual_channel = False
        self._correlation_result = None

        if self._correlation_plot:
            self._correlation_plot.clear()

        # Show single-channel message
        self._show_single_channel_message()

        # Disable controls
        self._enable_controls(False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        logger.debug("Correlation tab showing single-channel message")

    def set_correlation_result(self, result: CorrelationResult) -> None:
        """Set the correlation analysis result.

        Args:
            result: The CorrelationResult from calculate_g2.
        """
        self._correlation_result = result

        # Update plot
        if self._correlation_plot:
            self._correlation_plot.set_data(result)

        # Enable rebin button
        if dpg.does_item_exist(self._tags.rebin_button):
            dpg.configure_item(self._tags.rebin_button, enabled=True)

        # Update info text
        self._update_info_text()

        logger.debug(
            f"Correlation result set: {result.num_events} events, "
            f"g2(0) bin = {self._correlation_plot.get_g2_at_zero() if self._correlation_plot else 'N/A'}"
        )

    def clear(self) -> None:
        """Clear the tab data."""
        self._abstimes1 = None
        self._abstimes2 = None
        self._microtimes1 = None
        self._microtimes2 = None
        self._is_dual_channel = False
        self._correlation_result = None

        if self._correlation_plot:
            self._correlation_plot.clear()

        # Hide plot, show placeholder
        self._show_plot_area(False)

        # Disable controls
        self._enable_controls(False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        logger.debug("Correlation tab cleared")

    def _show_plot_area(self, show: bool) -> None:
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

        # Hide the single-channel message when showing plot
        if show and dpg.does_item_exist(self._tags.single_channel_text):
            dpg.configure_item(self._tags.single_channel_text, show=False)

    def _show_single_channel_message(self) -> None:
        """Show the 'single channel only' message."""
        if dpg.does_item_exist(self._tags.plot_area):
            dpg.configure_item(self._tags.plot_area, show=False)

        if dpg.does_item_exist(self._tags.no_data_text):
            dpg.configure_item(self._tags.no_data_text, show=False)

        if dpg.does_item_exist(self._tags.single_channel_text):
            dpg.configure_item(self._tags.single_channel_text, show=True)

    def _enable_controls(self, enable: bool) -> None:
        """Enable or disable control widgets.

        Args:
            enable: Whether to enable the controls.
        """
        for tag in [
            self._tags.window_input,
            self._tags.binsize_input,
            self._tags.difftime_input,
            self._tags.correlate_button,
            self._tags.fit_view_button,
        ]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=enable)

        # Rebin button only enabled if we have a result
        if dpg.does_item_exist(self._tags.rebin_button):
            dpg.configure_item(
                self._tags.rebin_button,
                enabled=enable and self._correlation_result is not None,
            )

    def _update_info_text(self) -> None:
        """Update the info text with current data/result stats."""
        if not dpg.does_item_exist(self._tags.info_text):
            return

        if not self._is_dual_channel:
            dpg.set_value(self._tags.info_text, "")
            return

        parts = []

        # Show photon counts
        if self._abstimes1 is not None and self._abstimes2 is not None:
            parts.append(f"Ch1: {len(self._abstimes1):,} | Ch2: {len(self._abstimes2):,}")

        # Show correlation stats if available
        if self._correlation_result is not None:
            result = self._correlation_result
            parts.append(f"{result.num_events:,} events")
            parts.append(f"{result.window_ns:.0f}ns window")
            parts.append(f"{result.binsize_ns:.1f}ns bins")

            # Show g2(0) value
            if self._correlation_plot:
                g2_zero = self._correlation_plot.get_g2_at_zero()
                if g2_zero is not None:
                    parts.append(f"g2(0)={g2_zero}")

        info = " | ".join(parts)
        dpg.set_value(self._tags.info_text, info)

    def enable_correlate_button(self, enabled: bool = True) -> None:
        """Enable or disable the correlate button.

        Args:
            enabled: Whether to enable the button.
        """
        if dpg.does_item_exist(self._tags.correlate_button):
            dpg.configure_item(self._tags.correlate_button, enabled=enabled)

    def get_channel_data(
        self,
    ) -> tuple[
        NDArray[np.uint64] | None,
        NDArray[np.uint64] | None,
        NDArray[np.float64] | None,
        NDArray[np.float64] | None,
    ]:
        """Get the current channel data for correlation.

        Returns:
            Tuple of (abstimes1, abstimes2, microtimes1, microtimes2).
        """
        return (
            self._abstimes1,
            self._abstimes2,
            self._microtimes1,
            self._microtimes2,
        )
