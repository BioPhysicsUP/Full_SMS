"""Lifetime analysis tab view.

Provides the fluorescence decay histogram visualization with:
- Decay plot showing TCSPC histogram
- Fit curve overlay with results display
- IRF display
- Log/linear scale toggle
- Basic plot controls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.models.fit import FitResult
from full_sms.ui.plots.decay_plot import DecayPlot
from full_sms.ui.plots.residuals_plot import ResidualsPlot

logger = logging.getLogger(__name__)


@dataclass
class LifetimeTabTags:
    """Tags for lifetime tab elements."""

    container: str = "lifetime_tab_view_container"
    controls_group: str = "lifetime_tab_controls"
    log_scale_checkbox: str = "lifetime_tab_log_scale"
    show_fit_checkbox: str = "lifetime_tab_show_fit"
    show_irf_checkbox: str = "lifetime_tab_show_irf"
    show_residuals_checkbox: str = "lifetime_tab_show_residuals"
    fit_view_button: str = "lifetime_tab_fit_view"
    fit_button: str = "lifetime_tab_fit_button"
    info_text: str = "lifetime_tab_info"
    plot_container: str = "lifetime_tab_plot_container"
    plot_area: str = "lifetime_tab_plot_area"
    decay_plot_group: str = "lifetime_tab_decay_plot_group"
    residuals_plot_group: str = "lifetime_tab_residuals_plot_group"
    no_data_text: str = "lifetime_tab_no_data"
    # Fit results display
    results_group: str = "lifetime_tab_results_group"
    results_header: str = "lifetime_tab_results_header"
    tau_text: str = "lifetime_tab_tau_text"
    chi_squared_text: str = "lifetime_tab_chi_squared_text"
    dw_text: str = "lifetime_tab_dw_text"
    avg_lifetime_text: str = "lifetime_tab_avg_lifetime_text"


LIFETIME_TAB_TAGS = LifetimeTabTags()


class LifetimeTab:
    """Lifetime analysis tab view.

    Contains the decay histogram plot and controls for scale adjustment.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the lifetime tab.

        Args:
            parent: The parent container to build the tab in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._microtimes: NDArray[np.float64] | None = None
        self._channelwidth: float = 0.1  # Default 100ps
        self._log_scale: bool = True

        # Fit/IRF/Residuals state
        self._fit_result: Optional[FitResult] = None
        self._show_fit: bool = True
        self._show_irf: bool = False
        self._show_residuals: bool = True

        # Callbacks
        self._on_log_scale_changed: Callable[[bool], None] | None = None
        self._on_fit_requested: Callable[[], None] | None = None

        # UI components
        self._decay_plot: DecayPlot | None = None
        self._residuals_plot: ResidualsPlot | None = None

        # Generate unique tags
        self._tags = LifetimeTabTags(
            container=f"{tag_prefix}lifetime_tab_view_container",
            controls_group=f"{tag_prefix}lifetime_tab_controls",
            log_scale_checkbox=f"{tag_prefix}lifetime_tab_log_scale",
            show_fit_checkbox=f"{tag_prefix}lifetime_tab_show_fit",
            show_irf_checkbox=f"{tag_prefix}lifetime_tab_show_irf",
            show_residuals_checkbox=f"{tag_prefix}lifetime_tab_show_residuals",
            fit_view_button=f"{tag_prefix}lifetime_tab_fit_view",
            fit_button=f"{tag_prefix}lifetime_tab_fit_button",
            info_text=f"{tag_prefix}lifetime_tab_info",
            plot_container=f"{tag_prefix}lifetime_tab_plot_container",
            plot_area=f"{tag_prefix}lifetime_tab_plot_area",
            decay_plot_group=f"{tag_prefix}lifetime_tab_decay_plot_group",
            residuals_plot_group=f"{tag_prefix}lifetime_tab_residuals_plot_group",
            no_data_text=f"{tag_prefix}lifetime_tab_no_data",
            results_group=f"{tag_prefix}lifetime_tab_results_group",
            results_header=f"{tag_prefix}lifetime_tab_results_header",
            tau_text=f"{tag_prefix}lifetime_tab_tau_text",
            chi_squared_text=f"{tag_prefix}lifetime_tab_chi_squared_text",
            dw_text=f"{tag_prefix}lifetime_tab_dw_text",
            avg_lifetime_text=f"{tag_prefix}lifetime_tab_avg_lifetime_text",
        )

    @property
    def tags(self) -> LifetimeTabTags:
        """Get the tags for this tab instance."""
        return self._tags

    @property
    def log_scale(self) -> bool:
        """Get the current log scale setting."""
        return self._log_scale

    @property
    def decay_plot(self) -> DecayPlot | None:
        """Get the decay plot widget."""
        return self._decay_plot

    def build(self) -> None:
        """Build the tab UI structure."""
        if self._is_built:
            return

        # Main container
        with dpg.group(parent=self._parent, tag=self._tags.container):
            # Controls bar at top
            self._build_controls()

            # Fit results display (hidden until fit is available)
            self._build_fit_results_display()

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
                    "Load an HDF5 file and select a particle to view decay histogram.",
                    tag=self._tags.no_data_text,
                    color=(128, 128, 128),
                )

                # Plot area
                with dpg.group(
                    tag=self._tags.plot_area,
                    show=False,  # Hidden until data loaded
                ):
                    # Main decay plot (takes most of the space)
                    with dpg.child_window(
                        tag=self._tags.decay_plot_group,
                        border=False,
                        autosize_x=True,
                        height=-140,  # Leave room for residuals plot
                    ):
                        self._decay_plot = DecayPlot(
                            parent=self._tags.decay_plot_group,
                            tag_prefix=f"{self._tag_prefix}main_",
                        )
                        self._decay_plot.build()

                    # Residuals plot (below decay plot, hidden until fit)
                    with dpg.group(
                        tag=self._tags.residuals_plot_group,
                        show=False,  # Hidden until fit is available
                    ):
                        self._residuals_plot = ResidualsPlot(
                            parent=self._tags.residuals_plot_group,
                            tag_prefix=f"{self._tag_prefix}main_",
                            height=120,
                        )
                        self._residuals_plot.build()

                        # Link X axis to decay plot for synchronized panning
                        if self._decay_plot:
                            self._residuals_plot.link_x_axis(
                                self._decay_plot.tags.x_axis
                            )

        self._is_built = True
        logger.debug("Lifetime tab built")

    def _build_controls(self) -> None:
        """Build the controls bar at the top of the tab."""
        with dpg.group(horizontal=True, tag=self._tags.controls_group):
            # Log scale toggle
            dpg.add_checkbox(
                label="Log Scale",
                tag=self._tags.log_scale_checkbox,
                default_value=self._log_scale,
                callback=self._on_log_scale_checkbox_changed,
                enabled=False,
            )

            # Spacer
            dpg.add_spacer(width=15)

            # Show fit checkbox
            dpg.add_checkbox(
                label="Show Fit",
                tag=self._tags.show_fit_checkbox,
                default_value=self._show_fit,
                callback=self._on_show_fit_checkbox_changed,
                enabled=False,
            )

            # Spacer
            dpg.add_spacer(width=15)

            # Show IRF checkbox
            dpg.add_checkbox(
                label="Show IRF",
                tag=self._tags.show_irf_checkbox,
                default_value=self._show_irf,
                callback=self._on_show_irf_checkbox_changed,
                enabled=False,
            )

            # Spacer
            dpg.add_spacer(width=15)

            # Show Residuals checkbox
            dpg.add_checkbox(
                label="Show Residuals",
                tag=self._tags.show_residuals_checkbox,
                default_value=self._show_residuals,
                callback=self._on_show_residuals_checkbox_changed,
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
            dpg.add_spacer(width=15)

            # Fit button (opens fitting dialog)
            dpg.add_button(
                label="Fit...",
                tag=self._tags.fit_button,
                callback=self._on_fit_button_clicked,
                enabled=False,
            )

            # Spacer
            dpg.add_spacer(width=30)

            # Info text (shows photon count, time range)
            dpg.add_text(
                "",
                tag=self._tags.info_text,
                color=(128, 128, 128),
            )

    def _build_fit_results_display(self) -> None:
        """Build the fit results display section."""
        # Fit results group (hidden until fit is available)
        with dpg.group(
            tag=self._tags.results_group,
            horizontal=True,
            show=False,
        ):
            # Results header
            dpg.add_text(
                "Fit Results:",
                tag=self._tags.results_header,
                color=(180, 180, 180),
            )

            dpg.add_spacer(width=15)

            # Tau value(s)
            dpg.add_text(
                "",
                tag=self._tags.tau_text,
                color=(100, 180, 255),  # Blue for tau
            )

            dpg.add_spacer(width=20)

            # Average lifetime
            dpg.add_text(
                "",
                tag=self._tags.avg_lifetime_text,
                color=(180, 150, 255),  # Purple for average
            )

            dpg.add_spacer(width=20)

            # Chi-squared
            dpg.add_text(
                "",
                tag=self._tags.chi_squared_text,
                color=(100, 220, 150),  # Green for good fit
            )

            dpg.add_spacer(width=20)

            # Durbin-Watson
            dpg.add_text(
                "",
                tag=self._tags.dw_text,
                color=(200, 200, 200),
            )

    def _on_show_fit_checkbox_changed(
        self, sender: int, app_data: bool
    ) -> None:
        """Handle show fit checkbox changes.

        Args:
            sender: The checkbox widget.
            app_data: The new checkbox value.
        """
        self._show_fit = app_data

        if self._decay_plot:
            self._decay_plot.set_show_fit(app_data)

        logger.debug(f"Show fit changed to {app_data}")

    def _on_show_irf_checkbox_changed(
        self, sender: int, app_data: bool
    ) -> None:
        """Handle show IRF checkbox changes.

        Args:
            sender: The checkbox widget.
            app_data: The new checkbox value.
        """
        self._show_irf = app_data

        if self._decay_plot:
            self._decay_plot.set_show_irf(app_data)

        logger.debug(f"Show IRF changed to {app_data}")

    def _on_show_residuals_checkbox_changed(
        self, sender: int, app_data: bool
    ) -> None:
        """Handle show residuals checkbox changes.

        Args:
            sender: The checkbox widget.
            app_data: The new checkbox value.
        """
        self._show_residuals = app_data

        # Show or hide the residuals plot
        if dpg.does_item_exist(self._tags.residuals_plot_group):
            # Only show if we have fit data and the checkbox is enabled
            should_show = app_data and self._fit_result is not None
            dpg.configure_item(self._tags.residuals_plot_group, show=should_show)

        logger.debug(f"Show residuals changed to {app_data}")

    def _on_log_scale_checkbox_changed(
        self, sender: int, app_data: bool
    ) -> None:
        """Handle log scale checkbox changes.

        Args:
            sender: The checkbox widget.
            app_data: The new checkbox value.
        """
        self._log_scale = app_data

        # Update the plot
        if self._decay_plot:
            self._decay_plot.set_log_scale(app_data)

        # Call callback if set
        if self._on_log_scale_changed:
            self._on_log_scale_changed(app_data)

        logger.debug(f"Log scale changed to {app_data}")

    def _on_fit_view_clicked(self) -> None:
        """Handle fit view button click."""
        if self._decay_plot:
            self._decay_plot.fit_view()

    def _on_fit_button_clicked(self) -> None:
        """Handle fit button click - opens fitting dialog."""
        if self._on_fit_requested:
            self._on_fit_requested()
        logger.debug("Fit button clicked")

    def set_on_fit_requested(self, callback: Callable[[], None]) -> None:
        """Set callback for when Fit button is clicked.

        Args:
            callback: Function called when user clicks Fit button.
        """
        self._on_fit_requested = callback

    def set_data(
        self,
        microtimes: NDArray[np.float64],
        channelwidth: float,
    ) -> None:
        """Set the microtime data.

        Args:
            microtimes: TCSPC microtime values in nanoseconds.
            channelwidth: TCSPC channel width in nanoseconds.
        """
        self._microtimes = microtimes
        self._channelwidth = channelwidth

        if len(microtimes) == 0:
            self.clear()
            return

        # Update plot
        if self._decay_plot:
            self._decay_plot.set_data(microtimes, channelwidth)

        # Show plot, hide placeholder
        self._show_plot(True)

        # Enable controls
        if dpg.does_item_exist(self._tags.fit_view_button):
            dpg.configure_item(self._tags.fit_view_button, enabled=True)

        if dpg.does_item_exist(self._tags.log_scale_checkbox):
            dpg.configure_item(self._tags.log_scale_checkbox, enabled=True)

        if dpg.does_item_exist(self._tags.fit_button):
            dpg.configure_item(self._tags.fit_button, enabled=True)

        # Update info text
        self._update_info_text()

        logger.debug(f"Lifetime tab data set: {len(microtimes)} photons")

    def clear(self) -> None:
        """Clear the tab data."""
        self._microtimes = None
        self._fit_result = None

        if self._decay_plot:
            self._decay_plot.clear()

        # Hide plot, show placeholder
        self._show_plot(False)

        # Hide fit results
        self._show_fit_results(False)

        # Disable controls
        if dpg.does_item_exist(self._tags.fit_view_button):
            dpg.configure_item(self._tags.fit_view_button, enabled=False)

        if dpg.does_item_exist(self._tags.fit_button):
            dpg.configure_item(self._tags.fit_button, enabled=False)

        if dpg.does_item_exist(self._tags.log_scale_checkbox):
            dpg.configure_item(self._tags.log_scale_checkbox, enabled=False)

        if dpg.does_item_exist(self._tags.show_fit_checkbox):
            dpg.configure_item(self._tags.show_fit_checkbox, enabled=False)

        if dpg.does_item_exist(self._tags.show_irf_checkbox):
            dpg.configure_item(self._tags.show_irf_checkbox, enabled=False)

        if dpg.does_item_exist(self._tags.show_residuals_checkbox):
            dpg.configure_item(self._tags.show_residuals_checkbox, enabled=False)

        # Hide the residuals plot
        if dpg.does_item_exist(self._tags.residuals_plot_group):
            dpg.configure_item(self._tags.residuals_plot_group, show=False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        logger.debug("Lifetime tab cleared")

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

    def _update_info_text(self) -> None:
        """Update the info text with current data stats."""
        if not dpg.does_item_exist(self._tags.info_text):
            return

        if self._microtimes is None or len(self._microtimes) == 0:
            dpg.set_value(self._tags.info_text, "")
            return

        # Calculate stats
        num_photons = len(self._microtimes)
        time_range = self._decay_plot.get_time_range() if self._decay_plot else None
        max_counts = self._decay_plot.get_max_counts() if self._decay_plot else None

        if time_range and max_counts:
            duration_ns = time_range[1] - time_range[0]
            info = (
                f"{num_photons:,} photons | "
                f"{duration_ns:.1f} ns range | "
                f"Max: {max_counts:,} counts | "
                f"Channel: {self._channelwidth:.3f} ns"
            )
        else:
            info = f"{num_photons:,} photons | Channel: {self._channelwidth:.3f} ns"

        dpg.set_value(self._tags.info_text, info)

    def set_on_log_scale_changed(self, callback: Callable[[bool], None]) -> None:
        """Set callback for log scale changes.

        Args:
            callback: Function called when log scale changes, receives new value.
        """
        self._on_log_scale_changed = callback

    def set_log_scale(self, log_scale: bool) -> None:
        """Programmatically set the log scale.

        Args:
            log_scale: Whether to use log scale.
        """
        self._log_scale = log_scale

        # Update checkbox
        if dpg.does_item_exist(self._tags.log_scale_checkbox):
            dpg.set_value(self._tags.log_scale_checkbox, log_scale)

        # Update plot
        if self._decay_plot:
            self._decay_plot.set_log_scale(log_scale)

    @property
    def has_data(self) -> bool:
        """Whether the tab has data loaded."""
        return self._microtimes is not None and len(self._microtimes) > 0

    @property
    def channelwidth(self) -> float:
        """Get the current channel width in nanoseconds."""
        return self._channelwidth

    # -------------------------------------------------------------------------
    # Fit Result Methods
    # -------------------------------------------------------------------------

    @property
    def has_fit(self) -> bool:
        """Whether a fit result is currently set."""
        return self._fit_result is not None

    @property
    def fit_result(self) -> Optional[FitResult]:
        """Get the current fit result."""
        return self._fit_result

    @property
    def show_fit(self) -> bool:
        """Whether the fit curve is currently visible."""
        return self._show_fit

    def set_fit(self, fit_result: FitResult) -> None:
        """Set the fit result and display it.

        Args:
            fit_result: The FitResult from lifetime fitting.
        """
        self._fit_result = fit_result

        # Update the decay plot
        if self._decay_plot:
            self._decay_plot.set_fit(fit_result)

        # Update the residuals plot
        if self._residuals_plot and self._decay_plot:
            # Get time array for the fit range from decay plot
            if self._decay_plot._t is not None:
                fit_t = self._decay_plot._t[
                    fit_result.fit_start_index : fit_result.fit_end_index
                ]
                # Ensure residuals match the time array length
                residuals = fit_result.residuals
                if len(residuals) == len(fit_t):
                    self._residuals_plot.set_residuals(fit_t, residuals)
                else:
                    # Trim to match
                    min_len = min(len(residuals), len(fit_t))
                    self._residuals_plot.set_residuals(
                        fit_t[:min_len], residuals[:min_len]
                    )

        # Enable the show fit checkbox
        if dpg.does_item_exist(self._tags.show_fit_checkbox):
            dpg.configure_item(self._tags.show_fit_checkbox, enabled=True)

        # Enable the show residuals checkbox
        if dpg.does_item_exist(self._tags.show_residuals_checkbox):
            dpg.configure_item(self._tags.show_residuals_checkbox, enabled=True)

        # Show the residuals plot (if checkbox is checked)
        if dpg.does_item_exist(self._tags.residuals_plot_group):
            dpg.configure_item(
                self._tags.residuals_plot_group, show=self._show_residuals
            )

        # Update the fit results text display
        self._update_fit_results_text()

        # Show the fit results group
        self._show_fit_results(True)

        logger.debug(
            f"Fit set: tau={fit_result.tau}, chi2={fit_result.chi_squared:.3f}"
        )

    def clear_fit(self) -> None:
        """Clear the fit result."""
        self._fit_result = None

        if self._decay_plot:
            self._decay_plot.clear_fit()

        # Clear the residuals plot
        if self._residuals_plot:
            self._residuals_plot.clear()

        # Hide the residuals plot
        if dpg.does_item_exist(self._tags.residuals_plot_group):
            dpg.configure_item(self._tags.residuals_plot_group, show=False)

        # Disable the show fit checkbox
        if dpg.does_item_exist(self._tags.show_fit_checkbox):
            dpg.configure_item(self._tags.show_fit_checkbox, enabled=False)

        # Disable the show residuals checkbox
        if dpg.does_item_exist(self._tags.show_residuals_checkbox):
            dpg.configure_item(self._tags.show_residuals_checkbox, enabled=False)

        # Hide the fit results group
        self._show_fit_results(False)

        logger.debug("Fit cleared")

    def _show_fit_results(self, show: bool) -> None:
        """Show or hide the fit results display section.

        Args:
            show: Whether to show the fit results section.
        """
        if dpg.does_item_exist(self._tags.results_group):
            dpg.configure_item(self._tags.results_group, show=show)

    def _update_fit_results_text(self) -> None:
        """Update the fit results display text."""
        if self._fit_result is None:
            return

        fit = self._fit_result

        # Format tau values
        if fit.num_exponentials == 1:
            tau_text = f"tau = {fit.tau[0]:.2f} +/- {fit.tau_std[0]:.2f} ns"
        else:
            tau_parts = []
            for i in range(fit.num_exponentials):
                tau_parts.append(f"tau{i+1} = {fit.tau[i]:.2f} ns")
            tau_text = ", ".join(tau_parts)

        if dpg.does_item_exist(self._tags.tau_text):
            dpg.set_value(self._tags.tau_text, tau_text)

        # Average lifetime (for multi-exponential)
        if fit.num_exponentials > 1:
            avg_text = f"<tau> = {fit.average_lifetime:.2f} ns"
        else:
            avg_text = ""

        if dpg.does_item_exist(self._tags.avg_lifetime_text):
            dpg.set_value(self._tags.avg_lifetime_text, avg_text)

        # Chi-squared with color coding
        chi2 = fit.chi_squared
        if fit.is_good_fit:
            chi2_color = (100, 220, 150, 255)  # Green for good fit
        elif chi2 < 0.8:
            chi2_color = (255, 220, 100, 255)  # Yellow for overfitted
        else:
            chi2_color = (255, 150, 100, 255)  # Orange for poor fit

        if dpg.does_item_exist(self._tags.chi_squared_text):
            dpg.set_value(self._tags.chi_squared_text, f"chi2 = {chi2:.3f}")
            dpg.configure_item(self._tags.chi_squared_text, color=chi2_color)

        # Durbin-Watson
        dw = fit.durbin_watson
        dw_text = f"DW = {dw:.2f}"

        # Color code DW if bounds are available
        if fit.dw_is_acceptable is not None:
            if fit.dw_is_acceptable:
                dw_color = (100, 220, 150, 255)  # Green
            else:
                dw_color = (255, 150, 100, 255)  # Orange
        else:
            dw_color = (200, 200, 200, 255)  # Gray

        if dpg.does_item_exist(self._tags.dw_text):
            dpg.set_value(self._tags.dw_text, dw_text)
            dpg.configure_item(self._tags.dw_text, color=dw_color)

    def set_show_fit(self, show: bool) -> None:
        """Programmatically set the show fit state.

        Args:
            show: Whether to show the fit curve.
        """
        self._show_fit = show

        if dpg.does_item_exist(self._tags.show_fit_checkbox):
            dpg.set_value(self._tags.show_fit_checkbox, show)

        if self._decay_plot:
            self._decay_plot.set_show_fit(show)

    # -------------------------------------------------------------------------
    # IRF Methods
    # -------------------------------------------------------------------------

    @property
    def show_irf(self) -> bool:
        """Whether the IRF is currently visible."""
        return self._show_irf

    @property
    def has_irf(self) -> bool:
        """Whether IRF data is currently set."""
        return self._decay_plot.has_irf if self._decay_plot else False

    def set_irf(
        self,
        t: NDArray[np.float64],
        counts: NDArray[np.float64],
        normalize: bool = True,
    ) -> None:
        """Set the IRF data and display it.

        Args:
            t: Time array in nanoseconds.
            counts: Count array.
            normalize: If True, normalize IRF to match data peak.
        """
        if self._decay_plot:
            self._decay_plot.set_irf(t, counts, normalize=normalize)

        # Enable the show IRF checkbox
        if dpg.does_item_exist(self._tags.show_irf_checkbox):
            dpg.configure_item(self._tags.show_irf_checkbox, enabled=True)

        logger.debug(f"IRF set: {len(t)} points")

    def clear_irf(self) -> None:
        """Clear the IRF data."""
        if self._decay_plot:
            self._decay_plot.clear_irf()

        # Disable the show IRF checkbox
        if dpg.does_item_exist(self._tags.show_irf_checkbox):
            dpg.configure_item(self._tags.show_irf_checkbox, enabled=False)

        logger.debug("IRF cleared")

    def set_show_irf(self, show: bool) -> None:
        """Programmatically set the show IRF state.

        Args:
            show: Whether to show the IRF.
        """
        self._show_irf = show

        if dpg.does_item_exist(self._tags.show_irf_checkbox):
            dpg.set_value(self._tags.show_irf_checkbox, show)

        if self._decay_plot:
            self._decay_plot.set_show_irf(show)

    # -------------------------------------------------------------------------
    # Residuals Methods
    # -------------------------------------------------------------------------

    @property
    def residuals_plot(self) -> ResidualsPlot | None:
        """Get the residuals plot widget."""
        return self._residuals_plot

    @property
    def show_residuals(self) -> bool:
        """Whether the residuals plot is currently visible."""
        return self._show_residuals

    def set_show_residuals(self, show: bool) -> None:
        """Programmatically set the show residuals state.

        Args:
            show: Whether to show the residuals plot.
        """
        self._show_residuals = show

        if dpg.does_item_exist(self._tags.show_residuals_checkbox):
            dpg.set_value(self._tags.show_residuals_checkbox, show)

        # Only show if we have fit data
        if dpg.does_item_exist(self._tags.residuals_plot_group):
            should_show = show and self._fit_result is not None
            dpg.configure_item(self._tags.residuals_plot_group, show=should_show)
