"""Export tab view.

Provides data export functionality with:
- Checkboxes for what to export (intensity, levels, groups, fits)
- Format selection (CSV, Parquet, Excel, JSON)
- Output directory picker
- Export button with progress feedback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import dearpygui.dearpygui as dpg
import numpy as np

from full_sms.io.exporters import ExportFormat
from full_sms.io.plot_exporters import PlotFormat
from full_sms.models.session import ChannelSelection, SessionState

logger = logging.getLogger(__name__)


# Format display names
FORMAT_OPTIONS = {
    "CSV": ExportFormat.CSV,
    "Parquet": ExportFormat.PARQUET,
    "Excel (.xlsx)": ExportFormat.EXCEL,
    "JSON": ExportFormat.JSON,
}
DEFAULT_FORMAT = "CSV"

# Plot format options
PLOT_FORMAT_OPTIONS = {
    "PNG": PlotFormat.PNG,
    "PDF": PlotFormat.PDF,
    "SVG": PlotFormat.SVG,
}
DEFAULT_PLOT_FORMAT = "PNG"


@dataclass
class ExportTabTags:
    """Tags for export tab elements."""

    container: str = "export_tab_view_container"
    controls_group: str = "export_tab_controls"

    # What to export checkboxes - Data
    export_intensity: str = "export_tab_intensity_cb"
    export_levels: str = "export_tab_levels_cb"
    export_groups: str = "export_tab_groups_cb"
    export_fits: str = "export_tab_fits_cb"

    # What to export checkboxes - Plots
    export_intensity_plot: str = "export_tab_intensity_plot_cb"
    export_decay_plot: str = "export_tab_decay_plot_cb"
    export_bic_plot: str = "export_tab_bic_plot_cb"
    export_correlation_plot: str = "export_tab_correlation_plot_cb"

    # Plot customization checkboxes
    intensity_include_levels: str = "export_tab_intensity_include_levels_cb"
    intensity_include_groups: str = "export_tab_intensity_include_groups_cb"
    decay_include_fit: str = "export_tab_decay_include_fit_cb"
    decay_include_irf: str = "export_tab_decay_include_irf_cb"

    # Format selection
    format_combo: str = "export_tab_format"
    plot_format_combo: str = "export_tab_plot_format"
    plot_dpi_input: str = "export_tab_plot_dpi"

    # Output directory
    output_dir_input: str = "export_tab_output_dir"
    browse_button: str = "export_tab_browse"

    # Bin size
    bin_size_input: str = "export_tab_bin_size"
    use_intensity_bin_size_cb: str = "export_tab_use_intensity_bin_size"

    # Export buttons
    export_current_button: str = "export_tab_export_current"
    export_selected_button: str = "export_tab_export_selected"
    export_all_button: str = "export_tab_export_all"

    # Status
    status_text: str = "export_tab_status"
    results_text: str = "export_tab_results"

    # No data message
    no_data_text: str = "export_tab_no_data"
    main_content: str = "export_tab_main_content"


EXPORT_TAB_TAGS = ExportTabTags()


class ExportTab:
    """Export tab view.

    Provides UI for exporting analysis data to various file formats.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the export tab.

        Args:
            parent: The parent container to build the tab in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # State
        self._session_state: SessionState | None = None
        self._output_dir: Path | None = None  # Will be set from session or smart default
        self._has_data = False
        self._current_selection: ChannelSelection | None = None
        self._batch_selection: list[ChannelSelection] = []

        # Callbacks
        self._on_export: Callable[[list[tuple[int, int]], Path, ExportFormat, dict], None] | None = None

        # Generate unique tags
        self._tags = ExportTabTags(
            container=f"{tag_prefix}export_tab_view_container",
            controls_group=f"{tag_prefix}export_tab_controls",
            export_intensity=f"{tag_prefix}export_tab_intensity_cb",
            export_levels=f"{tag_prefix}export_tab_levels_cb",
            export_groups=f"{tag_prefix}export_tab_groups_cb",
            export_fits=f"{tag_prefix}export_tab_fits_cb",
            export_intensity_plot=f"{tag_prefix}export_tab_intensity_plot_cb",
            export_decay_plot=f"{tag_prefix}export_tab_decay_plot_cb",
            export_bic_plot=f"{tag_prefix}export_tab_bic_plot_cb",
            export_correlation_plot=f"{tag_prefix}export_tab_correlation_plot_cb",
            intensity_include_levels=f"{tag_prefix}export_tab_intensity_include_levels_cb",
            intensity_include_groups=f"{tag_prefix}export_tab_intensity_include_groups_cb",
            decay_include_fit=f"{tag_prefix}export_tab_decay_include_fit_cb",
            decay_include_irf=f"{tag_prefix}export_tab_decay_include_irf_cb",
            format_combo=f"{tag_prefix}export_tab_format",
            plot_format_combo=f"{tag_prefix}export_tab_plot_format",
            plot_dpi_input=f"{tag_prefix}export_tab_plot_dpi",
            output_dir_input=f"{tag_prefix}export_tab_output_dir",
            browse_button=f"{tag_prefix}export_tab_browse",
            bin_size_input=f"{tag_prefix}export_tab_bin_size",
            use_intensity_bin_size_cb=f"{tag_prefix}export_tab_use_intensity_bin_size",
            export_current_button=f"{tag_prefix}export_tab_export_current",
            export_selected_button=f"{tag_prefix}export_tab_export_selected",
            export_all_button=f"{tag_prefix}export_tab_export_all",
            status_text=f"{tag_prefix}export_tab_status",
            results_text=f"{tag_prefix}export_tab_results",
            no_data_text=f"{tag_prefix}export_tab_no_data",
            main_content=f"{tag_prefix}export_tab_main_content",
        )

    @property
    def tags(self) -> ExportTabTags:
        """Get the tags for this tab instance."""
        return self._tags

    @property
    def has_data(self) -> bool:
        """Whether the tab has data loaded."""
        return self._has_data

    def build(self) -> None:
        """Build the tab UI structure."""
        if self._is_built:
            return

        # Main container
        with dpg.group(parent=self._parent, tag=self._tags.container):
            # No data placeholder
            dpg.add_text(
                "Load an HDF5 file to export data.",
                tag=self._tags.no_data_text,
                color=(128, 128, 128),
            )

            # Main content (hidden until data loaded)
            with dpg.group(tag=self._tags.main_content, show=False):
                # Title
                dpg.add_text("Data Export", color=(100, 180, 255))
                dpg.add_separator()
                dpg.add_spacer(height=10)

                # Two column layout
                with dpg.group(horizontal=True):
                    # Left column: Export options
                    with dpg.child_window(width=420, height=480, border=True):
                        self._build_export_options()

                    dpg.add_spacer(width=20)

                    # Right column: Output settings
                    with dpg.child_window(width=400, height=480, border=True):
                        self._build_output_settings()

                dpg.add_spacer(height=20)

                # Export buttons
                self._build_export_buttons()

                dpg.add_spacer(height=20)

                # Status and results
                self._build_status_area()

        self._is_built = True
        logger.debug("Export tab built")

    def _build_export_options(self) -> None:
        """Build the export options section."""
        # Data export section
        dpg.add_text("Data Export", color=(180, 180, 180))
        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Checkboxes for data export
        dpg.add_checkbox(
            label="Intensity Plot",
            default_value=True,
            tag=self._tags.export_intensity,
        )
        dpg.add_checkbox(
            label="Levels (Change Points)",
            default_value=True,
            tag=self._tags.export_levels,
        )
        dpg.add_checkbox(
            label="Groups (Clusters)",
            default_value=True,
            tag=self._tags.export_groups,
        )
        dpg.add_checkbox(
            label="Lifetime Fit Parameters",
            default_value=True,
            tag=self._tags.export_fits,
        )

        dpg.add_spacer(height=15)

        # Plot export section
        dpg.add_text("Plot Export", color=(180, 180, 180))
        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Intensity Plot
        dpg.add_checkbox(
            label="Intensity Plot",
            default_value=True,
            tag=self._tags.export_intensity_plot,
            callback=self._on_plot_checkbox_changed,
        )
        # Indented sub-options
        with dpg.group(indent=20):
            dpg.add_checkbox(
                label="Include Levels",
                default_value=True,
                tag=self._tags.intensity_include_levels,
            )
            dpg.add_checkbox(
                label="Include Groups",
                default_value=False,
                tag=self._tags.intensity_include_groups,
                enabled=False,  # Disabled for now per user request
            )

        dpg.add_spacer(height=5)

        # Decay Plot
        dpg.add_checkbox(
            label="Decay Plot",
            default_value=True,
            tag=self._tags.export_decay_plot,
            callback=self._on_plot_checkbox_changed,
        )
        # Indented sub-options
        with dpg.group(indent=20):
            dpg.add_checkbox(
                label="Include Fit",
                default_value=True,
                tag=self._tags.decay_include_fit,
            )
            dpg.add_checkbox(
                label="Include IRF",
                default_value=True,
                tag=self._tags.decay_include_irf,
            )

        dpg.add_spacer(height=5)

        # BIC Plot (no sub-options)
        dpg.add_checkbox(
            label="BIC Plot",
            default_value=True,
            tag=self._tags.export_bic_plot,
        )

        dpg.add_spacer(height=5)

        # Correlation Plot (no sub-options)
        dpg.add_checkbox(
            label="Correlation Plot",
            default_value=False,
            tag=self._tags.export_correlation_plot,
        )

    def _build_output_settings(self) -> None:
        """Build the output settings section."""
        dpg.add_text("Output Settings", color=(180, 180, 180))
        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Data format selection
        with dpg.group(horizontal=True):
            dpg.add_text("Data Format:")
            dpg.add_combo(
                items=list(FORMAT_OPTIONS.keys()),
                default_value=DEFAULT_FORMAT,
                tag=self._tags.format_combo,
                width=120,
            )

        dpg.add_spacer(height=5)

        # Plot format selection
        with dpg.group(horizontal=True):
            dpg.add_text("Plot Format:")
            dpg.add_combo(
                items=list(PLOT_FORMAT_OPTIONS.keys()),
                default_value=DEFAULT_PLOT_FORMAT,
                tag=self._tags.plot_format_combo,
                width=120,
            )

        dpg.add_spacer(height=5)

        # Plot DPI
        with dpg.group(horizontal=True):
            dpg.add_text("Plot DPI:")
            dpg.add_input_int(
                default_value=150,
                min_value=72,
                max_value=600,
                step=50,
                tag=self._tags.plot_dpi_input,
                width=120,
            )

        dpg.add_spacer(height=10)

        # Bin size section
        dpg.add_text("Bin Size", color=(180, 180, 180))
        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Checkbox to use intensity tab bin size
        dpg.add_checkbox(
            label="Use bin size from Intensity tab",
            default_value=True,
            tag=self._tags.use_intensity_bin_size_cb,
            callback=self._on_use_intensity_bin_size_changed,
        )

        dpg.add_spacer(height=5)

        # Bin size for intensity export
        with dpg.group(horizontal=True):
            dpg.add_text("Bin Size (ms):")
            dpg.add_input_float(
                default_value=10.0,
                min_value=0.1,
                max_value=1000.0,
                step=1.0,
                tag=self._tags.bin_size_input,
                width=120,
                enabled=False,  # Disabled by default since checkbox is checked
                callback=self._on_export_bin_size_changed,
            )

        dpg.add_spacer(height=10)

        # Output directory
        dpg.add_text("Output Directory:")
        dpg.add_spacer(height=3)

        with dpg.group(horizontal=True):
            dpg.add_input_text(
                default_value="",  # Will be set when session state is loaded
                tag=self._tags.output_dir_input,
                width=280,
                readonly=True,
            )
            dpg.add_button(
                label="Browse...",
                tag=self._tags.browse_button,
                callback=self._on_browse_clicked,
            )

    def _build_export_buttons(self) -> None:
        """Build the export action buttons."""
        logger.debug("Building export buttons")
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Export Current",
                tag=self._tags.export_current_button,
                width=140,
                callback=self._on_export_current,
                enabled=False,
            )
            dpg.add_spacer(width=10)

            dpg.add_button(
                label="Export Selected",
                tag=self._tags.export_selected_button,
                width=140,
                callback=self._on_export_selected,
                enabled=False,
            )
            dpg.add_spacer(width=10)

            dpg.add_button(
                label="Export All",
                tag=self._tags.export_all_button,
                width=140,
                callback=self._on_export_all,
                enabled=False,
            )
        logger.debug(f"Export buttons created with tags: current={self._tags.export_current_button}, selected={self._tags.export_selected_button}, all={self._tags.export_all_button}")

    def _build_status_area(self) -> None:
        """Build the status display area."""
        dpg.add_separator()
        dpg.add_spacer(height=10)

        dpg.add_text(
            "",
            tag=self._tags.status_text,
            color=(128, 128, 128),
        )

        dpg.add_spacer(height=5)

        dpg.add_text(
            "",
            tag=self._tags.results_text,
            color=(100, 200, 100),
            wrap=600,
        )

    def _on_plot_checkbox_changed(self, sender: int | str) -> None:
        """Handle plot checkbox state change to enable/disable sub-options.

        Args:
            sender: The checkbox tag.
        """
        if sender == self._tags.export_intensity_plot:
            enabled = dpg.get_value(self._tags.export_intensity_plot)
            if dpg.does_item_exist(self._tags.intensity_include_levels):
                dpg.configure_item(self._tags.intensity_include_levels, enabled=enabled)
            # Note: intensity_include_groups stays disabled per user request

        elif sender == self._tags.export_decay_plot:
            enabled = dpg.get_value(self._tags.export_decay_plot)
            if dpg.does_item_exist(self._tags.decay_include_fit):
                dpg.configure_item(self._tags.decay_include_fit, enabled=enabled)
            if dpg.does_item_exist(self._tags.decay_include_irf):
                dpg.configure_item(self._tags.decay_include_irf, enabled=enabled)

    def _on_use_intensity_bin_size_changed(self, sender: int | str) -> None:
        """Handle checkbox change for using intensity tab bin size.

        Args:
            sender: The checkbox tag.
        """
        use_intensity = dpg.get_value(self._tags.use_intensity_bin_size_cb)

        # Enable/disable the bin size input
        if dpg.does_item_exist(self._tags.bin_size_input):
            dpg.configure_item(self._tags.bin_size_input, enabled=not use_intensity)

        # Update session state if available
        if self._session_state is not None:
            self._session_state.ui_state.export_use_intensity_bin_size = use_intensity

            # If switching to use intensity bin size, sync the value
            if use_intensity:
                self._sync_bin_size_from_intensity()
            else:
                # Store current value as custom value
                if dpg.does_item_exist(self._tags.bin_size_input):
                    custom_value = dpg.get_value(self._tags.bin_size_input)
                    self._session_state.ui_state.export_bin_size_ms = custom_value

    def _on_export_bin_size_changed(self, sender: int | str, app_data: float) -> None:
        """Handle export bin size input change.

        Args:
            sender: The input field tag.
            app_data: The new bin size value.
        """
        # Only save if we're in custom mode (checkbox unchecked)
        if self._session_state is not None:
            use_intensity = dpg.get_value(self._tags.use_intensity_bin_size_cb) if dpg.does_item_exist(self._tags.use_intensity_bin_size_cb) else True
            if not use_intensity:
                self._session_state.ui_state.export_bin_size_ms = app_data
                logger.debug(f"Export bin size saved to session state: {app_data} ms")

    def _sync_bin_size_from_intensity(self) -> None:
        """Sync the export bin size from the intensity tab."""
        if self._session_state is not None and dpg.does_item_exist(self._tags.bin_size_input):
            intensity_bin_size = self._session_state.ui_state.bin_size_ms
            dpg.set_value(self._tags.bin_size_input, intensity_bin_size)

    def _get_default_export_directory(self) -> Path:
        """Get the default export directory using smart logic.

        Returns:
            Path to use as default export directory.
        """
        # 1. If already set in current state, use that
        if self._output_dir is not None:
            return self._output_dir

        # 2. If session state has an export directory, use that
        if self._session_state is not None and self._session_state.export_directory is not None:
            return self._session_state.export_directory

        # 3. If a file is loaded, use the file's directory
        if (
            self._session_state is not None
            and self._session_state.file_metadata is not None
        ):
            return self._session_state.file_metadata.path.parent

        # 4. Fall back to Desktop
        return Path.home() / "Desktop"

    def _on_browse_clicked(self) -> None:
        """Handle browse button click - open directory picker."""
        # Determine starting path using smart default logic
        start_path = self._get_default_export_directory()

        # Create file dialog for directory selection
        def dir_selected(sender, app_data):
            if app_data and "file_path_name" in app_data:
                dir_path = app_data["file_path_name"]
                # For directories, the path is in the file_path_name
                if Path(dir_path).is_dir():
                    self._output_dir = Path(dir_path)
                else:
                    # Use parent directory if a file was somehow selected
                    self._output_dir = Path(dir_path).parent

                # Save to session state
                if self._session_state is not None:
                    self._session_state.export_directory = self._output_dir

                # Update UI
                if dpg.does_item_exist(self._tags.output_dir_input):
                    dpg.set_value(self._tags.output_dir_input, str(self._output_dir))
            # Delete the dialog
            dpg.delete_item(sender)

        def cancel_callback(sender, app_data):
            dpg.delete_item(sender)

        # Create and show directory dialog
        with dpg.file_dialog(
            label="Select Output Directory",
            directory_selector=True,
            default_path=str(start_path),
            callback=dir_selected,
            cancel_callback=cancel_callback,
            width=600,
            height=400,
        ):
            pass

    def _get_export_options(self) -> dict:
        """Get current export options from UI."""
        # Get bin size - sync from intensity tab if checkbox is checked
        bin_size = 10.0
        if dpg.does_item_exist(self._tags.bin_size_input):
            use_intensity = dpg.get_value(self._tags.use_intensity_bin_size_cb) if dpg.does_item_exist(self._tags.use_intensity_bin_size_cb) else True
            if use_intensity and self._session_state is not None:
                bin_size = self._session_state.ui_state.bin_size_ms
            else:
                bin_size = dpg.get_value(self._tags.bin_size_input)
                # Save custom value to session state
                if self._session_state is not None:
                    self._session_state.ui_state.export_bin_size_ms = bin_size

        return {
            # Data export options
            "export_intensity": dpg.get_value(self._tags.export_intensity) if dpg.does_item_exist(self._tags.export_intensity) else True,
            "export_levels": dpg.get_value(self._tags.export_levels) if dpg.does_item_exist(self._tags.export_levels) else True,
            "export_groups": dpg.get_value(self._tags.export_groups) if dpg.does_item_exist(self._tags.export_groups) else True,
            "export_fits": dpg.get_value(self._tags.export_fits) if dpg.does_item_exist(self._tags.export_fits) else True,
            # Plot export options
            "export_intensity_plot": dpg.get_value(self._tags.export_intensity_plot) if dpg.does_item_exist(self._tags.export_intensity_plot) else True,
            "export_decay_plot": dpg.get_value(self._tags.export_decay_plot) if dpg.does_item_exist(self._tags.export_decay_plot) else True,
            "export_bic_plot": dpg.get_value(self._tags.export_bic_plot) if dpg.does_item_exist(self._tags.export_bic_plot) else True,
            "export_correlation_plot": dpg.get_value(self._tags.export_correlation_plot) if dpg.does_item_exist(self._tags.export_correlation_plot) else False,
            # Plot customization options
            "intensity_include_levels": dpg.get_value(self._tags.intensity_include_levels) if dpg.does_item_exist(self._tags.intensity_include_levels) else True,
            "intensity_include_groups": dpg.get_value(self._tags.intensity_include_groups) if dpg.does_item_exist(self._tags.intensity_include_groups) else False,
            "decay_include_fit": dpg.get_value(self._tags.decay_include_fit) if dpg.does_item_exist(self._tags.decay_include_fit) else True,
            "decay_include_irf": dpg.get_value(self._tags.decay_include_irf) if dpg.does_item_exist(self._tags.decay_include_irf) else True,
            # Common options
            "bin_size_ms": bin_size,
            "plot_dpi": dpg.get_value(self._tags.plot_dpi_input) if dpg.does_item_exist(self._tags.plot_dpi_input) else 150,
        }

    def _get_selected_plot_format(self) -> PlotFormat:
        """Get the currently selected plot export format."""
        if dpg.does_item_exist(self._tags.plot_format_combo):
            format_name = dpg.get_value(self._tags.plot_format_combo)
            return PLOT_FORMAT_OPTIONS.get(format_name, PlotFormat.PNG)
        return PlotFormat.PNG

    def _get_selected_format(self) -> ExportFormat:
        """Get the currently selected export format."""
        if dpg.does_item_exist(self._tags.format_combo):
            format_name = dpg.get_value(self._tags.format_combo)
            return FORMAT_OPTIONS.get(format_name, ExportFormat.CSV)
        return ExportFormat.CSV

    def _on_export_current(self) -> None:
        """Handle export current button click."""
        logger.info("Export current button clicked")
        if self._current_selection is None:
            logger.warning("No particle selected for export")
            self._set_status("No particle selected", error=True)
            return

        selections = [(self._current_selection.particle_id, self._current_selection.channel)]
        logger.info(f"Exporting current selection: {selections}")
        self._do_export(selections)

    def _on_export_selected(self) -> None:
        """Handle export selected button click."""
        if not self._batch_selection:
            self._set_status("No particles selected", error=True)
            return

        selections = [(s.particle_id, s.channel) for s in self._batch_selection]
        self._do_export(selections)

    def _on_export_all(self) -> None:
        """Handle export all button click."""
        if self._session_state is None or not self._session_state.particles:
            self._set_status("No particles to export", error=True)
            return

        selections = []
        for particle in self._session_state.particles:
            selections.append((particle.id, 1))
            if particle.has_dual_channel:
                selections.append((particle.id, 2))

        self._do_export(selections)

    def _do_export(self, selections: list[tuple[int, int]]) -> None:
        """Perform the export operation.

        Args:
            selections: List of (particle_id, channel) tuples to export.
        """
        logger.info(f"_do_export called with {len(selections)} selections")
        fmt = self._get_selected_format()
        plot_fmt = self._get_selected_plot_format()
        options = self._get_export_options()
        logger.debug(f"Export options: {options}")

        self._set_status(f"Exporting {len(selections)} particle(s)...")

        if self._on_export:
            logger.info("Using export callback")
            self._on_export(selections, self._output_dir, fmt, options)
        else:
            # No callback set - do export directly
            logger.info("No export callback, using direct export")
            self._do_export_sync(selections, fmt, plot_fmt, options)

    def _export_plots_for_particle(
        self,
        particle,
        channel: int,
        levels,
        clustering,
        fit_result,
        plot_fmt: PlotFormat,
        options: dict,
    ) -> list[Path]:
        """Export plots for a single particle with customization options.

        Args:
            particle: ParticleData instance.
            channel: Channel number (1 or 2).
            levels: Optional levels from CPA.
            clustering: Optional clustering result.
            fit_result: Optional fit result.
            plot_fmt: Plot export format.
            options: Export options dictionary.

        Returns:
            List of exported file paths.
        """
        from full_sms.io.plot_exporters import (
            export_intensity_plot,
            export_decay_plot,
            export_bic_plot,
        )
        from full_sms.analysis.histograms import build_decay_histogram

        exported: list[Path] = []
        prefix = f"particle_{particle.id}_ch{channel}"
        bin_size_ms = options.get("bin_size_ms", 10.0)
        dpi = options.get("plot_dpi", 150)

        # Get channel data
        channel_data = particle.channel1 if channel == 1 else particle.channel2
        if channel_data is None:
            logger.warning(f"No channel {channel} data for particle {particle.id}")
            return exported

        # Export intensity plot
        if options.get("export_intensity_plot", True):
            try:
                # Determine what to include in the intensity plot
                show_levels = options.get("intensity_include_levels", True)
                show_groups = options.get("intensity_include_groups", False)

                # Get groups from clustering if needed
                groups = list(clustering.groups) if (clustering and show_groups) else None

                path = export_intensity_plot(
                    abstimes=channel_data.abstimes,
                    output_path=self._output_dir / f"{prefix}_intensity",
                    bin_size_ms=bin_size_ms,
                    levels=levels if show_levels else None,
                    groups=groups,
                    show_levels=show_levels,
                    show_groups=show_groups,
                    fmt=plot_fmt,
                    title=f"Particle {particle.id} - Channel {channel}",
                    dpi=dpi,
                )
                exported.append(path)
            except Exception as e:
                logger.warning(f"Failed to export intensity plot: {e}")

        # Export decay plot
        if options.get("export_decay_plot", True) and channel_data.microtimes is not None:
            try:
                # Build decay histogram
                t_ns, counts = build_decay_histogram(
                    channel_data.microtimes.astype(np.float64),
                    particle.channelwidth,
                )

                if len(t_ns) > 0:
                    # Determine what to include in the decay plot
                    include_fit = options.get("decay_include_fit", True)
                    include_irf = options.get("decay_include_irf", True)

                    # Note: fit_result is FitResultData (serializable), not FitResult (with arrays)
                    # The plot exporter needs FitResult with fitted_curve and residuals arrays
                    # For now, we skip the fit in exported plots (would need to recompute)
                    # TODO: Recompute fitted curve from FitResultData parameters

                    # TODO: Get IRF data from session state when available
                    # For now, IRF export is not implemented
                    irf_t = None
                    irf_counts = None

                    path = export_decay_plot(
                        t_ns=t_ns,
                        counts=counts,
                        output_path=self._output_dir / f"{prefix}_decay",
                        fit_result=None,  # Skip fit for now - need FitResult, not FitResultData
                        irf_t=irf_t if include_irf else None,
                        irf_counts=irf_counts if include_irf else None,
                        fmt=plot_fmt,
                        title=f"Particle {particle.id} - Channel {channel} Decay",
                        dpi=dpi,
                    )
                    exported.append(path)
            except Exception as e:
                logger.warning(f"Failed to export decay plot: {e}")

        # Export BIC plot
        if options.get("export_bic_plot", True) and clustering is not None:
            try:
                path = export_bic_plot(
                    clustering_result=clustering,
                    output_path=self._output_dir / f"{prefix}_bic",
                    fmt=plot_fmt,
                    title=f"Particle {particle.id} - BIC Optimization",
                    dpi=dpi,
                )
                exported.append(path)
            except Exception as e:
                logger.warning(f"Failed to export BIC plot: {e}")

        # Note: Correlation plot not yet implemented in this method
        # Would be added here when correlation functionality is available

        return exported

    def _do_export_sync(
        self,
        selections: list[tuple[int, int]],
        fmt: ExportFormat,
        plot_fmt: PlotFormat,
        options: dict,
    ) -> None:
        """Perform synchronous export (fallback when no callback is set).

        Args:
            selections: List of (particle_id, channel) tuples.
            fmt: Export format for data.
            plot_fmt: Export format for plots.
            options: Export options dict.
        """
        logger.info(f"_do_export_sync started: {len(selections)} selections, format={fmt.value}, plot_format={plot_fmt.value}")
        logger.info(f"Output directory: {self._output_dir}")

        if self._session_state is None:
            logger.error("No session state available for export")
            self._set_status("No data to export", error=True)
            return

        from full_sms.io.exporters import export_batch
        from full_sms.io.plot_exporters import export_all_plots

        all_files: list[Path] = []

        try:
            # Export data files
            any_data_export = any([
                options.get("export_intensity", True),
                options.get("export_levels", True),
                options.get("export_groups", True),
                options.get("export_fits", True),
            ])

            if any_data_export:
                files = export_batch(
                    state=self._session_state,
                    selections=selections,
                    output_dir=self._output_dir,
                    fmt=fmt,
                    export_intensity=options.get("export_intensity", True),
                    export_levels=options.get("export_levels", True),
                    export_groups=options.get("export_groups", True),
                    export_fits=options.get("export_fits", True),
                    bin_size_ms=options.get("bin_size_ms", 10.0),
                )
                all_files.extend(files)

            # Export plot files
            any_plot_export = any([
                options.get("export_intensity_plot", True),
                options.get("export_decay_plot", True),
                options.get("export_bic_plot", True),
                options.get("export_correlation_plot", False),
            ])

            if any_plot_export:
                for particle_id, channel in selections:
                    particle = self._session_state.get_particle(particle_id)
                    if particle is None:
                        continue

                    levels = self._session_state.get_levels(particle_id, channel)
                    clustering = self._session_state.get_clustering(particle_id, channel)

                    # Get fit result for the whole particle
                    fit_result = self._session_state.get_particle_fit(particle_id, channel)

                    # Export individual plots with customization options
                    plot_files = self._export_plots_for_particle(
                        particle=particle,
                        channel=channel,
                        levels=levels,
                        clustering=clustering,
                        fit_result=fit_result,
                        plot_fmt=plot_fmt,
                        options=options,
                    )
                    all_files.extend(plot_files)

            self._set_status(f"Export complete: {len(all_files)} files", success=True)
            self._set_results(f"Files exported to: {self._output_dir}")

        except Exception as e:
            logger.exception("Export failed")
            self._set_status(f"Export failed: {e}", error=True)

    def _set_status(
        self,
        message: str,
        error: bool = False,
        success: bool = False,
    ) -> None:
        """Set the status message.

        Args:
            message: Status message to display.
            error: Whether this is an error message.
            success: Whether this is a success message.
        """
        if dpg.does_item_exist(self._tags.status_text):
            color = (128, 128, 128)
            if error:
                color = (255, 100, 100)
            elif success:
                color = (100, 255, 100)

            dpg.set_value(self._tags.status_text, message)
            dpg.configure_item(self._tags.status_text, color=color)

    def _set_results(self, message: str) -> None:
        """Set the results message.

        Args:
            message: Results message to display.
        """
        if dpg.does_item_exist(self._tags.results_text):
            dpg.set_value(self._tags.results_text, message)

    def set_session_state(self, state: SessionState) -> None:
        """Set the session state for export.

        Args:
            state: The current session state.
        """
        self._session_state = state
        self._has_data = state.has_file

        # Set output directory using smart default logic
        self._output_dir = self._get_default_export_directory()

        # Update UI to show the directory
        if dpg.does_item_exist(self._tags.output_dir_input):
            dpg.set_value(self._tags.output_dir_input, str(self._output_dir))

        # Save to session state if not already set
        if state.export_directory is None:
            state.export_directory = self._output_dir

        # Restore bin size settings from state
        if dpg.does_item_exist(self._tags.use_intensity_bin_size_cb):
            dpg.set_value(
                self._tags.use_intensity_bin_size_cb,
                state.ui_state.export_use_intensity_bin_size,
            )

        if dpg.does_item_exist(self._tags.bin_size_input):
            # Enable/disable based on checkbox state
            use_intensity = state.ui_state.export_use_intensity_bin_size
            dpg.configure_item(self._tags.bin_size_input, enabled=not use_intensity)

            # Set the value
            if use_intensity:
                # Use intensity tab bin size
                dpg.set_value(self._tags.bin_size_input, state.ui_state.bin_size_ms)
            else:
                # Use custom export bin size
                dpg.set_value(self._tags.bin_size_input, state.ui_state.export_bin_size_ms)

        self._update_ui_state()

    def set_current_selection(self, selection: ChannelSelection | None) -> None:
        """Set the current particle selection.

        Args:
            selection: The current selection, or None.
        """
        logger.debug(f"Export tab: set_current_selection called with {selection}")
        self._current_selection = selection
        self._update_button_states()

    def set_batch_selection(self, selections: list[ChannelSelection]) -> None:
        """Set the batch selection.

        Args:
            selections: List of selected particle/channels.
        """
        self._batch_selection = selections
        self._update_button_states()

    def _update_ui_state(self) -> None:
        """Update UI visibility based on data state."""
        if dpg.does_item_exist(self._tags.no_data_text):
            dpg.configure_item(self._tags.no_data_text, show=not self._has_data)

        if dpg.does_item_exist(self._tags.main_content):
            dpg.configure_item(self._tags.main_content, show=self._has_data)

        self._update_button_states()

    def _update_button_states(self) -> None:
        """Update export button enabled states."""
        has_current = self._current_selection is not None
        has_batch = len(self._batch_selection) > 0
        has_any = self._has_data and self._session_state is not None and len(self._session_state.particles) > 0

        logger.debug(f"Updating export button states: has_current={has_current}, has_batch={has_batch}, has_any={has_any}")

        if dpg.does_item_exist(self._tags.export_current_button):
            dpg.configure_item(self._tags.export_current_button, enabled=has_current)
            logger.debug(f"Export Current button enabled: {has_current}")

        if dpg.does_item_exist(self._tags.export_selected_button):
            dpg.configure_item(self._tags.export_selected_button, enabled=has_batch)
            # Update label to show count
            if has_batch:
                dpg.configure_item(
                    self._tags.export_selected_button,
                    label=f"Export Selected ({len(self._batch_selection)})",
                )
            else:
                dpg.configure_item(
                    self._tags.export_selected_button,
                    label="Export Selected",
                )

        if dpg.does_item_exist(self._tags.export_all_button):
            dpg.configure_item(self._tags.export_all_button, enabled=has_any)

    def set_on_export(
        self,
        callback: Callable[[list[tuple[int, int]], Path, ExportFormat, dict], None],
    ) -> None:
        """Set callback for export action.

        Args:
            callback: Function called with (selections, output_dir, format, options).
        """
        self._on_export = callback

    def set_output_directory(self, path: Path) -> None:
        """Set the output directory.

        Args:
            path: Path to the output directory.
        """
        self._output_dir = path
        if dpg.does_item_exist(self._tags.output_dir_input):
            dpg.set_value(self._tags.output_dir_input, str(path))

    def clear(self) -> None:
        """Clear the tab data."""
        self._session_state = None
        self._has_data = False
        self._current_selection = None
        self._batch_selection = []
        self._update_ui_state()
        self._set_status("")
        self._set_results("")
        logger.debug("Export tab cleared")

    def show_export_success(self, num_files: int, output_dir: Path) -> None:
        """Show export success message.

        Args:
            num_files: Number of files exported.
            output_dir: Directory files were exported to.
        """
        self._set_status(f"Export complete: {num_files} files", success=True)
        self._set_results(f"Files exported to: {output_dir}")

    def show_export_error(self, error: str) -> None:
        """Show export error message.

        Args:
            error: Error message to display.
        """
        self._set_status(f"Export failed: {error}", error=True)
        self._set_results("")

    def on_intensity_bin_size_changed(self, new_bin_size_ms: float) -> None:
        """Called when the intensity tab bin size changes.

        If the "Use bin size from Intensity tab" checkbox is checked,
        this will update the export bin size to match.

        Args:
            new_bin_size_ms: The new bin size from the intensity tab.
        """
        if dpg.does_item_exist(self._tags.use_intensity_bin_size_cb):
            use_intensity = dpg.get_value(self._tags.use_intensity_bin_size_cb)
            if use_intensity and dpg.does_item_exist(self._tags.bin_size_input):
                dpg.set_value(self._tags.bin_size_input, new_bin_size_ms)
