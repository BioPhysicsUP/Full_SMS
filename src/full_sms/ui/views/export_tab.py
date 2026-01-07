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

from full_sms.io.exporters import ExportFormat
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


@dataclass
class ExportTabTags:
    """Tags for export tab elements."""

    container: str = "export_tab_view_container"
    controls_group: str = "export_tab_controls"

    # What to export checkboxes
    export_intensity: str = "export_tab_intensity_cb"
    export_levels: str = "export_tab_levels_cb"
    export_groups: str = "export_tab_groups_cb"
    export_fits: str = "export_tab_fits_cb"

    # Format selection
    format_combo: str = "export_tab_format"

    # Output directory
    output_dir_input: str = "export_tab_output_dir"
    browse_button: str = "export_tab_browse"

    # Bin size
    bin_size_input: str = "export_tab_bin_size"

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
        self._output_dir: Path = Path.home() / "Desktop"
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
            format_combo=f"{tag_prefix}export_tab_format",
            output_dir_input=f"{tag_prefix}export_tab_output_dir",
            browse_button=f"{tag_prefix}export_tab_browse",
            bin_size_input=f"{tag_prefix}export_tab_bin_size",
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
                    with dpg.child_window(width=350, height=300, border=True):
                        self._build_export_options()

                    dpg.add_spacer(width=20)

                    # Right column: Output settings
                    with dpg.child_window(width=400, height=300, border=True):
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
        dpg.add_text("What to Export", color=(180, 180, 180))
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Checkboxes for what to export
        dpg.add_checkbox(
            label="Intensity Trace",
            default_value=True,
            tag=self._tags.export_intensity,
        )
        dpg.add_text(
            "  Binned photon counts over time",
            color=(128, 128, 128),
        )
        dpg.add_spacer(height=5)

        dpg.add_checkbox(
            label="Levels (Change Points)",
            default_value=True,
            tag=self._tags.export_levels,
        )
        dpg.add_text(
            "  Detected intensity states from CPA",
            color=(128, 128, 128),
        )
        dpg.add_spacer(height=5)

        dpg.add_checkbox(
            label="Groups (Clusters)",
            default_value=True,
            tag=self._tags.export_groups,
        )
        dpg.add_text(
            "  Clustered intensity states from AHCA",
            color=(128, 128, 128),
        )
        dpg.add_spacer(height=5)

        dpg.add_checkbox(
            label="Fit Results",
            default_value=True,
            tag=self._tags.export_fits,
        )
        dpg.add_text(
            "  Lifetime fitting parameters",
            color=(128, 128, 128),
        )

    def _build_output_settings(self) -> None:
        """Build the output settings section."""
        dpg.add_text("Output Settings", color=(180, 180, 180))
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Format selection
        with dpg.group(horizontal=True):
            dpg.add_text("Format:")
            dpg.add_combo(
                items=list(FORMAT_OPTIONS.keys()),
                default_value=DEFAULT_FORMAT,
                tag=self._tags.format_combo,
                width=150,
            )

        dpg.add_spacer(height=10)

        # Bin size for intensity export
        with dpg.group(horizontal=True):
            dpg.add_text("Bin Size (ms):")
            dpg.add_input_float(
                default_value=10.0,
                min_value=0.1,
                max_value=1000.0,
                step=1.0,
                tag=self._tags.bin_size_input,
                width=100,
            )

        dpg.add_spacer(height=15)

        # Output directory
        dpg.add_text("Output Directory:")
        dpg.add_spacer(height=5)

        with dpg.group(horizontal=True):
            dpg.add_input_text(
                default_value=str(self._output_dir),
                tag=self._tags.output_dir_input,
                width=280,
                readonly=True,
            )
            dpg.add_button(
                label="Browse...",
                tag=self._tags.browse_button,
                callback=self._on_browse_clicked,
            )

        dpg.add_spacer(height=15)

        # Format notes
        dpg.add_text("Format Notes:", color=(180, 180, 180))
        dpg.add_text(
            "  CSV - Universal, human-readable",
            color=(128, 128, 128),
        )
        dpg.add_text(
            "  Parquet - Fast, compact (needs pyarrow)",
            color=(128, 128, 128),
        )
        dpg.add_text(
            "  Excel - Spreadsheet (needs openpyxl)",
            color=(128, 128, 128),
        )
        dpg.add_text(
            "  JSON - Structured with metadata",
            color=(128, 128, 128),
        )

    def _build_export_buttons(self) -> None:
        """Build the export action buttons."""
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

    def _on_browse_clicked(self) -> None:
        """Handle browse button click - open directory picker."""
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
            default_path=str(self._output_dir),
            callback=dir_selected,
            cancel_callback=cancel_callback,
            width=600,
            height=400,
        ):
            pass

    def _get_export_options(self) -> dict:
        """Get current export options from UI."""
        return {
            "export_intensity": dpg.get_value(self._tags.export_intensity) if dpg.does_item_exist(self._tags.export_intensity) else True,
            "export_levels": dpg.get_value(self._tags.export_levels) if dpg.does_item_exist(self._tags.export_levels) else True,
            "export_groups": dpg.get_value(self._tags.export_groups) if dpg.does_item_exist(self._tags.export_groups) else True,
            "export_fits": dpg.get_value(self._tags.export_fits) if dpg.does_item_exist(self._tags.export_fits) else True,
            "bin_size_ms": dpg.get_value(self._tags.bin_size_input) if dpg.does_item_exist(self._tags.bin_size_input) else 10.0,
        }

    def _get_selected_format(self) -> ExportFormat:
        """Get the currently selected export format."""
        if dpg.does_item_exist(self._tags.format_combo):
            format_name = dpg.get_value(self._tags.format_combo)
            return FORMAT_OPTIONS.get(format_name, ExportFormat.CSV)
        return ExportFormat.CSV

    def _on_export_current(self) -> None:
        """Handle export current button click."""
        if self._current_selection is None:
            self._set_status("No particle selected", error=True)
            return

        selections = [(self._current_selection.particle_id, self._current_selection.channel)]
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
        fmt = self._get_selected_format()
        options = self._get_export_options()

        self._set_status(f"Exporting {len(selections)} particle(s)...")

        if self._on_export:
            self._on_export(selections, self._output_dir, fmt, options)
        else:
            # No callback set - do export directly
            self._do_export_sync(selections, fmt, options)

    def _do_export_sync(
        self,
        selections: list[tuple[int, int]],
        fmt: ExportFormat,
        options: dict,
    ) -> None:
        """Perform synchronous export (fallback when no callback is set).

        Args:
            selections: List of (particle_id, channel) tuples.
            fmt: Export format.
            options: Export options dict.
        """
        if self._session_state is None:
            self._set_status("No data to export", error=True)
            return

        from full_sms.io.exporters import export_batch

        try:
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

            self._set_status(f"Export complete: {len(files)} files", success=True)
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
        self._update_ui_state()

    def set_current_selection(self, selection: ChannelSelection | None) -> None:
        """Set the current particle selection.

        Args:
            selection: The current selection, or None.
        """
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

        if dpg.does_item_exist(self._tags.export_current_button):
            dpg.configure_item(self._tags.export_current_button, enabled=has_current)

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
