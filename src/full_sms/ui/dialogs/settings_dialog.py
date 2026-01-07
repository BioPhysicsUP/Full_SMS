"""Settings dialog for configuring application preferences.

Provides a modal dialog for configuring:
- Change point analysis settings (min_photons, min_boundary_offset)
- Lifetime fitting settings (moving average, fit boundaries)
- Display settings (default bin size, auto-resolve)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import dearpygui.dearpygui as dpg

from full_sms.config import Settings, get_settings, save_settings

logger = logging.getLogger(__name__)


@dataclass
class SettingsDialogTags:
    """Tags for settings dialog UI elements."""

    dialog: str = "settings_dialog"
    # Change point analysis
    cpa_min_photons: str = "settings_cpa_min_photons"
    cpa_min_boundary: str = "settings_cpa_min_boundary"
    # Lifetime settings
    lt_use_moving_avg: str = "settings_lt_use_moving_avg"
    lt_moving_avg_window: str = "settings_lt_moving_avg_window"
    lt_moving_avg_row: str = "settings_lt_moving_avg_row"
    lt_start_percent: str = "settings_lt_start_percent"
    lt_end_multiple: str = "settings_lt_end_multiple"
    lt_end_percent: str = "settings_lt_end_percent"
    lt_min_decay_window: str = "settings_lt_min_decay_window"
    lt_bg_percent: str = "settings_lt_bg_percent"
    # Display settings
    disp_default_bin_size: str = "settings_disp_default_bin_size"
    disp_auto_resolve: str = "settings_disp_auto_resolve"
    # Buttons
    save_button: str = "settings_save_button"
    cancel_button: str = "settings_cancel_button"
    reset_button: str = "settings_reset_button"


class SettingsDialog:
    """Modal dialog for configuring application settings.

    Usage:
        dialog = SettingsDialog()
        dialog.set_on_save(callback)  # Called when settings are saved
        dialog.show()
    """

    def __init__(self, tag_prefix: str = "") -> None:
        """Initialize the settings dialog.

        Args:
            tag_prefix: Optional prefix for unique tags.
        """
        self._tag_prefix = tag_prefix
        self._is_built = False
        self._on_save: Optional[Callable[[Settings], None]] = None
        self._on_cancel: Optional[Callable[[], None]] = None

        # Generate unique tags
        self._tags = SettingsDialogTags(
            dialog=f"{tag_prefix}settings_dialog",
            cpa_min_photons=f"{tag_prefix}settings_cpa_min_photons",
            cpa_min_boundary=f"{tag_prefix}settings_cpa_min_boundary",
            lt_use_moving_avg=f"{tag_prefix}settings_lt_use_moving_avg",
            lt_moving_avg_window=f"{tag_prefix}settings_lt_moving_avg_window",
            lt_moving_avg_row=f"{tag_prefix}settings_lt_moving_avg_row",
            lt_start_percent=f"{tag_prefix}settings_lt_start_percent",
            lt_end_multiple=f"{tag_prefix}settings_lt_end_multiple",
            lt_end_percent=f"{tag_prefix}settings_lt_end_percent",
            lt_min_decay_window=f"{tag_prefix}settings_lt_min_decay_window",
            lt_bg_percent=f"{tag_prefix}settings_lt_bg_percent",
            disp_default_bin_size=f"{tag_prefix}settings_disp_default_bin_size",
            disp_auto_resolve=f"{tag_prefix}settings_disp_auto_resolve",
            save_button=f"{tag_prefix}settings_save_button",
            cancel_button=f"{tag_prefix}settings_cancel_button",
            reset_button=f"{tag_prefix}settings_reset_button",
        )

    @property
    def tags(self) -> SettingsDialogTags:
        """Get the dialog tags."""
        return self._tags

    def set_on_save(self, callback: Callable[[Settings], None]) -> None:
        """Set callback for when settings are saved.

        Args:
            callback: Function called with Settings when user clicks Save.
        """
        self._on_save = callback

    def set_on_cancel(self, callback: Callable[[], None]) -> None:
        """Set callback for when Cancel button is clicked.

        Args:
            callback: Function called when user clicks Cancel.
        """
        self._on_cancel = callback

    def build(self) -> None:
        """Build the dialog UI (but don't show it)."""
        if self._is_built:
            return

        # Delete existing dialog if present
        if dpg.does_item_exist(self._tags.dialog):
            dpg.delete_item(self._tags.dialog)

        # Create modal window
        with dpg.window(
            label="Settings",
            tag=self._tags.dialog,
            modal=True,
            show=False,
            width=450,
            height=520,
            no_resize=True,
            no_move=False,
            on_close=self._on_close,
        ):
            # Use a tab bar for organization
            with dpg.tab_bar():
                # Change Point Analysis tab
                with dpg.tab(label="Change Point"):
                    dpg.add_spacer(height=10)
                    dpg.add_text(
                        "Change Point Analysis Settings",
                        color=(100, 180, 255),
                    )
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    # Min photons
                    with dpg.group(horizontal=True):
                        dpg.add_text("Min photons per segment:", indent=10)
                        dpg.add_input_int(
                            tag=self._tags.cpa_min_photons,
                            default_value=20,
                            width=100,
                            min_value=5,
                            max_value=1000,
                            min_clamped=True,
                            max_clamped=True,
                        )

                    dpg.add_spacer(height=5)

                    # Min boundary offset
                    with dpg.group(horizontal=True):
                        dpg.add_text("Min boundary offset:", indent=10)
                        dpg.add_input_int(
                            tag=self._tags.cpa_min_boundary,
                            default_value=7,
                            width=100,
                            min_value=1,
                            max_value=100,
                            min_clamped=True,
                            max_clamped=True,
                        )

                    dpg.add_spacer(height=10)
                    dpg.add_text(
                        "These settings control the sensitivity of\n"
                        "level detection in intensity traces.",
                        color=(128, 128, 128),
                        indent=10,
                    )

                # Lifetime Fitting tab
                with dpg.tab(label="Lifetime"):
                    dpg.add_spacer(height=10)
                    dpg.add_text(
                        "Lifetime Fitting Settings",
                        color=(100, 180, 255),
                    )
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    # Moving average
                    dpg.add_checkbox(
                        label="Use moving average smoothing",
                        tag=self._tags.lt_use_moving_avg,
                        default_value=True,
                        callback=self._on_moving_avg_changed,
                        indent=10,
                    )

                    with dpg.group(
                        horizontal=True,
                        tag=self._tags.lt_moving_avg_row,
                    ):
                        dpg.add_text("Window size:", indent=20)
                        dpg.add_input_int(
                            tag=self._tags.lt_moving_avg_window,
                            default_value=10,
                            width=80,
                            min_value=3,
                            max_value=100,
                            min_clamped=True,
                            max_clamped=True,
                        )

                    dpg.add_separator()
                    dpg.add_spacer(height=5)
                    dpg.add_text("Auto Fit Range Detection", indent=10)

                    # Start percent
                    with dpg.group(horizontal=True):
                        dpg.add_text("Start at % of max:", indent=20)
                        dpg.add_input_int(
                            tag=self._tags.lt_start_percent,
                            default_value=80,
                            width=80,
                            min_value=1,
                            max_value=100,
                            min_clamped=True,
                            max_clamped=True,
                        )

                    # End multiple
                    with dpg.group(horizontal=True):
                        dpg.add_text("End at tau multiple:", indent=20)
                        dpg.add_input_int(
                            tag=self._tags.lt_end_multiple,
                            default_value=20,
                            width=80,
                            min_value=1,
                            max_value=100,
                            min_clamped=True,
                            max_clamped=True,
                        )

                    # End percent
                    with dpg.group(horizontal=True):
                        dpg.add_text("End at % of max:", indent=20)
                        dpg.add_input_int(
                            tag=self._tags.lt_end_percent,
                            default_value=1,
                            width=80,
                            min_value=0,
                            max_value=50,
                            min_clamped=True,
                            max_clamped=True,
                        )

                    # Min decay window
                    with dpg.group(horizontal=True):
                        dpg.add_text("Min decay window (ns):", indent=20)
                        dpg.add_input_float(
                            tag=self._tags.lt_min_decay_window,
                            default_value=0.5,
                            width=80,
                            min_value=0.1,
                            max_value=100.0,
                            min_clamped=True,
                            max_clamped=True,
                            format="%.1f",
                        )

                    dpg.add_separator()
                    dpg.add_spacer(height=5)

                    # Background percent
                    with dpg.group(horizontal=True):
                        dpg.add_text("Background % of data:", indent=10)
                        dpg.add_input_int(
                            tag=self._tags.lt_bg_percent,
                            default_value=5,
                            width=80,
                            min_value=1,
                            max_value=50,
                            min_clamped=True,
                            max_clamped=True,
                        )

                # Display tab
                with dpg.tab(label="Display"):
                    dpg.add_spacer(height=10)
                    dpg.add_text(
                        "Display Settings",
                        color=(100, 180, 255),
                    )
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    # Default bin size
                    with dpg.group(horizontal=True):
                        dpg.add_text("Default bin size (ms):", indent=10)
                        dpg.add_input_float(
                            tag=self._tags.disp_default_bin_size,
                            default_value=10.0,
                            width=100,
                            min_value=0.1,
                            max_value=1000.0,
                            min_clamped=True,
                            max_clamped=True,
                            format="%.1f",
                        )

                    dpg.add_spacer(height=10)

                    # Auto resolve levels
                    dpg.add_checkbox(
                        label="Auto-resolve levels on file load",
                        tag=self._tags.disp_auto_resolve,
                        default_value=False,
                        indent=10,
                    )

                    dpg.add_spacer(height=10)
                    dpg.add_text(
                        "Auto-resolve will automatically run change\n"
                        "point analysis when loading a new file.",
                        color=(128, 128, 128),
                        indent=10,
                    )

            dpg.add_spacer(height=15)

            # Buttons row
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Save",
                    tag=self._tags.save_button,
                    callback=self._on_save_clicked,
                    width=100,
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Cancel",
                    tag=self._tags.cancel_button,
                    callback=self._on_cancel_clicked,
                    width=100,
                )
                dpg.add_spacer(width=30)
                dpg.add_button(
                    label="Reset to Defaults",
                    tag=self._tags.reset_button,
                    callback=self._on_reset_clicked,
                    width=130,
                )

        self._is_built = True
        logger.debug("Settings dialog built")

    def show(self) -> None:
        """Show the settings dialog."""
        if not self._is_built:
            self.build()

        # Populate from current settings
        self._populate_from_settings()

        # Center the dialog
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.set_item_pos(
            self._tags.dialog,
            [viewport_width // 2 - 225, viewport_height // 2 - 260],
        )

        dpg.configure_item(self._tags.dialog, show=True)
        logger.debug("Settings dialog shown")

    def hide(self) -> None:
        """Hide the settings dialog."""
        if dpg.does_item_exist(self._tags.dialog):
            dpg.configure_item(self._tags.dialog, show=False)

    def _populate_from_settings(self) -> None:
        """Populate dialog fields from current settings."""
        settings = get_settings()

        # Change point settings
        dpg.set_value(self._tags.cpa_min_photons, settings.change_point.min_photons)
        dpg.set_value(
            self._tags.cpa_min_boundary, settings.change_point.min_boundary_offset
        )

        # Lifetime settings
        dpg.set_value(self._tags.lt_use_moving_avg, settings.lifetime.use_moving_avg)
        dpg.set_value(
            self._tags.lt_moving_avg_window, settings.lifetime.moving_avg_window
        )
        self._update_moving_avg_visibility()
        dpg.set_value(self._tags.lt_start_percent, settings.lifetime.start_percent)
        dpg.set_value(self._tags.lt_end_multiple, settings.lifetime.end_multiple)
        dpg.set_value(self._tags.lt_end_percent, settings.lifetime.end_percent)
        dpg.set_value(
            self._tags.lt_min_decay_window, settings.lifetime.minimum_decay_window
        )
        dpg.set_value(self._tags.lt_bg_percent, settings.lifetime.bg_percent)

        # Display settings
        dpg.set_value(
            self._tags.disp_default_bin_size, settings.display.default_bin_size_ms
        )
        dpg.set_value(self._tags.disp_auto_resolve, settings.display.auto_resolve_levels)

    def _collect_settings(self) -> Settings:
        """Collect settings from dialog fields.

        Returns:
            Settings instance with values from the dialog.
        """
        settings = get_settings()

        # Change point settings
        settings.change_point.min_photons = dpg.get_value(self._tags.cpa_min_photons)
        settings.change_point.min_boundary_offset = dpg.get_value(
            self._tags.cpa_min_boundary
        )

        # Lifetime settings
        settings.lifetime.use_moving_avg = dpg.get_value(self._tags.lt_use_moving_avg)
        settings.lifetime.moving_avg_window = dpg.get_value(
            self._tags.lt_moving_avg_window
        )
        settings.lifetime.start_percent = dpg.get_value(self._tags.lt_start_percent)
        settings.lifetime.end_multiple = dpg.get_value(self._tags.lt_end_multiple)
        settings.lifetime.end_percent = dpg.get_value(self._tags.lt_end_percent)
        settings.lifetime.minimum_decay_window = dpg.get_value(
            self._tags.lt_min_decay_window
        )
        settings.lifetime.bg_percent = dpg.get_value(self._tags.lt_bg_percent)

        # Display settings
        settings.display.default_bin_size_ms = dpg.get_value(
            self._tags.disp_default_bin_size
        )
        settings.display.auto_resolve_levels = dpg.get_value(
            self._tags.disp_auto_resolve
        )

        return settings

    # -------------------------------------------------------------------------
    # UI Event Handlers
    # -------------------------------------------------------------------------

    def _on_moving_avg_changed(self, sender: int, app_data: bool) -> None:
        """Handle moving average checkbox change."""
        self._update_moving_avg_visibility()

    def _update_moving_avg_visibility(self) -> None:
        """Update visibility of moving average window input."""
        use_moving_avg = dpg.get_value(self._tags.lt_use_moving_avg)
        if dpg.does_item_exist(self._tags.lt_moving_avg_row):
            dpg.configure_item(self._tags.lt_moving_avg_row, show=use_moving_avg)

    def _on_save_clicked(self) -> None:
        """Handle Save button click."""
        settings = self._collect_settings()

        # Save to disk
        save_settings()

        self.hide()

        if self._on_save:
            self._on_save(settings)

        logger.info("Settings saved")

    def _on_cancel_clicked(self) -> None:
        """Handle Cancel button click."""
        self.hide()

        if self._on_cancel:
            self._on_cancel()

        logger.debug("Settings dialog cancelled")

    def _on_reset_clicked(self) -> None:
        """Handle Reset to Defaults button click."""
        settings = get_settings()
        settings.reset_to_defaults()

        # Repopulate dialog with defaults
        self._populate_from_settings()

        logger.info("Settings reset to defaults")

    def _on_close(self) -> None:
        """Handle dialog close (X button)."""
        if self._on_cancel:
            self._on_cancel()
