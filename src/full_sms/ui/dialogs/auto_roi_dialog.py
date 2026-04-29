"""Auto-ROI trim dialog for detecting bleaching and trimming ROI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import dearpygui.dearpygui as dpg

logger = logging.getLogger(__name__)

_DIALOG_TAG = "auto_roi_dialog"


class AutoROIScope(Enum):
    """Scope for auto-ROI trimming."""

    CURRENT = "current"
    SELECTED = "selected"
    ALL = "all"


@dataclass
class AutoROIParameters:
    """Parameters for auto-ROI trimming."""

    threshold_cps: float = 1000.0
    min_duration_s: float = 1.0
    scope: AutoROIScope = AutoROIScope.CURRENT


class AutoROIDialog:
    """Modal dialog for auto-ROI trim parameters.

    Allows the user to set threshold, minimum duration, and scope
    for automatic bleaching detection and ROI trimming.
    """

    def __init__(self) -> None:
        self._on_accept: Optional[Callable[[AutoROIParameters], None]] = None
        self._is_built = False

    def build(self) -> None:
        """Build the dialog (hidden initially)."""
        self._is_built = True

    def set_on_accept(self, callback: Callable[[AutoROIParameters], None]) -> None:
        """Set callback for when the user accepts the dialog.

        Args:
            callback: Function called with AutoROIParameters.
        """
        self._on_accept = callback

    def show(self) -> None:
        """Show the auto-ROI dialog."""
        # Clean up any existing dialog
        if dpg.does_item_exist(_DIALOG_TAG):
            dpg.delete_item(_DIALOG_TAG)

        dialog_width = 350
        dialog_height = 200

        with dpg.window(
            label="Auto-Trim ROI",
            tag=_DIALOG_TAG,
            modal=True,
            no_close=True,
            no_resize=True,
            no_move=True,
            width=dialog_width,
            height=dialog_height,
        ):
            dpg.add_text(
                "Detect trailing bleached regions and trim ROI.",
                wrap=dialog_width - 40,
                color=(180, 180, 180),
            )
            dpg.add_spacer(height=8)

            with dpg.group(horizontal=True):
                dpg.add_text("Threshold (c/s):")
                dpg.add_input_float(
                    default_value=1000.0,
                    tag=f"{_DIALOG_TAG}_threshold",
                    width=120,
                    format="%.1f",
                    step=0,
                    min_value=0.0,
                    min_clamped=True,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("Min Duration (s):")
                dpg.add_input_float(
                    default_value=1.0,
                    tag=f"{_DIALOG_TAG}_duration",
                    width=120,
                    format="%.1f",
                    step=0,
                    min_value=0.01,
                    min_clamped=True,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("Scope:")
                dpg.add_combo(
                    items=["Current", "Selected", "All"],
                    default_value="Current",
                    tag=f"{_DIALOG_TAG}_scope",
                    width=120,
                )

            dpg.add_spacer(height=10)

            button_total = 180
            button_indent = (dialog_width - button_total) // 2
            with dpg.group(horizontal=True, indent=button_indent):
                dpg.add_button(
                    label="Apply",
                    callback=self._on_apply,
                    width=80,
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Cancel",
                    callback=self._on_cancel,
                    width=80,
                )

        # Center the dialog
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.set_item_pos(
            _DIALOG_TAG,
            [
                viewport_width // 2 - dialog_width // 2,
                viewport_height // 2 - dialog_height // 2,
            ],
        )

    def _on_apply(self) -> None:
        """Handle Apply button click."""
        threshold = dpg.get_value(f"{_DIALOG_TAG}_threshold")
        duration = dpg.get_value(f"{_DIALOG_TAG}_duration")
        scope_str = dpg.get_value(f"{_DIALOG_TAG}_scope")

        scope_map = {
            "Current": AutoROIScope.CURRENT,
            "Selected": AutoROIScope.SELECTED,
            "All": AutoROIScope.ALL,
        }
        scope = scope_map.get(scope_str, AutoROIScope.CURRENT)

        params = AutoROIParameters(
            threshold_cps=threshold,
            min_duration_s=duration,
            scope=scope,
        )

        self._cleanup()
        if self._on_accept:
            self._on_accept(params)

    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove the dialog."""
        if dpg.does_item_exist(_DIALOG_TAG):
            dpg.delete_item(_DIALOG_TAG)
