"""Dialog shown when matching session files are found for an HDF5 file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import dearpygui.dearpygui as dpg

logger = logging.getLogger(__name__)

# Dialog tag
_DIALOG_TAG = "session_found_dialog"


class SessionFoundDialog:
    """Modal dialog to prompt user when matching sessions are found.

    Shows either a single-match or multi-match message and lets the user
    choose to load a session or skip.

    Args:
        session_paths: List of matching .smsa file paths.
        on_load: Callback with the chosen session path.
        on_skip: Callback when user chooses to open without session.
    """

    def __init__(
        self,
        session_paths: List[Path],
        on_load: Callable[[Path], None],
        on_skip: Callable[[], None],
    ) -> None:
        self._session_paths = session_paths
        self._on_load = on_load
        self._on_skip = on_skip

    def show(self) -> None:
        """Show the dialog."""
        # Clean up any existing dialog
        if dpg.does_item_exist(_DIALOG_TAG):
            dpg.delete_item(_DIALOG_TAG)

        dialog_width = 450
        dialog_height = 150
        button_total = 310  # 140 + 10 spacing + 160
        button_indent = (dialog_width - button_total) // 2

        with dpg.window(
            label="Session Found",
            tag=_DIALOG_TAG,
            modal=True,
            no_close=True,
            no_resize=True,
            no_move=True,
            width=dialog_width,
            height=dialog_height,
        ):
            if len(self._session_paths) == 1:
                name = self._session_paths[0].name
                dpg.add_text(
                    f"A matching analysis session was found:\n{name}\n\n"
                    "Would you like to load it?",
                    wrap=dialog_width - 40,
                )
                dpg.add_spacer(height=5)
                with dpg.group(horizontal=True, indent=button_indent):
                    dpg.add_button(
                        label="Load Session",
                        callback=self._load_single,
                        width=140,
                    )
                    dpg.add_button(
                        label="Open Without Session",
                        callback=self._skip,
                        width=160,
                    )
            else:
                dpg.add_text(
                    f"{len(self._session_paths)} matching analysis sessions "
                    "were found.\nWould you like to load one?",
                    wrap=dialog_width - 40,
                )
                dpg.add_spacer(height=5)
                combo_width = 300
                combo_indent = (dialog_width - combo_width) // 2
                dpg.add_combo(
                    items=[p.name for p in self._session_paths],
                    default_value=self._session_paths[0].name,
                    tag=f"{_DIALOG_TAG}_combo",
                    width=combo_width,
                    indent=combo_indent,
                )
                dpg.add_spacer(height=5)
                with dpg.group(horizontal=True, indent=button_indent):
                    dpg.add_button(
                        label="Load Session",
                        callback=self._load_selected,
                        width=140,
                    )
                    dpg.add_button(
                        label="Open Without Session",
                        callback=self._skip,
                        width=160,
                    )

        # Center the dialog in the viewport
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.set_item_pos(
            _DIALOG_TAG,
            [viewport_width // 2 - dialog_width // 2,
             viewport_height // 2 - dialog_height // 2],
        )

    def _load_single(self) -> None:
        """Load the single matching session."""
        self._cleanup()
        self._on_load(self._session_paths[0])

    def _load_selected(self) -> None:
        """Load the session selected in the combo."""
        selected_name = dpg.get_value(f"{_DIALOG_TAG}_combo")
        for p in self._session_paths:
            if p.name == selected_name:
                self._cleanup()
                self._on_load(p)
                return
        # Fallback to first
        self._cleanup()
        self._on_load(self._session_paths[0])

    def _skip(self) -> None:
        """Skip loading a session."""
        self._cleanup()
        self._on_skip()

    def _cleanup(self) -> None:
        """Remove the dialog."""
        if dpg.does_item_exist(_DIALOG_TAG):
            dpg.delete_item(_DIALOG_TAG)
