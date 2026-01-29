"""File dialogs for opening, saving, and loading files.

Provides native file dialogs using DearPyGui's file_dialog for:
- Open HDF5 file
- Save/Load analysis sessions (.smsa)
- Export directory selection

File dialog paths are persisted in application settings so dialogs
remember the last used location between sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import dearpygui.dearpygui as dpg

from full_sms.config import get_settings, save_settings

logger = logging.getLogger(__name__)

# File extension constants
HDF5_EXTENSIONS = (".h5", ".hdf5", ".smsh5")
SESSION_EXTENSION = ".smsa"

# Default paths
DEFAULT_OPEN_PATH = Path.home() / "Documents"
DEFAULT_SAVE_PATH = Path.home() / "Desktop"


@dataclass
class FileDialogTags:
    """Tags for file dialog elements."""

    open_h5_dialog: str = "file_dialog_open_h5"
    save_session_dialog: str = "file_dialog_save_session"
    load_session_dialog: str = "file_dialog_load_session"


class FileDialogs:
    """Manager for file dialogs.

    Provides methods to show file dialogs for various operations.
    Each dialog is created on demand and destroyed after use.

    Usage:
        dialogs = FileDialogs()
        dialogs.set_on_open_h5(callback)
        dialogs.show_open_h5_dialog()
    """

    def __init__(self, tag_prefix: str = "") -> None:
        """Initialize file dialogs manager.

        Args:
            tag_prefix: Optional prefix for unique tags.
        """
        self._tag_prefix = tag_prefix

        # Callbacks
        self._on_open_h5: Optional[Callable[[Path], None]] = None
        self._on_save_session: Optional[Callable[[Path], None]] = None
        self._on_load_session: Optional[Callable[[Path], None]] = None

        # Load paths from persistent settings
        settings = get_settings()
        fd_settings = settings.file_dialogs

        # Initialize paths from settings, falling back to defaults
        self._last_open_path: Path = (
            Path(fd_settings.last_open_directory)
            if fd_settings.last_open_directory
            else DEFAULT_OPEN_PATH
        )
        self._last_save_path: Path = DEFAULT_SAVE_PATH
        self._last_session_path: Path = (
            Path(fd_settings.last_session_directory)
            if fd_settings.last_session_directory
            else DEFAULT_SAVE_PATH
        )

        # Current open file path (for suggesting save location)
        self._current_file_path: Optional[Path] = None

        # Generate unique tags
        self._tags = FileDialogTags(
            open_h5_dialog=f"{tag_prefix}file_dialog_open_h5",
            save_session_dialog=f"{tag_prefix}file_dialog_save_session",
            load_session_dialog=f"{tag_prefix}file_dialog_load_session",
        )

    @property
    def tags(self) -> FileDialogTags:
        """Get the dialog tags."""
        return self._tags

    def _save_paths_to_settings(self) -> None:
        """Save the current paths to persistent settings."""
        settings = get_settings()
        settings.file_dialogs.last_open_directory = str(self._last_open_path)
        settings.file_dialogs.last_session_directory = str(self._last_session_path)
        save_settings()

    @property
    def last_open_path(self) -> Path:
        """Get the last path used for opening files."""
        return self._last_open_path

    def set_current_file_path(self, path: Optional[Path]) -> None:
        """Set the current open file path.

        Used to suggest save locations based on the open file.

        Args:
            path: Path to the currently open file, or None.
        """
        self._current_file_path = path
        if path is not None:
            self._last_open_path = path.parent
            self._last_session_path = path.parent

    # -------------------------------------------------------------------------
    # Callback setters
    # -------------------------------------------------------------------------

    def set_on_open_h5(self, callback: Callable[[Path], None]) -> None:
        """Set callback for when an HDF5 file is selected.

        Args:
            callback: Function called with the selected file path.
        """
        self._on_open_h5 = callback

    def set_on_save_session(self, callback: Callable[[Path], None]) -> None:
        """Set callback for when a session save path is selected.

        Args:
            callback: Function called with the save path.
        """
        self._on_save_session = callback

    def set_on_load_session(self, callback: Callable[[Path], None]) -> None:
        """Set callback for when a session file is selected for loading.

        Args:
            callback: Function called with the session file path.
        """
        self._on_load_session = callback

    # -------------------------------------------------------------------------
    # Open HDF5 Dialog
    # -------------------------------------------------------------------------

    def show_open_h5_dialog(self, default_path: Optional[Path] = None) -> None:
        """Show the Open HDF5 file dialog.

        Args:
            default_path: Optional starting directory.
        """
        # Clean up any existing dialog
        if dpg.does_item_exist(self._tags.open_h5_dialog):
            dpg.delete_item(self._tags.open_h5_dialog)

        start_path = default_path or self._last_open_path
        if not start_path.exists():
            start_path = Path.home()

        def file_selected(sender, app_data):
            """Handle file selection."""
            self._handle_open_h5_selection(app_data)
            dpg.delete_item(sender)

        def cancel_callback(sender, app_data):
            """Handle cancel."""
            dpg.delete_item(sender)

        # Create file dialog with HDF5 filter
        with dpg.file_dialog(
            label="Open HDF5 File",
            tag=self._tags.open_h5_dialog,
            directory_selector=False,
            default_path=str(start_path),
            callback=file_selected,
            cancel_callback=cancel_callback,
            width=700,
            height=450,
            modal=True,
            show=True,
        ):
            # Add file type filter for HDF5 files
            dpg.add_file_extension(
                ".h5",
                color=(0, 255, 100, 255),
                custom_text="[HDF5]",
            )
            dpg.add_file_extension(
                ".hdf5",
                color=(0, 255, 100, 255),
                custom_text="[HDF5]",
            )
            dpg.add_file_extension(
                ".smsh5",
                color=(0, 255, 100, 255),
                custom_text="[SMS]",
            )
            # Also show all files option
            dpg.add_file_extension(".*", color=(255, 255, 255, 150))

        logger.debug("Open HDF5 dialog shown")

    def _handle_open_h5_selection(self, app_data: dict) -> None:
        """Handle HDF5 file selection from dialog.

        Args:
            app_data: Data from the file dialog callback.
        """
        if not app_data:
            return

        file_path_name = app_data.get("file_path_name", "")
        if not file_path_name:
            return

        path = Path(file_path_name)

        # Validate file extension
        if path.suffix.lower() not in HDF5_EXTENSIONS:
            logger.warning(f"Selected file is not an HDF5 file: {path}")
            # Still allow it - user may know what they're doing

        # Update last path and save to persistent settings
        self._last_open_path = path.parent
        self._save_paths_to_settings()

        logger.info(f"HDF5 file selected: {path}")

        if self._on_open_h5:
            self._on_open_h5(path)

    # -------------------------------------------------------------------------
    # Save Session Dialog
    # -------------------------------------------------------------------------

    def show_save_session_dialog(
        self,
        default_path: Optional[Path] = None,
        default_filename: Optional[str] = None,
    ) -> None:
        """Show the Save Session dialog.

        Args:
            default_path: Optional starting directory.
            default_filename: Optional default filename (without extension).
        """
        # Clean up any existing dialog
        if dpg.does_item_exist(self._tags.save_session_dialog):
            dpg.delete_item(self._tags.save_session_dialog)

        start_path = default_path or self._last_session_path
        if not start_path.exists():
            start_path = Path.home()

        # Generate default filename from current file if available
        if default_filename is None and self._current_file_path is not None:
            default_filename = self._current_file_path.stem

        def file_selected(sender, app_data):
            """Handle file selection."""
            self._handle_save_session_selection(app_data)
            dpg.delete_item(sender)

        def cancel_callback(sender, app_data):
            """Handle cancel."""
            dpg.delete_item(sender)

        # Create file dialog for saving
        with dpg.file_dialog(
            label="Save Session",
            tag=self._tags.save_session_dialog,
            directory_selector=False,
            default_path=str(start_path),
            default_filename=default_filename or "analysis",
            callback=file_selected,
            cancel_callback=cancel_callback,
            width=700,
            height=450,
            modal=True,
            show=True,
        ):
            # Add session file extension filter
            dpg.add_file_extension(
                SESSION_EXTENSION,
                color=(100, 180, 255, 255),
                custom_text="[Session]",
            )

        logger.debug("Save session dialog shown")

    def _handle_save_session_selection(self, app_data: dict) -> None:
        """Handle session save path selection from dialog.

        Args:
            app_data: Data from the file dialog callback.
        """
        if not app_data:
            return

        file_path_name = app_data.get("file_path_name", "")
        if not file_path_name:
            return

        path = Path(file_path_name)

        # Ensure correct extension
        if path.suffix.lower() != SESSION_EXTENSION:
            path = path.with_suffix(SESSION_EXTENSION)

        # Update last path and save to persistent settings
        self._last_session_path = path.parent
        self._save_paths_to_settings()

        logger.info(f"Session save path selected: {path}")

        if self._on_save_session:
            self._on_save_session(path)

    # -------------------------------------------------------------------------
    # Load Session Dialog
    # -------------------------------------------------------------------------

    def show_load_session_dialog(self, default_path: Optional[Path] = None) -> None:
        """Show the Load Session dialog.

        Args:
            default_path: Optional starting directory.
        """
        # Clean up any existing dialog
        if dpg.does_item_exist(self._tags.load_session_dialog):
            dpg.delete_item(self._tags.load_session_dialog)

        start_path = default_path or self._last_session_path
        if not start_path.exists():
            start_path = Path.home()

        def file_selected(sender, app_data):
            """Handle file selection."""
            self._handle_load_session_selection(app_data)
            dpg.delete_item(sender)

        def cancel_callback(sender, app_data):
            """Handle cancel."""
            dpg.delete_item(sender)

        # Create file dialog for loading
        with dpg.file_dialog(
            label="Load Session",
            tag=self._tags.load_session_dialog,
            directory_selector=False,
            default_path=str(start_path),
            callback=file_selected,
            cancel_callback=cancel_callback,
            width=700,
            height=450,
            modal=True,
            show=True,
        ):
            # Add session file extension filter
            dpg.add_file_extension(
                SESSION_EXTENSION,
                color=(100, 180, 255, 255),
                custom_text="[Session]",
            )
            # Also show all files option
            dpg.add_file_extension(".*", color=(255, 255, 255, 150))

        logger.debug("Load session dialog shown")

    def _handle_load_session_selection(self, app_data: dict) -> None:
        """Handle session file selection from dialog.

        Args:
            app_data: Data from the file dialog callback.
        """
        if not app_data:
            return

        file_path_name = app_data.get("file_path_name", "")
        if not file_path_name:
            return

        path = Path(file_path_name)

        # Validate file extension
        if path.suffix.lower() != SESSION_EXTENSION:
            logger.warning(f"Selected file is not a session file: {path}")
            # Still allow it - user may know what they're doing

        # Update last path and save to persistent settings
        self._last_session_path = path.parent
        self._save_paths_to_settings()

        logger.info(f"Session file selected: {path}")

        if self._on_load_session:
            self._on_load_session(path)

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def destroy(self) -> None:
        """Clean up all dialogs."""
        for tag in [
            self._tags.open_h5_dialog,
            self._tags.save_session_dialog,
            self._tags.load_session_dialog,
        ]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        logger.debug("File dialogs destroyed")
