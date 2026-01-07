"""Status bar widget for displaying status messages and progress.

Provides a status bar with:
- Status message text
- Progress bar (visible during processing)
- File info display
- Automatic sync with ProcessingState
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg

from full_sms.models.session import ProcessingState

logger = logging.getLogger(__name__)


@dataclass
class StatusBarTags:
    """Tags for status bar elements."""

    container: str = "status_bar_widget_container"
    status_text: str = "status_bar_status_text"
    progress_bar: str = "status_bar_progress"
    file_info: str = "status_bar_file_info"
    task_info: str = "status_bar_task_info"


STATUS_BAR_TAGS = StatusBarTags()


# Colors for different status types
COLOR_NORMAL = (200, 200, 200, 255)
COLOR_BUSY = (100, 180, 255, 255)
COLOR_SUCCESS = (100, 200, 100, 255)
COLOR_ERROR = (255, 100, 100, 255)
COLOR_FILE_INFO = (128, 128, 128, 255)


class StatusBar:
    """Status bar widget for the application.

    Displays status messages, progress indicator, and file information.
    Can be synchronized with ProcessingState for automatic updates.
    """

    def __init__(self, parent: int | str) -> None:
        """Initialize the status bar.

        Args:
            parent: The parent container to build the status bar in.
        """
        self._parent = parent
        self._is_built = False
        self._last_state: ProcessingState | None = None

    def build(self) -> None:
        """Build the status bar UI structure."""
        if self._is_built:
            return

        # Separator above status bar
        dpg.add_separator(parent=self._parent)

        with dpg.group(
            parent=self._parent,
            horizontal=True,
            tag=STATUS_BAR_TAGS.container,
        ):
            # Status message (left side)
            dpg.add_text(
                "Ready",
                tag=STATUS_BAR_TAGS.status_text,
                color=COLOR_NORMAL,
            )

            # Task info (shows current task when busy)
            dpg.add_text(
                "",
                tag=STATUS_BAR_TAGS.task_info,
                color=COLOR_BUSY,
                show=False,
            )

            # Flexible spacer
            dpg.add_spacer(width=-1)

            # Progress bar (hidden by default)
            dpg.add_progress_bar(
                tag=STATUS_BAR_TAGS.progress_bar,
                default_value=0.0,
                width=150,
                show=False,
            )

            # Spacer between progress and file info
            dpg.add_spacer(width=10)

            # File info (right side)
            dpg.add_text(
                "",
                tag=STATUS_BAR_TAGS.file_info,
                color=COLOR_FILE_INFO,
            )

        self._is_built = True
        logger.debug("Status bar built")

    def set_status(self, message: str, color: tuple[int, ...] | None = None) -> None:
        """Update the status message.

        Args:
            message: The status message to display.
            color: Optional color tuple (r, g, b, a). Uses default if None.
        """
        if dpg.does_item_exist(STATUS_BAR_TAGS.status_text):
            dpg.set_value(STATUS_BAR_TAGS.status_text, message)
            if color:
                dpg.configure_item(STATUS_BAR_TAGS.status_text, color=color)

    def set_file_info(self, info: str) -> None:
        """Update the file info display.

        Args:
            info: The file info text to display.
        """
        if dpg.does_item_exist(STATUS_BAR_TAGS.file_info):
            dpg.set_value(STATUS_BAR_TAGS.file_info, info)

    def show_progress(self, value: float = 0.0, task: str = "") -> None:
        """Show the progress bar and optionally task info.

        Args:
            value: Initial progress value (0.0 to 1.0).
            task: Optional task description to display.
        """
        if dpg.does_item_exist(STATUS_BAR_TAGS.progress_bar):
            dpg.configure_item(STATUS_BAR_TAGS.progress_bar, show=True)
            dpg.set_value(STATUS_BAR_TAGS.progress_bar, max(0.0, min(1.0, value)))

        if dpg.does_item_exist(STATUS_BAR_TAGS.task_info):
            if task:
                dpg.set_value(STATUS_BAR_TAGS.task_info, f"  [{task}]")
                dpg.configure_item(STATUS_BAR_TAGS.task_info, show=True)
            else:
                dpg.configure_item(STATUS_BAR_TAGS.task_info, show=False)

    def update_progress(self, value: float, message: str = "") -> None:
        """Update the progress bar value and optionally the status message.

        Args:
            value: Progress value (0.0 to 1.0).
            message: Optional status message to display.
        """
        if dpg.does_item_exist(STATUS_BAR_TAGS.progress_bar):
            dpg.set_value(STATUS_BAR_TAGS.progress_bar, max(0.0, min(1.0, value)))

        if message:
            self.set_status(message, COLOR_BUSY)

    def hide_progress(self) -> None:
        """Hide the progress bar and task info."""
        if dpg.does_item_exist(STATUS_BAR_TAGS.progress_bar):
            dpg.configure_item(STATUS_BAR_TAGS.progress_bar, show=False)
        if dpg.does_item_exist(STATUS_BAR_TAGS.task_info):
            dpg.configure_item(STATUS_BAR_TAGS.task_info, show=False)

    def sync_with_state(self, state: ProcessingState) -> None:
        """Synchronize the status bar with a ProcessingState.

        This method efficiently updates only the elements that have changed.
        Call this periodically (e.g., each frame) to keep the status bar in sync.

        Args:
            state: The ProcessingState to sync with.
        """
        # Check if state has changed to avoid unnecessary updates
        if self._last_state is not None:
            if (
                self._last_state.is_busy == state.is_busy
                and self._last_state.progress == state.progress
                and self._last_state.message == state.message
                and self._last_state.current_task == state.current_task
            ):
                return

        # Update UI based on state
        if state.is_busy:
            self.set_status(state.message or "Processing...", COLOR_BUSY)
            self.show_progress(state.progress, state.current_task)
        else:
            if state.message:
                self.set_status(state.message, COLOR_SUCCESS)
            else:
                self.set_status("Ready", COLOR_NORMAL)
            self.hide_progress()

        # Cache state for comparison
        self._last_state = ProcessingState(
            is_busy=state.is_busy,
            progress=state.progress,
            message=state.message,
            current_task=state.current_task,
        )

    def show_error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: The error message to display.
        """
        self.set_status(message, COLOR_ERROR)
        self.hide_progress()

    def show_success(self, message: str) -> None:
        """Display a success message.

        Args:
            message: The success message to display.
        """
        self.set_status(message, COLOR_SUCCESS)
        self.hide_progress()

    def reset(self) -> None:
        """Reset the status bar to its default state."""
        self.set_status("Ready", COLOR_NORMAL)
        self.set_file_info("")
        self.hide_progress()
        self._last_state = None
