"""Keyboard shortcut handling for Full SMS.

Provides keyboard shortcuts for common operations:
- Cmd/Ctrl+O: Open file
- Cmd/Ctrl+S: Save analysis
- Cmd/Ctrl+E: Export
- Cmd/Ctrl+R: Resolve current
- Tab/Shift+Tab: Navigate tabs
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Platform detection
IS_MAC = sys.platform == "darwin"


@dataclass
class ShortcutHandler:
    """Container for shortcut handlers."""

    on_open: Callable[[], None] | None = None
    on_save: Callable[[], None] | None = None
    on_export: Callable[[], None] | None = None
    on_resolve: Callable[[], None] | None = None
    on_next_tab: Callable[[], None] | None = None
    on_prev_tab: Callable[[], None] | None = None
    on_select_all: Callable[[], None] | None = None
    on_quit: Callable[[], None] | None = None


def _is_modifier_down() -> bool:
    """Check if the platform-specific modifier key (Cmd on Mac, Ctrl elsewhere) is down."""
    if IS_MAC:
        # On macOS, check for Command key (Super)
        return dpg.is_key_down(dpg.mvKey_LSuper) or dpg.is_key_down(dpg.mvKey_RSuper)
    else:
        # On Windows/Linux, check for Ctrl key
        return dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)


def _is_shift_down() -> bool:
    """Check if Shift key is down."""
    return dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)


def _is_ctrl_down() -> bool:
    """Check if Ctrl key is down (for Ctrl+Q on Mac)."""
    return dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)


class KeyboardShortcuts:
    """Manages keyboard shortcuts for the application.

    Uses DearPyGui's handler registry to listen for key presses
    and check modifier keys.
    """

    def __init__(self, handlers: ShortcutHandler | None = None) -> None:
        """Initialize keyboard shortcuts.

        Args:
            handlers: Optional shortcut handler configuration.
        """
        self._handlers = handlers or ShortcutHandler()
        self._registry_tag = "keyboard_handler_registry"
        self._built = False

        # Track key states to prevent repeated firing
        self._key_was_pressed: dict[int, bool] = {}

    def set_handlers(self, handlers: ShortcutHandler) -> None:
        """Set the shortcut handlers.

        Args:
            handlers: The shortcut handler configuration.
        """
        self._handlers = handlers

    def build(self) -> None:
        """Build the keyboard handler registry."""
        if self._built:
            return

        with dpg.handler_registry(tag=self._registry_tag):
            # Cmd/Ctrl+O: Open file
            dpg.add_key_release_handler(
                key=dpg.mvKey_O,
                callback=self._on_key_o,
            )

            # Cmd/Ctrl+S: Save
            dpg.add_key_release_handler(
                key=dpg.mvKey_S,
                callback=self._on_key_s,
            )

            # Cmd/Ctrl+E: Export
            dpg.add_key_release_handler(
                key=dpg.mvKey_E,
                callback=self._on_key_e,
            )

            # Cmd/Ctrl+R: Resolve current
            dpg.add_key_release_handler(
                key=dpg.mvKey_R,
                callback=self._on_key_r,
            )

            # Cmd/Ctrl+A: Select all
            dpg.add_key_release_handler(
                key=dpg.mvKey_A,
                callback=self._on_key_a,
            )

            # Cmd/Ctrl+Q or Cmd+Q: Quit (Mac only via Cmd+Q)
            dpg.add_key_release_handler(
                key=dpg.mvKey_Q,
                callback=self._on_key_q,
            )

            # Tab: Next tab (with optional Shift for previous)
            dpg.add_key_release_handler(
                key=dpg.mvKey_Tab,
                callback=self._on_key_tab,
            )

        self._built = True
        logger.info("Keyboard shortcuts initialized")

    def _on_key_o(self, sender: Any, app_data: Any) -> None:
        """Handle O key release."""
        if _is_modifier_down() and self._handlers.on_open:
            logger.debug("Shortcut: Open file (Cmd/Ctrl+O)")
            self._handlers.on_open()

    def _on_key_s(self, sender: Any, app_data: Any) -> None:
        """Handle S key release."""
        if _is_modifier_down() and self._handlers.on_save:
            logger.debug("Shortcut: Save (Cmd/Ctrl+S)")
            self._handlers.on_save()

    def _on_key_e(self, sender: Any, app_data: Any) -> None:
        """Handle E key release."""
        if _is_modifier_down() and self._handlers.on_export:
            logger.debug("Shortcut: Export (Cmd/Ctrl+E)")
            self._handlers.on_export()

    def _on_key_r(self, sender: Any, app_data: Any) -> None:
        """Handle R key release."""
        if _is_modifier_down() and self._handlers.on_resolve:
            logger.debug("Shortcut: Resolve current (Cmd/Ctrl+R)")
            self._handlers.on_resolve()

    def _on_key_a(self, sender: Any, app_data: Any) -> None:
        """Handle A key release."""
        if _is_modifier_down() and self._handlers.on_select_all:
            logger.debug("Shortcut: Select all (Cmd/Ctrl+A)")
            self._handlers.on_select_all()

    def _on_key_q(self, sender: Any, app_data: Any) -> None:
        """Handle Q key release."""
        # On Mac, Cmd+Q quits. On Windows/Linux, we don't use Ctrl+Q
        # (Alt+F4 is the standard, which is handled by the OS)
        if IS_MAC and _is_modifier_down() and self._handlers.on_quit:
            logger.debug("Shortcut: Quit (Cmd+Q)")
            self._handlers.on_quit()

    def _on_key_tab(self, sender: Any, app_data: Any) -> None:
        """Handle Tab key release."""
        # Only handle Tab when Ctrl is held (to not interfere with normal tab navigation)
        if _is_ctrl_down():
            if _is_shift_down() and self._handlers.on_prev_tab:
                logger.debug("Shortcut: Previous tab (Ctrl+Shift+Tab)")
                self._handlers.on_prev_tab()
            elif self._handlers.on_next_tab:
                logger.debug("Shortcut: Next tab (Ctrl+Tab)")
                self._handlers.on_next_tab()

    def destroy(self) -> None:
        """Clean up the keyboard handler registry."""
        if self._built and dpg.does_item_exist(self._registry_tag):
            dpg.delete_item(self._registry_tag)
            self._built = False
            logger.info("Keyboard shortcuts destroyed")
