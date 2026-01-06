"""Full SMS application entry point.

Single Molecule Spectroscopy analysis application using DearPyGui.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg

from full_sms.ui.theme import APP_VERSION, create_plot_theme, create_theme

if TYPE_CHECKING:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Tag constants for UI elements
TAGS = {
    "primary_window": "primary_window",
    "status_bar": "status_bar",
    "status_text": "status_text",
    "content_area": "content_area",
}


class Application:
    """Main application class for Full SMS."""

    def __init__(self) -> None:
        """Initialize the application."""
        self._running = False
        self._theme: int | None = None
        self._plot_theme: int | None = None

    def setup(self) -> None:
        """Set up the DearPyGui context, viewport, and UI."""
        logger.info("Initializing Full SMS application")

        # Create DearPyGui context
        dpg.create_context()

        # Create viewport (the OS window)
        dpg.create_viewport(
            title=f"Full SMS v{APP_VERSION}",
            width=1440,
            height=900,
            min_width=800,
            min_height=600,
        )

        # Set up themes
        self._theme = create_theme()
        self._plot_theme = create_plot_theme()
        dpg.bind_theme(self._theme)

        # Create the main UI
        self._create_main_window()

        # Set up DearPyGui
        dpg.setup_dearpygui()

        logger.info("Application setup complete")

    def _create_main_window(self) -> None:
        """Create the main application window with menu bar."""
        with dpg.window(tag=TAGS["primary_window"]):
            # Menu bar
            self._create_menu_bar()

            # Main content area (placeholder for now)
            with dpg.child_window(tag=TAGS["content_area"], border=False):
                dpg.add_text("Full SMS - Single Molecule Spectroscopy Analysis")
                dpg.add_separator()
                dpg.add_text(
                    "Welcome to Full SMS. Use File > Open H5 to load a data file.",
                    color=(180, 180, 180),
                )
                dpg.add_spacer(height=20)
                dpg.add_text(f"Version: {APP_VERSION}", color=(128, 128, 128))

            # Status bar at the bottom
            self._create_status_bar()

        # Set as primary window (fills viewport)
        dpg.set_primary_window(TAGS["primary_window"], True)

    def _create_menu_bar(self) -> None:
        """Create the application menu bar."""
        with dpg.menu_bar():
            # File menu
            with dpg.menu(label="File"):
                dpg.add_menu_item(
                    label="Open H5...",
                    callback=self._on_open_h5,
                    shortcut="Cmd+O" if sys.platform == "darwin" else "Ctrl+O",
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="Load Session...",
                    callback=self._on_load_session,
                )
                dpg.add_menu_item(
                    label="Save Session",
                    callback=self._on_save_session,
                    shortcut="Cmd+S" if sys.platform == "darwin" else "Ctrl+S",
                    enabled=False,  # Disabled until data is loaded
                    tag="menu_save_session",
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="Close File",
                    callback=self._on_close_file,
                    enabled=False,
                    tag="menu_close_file",
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="Exit",
                    callback=self._on_exit,
                    shortcut="Cmd+Q" if sys.platform == "darwin" else "Alt+F4",
                )

            # Edit menu
            with dpg.menu(label="Edit"):
                dpg.add_menu_item(
                    label="Select All",
                    callback=self._on_select_all,
                    shortcut="Cmd+A" if sys.platform == "darwin" else "Ctrl+A",
                    enabled=False,
                    tag="menu_select_all",
                )
                dpg.add_menu_item(
                    label="Deselect All",
                    callback=self._on_deselect_all,
                    enabled=False,
                    tag="menu_deselect_all",
                )
                dpg.add_menu_item(
                    label="Invert Selection",
                    callback=self._on_invert_selection,
                    enabled=False,
                    tag="menu_invert_selection",
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="Settings...",
                    callback=self._on_settings,
                )

            # Analysis menu (disabled until data loaded)
            with dpg.menu(label="Analysis", tag="menu_analysis", enabled=False):
                with dpg.menu(label="Intensity"):
                    dpg.add_menu_item(
                        label="Resolve Current",
                        callback=lambda: self._on_resolve("current"),
                    )
                    dpg.add_menu_item(
                        label="Resolve Selected",
                        callback=lambda: self._on_resolve("selected"),
                    )
                    dpg.add_menu_item(
                        label="Resolve All",
                        callback=lambda: self._on_resolve("all"),
                    )
                with dpg.menu(label="Grouping"):
                    dpg.add_menu_item(
                        label="Group Current",
                        callback=lambda: self._on_group("current"),
                    )
                    dpg.add_menu_item(
                        label="Group Selected",
                        callback=lambda: self._on_group("selected"),
                    )
                    dpg.add_menu_item(
                        label="Group All",
                        callback=lambda: self._on_group("all"),
                    )
                with dpg.menu(label="Lifetime"):
                    dpg.add_menu_item(
                        label="Fit Current",
                        callback=lambda: self._on_fit("current"),
                    )
                    dpg.add_menu_item(
                        label="Fit Selected",
                        callback=lambda: self._on_fit("selected"),
                    )
                    dpg.add_menu_item(
                        label="Fit All",
                        callback=lambda: self._on_fit("all"),
                    )

            # Help menu
            with dpg.menu(label="Help"):
                dpg.add_menu_item(
                    label="Documentation",
                    callback=self._on_documentation,
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="About Full SMS",
                    callback=self._on_about,
                )

    def _create_status_bar(self) -> None:
        """Create the status bar at the bottom of the window."""
        dpg.add_separator()
        with dpg.group(horizontal=True, tag=TAGS["status_bar"]):
            dpg.add_text("Ready", tag=TAGS["status_text"])

    def set_status(self, message: str) -> None:
        """Update the status bar message.

        Args:
            message: The status message to display.
        """
        if dpg.does_item_exist(TAGS["status_text"]):
            dpg.set_value(TAGS["status_text"], message)

    # Menu callbacks - File menu

    def _on_open_h5(self) -> None:
        """Handle Open H5 menu action."""
        logger.info("Open H5 triggered")
        self.set_status("Opening file...")
        # File dialog will be implemented in Task 6.3/12.3

    def _on_load_session(self) -> None:
        """Handle Load Session menu action."""
        logger.info("Load Session triggered")
        self.set_status("Loading session...")

    def _on_save_session(self) -> None:
        """Handle Save Session menu action."""
        logger.info("Save Session triggered")
        self.set_status("Saving session...")

    def _on_close_file(self) -> None:
        """Handle Close File menu action."""
        logger.info("Close File triggered")
        self.set_status("File closed")

    def _on_exit(self) -> None:
        """Handle Exit menu action."""
        logger.info("Exit triggered")
        dpg.stop_dearpygui()

    # Menu callbacks - Edit menu

    def _on_select_all(self) -> None:
        """Handle Select All menu action."""
        logger.info("Select All triggered")

    def _on_deselect_all(self) -> None:
        """Handle Deselect All menu action."""
        logger.info("Deselect All triggered")

    def _on_invert_selection(self) -> None:
        """Handle Invert Selection menu action."""
        logger.info("Invert Selection triggered")

    def _on_settings(self) -> None:
        """Handle Settings menu action."""
        logger.info("Settings triggered")

    # Menu callbacks - Analysis menu

    def _on_resolve(self, mode: str) -> None:
        """Handle Resolve menu action."""
        logger.info(f"Resolve {mode} triggered")

    def _on_group(self, mode: str) -> None:
        """Handle Group menu action."""
        logger.info(f"Group {mode} triggered")

    def _on_fit(self, mode: str) -> None:
        """Handle Fit menu action."""
        logger.info(f"Fit {mode} triggered")

    # Menu callbacks - Help menu

    def _on_documentation(self) -> None:
        """Handle Documentation menu action."""
        logger.info("Documentation triggered")
        import webbrowser

        webbrowser.open("https://up-biophysics-sms.readthedocs.io/en/latest/")

    def _on_about(self) -> None:
        """Handle About menu action."""
        logger.info("About triggered")
        self._show_about_dialog()

    def _show_about_dialog(self) -> None:
        """Show the About dialog."""
        # Check if dialog already exists
        if dpg.does_item_exist("about_dialog"):
            dpg.delete_item("about_dialog")

        with dpg.window(
            label="About Full SMS",
            modal=True,
            tag="about_dialog",
            width=400,
            height=250,
            pos=(
                dpg.get_viewport_width() // 2 - 200,
                dpg.get_viewport_height() // 2 - 125,
            ),
            no_resize=True,
            no_move=False,
        ):
            dpg.add_text("Full SMS", color=(100, 180, 255))
            dpg.add_text(f"Version {APP_VERSION}")
            dpg.add_spacer(height=10)
            dpg.add_text("Single Molecule Spectroscopy Analysis")
            dpg.add_spacer(height=10)
            dpg.add_text("Biophysics Group", color=(180, 180, 180))
            dpg.add_text("University of Pretoria", color=(180, 180, 180))
            dpg.add_spacer(height=10)
            dpg.add_text(
                "Developed by Bertus van Heerden and Joshua Botha",
                color=(128, 128, 128),
            )
            dpg.add_spacer(height=20)
            dpg.add_button(
                label="Close",
                width=100,
                callback=lambda: dpg.delete_item("about_dialog"),
            )

    def run(self) -> None:
        """Run the main application loop."""
        logger.info("Starting application main loop")
        self._running = True

        # Show the viewport
        dpg.show_viewport()

        # Main render loop with frame callback support
        while dpg.is_dearpygui_running():
            # Frame callback placeholder - will be used for async operations
            self._on_frame()

            # Render frame
            dpg.render_dearpygui_frame()

        self._running = False
        logger.info("Application main loop ended")

    def _on_frame(self) -> None:
        """Called every frame - use for async operations and updates."""
        # Placeholder for future async task checking
        pass

    def shutdown(self) -> None:
        """Clean up and destroy the DearPyGui context."""
        logger.info("Shutting down application")
        dpg.destroy_context()
        logger.info("Application shutdown complete")


def main() -> None:
    """Run the Full SMS application."""
    app = Application()

    try:
        app.setup()
        app.run()
    except Exception as e:
        logger.exception(f"Application error: {e}")
        raise
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
