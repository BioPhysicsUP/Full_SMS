"""Full SMS application entry point.

Single Molecule Spectroscopy analysis application using DearPyGui.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import Future
from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg
import numpy as np

from full_sms.models.fit import FitResult
from full_sms.models.level import LevelData
from full_sms.models.session import (
    ActiveTab,
    ChannelSelection,
    ConfidenceLevel,
    SessionState,
)
from full_sms.config import Settings, get_settings, save_settings
from full_sms.ui.dialogs import FittingDialog, FittingParameters, SettingsDialog
from full_sms.ui.keyboard import KeyboardShortcuts, ShortcutHandler
from full_sms.ui.layout import MainLayout
from full_sms.ui.theme import APP_VERSION, create_plot_theme, create_theme
from full_sms.workers.pool import AnalysisPool, TaskResult
from full_sms.workers.tasks import (
    run_clustering_task,
    run_correlation_task,
    run_cpa_task,
    run_fit_task,
)

if TYPE_CHECKING:
    from full_sms.models.particle import ParticleData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Tag constants for UI elements
TAGS = {
    "primary_window": "primary_window",
}


class Application:
    """Main application class for Full SMS."""

    def __init__(self) -> None:
        """Initialize the application."""
        self._running = False
        self._theme: int | None = None
        self._plot_theme: int | None = None
        self._layout: MainLayout | None = None
        self._session = SessionState()

        # Analysis worker pool
        self._pool: AnalysisPool | None = None

        # Pending futures for async operations
        self._pending_futures: list[Future] = []
        self._resolve_mode: str | None = None  # Track current resolve operation
        self._grouping_mode: str | None = None  # Track current grouping operation
        self._correlation_pending: bool = False  # Track correlation operation

        # Fitting dialog and parameters
        self._fitting_dialog: FittingDialog | None = None
        self._fitting_params = FittingParameters()

        # Settings dialog
        self._settings_dialog: SettingsDialog | None = None

        # Keyboard shortcuts
        self._keyboard: KeyboardShortcuts | None = None

    def setup(self) -> None:
        """Set up the DearPyGui context, viewport, and UI."""
        logger.info("Initializing Full SMS application")

        # Initialize worker pool
        self._pool = AnalysisPool()
        logger.info(f"Analysis pool initialized with {self._pool.max_workers} workers")

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

        # Set up keyboard shortcuts
        self._setup_keyboard_shortcuts()

        # Set up DearPyGui
        dpg.setup_dearpygui()

        logger.info("Application setup complete")

    def _create_main_window(self) -> None:
        """Create the main application window with menu bar."""
        with dpg.window(tag=TAGS["primary_window"]):
            # Menu bar
            self._create_menu_bar()

            # Main layout with sidebar, tabs, and status bar
            self._layout = MainLayout(parent=TAGS["primary_window"])
            self._layout.build()

            # Set up callbacks
            self._layout.set_on_tab_change(self._on_tab_changed)
            self._layout.set_on_selection_change(self._on_selection_changed)
            self._layout.set_on_batch_change(self._on_batch_changed)

            # Set up intensity tab resolve callback
            if self._layout.intensity_tab:
                self._layout.intensity_tab.set_on_resolve(self._on_resolve_from_tab)

            # Set up lifetime tab fit callback
            if self._layout.lifetime_tab:
                self._layout.lifetime_tab.set_on_fit_requested(
                    self._on_fit_requested_from_tab
                )

            # Set up grouping tab callback
            if self._layout.grouping_tab:
                self._layout.grouping_tab.set_on_grouping_requested(
                    self._on_grouping_from_tab
                )

            # Set up correlation tab callback
            if self._layout.correlation_tab:
                self._layout.correlation_tab.set_on_correlate(
                    self._on_correlate_from_tab
                )

            # Create fitting dialog (must be after DearPyGui context is created)
            self._fitting_dialog = FittingDialog()
            self._fitting_dialog.build()
            self._fitting_dialog.set_on_fit(self._on_fit_dialog_accepted)

            # Create settings dialog
            self._settings_dialog = SettingsDialog()
            self._settings_dialog.build()
            self._settings_dialog.set_on_save(self._on_settings_saved)

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

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up global keyboard shortcuts."""
        handlers = ShortcutHandler(
            on_open=self._on_open_h5,
            on_save=self._on_save_session,
            on_export=self._on_export_shortcut,
            on_resolve=self._on_resolve_current_shortcut,
            on_next_tab=self._on_next_tab,
            on_prev_tab=self._on_prev_tab,
            on_select_all=self._on_select_all,
            on_quit=self._on_exit,
        )
        self._keyboard = KeyboardShortcuts(handlers)
        self._keyboard.build()

    def _on_export_shortcut(self) -> None:
        """Handle Cmd/Ctrl+E shortcut to switch to export tab."""
        logger.info("Export shortcut triggered")
        if self._layout:
            from full_sms.models.session import ActiveTab
            self._layout.set_active_tab(ActiveTab.EXPORT)

    def _on_resolve_current_shortcut(self) -> None:
        """Handle Cmd/Ctrl+R shortcut to resolve current particle."""
        logger.info("Resolve current shortcut triggered")
        if self._session.current_selection:
            self._on_resolve("current")
        else:
            self.set_status("Select a particle to resolve")

    def _on_next_tab(self) -> None:
        """Handle Ctrl+Tab shortcut to navigate to next tab."""
        if self._layout:
            self._layout.next_tab()

    def _on_prev_tab(self) -> None:
        """Handle Ctrl+Shift+Tab shortcut to navigate to previous tab."""
        if self._layout:
            self._layout.prev_tab()

    def set_status(self, message: str) -> None:
        """Update the status bar message.

        Args:
            message: The status message to display.
        """
        if self._layout:
            self._layout.set_status(message)

    def _on_tab_changed(self, tab: ActiveTab) -> None:
        """Handle tab change events.

        Args:
            tab: The newly selected tab.
        """
        self._session.ui_state.active_tab = tab
        logger.debug(f"Tab changed to: {tab.value}")

    def _on_selection_changed(self, selection: ChannelSelection | None) -> None:
        """Handle current selection change from particle tree.

        Args:
            selection: The new current selection, or None if cleared.
        """
        self._session.current_selection = selection
        if selection:
            # Add to batch selection if not already there
            if selection not in self._session.selected:
                self._session.selected.append(selection)
            logger.debug(
                f"Selection changed: Particle {selection.particle_id}, "
                f"Channel {selection.channel}"
            )

            # Update intensity display with levels if they exist
            self._update_intensity_display()

            # Update correlation tab for dual-channel particles
            self._update_correlation_display()
        else:
            logger.debug("Selection cleared")

        # Update resolve button states
        self._update_resolve_buttons_state()

    def _on_batch_changed(self, selections: list[ChannelSelection]) -> None:
        """Handle batch selection change from particle tree.

        Args:
            selections: List of all selected items.
        """
        self._session.selected = selections
        logger.debug(f"Batch selection changed: {len(selections)} items")

        # Update resolve button states
        self._update_resolve_buttons_state()

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
        if self._layout:
            self._layout.select_all()

    def _on_deselect_all(self) -> None:
        """Handle Deselect All menu action."""
        logger.info("Deselect All triggered")
        if self._layout:
            self._layout.clear_selection()

    def _on_invert_selection(self) -> None:
        """Handle Invert Selection menu action."""
        logger.info("Invert Selection triggered")
        # Will be implemented when needed

    def _on_settings(self) -> None:
        """Handle Settings menu action."""
        logger.info("Settings triggered")
        if self._settings_dialog:
            self._settings_dialog.show()

    def _on_settings_saved(self, settings: Settings) -> None:
        """Handle settings dialog save.

        Args:
            settings: The updated settings.
        """
        logger.info("Settings saved")

        # Update UI state with new default bin size
        if self._layout and self._layout.intensity_tab:
            # Only update if no data is loaded (don't override user's current choice)
            if not self._session.has_file:
                self._session.ui_state.bin_size_ms = settings.display.default_bin_size_ms

        self.set_status("Settings saved")

    # Menu callbacks - Analysis menu

    def _on_resolve(self, mode: str) -> None:
        """Handle Resolve menu action."""
        logger.info(f"Resolve {mode} triggered")
        # Use the current confidence level from session state
        confidence = self._session.ui_state.confidence
        self._on_resolve_from_tab(mode, confidence)

    def _on_group(self, mode: str) -> None:
        """Handle Group menu action."""
        logger.info(f"Group {mode} triggered")
        self._on_grouping_from_tab(mode)

    def _on_grouping_from_tab(self, mode: str) -> None:
        """Handle grouping request from the grouping tab.

        Args:
            mode: The grouping mode ("current", "selected", or "all").
        """
        logger.info(f"Grouping from tab: mode={mode}")

        # Check if already processing
        if self._session.processing.is_busy:
            logger.warning("Already processing, ignoring grouping request")
            return

        # Get grouping options from the tab
        use_lifetime = False
        global_grouping = False
        if self._layout and self._layout.grouping_tab:
            use_lifetime = self._layout.grouping_tab.use_lifetime
            global_grouping = self._layout.grouping_tab.global_grouping

        # Determine which particles to group
        targets: list[ChannelSelection] = []

        if mode == "current":
            if self._session.current_selection:
                targets = [self._session.current_selection]
        elif mode == "selected":
            targets = list(self._session.selected)
        elif mode == "all":
            # Create selections for all particles that have levels
            for particle in self._session.particles:
                # Channel 1
                if self._session.get_levels(particle.id, 1):
                    targets.append(ChannelSelection(particle.id, 1))
                # Channel 2 if dual channel and has levels
                if particle.channel2 is not None and self._session.get_levels(particle.id, 2):
                    targets.append(ChannelSelection(particle.id, 2))

        # Filter to only targets with levels
        targets = [
            t for t in targets
            if self._session.get_levels(t.particle_id, t.channel) is not None
        ]

        if not targets:
            logger.warning("No particles with levels to group")
            if self._layout:
                self._layout.set_status("No particles with levels to group")
            return

        # Start processing
        self._grouping_mode = mode
        self._session.processing.start(
            "Clustering",
            f"Grouping {len(targets)} particle(s)...",
        )

        # Disable group buttons
        if self._layout and self._layout.grouping_tab:
            self._layout.grouping_tab.set_grouping(True)

        # Submit tasks
        self._submit_clustering_tasks(targets, use_lifetime, global_grouping)

    def _submit_clustering_tasks(
        self,
        targets: list[ChannelSelection],
        use_lifetime: bool,
        global_grouping: bool,
    ) -> None:
        """Submit clustering tasks to the worker pool.

        Args:
            targets: List of particle/channel selections to analyze.
            use_lifetime: Whether to use lifetime in clustering.
            global_grouping: Whether to group all particles together.
        """
        if not self._pool:
            logger.error("Worker pool not initialized")
            return

        if global_grouping:
            # Combine all levels from all targets into a single clustering task
            all_levels: list[dict] = []
            for selection in targets:
                levels = self._session.get_levels(selection.particle_id, selection.channel)
                if levels:
                    for level in levels:
                        all_levels.append({
                            "start_index": level.start_index,
                            "end_index": level.end_index,
                            "start_time_ns": level.start_time_ns,
                            "end_time_ns": level.end_time_ns,
                            "num_photons": level.num_photons,
                            "dwell_time_s": level.dwell_time_s,
                            "intensity_cps": level.intensity_cps,
                            "group_id": level.group_id,
                        })

            if all_levels:
                params = {
                    "levels": all_levels,
                    "use_lifetime": use_lifetime,
                    "particle_id": "global",  # Special marker for global grouping
                    "channel_id": 0,
                }
                future = self._pool.submit(run_clustering_task, params)
                self._pending_futures.append(future)

            logger.info(f"Submitted 1 global clustering task with {len(all_levels)} levels")
        else:
            # Submit individual clustering tasks for each target
            for selection in targets:
                levels = self._session.get_levels(selection.particle_id, selection.channel)
                if not levels:
                    continue

                # Convert levels to dicts
                level_dicts = [
                    {
                        "start_index": level.start_index,
                        "end_index": level.end_index,
                        "start_time_ns": level.start_time_ns,
                        "end_time_ns": level.end_time_ns,
                        "num_photons": level.num_photons,
                        "dwell_time_s": level.dwell_time_s,
                        "intensity_cps": level.intensity_cps,
                        "group_id": level.group_id,
                    }
                    for level in levels
                ]

                params = {
                    "levels": level_dicts,
                    "use_lifetime": use_lifetime,
                    "particle_id": selection.particle_id,
                    "channel_id": selection.channel,
                }

                future = self._pool.submit(run_clustering_task, params)
                self._pending_futures.append(future)

            logger.info(f"Submitted {len(self._pending_futures)} clustering tasks")

    def _on_fit(self, mode: str) -> None:
        """Handle Fit menu action."""
        logger.info(f"Fit {mode} triggered")
        # For now, just show the fitting dialog for current selection
        self._on_fit_requested_from_tab()

    def _on_fit_requested_from_tab(self) -> None:
        """Handle fit request from lifetime tab's Fit button."""
        logger.info("Fit requested from tab")

        # Check if we have data
        if not self._session.current_selection:
            logger.warning("No particle selected for fitting")
            if self._layout:
                self._layout.set_status("Select a particle to fit")
            return

        # Show the fitting dialog
        if self._fitting_dialog:
            self._fitting_dialog.show(self._fitting_params)

    def _on_fit_dialog_accepted(self, params: FittingParameters) -> None:
        """Handle fit dialog acceptance - run the fit.

        Args:
            params: The fitting parameters from the dialog.
        """
        logger.info(
            f"Fit dialog accepted: {params.num_exponentials} exp, "
            f"tau_init={params.tau_init}"
        )

        # Store parameters for next time
        self._fitting_params = params

        # Check if we have data
        if not self._session.current_selection:
            logger.warning("No particle selected for fitting")
            return

        # Get the current particle and channel data
        selection = self._session.current_selection
        particle = self._session.get_particle(selection.particle_id)
        if particle is None:
            logger.warning(f"Particle {selection.particle_id} not found")
            return

        channel_data = (
            particle.channel1 if selection.channel == 1 else particle.channel2
        )
        if channel_data is None:
            logger.warning(f"No channel {selection.channel} data")
            return

        # Run the fit
        self._run_fit_task(particle, channel_data, selection, params)

    def _run_fit_task(
        self,
        particle,
        channel_data,
        selection: ChannelSelection,
        params: FittingParameters,
    ) -> None:
        """Submit a lifetime fitting task.

        Args:
            particle: The particle data.
            channel_data: The channel data with microtimes.
            selection: The current selection.
            params: Fitting parameters.
        """
        if not self._pool:
            logger.error("Worker pool not initialized")
            return

        # Build decay histogram
        from full_sms.analysis.histograms import build_decay_histogram

        t, counts = build_decay_histogram(
            channel_data.microtimes, particle.channelwidth
        )

        # Prepare task parameters
        task_params = {
            "t": t,
            "counts": counts,
            "channelwidth": particle.channelwidth,
            "num_exponentials": params.num_exponentials,
            "tau_init": params.get_tau_init_for_fit(),
            "tau_bounds": (params.tau_min, params.tau_max),
            "shift_init": params.shift_init,
            "shift_bounds": (params.shift_min, params.shift_max),
            "autostart": params.start_mode.value,
            "autoend": params.auto_end,
            "particle_id": selection.particle_id,
            "channel_id": selection.channel,
        }

        # Add manual start/end if provided
        if params.start_channel is not None:
            task_params["start"] = params.start_channel
        if params.end_channel is not None:
            task_params["end"] = params.end_channel

        # Add background if manual
        if not params.background_auto:
            task_params["background"] = params.background_value

        # Add IRF if using it (not simulated for now)
        if params.use_irf and not params.use_simulated_irf:
            # TODO: Get IRF from file metadata
            pass

        # Start processing
        self._session.processing.start("Lifetime fit", "Fitting decay curve...")
        if self._layout:
            self._layout.set_status("Fitting...")

        # Submit task
        future = self._pool.submit(run_fit_task, task_params)
        self._pending_futures.append(future)

        logger.info(f"Submitted fit task for particle {selection.particle_id}")

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
        # Check for completed futures
        self._check_pending_futures()

        # Sync status bar with processing state
        if self._layout:
            self._layout.sync_status_with_state(self._session.processing)

    def _check_pending_futures(self) -> None:
        """Check for completed futures and process their results."""
        if not self._pending_futures:
            return

        completed = []
        for future in self._pending_futures:
            if future.done():
                completed.append(future)
                try:
                    result = future.result()
                    self._handle_future_result(result)
                except Exception as e:
                    logger.exception(f"Future failed: {e}")
                    self._session.processing.finish(f"Error: {e}")
                    if self._layout:
                        self._layout.show_error(str(e))

        # Remove completed futures
        for future in completed:
            self._pending_futures.remove(future)

        # If all futures completed, finish processing
        if not self._pending_futures and self._session.processing.is_busy:
            if self._resolve_mode is not None:
                self._finish_resolve()
            elif self._grouping_mode is not None:
                self._finish_grouping()
            elif self._correlation_pending:
                self._finish_correlation()
            else:
                # Unknown operation type, just finish
                self._session.processing.finish("Complete")

    def _handle_future_result(self, result: TaskResult) -> None:
        """Handle a completed task result.

        Args:
            result: The TaskResult from the worker.
        """
        if not result.success:
            logger.error(f"Task failed: {result.error}")
            return

        if result.value is None:
            return

        data = result.value
        particle_id = data.get("particle_id")
        channel_id = data.get("channel_id", 1)

        # Check if this is a CPA result (has "levels" key but not "steps")
        if "levels" in data and "steps" not in data:
            self._handle_cpa_result(particle_id, channel_id, data)
        # Check if this is a clustering result (has "steps" key)
        elif "steps" in data:
            self._handle_clustering_result(particle_id, channel_id, data)
        # Check if this is a fit result (has "tau" key but not "g2")
        elif "tau" in data and "g2" not in data:
            self._handle_fit_result(particle_id, channel_id, data)
        # Check if this is a correlation result (has "g2" key)
        elif "g2" in data:
            self._handle_correlation_result(particle_id, data)

    def _handle_cpa_result(
        self, particle_id: int, channel_id: int, data: dict
    ) -> None:
        """Handle completed CPA result.

        Args:
            particle_id: The particle ID.
            channel_id: The channel ID.
            data: The result data containing levels.
        """
        # Convert level dicts back to LevelData objects
        levels = []
        for level_dict in data["levels"]:
            level = LevelData(
                start_index=level_dict["start_index"],
                end_index=level_dict["end_index"],
                start_time_ns=int(level_dict["start_time_ns"]),
                end_time_ns=int(level_dict["end_time_ns"]),
                num_photons=level_dict["num_photons"],
                intensity_cps=level_dict["intensity_cps"],
                group_id=level_dict.get("group_id"),
            )
            levels.append(level)

        # Store in session state
        self._session.set_levels(particle_id, channel_id, levels)

        logger.info(
            f"CPA complete for particle {particle_id}, channel {channel_id}: "
            f"{len(levels)} levels detected"
        )

        # Update progress
        completed = sum(
            1
            for key in self._session.levels
            if self._session.levels[key] is not None
        )
        total = len(self._pending_futures) + completed
        if total > 0:
            self._session.processing.update(
                completed / total,
                f"Resolved {completed}/{total} particles...",
            )

    def _handle_clustering_result(
        self, particle_id: int | str, channel_id: int, data: dict
    ) -> None:
        """Handle completed clustering result.

        Args:
            particle_id: The particle ID (or "global" for global grouping).
            channel_id: The channel ID.
            data: The result data containing clustering steps.
        """
        from full_sms.models.group import ClusteringResult, ClusteringStep, GroupData

        # Check for null result (not enough levels to cluster)
        if data.get("result") is None and "steps" not in data:
            logger.warning(f"Clustering returned null for {particle_id}, {channel_id}")
            return

        # Convert step dicts back to ClusteringStep objects
        steps = []
        for step_dict in data["steps"]:
            groups = tuple(
                GroupData(
                    group_id=g["group_id"],
                    level_indices=tuple(g["level_indices"]),
                    total_photons=g["total_photons"],
                    total_dwell_time_s=g["total_dwell_time_s"],
                    intensity_cps=g["intensity_cps"],
                )
                for g in step_dict["groups"]
            )
            step = ClusteringStep(
                groups=groups,
                level_group_assignments=tuple(step_dict["level_group_assignments"]),
                bic=step_dict["bic"],
            )
            steps.append(step)

        # Create ClusteringResult
        result = ClusteringResult(
            steps=tuple(steps),
            optimal_step_index=data["optimal_step_index"],
            selected_step_index=data["selected_step_index"],
            num_original_levels=data["num_original_levels"],
        )

        # Store in session state (if not global grouping)
        if particle_id != "global":
            self._session.set_clustering(particle_id, channel_id, result)

        logger.info(
            f"Clustering complete for particle {particle_id}, channel {channel_id}: "
            f"{result.num_groups} groups (optimal), {len(steps)} steps"
        )

        # Update progress
        completed = sum(
            1
            for key in self._session.clustering_results
            if self._session.clustering_results[key] is not None
        )
        total = len(self._pending_futures) + completed
        if total > 0:
            self._session.processing.update(
                completed / total,
                f"Grouped {completed}/{total} particles...",
            )

    def _handle_fit_result(
        self, particle_id: int, channel_id: int, data: dict
    ) -> None:
        """Handle completed fit result.

        Args:
            particle_id: The particle ID.
            channel_id: The channel ID.
            data: The result data containing fit parameters.
        """
        # Check for error
        if data.get("error"):
            logger.error(f"Fit failed: {data['error']}")
            self._session.processing.finish(f"Fit error: {data['error']}")
            if self._layout:
                self._layout.show_error(f"Fit failed: {data['error']}")
            return

        # Convert to FitResult
        fit_result = FitResult.from_fit_parameters(
            tau=data["tau"],
            tau_std=data["tau_std"],
            amplitude=data["amplitude"],
            amplitude_std=data["amplitude_std"],
            shift=data["shift"],
            shift_std=data["shift_std"],
            chi_squared=data["chi_squared"],
            durbin_watson=data["durbin_watson"],
            residuals=np.array(data["residuals"]),
            fitted_curve=np.array(data["fitted_curve"]),
            fit_start_index=data["fit_start_index"],
            fit_end_index=data["fit_end_index"],
            background=data["background"],
            dw_bounds=data.get("dw_bounds"),
        )

        logger.info(
            f"Fit complete for particle {particle_id}, channel {channel_id}: "
            f"tau={fit_result.tau}, chi2={fit_result.chi_squared:.3f}"
        )

        # Update the lifetime tab display
        if self._layout and self._layout.lifetime_tab:
            self._layout.lifetime_tab.set_fit(fit_result)

        # Finish processing
        self._session.processing.finish("Fit complete")

        # Show success message
        if self._layout:
            tau_str = ", ".join(f"{t:.2f}" for t in fit_result.tau)
            self._layout.show_success(
                f"Fit complete: tau = {tau_str} ns, chi2 = {fit_result.chi_squared:.3f}"
            )

    def _handle_correlation_result(self, particle_id: int, data: dict) -> None:
        """Handle completed correlation result.

        Args:
            particle_id: The particle ID.
            data: The result data containing g2 correlation values.
        """
        from full_sms.analysis.correlation import CorrelationResult

        # Create CorrelationResult from the data
        result = CorrelationResult(
            tau=np.array(data["tau"]),
            g2=np.array(data["g2"]),
            events=np.array(data["events"]),
            window_ns=data["window_ns"],
            binsize_ns=data["binsize_ns"],
            num_photons_ch1=data["num_photons_ch1"],
            num_photons_ch2=data["num_photons_ch2"],
        )

        logger.info(
            f"Correlation complete for particle {particle_id}: "
            f"{result.num_events} events, window={result.window_ns}ns"
        )

        # Update the correlation tab display
        if self._layout:
            self._layout.set_correlation_result(result)

        # Finish processing
        self._correlation_pending = False
        self._session.processing.finish("Correlation complete")

        # Show success message
        if self._layout:
            g2_zero = result.g2[len(result.g2) // 2] if len(result.g2) > 0 else 0
            self._layout.show_success(
                f"Correlation complete: {result.num_events:,} events, g2(0)={g2_zero}"
            )

    def _on_correlate_from_tab(
        self, window_ns: float, binsize_ns: float, difftime_ns: float
    ) -> None:
        """Handle correlate request from the correlation tab.

        Args:
            window_ns: Correlation window in nanoseconds.
            binsize_ns: Histogram bin size in nanoseconds.
            difftime_ns: Channel time offset in nanoseconds.
        """
        logger.info(
            f"Correlate from tab: window={window_ns}ns, binsize={binsize_ns}ns, "
            f"offset={difftime_ns}ns"
        )

        # Check if already processing
        if self._session.processing.is_busy:
            logger.warning("Already processing, ignoring correlation request")
            return

        # Check if we have a current selection
        if not self._session.current_selection:
            logger.warning("No particle selected for correlation")
            if self._layout:
                self._layout.set_status("Select a particle for correlation")
            return

        selection = self._session.current_selection
        particle = self._session.get_particle(selection.particle_id)
        if particle is None:
            logger.warning(f"Particle {selection.particle_id} not found")
            return

        # Check for dual channel
        if not particle.has_dual_channel:
            logger.warning("Particle does not have dual channels")
            if self._layout:
                self._layout.set_status("Correlation requires dual-channel data")
            return

        # Start processing
        self._correlation_pending = True
        self._session.processing.start("Correlation", "Calculating g2...")

        # Disable correlate button
        if self._layout and self._layout.correlation_tab:
            self._layout.correlation_tab.enable_correlate_button(False)

        # Submit correlation task
        self._submit_correlation_task(particle, window_ns, binsize_ns, difftime_ns)

    def _submit_correlation_task(
        self,
        particle,
        window_ns: float,
        binsize_ns: float,
        difftime_ns: float,
    ) -> None:
        """Submit a correlation task to the worker pool.

        Args:
            particle: The particle with dual-channel data.
            window_ns: Correlation window in nanoseconds.
            binsize_ns: Histogram bin size in nanoseconds.
            difftime_ns: Channel time offset in nanoseconds.
        """
        if not self._pool:
            logger.error("Worker pool not initialized")
            return

        params = {
            "abstimes1": particle.channel1.abstimes.astype(np.float64),
            "abstimes2": particle.channel2.abstimes.astype(np.float64),
            "microtimes1": particle.channel1.microtimes,
            "microtimes2": particle.channel2.microtimes,
            "window_ns": window_ns,
            "binsize_ns": binsize_ns,
            "difftime_ns": difftime_ns,
            "particle_id": particle.id,
        }

        future = self._pool.submit(run_correlation_task, params)
        self._pending_futures.append(future)

        logger.info(f"Submitted correlation task for particle {particle.id}")

    def _finish_correlation(self) -> None:
        """Finish the correlation operation and update the UI."""
        self._correlation_pending = False
        self._session.processing.finish("Correlation complete")

        # Re-enable correlate button
        if self._layout and self._layout.correlation_tab:
            self._layout.correlation_tab.enable_correlate_button(True)

    def _finish_resolve(self) -> None:
        """Finish the resolve operation and update the UI."""
        self._session.processing.finish("Change point analysis complete")
        self._resolve_mode = None

        # Re-enable resolve buttons
        if self._layout and self._layout.intensity_tab:
            self._layout.intensity_tab.set_resolving(False)
            self._update_resolve_buttons_state()

        # Update display with levels for current selection
        self._update_intensity_display()

        # Update group buttons state (now that levels exist)
        self._update_group_buttons_state()

        if self._layout:
            num_levels = 0
            if self._session.current_selection:
                levels = self._session.get_current_levels()
                num_levels = len(levels) if levels else 0
            self._layout.show_success(
                f"Change point analysis complete ({num_levels} levels detected)"
            )

    def _finish_grouping(self) -> None:
        """Finish the grouping operation and update the UI."""
        self._session.processing.finish("Clustering complete")
        self._grouping_mode = None

        # Re-enable group buttons
        if self._layout and self._layout.grouping_tab:
            self._layout.grouping_tab.set_grouping(False)
            self._update_group_buttons_state()

        # Update display with clustering results for current selection
        self._update_grouping_display()

        if self._layout:
            num_groups = 0
            if self._session.current_selection:
                clustering = self._session.get_current_clustering()
                num_groups = clustering.num_groups if clustering else 0
            self._layout.show_success(
                f"Clustering complete ({num_groups} groups)"
            )

    def _update_grouping_display(self) -> None:
        """Update the grouping tab display with current data."""
        if not self._layout or not self._layout.grouping_tab:
            return

        if not self._session.current_selection:
            return

        # Get clustering result for current selection
        clustering = self._session.get_current_clustering()
        if clustering:
            self._layout.grouping_tab.set_clustering_result(clustering)

            # Also update the intensity tab's level display to show group colors
            levels = self._session.get_current_levels()
            if levels and self._layout.intensity_tab:
                # Update levels with group assignments from clustering
                updated_levels = self._apply_group_assignments_to_levels(
                    levels, clustering
                )
                self._layout.intensity_tab.set_levels(updated_levels, color_by_group=True)

    def _apply_group_assignments_to_levels(
        self, levels: list[LevelData], clustering
    ) -> list[LevelData]:
        """Apply group assignments from clustering result to levels.

        Args:
            levels: The original levels.
            clustering: The ClusteringResult with group assignments.

        Returns:
            New list of LevelData with group_id set.
        """
        assignments = clustering.selected_assignments
        updated_levels = []
        for i, level in enumerate(levels):
            group_id = assignments[i] if i < len(assignments) else None
            updated_levels.append(
                LevelData(
                    start_index=level.start_index,
                    end_index=level.end_index,
                    start_time_ns=level.start_time_ns,
                    end_time_ns=level.end_time_ns,
                    num_photons=level.num_photons,
                    intensity_cps=level.intensity_cps,
                    group_id=group_id,
                )
            )
        return updated_levels

    def _update_group_buttons_state(self) -> None:
        """Update group button states based on current selections."""
        if not self._layout or not self._layout.grouping_tab:
            return

        # Check if current selection has levels
        has_current = False
        if self._session.current_selection:
            levels = self._session.get_current_levels()
            has_current = levels is not None and len(levels) > 0

        # Check if any selected particles have levels
        has_selected = any(
            self._session.get_levels(s.particle_id, s.channel) is not None
            for s in self._session.selected
        )

        # Check if any particles have levels
        has_any = any(
            self._session.get_levels(p.id, 1) is not None
            or (p.channel2 is not None and self._session.get_levels(p.id, 2) is not None)
            for p in self._session.particles
        )

        self._layout.grouping_tab.set_group_buttons_state(
            has_current=has_current,
            has_selected=has_selected,
            has_any=has_any,
        )

    def _on_resolve_from_tab(self, mode: str, confidence: ConfidenceLevel) -> None:
        """Handle resolve request from the intensity tab.

        Args:
            mode: The resolve mode ("current", "selected", or "all").
            confidence: The confidence level for CPA.
        """
        logger.info(f"Resolve from tab: mode={mode}, confidence={confidence.value}")

        # Check if already processing
        if self._session.processing.is_busy:
            logger.warning("Already processing, ignoring resolve request")
            return

        # Determine which particles to resolve
        targets: list[ChannelSelection] = []

        if mode == "current":
            if self._session.current_selection:
                targets = [self._session.current_selection]
        elif mode == "selected":
            targets = list(self._session.selected)
        elif mode == "all":
            # Create selections for all particles
            for particle in self._session.particles:
                # Channel 1
                targets.append(ChannelSelection(particle.id, 1))
                # Channel 2 if dual channel
                if particle.channel2 is not None:
                    targets.append(ChannelSelection(particle.id, 2))

        if not targets:
            logger.warning("No particles to resolve")
            if self._layout:
                self._layout.set_status("No particles selected for analysis")
            return

        # Start processing
        self._resolve_mode = mode
        self._session.processing.start(
            "Change point analysis",
            f"Resolving {len(targets)} particle(s)...",
        )

        # Disable resolve buttons
        if self._layout and self._layout.intensity_tab:
            self._layout.intensity_tab.set_resolving(True)

        # Submit tasks
        self._submit_cpa_tasks(targets, confidence)

    def _submit_cpa_tasks(
        self, targets: list[ChannelSelection], confidence: ConfidenceLevel
    ) -> None:
        """Submit CPA tasks to the worker pool.

        Args:
            targets: List of particle/channel selections to analyze.
            confidence: The confidence level.
        """
        if not self._pool:
            logger.error("Worker pool not initialized")
            return

        for selection in targets:
            particle = self._session.get_particle(selection.particle_id)
            if particle is None:
                logger.warning(f"Particle {selection.particle_id} not found")
                continue

            # Get abstimes for the selected channel
            if selection.channel == 1:
                channel_data = particle.channel1
            else:
                channel_data = particle.channel2

            if channel_data is None:
                logger.warning(
                    f"No channel {selection.channel} data for particle "
                    f"{selection.particle_id}"
                )
                continue

            # Build task parameters
            params = {
                "abstimes": channel_data.abstimes.astype(np.float64),
                "confidence": confidence.value,
                "particle_id": selection.particle_id,
                "channel_id": selection.channel,
            }

            # Get end time if available
            if len(channel_data.abstimes) > 0:
                params["end_time_ns"] = float(channel_data.abstimes[-1])

            # Submit task
            future = self._pool.submit(run_cpa_task, params)
            self._pending_futures.append(future)

        logger.info(f"Submitted {len(self._pending_futures)} CPA tasks")

    def _update_intensity_display(self) -> None:
        """Update the intensity tab display with current data."""
        if not self._layout or not self._layout.intensity_tab:
            return

        if not self._session.current_selection:
            return

        # Get levels for current selection
        levels = self._session.get_current_levels()
        if levels:
            self._layout.intensity_tab.set_levels(levels)

    def _update_correlation_display(self) -> None:
        """Update the correlation tab display for the current selection."""
        if not self._layout or not self._layout.correlation_tab:
            return

        if not self._session.current_selection:
            self._layout.clear_correlation_data()
            return

        # Get the current particle
        particle = self._session.get_particle(
            self._session.current_selection.particle_id
        )
        if particle is None:
            self._layout.clear_correlation_data()
            return

        # Check if this particle has dual channels
        if particle.has_dual_channel and particle.channel2 is not None:
            # Set dual-channel data
            self._layout.set_correlation_data(
                particle.channel1.abstimes,
                particle.channel2.abstimes,
                particle.channel1.microtimes,
                particle.channel2.microtimes,
            )
        else:
            # Single channel - show message
            self._layout.set_correlation_single_channel()

    def _update_resolve_buttons_state(self) -> None:
        """Update resolve button states based on current selections."""
        if not self._layout or not self._layout.intensity_tab:
            return

        has_current = self._session.current_selection is not None
        has_selected = len(self._session.selected) > 0
        has_any = len(self._session.particles) > 0

        self._layout.intensity_tab.set_resolve_buttons_state(
            has_current=has_current and self._layout.intensity_tab.has_data,
            has_selected=has_selected,
            has_any=has_any,
        )

    def shutdown(self) -> None:
        """Clean up and destroy the DearPyGui context."""
        logger.info("Shutting down application")

        # Shutdown worker pool
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
            logger.info("Worker pool shutdown")

        # Clean up keyboard shortcuts
        if self._keyboard:
            self._keyboard.destroy()

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
