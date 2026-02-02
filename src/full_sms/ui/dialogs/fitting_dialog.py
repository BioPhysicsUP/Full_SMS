"""Fitting dialog for configuring lifetime fit parameters.

Provides a modal dialog for configuring:
- Number of exponentials (1/2/3)
- IRF settings (use IRF, shift)
- Fit range (auto or manual)
- Initial guesses for tau
- Background correction settings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import dearpygui.dearpygui as dpg

from full_sms.analysis.lifetime import StartpointMode

logger = logging.getLogger(__name__)


class NumExponentials(Enum):
    """Number of exponential components for fitting."""

    ONE = 1
    TWO = 2
    THREE = 3


class FitTarget(Enum):
    """What to fit - measurement full decay or individual levels."""

    MEASUREMENT = "Measurement (full decay)"
    SELECTED_LEVEL = "Selected Level"
    ALL_LEVELS = "All Levels"


class FitScope(Enum):
    """Which measurements to fit."""

    CURRENT = "Current"
    SELECTED = "Selected"
    ALL = "All"


@dataclass
class FittingParameters:
    """Parameters for lifetime fitting.

    Attributes:
        fit_target: What to fit - measurement full decay or levels.
        fit_scope: Which measurements to fit (current, selected, all).
        selected_level_index: Index of selected level (for SELECTED_LEVEL target).
        num_exponentials: Number of exponential components (1, 2, or 3).
        tau_init: Initial guess for tau values in nanoseconds.
        tau_min: Minimum bounds for tau values.
        tau_max: Maximum bounds for tau values.
        use_irf: Whether to use IRF convolution.
        shift_init: Initial IRF shift in channels.
        shift_min: Minimum IRF shift bound.
        shift_max: Maximum IRF shift bound.
        use_simulated_irf: Whether to use simulated IRF.
        simulated_irf_fwhm: FWHM for simulated IRF in nanoseconds.
        fit_simulated_irf_fwhm: Whether to fit the FWHM (only for simulated IRF).
        simulated_irf_fwhm_min: Minimum bound for fitted FWHM.
        simulated_irf_fwhm_max: Maximum bound for fitted FWHM.
        start_mode: Automatic startpoint detection mode.
        start_channel: Manual start channel index.
        auto_end: Whether to automatically determine endpoint.
        end_channel: Manual end channel index.
        background_auto: Whether to auto-estimate background.
        background_value: Manual background value.
    """

    fit_target: FitTarget = FitTarget.MEASUREMENT
    fit_scope: FitScope = FitScope.CURRENT
    selected_level_index: Optional[int] = None
    num_exponentials: int = 1
    tau_init: list[float] = field(default_factory=lambda: [5.0])
    tau_min: float = 0.01
    tau_max: float = 100.0
    use_irf: bool = True
    shift_init: float = 0.0
    shift_min: float = -2000.0
    shift_max: float = 2000.0
    use_simulated_irf: bool = False
    simulated_irf_fwhm: float = 0.1
    fit_simulated_irf_fwhm: bool = False
    simulated_irf_fwhm_min: float = 0.01
    simulated_irf_fwhm_max: float = 2.0
    start_mode: StartpointMode = StartpointMode.CLOSE_TO_MAX
    start_channel: Optional[int] = None
    auto_end: bool = True
    end_channel: Optional[int] = None
    background_auto: bool = True
    background_value: float = 0.0

    def get_tau_init_for_fit(self) -> list[float]:
        """Get tau initial values padded/trimmed for num_exponentials."""
        defaults = [5.0, 1.0, 0.1]
        result = []
        for i in range(self.num_exponentials):
            if i < len(self.tau_init):
                result.append(self.tau_init[i])
            else:
                result.append(defaults[i] if i < len(defaults) else 5.0)
        return result


@dataclass
class FittingDialogTags:
    """Tags for fitting dialog UI elements."""

    dialog: str = "fitting_dialog"
    # Fit target and scope
    fit_target_combo: str = "fitting_fit_target"
    fit_scope_combo: str = "fitting_fit_scope"
    fit_scope_row: str = "fitting_fit_scope_row"
    num_exp_combo: str = "fitting_num_exp_combo"
    # Tau parameters
    tau_group: str = "fitting_tau_group"
    tau1_init: str = "fitting_tau1_init"
    tau2_init: str = "fitting_tau2_init"
    tau3_init: str = "fitting_tau3_init"
    tau2_row: str = "fitting_tau2_row"
    tau3_row: str = "fitting_tau3_row"
    tau_min: str = "fitting_tau_min"
    tau_max: str = "fitting_tau_max"
    # IRF settings
    use_irf_checkbox: str = "fitting_use_irf"
    shift_init: str = "fitting_shift_init"
    shift_min: str = "fitting_shift_min"
    shift_max: str = "fitting_shift_max"
    use_simulated_irf: str = "fitting_use_simulated_irf"
    simulated_irf_fwhm: str = "fitting_simulated_irf_fwhm"
    fit_simulated_irf_fwhm: str = "fitting_fit_simulated_irf_fwhm"
    simulated_irf_fwhm_min: str = "fitting_simulated_irf_fwhm_min"
    simulated_irf_fwhm_max: str = "fitting_simulated_irf_fwhm_max"
    simulated_irf_fwhm_bounds_row: str = "fitting_simulated_irf_fwhm_bounds_row"
    irf_group: str = "fitting_irf_group"
    simulated_irf_row: str = "fitting_simulated_irf_row"
    # Fit range
    start_mode_combo: str = "fitting_start_mode"
    start_channel: str = "fitting_start_channel"
    start_manual_row: str = "fitting_start_manual_row"
    auto_end_checkbox: str = "fitting_auto_end"
    end_channel: str = "fitting_end_channel"
    end_manual_row: str = "fitting_end_manual_row"
    # Background
    background_auto: str = "fitting_bg_auto"
    background_value: str = "fitting_bg_value"
    background_manual_row: str = "fitting_bg_manual_row"
    # Buttons
    fit_button: str = "fitting_fit_button"
    cancel_button: str = "fitting_cancel_button"


class FittingDialog:
    """Modal dialog for configuring lifetime fitting parameters.

    Usage:
        dialog = FittingDialog()
        dialog.set_on_fit(callback)  # Called with FittingParameters when Fit is clicked
        dialog.show()
    """

    def __init__(self, tag_prefix: str = "") -> None:
        """Initialize the fitting dialog.

        Args:
            tag_prefix: Optional prefix for unique tags.
        """
        self._tag_prefix = tag_prefix
        self._is_built = False
        self._parameters = FittingParameters()
        self._on_fit: Optional[Callable[[FittingParameters], None]] = None
        self._on_cancel: Optional[Callable[[], None]] = None
        self._has_irf: bool = False

        # Level state for fit target options
        self._has_levels: bool = False
        self._has_selected_level: bool = False
        self._selected_level_index: Optional[int] = None

        # Generate unique tags
        self._tags = FittingDialogTags(
            dialog=f"{tag_prefix}fitting_dialog",
            fit_target_combo=f"{tag_prefix}fitting_fit_target",
            fit_scope_combo=f"{tag_prefix}fitting_fit_scope",
            fit_scope_row=f"{tag_prefix}fitting_fit_scope_row",
            num_exp_combo=f"{tag_prefix}fitting_num_exp_combo",
            tau_group=f"{tag_prefix}fitting_tau_group",
            tau1_init=f"{tag_prefix}fitting_tau1_init",
            tau2_init=f"{tag_prefix}fitting_tau2_init",
            tau3_init=f"{tag_prefix}fitting_tau3_init",
            tau2_row=f"{tag_prefix}fitting_tau2_row",
            tau3_row=f"{tag_prefix}fitting_tau3_row",
            tau_min=f"{tag_prefix}fitting_tau_min",
            tau_max=f"{tag_prefix}fitting_tau_max",
            use_irf_checkbox=f"{tag_prefix}fitting_use_irf",
            shift_init=f"{tag_prefix}fitting_shift_init",
            shift_min=f"{tag_prefix}fitting_shift_min",
            shift_max=f"{tag_prefix}fitting_shift_max",
            use_simulated_irf=f"{tag_prefix}fitting_use_simulated_irf",
            simulated_irf_fwhm=f"{tag_prefix}fitting_simulated_irf_fwhm",
            fit_simulated_irf_fwhm=f"{tag_prefix}fitting_fit_simulated_irf_fwhm",
            simulated_irf_fwhm_min=f"{tag_prefix}fitting_simulated_irf_fwhm_min",
            simulated_irf_fwhm_max=f"{tag_prefix}fitting_simulated_irf_fwhm_max",
            simulated_irf_fwhm_bounds_row=f"{tag_prefix}fitting_simulated_irf_fwhm_bounds_row",
            irf_group=f"{tag_prefix}fitting_irf_group",
            simulated_irf_row=f"{tag_prefix}fitting_simulated_irf_row",
            start_mode_combo=f"{tag_prefix}fitting_start_mode",
            start_channel=f"{tag_prefix}fitting_start_channel",
            start_manual_row=f"{tag_prefix}fitting_start_manual_row",
            auto_end_checkbox=f"{tag_prefix}fitting_auto_end",
            end_channel=f"{tag_prefix}fitting_end_channel",
            end_manual_row=f"{tag_prefix}fitting_end_manual_row",
            background_auto=f"{tag_prefix}fitting_bg_auto",
            background_value=f"{tag_prefix}fitting_bg_value",
            background_manual_row=f"{tag_prefix}fitting_bg_manual_row",
            fit_button=f"{tag_prefix}fitting_fit_button",
            cancel_button=f"{tag_prefix}fitting_cancel_button",
        )

    @property
    def tags(self) -> FittingDialogTags:
        """Get the dialog tags."""
        return self._tags

    @property
    def parameters(self) -> FittingParameters:
        """Get the current fitting parameters."""
        return self._parameters

    def set_on_fit(self, callback: Callable[[FittingParameters], None]) -> None:
        """Set callback for when Fit button is clicked.

        Args:
            callback: Function called with FittingParameters when user clicks Fit.
        """
        self._on_fit = callback

    def set_on_cancel(self, callback: Callable[[], None]) -> None:
        """Set callback for when Cancel button is clicked.

        Args:
            callback: Function called when user clicks Cancel.
        """
        self._on_cancel = callback

    def set_has_irf(self, has_irf: bool) -> None:
        """Set whether IRF data is available.

        Args:
            has_irf: True if IRF data exists in the file.
        """
        self._has_irf = has_irf

    def set_level_state(
        self,
        has_levels: bool,
        selected_level_index: Optional[int] = None,
    ) -> None:
        """Set the level state for fit target options.

        Args:
            has_levels: True if the current measurement has resolved levels.
            selected_level_index: Index of currently selected level, or None.
        """
        self._has_levels = has_levels
        self._has_selected_level = selected_level_index is not None
        self._selected_level_index = selected_level_index

    def build(self) -> None:
        """Build the dialog UI (but don't show it)."""
        if self._is_built:
            return

        # Delete existing dialog if present
        if dpg.does_item_exist(self._tags.dialog):
            dpg.delete_item(self._tags.dialog)

        # Create modal window
        with dpg.window(
            label="Lifetime Fitting",
            tag=self._tags.dialog,
            modal=True,
            show=False,
            width=550,
            height=600,
            no_resize=True,
            no_move=False,
            on_close=self._on_close,
        ):
            # Fit target selection
            dpg.add_text("Fit Target")
            dpg.add_combo(
                items=[FitTarget.MEASUREMENT.value],  # Options set dynamically in show()
                default_value=FitTarget.MEASUREMENT.value,
                tag=self._tags.fit_target_combo,
                callback=self._on_fit_target_changed,
                width=220,
            )

            # Fit scope selection
            with dpg.group(horizontal=True, tag=self._tags.fit_scope_row):
                dpg.add_text("Scope:")
                dpg.add_combo(
                    items=[s.value for s in FitScope],
                    default_value=FitScope.CURRENT.value,
                    tag=self._tags.fit_scope_combo,
                    width=120,
                )

            dpg.add_separator()

            # Number of exponentials
            dpg.add_text("Number of Exponentials")
            dpg.add_combo(
                items=["1", "2", "3"],
                default_value="1",
                tag=self._tags.num_exp_combo,
                callback=self._on_num_exp_changed,
                width=100,
            )

            dpg.add_separator()

            # Tau parameters section
            dpg.add_text("Lifetime Parameters (tau, ns)")

            with dpg.group(tag=self._tags.tau_group):
                # Tau 1
                with dpg.group(horizontal=True):
                    dpg.add_text("tau1 init:", indent=10)
                    dpg.add_input_float(
                        tag=self._tags.tau1_init,
                        default_value=5.0,
                        width=100,
                        min_value=0.001,
                        min_clamped=True,
                    )

                # Tau 2 (hidden by default)
                with dpg.group(horizontal=True, tag=self._tags.tau2_row, show=False):
                    dpg.add_text("tau2 init:", indent=10)
                    dpg.add_input_float(
                        tag=self._tags.tau2_init,
                        default_value=1.0,
                        width=100,
                        min_value=0.001,
                        min_clamped=True,
                    )

                # Tau 3 (hidden by default)
                with dpg.group(horizontal=True, tag=self._tags.tau3_row, show=False):
                    dpg.add_text("tau3 init:", indent=10)
                    dpg.add_input_float(
                        tag=self._tags.tau3_init,
                        default_value=0.1,
                        width=100,
                        min_value=0.001,
                        min_clamped=True,
                    )

                # Tau bounds
                with dpg.group(horizontal=True):
                    dpg.add_text("Bounds:", indent=10)
                    dpg.add_input_float(
                        tag=self._tags.tau_min,
                        default_value=0.01,
                        width=110,
                        label="min",
                    )
                    dpg.add_input_float(
                        tag=self._tags.tau_max,
                        default_value=100.0,
                        width=110,
                        label="max",
                    )

            dpg.add_separator()

            # IRF settings section
            dpg.add_text("IRF Settings")

            with dpg.group(tag=self._tags.irf_group):
                dpg.add_checkbox(
                    label="Use IRF",
                    tag=self._tags.use_irf_checkbox,
                    default_value=True,
                    callback=self._on_use_irf_changed,
                )

                # Simulated IRF option
                with dpg.group(
                    horizontal=True,
                    tag=self._tags.simulated_irf_row,
                ):
                    dpg.add_checkbox(
                        label="Use Simulated IRF",
                        tag=self._tags.use_simulated_irf,
                        default_value=False,
                        callback=self._on_use_simulated_irf_changed,
                        indent=20,
                    )
                    dpg.add_input_float(
                        tag=self._tags.simulated_irf_fwhm,
                        default_value=0.1,
                        width=100,
                        label="FWHM (ns)",
                        enabled=False,
                    )
                    dpg.add_checkbox(
                        label="Fit FWHM",
                        tag=self._tags.fit_simulated_irf_fwhm,
                        default_value=False,
                        callback=self._on_fit_fwhm_changed,
                        enabled=False,
                    )

                # FWHM bounds (only shown when fitting FWHM)
                with dpg.group(
                    horizontal=True,
                    tag=self._tags.simulated_irf_fwhm_bounds_row,
                    show=False,
                ):
                    dpg.add_text("FWHM bounds:", indent=40)
                    dpg.add_input_float(
                        tag=self._tags.simulated_irf_fwhm_min,
                        default_value=0.01,
                        width=80,
                        label="min",
                    )
                    dpg.add_input_float(
                        tag=self._tags.simulated_irf_fwhm_max,
                        default_value=2.0,
                        width=80,
                        label="max (ns)",
                    )

                # Shift parameters
                with dpg.group(horizontal=True):
                    dpg.add_text("Shift:", indent=10)
                    dpg.add_input_float(
                        tag=self._tags.shift_init,
                        default_value=0.0,
                        width=100,
                        label="init",
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Shift bounds:", indent=10)
                    dpg.add_input_float(
                        tag=self._tags.shift_min,
                        default_value=-2000.0,
                        width=110,
                        label="min",
                    )
                    dpg.add_input_float(
                        tag=self._tags.shift_max,
                        default_value=2000.0,
                        width=110,
                        label="max",
                    )

            dpg.add_separator()

            # Fit range section
            dpg.add_text("Fit Range")

            # Start mode
            with dpg.group(horizontal=True):
                dpg.add_text("Start mode:", indent=10)
                dpg.add_combo(
                    items=[mode.value for mode in StartpointMode],
                    default_value=StartpointMode.CLOSE_TO_MAX.value,
                    tag=self._tags.start_mode_combo,
                    callback=self._on_start_mode_changed,
                    width=180,
                )

            # Manual start channel (hidden by default)
            with dpg.group(
                horizontal=True, tag=self._tags.start_manual_row, show=False
            ):
                dpg.add_text("Start channel:", indent=20)
                dpg.add_input_int(
                    tag=self._tags.start_channel,
                    default_value=0,
                    width=100,
                    min_value=0,
                    min_clamped=True,
                )

            # Auto end
            dpg.add_checkbox(
                label="Auto-detect endpoint",
                tag=self._tags.auto_end_checkbox,
                default_value=True,
                callback=self._on_auto_end_changed,
                indent=10,
            )

            # Manual end channel (hidden by default)
            with dpg.group(horizontal=True, tag=self._tags.end_manual_row, show=False):
                dpg.add_text("End channel:", indent=20)
                dpg.add_input_int(
                    tag=self._tags.end_channel,
                    default_value=4096,
                    width=100,
                    min_value=1,
                    min_clamped=True,
                )

            dpg.add_separator()

            # Background section
            dpg.add_text("Background")

            dpg.add_checkbox(
                label="Auto-estimate background",
                tag=self._tags.background_auto,
                default_value=True,
                callback=self._on_bg_auto_changed,
                indent=10,
            )

            with dpg.group(
                horizontal=True, tag=self._tags.background_manual_row, show=False
            ):
                dpg.add_text("Background value:", indent=20)
                dpg.add_input_float(
                    tag=self._tags.background_value,
                    default_value=0.0,
                    width=100,
                    min_value=0.0,
                    min_clamped=True,
                )

            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Fit",
                    tag=self._tags.fit_button,
                    callback=self._on_fit_clicked,
                    width=120,
                )
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label="Cancel",
                    tag=self._tags.cancel_button,
                    callback=self._on_cancel_clicked,
                    width=120,
                )

        self._is_built = True
        logger.debug("Fitting dialog built")

    def show(self, parameters: Optional[FittingParameters] = None) -> None:
        """Show the fitting dialog.

        Args:
            parameters: Optional existing parameters to populate the dialog.
        """
        if not self._is_built:
            self.build()

        # Configure fit target options based on level state
        self._update_fit_target_options()

        if parameters:
            self._parameters = parameters
            self._populate_from_parameters()

        # Center the dialog
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.set_item_pos(
            self._tags.dialog,
            [viewport_width // 2 - 275, viewport_height // 2 - 300],
        )

        dpg.configure_item(self._tags.dialog, show=True)
        logger.debug("Fitting dialog shown")

    def _update_fit_target_options(self) -> None:
        """Update fit target combo options based on current level state."""
        if not dpg.does_item_exist(self._tags.fit_target_combo):
            return

        # Build list of available options
        options = [FitTarget.MEASUREMENT.value]

        if self._has_selected_level:
            options.append(FitTarget.SELECTED_LEVEL.value)

        if self._has_levels:
            options.append(FitTarget.ALL_LEVELS.value)

        # Update combo items
        dpg.configure_item(self._tags.fit_target_combo, items=options)

        # Ensure current value is valid
        current_value = dpg.get_value(self._tags.fit_target_combo)
        if current_value not in options:
            dpg.set_value(self._tags.fit_target_combo, FitTarget.MEASUREMENT.value)

        # Update scope enabled state based on current target
        self._on_fit_target_changed(
            self._tags.fit_target_combo,
            dpg.get_value(self._tags.fit_target_combo),
        )

    def hide(self) -> None:
        """Hide the fitting dialog."""
        if dpg.does_item_exist(self._tags.dialog):
            dpg.configure_item(self._tags.dialog, show=False)

    def _populate_from_parameters(self) -> None:
        """Populate dialog fields from current parameters."""
        p = self._parameters

        # Fit target and scope (only if the value is available in current options)
        if dpg.does_item_exist(self._tags.fit_target_combo):
            available_targets = dpg.get_item_configuration(
                self._tags.fit_target_combo
            ).get("items", [])
            if p.fit_target.value in available_targets:
                dpg.set_value(self._tags.fit_target_combo, p.fit_target.value)
                # Update scope enabled state
                self._on_fit_target_changed(
                    self._tags.fit_target_combo, p.fit_target.value
                )

        if dpg.does_item_exist(self._tags.fit_scope_combo):
            dpg.set_value(self._tags.fit_scope_combo, p.fit_scope.value)

        # Number of exponentials
        dpg.set_value(self._tags.num_exp_combo, str(p.num_exponentials))
        self._update_tau_rows_visibility()

        # Tau values
        tau_inits = p.get_tau_init_for_fit()
        if len(tau_inits) >= 1:
            dpg.set_value(self._tags.tau1_init, tau_inits[0])
        if len(tau_inits) >= 2:
            dpg.set_value(self._tags.tau2_init, tau_inits[1])
        if len(tau_inits) >= 3:
            dpg.set_value(self._tags.tau3_init, tau_inits[2])

        dpg.set_value(self._tags.tau_min, p.tau_min)
        dpg.set_value(self._tags.tau_max, p.tau_max)

        # IRF settings
        dpg.set_value(self._tags.use_irf_checkbox, p.use_irf)
        dpg.set_value(self._tags.shift_init, p.shift_init)
        dpg.set_value(self._tags.shift_min, p.shift_min)
        dpg.set_value(self._tags.shift_max, p.shift_max)
        dpg.set_value(self._tags.use_simulated_irf, p.use_simulated_irf)
        dpg.set_value(self._tags.simulated_irf_fwhm, p.simulated_irf_fwhm)
        dpg.set_value(self._tags.fit_simulated_irf_fwhm, p.fit_simulated_irf_fwhm)
        dpg.set_value(self._tags.simulated_irf_fwhm_min, p.simulated_irf_fwhm_min)
        dpg.set_value(self._tags.simulated_irf_fwhm_max, p.simulated_irf_fwhm_max)
        # Update UI state based on simulated IRF settings
        self._on_use_simulated_irf_changed(
            self._tags.use_simulated_irf, p.use_simulated_irf
        )

        # Fit range
        dpg.set_value(self._tags.start_mode_combo, p.start_mode.value)
        self._update_start_manual_visibility()
        if p.start_channel is not None:
            dpg.set_value(self._tags.start_channel, p.start_channel)
        dpg.set_value(self._tags.auto_end_checkbox, p.auto_end)
        self._update_end_manual_visibility()
        if p.end_channel is not None:
            dpg.set_value(self._tags.end_channel, p.end_channel)

        # Background
        dpg.set_value(self._tags.background_auto, p.background_auto)
        dpg.set_value(self._tags.background_value, p.background_value)
        self._update_bg_manual_visibility()

    def _collect_parameters(self) -> FittingParameters:
        """Collect parameters from dialog fields."""
        # Get fit target
        fit_target_str = dpg.get_value(self._tags.fit_target_combo)
        fit_target = FitTarget.MEASUREMENT
        for target in FitTarget:
            if target.value == fit_target_str:
                fit_target = target
                break

        # Get fit scope
        fit_scope_str = dpg.get_value(self._tags.fit_scope_combo)
        fit_scope = FitScope.CURRENT
        for scope in FitScope:
            if scope.value == fit_scope_str:
                fit_scope = scope
                break

        # Get selected level index (stored when set_level_state was called)
        selected_level_index = self._selected_level_index

        num_exp = int(dpg.get_value(self._tags.num_exp_combo))

        # Collect tau inits based on number of exponentials
        tau_inits = [dpg.get_value(self._tags.tau1_init)]
        if num_exp >= 2:
            tau_inits.append(dpg.get_value(self._tags.tau2_init))
        if num_exp >= 3:
            tau_inits.append(dpg.get_value(self._tags.tau3_init))

        # Get start mode
        start_mode_str = dpg.get_value(self._tags.start_mode_combo)
        start_mode = StartpointMode.MANUAL
        for mode in StartpointMode:
            if mode.value == start_mode_str:
                start_mode = mode
                break

        # Get start/end channels
        start_channel = None
        if start_mode == StartpointMode.MANUAL:
            start_channel = dpg.get_value(self._tags.start_channel)

        auto_end = dpg.get_value(self._tags.auto_end_checkbox)
        end_channel = None
        if not auto_end:
            end_channel = dpg.get_value(self._tags.end_channel)

        # Get background
        background_auto = dpg.get_value(self._tags.background_auto)
        background_value = 0.0
        if not background_auto:
            background_value = dpg.get_value(self._tags.background_value)

        return FittingParameters(
            fit_target=fit_target,
            fit_scope=fit_scope,
            selected_level_index=selected_level_index,
            num_exponentials=num_exp,
            tau_init=tau_inits,
            tau_min=dpg.get_value(self._tags.tau_min),
            tau_max=dpg.get_value(self._tags.tau_max),
            use_irf=dpg.get_value(self._tags.use_irf_checkbox),
            shift_init=dpg.get_value(self._tags.shift_init),
            shift_min=dpg.get_value(self._tags.shift_min),
            shift_max=dpg.get_value(self._tags.shift_max),
            use_simulated_irf=dpg.get_value(self._tags.use_simulated_irf),
            simulated_irf_fwhm=dpg.get_value(self._tags.simulated_irf_fwhm),
            fit_simulated_irf_fwhm=dpg.get_value(self._tags.fit_simulated_irf_fwhm),
            simulated_irf_fwhm_min=dpg.get_value(self._tags.simulated_irf_fwhm_min),
            simulated_irf_fwhm_max=dpg.get_value(self._tags.simulated_irf_fwhm_max),
            start_mode=start_mode,
            start_channel=start_channel,
            auto_end=auto_end,
            end_channel=end_channel,
            background_auto=background_auto,
            background_value=background_value,
        )

    # -------------------------------------------------------------------------
    # UI Event Handlers
    # -------------------------------------------------------------------------

    def _on_fit_target_changed(self, sender: int, app_data: str) -> None:
        """Handle fit target combo change.

        When "Selected Level" is chosen, scope must be locked to "Current".
        """
        is_selected_level = app_data == FitTarget.SELECTED_LEVEL.value

        if dpg.does_item_exist(self._tags.fit_scope_combo):
            if is_selected_level:
                # Lock scope to Current when fitting a selected level
                dpg.set_value(self._tags.fit_scope_combo, FitScope.CURRENT.value)
                dpg.configure_item(self._tags.fit_scope_combo, enabled=False)
            else:
                # Enable scope selection for other targets
                dpg.configure_item(self._tags.fit_scope_combo, enabled=True)

    def _on_num_exp_changed(self, sender: int, app_data: str) -> None:
        """Handle number of exponentials combo change."""
        self._update_tau_rows_visibility()

    def _update_tau_rows_visibility(self) -> None:
        """Update visibility of tau input rows based on num_exponentials."""
        num_exp = int(dpg.get_value(self._tags.num_exp_combo))

        if dpg.does_item_exist(self._tags.tau2_row):
            dpg.configure_item(self._tags.tau2_row, show=(num_exp >= 2))

        if dpg.does_item_exist(self._tags.tau3_row):
            dpg.configure_item(self._tags.tau3_row, show=(num_exp >= 3))

    def _on_use_irf_changed(self, sender: int, app_data: bool) -> None:
        """Handle use IRF checkbox change."""
        # Enable/disable shift controls based on IRF usage
        enabled = app_data
        for tag in [
            self._tags.shift_init,
            self._tags.shift_min,
            self._tags.shift_max,
            self._tags.use_simulated_irf,
        ]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=enabled)

        if not enabled:
            dpg.configure_item(self._tags.simulated_irf_fwhm, enabled=False)
        else:
            self._on_use_simulated_irf_changed(
                self._tags.use_simulated_irf,
                dpg.get_value(self._tags.use_simulated_irf),
            )

    def _on_use_simulated_irf_changed(self, sender: int, app_data: bool) -> None:
        """Handle use simulated IRF checkbox change."""
        if dpg.does_item_exist(self._tags.simulated_irf_fwhm):
            dpg.configure_item(self._tags.simulated_irf_fwhm, enabled=app_data)
        if dpg.does_item_exist(self._tags.fit_simulated_irf_fwhm):
            dpg.configure_item(self._tags.fit_simulated_irf_fwhm, enabled=app_data)
            if not app_data:
                # Hide FWHM bounds row when simulated IRF is disabled
                dpg.configure_item(self._tags.simulated_irf_fwhm_bounds_row, show=False)
            else:
                # Update bounds visibility based on fit FWHM checkbox
                fit_fwhm = dpg.get_value(self._tags.fit_simulated_irf_fwhm)
                dpg.configure_item(self._tags.simulated_irf_fwhm_bounds_row, show=fit_fwhm)

    def _on_fit_fwhm_changed(self, sender: int, app_data: bool) -> None:
        """Handle fit FWHM checkbox change."""
        if dpg.does_item_exist(self._tags.simulated_irf_fwhm_bounds_row):
            dpg.configure_item(self._tags.simulated_irf_fwhm_bounds_row, show=app_data)

    def _on_start_mode_changed(self, sender: int, app_data: str) -> None:
        """Handle start mode combo change."""
        self._update_start_manual_visibility()

    def _update_start_manual_visibility(self) -> None:
        """Update visibility of manual start channel input."""
        start_mode_str = dpg.get_value(self._tags.start_mode_combo)
        is_manual = start_mode_str == StartpointMode.MANUAL.value

        if dpg.does_item_exist(self._tags.start_manual_row):
            dpg.configure_item(self._tags.start_manual_row, show=is_manual)

    def _on_auto_end_changed(self, sender: int, app_data: bool) -> None:
        """Handle auto end checkbox change."""
        self._update_end_manual_visibility()

    def _update_end_manual_visibility(self) -> None:
        """Update visibility of manual end channel input."""
        auto_end = dpg.get_value(self._tags.auto_end_checkbox)

        if dpg.does_item_exist(self._tags.end_manual_row):
            dpg.configure_item(self._tags.end_manual_row, show=not auto_end)

    def _on_bg_auto_changed(self, sender: int, app_data: bool) -> None:
        """Handle background auto checkbox change."""
        self._update_bg_manual_visibility()

    def _update_bg_manual_visibility(self) -> None:
        """Update visibility of manual background input."""
        bg_auto = dpg.get_value(self._tags.background_auto)

        if dpg.does_item_exist(self._tags.background_manual_row):
            dpg.configure_item(self._tags.background_manual_row, show=not bg_auto)

    def _on_fit_clicked(self) -> None:
        """Handle Fit button click."""
        self._parameters = self._collect_parameters()
        self.hide()

        if self._on_fit:
            self._on_fit(self._parameters)

        logger.debug(f"Fit clicked with {self._parameters.num_exponentials} exp")

    def _on_cancel_clicked(self) -> None:
        """Handle Cancel button click."""
        self.hide()

        if self._on_cancel:
            self._on_cancel()

        logger.debug("Fitting dialog cancelled")

    def _on_close(self) -> None:
        """Handle dialog close (X button)."""
        if self._on_cancel:
            self._on_cancel()
