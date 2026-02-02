"""Grouping analysis tab view.

Provides the BIC optimization curve visualization and group count selection
for hierarchical clustering results. Includes an intensity plot with group
band visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.models.group import ClusteringResult, GroupData
from full_sms.models.level import LevelData
from full_sms.ui.plots.bic_plot import BICPlot
from full_sms.ui.plots.intensity_plot import IntensityPlot

# Height of the compact intensity plot in the grouping tab
GROUPING_INTENSITY_PLOT_HEIGHT = 400

logger = logging.getLogger(__name__)


@dataclass
class GroupingTabTags:
    """Tags for grouping tab elements."""

    container: str = "grouping_tab_view_container"
    controls_group: str = "grouping_tab_controls"
    controls_row2: str = "grouping_tab_controls_row2"
    group_button: str = "grouping_tab_group_button"
    group_selected_button: str = "grouping_tab_group_selected_button"
    group_all_button: str = "grouping_tab_group_all_button"
    reset_button: str = "grouping_tab_reset_button"
    fit_view_button: str = "grouping_tab_fit_view"
    info_text: str = "grouping_tab_info"
    plot_container: str = "grouping_tab_plot_container"
    plot_area: str = "grouping_tab_plot_area"
    no_data_text: str = "grouping_tab_no_data"
    # Grouping options
    use_lifetime_checkbox: str = "grouping_tab_use_lifetime"
    global_grouping_checkbox: str = "grouping_tab_global_grouping"
    group_count_slider: str = "grouping_tab_group_count"
    group_count_label: str = "grouping_tab_group_count_label"
    # Results display
    results_group: str = "grouping_tab_results_group"
    results_header: str = "grouping_tab_results_header"
    groups_text: str = "grouping_tab_groups_text"
    bic_text: str = "grouping_tab_bic_text"
    optimal_text: str = "grouping_tab_optimal_text"
    # Group list
    group_list_group: str = "grouping_tab_group_list_group"
    group_list_header: str = "grouping_tab_group_list_header"
    group_table: str = "grouping_tab_group_table"
    # Intensity plot for group visualization
    intensity_plot_group: str = "grouping_tab_intensity_plot_group"
    intensity_plot_container: str = "grouping_tab_intensity_plot_container"


GROUPING_TAB_TAGS = GroupingTabTags()


class GroupingTab:
    """Grouping analysis tab view.

    Contains the BIC optimization plot and controls for viewing clustering results
    and selecting the number of groups.
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the grouping tab.

        Args:
            parent: The parent container to build the tab in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._clustering_result: Optional[ClusteringResult] = None
        self._selected_num_groups: int = 0
        self._selected_group_id: int | None = None  # Currently selected group for highlighting

        # Callbacks
        self._on_group_count_changed: Callable[[int], None] | None = None
        self._on_grouping_requested: Callable[[str], None] | None = None
        self._on_group_selected: Callable[[int | None], None] | None = None
        self._on_grouping_options_changed: Callable[[bool, bool], None] | None = None

        # Grouping options
        self._use_lifetime: bool = False
        self._global_grouping: bool = False

        # Intensity plot data state
        self._abstimes: NDArray[np.uint64] | None = None
        self._bin_size_ms: float = 10.0
        self._levels: list[LevelData] | None = None
        self._group_band_tags: list[str] = []  # Tags for group band shade series

        # UI components
        self._bic_plot: BICPlot | None = None
        self._intensity_plot: IntensityPlot | None = None

        # Generate unique tags
        self._tags = GroupingTabTags(
            container=f"{tag_prefix}grouping_tab_view_container",
            controls_group=f"{tag_prefix}grouping_tab_controls",
            controls_row2=f"{tag_prefix}grouping_tab_controls_row2",
            group_button=f"{tag_prefix}grouping_tab_group_button",
            group_selected_button=f"{tag_prefix}grouping_tab_group_selected_button",
            group_all_button=f"{tag_prefix}grouping_tab_group_all_button",
            reset_button=f"{tag_prefix}grouping_tab_reset_button",
            fit_view_button=f"{tag_prefix}grouping_tab_fit_view",
            info_text=f"{tag_prefix}grouping_tab_info",
            plot_container=f"{tag_prefix}grouping_tab_plot_container",
            plot_area=f"{tag_prefix}grouping_tab_plot_area",
            no_data_text=f"{tag_prefix}grouping_tab_no_data",
            use_lifetime_checkbox=f"{tag_prefix}grouping_tab_use_lifetime",
            global_grouping_checkbox=f"{tag_prefix}grouping_tab_global_grouping",
            group_count_slider=f"{tag_prefix}grouping_tab_group_count",
            group_count_label=f"{tag_prefix}grouping_tab_group_count_label",
            results_group=f"{tag_prefix}grouping_tab_results_group",
            results_header=f"{tag_prefix}grouping_tab_results_header",
            groups_text=f"{tag_prefix}grouping_tab_groups_text",
            bic_text=f"{tag_prefix}grouping_tab_bic_text",
            optimal_text=f"{tag_prefix}grouping_tab_optimal_text",
            group_list_group=f"{tag_prefix}grouping_tab_group_list_group",
            group_list_header=f"{tag_prefix}grouping_tab_group_list_header",
            group_table=f"{tag_prefix}grouping_tab_group_table",
            intensity_plot_group=f"{tag_prefix}grouping_tab_intensity_plot_group",
            intensity_plot_container=f"{tag_prefix}grouping_tab_intensity_plot_container",
        )

    @property
    def tags(self) -> GroupingTabTags:
        """Get the tags for this tab instance."""
        return self._tags

    @property
    def bic_plot(self) -> BICPlot | None:
        """Get the BIC plot widget."""
        return self._bic_plot

    @property
    def has_data(self) -> bool:
        """Whether the tab has clustering data loaded."""
        return self._clustering_result is not None

    @property
    def intensity_plot(self) -> IntensityPlot | None:
        """Get the intensity plot widget."""
        return self._intensity_plot

    @property
    def clustering_result(self) -> Optional[ClusteringResult]:
        """Get the current clustering result."""
        return self._clustering_result

    @property
    def selected_num_groups(self) -> int:
        """Get the currently selected number of groups."""
        return self._selected_num_groups

    @property
    def selected_group_id(self) -> int | None:
        """Get the currently selected group ID (for highlighting)."""
        return self._selected_group_id

    @property
    def use_lifetime(self) -> bool:
        """Whether to include lifetime in clustering."""
        return self._use_lifetime

    def set_use_lifetime(self, value: bool) -> None:
        """Set whether to use lifetime in clustering.

        Args:
            value: Whether to use lifetime.
        """
        self._use_lifetime = value
        if self._is_built and dpg.does_item_exist(self._tags.use_lifetime_checkbox):
            dpg.set_value(self._tags.use_lifetime_checkbox, value)

    @property
    def global_grouping(self) -> bool:
        """Whether global grouping mode is enabled."""
        return self._global_grouping

    def set_global_grouping(self, value: bool) -> None:
        """Set whether global grouping mode is enabled.

        Args:
            value: Whether to use global grouping.
        """
        self._global_grouping = value
        if self._is_built and dpg.does_item_exist(self._tags.global_grouping_checkbox):
            dpg.set_value(self._tags.global_grouping_checkbox, value)

    def build(self) -> None:
        """Build the tab UI structure."""
        if self._is_built:
            return

        # Main container
        with dpg.group(parent=self._parent, tag=self._tags.container):
            # Controls bar at top
            self._build_controls()

            # Results display (hidden until clustering is done)
            self._build_results_display()

            # Separator
            dpg.add_separator()

            # Main content area with plot and group list side by side
            with dpg.child_window(
                tag=self._tags.plot_container,
                border=False,
                autosize_x=True,
                autosize_y=True,
            ):
                # No data placeholder (shown when no clustering done)
                dpg.add_text(
                    "Run change point analysis and grouping to view BIC optimization curve.",
                    tag=self._tags.no_data_text,
                    color=(128, 128, 128),
                )

                # Plot area (hidden until data loaded)
                with dpg.group(
                    tag=self._tags.plot_area,
                    show=False,
                ):
                    # Intensity plot with group bands (top, compact)
                    with dpg.group(
                        tag=self._tags.intensity_plot_group,
                        show=False,  # Hidden until intensity data is set
                    ):
                        with dpg.child_window(
                            tag=self._tags.intensity_plot_container,
                            border=False,
                            autosize_x=True,
                            height=GROUPING_INTENSITY_PLOT_HEIGHT,
                        ):
                            self._intensity_plot = IntensityPlot(
                                parent=dpg.last_container(),
                                tag_prefix=f"{self._tag_prefix}grouping_int_",
                            )
                            self._intensity_plot.build(show_legend=False)

                    # Horizontal layout for BIC plot and group list
                    with dpg.group(horizontal=True):
                        # Left side: BIC plot
                        with dpg.child_window(
                            border=False,
                            width=-320,  # Leave room for group list
                            autosize_y=True,
                        ):
                            self._bic_plot = BICPlot(
                                parent=dpg.last_item(),
                                tag_prefix=f"{self._tag_prefix}main_",
                            )
                            self._bic_plot.build()
                            self._bic_plot.set_on_group_selected(self._on_bic_point_selected)

                        # Right side: Group list
                        self._build_group_list()

        self._is_built = True
        logger.debug("Grouping tab built")

    def _build_controls(self) -> None:
        """Build the controls bar at the top of the tab."""
        # First row: Group buttons and options
        with dpg.group(horizontal=True, tag=self._tags.controls_group):
            # Group buttons
            dpg.add_button(
                label="Group Current",
                tag=self._tags.group_button,
                callback=lambda: self._on_group_button_clicked("current"),
                enabled=False,
            )

            dpg.add_spacer(width=5)

            dpg.add_button(
                label="Group Selected",
                tag=self._tags.group_selected_button,
                callback=lambda: self._on_group_button_clicked("selected"),
                enabled=False,
            )

            dpg.add_spacer(width=5)

            dpg.add_button(
                label="Group All",
                tag=self._tags.group_all_button,
                callback=lambda: self._on_group_button_clicked("all"),
                enabled=False,
            )

            # Spacer
            dpg.add_spacer(width=20)

            # Use lifetime checkbox
            dpg.add_checkbox(
                label="Use Lifetime",
                tag=self._tags.use_lifetime_checkbox,
                default_value=False,
                callback=self._on_use_lifetime_changed,
            )

            dpg.add_spacer(width=15)

            # Global grouping checkbox
            dpg.add_checkbox(
                label="Global Grouping",
                tag=self._tags.global_grouping_checkbox,
                default_value=False,
                callback=self._on_global_grouping_changed,
            )
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("Group all selected measurements together\nas a single dataset")

            # Spacer
            dpg.add_spacer(width=30)

            # Info text
            dpg.add_text(
                "",
                tag=self._tags.info_text,
                color=(128, 128, 128),
            )

        # Second row: Post-clustering controls
        with dpg.group(horizontal=True, tag=self._tags.controls_row2):
            # Reset to optimal button
            dpg.add_button(
                label="Reset to Optimal",
                tag=self._tags.reset_button,
                callback=self._on_reset_button_clicked,
                enabled=False,
            )

            dpg.add_spacer(width=15)

            # Fit view button
            dpg.add_button(
                label="Fit View",
                tag=self._tags.fit_view_button,
                callback=self._on_fit_view_clicked,
                enabled=False,
            )

            dpg.add_spacer(width=30)

            # Manual group count override
            dpg.add_text("Groups:")
            dpg.add_slider_int(
                tag=self._tags.group_count_slider,
                default_value=1,
                min_value=1,
                max_value=100,
                width=150,
                callback=self._on_group_count_slider_changed,
                enabled=False,
            )
            dpg.add_text(
                "1",
                tag=self._tags.group_count_label,
                color=(180, 180, 180),
            )

    def _build_results_display(self) -> None:
        """Build the results display section."""
        # Results group (hidden until clustering is done)
        with dpg.group(
            tag=self._tags.results_group,
            horizontal=True,
            show=False,
        ):
            # Results header
            dpg.add_text(
                "Clustering:",
                tag=self._tags.results_header,
                color=(180, 180, 180),
            )

            dpg.add_spacer(width=15)

            # Number of groups
            dpg.add_text(
                "",
                tag=self._tags.groups_text,
                color=(100, 180, 255),  # Blue
            )

            dpg.add_spacer(width=20)

            # BIC value
            dpg.add_text(
                "",
                tag=self._tags.bic_text,
                color=(100, 220, 150),  # Green
            )

            dpg.add_spacer(width=20)

            # Optimal indicator
            dpg.add_text(
                "",
                tag=self._tags.optimal_text,
                color=(255, 220, 100),  # Yellow
            )

    def _build_group_list(self) -> None:
        """Build the group list panel."""
        with dpg.child_window(
            tag=self._tags.group_list_group,
            width=310,
            border=True,
            autosize_y=True,
        ):
            # Header
            dpg.add_text(
                "Groups",
                tag=self._tags.group_list_header,
                color=(180, 180, 180),
            )
            dpg.add_separator()

            # Group table
            with dpg.table(
                tag=self._tags.group_table,
                header_row=True,
                borders_innerH=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_outerV=True,
                row_background=True,
                resizable=True,
            ):
                dpg.add_table_column(label="Group", width_fixed=True, init_width_or_weight=30)
                dpg.add_table_column(label="Levels")
                dpg.add_table_column(label="Int. (cps)")
                dpg.add_table_column(label="Dwell (s)")

    def _update_group_table(self, groups: tuple[GroupData, ...]) -> None:
        """Update the group table with current groups.

        Args:
            groups: Tuple of GroupData objects to display.
        """
        if not dpg.does_item_exist(self._tags.group_table):
            return

        # Clear existing rows (keep header columns)
        children = dpg.get_item_children(self._tags.group_table, 1)
        if children:
            for child in children:
                dpg.delete_item(child)

        # Create rows with clickable selectables in the ID column
        for group in groups:
            is_selected = group.group_id == self._selected_group_id

            with dpg.table_row(parent=self._tags.group_table):
                # First column - ID with selectable for clicking
                dpg.add_selectable(
                    label=str(group.group_id + 1),  # 1-indexed for display
                    default_value=is_selected,
                    callback=self._on_group_row_clicked,
                    user_data=group.group_id,
                )
                # Data columns
                dpg.add_text(str(group.num_levels))
                dpg.add_text(f"{group.intensity_cps:,.0f}")
                dpg.add_text(f"{group.total_dwell_time_s:.3f}")

    def _on_group_row_clicked(self, sender: int, app_data: bool, user_data: int) -> None:
        """Handle group row click.

        Args:
            sender: The selectable widget.
            app_data: Whether the selectable is now selected.
            user_data: The group_id.
        """
        group_id = user_data

        if app_data:
            # Select this group
            self._selected_group_id = group_id
        else:
            # Deselect if clicking the same row again
            if self._selected_group_id == group_id:
                self._selected_group_id = None

        # Update the table to reflect selection state
        if self._clustering_result:
            self._update_group_table(self._clustering_result.groups)

        # Notify callback
        if self._on_group_selected:
            self._on_group_selected(self._selected_group_id)

        logger.debug(f"Group selection changed to {self._selected_group_id}")

    def set_clustering_result(self, result: ClusteringResult) -> None:
        """Set the clustering result and update displays.

        Args:
            result: The ClusteringResult from hierarchical clustering.
        """
        self._clustering_result = result
        self._selected_num_groups = result.num_groups

        # Update BIC plot
        if self._bic_plot:
            self._bic_plot.set_clustering_result(result)

        # Update group table
        self._update_group_table(result.groups)

        # Update group count slider range and value
        if dpg.does_item_exist(self._tags.group_count_slider):
            # Get the range of possible group counts from the steps
            min_groups = min(step.num_groups for step in result.steps)
            max_groups = max(step.num_groups for step in result.steps)
            dpg.configure_item(
                self._tags.group_count_slider,
                min_value=min_groups,
                max_value=max_groups,
            )
            dpg.set_value(self._tags.group_count_slider, result.num_groups)

        # Update group count label
        if dpg.does_item_exist(self._tags.group_count_label):
            dpg.set_value(self._tags.group_count_label, str(result.num_groups))

        # Show the plot area, hide placeholder
        self._show_plot(True)

        # Enable controls
        self._enable_controls(True)

        # Update results display
        self._update_results_text()

        # Show results section
        if dpg.does_item_exist(self._tags.results_group):
            dpg.configure_item(self._tags.results_group, show=True)

        logger.debug(
            f"Grouping tab updated: {result.num_steps} steps, "
            f"{result.num_groups} groups selected"
        )

    def clear(self) -> None:
        """Clear the tab data."""
        self._clustering_result = None
        self._selected_num_groups = 0
        self._selected_group_id = None

        if self._bic_plot:
            self._bic_plot.clear()

        # Clear intensity plot data
        self._abstimes = None
        self._levels = None
        self._clear_group_bands()
        if self._intensity_plot:
            self._intensity_plot.clear()

        # Hide intensity plot
        if dpg.does_item_exist(self._tags.intensity_plot_group):
            dpg.configure_item(self._tags.intensity_plot_group, show=False)

        # Clear the group table
        if dpg.does_item_exist(self._tags.group_table):
            children = dpg.get_item_children(self._tags.group_table, 1)
            if children:
                for child in children:
                    dpg.delete_item(child)

        # Hide plot, show placeholder
        self._show_plot(False)

        # Disable controls
        self._enable_controls(False)

        # Hide results section
        if dpg.does_item_exist(self._tags.results_group):
            dpg.configure_item(self._tags.results_group, show=False)

        # Clear info text
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, "")

        logger.debug("Grouping tab cleared")

    def _show_plot(self, show: bool) -> None:
        """Show or hide the plot area.

        Args:
            show: Whether to show the plot area (True) or placeholder (False).
        """
        if dpg.does_item_exist(self._tags.plot_area):
            dpg.configure_item(self._tags.plot_area, show=show)

        if dpg.does_item_exist(self._tags.no_data_text):
            dpg.configure_item(self._tags.no_data_text, show=not show)

    def _enable_controls(self, enable: bool) -> None:
        """Enable or disable control buttons.

        Args:
            enable: Whether to enable the controls.
        """
        for tag in [
            self._tags.reset_button,
            self._tags.fit_view_button,
            self._tags.group_count_slider,
        ]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=enable)

    def _update_results_text(self) -> None:
        """Update the results display text."""
        if self._clustering_result is None:
            return

        result = self._clustering_result

        # Number of groups
        if dpg.does_item_exist(self._tags.groups_text):
            dpg.set_value(
                self._tags.groups_text,
                f"{result.num_groups} groups"
            )

        # BIC value
        if dpg.does_item_exist(self._tags.bic_text):
            dpg.set_value(
                self._tags.bic_text,
                f"BIC = {result.selected_bic:.2f}"
            )

        # Optimal indicator
        if dpg.does_item_exist(self._tags.optimal_text):
            if result.is_optimal_selected:
                dpg.set_value(self._tags.optimal_text, "(optimal)")
                dpg.configure_item(self._tags.optimal_text, color=(100, 220, 150, 255))
            else:
                optimal_groups = result.steps[result.optimal_step_index].num_groups
                dpg.set_value(
                    self._tags.optimal_text,
                    f"(optimal: {optimal_groups} groups)"
                )
                dpg.configure_item(self._tags.optimal_text, color=(255, 220, 100, 255))

    def _on_bic_point_selected(self, num_groups: int) -> None:
        """Handle BIC plot click to select group count.

        Args:
            num_groups: The number of groups selected.
        """
        if self._clustering_result is None:
            return

        # Find the step with this group count
        for i, step in enumerate(self._clustering_result.steps):
            if step.num_groups == num_groups:
                # Update the clustering result with new selection
                self._clustering_result = self._clustering_result.with_selected_step(i)
                self._selected_num_groups = num_groups

                # Update slider and label
                if dpg.does_item_exist(self._tags.group_count_slider):
                    dpg.set_value(self._tags.group_count_slider, num_groups)
                if dpg.does_item_exist(self._tags.group_count_label):
                    dpg.set_value(self._tags.group_count_label, str(num_groups))

                # Update displays
                self._update_group_table(self._clustering_result.groups)
                self._update_results_text()

                # Update group bands on intensity plot
                self.update_group_bands()

                # Call callback if set
                if self._on_group_count_changed:
                    self._on_group_count_changed(num_groups)

                logger.debug(f"User selected {num_groups} groups")
                break

    def _on_group_button_clicked(self, mode: str) -> None:
        """Handle group button click.

        Args:
            mode: The grouping mode ("current", "selected", or "all").
        """
        if self._on_grouping_requested:
            self._on_grouping_requested(mode)
        logger.debug(f"Group button clicked: mode={mode}")

    def _on_use_lifetime_changed(self, sender: int, app_data: bool) -> None:
        """Handle use lifetime checkbox change.

        Args:
            sender: The checkbox widget.
            app_data: Whether to use lifetime in clustering.
        """
        self._use_lifetime = app_data
        logger.debug(f"Use lifetime changed to {app_data}")
        # Notify callback if set (app will save to session)
        if self._on_grouping_options_changed:
            self._on_grouping_options_changed(
                use_lifetime=app_data, global_grouping=self._global_grouping
            )

    def _on_global_grouping_changed(self, sender: int, app_data: bool) -> None:
        """Handle global grouping checkbox change.

        Args:
            sender: The checkbox widget.
            app_data: Whether to use global grouping mode.
        """
        self._global_grouping = app_data
        logger.debug(f"Global grouping changed to {app_data}")
        # Notify callback if set (app will save to session)
        if self._on_grouping_options_changed:
            self._on_grouping_options_changed(
                use_lifetime=self._use_lifetime, global_grouping=app_data
            )

    def _on_group_count_slider_changed(self, sender: int, app_data: int) -> None:
        """Handle group count slider change.

        Args:
            sender: The slider widget.
            app_data: The new group count value.
        """
        if self._clustering_result is None:
            return

        # Update the label
        if dpg.does_item_exist(self._tags.group_count_label):
            dpg.set_value(self._tags.group_count_label, str(app_data))

        # Find the step with this group count
        for i, step in enumerate(self._clustering_result.steps):
            if step.num_groups == app_data:
                self._clustering_result = self._clustering_result.with_selected_step(i)
                self._selected_num_groups = app_data

                # Update BIC plot selection
                if self._bic_plot:
                    self._bic_plot.set_selected_num_groups(app_data)

                # Update displays
                self._update_group_table(self._clustering_result.groups)
                self._update_results_text()

                # Update group bands on intensity plot
                self.update_group_bands()

                # Call callback
                if self._on_group_count_changed:
                    self._on_group_count_changed(app_data)

                logger.debug(f"Group count slider set to {app_data}")
                break

    def _on_reset_button_clicked(self) -> None:
        """Handle reset to optimal button click."""
        if self._clustering_result is None:
            return

        # Reset to optimal step
        optimal_index = self._clustering_result.optimal_step_index
        optimal_groups = self._clustering_result.steps[optimal_index].num_groups

        self._clustering_result = self._clustering_result.with_selected_step(optimal_index)
        self._selected_num_groups = optimal_groups

        # Update BIC plot selection
        if self._bic_plot:
            self._bic_plot.set_selected_num_groups(optimal_groups)

        # Update slider and label
        if dpg.does_item_exist(self._tags.group_count_slider):
            dpg.set_value(self._tags.group_count_slider, optimal_groups)
        if dpg.does_item_exist(self._tags.group_count_label):
            dpg.set_value(self._tags.group_count_label, str(optimal_groups))

        # Update displays
        self._update_group_table(self._clustering_result.groups)
        self._update_results_text()

        # Update group bands on intensity plot
        self.update_group_bands()

        # Call callback
        if self._on_group_count_changed:
            self._on_group_count_changed(optimal_groups)

        logger.debug(f"Reset to optimal: {optimal_groups} groups")

    def _on_fit_view_clicked(self) -> None:
        """Handle fit view button click."""
        if self._bic_plot:
            self._bic_plot.fit_view()

    def set_on_group_count_changed(
        self, callback: Callable[[int], None]
    ) -> None:
        """Set callback for when group count changes.

        Args:
            callback: Function called when user selects different group count.
                Receives the new number of groups.
        """
        self._on_group_count_changed = callback

    def set_on_grouping_requested(
        self, callback: Callable[[str], None]
    ) -> None:
        """Set callback for when grouping button is clicked.

        Args:
            callback: Function called when user clicks a Group button.
                Receives the mode ("current", "selected", or "all").
        """
        self._on_grouping_requested = callback

    def set_on_group_selected(
        self, callback: Callable[[int | None], None]
    ) -> None:
        """Set callback for when a group is selected/deselected in the table.

        Args:
            callback: Function called when group selection changes.
                Receives the group_id (0-indexed) or None if deselected.
        """
        self._on_group_selected = callback

    def set_on_grouping_options_changed(
        self, callback: Callable[[bool, bool], None]
    ) -> None:
        """Set callback for when grouping options change.

        Args:
            callback: Function called when use_lifetime or global_grouping changes.
                Receives (use_lifetime, global_grouping) as parameters.
        """
        self._on_grouping_options_changed = callback

    def set_selected_group(self, group_id: int | None) -> None:
        """Programmatically set the selected group for highlighting.

        Args:
            group_id: The group ID to select (0-indexed), or None to clear.
        """
        if self._selected_group_id == group_id:
            return

        self._selected_group_id = group_id

        # Update the table display
        if self._clustering_result:
            self._update_group_table(self._clustering_result.groups)

        logger.debug(f"Selected group set to {group_id}")

    def clear_selected_group(self) -> None:
        """Clear the currently selected group."""
        self.set_selected_group(None)

    def enable_group_button(self, enabled: bool = True) -> None:
        """Enable or disable the Group Current button.

        Args:
            enabled: Whether to enable the button.
        """
        if dpg.does_item_exist(self._tags.group_button):
            dpg.configure_item(self._tags.group_button, enabled=enabled)

    def set_group_buttons_state(
        self,
        has_current: bool = False,
        has_selected: bool = False,
        has_any: bool = False,
    ) -> None:
        """Enable/disable group buttons based on application state.

        Args:
            has_current: Whether there is a currently viewed measurement with levels.
            has_selected: Whether there are batch-selected measurements with levels.
            has_any: Whether there are any measurements with levels loaded.
        """
        if dpg.does_item_exist(self._tags.group_button):
            dpg.configure_item(self._tags.group_button, enabled=has_current)
        if dpg.does_item_exist(self._tags.group_selected_button):
            dpg.configure_item(self._tags.group_selected_button, enabled=has_selected)
        if dpg.does_item_exist(self._tags.group_all_button):
            dpg.configure_item(self._tags.group_all_button, enabled=has_any)

    def set_grouping(self, is_grouping: bool) -> None:
        """Set the grouping state, disabling buttons while analysis is running.

        Args:
            is_grouping: Whether clustering analysis is currently running.
        """
        # Disable all group buttons during analysis
        for tag in [
            self._tags.group_button,
            self._tags.group_selected_button,
            self._tags.group_all_button,
        ]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=not is_grouping)

    def set_selected_groups(self, num_groups: int) -> None:
        """Programmatically set the selected number of groups.

        Args:
            num_groups: Number of groups to select.
        """
        if self._clustering_result is None:
            return

        # Find the step with this group count
        for i, step in enumerate(self._clustering_result.steps):
            if step.num_groups == num_groups:
                self._clustering_result = self._clustering_result.with_selected_step(i)
                self._selected_num_groups = num_groups

                # Update BIC plot
                if self._bic_plot:
                    self._bic_plot.set_selected_num_groups(num_groups)

                # Update displays
                self._update_group_table(self._clustering_result.groups)
                self._update_results_text()

                # Update group bands on intensity plot
                self.update_group_bands()
                break

    def update_info(self, text: str) -> None:
        """Update the info text display.

        Args:
            text: Text to display in the info area.
        """
        if dpg.does_item_exist(self._tags.info_text):
            dpg.set_value(self._tags.info_text, text)

    # -------------------------------------------------------------------------
    # Intensity Plot Methods
    # -------------------------------------------------------------------------

    def set_intensity_data(
        self,
        abstimes: NDArray[np.uint64],
        bin_size_ms: float = 10.0,
    ) -> None:
        """Set the intensity data for the plot.

        Args:
            abstimes: Absolute photon arrival times in nanoseconds.
            bin_size_ms: Bin size in milliseconds.
        """
        self._abstimes = abstimes
        self._bin_size_ms = bin_size_ms

        if self._intensity_plot:
            self._intensity_plot.set_data(abstimes, bin_size_ms)

        logger.debug(f"Grouping tab intensity data set: {len(abstimes)} photons")

    def set_levels_with_groups(
        self,
        levels: Sequence[LevelData],
        clustering: ClusteringResult | None = None,
    ) -> None:
        """Set levels and render group bands on the intensity plot.

        After setting levels, if clustering is available, renders alternating
        color bands for each group to visualize grouping on the intensity trace.

        Args:
            levels: Sequence of LevelData objects.
            clustering: Optional clustering result with group assignments.
        """
        self._levels = list(levels)

        if not self._intensity_plot:
            return

        # Show the intensity plot group
        if dpg.does_item_exist(self._tags.intensity_plot_group):
            dpg.configure_item(self._tags.intensity_plot_group, show=True)

        # Ensure intensity data is set on the plot (may not have been rendered
        # if the plot was hidden when set_intensity_data was called)
        if self._abstimes is not None:
            self._intensity_plot.set_data(self._abstimes, self._bin_size_ms)

        # Set levels on the intensity plot (shows step line)
        self._intensity_plot.set_levels(levels, color_by_group=False)

        # If clustering is available, render group bands
        if clustering:
            self._render_group_bands(levels, clustering)

        # Fix Y-axis to data range
        self._intensity_plot.fix_y_axis_to_data()

        logger.debug(
            f"Grouping tab levels set: {len(levels)} levels, "
            f"clustering={'yes' if clustering else 'no'}"
        )

    def _render_group_bands(
        self,
        levels: Sequence[LevelData],
        clustering: ClusteringResult,
    ) -> None:
        """Render alternating horizontal bands for groups on the intensity plot.

        Calculates non-overlapping bounds between groups based on midpoints
        of adjacent group intensities. Draws bands for alternating groups
        and dashed lines at each group's average intensity.

        Args:
            levels: The levels (needed for intensity values).
            clustering: The clustering result with group assignments.
        """
        if not self._intensity_plot:
            return

        # Clear any existing group bands
        self._clear_group_bands()

        y_axis_tag = self._intensity_plot.tags.y_axis
        if not dpg.does_item_exist(y_axis_tag):
            return

        # Get the full time range for bands to span
        time_range = self._intensity_plot.get_time_range()
        if not time_range:
            return
        x_min, x_max = time_range

        # Get groups and calculate their average intensities
        groups = clustering.groups
        if len(groups) < 2:
            return

        # Convert group intensities to counts/bin and sort descending
        int_conv = self._bin_size_ms / 1000.0 if self._bin_size_ms > 0 else 0.01
        group_ints = [(g.group_id, g.intensity_cps * int_conv) for g in groups]
        group_ints.sort(key=lambda x: x[1], reverse=True)  # Sort descending by intensity

        # Calculate non-overlapping bounds between groups (midpoints)
        # Format: list of (group_id, intensity, lower_bound, upper_bound)
        bounds_data: list[tuple[int, float, float, float]] = []

        for i, (group_id, g_int) in enumerate(group_ints):
            if i == 0:
                # Highest intensity group: upper bound is infinity (use large value)
                upper = g_int * 2  # Large enough to cover the plot
                if i == len(group_ints) - 1:
                    # Only one group
                    lower = 0
                else:
                    # Midpoint with next group
                    lower = (g_int + group_ints[i + 1][1]) / 2
            elif i == len(group_ints) - 1:
                # Lowest intensity group: lower bound is 0
                upper = prev_mid
                lower = 0
            else:
                # Middle groups: bounds are midpoints with neighbors
                upper = prev_mid
                lower = (g_int + group_ints[i + 1][1]) / 2

            prev_mid = lower if i < len(group_ints) - 1 else 0
            bounds_data.append((group_id, g_int, lower, upper))

        # Band color (semi-transparent light green for visibility)
        band_color = (60, 200, 120, 130)  # Light green

        # Draw bands for alternating groups (every other one, starting from index 1)
        for i, (group_id, g_int, lower, upper) in enumerate(bounds_data):
            # Draw band for odd indices (alternating pattern)
            if i % 2 == 1:
                band_tag = f"{self._tag_prefix}group_band_{i}"
                self._group_band_tags.append(band_tag)

                # X coordinates span the full time range
                x_coords = [x_min, x_max]
                # Y coordinates define the horizontal band
                y1_coords = [lower, lower]
                y2_coords = [upper, upper]

                dpg.add_shade_series(
                    x_coords,
                    y1_coords,
                    y2=y2_coords,
                    parent=y_axis_tag,
                    tag=band_tag,
                )
                self._apply_group_band_theme(band_tag, band_color)

            # Draw dashed line at group average intensity
            # Create dashed effect by drawing segments with gaps
            line_tag = f"{self._tag_prefix}group_line_{i}"
            self._group_band_tags.append(line_tag)

            # Generate dashed line points (dash-gap pattern)
            total_width = x_max - x_min
            num_dashes = 40  # Number of dash segments
            dash_width = total_width / (num_dashes * 2)  # Dash and gap equal width

            x_dashes = []
            y_dashes = []
            for d in range(num_dashes):
                dash_start = x_min + d * 2 * dash_width
                dash_end = dash_start + dash_width
                # Add dash segment with NaN to create gap
                x_dashes.extend([dash_start, dash_end, float('nan')])
                y_dashes.extend([g_int, g_int, float('nan')])

            dpg.add_line_series(
                x_dashes,
                y_dashes,
                parent=y_axis_tag,
                tag=line_tag,
            )
            self._apply_group_line_theme(line_tag)

        logger.debug(f"Rendered {len(self._group_band_tags)} group bands and lines")

    def _apply_group_band_theme(
        self,
        band_tag: str,
        color: tuple[int, int, int, int],
    ) -> None:
        """Apply a theme to a group band shade series.

        Args:
            band_tag: The tag of the shade series.
            color: RGBA color tuple for the band fill.
        """
        if not dpg.does_item_exist(band_tag):
            return

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvShadeSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Fill,
                    color,
                    category=dpg.mvThemeCat_Plots,
                )

        dpg.bind_item_theme(band_tag, theme)

    def _apply_group_line_theme(self, line_tag: str) -> None:
        """Apply a theme for group average intensity lines.

        Args:
            line_tag: The tag of the line series.
        """
        if not dpg.does_item_exist(line_tag):
            return

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (230, 220, 60, 260),
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight,
                    0.5,
                    category=dpg.mvThemeCat_Plots,
                )

        dpg.bind_item_theme(line_tag, theme)

    def _clear_group_bands(self) -> None:
        """Remove all group band shade series from the intensity plot."""
        for tag in self._group_band_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self._group_band_tags.clear()

    def update_group_bands(self) -> None:
        """Update group bands after group count selection changes.

        Call this when the user changes the selected number of groups.
        """
        if not self._levels or not self._clustering_result:
            return

        self._render_group_bands(self._levels, self._clustering_result)
