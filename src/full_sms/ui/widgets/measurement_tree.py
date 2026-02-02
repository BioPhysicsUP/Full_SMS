"""Measurement tree widget for hierarchical measurement/channel selection.

Provides a tree view with:
- Tree nodes for each measurement
- Nested selectables for SPAD channels
- Checkboxes for batch selection
- Visual indicator for current selection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import dearpygui.dearpygui as dpg

from full_sms.models.measurement import MeasurementData
from full_sms.models.session import ChannelSelection

logger = logging.getLogger(__name__)


@dataclass
class MeasurementTreeTags:
    """Tags for measurement tree elements."""

    container: str = "measurement_tree_container"
    tree_group: str = "measurement_tree_group"


TREE_TAGS = MeasurementTreeTags()


# Colors for selection state
COLOR_NORMAL = (200, 200, 200, 255)
COLOR_SELECTED = (100, 180, 255, 255)
COLOR_CURRENT = (255, 200, 100, 255)
COLOR_CHANNEL = (160, 160, 160, 255)
COLOR_CHANNEL_SELECTED = (100, 180, 255, 255)


class MeasurementTree:
    """Hierarchical measurement/channel selection widget.

    Displays measurements in a tree structure with expandable nodes.
    Each measurement can have one or two channels (SPAD detectors).
    Supports single selection (current) and multi-selection (batch).
    """

    def __init__(
        self,
        parent: int | str,
        on_selection_changed: Callable[[ChannelSelection | None], None] | None = None,
        on_batch_changed: Callable[[list[ChannelSelection]], None] | None = None,
    ) -> None:
        """Initialize the measurement tree.

        Args:
            parent: The parent container to build the tree in.
            on_selection_changed: Callback when the current selection changes.
            on_batch_changed: Callback when the batch selection changes.
        """
        self._parent = parent
        self._on_selection_changed = on_selection_changed
        self._on_batch_changed = on_batch_changed

        self._measurements: list[MeasurementData] = []
        self._current_selection: ChannelSelection | None = None
        self._batch_selection: set[tuple[int, int]] = set()  # (measurement_id, channel)

        # Track UI element tags for updates
        self._measurement_nodes: dict[int, str] = {}  # measurement_id -> tree node tag
        self._channel_items: dict[tuple[int, int], str] = {}  # (mid, ch) -> selectable tag
        self._checkbox_items: dict[tuple[int, int], str] = {}  # (mid, ch) -> checkbox tag

        self._is_built = False

    def build(self) -> None:
        """Build the measurement tree UI structure."""
        if self._is_built:
            return

        # Create container for the tree
        with dpg.child_window(
            tag=TREE_TAGS.container,
            parent=self._parent,
            autosize_x=True,
            autosize_y=True,
            border=False,
        ):
            # Initial empty state message
            dpg.add_text(
                "No file loaded",
                color=(128, 128, 128),
                tag="measurement_tree_empty_msg",
            )
            dpg.add_text(
                "File > Open H5",
                color=(100, 100, 100),
                tag="measurement_tree_empty_hint",
            )

            # Container for the actual tree (hidden initially)
            with dpg.group(tag=TREE_TAGS.tree_group, show=False):
                pass

        self._is_built = True

    def set_measurements(self, measurements: list[MeasurementData]) -> None:
        """Set the measurements to display in the tree.

        This rebuilds the entire tree structure.

        Args:
            measurements: List of measurements to display.
        """
        self._measurements = measurements
        self._current_selection = None
        self._batch_selection.clear()
        self._measurement_nodes.clear()
        self._channel_items.clear()
        self._checkbox_items.clear()

        # Clear existing tree items
        if dpg.does_item_exist(TREE_TAGS.tree_group):
            dpg.delete_item(TREE_TAGS.tree_group, children_only=True)

        if not measurements:
            # Show empty state
            if dpg.does_item_exist("measurement_tree_empty_msg"):
                dpg.configure_item("measurement_tree_empty_msg", show=True)
            if dpg.does_item_exist("measurement_tree_empty_hint"):
                dpg.configure_item("measurement_tree_empty_hint", show=True)
            if dpg.does_item_exist(TREE_TAGS.tree_group):
                dpg.configure_item(TREE_TAGS.tree_group, show=False)
            return

        # Hide empty state, show tree
        if dpg.does_item_exist("measurement_tree_empty_msg"):
            dpg.configure_item("measurement_tree_empty_msg", show=False)
        if dpg.does_item_exist("measurement_tree_empty_hint"):
            dpg.configure_item("measurement_tree_empty_hint", show=False)
        if dpg.does_item_exist(TREE_TAGS.tree_group):
            dpg.configure_item(TREE_TAGS.tree_group, show=True)

        # Build tree nodes for each measurement
        for measurement in measurements:
            self._build_measurement_node(measurement)

        logger.info(f"Measurement tree updated with {len(measurements)} measurements")

    def _build_measurement_node(self, measurement: MeasurementData) -> None:
        """Build a tree node for a single measurement.

        For single-channel measurements, creates a flat selectable item.
        For dual-channel measurements, creates a tree node with nested channels.

        Args:
            measurement: The measurement data to display.
        """
        # Single-channel measurements: flat item without nesting
        if not measurement.has_dual_channel:
            self._build_single_channel_item(measurement)
            return

        # Dual-channel measurements: tree node with nested channels
        node_tag = f"measurement_node_{measurement.id}"
        self._measurement_nodes[measurement.id] = node_tag

        with dpg.tree_node(
            label=measurement.name,
            tag=node_tag,
            parent=TREE_TAGS.tree_group,
            default_open=False,
            selectable=False,
        ):
            # Channel 1 (always present)
            self._build_channel_item(measurement.id, 1)

            # Channel 2
            if measurement.channel2 is not None:
                self._build_channel_item(measurement.id, 2)

    def _build_single_channel_item(self, measurement: MeasurementData) -> None:
        """Build a flat item for a single-channel measurement.

        Args:
            measurement: The measurement data to display.
        """
        key = (measurement.id, 1)
        selectable_tag = f"channel_sel_{measurement.id}_1"
        checkbox_tag = f"channel_chk_{measurement.id}_1"

        self._channel_items[key] = selectable_tag
        self._checkbox_items[key] = checkbox_tag

        with dpg.group(horizontal=True, parent=TREE_TAGS.tree_group):
            # Checkbox for batch selection
            dpg.add_checkbox(
                tag=checkbox_tag,
                default_value=False,
                callback=self._on_checkbox_changed,
                user_data=key,
            )

            # Selectable label showing measurement name directly
            dpg.add_selectable(
                label=measurement.name,
                tag=selectable_tag,
                default_value=False,
                callback=self._on_selectable_clicked,
                user_data=key,
            )

    def _build_channel_item(self, measurement_id: int, channel: int) -> None:
        """Build a channel selection item.

        Args:
            measurement_id: The measurement ID.
            channel: The channel number (1 or 2).
        """
        key = (measurement_id, channel)
        selectable_tag = f"channel_sel_{measurement_id}_{channel}"
        checkbox_tag = f"channel_chk_{measurement_id}_{channel}"

        self._channel_items[key] = selectable_tag
        self._checkbox_items[key] = checkbox_tag

        with dpg.group(horizontal=True):
            # Checkbox for batch selection
            dpg.add_checkbox(
                tag=checkbox_tag,
                default_value=False,
                callback=self._on_checkbox_changed,
                user_data=key,
            )

            # Selectable label for current selection
            dpg.add_selectable(
                label=f"Channel {channel}",
                tag=selectable_tag,
                default_value=False,
                callback=self._on_selectable_clicked,
                user_data=key,
            )

    def _on_selectable_clicked(
        self, sender: int, app_data: bool, user_data: tuple[int, int]
    ) -> None:
        """Handle channel selectable click.

        Args:
            sender: The selectable widget.
            app_data: Whether it's now selected (always True on click).
            user_data: Tuple of (measurement_id, channel).
        """
        measurement_id, channel = user_data
        self._set_current_selection(measurement_id, channel)

    def _on_checkbox_changed(
        self, sender: int, app_data: bool, user_data: tuple[int, int]
    ) -> None:
        """Handle batch selection checkbox change.

        Args:
            sender: The checkbox widget.
            app_data: The new checked state.
            user_data: Tuple of (measurement_id, channel).
        """
        measurement_id, channel = user_data
        key = (measurement_id, channel)

        if app_data:
            self._batch_selection.add(key)
        else:
            self._batch_selection.discard(key)

        self._notify_batch_changed()

    def _set_current_selection(
        self, measurement_id: int, channel: int
    ) -> None:
        """Set the current selection and update visuals.

        Args:
            measurement_id: The measurement ID.
            channel: The channel number.
        """
        # Clear previous selection visual
        if self._current_selection is not None:
            old_key = (
                self._current_selection.measurement_id,
                self._current_selection.channel,
            )
            if old_key in self._channel_items:
                old_tag = self._channel_items[old_key]
                if dpg.does_item_exist(old_tag):
                    dpg.set_value(old_tag, False)

        # Set new selection
        self._current_selection = ChannelSelection(
            measurement_id=measurement_id, channel=channel
        )

        # Update visual
        key = (measurement_id, channel)
        if key in self._channel_items:
            tag = self._channel_items[key]
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, True)

        # Expand parent node if collapsed (only for dual-channel measurements)
        if measurement_id in self._measurement_nodes:
            node_tag = self._measurement_nodes[measurement_id]
            if dpg.does_item_exist(node_tag):
                dpg.set_value(node_tag, True)

        # Notify callback
        self._notify_selection_changed()

    def _notify_selection_changed(self) -> None:
        """Notify the selection changed callback."""
        if self._on_selection_changed:
            self._on_selection_changed(self._current_selection)

    def _notify_batch_changed(self) -> None:
        """Notify the batch selection changed callback."""
        if self._on_batch_changed:
            selections = [
                ChannelSelection(measurement_id=mid, channel=ch)
                for mid, ch in sorted(self._batch_selection)
            ]
            self._on_batch_changed(selections)

    @property
    def current_selection(self) -> ChannelSelection | None:
        """Get the current (primary) selection."""
        return self._current_selection

    @property
    def batch_selection(self) -> list[ChannelSelection]:
        """Get all selected items for batch operations."""
        return [
            ChannelSelection(measurement_id=mid, channel=ch)
            for mid, ch in sorted(self._batch_selection)
        ]

    @property
    def batch_count(self) -> int:
        """Get the number of items selected for batch operations."""
        return len(self._batch_selection)

    def select(self, measurement_id: int, channel: int = 1) -> None:
        """Programmatically select a measurement/channel.

        Args:
            measurement_id: The measurement ID.
            channel: The channel number (default 1).
        """
        self._set_current_selection(measurement_id, channel)

    def select_all(self) -> None:
        """Select all measurements/channels for batch operations."""
        for measurement in self._measurements:
            key1 = (measurement.id, 1)
            self._batch_selection.add(key1)
            if key1 in self._checkbox_items:
                chk_tag = self._checkbox_items[key1]
                if dpg.does_item_exist(chk_tag):
                    dpg.set_value(chk_tag, True)

            if measurement.has_dual_channel:
                key2 = (measurement.id, 2)
                self._batch_selection.add(key2)
                if key2 in self._checkbox_items:
                    chk_tag = self._checkbox_items[key2]
                    if dpg.does_item_exist(chk_tag):
                        dpg.set_value(chk_tag, True)

        self._notify_batch_changed()
        logger.debug(f"Selected all: {len(self._batch_selection)} items")

    def clear_selection(self) -> None:
        """Clear all selections."""
        # Clear current selection visual
        if self._current_selection is not None:
            old_key = (
                self._current_selection.measurement_id,
                self._current_selection.channel,
            )
            if old_key in self._channel_items:
                old_tag = self._channel_items[old_key]
                if dpg.does_item_exist(old_tag):
                    dpg.set_value(old_tag, False)

        self._current_selection = None

        # Clear batch selection
        for key in self._batch_selection:
            if key in self._checkbox_items:
                chk_tag = self._checkbox_items[key]
                if dpg.does_item_exist(chk_tag):
                    dpg.set_value(chk_tag, False)

        self._batch_selection.clear()

        self._notify_selection_changed()
        self._notify_batch_changed()
        logger.debug("Cleared all selections")

    def expand_all(self) -> None:
        """Expand all measurement tree nodes."""
        for node_tag in self._measurement_nodes.values():
            if dpg.does_item_exist(node_tag):
                dpg.set_value(node_tag, True)

    def collapse_all(self) -> None:
        """Collapse all measurement tree nodes."""
        for node_tag in self._measurement_nodes.values():
            if dpg.does_item_exist(node_tag):
                dpg.set_value(node_tag, False)

    def get_measurement_ids(self) -> list[int]:
        """Get list of all measurement IDs."""
        return [m.id for m in self._measurements]

    def clear(self) -> None:
        """Clear the tree and reset to empty state.

        Called when closing a file.
        """
        self.set_measurements([])
        logger.debug("Measurement tree cleared")
