"""Particle tree widget for hierarchical particle/channel selection.

Provides a tree view with:
- Tree nodes for each particle
- Nested selectables for SPAD channels
- Checkboxes for batch selection
- Visual indicator for current selection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import dearpygui.dearpygui as dpg

from full_sms.models.particle import ParticleData
from full_sms.models.session import ChannelSelection

logger = logging.getLogger(__name__)


@dataclass
class ParticleTreeTags:
    """Tags for particle tree elements."""

    container: str = "particle_tree_container"
    tree_group: str = "particle_tree_group"


TREE_TAGS = ParticleTreeTags()


# Colors for selection state
COLOR_NORMAL = (200, 200, 200, 255)
COLOR_SELECTED = (100, 180, 255, 255)
COLOR_CURRENT = (255, 200, 100, 255)
COLOR_CHANNEL = (160, 160, 160, 255)
COLOR_CHANNEL_SELECTED = (100, 180, 255, 255)


class ParticleTree:
    """Hierarchical particle/channel selection widget.

    Displays particles in a tree structure with expandable nodes.
    Each particle can have one or two channels (SPAD detectors).
    Supports single selection (current) and multi-selection (batch).
    """

    def __init__(
        self,
        parent: int | str,
        on_selection_changed: Callable[[ChannelSelection | None], None] | None = None,
        on_batch_changed: Callable[[list[ChannelSelection]], None] | None = None,
    ) -> None:
        """Initialize the particle tree.

        Args:
            parent: The parent container to build the tree in.
            on_selection_changed: Callback when the current selection changes.
            on_batch_changed: Callback when the batch selection changes.
        """
        self._parent = parent
        self._on_selection_changed = on_selection_changed
        self._on_batch_changed = on_batch_changed

        self._particles: list[ParticleData] = []
        self._current_selection: ChannelSelection | None = None
        self._batch_selection: set[tuple[int, int]] = set()  # (particle_id, channel)

        # Track UI element tags for updates
        self._particle_nodes: dict[int, str] = {}  # particle_id -> tree node tag
        self._channel_items: dict[tuple[int, int], str] = {}  # (pid, ch) -> selectable tag
        self._checkbox_items: dict[tuple[int, int], str] = {}  # (pid, ch) -> checkbox tag

        self._is_built = False

    def build(self) -> None:
        """Build the particle tree UI structure."""
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
                tag="particle_tree_empty_msg",
            )
            dpg.add_text(
                "Use File > Open H5 to load data",
                color=(100, 100, 100),
                tag="particle_tree_empty_hint",
            )

            # Container for the actual tree (hidden initially)
            with dpg.group(tag=TREE_TAGS.tree_group, show=False):
                pass

        self._is_built = True

    def set_particles(self, particles: list[ParticleData]) -> None:
        """Set the particles to display in the tree.

        This rebuilds the entire tree structure.

        Args:
            particles: List of particles to display.
        """
        self._particles = particles
        self._current_selection = None
        self._batch_selection.clear()
        self._particle_nodes.clear()
        self._channel_items.clear()
        self._checkbox_items.clear()

        # Clear existing tree items
        if dpg.does_item_exist(TREE_TAGS.tree_group):
            dpg.delete_item(TREE_TAGS.tree_group, children_only=True)

        if not particles:
            # Show empty state
            if dpg.does_item_exist("particle_tree_empty_msg"):
                dpg.configure_item("particle_tree_empty_msg", show=True)
            if dpg.does_item_exist("particle_tree_empty_hint"):
                dpg.configure_item("particle_tree_empty_hint", show=True)
            if dpg.does_item_exist(TREE_TAGS.tree_group):
                dpg.configure_item(TREE_TAGS.tree_group, show=False)
            return

        # Hide empty state, show tree
        if dpg.does_item_exist("particle_tree_empty_msg"):
            dpg.configure_item("particle_tree_empty_msg", show=False)
        if dpg.does_item_exist("particle_tree_empty_hint"):
            dpg.configure_item("particle_tree_empty_hint", show=False)
        if dpg.does_item_exist(TREE_TAGS.tree_group):
            dpg.configure_item(TREE_TAGS.tree_group, show=True)

        # Build tree nodes for each particle
        for particle in particles:
            self._build_particle_node(particle)

        logger.info(f"Particle tree updated with {len(particles)} particles")

    def _build_particle_node(self, particle: ParticleData) -> None:
        """Build a tree node for a single particle.

        Args:
            particle: The particle data to display.
        """
        node_tag = f"particle_node_{particle.id}"
        self._particle_nodes[particle.id] = node_tag

        with dpg.tree_node(
            label=particle.name,
            tag=node_tag,
            parent=TREE_TAGS.tree_group,
            default_open=False,
            selectable=False,
        ):
            # Channel 1 (always present)
            self._build_channel_item(particle.id, 1, particle.channel1.num_photons)

            # Channel 2 (if dual-channel)
            if particle.has_dual_channel and particle.channel2 is not None:
                self._build_channel_item(particle.id, 2, particle.channel2.num_photons)

    def _build_channel_item(
        self, particle_id: int, channel: int, num_photons: int
    ) -> None:
        """Build a channel selection item.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            num_photons: Number of photons in this channel.
        """
        key = (particle_id, channel)
        selectable_tag = f"channel_sel_{particle_id}_{channel}"
        checkbox_tag = f"channel_chk_{particle_id}_{channel}"

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
            photon_str = f"{num_photons:,}"
            label = f"Channel {channel} ({photon_str} photons)"
            dpg.add_selectable(
                label=label,
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
            user_data: Tuple of (particle_id, channel).
        """
        particle_id, channel = user_data
        self._set_current_selection(particle_id, channel)

    def _on_checkbox_changed(
        self, sender: int, app_data: bool, user_data: tuple[int, int]
    ) -> None:
        """Handle batch selection checkbox change.

        Args:
            sender: The checkbox widget.
            app_data: The new checked state.
            user_data: Tuple of (particle_id, channel).
        """
        particle_id, channel = user_data
        key = (particle_id, channel)

        if app_data:
            self._batch_selection.add(key)
        else:
            self._batch_selection.discard(key)

        self._notify_batch_changed()

    def _set_current_selection(
        self, particle_id: int, channel: int
    ) -> None:
        """Set the current selection and update visuals.

        Args:
            particle_id: The particle ID.
            channel: The channel number.
        """
        # Clear previous selection visual
        if self._current_selection is not None:
            old_key = (
                self._current_selection.particle_id,
                self._current_selection.channel,
            )
            if old_key in self._channel_items:
                old_tag = self._channel_items[old_key]
                if dpg.does_item_exist(old_tag):
                    dpg.set_value(old_tag, False)

        # Set new selection
        self._current_selection = ChannelSelection(
            particle_id=particle_id, channel=channel
        )

        # Update visual
        key = (particle_id, channel)
        if key in self._channel_items:
            tag = self._channel_items[key]
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, True)

        # Also ensure it's in batch selection
        if key not in self._batch_selection:
            self._batch_selection.add(key)
            if key in self._checkbox_items:
                chk_tag = self._checkbox_items[key]
                if dpg.does_item_exist(chk_tag):
                    dpg.set_value(chk_tag, True)
            self._notify_batch_changed()

        # Expand parent node if collapsed
        if particle_id in self._particle_nodes:
            node_tag = self._particle_nodes[particle_id]
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
                ChannelSelection(particle_id=pid, channel=ch)
                for pid, ch in sorted(self._batch_selection)
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
            ChannelSelection(particle_id=pid, channel=ch)
            for pid, ch in sorted(self._batch_selection)
        ]

    @property
    def batch_count(self) -> int:
        """Get the number of items selected for batch operations."""
        return len(self._batch_selection)

    def select(self, particle_id: int, channel: int = 1) -> None:
        """Programmatically select a particle/channel.

        Args:
            particle_id: The particle ID.
            channel: The channel number (default 1).
        """
        self._set_current_selection(particle_id, channel)

    def select_all(self) -> None:
        """Select all particles/channels for batch operations."""
        for particle in self._particles:
            key1 = (particle.id, 1)
            self._batch_selection.add(key1)
            if key1 in self._checkbox_items:
                chk_tag = self._checkbox_items[key1]
                if dpg.does_item_exist(chk_tag):
                    dpg.set_value(chk_tag, True)

            if particle.has_dual_channel:
                key2 = (particle.id, 2)
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
                self._current_selection.particle_id,
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
        """Expand all particle tree nodes."""
        for node_tag in self._particle_nodes.values():
            if dpg.does_item_exist(node_tag):
                dpg.set_value(node_tag, True)

    def collapse_all(self) -> None:
        """Collapse all particle tree nodes."""
        for node_tag in self._particle_nodes.values():
            if dpg.does_item_exist(node_tag):
                dpg.set_value(node_tag, False)

    def get_particle_ids(self) -> list[int]:
        """Get list of all particle IDs."""
        return [p.id for p in self._particles]
