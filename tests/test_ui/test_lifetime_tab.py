"""Tests for lifetime tab view."""

import numpy as np
import pytest

from full_sms.ui.views.lifetime_tab import LifetimeTab, LifetimeTabTags, LIFETIME_TAB_TAGS


class TestLifetimeTabTags:
    """Tests for LifetimeTabTags."""

    def test_default_tags(self):
        """Default tags have expected values."""
        tags = LIFETIME_TAB_TAGS

        assert tags.container == "lifetime_tab_view_container"
        assert tags.controls_group == "lifetime_tab_controls"
        assert tags.log_scale_checkbox == "lifetime_tab_log_scale"
        assert tags.fit_view_button == "lifetime_tab_fit_view"
        assert tags.info_text == "lifetime_tab_info"
        assert tags.plot_container == "lifetime_tab_plot_container"
        assert tags.plot_area == "lifetime_tab_plot_area"
        assert tags.no_data_text == "lifetime_tab_no_data"

    def test_custom_tags(self):
        """Custom tags can be created."""
        tags = LifetimeTabTags(
            container="custom_container",
            controls_group="custom_controls",
            log_scale_checkbox="custom_log",
            fit_view_button="custom_fit",
            info_text="custom_info",
            plot_container="custom_plot_container",
            plot_area="custom_plot_area",
            no_data_text="custom_no_data",
        )

        assert tags.container == "custom_container"
        assert tags.controls_group == "custom_controls"


class TestLifetimeTabInit:
    """Tests for LifetimeTab initialization (no GUI context required)."""

    def test_default_state(self):
        """LifetimeTab initializes with correct default state."""
        tab = LifetimeTab(parent="test_parent")

        assert tab._parent == "test_parent"
        assert tab._tag_prefix == ""
        assert tab._is_built is False
        assert tab._microtimes is None
        assert tab._channelwidth == 0.1
        assert tab._log_scale is True
        assert tab._on_log_scale_changed is None
        assert tab._decay_plot is None

    def test_custom_prefix(self):
        """LifetimeTab accepts custom tag prefix."""
        tab = LifetimeTab(parent="test_parent", tag_prefix="custom_")

        assert tab._tag_prefix == "custom_"
        assert tab._tags.container == "custom_lifetime_tab_view_container"
        assert tab._tags.controls_group == "custom_lifetime_tab_controls"

    def test_properties(self):
        """Properties return expected values."""
        tab = LifetimeTab(parent="test_parent")

        assert tab.tags is not None
        assert tab.log_scale is True
        assert tab.decay_plot is None  # Not built yet
        assert tab.has_data is False
        assert tab.channelwidth == 0.1

    def test_has_data_false_when_no_data(self):
        """has_data returns False when no data set."""
        tab = LifetimeTab(parent="test_parent")

        assert tab.has_data is False


class TestLifetimeTabDataState:
    """Tests for LifetimeTab data state management (no GUI context)."""

    def test_internal_data_storage(self):
        """Internal data arrays can be set manually for testing."""
        tab = LifetimeTab(parent="test_parent")

        # Manually set data (bypassing GUI)
        tab._microtimes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tab._channelwidth = 0.5

        assert tab.has_data is True
        assert tab.channelwidth == 0.5

    def test_empty_arrays_has_data_false(self):
        """Empty arrays return has_data False."""
        tab = LifetimeTab(parent="test_parent")

        tab._microtimes = np.array([])

        assert tab.has_data is False

    def test_log_scale_toggle_state(self):
        """Log scale state can be toggled internally."""
        tab = LifetimeTab(parent="test_parent")

        assert tab._log_scale is True

        # Toggle state directly (GUI update would fail without context)
        tab._log_scale = False
        assert tab._log_scale is False
        assert tab.log_scale is False

        tab._log_scale = True
        assert tab._log_scale is True
        assert tab.log_scale is True


class TestLifetimeTabTagGeneration:
    """Tests for tag generation with different prefixes."""

    def test_empty_prefix(self):
        """Empty prefix generates base tags."""
        tab = LifetimeTab(parent="parent", tag_prefix="")
        tags = tab.tags

        assert "lifetime_tab" in tags.container
        assert not tags.container.startswith("_")

    def test_prefix_applied_to_all_tags(self):
        """Prefix is applied to all generated tags."""
        prefix = "my_widget_"
        tab = LifetimeTab(parent="parent", tag_prefix=prefix)
        tags = tab.tags

        assert tags.container.startswith(prefix)
        assert tags.controls_group.startswith(prefix)
        assert tags.log_scale_checkbox.startswith(prefix)
        assert tags.fit_view_button.startswith(prefix)
        assert tags.info_text.startswith(prefix)
        assert tags.plot_container.startswith(prefix)
        assert tags.plot_area.startswith(prefix)
        assert tags.no_data_text.startswith(prefix)

    def test_multiple_instances_unique_tags(self):
        """Multiple instances with different prefixes have unique tags."""
        tab1 = LifetimeTab(parent="parent", tag_prefix="tab1_")
        tab2 = LifetimeTab(parent="parent", tag_prefix="tab2_")

        assert tab1.tags.container != tab2.tags.container
        assert tab1.tags.controls_group != tab2.tags.controls_group
        assert tab1.tags.log_scale_checkbox != tab2.tags.log_scale_checkbox


class TestLifetimeTabCallbacks:
    """Tests for callback registration."""

    def test_set_log_scale_callback(self):
        """Log scale callback can be registered."""
        tab = LifetimeTab(parent="test_parent")

        callback_called = []

        def my_callback(value: bool) -> None:
            callback_called.append(value)

        tab.set_on_log_scale_changed(my_callback)

        assert tab._on_log_scale_changed is not None

    def test_callback_not_called_on_registration(self):
        """Callback is not called immediately on registration."""
        tab = LifetimeTab(parent="test_parent")

        callback_calls = []

        def my_callback(value: bool) -> None:
            callback_calls.append(value)

        tab.set_on_log_scale_changed(my_callback)

        # Should not be called yet
        assert len(callback_calls) == 0
