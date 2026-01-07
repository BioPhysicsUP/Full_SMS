"""Tests for decay plot widget."""

import numpy as np
import pytest

from full_sms.ui.plots.decay_plot import DecayPlot, DecayPlotTags, DECAY_PLOT_TAGS


class TestDecayPlotTags:
    """Tests for DecayPlotTags."""

    def test_default_tags(self):
        """Default tags have expected values."""
        tags = DECAY_PLOT_TAGS

        assert tags.container == "decay_plot_container"
        assert tags.plot == "decay_plot"
        assert tags.x_axis == "decay_plot_x_axis"
        assert tags.y_axis == "decay_plot_y_axis"
        assert tags.series == "decay_plot_series"

    def test_custom_tags(self):
        """Custom tags can be created."""
        tags = DecayPlotTags(
            container="custom_container",
            plot="custom_plot",
            x_axis="custom_x",
            y_axis="custom_y",
            series="custom_series",
        )

        assert tags.container == "custom_container"
        assert tags.plot == "custom_plot"


class TestDecayPlotInit:
    """Tests for DecayPlot initialization (no GUI context required)."""

    def test_default_state(self):
        """DecayPlot initializes with correct default state."""
        plot = DecayPlot(parent="test_parent")

        assert plot._parent == "test_parent"
        assert plot._tag_prefix == ""
        assert plot._is_built is False
        assert plot._t is None
        assert plot._counts is None
        assert plot._channelwidth == 0.1
        assert plot._log_scale is True

    def test_custom_prefix(self):
        """DecayPlot accepts custom tag prefix."""
        plot = DecayPlot(parent="test_parent", tag_prefix="custom_")

        assert plot._tag_prefix == "custom_"
        assert plot._tags.container == "custom_decay_plot_container"
        assert plot._tags.plot == "custom_decay_plot"
        assert plot._tags.x_axis == "custom_decay_plot_x_axis"
        assert plot._tags.y_axis == "custom_decay_plot_y_axis"
        assert plot._tags.series == "custom_decay_plot_series"

    def test_properties(self):
        """Properties return expected values."""
        plot = DecayPlot(parent="test_parent")

        assert plot.tags is not None
        assert plot.log_scale is True
        assert plot.channelwidth == 0.1
        assert plot.has_data is False

    def test_has_data_false_when_no_data(self):
        """has_data returns False when no data set."""
        plot = DecayPlot(parent="test_parent")

        assert plot.has_data is False
        assert plot.get_time_range() is None
        assert plot.get_count_range() is None
        assert plot.get_max_counts() is None


class TestDecayPlotDataState:
    """Tests for DecayPlot data state management (no GUI context)."""

    def test_internal_data_storage(self):
        """Internal data arrays can be set manually for testing."""
        plot = DecayPlot(parent="test_parent")

        # Manually set data (bypassing GUI)
        plot._t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        plot._counts = np.array([100, 200, 150, 75, 30], dtype=np.int64)

        assert plot.has_data is True
        assert plot.get_time_range() == (0.0, 4.0)
        assert plot.get_count_range() == (30, 200)
        assert plot.get_max_counts() == 200

    def test_empty_arrays_has_data_false(self):
        """Empty arrays return has_data False."""
        plot = DecayPlot(parent="test_parent")

        plot._t = np.array([])
        plot._counts = np.array([], dtype=np.int64)

        assert plot.has_data is False
        assert plot.get_time_range() is None
        assert plot.get_count_range() is None

    def test_log_scale_toggle_state(self):
        """Log scale state can be toggled internally."""
        plot = DecayPlot(parent="test_parent")

        assert plot._log_scale is True

        # Toggle state directly (GUI update would fail without context)
        plot._log_scale = False
        assert plot._log_scale is False

        plot._log_scale = True
        assert plot._log_scale is True


class TestDecayPlotTagGeneration:
    """Tests for tag generation with different prefixes."""

    def test_empty_prefix(self):
        """Empty prefix generates base tags."""
        plot = DecayPlot(parent="parent", tag_prefix="")
        tags = plot.tags

        assert "decay_plot" in tags.plot
        assert not tags.plot.startswith("_")

    def test_prefix_applied_to_all_tags(self):
        """Prefix is applied to all generated tags."""
        prefix = "my_widget_"
        plot = DecayPlot(parent="parent", tag_prefix=prefix)
        tags = plot.tags

        assert tags.container.startswith(prefix)
        assert tags.plot.startswith(prefix)
        assert tags.x_axis.startswith(prefix)
        assert tags.y_axis.startswith(prefix)
        assert tags.series.startswith(prefix)

    def test_multiple_instances_unique_tags(self):
        """Multiple instances with different prefixes have unique tags."""
        plot1 = DecayPlot(parent="parent", tag_prefix="plot1_")
        plot2 = DecayPlot(parent="parent", tag_prefix="plot2_")

        assert plot1.tags.plot != plot2.tags.plot
        assert plot1.tags.container != plot2.tags.container
        assert plot1.tags.x_axis != plot2.tags.x_axis
        assert plot1.tags.y_axis != plot2.tags.y_axis
        assert plot1.tags.series != plot2.tags.series
