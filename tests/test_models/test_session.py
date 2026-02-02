"""Tests for session state models."""

from pathlib import Path

import numpy as np
import pytest

from conftest import make_clustering_result
from full_sms.models import ChannelData, ClusteringResult, FitResultData, GroupData, LevelData, MeasurementData
from full_sms.models.session import (
    ActiveTab,
    ChannelSelection,
    ConfidenceLevel,
    FileMetadata,
    ProcessingState,
    SessionState,
    UIState,
)


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_confidence_values(self) -> None:
        """Confidence levels have correct values."""
        assert ConfidenceLevel.CONF_69.value == 0.69
        assert ConfidenceLevel.CONF_90.value == 0.90
        assert ConfidenceLevel.CONF_95.value == 0.95
        assert ConfidenceLevel.CONF_99.value == 0.99


class TestActiveTab:
    """Tests for ActiveTab enum."""

    def test_all_tabs_defined(self) -> None:
        """All expected tabs are defined."""
        tabs = {t.value for t in ActiveTab}
        expected = {"intensity", "lifetime", "grouping", "spectra", "raster", "correlation", "export"}
        assert tabs == expected


class TestFileMetadata:
    """Tests for FileMetadata."""

    def test_create_metadata(self) -> None:
        """Can create FileMetadata with required fields."""
        metadata = FileMetadata(
            path=Path("/path/to/file.h5"),
            filename="file.h5",
            num_measurements=10,
        )

        assert metadata.path == Path("/path/to/file.h5")
        assert metadata.filename == "file.h5"
        assert metadata.num_measurements == 10
        assert metadata.has_irf is False
        assert metadata.has_spectra is False
        assert metadata.has_raster is False

    def test_create_metadata_with_features(self) -> None:
        """Can create FileMetadata with optional features."""
        metadata = FileMetadata(
            path=Path("/path/to/file.h5"),
            filename="file.h5",
            num_measurements=5,
            has_irf=True,
            has_spectra=True,
            has_raster=True,
        )

        assert metadata.has_irf is True
        assert metadata.has_spectra is True
        assert metadata.has_raster is True


class TestChannelSelection:
    """Tests for ChannelSelection."""

    def test_create_selection(self) -> None:
        """Can create a ChannelSelection."""
        selection = ChannelSelection(measurement_id=1, channel=1)

        assert selection.measurement_id == 1
        assert selection.channel == 1

    def test_default_channel(self) -> None:
        """Default channel is 1."""
        selection = ChannelSelection(measurement_id=5)

        assert selection.channel == 1

    def test_channel_2(self) -> None:
        """Can select channel 2."""
        selection = ChannelSelection(measurement_id=1, channel=2)

        assert selection.channel == 2

    def test_invalid_channel_raises(self) -> None:
        """Invalid channel raises ValueError."""
        with pytest.raises(ValueError, match="channel must be 1 or 2"):
            ChannelSelection(measurement_id=1, channel=3)

        with pytest.raises(ValueError, match="channel must be 1 or 2"):
            ChannelSelection(measurement_id=1, channel=0)


class TestUIState:
    """Tests for UIState."""

    def test_defaults(self) -> None:
        """UIState has sensible defaults."""
        ui = UIState()

        assert ui.bin_size_ms == 10.0
        assert ui.confidence == ConfidenceLevel.CONF_95
        assert ui.active_tab == ActiveTab.INTENSITY
        assert ui.show_levels is True
        assert ui.show_groups is True
        assert ui.log_scale_decay is True

    def test_custom_values(self) -> None:
        """Can create UIState with custom values."""
        ui = UIState(
            bin_size_ms=5.0,
            confidence=ConfidenceLevel.CONF_99,
            active_tab=ActiveTab.LIFETIME,
            show_levels=False,
            show_groups=False,
            log_scale_decay=False,
        )

        assert ui.bin_size_ms == 5.0
        assert ui.confidence == ConfidenceLevel.CONF_99
        assert ui.active_tab == ActiveTab.LIFETIME
        assert ui.show_levels is False
        assert ui.show_groups is False
        assert ui.log_scale_decay is False


class TestProcessingState:
    """Tests for ProcessingState."""

    def test_defaults(self) -> None:
        """ProcessingState starts in idle state."""
        processing = ProcessingState()

        assert processing.is_busy is False
        assert processing.progress == 0.0
        assert processing.message == ""
        assert processing.current_task == ""

    def test_start(self) -> None:
        """start() sets busy state and task."""
        processing = ProcessingState()

        processing.start("Change Point Analysis")

        assert processing.is_busy is True
        assert processing.progress == 0.0
        assert processing.current_task == "Change Point Analysis"
        assert "Starting" in processing.message

    def test_start_with_message(self) -> None:
        """start() can use custom message."""
        processing = ProcessingState()

        processing.start("CPA", message="Analyzing measurement 1...")

        assert processing.message == "Analyzing measurement 1..."

    def test_update(self) -> None:
        """update() updates progress."""
        processing = ProcessingState()
        processing.start("Task")

        processing.update(0.5, "Halfway done")

        assert processing.progress == 0.5
        assert processing.message == "Halfway done"

    def test_update_clamps_progress(self) -> None:
        """update() clamps progress to [0, 1]."""
        processing = ProcessingState()

        processing.update(-0.5)
        assert processing.progress == 0.0

        processing.update(1.5)
        assert processing.progress == 1.0

    def test_finish(self) -> None:
        """finish() sets idle state."""
        processing = ProcessingState()
        processing.start("Task")

        processing.finish()

        assert processing.is_busy is False
        assert processing.progress == 1.0
        assert processing.current_task == ""

    def test_finish_with_message(self) -> None:
        """finish() can use custom message."""
        processing = ProcessingState()
        processing.start("Task")

        processing.finish("All done!")

        assert processing.message == "All done!"


class TestSessionState:
    """Tests for SessionState."""

    @pytest.fixture
    def sample_channel(self) -> ChannelData:
        """Create a sample channel."""
        return ChannelData(
            abstimes=np.array([0, 100_000_000, 200_000_000], dtype=np.uint64),
            microtimes=np.array([1.5, 2.3, 1.8], dtype=np.float64),
        )

    @pytest.fixture
    def sample_measurement(self, sample_channel: ChannelData) -> MeasurementData:
        """Create a sample measurement."""
        return MeasurementData(
            id=1,
            name="Measurement 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )

    @pytest.fixture
    def sample_level(self) -> LevelData:
        """Create a sample level."""
        return LevelData(
            start_index=0,
            end_index=99,
            start_time_ns=0,
            end_time_ns=100_000_000,
            num_photons=100,
            intensity_cps=1000.0,
        )

    def test_empty_state(self) -> None:
        """New SessionState is empty."""
        state = SessionState()

        assert state.file_metadata is None
        assert state.measurements == []
        assert state.selected == []
        assert state.current_selection is None
        assert state.levels == {}
        assert state.clustering_results == {}
        assert state.measurement_fits == {}
        assert state.level_fits == {}
        assert state.has_file is False
        assert state.num_measurements == 0
        assert state.has_selection is False

    def test_has_file(self) -> None:
        """has_file returns True when file is loaded."""
        state = SessionState()

        state.file_metadata = FileMetadata(
            path=Path("/test.h5"),
            filename="test.h5",
            num_measurements=1,
        )

        assert state.has_file is True

    def test_num_measurements(self, sample_measurement: MeasurementData) -> None:
        """num_measurements returns measurement count."""
        state = SessionState()
        state.measurements = [sample_measurement]

        assert state.num_measurements == 1

    def test_get_measurement(self, sample_measurement: MeasurementData, sample_channel: ChannelData) -> None:
        """get_measurement returns measurement by ID."""
        measurement2 = MeasurementData(
            id=2,
            name="Measurement 2",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )
        state = SessionState()
        state.measurements = [sample_measurement, measurement2]

        assert state.get_measurement(1) is sample_measurement
        assert state.get_measurement(2) is measurement2
        assert state.get_measurement(999) is None

    def test_get_current_measurement(self, sample_measurement: MeasurementData) -> None:
        """get_current_measurement returns selected measurement."""
        state = SessionState()
        state.measurements = [sample_measurement]

        # No selection
        assert state.get_current_measurement() is None

        # With selection
        state.select(1)
        assert state.get_current_measurement() is sample_measurement

    def test_select(self, sample_measurement: MeasurementData) -> None:
        """select() sets current selection and adds to selected list."""
        state = SessionState()
        state.measurements = [sample_measurement]

        state.select(1)

        assert state.current_selection is not None
        assert state.current_selection.measurement_id == 1
        assert state.current_selection.channel == 1
        assert state.has_selection is True
        assert len(state.selected) == 1

    def test_select_multiple(self, sample_measurement: MeasurementData, sample_channel: ChannelData) -> None:
        """Selecting multiple measurements adds to list."""
        measurement2 = MeasurementData(
            id=2,
            name="Measurement 2",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )
        state = SessionState()
        state.measurements = [sample_measurement, measurement2]

        state.select(1)
        state.select(2)

        assert len(state.selected) == 2
        assert state.current_selection.measurement_id == 2

    def test_select_same_measurement_no_duplicate(self, sample_measurement: MeasurementData) -> None:
        """Selecting same measurement again doesn't duplicate in list."""
        state = SessionState()
        state.measurements = [sample_measurement]

        state.select(1)
        state.select(1)

        assert len(state.selected) == 1

    def test_select_channel_2(self, sample_measurement: MeasurementData, sample_channel: ChannelData) -> None:
        """Can select channel 2."""
        dual_measurement = MeasurementData(
            id=1,
            name="Measurement 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
            channel2=sample_channel,
        )
        state = SessionState()
        state.measurements = [dual_measurement]

        state.select(1, channel=2)

        assert state.current_selection.channel == 2

    def test_clear_selection(self, sample_measurement: MeasurementData) -> None:
        """clear_selection() removes all selections."""
        state = SessionState()
        state.measurements = [sample_measurement]
        state.select(1)

        state.clear_selection()

        assert state.selected == []
        assert state.current_selection is None
        assert state.has_selection is False

    def test_set_and_get_levels(self, sample_measurement: MeasurementData, sample_level: LevelData) -> None:
        """Can store and retrieve levels."""
        state = SessionState()
        state.measurements = [sample_measurement]
        levels = [sample_level]

        state.set_levels(1, 1, levels)

        assert state.get_levels(1, 1) == levels
        assert state.get_levels(1, 2) is None
        assert state.get_levels(999, 1) is None

    def test_get_current_levels(self, sample_measurement: MeasurementData, sample_level: LevelData) -> None:
        """get_current_levels returns levels for current selection."""
        state = SessionState()
        state.measurements = [sample_measurement]
        levels = [sample_level]
        state.set_levels(1, 1, levels)

        # No selection
        assert state.get_current_levels() is None

        # With selection
        state.select(1)
        assert state.get_current_levels() == levels

    def test_set_and_get_clustering(self, sample_measurement: MeasurementData) -> None:
        """Can store and retrieve clustering results."""
        result = make_clustering_result(num_steps=1, num_original_levels=2)
        state = SessionState()
        state.measurements = [sample_measurement]

        state.set_clustering(1, 1, result)

        assert state.get_clustering(1, 1) is result
        assert state.get_clustering(999, 1) is None

    def test_get_current_clustering(self, sample_measurement: MeasurementData) -> None:
        """get_current_clustering returns result for current selection."""
        result = make_clustering_result(num_steps=1, num_original_levels=1)
        state = SessionState()
        state.measurements = [sample_measurement]
        state.set_clustering(1, 1, result)

        # No selection
        assert state.get_current_clustering() is None

        # With selection
        state.select(1)
        assert state.get_current_clustering() is result

    def test_get_groups(self, sample_measurement: MeasurementData) -> None:
        """get_groups returns groups from clustering result."""
        result = make_clustering_result(num_steps=1, num_original_levels=1)
        state = SessionState()
        state.measurements = [sample_measurement]
        state.set_clustering(1, 1, result)

        groups = state.get_groups(1, 1)

        assert groups is not None
        assert len(groups) == 1
        assert groups[0] is result.groups[0]

    def test_get_groups_no_clustering(self, sample_measurement: MeasurementData) -> None:
        """get_groups returns None if no clustering."""
        state = SessionState()
        state.measurements = [sample_measurement]

        assert state.get_groups(1, 1) is None

    def test_get_current_groups(self, sample_measurement: MeasurementData) -> None:
        """get_current_groups returns groups for current selection."""
        result = make_clustering_result(num_steps=1, num_original_levels=1)
        state = SessionState()
        state.measurements = [sample_measurement]
        state.set_clustering(1, 1, result)

        # No selection
        assert state.get_current_groups() is None

        # With selection
        state.select(1)
        groups = state.get_current_groups()
        assert groups is not None
        assert len(groups) == 1

    def test_set_and_get_measurement_fit(self, sample_measurement: MeasurementData) -> None:
        """Can store and retrieve measurement fit results."""
        fit = FitResultData(
            tau=(5.0,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.01,),
            shift=0.5,
            shift_std=0.1,
            chi_squared=1.0,
            durbin_watson=2.0,
            dw_bounds=(1.5, 2.5),
            fit_start_index=0,
            fit_end_index=100,
            background=10.0,
            num_exponentials=1,
            average_lifetime=5.0,
            level_index=None,
        )
        state = SessionState()
        state.measurements = [sample_measurement]

        state.set_measurement_fit(1, 1, fit)

        assert state.get_measurement_fit(1, 1) is fit
        assert state.get_measurement_fit(999, 1) is None

    def test_set_and_get_level_fit(self, sample_measurement: MeasurementData) -> None:
        """Can store and retrieve level fit results."""
        fit = FitResultData(
            tau=(5.0,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.01,),
            shift=0.5,
            shift_std=0.1,
            chi_squared=1.0,
            durbin_watson=2.0,
            dw_bounds=(1.5, 2.5),
            fit_start_index=0,
            fit_end_index=100,
            background=10.0,
            num_exponentials=1,
            average_lifetime=5.0,
            level_index=0,
        )
        state = SessionState()
        state.measurements = [sample_measurement]

        state.set_level_fit(1, 1, 0, fit)

        assert state.get_level_fit(1, 1, 0) is fit
        assert state.get_level_fit(1, 1, 1) is None
        assert state.get_level_fit(999, 1, 0) is None

    def test_clear_analysis(self, sample_measurement: MeasurementData, sample_level: LevelData) -> None:
        """clear_analysis() removes all analysis results."""
        state = SessionState()
        state.measurements = [sample_measurement]
        state.set_levels(1, 1, [sample_level])

        state.clear_analysis()

        assert state.levels == {}
        assert state.clustering_results == {}
        assert state.measurement_fits == {}
        assert state.level_fits == {}
        assert state.measurements == [sample_measurement]  # Measurements preserved

    def test_reset(self, sample_measurement: MeasurementData, sample_level: LevelData) -> None:
        """reset() returns to initial empty state."""
        state = SessionState()
        state.file_metadata = FileMetadata(
            path=Path("/test.h5"),
            filename="test.h5",
            num_measurements=1,
        )
        state.measurements = [sample_measurement]
        state.select(1)
        state.set_levels(1, 1, [sample_level])
        state.ui_state.bin_size_ms = 5.0
        state.processing.start("Test")

        state.reset()

        assert state.file_metadata is None
        assert state.measurements == []
        assert state.selected == []
        assert state.current_selection is None
        assert state.levels == {}
        assert state.ui_state.bin_size_ms == 10.0  # Default
        assert state.processing.is_busy is False

    def test_ui_state_modifiable(self) -> None:
        """UI state can be modified."""
        state = SessionState()

        state.ui_state.bin_size_ms = 5.0
        state.ui_state.active_tab = ActiveTab.LIFETIME

        assert state.ui_state.bin_size_ms == 5.0
        assert state.ui_state.active_tab == ActiveTab.LIFETIME

    def test_processing_state_workflow(self) -> None:
        """Processing state workflow works."""
        state = SessionState()

        state.processing.start("Analysis")
        assert state.processing.is_busy is True

        state.processing.update(0.5, "Halfway")
        assert state.processing.progress == 0.5

        state.processing.finish("Done")
        assert state.processing.is_busy is False
        assert state.processing.message == "Done"
