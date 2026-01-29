"""Tests for session save/load functionality."""

import json
from pathlib import Path

import numpy as np
import pytest

from full_sms.io.session import (
    SESSION_VERSION,
    SessionSerializationError,
    _clustering_to_dict,
    _dict_to_clustering,
    _dict_to_fit_result_data,
    _dict_to_group,
    _dict_to_level,
    _dict_to_selection,
    _dict_to_ui_state,
    _fit_result_data_to_dict,
    _group_to_dict,
    _level_to_dict,
    _selection_to_dict,
    _str_to_tuple_key,
    _tuple_key_to_str,
    _ui_state_to_dict,
    apply_session_to_state,
    load_session,
    save_session,
    scan_sessions_for_uuid,
)
from conftest import make_clustering_result
from full_sms.models import ChannelData, ClusteringResult, ClusteringStep, FitResultData, GroupData, LevelData, ParticleData
from full_sms.models.session import (
    ActiveTab,
    ChannelSelection,
    ConfidenceLevel,
    FileMetadata,
    SessionState,
    UIState,
)


class TestTupleKeyConversion:
    """Tests for tuple key to string conversion."""

    def test_tuple_to_str(self) -> None:
        """Tuple keys convert to comma-separated strings."""
        assert _tuple_key_to_str((1, 2)) == "1,2"
        assert _tuple_key_to_str((10, 1, 5)) == "10,1,5"
        assert _tuple_key_to_str((0,)) == "0"

    def test_str_to_tuple(self) -> None:
        """String keys convert back to tuples with correct types."""
        assert _str_to_tuple_key("1,2", (int, int)) == (1, 2)
        assert _str_to_tuple_key("10,1,5", (int, int, int)) == (10, 1, 5)

    def test_str_to_tuple_wrong_length_raises(self) -> None:
        """Wrong number of parts raises error."""
        with pytest.raises(SessionSerializationError, match="has 2 parts, expected 3"):
            _str_to_tuple_key("1,2", (int, int, int))


class TestLevelSerialization:
    """Tests for LevelData serialization."""

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
            group_id=2,
        )

    def test_level_to_dict(self, sample_level: LevelData) -> None:
        """LevelData converts to dict correctly."""
        d = _level_to_dict(sample_level)

        assert d["start_index"] == 0
        assert d["end_index"] == 99
        assert d["start_time_ns"] == 0
        assert d["end_time_ns"] == 100_000_000
        assert d["num_photons"] == 100
        assert d["intensity_cps"] == 1000.0
        assert d["group_id"] == 2

    def test_dict_to_level(self, sample_level: LevelData) -> None:
        """Dict converts back to LevelData correctly."""
        d = _level_to_dict(sample_level)
        restored = _dict_to_level(d)

        assert restored == sample_level

    def test_level_roundtrip_none_group(self) -> None:
        """Level with None group_id roundtrips correctly."""
        level = LevelData(
            start_index=0,
            end_index=50,
            start_time_ns=0,
            end_time_ns=50_000_000,
            num_photons=50,
            intensity_cps=1000.0,
            group_id=None,
        )

        d = _level_to_dict(level)
        restored = _dict_to_level(d)

        assert restored.group_id is None


class TestGroupSerialization:
    """Tests for GroupData serialization."""

    @pytest.fixture
    def sample_group(self) -> GroupData:
        """Create a sample group."""
        return GroupData(
            group_id=1,
            level_indices=(0, 2, 5),
            total_photons=500,
            total_dwell_time_s=2.5,
            intensity_cps=200.0,
        )

    def test_group_to_dict(self, sample_group: GroupData) -> None:
        """GroupData converts to dict correctly."""
        d = _group_to_dict(sample_group)

        assert d["group_id"] == 1
        assert d["level_indices"] == [0, 2, 5]  # Tuple becomes list
        assert d["total_photons"] == 500
        assert d["total_dwell_time_s"] == 2.5
        assert d["intensity_cps"] == 200.0

    def test_dict_to_group(self, sample_group: GroupData) -> None:
        """Dict converts back to GroupData correctly."""
        d = _group_to_dict(sample_group)
        restored = _dict_to_group(d)

        assert restored == sample_group


class TestClusteringResultSerialization:
    """Tests for ClusteringResult serialization."""

    @pytest.fixture
    def sample_clustering(self) -> ClusteringResult:
        """Create a sample clustering result."""
        return make_clustering_result(
            num_steps=3, num_original_levels=3, optimal_step_index=1, selected_step_index=1
        )

    def test_clustering_to_dict(self, sample_clustering: ClusteringResult) -> None:
        """ClusteringResult converts to dict correctly."""
        d = _clustering_to_dict(sample_clustering)

        assert len(d["steps"]) == 3
        assert d["optimal_step_index"] == 1
        assert d["selected_step_index"] == 1
        assert d["num_original_levels"] == 3
        # Check that each step has expected structure
        assert "groups" in d["steps"][0]
        assert "bic" in d["steps"][0]
        assert "level_group_assignments" in d["steps"][0]

    def test_dict_to_clustering(self, sample_clustering: ClusteringResult) -> None:
        """Dict converts back to ClusteringResult correctly."""
        d = _clustering_to_dict(sample_clustering)
        restored = _dict_to_clustering(d)

        assert restored.num_groups == sample_clustering.num_groups
        assert restored.all_bic_values == sample_clustering.all_bic_values
        assert restored.optimal_step_index == sample_clustering.optimal_step_index
        assert restored.num_steps == sample_clustering.num_steps


class TestFitResultDataSerialization:
    """Tests for FitResultData serialization."""

    @pytest.fixture
    def sample_fit(self) -> FitResultData:
        """Create a sample fit result data."""
        return FitResultData(
            tau=(5.0,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.01,),
            shift=0.5,
            shift_std=0.05,
            chi_squared=1.05,
            durbin_watson=2.1,
            dw_bounds=(1.5, 2.5),
            fit_start_index=10,
            fit_end_index=500,
            background=5.0,
            num_exponentials=1,
            average_lifetime=5.0,
            level_index=None,
        )

    def test_fit_to_dict(self, sample_fit: FitResultData) -> None:
        """FitResultData converts to dict correctly."""
        d = _fit_result_data_to_dict(sample_fit)

        assert d["tau"] == [5.0]
        assert d["tau_std"] == [0.1]
        assert d["amplitude"] == [1.0]
        assert d["chi_squared"] == 1.05
        assert d["dw_bounds"] == [1.5, 2.5]

    def test_dict_to_fit(self, sample_fit: FitResultData) -> None:
        """Dict converts back to FitResultData correctly."""
        d = _fit_result_data_to_dict(sample_fit)
        restored = _dict_to_fit_result_data(d)

        assert restored.tau == sample_fit.tau
        assert restored.chi_squared == sample_fit.chi_squared

    def test_fit_none_dw_bounds(self) -> None:
        """FitResultData with None dw_bounds roundtrips correctly."""
        fit = FitResultData(
            tau=(5.0,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.01,),
            shift=0.5,
            shift_std=0.05,
            chi_squared=1.05,
            durbin_watson=2.1,
            dw_bounds=None,
            fit_start_index=10,
            fit_end_index=500,
            background=5.0,
            num_exponentials=1,
            average_lifetime=5.0,
            level_index=None,
        )

        d = _fit_result_data_to_dict(fit)
        restored = _dict_to_fit_result_data(d)

        assert restored.dw_bounds is None

    def test_fit_with_level_index(self) -> None:
        """FitResultData with level_index roundtrips correctly."""
        fit = FitResultData(
            tau=(5.0,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.01,),
            shift=0.5,
            shift_std=0.05,
            chi_squared=1.05,
            durbin_watson=2.1,
            dw_bounds=None,
            fit_start_index=10,
            fit_end_index=500,
            background=5.0,
            num_exponentials=1,
            average_lifetime=5.0,
            level_index=3,
        )

        d = _fit_result_data_to_dict(fit)
        restored = _dict_to_fit_result_data(d)

        assert restored.level_index == 3


class TestUIStateSerialization:
    """Tests for UIState serialization."""

    def test_ui_state_to_dict(self) -> None:
        """UIState converts to dict correctly."""
        ui = UIState(
            bin_size_ms=5.0,
            confidence=ConfidenceLevel.CONF_99,
            active_tab=ActiveTab.LIFETIME,
            show_levels=False,
            show_groups=True,
            log_scale_decay=False,
        )

        d = _ui_state_to_dict(ui)

        assert d["bin_size_ms"] == 5.0
        assert d["confidence"] == 0.99
        assert d["active_tab"] == "lifetime"
        assert d["show_levels"] is False
        assert d["show_groups"] is True
        assert d["log_scale_decay"] is False

    def test_dict_to_ui_state(self) -> None:
        """Dict converts back to UIState correctly."""
        d = {
            "bin_size_ms": 5.0,
            "confidence": 0.99,
            "active_tab": "lifetime",
            "show_levels": False,
            "show_groups": True,
            "log_scale_decay": False,
        }

        ui = _dict_to_ui_state(d)

        assert ui.bin_size_ms == 5.0
        assert ui.confidence == ConfidenceLevel.CONF_99
        assert ui.active_tab == ActiveTab.LIFETIME
        assert ui.show_levels is False


class TestSelectionSerialization:
    """Tests for ChannelSelection serialization."""

    def test_selection_roundtrip(self) -> None:
        """ChannelSelection roundtrips correctly."""
        sel = ChannelSelection(particle_id=5, channel=2)

        d = _selection_to_dict(sel)
        restored = _dict_to_selection(d)

        assert restored == sel


class TestSaveSession:
    """Tests for save_session function."""

    @pytest.fixture
    def sample_channel(self) -> ChannelData:
        """Create a sample channel."""
        return ChannelData(
            abstimes=np.array([0, 100_000_000, 200_000_000], dtype=np.uint64),
            microtimes=np.array([1.5, 2.3, 1.8], dtype=np.float64),
        )

    @pytest.fixture
    def sample_state(self, sample_channel: ChannelData) -> SessionState:
        """Create a sample session state with analysis results."""
        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )

        state = SessionState()
        state.file_metadata = FileMetadata(
            path=Path("/path/to/test.h5"),
            filename="test.h5",
            num_particles=1,
            has_irf=True,
        )
        state.particles = [particle]
        state.select(1)

        # Add a level
        level = LevelData(
            start_index=0,
            end_index=2,
            start_time_ns=0,
            end_time_ns=200_000_000,
            num_photons=3,
            intensity_cps=15.0,
        )
        state.set_levels(1, 1, [level])

        # Add clustering
        clustering = make_clustering_result(num_steps=1, num_original_levels=1)
        state.set_clustering(1, 1, clustering)

        # Modify UI state
        state.ui_state.bin_size_ms = 5.0
        state.ui_state.confidence = ConfidenceLevel.CONF_90

        return state

    def test_save_creates_file(self, sample_state: SessionState, tmp_path: Path) -> None:
        """save_session creates a file."""
        session_path = tmp_path / "test.smsa"

        save_session(sample_state, session_path)

        assert session_path.exists()

    def test_save_creates_valid_json(self, sample_state: SessionState, tmp_path: Path) -> None:
        """save_session creates valid JSON."""
        session_path = tmp_path / "test.smsa"

        save_session(sample_state, session_path)

        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == SESSION_VERSION

    def test_save_includes_file_metadata(self, sample_state: SessionState, tmp_path: Path) -> None:
        """save_session includes file metadata."""
        session_path = tmp_path / "test.smsa"

        save_session(sample_state, session_path)

        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["file_metadata"]["filename"] == "test.h5"
        assert data["file_metadata"]["has_irf"] is True

    def test_save_includes_levels(self, sample_state: SessionState, tmp_path: Path) -> None:
        """save_session includes levels."""
        session_path = tmp_path / "test.smsa"

        save_session(sample_state, session_path)

        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "1,1" in data["levels"]
        assert len(data["levels"]["1,1"]) == 1

    def test_save_includes_clustering(self, sample_state: SessionState, tmp_path: Path) -> None:
        """save_session includes clustering results."""
        session_path = tmp_path / "test.smsa"

        save_session(sample_state, session_path)

        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "1,1" in data["clustering_results"]

    def test_save_includes_ui_state(self, sample_state: SessionState, tmp_path: Path) -> None:
        """save_session includes UI state."""
        session_path = tmp_path / "test.smsa"

        save_session(sample_state, session_path)

        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["ui_state"]["bin_size_ms"] == 5.0
        assert data["ui_state"]["confidence"] == 0.90

    def test_save_without_file_raises(self, tmp_path: Path) -> None:
        """save_session raises if no file is loaded."""
        state = SessionState()  # No file loaded
        session_path = tmp_path / "test.smsa"

        with pytest.raises(SessionSerializationError, match="without loaded file"):
            save_session(state, session_path)


class TestLoadSession:
    """Tests for load_session function."""

    @pytest.fixture
    def sample_session_file(self, tmp_path: Path) -> Path:
        """Create a sample session file."""
        data = {
            "version": SESSION_VERSION,
            "file_metadata": {
                "path": "/path/to/test.h5",
                "filename": "test.h5",
                "num_particles": 2,
                "has_irf": True,
                "has_spectra": False,
                "has_raster": False,
            },
            "levels": {
                "1,1": [
                    {
                        "start_index": 0,
                        "end_index": 99,
                        "start_time_ns": 0,
                        "end_time_ns": 100_000_000,
                        "num_photons": 100,
                        "intensity_cps": 1000.0,
                        "group_id": None,
                    }
                ]
            },
            "clustering_results": {},
            "particle_fits": {},
            "level_fits": {},
            "selected": [{"particle_id": 1, "channel": 1}],
            "current_selection": {"particle_id": 1, "channel": 1},
            "ui_state": {
                "bin_size_ms": 5.0,
                "confidence": 0.95,
                "active_tab": "intensity",
                "show_levels": True,
                "show_groups": True,
                "log_scale_decay": True,
            },
        }

        session_path = tmp_path / "test.smsa"
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return session_path

    def test_load_returns_dict(self, sample_session_file: Path) -> None:
        """load_session returns a dictionary."""
        result = load_session(sample_session_file)

        assert isinstance(result, dict)

    def test_load_includes_version(self, sample_session_file: Path) -> None:
        """load_session includes version."""
        result = load_session(sample_session_file)

        assert result["version"] == SESSION_VERSION

    def test_load_parses_file_metadata(self, sample_session_file: Path) -> None:
        """load_session parses file metadata."""
        result = load_session(sample_session_file)

        assert result["file_metadata"]["path"] == Path("/path/to/test.h5")
        assert result["file_metadata"]["filename"] == "test.h5"
        assert result["file_metadata"]["has_irf"] is True

    def test_load_parses_levels(self, sample_session_file: Path) -> None:
        """load_session parses levels."""
        result = load_session(sample_session_file)

        assert (1, 1) in result["levels"]
        levels = result["levels"][(1, 1)]
        assert len(levels) == 1
        assert isinstance(levels[0], LevelData)
        assert levels[0].num_photons == 100

    def test_load_parses_selections(self, sample_session_file: Path) -> None:
        """load_session parses selections."""
        result = load_session(sample_session_file)

        assert len(result["selected"]) == 1
        assert isinstance(result["selected"][0], ChannelSelection)
        assert result["current_selection"].particle_id == 1

    def test_load_parses_ui_state(self, sample_session_file: Path) -> None:
        """load_session parses UI state."""
        result = load_session(sample_session_file)

        assert isinstance(result["ui_state"], UIState)
        assert result["ui_state"].bin_size_ms == 5.0

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """load_session raises for missing file."""
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nonexistent.smsa")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """load_session raises for invalid JSON."""
        invalid_file = tmp_path / "invalid.smsa"
        with open(invalid_file, "w") as f:
            f.write("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_session(invalid_file)

    def test_load_wrong_version_raises(self, tmp_path: Path) -> None:
        """load_session raises for wrong version."""
        session_file = tmp_path / "old.smsa"
        with open(session_file, "w") as f:
            json.dump({"version": "0.1", "file_metadata": {}}, f)

        with pytest.raises(SessionSerializationError, match="Unsupported session version"):
            load_session(session_file)

    def test_load_missing_metadata_raises(self, tmp_path: Path) -> None:
        """load_session raises if file_metadata is missing."""
        session_file = tmp_path / "bad.smsa"
        with open(session_file, "w") as f:
            json.dump({"version": SESSION_VERSION}, f)

        with pytest.raises(SessionSerializationError, match="missing file_metadata"):
            load_session(session_file)


class TestRoundTrip:
    """Tests for full save/load round-trip."""

    @pytest.fixture
    def sample_channel(self) -> ChannelData:
        """Create a sample channel."""
        return ChannelData(
            abstimes=np.array([0, 100_000_000, 200_000_000, 300_000_000], dtype=np.uint64),
            microtimes=np.array([1.5, 2.3, 1.8, 2.0], dtype=np.float64),
        )

    @pytest.fixture
    def complex_state(self, sample_channel: ChannelData) -> SessionState:
        """Create a complex session state for round-trip testing."""
        particle1 = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )
        particle2 = ParticleData(
            id=2,
            name="Particle 2",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
            channel2=sample_channel,
        )

        state = SessionState()
        state.file_metadata = FileMetadata(
            path=Path("/data/experiment.h5"),
            filename="experiment.h5",
            num_particles=2,
            has_irf=True,
            has_spectra=True,
        )
        state.particles = [particle1, particle2]

        # Add levels for particle 1
        level1 = LevelData(
            start_index=0,
            end_index=1,
            start_time_ns=0,
            end_time_ns=100_000_000,
            num_photons=2,
            intensity_cps=20.0,
            group_id=0,
        )
        level2 = LevelData(
            start_index=2,
            end_index=3,
            start_time_ns=200_000_000,
            end_time_ns=300_000_000,
            num_photons=2,
            intensity_cps=20.0,
            group_id=1,
        )
        state.set_levels(1, 1, [level1, level2])

        # Add levels for particle 2 channel 2
        state.set_levels(2, 2, [level1])

        # Add clustering for particle 1
        clustering = make_clustering_result(
            num_steps=2, num_original_levels=2, optimal_step_index=1, selected_step_index=1
        )
        state.set_clustering(1, 1, clustering)

        # Add particle fit result
        particle_fit = FitResultData(
            tau=(4.5,),
            tau_std=(0.2,),
            amplitude=(1.0,),
            amplitude_std=(0.05,),
            shift=1.0,
            shift_std=0.1,
            chi_squared=1.1,
            durbin_watson=1.9,
            dw_bounds=(1.5, 2.5),
            fit_start_index=5,
            fit_end_index=200,
            background=3.0,
            num_exponentials=1,
            average_lifetime=4.5,
            level_index=None,
        )
        state.set_particle_fit(1, 1, particle_fit)

        # Add level fit result
        level_fit = FitResultData(
            tau=(3.5,),
            tau_std=(0.15,),
            amplitude=(1.0,),
            amplitude_std=(0.03,),
            shift=0.8,
            shift_std=0.08,
            chi_squared=1.05,
            durbin_watson=2.0,
            dw_bounds=(1.5, 2.5),
            fit_start_index=5,
            fit_end_index=150,
            background=2.5,
            num_exponentials=1,
            average_lifetime=3.5,
            level_index=0,
        )
        state.set_level_fit(1, 1, 0, level_fit)

        # Set selections
        state.select(1)
        state.select(2, channel=2)

        # Modify UI state
        state.ui_state.bin_size_ms = 2.5
        state.ui_state.confidence = ConfidenceLevel.CONF_69
        state.ui_state.active_tab = ActiveTab.GROUPING
        state.ui_state.show_levels = False

        return state

    def test_roundtrip_preserves_metadata(self, complex_state: SessionState, tmp_path: Path) -> None:
        """Round-trip preserves file metadata."""
        session_path = tmp_path / "session.smsa"

        save_session(complex_state, session_path)
        loaded = load_session(session_path)

        assert loaded["file_metadata"]["path"] == complex_state.file_metadata.path
        assert loaded["file_metadata"]["filename"] == complex_state.file_metadata.filename
        assert loaded["file_metadata"]["has_irf"] == complex_state.file_metadata.has_irf
        assert loaded["file_metadata"]["has_spectra"] == complex_state.file_metadata.has_spectra

    def test_roundtrip_preserves_levels(self, complex_state: SessionState, tmp_path: Path) -> None:
        """Round-trip preserves levels."""
        session_path = tmp_path / "session.smsa"

        save_session(complex_state, session_path)
        loaded = load_session(session_path)

        # Check particle 1 levels
        original_levels = complex_state.levels[(1, 1)]
        loaded_levels = loaded["levels"][(1, 1)]

        assert len(loaded_levels) == len(original_levels)
        for orig, load in zip(original_levels, loaded_levels):
            assert orig.start_index == load.start_index
            assert orig.end_index == load.end_index
            assert orig.intensity_cps == load.intensity_cps
            assert orig.group_id == load.group_id

        # Check particle 2 channel 2 levels
        assert (2, 2) in loaded["levels"]
        assert len(loaded["levels"][(2, 2)]) == 1

    def test_roundtrip_preserves_clustering(self, complex_state: SessionState, tmp_path: Path) -> None:
        """Round-trip preserves clustering results."""
        session_path = tmp_path / "session.smsa"

        save_session(complex_state, session_path)
        loaded = load_session(session_path)

        original_clustering = complex_state.clustering_results[(1, 1)]
        loaded_clustering = loaded["clustering_results"][(1, 1)]

        assert loaded_clustering.num_groups == original_clustering.num_groups
        assert loaded_clustering.all_bic_values == original_clustering.all_bic_values
        assert loaded_clustering.optimal_step_index == original_clustering.optimal_step_index
        assert loaded_clustering.groups[0].level_indices == original_clustering.groups[0].level_indices

    def test_roundtrip_preserves_fit_results(self, complex_state: SessionState, tmp_path: Path) -> None:
        """Round-trip preserves fit results."""
        session_path = tmp_path / "session.smsa"

        save_session(complex_state, session_path)
        loaded = load_session(session_path)

        # Check particle fit
        original_particle_fit = complex_state.particle_fits[(1, 1)]
        loaded_particle_fit = loaded["particle_fits"][(1, 1)]

        assert loaded_particle_fit.tau == original_particle_fit.tau
        assert loaded_particle_fit.chi_squared == original_particle_fit.chi_squared
        assert loaded_particle_fit.dw_bounds == original_particle_fit.dw_bounds
        assert loaded_particle_fit.level_index is None

        # Check level fit
        original_level_fit = complex_state.level_fits[(1, 1, 0)]
        loaded_level_fit = loaded["level_fits"][(1, 1, 0)]

        assert loaded_level_fit.tau == original_level_fit.tau
        assert loaded_level_fit.chi_squared == original_level_fit.chi_squared
        assert loaded_level_fit.level_index == 0

    def test_roundtrip_preserves_selections(self, complex_state: SessionState, tmp_path: Path) -> None:
        """Round-trip preserves selections."""
        session_path = tmp_path / "session.smsa"

        save_session(complex_state, session_path)
        loaded = load_session(session_path)

        assert len(loaded["selected"]) == len(complex_state.selected)
        assert loaded["current_selection"].particle_id == complex_state.current_selection.particle_id
        assert loaded["current_selection"].channel == complex_state.current_selection.channel

    def test_roundtrip_preserves_ui_state(self, complex_state: SessionState, tmp_path: Path) -> None:
        """Round-trip preserves UI state."""
        session_path = tmp_path / "session.smsa"

        save_session(complex_state, session_path)
        loaded = load_session(session_path)

        assert loaded["ui_state"].bin_size_ms == complex_state.ui_state.bin_size_ms
        assert loaded["ui_state"].confidence == complex_state.ui_state.confidence
        assert loaded["ui_state"].active_tab == complex_state.ui_state.active_tab
        assert loaded["ui_state"].show_levels == complex_state.ui_state.show_levels


class TestApplySessionToState:
    """Tests for apply_session_to_state function."""

    @pytest.fixture
    def sample_channel(self) -> ChannelData:
        """Create a sample channel."""
        return ChannelData(
            abstimes=np.array([0, 100_000_000], dtype=np.uint64),
            microtimes=np.array([1.5, 2.3], dtype=np.float64),
        )

    def test_apply_sets_levels(self, sample_channel: ChannelData) -> None:
        """apply_session_to_state sets levels."""
        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )
        state = SessionState()
        state.particles = [particle]

        level = LevelData(
            start_index=0,
            end_index=1,
            start_time_ns=0,
            end_time_ns=100_000_000,
            num_photons=2,
            intensity_cps=20.0,
        )
        session_data = {
            "levels": {(1, 1): [level]},
            "clustering_results": {},
            "particle_fits": {},
            "level_fits": {},
            "selected": [],
            "current_selection": None,
            "ui_state": UIState(),
        }

        apply_session_to_state(session_data, state)

        assert state.get_levels(1, 1) == [level]

    def test_apply_clears_existing_analysis(self, sample_channel: ChannelData) -> None:
        """apply_session_to_state clears existing analysis results."""
        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )
        state = SessionState()
        state.particles = [particle]

        # Add existing analysis
        old_level = LevelData(
            start_index=0,
            end_index=1,
            start_time_ns=0,
            end_time_ns=100_000_000,
            num_photons=2,
            intensity_cps=20.0,
        )
        state.set_levels(1, 1, [old_level])

        # Apply empty session
        session_data = {
            "levels": {},
            "clustering_results": {},
            "particle_fits": {},
            "level_fits": {},
            "selected": [],
            "current_selection": None,
            "ui_state": UIState(),
        }

        apply_session_to_state(session_data, state)

        assert state.levels == {}

    def test_apply_sets_ui_state(self, sample_channel: ChannelData) -> None:
        """apply_session_to_state sets UI state."""
        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )
        state = SessionState()
        state.particles = [particle]

        new_ui = UIState(bin_size_ms=2.5, confidence=ConfidenceLevel.CONF_69)
        session_data = {
            "levels": {},
            "clustering_results": {},
            "particle_fits": {},
            "level_fits": {},
            "selected": [],
            "current_selection": None,
            "ui_state": new_ui,
        }

        apply_session_to_state(session_data, state)

        assert state.ui_state.bin_size_ms == 2.5
        assert state.ui_state.confidence == ConfidenceLevel.CONF_69


class TestUUIDRoundTrip:
    """Tests for analysis_uuid in session save/load."""

    @pytest.fixture
    def sample_channel(self) -> ChannelData:
        """Create a sample channel."""
        return ChannelData(
            abstimes=np.array([0, 100_000_000], dtype=np.uint64),
            microtimes=np.array([1.5, 2.3], dtype=np.float64),
        )

    def test_uuid_roundtrip(self, sample_channel: ChannelData, tmp_path: Path) -> None:
        """analysis_uuid is preserved through save/load."""
        state = SessionState()
        state.file_metadata = FileMetadata(
            path=tmp_path / "test.h5",
            filename="test.h5",
            num_particles=1,
            analysis_uuid="abc-123-def",
        )
        state.particles = [
            ParticleData(
                id=1, name="Particle 1", tcspc_card="SPC-150",
                channelwidth=0.012, channel1=sample_channel,
            )
        ]

        session_path = tmp_path / "test.smsa"
        save_session(state, session_path)
        loaded = load_session(session_path)

        assert loaded["file_metadata"]["analysis_uuid"] == "abc-123-def"

    def test_uuid_none_roundtrip(self, sample_channel: ChannelData, tmp_path: Path) -> None:
        """None analysis_uuid is preserved through save/load."""
        state = SessionState()
        state.file_metadata = FileMetadata(
            path=tmp_path / "test.h5",
            filename="test.h5",
            num_particles=1,
        )
        state.particles = [
            ParticleData(
                id=1, name="Particle 1", tcspc_card="SPC-150",
                channelwidth=0.012, channel1=sample_channel,
            )
        ]

        session_path = tmp_path / "test.smsa"
        save_session(state, session_path)
        loaded = load_session(session_path)

        assert loaded["file_metadata"]["analysis_uuid"] is None


class TestRelativePathRoundTrip:
    """Tests for relative path storage in sessions."""

    @pytest.fixture
    def sample_channel(self) -> ChannelData:
        """Create a sample channel."""
        return ChannelData(
            abstimes=np.array([0, 100_000_000], dtype=np.uint64),
            microtimes=np.array([1.5, 2.3], dtype=np.float64),
        )

    def test_relative_path_same_directory(self, sample_channel: ChannelData, tmp_path: Path) -> None:
        """HDF5 path in same dir is stored as relative and resolves correctly."""
        h5_path = tmp_path / "data.h5"
        h5_path.touch()

        state = SessionState()
        state.file_metadata = FileMetadata(
            path=h5_path, filename="data.h5", num_particles=1,
        )
        state.particles = [
            ParticleData(
                id=1, name="Particle 1", tcspc_card="SPC-150",
                channelwidth=0.012, channel1=sample_channel,
            )
        ]

        session_path = tmp_path / "data.smsa"
        save_session(state, session_path)

        # Check raw JSON - path should be relative
        with open(session_path, "r") as f:
            raw = json.load(f)
        assert raw["file_metadata"]["path"] == "data.h5"

        # Load and verify resolved path
        loaded = load_session(session_path)
        assert loaded["file_metadata"]["path"] == h5_path.resolve()

    def test_relative_path_subdirectory(self, sample_channel: ChannelData, tmp_path: Path) -> None:
        """HDF5 path in subdirectory resolves correctly."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        h5_path = data_dir / "experiment.h5"
        h5_path.touch()

        state = SessionState()
        state.file_metadata = FileMetadata(
            path=h5_path, filename="experiment.h5", num_particles=1,
        )
        state.particles = [
            ParticleData(
                id=1, name="Particle 1", tcspc_card="SPC-150",
                channelwidth=0.012, channel1=sample_channel,
            )
        ]

        session_path = tmp_path / "session.smsa"
        save_session(state, session_path)
        loaded = load_session(session_path)

        assert loaded["file_metadata"]["path"] == h5_path.resolve()


class TestBackwardCompatV10:
    """Tests for backward compatibility with v1.0 session files."""

    def test_loads_v10_absolute_path(self, tmp_path: Path) -> None:
        """v1.0 sessions with absolute paths load correctly."""
        data = {
            "version": "1.0",
            "file_metadata": {
                "path": "/absolute/path/to/test.h5",
                "filename": "test.h5",
                "num_particles": 1,
            },
            "levels": {},
            "clustering_results": {},
            "particle_fits": {},
            "level_fits": {},
            "selected": [],
            "current_selection": None,
            "ui_state": {
                "bin_size_ms": 10.0,
                "confidence": 0.95,
                "active_tab": "intensity",
                "show_levels": True,
                "show_groups": True,
                "log_scale_decay": True,
            },
        }
        session_path = tmp_path / "old.smsa"
        with open(session_path, "w") as f:
            json.dump(data, f)

        loaded = load_session(session_path)

        assert loaded["file_metadata"]["path"] == Path("/absolute/path/to/test.h5")
        assert loaded["file_metadata"]["analysis_uuid"] is None

    def test_rejects_unsupported_version(self, tmp_path: Path) -> None:
        """Unsupported versions are rejected."""
        data = {
            "version": "0.5",
            "file_metadata": {"path": "/test.h5", "filename": "test.h5", "num_particles": 1},
        }
        session_path = tmp_path / "bad.smsa"
        with open(session_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(SessionSerializationError, match="Unsupported session version"):
            load_session(session_path)


class TestScanSessionsForUUID:
    """Tests for scan_sessions_for_uuid function."""

    def test_finds_matching_session(self, tmp_path: Path) -> None:
        """Finds session files with matching UUID."""
        target_uuid = "test-uuid-123"
        data = {
            "version": "1.1",
            "file_metadata": {
                "path": "test.h5",
                "filename": "test.h5",
                "num_particles": 1,
                "analysis_uuid": target_uuid,
            },
        }
        session_path = tmp_path / "test.smsa"
        with open(session_path, "w") as f:
            json.dump(data, f)

        matches = scan_sessions_for_uuid(tmp_path, target_uuid)
        assert len(matches) == 1
        assert matches[0] == session_path

    def test_no_matches(self, tmp_path: Path) -> None:
        """Returns empty list when no sessions match."""
        data = {
            "version": "1.1",
            "file_metadata": {
                "path": "test.h5",
                "filename": "test.h5",
                "num_particles": 1,
                "analysis_uuid": "other-uuid",
            },
        }
        session_path = tmp_path / "test.smsa"
        with open(session_path, "w") as f:
            json.dump(data, f)

        matches = scan_sessions_for_uuid(tmp_path, "target-uuid")
        assert len(matches) == 0

    def test_skips_corrupt_files(self, tmp_path: Path) -> None:
        """Silently skips corrupt session files."""
        # Write a corrupt file
        corrupt_path = tmp_path / "corrupt.smsa"
        with open(corrupt_path, "w") as f:
            f.write("not valid json {")

        # Write a valid file
        target_uuid = "good-uuid"
        data = {
            "version": "1.1",
            "file_metadata": {
                "path": "test.h5",
                "filename": "test.h5",
                "num_particles": 1,
                "analysis_uuid": target_uuid,
            },
        }
        good_path = tmp_path / "good.smsa"
        with open(good_path, "w") as f:
            json.dump(data, f)

        matches = scan_sessions_for_uuid(tmp_path, target_uuid)
        assert len(matches) == 1
        assert matches[0] == good_path

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Returns empty list for directory with no smsa files."""
        matches = scan_sessions_for_uuid(tmp_path, "any-uuid")
        assert len(matches) == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Returns empty list for nonexistent directory."""
        matches = scan_sessions_for_uuid(tmp_path / "nope", "any-uuid")
        assert len(matches) == 0

    def test_multiple_matches(self, tmp_path: Path) -> None:
        """Finds multiple matching session files."""
        target_uuid = "shared-uuid"
        for name in ["a.smsa", "b.smsa"]:
            data = {
                "version": "1.1",
                "file_metadata": {
                    "path": "test.h5",
                    "filename": "test.h5",
                    "num_particles": 1,
                    "analysis_uuid": target_uuid,
                },
            }
            with open(tmp_path / name, "w") as f:
                json.dump(data, f)

        matches = scan_sessions_for_uuid(tmp_path, target_uuid)
        assert len(matches) == 2
