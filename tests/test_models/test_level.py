"""Tests for level data models."""

import numpy as np
import pytest

from full_sms.models.level import LevelData


class TestLevelData:
    """Tests for LevelData."""

    def test_create_level(self) -> None:
        """Can create a LevelData with valid parameters."""
        level = LevelData(
            start_index=0,
            end_index=99,
            start_time_ns=0,
            end_time_ns=1_000_000_000,
            num_photons=100,
            intensity_cps=100.0,
        )

        assert level.start_index == 0
        assert level.end_index == 99
        assert level.start_time_ns == 0
        assert level.end_time_ns == 1_000_000_000
        assert level.num_photons == 100
        assert level.intensity_cps == 100.0
        assert level.group_id is None

    def test_create_level_with_group(self) -> None:
        """Can create a LevelData with group assignment."""
        level = LevelData(
            start_index=0,
            end_index=99,
            start_time_ns=0,
            end_time_ns=1_000_000_000,
            num_photons=100,
            intensity_cps=100.0,
            group_id=2,
        )

        assert level.group_id == 2

    def test_indices_property(self) -> None:
        """indices returns (start, end) tuple."""
        level = LevelData(
            start_index=10,
            end_index=50,
            start_time_ns=100_000_000,
            end_time_ns=500_000_000,
            num_photons=40,
            intensity_cps=100.0,
        )

        assert level.indices == (10, 50)

    def test_times_ns_property(self) -> None:
        """times_ns returns (start, end) tuple in nanoseconds."""
        level = LevelData(
            start_index=10,
            end_index=50,
            start_time_ns=100_000_000,
            end_time_ns=500_000_000,
            num_photons=40,
            intensity_cps=100.0,
        )

        assert level.times_ns == (100_000_000, 500_000_000)

    def test_dwell_time_ns(self) -> None:
        """dwell_time_ns calculates duration in nanoseconds."""
        level = LevelData(
            start_index=0,
            end_index=99,
            start_time_ns=100_000_000,
            end_time_ns=600_000_000,
            num_photons=100,
            intensity_cps=200.0,
        )

        assert level.dwell_time_ns == 500_000_000

    def test_dwell_time_s(self) -> None:
        """dwell_time_s calculates duration in seconds."""
        level = LevelData(
            start_index=0,
            end_index=99,
            start_time_ns=0,
            end_time_ns=1_000_000_000,
            num_photons=100,
            intensity_cps=100.0,
        )

        assert level.dwell_time_s == pytest.approx(1.0)

    def test_times_s_property(self) -> None:
        """times_s returns (start, end) tuple in seconds."""
        level = LevelData(
            start_index=0,
            end_index=99,
            start_time_ns=500_000_000,
            end_time_ns=1_500_000_000,
            num_photons=100,
            intensity_cps=100.0,
        )

        assert level.times_s == pytest.approx((0.5, 1.5))

    def test_negative_start_index_raises(self) -> None:
        """Raises ValueError for negative start_index."""
        with pytest.raises(ValueError, match="start_index must be non-negative"):
            LevelData(
                start_index=-1,
                end_index=99,
                start_time_ns=0,
                end_time_ns=1_000_000_000,
                num_photons=100,
                intensity_cps=100.0,
            )

    def test_end_before_start_raises(self) -> None:
        """Raises ValueError when end_index < start_index."""
        with pytest.raises(ValueError, match="end_index.*must be >= start_index"):
            LevelData(
                start_index=100,
                end_index=50,
                start_time_ns=0,
                end_time_ns=1_000_000_000,
                num_photons=100,
                intensity_cps=100.0,
            )

    def test_negative_start_time_raises(self) -> None:
        """Raises ValueError for negative start_time_ns."""
        with pytest.raises(ValueError, match="start_time_ns must be non-negative"):
            LevelData(
                start_index=0,
                end_index=99,
                start_time_ns=-1,
                end_time_ns=1_000_000_000,
                num_photons=100,
                intensity_cps=100.0,
            )

    def test_end_time_before_start_raises(self) -> None:
        """Raises ValueError when end_time_ns < start_time_ns."""
        with pytest.raises(ValueError, match="end_time_ns.*must be >= start_time_ns"):
            LevelData(
                start_index=0,
                end_index=99,
                start_time_ns=1_000_000_000,
                end_time_ns=500_000_000,
                num_photons=100,
                intensity_cps=100.0,
            )

    def test_negative_num_photons_raises(self) -> None:
        """Raises ValueError for negative num_photons."""
        with pytest.raises(ValueError, match="num_photons must be non-negative"):
            LevelData(
                start_index=0,
                end_index=99,
                start_time_ns=0,
                end_time_ns=1_000_000_000,
                num_photons=-1,
                intensity_cps=100.0,
            )

    def test_negative_intensity_raises(self) -> None:
        """Raises ValueError for negative intensity_cps."""
        with pytest.raises(ValueError, match="intensity_cps must be non-negative"):
            LevelData(
                start_index=0,
                end_index=99,
                start_time_ns=0,
                end_time_ns=1_000_000_000,
                num_photons=100,
                intensity_cps=-10.0,
            )

    def test_immutable(self) -> None:
        """LevelData is immutable (frozen dataclass)."""
        level = LevelData(
            start_index=0,
            end_index=99,
            start_time_ns=0,
            end_time_ns=1_000_000_000,
            num_photons=100,
            intensity_cps=100.0,
        )

        with pytest.raises(AttributeError):
            level.num_photons = 200


class TestLevelDataFromPhotonIndices:
    """Tests for LevelData.from_photon_indices factory method."""

    def test_from_photon_indices(self) -> None:
        """Can create LevelData from abstimes and indices."""
        # 1000 photons over 1 second = 1000 cps
        abstimes = np.arange(0, 1_000_000_000, 1_000_000, dtype=np.uint64)

        level = LevelData.from_photon_indices(
            abstimes=abstimes,
            start_index=0,
            end_index=99,
        )

        assert level.start_index == 0
        assert level.end_index == 99
        assert level.start_time_ns == 0
        assert level.end_time_ns == 99_000_000  # abstimes[99]
        assert level.num_photons == 100
        assert level.intensity_cps == pytest.approx(100 / 0.099, rel=1e-3)
        assert level.group_id is None

    def test_from_photon_indices_with_group(self) -> None:
        """Can specify group_id when creating from indices."""
        abstimes = np.arange(0, 1_000_000_000, 1_000_000, dtype=np.uint64)

        level = LevelData.from_photon_indices(
            abstimes=abstimes,
            start_index=0,
            end_index=99,
            group_id=5,
        )

        assert level.group_id == 5

    def test_from_photon_indices_mid_trace(self) -> None:
        """Can create level from middle of photon trace."""
        abstimes = np.arange(0, 1_000_000_000, 1_000_000, dtype=np.uint64)

        level = LevelData.from_photon_indices(
            abstimes=abstimes,
            start_index=200,
            end_index=299,
        )

        assert level.start_index == 200
        assert level.end_index == 299
        assert level.start_time_ns == 200_000_000
        assert level.end_time_ns == 299_000_000
        assert level.num_photons == 100

    def test_from_photon_indices_zero_dwell(self) -> None:
        """Handles zero dwell time (same start and end time)."""
        abstimes = np.array([0, 0, 0, 100_000_000], dtype=np.uint64)

        level = LevelData.from_photon_indices(
            abstimes=abstimes,
            start_index=0,
            end_index=2,
        )

        assert level.num_photons == 3
        assert level.intensity_cps == 0.0  # Can't calculate intensity with zero time
