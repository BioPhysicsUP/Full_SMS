"""Tests for ROI utility functions."""

import numpy as np
import pytest

from full_sms.analysis.roi import (
    auto_trim_roi,
    filter_levels_by_roi,
    get_default_roi,
    get_roi_photon_indices,
    slice_by_roi,
)
from full_sms.models.level import LevelData


class TestGetROIPhotonIndices:
    """Tests for get_roi_photon_indices."""

    def test_known_time_window(self) -> None:
        """Returns correct indices for a known time window."""
        # Photons at 0, 1, 2, 3, 4 seconds (in nanoseconds)
        abstimes = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000], dtype=np.uint64)

        start_idx, end_idx = get_roi_photon_indices(abstimes, 1.0, 3.0)

        assert start_idx == 1
        assert end_idx == 4  # side='right' includes photon at exactly 3.0s

    def test_full_range(self) -> None:
        """Full range returns all indices."""
        abstimes = np.array([100_000_000, 200_000_000, 300_000_000], dtype=np.uint64)

        start_idx, end_idx = get_roi_photon_indices(abstimes, 0.0, 1.0)

        assert start_idx == 0
        assert end_idx == 3

    def test_empty_range(self) -> None:
        """ROI outside data returns empty range."""
        abstimes = np.array([1_000_000_000, 2_000_000_000], dtype=np.uint64)

        start_idx, end_idx = get_roi_photon_indices(abstimes, 5.0, 10.0)

        assert start_idx == end_idx  # Empty range

    def test_partial_overlap(self) -> None:
        """ROI partially overlapping data returns correct subset."""
        abstimes = np.array([
            500_000_000, 1_000_000_000, 1_500_000_000,
            2_000_000_000, 2_500_000_000
        ], dtype=np.uint64)

        start_idx, end_idx = get_roi_photon_indices(abstimes, 1.0, 2.0)

        # Photons at 1.0s and 1.5s should be included, 2.0s via side='right'
        assert start_idx == 1
        assert end_idx == 4  # includes photon at 2.0s


class TestSliceByROI:
    """Tests for slice_by_roi."""

    def test_none_roi_returns_unmodified(self) -> None:
        """None ROI returns original arrays."""
        abstimes = np.array([100, 200, 300], dtype=np.uint64)
        microtimes = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        result_abs, result_micro = slice_by_roi(abstimes, microtimes, None)

        np.testing.assert_array_equal(result_abs, abstimes)
        np.testing.assert_array_equal(result_micro, microtimes)

    def test_subset_slicing(self) -> None:
        """ROI returns correct subset."""
        abstimes = np.array([
            0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000
        ], dtype=np.uint64)
        microtimes = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

        result_abs, result_micro = slice_by_roi(abstimes, microtimes, (1.0, 3.0))

        # Photons at 1.0s, 2.0s; 3.0s included via searchsorted side='right'
        assert len(result_abs) == 3
        assert len(result_micro) == 3
        np.testing.assert_array_equal(result_abs, [1_000_000_000, 2_000_000_000, 3_000_000_000])
        np.testing.assert_array_equal(result_micro, [0.2, 0.3, 0.4])


class TestGetDefaultROI:
    """Tests for get_default_roi."""

    def test_normal_trace(self) -> None:
        """Returns full trace range in seconds."""
        abstimes = np.array([0, 5_000_000_000], dtype=np.uint64)

        start, end = get_default_roi(abstimes)

        assert start == 0.0
        assert end == 5.0

    def test_empty_array(self) -> None:
        """Returns (0, 0) for empty array."""
        abstimes = np.array([], dtype=np.uint64)

        start, end = get_default_roi(abstimes)

        assert start == 0.0
        assert end == 0.0


class TestAutoTrimROI:
    """Tests for auto_trim_roi."""

    def _make_level(
        self,
        start_ns: int,
        end_ns: int,
        intensity_cps: float,
        num_photons: int = 100,
    ) -> LevelData:
        """Helper to create a LevelData."""
        return LevelData(
            start_index=0,
            end_index=num_photons - 1,
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            num_photons=num_photons,
            intensity_cps=intensity_cps,
        )

    def test_bleaching_detected(self) -> None:
        """Trailing low-intensity levels trigger trim."""
        levels = [
            self._make_level(0, 2_000_000_000, 5000.0),  # 0-2s, bright
            self._make_level(2_000_000_000, 4_000_000_000, 5000.0),  # 2-4s, bright
            self._make_level(4_000_000_000, 6_000_000_000, 100.0),  # 4-6s, dim
            self._make_level(6_000_000_000, 8_000_000_000, 50.0),  # 6-8s, dim
        ]

        result = auto_trim_roi(levels, threshold_cps=1000.0, min_duration_s=3.0)

        # 4s of trailing dim levels (>= 3s threshold)
        # Start of trailing run is at 4s
        assert result is not None
        assert result == 4.0

    def test_no_bleaching(self) -> None:
        """All levels above threshold returns None."""
        levels = [
            self._make_level(0, 2_000_000_000, 5000.0),
            self._make_level(2_000_000_000, 4_000_000_000, 3000.0),
        ]

        result = auto_trim_roi(levels, threshold_cps=1000.0, min_duration_s=1.0)

        assert result is None

    def test_insufficient_duration(self) -> None:
        """Short trailing bleach below min_duration returns None."""
        levels = [
            self._make_level(0, 5_000_000_000, 5000.0),  # 0-5s, bright
            self._make_level(5_000_000_000, 5_500_000_000, 100.0),  # 5-5.5s, dim
        ]

        result = auto_trim_roi(levels, threshold_cps=1000.0, min_duration_s=1.0)

        assert result is None

    def test_empty_levels(self) -> None:
        """Empty levels list returns None."""
        result = auto_trim_roi([], threshold_cps=1000.0, min_duration_s=1.0)

        assert result is None

    def test_interrupted_trailing_run(self) -> None:
        """Bright level in the middle breaks the trailing run."""
        levels = [
            self._make_level(0, 2_000_000_000, 5000.0),  # bright
            self._make_level(2_000_000_000, 4_000_000_000, 100.0),  # dim
            self._make_level(4_000_000_000, 6_000_000_000, 5000.0),  # bright (interrupts)
            self._make_level(6_000_000_000, 7_000_000_000, 100.0),  # dim
        ]

        result = auto_trim_roi(levels, threshold_cps=1000.0, min_duration_s=1.5)

        # Only 1s of trailing dim (below 1.5s threshold)
        assert result is None


class TestFilterLevelsByROI:
    """Tests for filter_levels_by_roi."""

    def _make_level(
        self,
        start_ns: int,
        end_ns: int,
        intensity_cps: float = 1000.0,
        num_photons: int = 100,
    ) -> LevelData:
        """Helper to create a LevelData."""
        return LevelData(
            start_index=0,
            end_index=num_photons - 1,
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            num_photons=num_photons,
            intensity_cps=intensity_cps,
        )

    def test_none_roi_returns_all(self) -> None:
        """None ROI returns all levels."""
        levels = [
            self._make_level(0, 1_000_000_000),
            self._make_level(1_000_000_000, 2_000_000_000),
        ]
        result = filter_levels_by_roi(levels, None)
        assert len(result) == 2

    def test_empty_levels_returns_empty(self) -> None:
        """Empty levels list returns empty."""
        result = filter_levels_by_roi([], (1.0, 5.0))
        assert result == []

    def test_levels_fully_inside_kept(self) -> None:
        """Levels fully inside ROI are kept."""
        levels = [
            self._make_level(2_000_000_000, 3_000_000_000),  # 2-3s
            self._make_level(3_000_000_000, 4_000_000_000),  # 3-4s
        ]
        result = filter_levels_by_roi(levels, (1.0, 5.0))
        assert len(result) == 2

    def test_levels_fully_outside_excluded(self) -> None:
        """Levels fully outside ROI are excluded."""
        levels = [
            self._make_level(0, 500_000_000),           # 0-0.5s
            self._make_level(6_000_000_000, 7_000_000_000),  # 6-7s
        ]
        result = filter_levels_by_roi(levels, (1.0, 5.0))
        assert len(result) == 0

    def test_levels_straddling_start_excluded(self) -> None:
        """Level straddling ROI start boundary is excluded."""
        levels = [
            self._make_level(500_000_000, 1_500_000_000),  # 0.5-1.5s, straddles start
            self._make_level(2_000_000_000, 3_000_000_000),  # 2-3s, inside
        ]
        result = filter_levels_by_roi(levels, (1.0, 5.0))
        assert len(result) == 1
        assert result[0].start_time_ns == 2_000_000_000

    def test_levels_straddling_end_excluded(self) -> None:
        """Level straddling ROI end boundary is excluded."""
        levels = [
            self._make_level(2_000_000_000, 3_000_000_000),  # 2-3s, inside
            self._make_level(4_500_000_000, 5_500_000_000),  # 4.5-5.5s, straddles end
        ]
        result = filter_levels_by_roi(levels, (1.0, 5.0))
        assert len(result) == 1
        assert result[0].start_time_ns == 2_000_000_000

    def test_level_exactly_at_roi_boundary_kept(self) -> None:
        """Level exactly matching ROI boundaries is kept."""
        levels = [
            self._make_level(1_000_000_000, 5_000_000_000),  # exactly 1-5s
        ]
        result = filter_levels_by_roi(levels, (1.0, 5.0))
        assert len(result) == 1

    def test_mixed_inside_outside_straddling(self) -> None:
        """Mix of inside, outside, and straddling levels."""
        levels = [
            self._make_level(0, 1_500_000_000),           # straddles start
            self._make_level(2_000_000_000, 3_000_000_000),  # inside
            self._make_level(3_000_000_000, 4_000_000_000),  # inside
            self._make_level(4_500_000_000, 5_500_000_000),  # straddles end
            self._make_level(6_000_000_000, 7_000_000_000),  # outside
        ]
        result = filter_levels_by_roi(levels, (1.0, 5.0))
        assert len(result) == 2
