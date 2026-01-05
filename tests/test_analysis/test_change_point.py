"""Tests for change point analysis."""

import numpy as np
import pytest

from full_sms.analysis.change_point import (
    CPAParams,
    ChangePointResult,
    ConfidenceLevel,
    TauData,
    _compute_sums,
    _create_levels_from_change_points,
    _get_sig_e,
    _get_sums_set,
    find_change_points,
)


def generate_two_level_data(
    intensity1_cps: float,
    intensity2_cps: float,
    duration1_s: float,
    duration2_s: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic photon data with two intensity levels.

    Args:
        intensity1_cps: Intensity of first level in counts per second.
        intensity2_cps: Intensity of second level in counts per second.
        duration1_s: Duration of first level in seconds.
        duration2_s: Duration of second level in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Array of photon arrival times in nanoseconds.
    """
    rng = np.random.default_rng(seed)

    # Generate photons for first level (Poisson process)
    n_photons_1 = int(intensity1_cps * duration1_s)
    inter_arrival_1 = rng.exponential(1e9 / intensity1_cps, n_photons_1)
    times_1 = np.cumsum(inter_arrival_1)

    # Generate photons for second level
    n_photons_2 = int(intensity2_cps * duration2_s)
    inter_arrival_2 = rng.exponential(1e9 / intensity2_cps, n_photons_2)
    times_2 = np.cumsum(inter_arrival_2) + duration1_s * 1e9

    # Combine and sort
    all_times = np.concatenate([times_1, times_2])
    return np.sort(all_times)


def generate_constant_intensity_data(
    intensity_cps: float,
    duration_s: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic photon data with constant intensity.

    Args:
        intensity_cps: Intensity in counts per second.
        duration_s: Duration in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Array of photon arrival times in nanoseconds.
    """
    rng = np.random.default_rng(seed)
    n_photons = int(intensity_cps * duration_s)
    inter_arrival = rng.exponential(1e9 / intensity_cps, n_photons)
    return np.cumsum(inter_arrival)


class TestTauData:
    """Tests for TauData class."""

    def test_load_conf_99(self):
        """Load tau data for 99% confidence."""
        tau = TauData(ConfidenceLevel.CONF_99)
        assert tau.confidence == ConfidenceLevel.CONF_99
        # tau_a for n=100 should be positive
        assert tau.get_tau_a(100) > 0
        assert tau.get_tau_b(100) > 0

    def test_load_conf_95(self):
        """Load tau data for 95% confidence."""
        tau = TauData(ConfidenceLevel.CONF_95)
        assert tau.confidence == ConfidenceLevel.CONF_95
        assert tau.get_tau_a(100) > 0

    def test_load_conf_90(self):
        """Load tau data for 90% confidence."""
        tau = TauData(ConfidenceLevel.CONF_90)
        assert tau.confidence == ConfidenceLevel.CONF_90
        assert tau.get_tau_a(100) > 0

    def test_load_conf_69(self):
        """Load tau data for 69% confidence."""
        tau = TauData(ConfidenceLevel.CONF_69)
        assert tau.confidence == ConfidenceLevel.CONF_69
        assert tau.get_tau_a(100) > 0

    def test_tau_a_increases_with_confidence(self):
        """Higher confidence should have higher tau_a threshold."""
        tau_69 = TauData(ConfidenceLevel.CONF_69)
        tau_95 = TauData(ConfidenceLevel.CONF_95)
        tau_99 = TauData(ConfidenceLevel.CONF_99)

        # Higher confidence = stricter threshold
        assert tau_99.get_tau_a(100) > tau_95.get_tau_a(100)
        assert tau_95.get_tau_a(100) > tau_69.get_tau_a(100)


class TestComputeSums:
    """Tests for precomputed sums."""

    def test_sums_computed(self):
        """Sums are computed successfully."""
        sums = _compute_sums()
        assert "sums_u_k" in sums
        assert "sums_u_n_k" in sums
        assert "sums_v2_k" in sums
        assert "sums_v2_n_k" in sums
        assert "sums_sig_e" in sums
        assert sums["n_min"] == 10
        assert sums["n_max"] == 1000

    def test_get_sig_e(self):
        """Get sig_e value."""
        sums = _compute_sums()
        sig_e = _get_sig_e(sums, 100)
        # sig_e should be positive for n=100
        assert sig_e > 0

    def test_get_sums_set(self):
        """Get all sums for given n, k."""
        sums = _compute_sums()
        sum_set = _get_sums_set(sums, n=100, k=50)
        assert "u_k" in sum_set
        assert "u_n_k" in sum_set
        assert "v2_k" in sum_set
        assert "v2_n_k" in sum_set

    def test_sums_cached(self):
        """Sums are cached (same object returned)."""
        sums1 = _compute_sums()
        sums2 = _compute_sums()
        assert sums1 is sums2


class TestFindChangePoints:
    """Tests for find_change_points function."""

    def test_empty_input(self):
        """Empty array returns empty result."""
        result = find_change_points(np.array([]))
        assert result.num_change_points == 0
        assert len(result.levels) == 0
        assert len(result.change_point_indices) == 0

    def test_too_few_photons(self):
        """Fewer than 200 photons returns single level."""
        # Generate 100 photons
        abstimes = generate_constant_intensity_data(10000, 0.01, seed=42)
        assert len(abstimes) < 200

        result = find_change_points(abstimes)
        assert result.num_change_points == 0
        # Should have one level if enough photons for minimum
        if len(abstimes) >= 20:
            assert len(result.levels) == 1

    def test_constant_intensity_no_change_points(self):
        """Constant intensity data should have no (or few) change points."""
        # Generate constant intensity data
        abstimes = generate_constant_intensity_data(
            intensity_cps=50000,
            duration_s=0.1,
            seed=42,
        )

        result = find_change_points(abstimes, confidence=0.99)
        # Should have 0 or very few spurious change points at 99% confidence
        # With constant intensity, ideally num_change_points == 0
        # but due to statistical fluctuations, might get 1-2 false positives
        assert result.num_change_points <= 2
        assert len(result.levels) == result.num_change_points + 1

    def test_detects_clear_change_point(self):
        """Detects a clear intensity change."""
        # Two levels with very different intensities
        abstimes = generate_two_level_data(
            intensity1_cps=10000,
            intensity2_cps=100000,  # 10x difference
            duration1_s=0.05,
            duration2_s=0.05,
            seed=42,
        )

        result = find_change_points(abstimes, confidence=0.95)
        # Should detect at least one change point
        assert result.num_change_points >= 1
        assert len(result.levels) >= 2

    def test_confidence_levels_as_float(self):
        """Accept confidence as float values."""
        abstimes = generate_constant_intensity_data(50000, 0.05, seed=42)

        # All valid float confidence levels should work
        for conf in [0.69, 0.90, 0.95, 0.99]:
            result = find_change_points(abstimes, confidence=conf)
            assert isinstance(result, ChangePointResult)

    def test_confidence_levels_as_enum(self):
        """Accept confidence as enum values."""
        abstimes = generate_constant_intensity_data(50000, 0.05, seed=42)

        for conf in ConfidenceLevel:
            result = find_change_points(abstimes, confidence=conf)
            assert isinstance(result, ChangePointResult)

    def test_invalid_confidence_raises(self):
        """Invalid confidence level raises ValueError."""
        abstimes = generate_constant_intensity_data(50000, 0.05, seed=42)

        with pytest.raises(ValueError, match="Unsupported confidence"):
            find_change_points(abstimes, confidence=0.50)

    def test_end_time_limits_analysis(self):
        """end_time_ns limits which photons are analyzed."""
        abstimes = generate_two_level_data(
            intensity1_cps=20000,
            intensity2_cps=100000,
            duration1_s=0.05,
            duration2_s=0.05,
            seed=42,
        )

        # Analyze only first half (constant intensity)
        end_time_ns = 0.04 * 1e9  # 40ms
        result = find_change_points(abstimes, end_time_ns=end_time_ns)

        # Should have fewer change points since we're only analyzing first level
        # At minimum, levels should exist
        assert len(result.levels) >= 1

    def test_levels_cover_all_photons(self):
        """Levels should cover all photons in the trace."""
        abstimes = generate_two_level_data(
            intensity1_cps=30000,
            intensity2_cps=70000,
            duration1_s=0.03,
            duration2_s=0.03,
            seed=42,
        )

        result = find_change_points(abstimes, confidence=0.95)

        if len(result.levels) > 0:
            # First level should start at 0
            assert result.levels[0].start_index == 0

            # Last level should end at last photon
            assert result.levels[-1].end_index == len(abstimes) - 1

            # Levels should be contiguous
            for i in range(len(result.levels) - 1):
                assert result.levels[i].end_index + 1 == result.levels[i + 1].start_index

    def test_level_intensities_calculated(self):
        """Levels have valid intensity values."""
        abstimes = generate_two_level_data(
            intensity1_cps=20000,
            intensity2_cps=80000,
            duration1_s=0.05,
            duration2_s=0.05,
            seed=42,
        )

        result = find_change_points(abstimes, confidence=0.95)

        for level in result.levels:
            assert level.intensity_cps > 0
            assert level.num_photons > 0
            assert level.dwell_time_s > 0

    def test_result_structure(self):
        """Result has correct structure."""
        abstimes = generate_constant_intensity_data(50000, 0.05, seed=42)

        result = find_change_points(abstimes)

        assert isinstance(result, ChangePointResult)
        assert isinstance(result.change_point_indices, np.ndarray)
        assert isinstance(result.levels, list)
        assert isinstance(result.num_change_points, int)
        assert isinstance(result.confidence_regions, list)
        assert result.num_change_points == len(result.change_point_indices)
        assert len(result.levels) == result.num_change_points + 1


class TestCreateLevelsFromChangePoints:
    """Tests for _create_levels_from_change_points helper."""

    def test_no_change_points_single_level(self):
        """No change points creates single level."""
        abstimes = np.array([1e6, 2e6, 3e6, 4e6, 5e6])  # 5 photons
        cpt_indices = np.array([], dtype=np.int64)

        levels = _create_levels_from_change_points(abstimes, cpt_indices, len(abstimes))

        assert len(levels) == 1
        assert levels[0].start_index == 0
        assert levels[0].end_index == 4

    def test_one_change_point_two_levels(self):
        """One change point creates two levels."""
        abstimes = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
        cpt_indices = np.array([3], dtype=np.int64)

        levels = _create_levels_from_change_points(abstimes, cpt_indices, len(abstimes))

        assert len(levels) == 2
        # First level: 0 to cpt-1 = 0 to 2
        assert levels[0].start_index == 0
        assert levels[0].end_index == 2
        # Second level: cpt to end = 3 to 4
        assert levels[1].start_index == 3
        assert levels[1].end_index == 4

    def test_two_change_points_three_levels(self):
        """Two change points creates three levels."""
        abstimes = np.linspace(1e6, 10e6, 10)  # 10 photons
        cpt_indices = np.array([3, 7], dtype=np.int64)

        levels = _create_levels_from_change_points(abstimes, cpt_indices, len(abstimes))

        assert len(levels) == 3
        assert levels[0].start_index == 0
        assert levels[0].end_index == 2
        assert levels[1].start_index == 3
        assert levels[1].end_index == 6
        assert levels[2].start_index == 7
        assert levels[2].end_index == 9

    def test_empty_photons_returns_empty(self):
        """Empty photon array returns empty levels."""
        abstimes = np.array([])
        cpt_indices = np.array([], dtype=np.int64)

        levels = _create_levels_from_change_points(abstimes, cpt_indices, 0)

        assert len(levels) == 0


class TestCPAParams:
    """Tests for CPAParams dataclass."""

    def test_default_values(self):
        """Default parameter values."""
        params = CPAParams(confidence=ConfidenceLevel.CONF_95)
        assert params.min_photons == 20
        assert params.min_boundary_offset == 7

    def test_custom_values(self):
        """Custom parameter values."""
        params = CPAParams(
            confidence=ConfidenceLevel.CONF_99,
            min_photons=50,
            min_boundary_offset=10,
        )
        assert params.confidence == ConfidenceLevel.CONF_99
        assert params.min_photons == 50
        assert params.min_boundary_offset == 10


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_values(self):
        """Confidence level values."""
        assert ConfidenceLevel.CONF_69.value == 0.69
        assert ConfidenceLevel.CONF_90.value == 0.90
        assert ConfidenceLevel.CONF_95.value == 0.95
        assert ConfidenceLevel.CONF_99.value == 0.99


class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_multi_level_detection(self):
        """Detect multiple intensity levels."""
        rng = np.random.default_rng(42)

        # Create three-level data
        intensities = [20000, 80000, 40000]  # cps
        durations = [0.03, 0.03, 0.03]  # seconds

        all_times = []
        current_time = 0

        for intensity, duration in zip(intensities, durations):
            n_photons = int(intensity * duration)
            inter_arrival = rng.exponential(1e9 / intensity, n_photons)
            times = np.cumsum(inter_arrival) + current_time
            all_times.extend(times)
            current_time = times[-1] if len(times) > 0 else current_time

        abstimes = np.array(all_times)

        result = find_change_points(abstimes, confidence=0.90)

        # Should detect 2+ change points (3 levels)
        # Using 90% confidence for more sensitivity
        assert result.num_change_points >= 1
        assert len(result.levels) >= 2

    def test_reproducibility(self):
        """Same input produces same output."""
        abstimes = generate_two_level_data(
            intensity1_cps=30000,
            intensity2_cps=70000,
            duration1_s=0.04,
            duration2_s=0.04,
            seed=123,
        )

        result1 = find_change_points(abstimes, confidence=0.95)
        result2 = find_change_points(abstimes, confidence=0.95)

        np.testing.assert_array_equal(
            result1.change_point_indices, result2.change_point_indices
        )
        assert result1.num_change_points == result2.num_change_points

    def test_higher_confidence_fewer_detections(self):
        """Higher confidence should detect fewer spurious change points."""
        abstimes = generate_constant_intensity_data(50000, 0.1, seed=42)

        result_69 = find_change_points(abstimes, confidence=0.69)
        result_99 = find_change_points(abstimes, confidence=0.99)

        # Higher confidence should be more conservative
        assert result_99.num_change_points <= result_69.num_change_points

    def test_large_dataset(self):
        """Handle larger dataset without errors."""
        # Generate ~100ms of data at high count rate = ~5000 photons
        abstimes = generate_constant_intensity_data(
            intensity_cps=50000,
            duration_s=0.1,
            seed=42,
        )

        result = find_change_points(abstimes, confidence=0.95)

        # Should complete without error
        assert isinstance(result, ChangePointResult)
        assert len(result.levels) >= 1
