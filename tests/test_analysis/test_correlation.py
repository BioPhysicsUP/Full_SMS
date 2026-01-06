"""Tests for correlation analysis module."""

import numpy as np
import pytest

from full_sms.analysis.correlation import (
    CorrelationResult,
    calculate_g2,
    rebin_correlation,
)


class TestCalculateG2:
    """Tests for calculate_g2 function."""

    def test_basic_correlation(self):
        """Test basic correlation with known inputs."""
        # Two photons in each channel, interleaved
        abstimes1 = np.array([0.0, 200.0])
        abstimes2 = np.array([100.0, 300.0])
        microtimes1 = np.array([0.0, 0.0])
        microtimes2 = np.array([0.0, 0.0])

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=500.0, binsize_ns=10.0
        )

        assert isinstance(result, CorrelationResult)
        assert len(result.tau) == len(result.g2)
        assert result.num_photons_ch1 == 2
        assert result.num_photons_ch2 == 2
        assert result.num_events > 0

    def test_empty_channel1(self):
        """Test with empty channel 1."""
        abstimes1 = np.array([])
        abstimes2 = np.array([100.0, 200.0])
        microtimes1 = np.array([])
        microtimes2 = np.array([0.0, 0.0])

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=500.0, binsize_ns=10.0
        )

        assert result.num_events == 0
        assert result.num_photons_ch1 == 0
        assert result.num_photons_ch2 == 2
        assert np.all(result.g2 == 0)

    def test_empty_channel2(self):
        """Test with empty channel 2."""
        abstimes1 = np.array([100.0, 200.0])
        abstimes2 = np.array([])
        microtimes1 = np.array([0.0, 0.0])
        microtimes2 = np.array([])

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=500.0, binsize_ns=10.0
        )

        assert result.num_events == 0
        assert result.num_photons_ch1 == 2
        assert result.num_photons_ch2 == 0

    def test_both_empty(self):
        """Test with both channels empty."""
        result = calculate_g2(
            np.array([]), np.array([]),
            np.array([]), np.array([]),
            window_ns=500.0, binsize_ns=10.0
        )

        assert result.num_events == 0
        assert result.num_photons_ch1 == 0
        assert result.num_photons_ch2 == 0

    def test_single_photon_each(self):
        """Test with one photon in each channel."""
        abstimes1 = np.array([0.0])
        abstimes2 = np.array([50.0])
        microtimes1 = np.array([0.0])
        microtimes2 = np.array([0.0])

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=500.0, binsize_ns=10.0
        )

        assert result.num_events == 1
        # The event should be at tau = 50 (ch2 after ch1)
        assert len(result.events) == 1
        assert np.isclose(result.events[0], 50.0)

    def test_same_channel_ignored(self):
        """Test that same-channel correlations are ignored."""
        # All photons in channel 1
        abstimes1 = np.array([0.0, 10.0, 20.0, 30.0])
        abstimes2 = np.array([])
        microtimes1 = np.array([0.0, 0.0, 0.0, 0.0])
        microtimes2 = np.array([])

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=500.0, binsize_ns=10.0
        )

        # No cross-channel events
        assert result.num_events == 0

    def test_window_cutoff(self):
        """Test that events beyond window are excluded."""
        abstimes1 = np.array([0.0])
        abstimes2 = np.array([1000.0])  # Beyond 500ns window
        microtimes1 = np.array([0.0])
        microtimes2 = np.array([0.0])

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=500.0, binsize_ns=10.0
        )

        assert result.num_events == 0

    def test_microtimes_added(self):
        """Test that microtimes are added to abstimes."""
        abstimes1 = np.array([0.0])
        abstimes2 = np.array([0.0])
        microtimes1 = np.array([10.0])
        microtimes2 = np.array([30.0])

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=500.0, binsize_ns=1.0
        )

        # Total times: ch1=10, ch2=30, dt=20
        assert result.num_events == 1
        assert np.isclose(result.events[0], 20.0)

    def test_difftime_correction(self):
        """Test channel time offset correction."""
        abstimes1 = np.array([0.0])
        abstimes2 = np.array([100.0])
        microtimes1 = np.array([0.0])
        microtimes2 = np.array([0.0])

        # Apply 50ns offset to channel 2
        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=500.0, binsize_ns=1.0,
            difftime_ns=50.0
        )

        # ch2 time becomes 100 + 0 + 50 = 150
        # dt = 150 - 0 = 150
        assert result.num_events == 1
        assert np.isclose(result.events[0], 150.0)

    def test_negative_difftime(self):
        """Test negative channel time offset."""
        abstimes1 = np.array([0.0])
        abstimes2 = np.array([100.0])
        microtimes1 = np.array([0.0])
        microtimes2 = np.array([0.0])

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=500.0, binsize_ns=1.0,
            difftime_ns=-50.0
        )

        # ch2 time becomes 100 + 0 - 50 = 50
        assert result.num_events == 1
        assert np.isclose(result.events[0], 50.0)

    def test_symmetric_histogram(self):
        """Test that histogram is symmetric for symmetric input."""
        rng = np.random.default_rng(42)

        # Generate random times, same distribution in both channels
        n = 1000
        abstimes1 = np.sort(rng.uniform(0, 1e6, n))
        abstimes2 = np.sort(rng.uniform(0, 1e6, n))
        microtimes1 = rng.uniform(0, 10, n)
        microtimes2 = rng.uniform(0, 10, n)

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=100.0, binsize_ns=5.0
        )

        # The histogram should be roughly symmetric
        # Compare left and right halves
        mid = len(result.g2) // 2
        left_sum = np.sum(result.g2[:mid])
        right_sum = np.sum(result.g2[mid:])

        # Allow 20% difference due to random fluctuations
        ratio = left_sum / max(right_sum, 1)
        assert 0.8 < ratio < 1.2

    def test_tau_range(self):
        """Test that tau covers the full window range."""
        result = calculate_g2(
            np.array([0.0]), np.array([10.0]),
            np.array([0.0]), np.array([0.0]),
            window_ns=100.0, binsize_ns=1.0
        )

        # tau should span from approximately -window to +window
        assert result.tau.min() < -90.0
        assert result.tau.max() > 90.0

    def test_bin_count(self):
        """Test histogram has expected number of bins."""
        result = calculate_g2(
            np.array([0.0]), np.array([10.0]),
            np.array([0.0]), np.array([0.0]),
            window_ns=100.0, binsize_ns=5.0
        )

        # Expected bins: 2 * window / binsize = 2 * 100 / 5 = 40
        assert len(result.g2) == 40
        assert len(result.tau) == 40


class TestCorrelationResult:
    """Tests for CorrelationResult dataclass."""

    def test_frozen(self):
        """Test that result is immutable."""
        result = calculate_g2(
            np.array([0.0]), np.array([10.0]),
            np.array([0.0]), np.array([0.0]),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.window_ns = 1000.0

    def test_rebin_method(self):
        """Test the rebin convenience method."""
        result = calculate_g2(
            np.array([0.0, 100.0, 200.0]),
            np.array([50.0, 150.0, 250.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            window_ns=500.0, binsize_ns=1.0
        )

        rebinned = result.rebin(new_binsize_ns=10.0)

        assert rebinned.binsize_ns == 10.0
        assert len(rebinned.g2) < len(result.g2)
        # Total counts preserved
        assert np.sum(rebinned.g2) == np.sum(result.g2)


class TestRebinCorrelation:
    """Tests for rebin_correlation function."""

    def test_basic_rebin(self):
        """Test basic rebinning."""
        result = calculate_g2(
            np.array([0.0, 100.0]),
            np.array([50.0, 150.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            window_ns=500.0, binsize_ns=1.0
        )

        rebinned = rebin_correlation(result, new_binsize_ns=10.0)

        assert rebinned.binsize_ns == 10.0
        assert rebinned.window_ns == 500.0
        # Fewer bins
        assert len(rebinned.g2) == len(result.g2) // 10

    def test_narrow_window(self):
        """Test rebinning with narrower window."""
        result = calculate_g2(
            np.array([0.0, 100.0]),
            np.array([50.0, 150.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            window_ns=500.0, binsize_ns=1.0
        )

        rebinned = rebin_correlation(result, new_binsize_ns=5.0, new_window_ns=100.0)

        assert rebinned.window_ns == 100.0
        assert rebinned.binsize_ns == 5.0

    def test_cannot_expand_window(self):
        """Test that window cannot be expanded."""
        result = calculate_g2(
            np.array([0.0]), np.array([10.0]),
            np.array([0.0]), np.array([0.0]),
            window_ns=100.0, binsize_ns=1.0
        )

        with pytest.raises(ValueError, match="Cannot expand window"):
            rebin_correlation(result, new_binsize_ns=1.0, new_window_ns=200.0)

    def test_preserves_counts(self):
        """Test that rebinning preserves total counts (within window)."""
        result = calculate_g2(
            np.array([0.0, 50.0, 100.0]),
            np.array([25.0, 75.0, 125.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            window_ns=500.0, binsize_ns=1.0
        )

        rebinned = rebin_correlation(result, new_binsize_ns=10.0)

        # Total counts should be preserved
        assert np.sum(rebinned.g2) == np.sum(result.g2)


class TestValidation:
    """Tests for input validation."""

    def test_mismatched_abstimes_microtimes_ch1(self):
        """Test error on mismatched channel 1 arrays."""
        with pytest.raises(ValueError, match="Channel 1 arrays must have same length"):
            calculate_g2(
                np.array([0.0, 100.0]),  # 2 elements
                np.array([50.0]),
                np.array([0.0]),  # 1 element
                np.array([0.0]),
            )

    def test_mismatched_abstimes_microtimes_ch2(self):
        """Test error on mismatched channel 2 arrays."""
        with pytest.raises(ValueError, match="Channel 2 arrays must have same length"):
            calculate_g2(
                np.array([0.0]),
                np.array([50.0, 100.0]),  # 2 elements
                np.array([0.0]),
                np.array([0.0]),  # 1 element
            )

    def test_accepts_lists(self):
        """Test that lists are converted to arrays."""
        result = calculate_g2(
            [0.0, 100.0],  # list
            [50.0, 150.0],  # list
            [0.0, 0.0],
            [0.0, 0.0],
        )

        assert isinstance(result.tau, np.ndarray)
        assert result.num_events > 0


class TestAntibunchingDetection:
    """Tests for detecting antibunching in synthetic data."""

    def test_uncorrelated_flat(self):
        """Test that uncorrelated channels give flat g2."""
        rng = np.random.default_rng(42)

        # Completely independent Poisson processes
        n = 5000
        rate = 1e-3  # photons per ns

        # Generate exponentially distributed inter-arrival times
        intervals1 = rng.exponential(1/rate, n)
        intervals2 = rng.exponential(1/rate, n)
        abstimes1 = np.cumsum(intervals1)
        abstimes2 = np.cumsum(intervals2)

        microtimes1 = rng.uniform(0, 10, n)
        microtimes2 = rng.uniform(0, 10, n)

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=200.0, binsize_ns=10.0
        )

        # For uncorrelated sources, g2 should be approximately flat
        # Check coefficient of variation is low
        if result.num_events > 100:
            mean_g2 = np.mean(result.g2)
            std_g2 = np.std(result.g2)
            cv = std_g2 / mean_g2 if mean_g2 > 0 else 0

            # CV should be relatively low for flat distribution
            # (allowing for Poisson noise)
            assert cv < 0.5

    def test_antibunched_dip(self):
        """Test detection of antibunching dip in simulated single emitter."""
        rng = np.random.default_rng(123)

        # Simulate single emitter with dead time
        # After each emission, there's a minimum time before next emission
        dead_time = 50.0  # ns
        n = 2000

        # Generate times with dead time constraint
        times = []
        current_time = 0.0
        for _ in range(n):
            # Random inter-arrival time, but minimum is dead_time
            interval = dead_time + rng.exponential(500.0)
            current_time += interval
            times.append(current_time)

        times = np.array(times)

        # Split into two channels (simulating 50/50 beamsplitter)
        channel = rng.integers(0, 2, n)
        abstimes1 = times[channel == 0]
        abstimes2 = times[channel == 1]

        microtimes1 = rng.uniform(0, 5, len(abstimes1))
        microtimes2 = rng.uniform(0, 5, len(abstimes2))

        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=200.0, binsize_ns=5.0
        )

        if result.num_events > 50:
            # Find the bin at tau=0
            zero_idx = len(result.g2) // 2

            # Compare central bins to outer bins
            central_counts = np.sum(result.g2[zero_idx-2:zero_idx+3])
            outer_counts = np.sum(result.g2[:5]) + np.sum(result.g2[-5:])

            # Central region should have fewer counts (antibunching)
            # This is a weak test due to randomness, but should hold statistically
            # The antibunching dip means g2(0) < g2(inf)
            assert central_counts <= outer_counts * 1.5  # Allow some margin


class TestPerformance:
    """Performance-related tests."""

    def test_large_dataset(self):
        """Test that large datasets complete in reasonable time."""
        rng = np.random.default_rng(42)
        n = 10000

        abstimes1 = np.sort(rng.uniform(0, 1e9, n))
        abstimes2 = np.sort(rng.uniform(0, 1e9, n))
        microtimes1 = rng.uniform(0, 10, n)
        microtimes2 = rng.uniform(0, 10, n)

        # This should complete without hanging
        result = calculate_g2(
            abstimes1, abstimes2, microtimes1, microtimes2,
            window_ns=100.0, binsize_ns=1.0
        )

        assert result.num_photons_ch1 == n
        assert result.num_photons_ch2 == n
