"""Tests for histogram utilities."""

import numpy as np
import pytest

from full_sms.analysis.histograms import (
    bin_photons,
    build_decay_histogram,
    compute_intensity_cps,
    rebin_histogram,
)


class TestBinPhotons:
    """Tests for bin_photons function."""

    def test_empty_input(self):
        """Empty abstimes returns empty arrays."""
        times, counts = bin_photons(np.array([]), bin_size_ms=1.0)
        assert len(times) == 0
        assert len(counts) == 0

    def test_single_photon(self):
        """Single photon falls in correct bin."""
        # Photon at 1.5 ms = 1.5e6 ns
        abstimes = np.array([1.5e6])
        times, counts = bin_photons(abstimes, bin_size_ms=1.0)

        # Should have bins at 0, 1 ms (photon in bin starting at 1ms)
        assert len(times) == 2
        assert times[0] == 0.0
        assert times[1] == 1.0
        assert counts[0] == 0
        assert counts[1] == 1

    def test_multiple_photons_same_bin(self):
        """Multiple photons in same bin are summed."""
        # Three photons all between 0-1 ms
        abstimes = np.array([0.1e6, 0.5e6, 0.9e6])
        times, counts = bin_photons(abstimes, bin_size_ms=1.0)

        assert counts[0] == 3

    def test_photons_across_bins(self):
        """Photons distributed across multiple bins."""
        # Photons at 0.5, 1.5, 1.8, 3.2 ms
        abstimes = np.array([0.5e6, 1.5e6, 1.8e6, 3.2e6])
        times, counts = bin_photons(abstimes, bin_size_ms=1.0)

        # Bins: [0-1): 1, [1-2): 2, [2-3): 0, [3-4): 1
        assert len(times) == 4
        assert list(counts) == [1, 2, 0, 1]

    def test_bin_size_conversion(self):
        """Bin size correctly converted from ms to ns."""
        # Photon at 0.7 ms = 700,000 ns
        abstimes = np.array([700_000])
        times, counts = bin_photons(abstimes, bin_size_ms=0.5)

        # With 0.5 ms bins, photon at 0.7 ms is in bin [0.5-1.0)
        assert times[0] == 0.0
        assert times[1] == 0.5
        assert counts[0] == 0
        assert counts[1] == 1

    def test_large_bin_size(self):
        """Large bin size combines many photons."""
        # 100 photons spread over 10 ms (not including boundary)
        abstimes = np.linspace(0, 9.9e6, 100)  # 0 to 9.9 ms
        times, counts = bin_photons(abstimes, bin_size_ms=10.0)

        # All in one bin [0-10)
        assert len(times) == 1
        assert counts[0] == 100

    def test_small_bin_size(self):
        """Small bin size creates more bins."""
        # Photon at 0.95 ms = 950,000 ns
        abstimes = np.array([950_000])
        times, counts = bin_photons(abstimes, bin_size_ms=0.1)

        # Should have bins at 0, 0.1, 0.2, ..., 0.9 (10 bins)
        # Photon at 0.95 ms is in bin [0.9-1.0)
        assert len(times) == 10
        assert counts[9] == 1  # Last bin has the photon

    def test_output_dtypes(self):
        """Output arrays have correct dtypes."""
        abstimes = np.array([1e6, 2e6])
        times, counts = bin_photons(abstimes, bin_size_ms=1.0)

        assert times.dtype == np.float64
        assert counts.dtype == np.int64

    def test_times_start_at_zero(self):
        """Time bins always start at zero."""
        # Photon way out at 100 ms
        abstimes = np.array([100e6])
        times, counts = bin_photons(abstimes, bin_size_ms=10.0)

        assert times[0] == 0.0


class TestBuildDecayHistogram:
    """Tests for build_decay_histogram function."""

    def test_empty_input(self):
        """Empty microtimes returns empty arrays."""
        t, counts = build_decay_histogram(np.array([]), channelwidth=0.1)
        assert len(t) == 0
        assert len(counts) == 0

    def test_single_microtime(self):
        """Single microtime creates histogram with explicit range."""
        microtimes = np.array([5.0])  # 5 ns
        # Need explicit range since single point has no span
        t, counts = build_decay_histogram(microtimes, channelwidth=1.0, tmin=0.0, tmax=10.0)

        # Should have bin containing 5.0
        assert 5.0 in t or any((t >= 4.5) & (t <= 5.5))
        assert counts.sum() == 1

    def test_histogram_shape(self):
        """Histogram has expected shape."""
        # Microtimes from 0 to 10 ns
        microtimes = np.linspace(1, 10, 100)
        t, counts = build_decay_histogram(microtimes, channelwidth=1.0)

        # Should have ~10 bins
        assert len(t) == len(counts)
        assert len(t) >= 9

    def test_channelwidth_determines_resolution(self):
        """Channel width determines histogram resolution."""
        microtimes = np.array([1.0, 1.5, 2.0, 2.5, 3.0])

        # Coarse bins
        t1, counts1 = build_decay_histogram(microtimes, channelwidth=1.0)

        # Fine bins
        t2, counts2 = build_decay_histogram(microtimes, channelwidth=0.5)

        # Fine resolution has more bins
        assert len(t2) > len(t1)
        # Total counts same
        assert counts1.sum() == counts2.sum()

    def test_custom_tmin_tmax(self):
        """Custom tmin/tmax limits histogram range."""
        microtimes = np.array([1.0, 5.0, 10.0, 15.0])

        t, counts = build_decay_histogram(
            microtimes, channelwidth=1.0, tmin=3.0, tmax=12.0
        )

        # Should only include microtimes within range
        assert t[0] >= 3.0
        assert t[-1] <= 12.0
        # Only 5.0 and 10.0 are in range
        assert counts.sum() == 2

    def test_exponential_decay_shape(self):
        """Histogram of exponential decay has expected shape."""
        # Generate exponential decay (tau=5 ns)
        np.random.seed(42)
        tau = 5.0
        microtimes = np.random.exponential(tau, size=10000)

        t, counts = build_decay_histogram(microtimes, channelwidth=0.5)

        # First half should have more counts than second half
        mid = len(counts) // 2
        assert counts[:mid].sum() > counts[mid:].sum()

    def test_output_dtypes(self):
        """Output arrays have correct dtypes."""
        microtimes = np.array([1.0, 2.0, 3.0])
        t, counts = build_decay_histogram(microtimes, channelwidth=0.5)

        assert t.dtype == np.float64
        assert counts.dtype == np.int64

    def test_negative_times_filtered(self):
        """Negative times are filtered out."""
        microtimes = np.array([-1.0, 0.5, 1.0, 2.0])
        t, counts = build_decay_histogram(microtimes, channelwidth=0.5)

        # All time values should be positive
        assert all(t > 0)

    def test_total_counts_preserved(self):
        """Total counts equals number of microtimes in range."""
        microtimes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t, counts = build_decay_histogram(microtimes, channelwidth=0.5)

        assert counts.sum() == 5


class TestComputeIntensityCps:
    """Tests for compute_intensity_cps function."""

    def test_empty_input(self):
        """Empty counts returns empty array."""
        cps = compute_intensity_cps(np.array([], dtype=np.int64), bin_size_ms=1.0)
        assert len(cps) == 0

    def test_conversion_1ms_bins(self):
        """1 ms bins convert correctly."""
        # 100 counts in 1 ms = 100,000 cps
        counts = np.array([100], dtype=np.int64)
        cps = compute_intensity_cps(counts, bin_size_ms=1.0)

        assert cps[0] == pytest.approx(100_000)

    def test_conversion_10ms_bins(self):
        """10 ms bins convert correctly."""
        # 100 counts in 10 ms = 10,000 cps
        counts = np.array([100], dtype=np.int64)
        cps = compute_intensity_cps(counts, bin_size_ms=10.0)

        assert cps[0] == pytest.approx(10_000)

    def test_conversion_100us_bins(self):
        """100 us (0.1 ms) bins convert correctly."""
        # 10 counts in 0.1 ms = 100,000 cps
        counts = np.array([10], dtype=np.int64)
        cps = compute_intensity_cps(counts, bin_size_ms=0.1)

        assert cps[0] == pytest.approx(100_000)

    def test_multiple_bins(self):
        """Multiple bins converted independently."""
        counts = np.array([100, 200, 50], dtype=np.int64)
        cps = compute_intensity_cps(counts, bin_size_ms=1.0)

        assert len(cps) == 3
        assert cps[0] == pytest.approx(100_000)
        assert cps[1] == pytest.approx(200_000)
        assert cps[2] == pytest.approx(50_000)

    def test_output_dtype(self):
        """Output has float64 dtype."""
        counts = np.array([100], dtype=np.int64)
        cps = compute_intensity_cps(counts, bin_size_ms=1.0)

        assert cps.dtype == np.float64


class TestRebinHistogram:
    """Tests for rebin_histogram function."""

    def test_factor_one_returns_copy(self):
        """Factor of 1 returns a copy."""
        t = np.array([1.0, 2.0, 3.0])
        counts = np.array([10, 20, 30], dtype=np.int64)

        new_t, new_counts = rebin_histogram(t, counts, factor=1)

        np.testing.assert_array_equal(new_t, t)
        np.testing.assert_array_equal(new_counts, counts)
        # Verify it's a copy, not same object
        assert new_t is not t
        assert new_counts is not counts

    def test_factor_two(self):
        """Factor of 2 combines pairs of bins."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        counts = np.array([10, 20, 30, 40], dtype=np.int64)

        new_t, new_counts = rebin_histogram(t, counts, factor=2)

        assert len(new_t) == 2
        assert len(new_counts) == 2
        assert list(new_t) == [0.0, 2.0]  # First time of each pair
        assert list(new_counts) == [30, 70]  # Sum of pairs

    def test_factor_three(self):
        """Factor of 3 combines triplets."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        counts = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        new_t, new_counts = rebin_histogram(t, counts, factor=3)

        assert len(new_t) == 2
        assert list(new_t) == [0.0, 3.0]
        assert list(new_counts) == [6, 15]  # 1+2+3=6, 4+5+6=15

    def test_truncation_with_remainder(self):
        """Extra bins that don't fill a group are truncated."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # 5 bins
        counts = np.array([10, 20, 30, 40, 50], dtype=np.int64)

        new_t, new_counts = rebin_histogram(t, counts, factor=2)

        # 5 bins / 2 = 2 complete groups (last bin dropped)
        assert len(new_t) == 2
        assert list(new_counts) == [30, 70]

    def test_empty_input(self):
        """Empty input returns empty output."""
        t = np.array([])
        counts = np.array([], dtype=np.int64)

        new_t, new_counts = rebin_histogram(t, counts, factor=2)

        assert len(new_t) == 0
        assert len(new_counts) == 0

    def test_factor_larger_than_length(self):
        """Factor larger than array length returns empty."""
        t = np.array([1.0, 2.0])
        counts = np.array([10, 20], dtype=np.int64)

        new_t, new_counts = rebin_histogram(t, counts, factor=3)

        assert len(new_t) == 0
        assert len(new_counts) == 0

    def test_invalid_factor_raises(self):
        """Factor < 1 raises ValueError."""
        t = np.array([1.0, 2.0])
        counts = np.array([10, 20], dtype=np.int64)

        with pytest.raises(ValueError, match="factor must be >= 1"):
            rebin_histogram(t, counts, factor=0)

        with pytest.raises(ValueError, match="factor must be >= 1"):
            rebin_histogram(t, counts, factor=-1)

    def test_output_dtypes(self):
        """Output arrays have correct dtypes."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        counts = np.array([10, 20, 30, 40], dtype=np.int64)

        new_t, new_counts = rebin_histogram(t, counts, factor=2)

        assert new_t.dtype == np.float64
        assert new_counts.dtype == np.int64

    def test_preserves_total_counts(self):
        """Total counts preserved (minus truncated bins)."""
        t = np.arange(100, dtype=np.float64)
        counts = np.ones(100, dtype=np.int64)

        new_t, new_counts = rebin_histogram(t, counts, factor=4)

        # 100 / 4 = 25 complete groups
        assert len(new_counts) == 25
        assert new_counts.sum() == 100


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_bin_and_compute_intensity(self):
        """Bin photons and compute intensity pipeline."""
        # 1000 photons over 10 ms = ~100,000 cps average
        np.random.seed(42)
        abstimes = np.random.uniform(0, 10e6, size=1000)  # 0-10 ms in ns

        times, counts = bin_photons(abstimes, bin_size_ms=1.0)
        cps = compute_intensity_cps(counts, bin_size_ms=1.0)

        # Average should be around 100k cps (with some variance)
        assert 80_000 < np.mean(cps) < 120_000

    def test_decay_histogram_and_rebin(self):
        """Build decay and rebin pipeline."""
        # Exponential decay
        np.random.seed(42)
        microtimes = np.random.exponential(5.0, size=10000)

        # High resolution
        t1, counts1 = build_decay_histogram(microtimes, channelwidth=0.1)

        # Rebin to lower resolution
        t2, counts2 = rebin_histogram(t1, counts1, factor=10)

        # Rebinned should have fewer points
        assert len(t2) < len(t1)
        # But similar total counts (minus truncation)
        assert counts2.sum() <= counts1.sum()
        assert counts2.sum() >= counts1.sum() * 0.9  # At least 90%
