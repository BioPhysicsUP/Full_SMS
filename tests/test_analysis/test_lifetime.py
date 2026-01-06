"""Tests for fluorescence lifetime fitting."""

import numpy as np
import pytest

from full_sms.analysis.lifetime import (
    BACKGROUND_SECTION_LENGTH,
    DEFAULT_SETTINGS,
    FitMethod,
    FitSettings,
    StartpointMode,
    _max_continuous_zeros,
    _moving_avg,
    calculate_boundaries,
    colorshift,
    durbin_watson_bounds,
    estimate_background,
    estimate_irf_background,
    fit_decay,
    simulate_irf,
)
from full_sms.models.fit import FitResult


class TestMovingAvg:
    """Tests for _moving_avg function."""

    def test_identity_window_1(self):
        """Window of 1 should return original data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _moving_avg(data, window_length=1)
        np.testing.assert_array_almost_equal(result, data)

    def test_smoothing_effect(self):
        """Moving average should smooth data."""
        data = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        result = _moving_avg(data, window_length=3)
        # Should reduce peak-to-peak variation
        assert result.max() - result.min() < data.max() - data.min()

    def test_preserves_length_with_padding(self):
        """With padding, output should have same length as input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _moving_avg(data, window_length=3, pad_same_size=True)
        assert len(result) == len(data)

    def test_uniform_data(self):
        """Uniform interior values should remain uniform."""
        # With zero-padding, edge values will be affected
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = _moving_avg(data, window_length=3)
        # Interior values should remain uniform
        np.testing.assert_array_almost_equal(result[1:-1], data[1:-1])


class TestMaxContinuousZeros:
    """Tests for _max_continuous_zeros function."""

    def test_no_zeros(self):
        """Should return 0 when no zeros present."""
        data = np.array([1.0, 2.0, 3.0])
        assert _max_continuous_zeros(data) == 0

    def test_single_zero(self):
        """Should return 1 for single zero."""
        data = np.array([1.0, 0.0, 2.0])
        assert _max_continuous_zeros(data) == 1

    def test_consecutive_zeros(self):
        """Should find longest run of consecutive zeros."""
        data = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        assert _max_continuous_zeros(data) == 3

    def test_empty_array(self):
        """Empty array should return 0."""
        assert _max_continuous_zeros(np.array([])) == 0

    def test_all_zeros(self):
        """Array of all zeros."""
        data = np.zeros(5)
        assert _max_continuous_zeros(data) == 5


class TestColorshift:
    """Tests for colorshift function."""

    def test_zero_shift(self):
        """Zero shift should return original."""
        irf = np.array([0.0, 0.0, 1.0, 0.5, 0.2, 0.0])
        result = colorshift(irf, 0.0)
        np.testing.assert_array_almost_equal(result, irf)

    def test_integer_shift(self):
        """Integer shift should shift exactly."""
        irf = np.array([1.0, 0.0, 0.0, 0.0])
        result = colorshift(irf, 1.0)
        expected = np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_shift(self):
        """Negative shift should shift left with wrapping."""
        irf = np.array([0.0, 1.0, 0.0, 0.0])
        result = colorshift(irf, -1.0)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_periodic_wrapping(self):
        """Shift past end should wrap around."""
        irf = np.array([0.0, 0.0, 0.0, 1.0])
        result = colorshift(irf, 1.0)
        # Value at index 3 should move to index 0 (wrap)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_fractional_shift(self):
        """Fractional shift should interpolate."""
        irf = np.array([0.0, 1.0, 0.0, 0.0])
        result = colorshift(irf, 0.5)
        # Should have non-zero values at positions 1 and 2
        assert result[1] > 0 and result[2] > 0

    def test_preserves_total(self):
        """Total should be approximately preserved."""
        irf = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
        result = colorshift(irf, 2.3)
        np.testing.assert_almost_equal(np.sum(result), np.sum(irf), decimal=10)


class TestEstimateBackground:
    """Tests for estimate_background function."""

    def test_flat_before_rise(self):
        """Should find background from flat region before rise."""
        # Create decay with flat background of 10
        bg_level = 10.0
        decay = np.ones(200) * bg_level
        decay[50:] = np.exp(-np.arange(150) / 20.0) * 1000 + bg_level

        bg = estimate_background(decay)
        # Background should be close to the flat level
        assert abs(bg - bg_level) < 5

    def test_empty_array(self):
        """Empty array should return 0."""
        assert estimate_background(np.array([])) == 0.0

    def test_return_rise_index(self):
        """Should return rise index when requested."""
        decay = np.concatenate([
            np.ones(20) * 5,  # flat region
            np.arange(1, 101) * 10,  # rise
        ])
        rise_idx = estimate_background(decay, return_rise_index=True)
        # Rise index should be around the transition
        assert isinstance(rise_idx, (int, np.integer))
        assert rise_idx < 50

    def test_all_zeros(self):
        """All zeros should return 0."""
        bg = estimate_background(np.zeros(100))
        assert bg == 0.0


class TestEstimateIrfBackground:
    """Tests for estimate_irf_background function."""

    def test_irf_with_offset(self):
        """Should estimate background from flat region before peak."""
        # Create IRF with background offset
        bg_level = 5.0
        irf = np.ones(100) * bg_level
        irf[30:50] += np.exp(-((np.arange(20) - 10) ** 2) / 10) * 100

        bg = estimate_irf_background(irf)
        assert abs(bg - bg_level) < 2

    def test_empty_irf(self):
        """Empty IRF should return 0."""
        assert estimate_irf_background(np.array([])) == 0.0

    def test_irf_no_background(self):
        """IRF starting from zero should return ~0."""
        irf = np.zeros(100)
        irf[20:30] = np.exp(-((np.arange(10) - 5) ** 2) / 5) * 100
        bg = estimate_irf_background(irf)
        assert bg < 1


class TestCalculateBoundaries:
    """Tests for calculate_boundaries function."""

    def test_manual_boundaries(self):
        """Manual boundaries should be returned directly."""
        decay = np.exp(-np.arange(100) / 20.0) * 1000 + 10
        start, end = calculate_boundaries(
            decay, channelwidth=0.1, start=10, end=80,
            autostart=StartpointMode.MANUAL, autoend=False
        )
        assert start == 10
        assert end == 80

    def test_auto_end(self):
        """Auto end should find decay endpoint."""
        decay = np.exp(-np.arange(200) / 20.0) * 1000 + 10
        start, end = calculate_boundaries(
            decay, channelwidth=0.1, start=0,
            autostart=StartpointMode.MANUAL, autoend=True
        )
        # End should be before the last point (decay drops to background)
        assert end < 200

    def test_empty_array(self):
        """Empty array should return (0, 0)."""
        start, end = calculate_boundaries(np.array([]), channelwidth=0.1)
        assert start == 0 and end == 0

    def test_autostart_close_to_max(self):
        """Close to max should start near peak."""
        decay = np.zeros(100)
        decay[40:] = np.exp(-np.arange(60) / 20.0) * 1000

        start, end = calculate_boundaries(
            decay, channelwidth=0.1,
            autostart=StartpointMode.CLOSE_TO_MAX, autoend=False
        )
        # Start should be near the peak (around index 40)
        assert 35 <= start <= 45

    def test_boundary_order(self):
        """End should always be greater than start."""
        decay = np.exp(-np.arange(50) / 20.0) * 1000 + 10
        start, end = calculate_boundaries(
            decay, channelwidth=0.1, start=0, autoend=True
        )
        assert end > start


class TestDurbinWatsonBounds:
    """Tests for durbin_watson_bounds function."""

    def test_returns_four_values(self):
        """Should return four bound values."""
        bounds = durbin_watson_bounds(100, 2)
        assert len(bounds) == 4

    def test_bounds_order(self):
        """5% bound should be > 1% bound > 0.3% > 0.1%."""
        bounds = durbin_watson_bounds(200, 2)
        assert bounds[0] > bounds[1] > bounds[2] > bounds[3]

    def test_bounds_less_than_2(self):
        """All bounds should be less than 2 (ideal DW value)."""
        bounds = durbin_watson_bounds(100, 2)
        assert all(b < 2 for b in bounds)

    def test_larger_sample_bounds(self):
        """Larger samples should have bounds closer to 2."""
        bounds_small = durbin_watson_bounds(50, 2)
        bounds_large = durbin_watson_bounds(500, 2)
        assert bounds_large[0] > bounds_small[0]


class TestSimulateIrf:
    """Tests for simulate_irf function."""

    def test_creates_gaussian(self):
        """Should create Gaussian-shaped IRF."""
        measured = np.zeros(100)
        measured[50] = 1000  # Peak at index 50

        irf, t = simulate_irf(channelwidth=0.1, fwhm=0.5, measured=measured)

        # IRF should peak near index 50
        assert np.argmax(irf) == 50

    def test_fwhm_affects_width(self):
        """Larger FWHM should create wider IRF."""
        measured = np.zeros(100)
        measured[50] = 1000

        irf_narrow, _ = simulate_irf(channelwidth=0.1, fwhm=0.3, measured=measured)
        irf_wide, _ = simulate_irf(channelwidth=0.1, fwhm=1.0, measured=measured)

        # Count bins above half max
        narrow_width = np.sum(irf_narrow > 0.5 * irf_narrow.max())
        wide_width = np.sum(irf_wide > 0.5 * irf_wide.max())

        assert wide_width > narrow_width

    def test_time_axis_length(self):
        """Time axis should match measured length."""
        measured = np.zeros(100)
        irf, t = simulate_irf(channelwidth=0.1, fwhm=0.5, measured=measured)
        assert len(t) == len(measured)
        assert len(irf) == len(measured)

    def test_time_axis_scaling(self):
        """Time axis should be in nanoseconds."""
        measured = np.zeros(100)
        irf, t = simulate_irf(channelwidth=0.1, fwhm=0.5, measured=measured)
        # With channelwidth=0.1ns, last time should be 9.9ns
        np.testing.assert_almost_equal(t[-1], 9.9)


class TestFitDecay:
    """Tests for fit_decay function."""

    @pytest.fixture
    def synthetic_single_exp(self):
        """Create synthetic single-exponential decay data.

        Creates data mimicking real TCSPC measurements with:
        - Flat background region before the decay (~100 channels)
        - Sharp rise at IRF position
        - Exponential decay following the rise
        """
        np.random.seed(42)  # For reproducibility
        tau_true = 5.0  # ns
        channelwidth = 0.1  # ns
        num_channels = 1000  # Longer trace for realistic decay
        background = 50.0

        t = np.arange(num_channels) * channelwidth

        # Create IRF - centered with enough pre-rise background
        irf_center_channel = 100  # 10ns in, leaves flat region for bg estimation
        irf_sigma_channels = 3  # ~0.3ns FWHM
        irf = np.exp(-((np.arange(num_channels) - irf_center_channel) ** 2) /
                     (2 * irf_sigma_channels ** 2))

        # Create exponential decay starting from origin
        decay_model = np.exp(-t / tau_true)

        # Convolve: result will have decay starting at IRF position
        from scipy.signal import convolve
        convolved = convolve(irf, decay_model, mode='full')[:num_channels]

        # Scale peak to reasonable counts
        peak_counts = 20000
        convolved = convolved / convolved.max() * peak_counts

        # Add uniform background
        signal_with_bg = convolved + background

        # Add Poisson noise
        counts = np.random.poisson(signal_with_bg).astype(np.int64)

        # Normalize IRF for fitting use
        irf_normed = irf / irf.max()

        return {
            't': t,
            'counts': counts,
            'irf': irf_normed,
            'channelwidth': channelwidth,
            'tau_true': tau_true,
            'background': background,
        }

    def test_recovers_tau(self, synthetic_single_exp):
        """Fit should recover known tau value."""
        data = synthetic_single_exp

        result = fit_decay(
            t=data['t'],
            counts=data['counts'],
            channelwidth=data['channelwidth'],
            irf=data['irf'],
            num_exponentials=1,
            tau_init=4.0,  # Close but not exact
            autostart=StartpointMode.CLOSE_TO_MAX,
            autoend=True,
            irf_background=0.0,  # Synthetic IRF is already baseline-corrected
        )

        # Should recover tau within 30% of true value
        relative_error = abs(result.tau[0] - data['tau_true']) / data['tau_true']
        assert relative_error < 0.3, f"Tau={result.tau[0]:.2f}, expected ~{data['tau_true']}"

    def test_returns_fit_result(self, synthetic_single_exp):
        """Should return FitResult object with all fields."""
        data = synthetic_single_exp

        result = fit_decay(
            t=data['t'],
            counts=data['counts'],
            channelwidth=data['channelwidth'],
            irf=data['irf'],
        )

        assert isinstance(result, FitResult)
        assert len(result.tau) == 1
        assert len(result.amplitude) == 1
        assert result.chi_squared > 0
        assert len(result.residuals) > 0
        assert len(result.fitted_curve) > 0

    def test_chi_squared_reasonable(self, synthetic_single_exp):
        """Chi-squared should be in a reasonable range."""
        data = synthetic_single_exp

        result = fit_decay(
            t=data['t'],
            counts=data['counts'],
            channelwidth=data['channelwidth'],
            irf=data['irf'],
            autostart=StartpointMode.CLOSE_TO_MAX,
            autoend=True,
        )

        # Chi-squared should be positive and finite
        # For Poisson-noisy data, chi-squared can vary widely
        assert result.chi_squared > 0
        assert np.isfinite(result.chi_squared)

    def test_durbin_watson_range(self, synthetic_single_exp):
        """Durbin-Watson should be between 0 and 4."""
        data = synthetic_single_exp

        result = fit_decay(
            t=data['t'],
            counts=data['counts'],
            channelwidth=data['channelwidth'],
            irf=data['irf'],
        )

        assert 0 <= result.durbin_watson <= 4

    def test_custom_tau_bounds(self, synthetic_single_exp):
        """Should respect tau bounds."""
        data = synthetic_single_exp

        result = fit_decay(
            t=data['t'],
            counts=data['counts'],
            channelwidth=data['channelwidth'],
            irf=data['irf'],
            tau_bounds=(4.0, 6.0),
        )

        assert 4.0 <= result.tau[0] <= 6.0

    def test_fit_without_irf(self, synthetic_single_exp):
        """Should work without IRF (uses delta function)."""
        data = synthetic_single_exp

        # This will be a poor fit but should not error
        result = fit_decay(
            t=data['t'],
            counts=data['counts'],
            channelwidth=data['channelwidth'],
            irf=None,  # No IRF
        )

        assert isinstance(result, FitResult)
        assert result.tau[0] > 0

    def test_raises_on_multi_exp(self, synthetic_single_exp):
        """Should raise error for multi-exponential (not yet supported)."""
        data = synthetic_single_exp

        with pytest.raises(ValueError, match="single exponential"):
            fit_decay(
                t=data['t'],
                counts=data['counts'],
                channelwidth=data['channelwidth'],
                num_exponentials=2,
            )

    def test_raises_on_empty_data(self):
        """Should raise error for empty data."""
        with pytest.raises(ValueError, match="Empty data"):
            fit_decay(
                t=np.array([]),
                counts=np.array([]),
                channelwidth=0.1,
            )

    def test_raises_on_length_mismatch(self):
        """Should raise error for length mismatch."""
        with pytest.raises(ValueError, match="Length mismatch"):
            fit_decay(
                t=np.arange(100) * 0.1,
                counts=np.ones(50, dtype=np.int64),
                channelwidth=0.1,
            )

    def test_fit_indices_in_result(self, synthetic_single_exp):
        """Fit result should include start/end indices."""
        data = synthetic_single_exp

        result = fit_decay(
            t=data['t'],
            counts=data['counts'],
            channelwidth=data['channelwidth'],
            irf=data['irf'],
            start=10,
            end=400,
        )

        assert result.fit_start_index == 10
        assert result.fit_end_index == 400

    def test_background_in_result(self, synthetic_single_exp):
        """Fit result should include background value."""
        data = synthetic_single_exp

        result = fit_decay(
            t=data['t'],
            counts=data['counts'],
            channelwidth=data['channelwidth'],
            irf=data['irf'],
        )

        # Background should be estimated
        assert result.background >= 0

    def test_average_lifetime_equals_tau_for_single_exp(self, synthetic_single_exp):
        """For single exponential, average lifetime should equal tau."""
        data = synthetic_single_exp

        result = fit_decay(
            t=data['t'],
            counts=data['counts'],
            channelwidth=data['channelwidth'],
            irf=data['irf'],
        )

        np.testing.assert_almost_equal(result.average_lifetime, result.tau[0])


class TestFitSettings:
    """Tests for FitSettings dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        settings = FitSettings()

        assert settings.use_moving_avg is True
        assert settings.moving_avg_window > 0
        assert 0 < settings.start_percent <= 100
        assert settings.end_multiple > 0
        assert 0 < settings.end_percent <= 100
        assert settings.minimum_decay_window_ns > 0

    def test_custom_values(self):
        """Should accept custom values."""
        settings = FitSettings(
            use_moving_avg=False,
            moving_avg_window=20,
            start_percent=50.0,
        )

        assert settings.use_moving_avg is False
        assert settings.moving_avg_window == 20
        assert settings.start_percent == 50.0


class TestEnums:
    """Tests for enum classes."""

    def test_fit_method_values(self):
        """FitMethod should have expected values."""
        assert FitMethod.LEAST_SQUARES.value == "ls"
        assert FitMethod.MAXIMUM_LIKELIHOOD.value == "ml"

    def test_startpoint_mode_values(self):
        """StartpointMode should have expected values."""
        assert StartpointMode.MANUAL.value == "Manual"
        assert StartpointMode.CLOSE_TO_MAX.value == "(Close to) max"
        assert StartpointMode.RISE_MIDDLE.value == "Rise middle"
        assert StartpointMode.RISE_START.value == "Rise start"
        assert StartpointMode.SAFE_RISE_START.value == "Safe rise start"


class TestIntegration:
    """Integration tests for lifetime fitting."""

    def test_full_fitting_pipeline(self):
        """Test complete fitting pipeline with realistic data."""
        np.random.seed(123)

        # Create realistic decay
        channelwidth = 0.064  # Common TCSPC channel width
        num_channels = 4096
        tau_true = 3.5  # ns

        t = np.arange(num_channels) * channelwidth

        # Create narrow IRF
        irf_center = 20 * channelwidth
        irf_sigma = 2 * channelwidth
        irf = np.exp(-((t - irf_center) ** 2) / (2 * irf_sigma ** 2))

        # Create and convolve decay
        from scipy.signal import convolve
        decay_model = np.exp(-t / tau_true)
        convolved = convolve(irf, decay_model, mode='full')[:num_channels]

        # Scale and add noise
        scale = 50000
        background = 100
        convolved = convolved / convolved.max() * scale
        counts = np.random.poisson(convolved + background).astype(np.int64)

        # Fit
        result = fit_decay(
            t=t,
            counts=counts,
            channelwidth=channelwidth,
            irf=irf,
            tau_init=3.0,
            autostart=StartpointMode.CLOSE_TO_MAX,
            autoend=True,
        )

        # Check that fit completed successfully with reasonable values
        assert result.tau[0] > 0
        assert result.chi_squared > 0
        assert np.isfinite(result.chi_squared)
        assert result.dw_bounds is not None

    def test_fitting_with_custom_boundaries(self):
        """Test fitting with manually specified boundaries."""
        channelwidth = 0.1
        num_channels = 500
        tau_true = 5.0

        t = np.arange(num_channels) * channelwidth
        decay = np.exp(-t / tau_true) * 10000
        decay = np.roll(decay, 50)  # Shift peak
        background = 20
        counts = np.random.poisson(decay + background).astype(np.int64)

        result = fit_decay(
            t=t,
            counts=counts,
            channelwidth=channelwidth,
            start=55,
            end=400,
            background=background,
        )

        assert result.fit_start_index == 55
        assert result.fit_end_index == 400
