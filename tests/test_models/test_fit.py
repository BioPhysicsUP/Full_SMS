"""Tests for fit result data models."""

import numpy as np
import pytest

from full_sms.models.fit import FitResult


class TestFitResult:
    """Tests for FitResult."""

    @pytest.fixture
    def single_exp_result(self) -> FitResult:
        """Create a sample single-exponential fit result."""
        return FitResult(
            tau=(2.5,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.05,),
            shift=5.2,
            shift_std=0.3,
            chi_squared=1.05,
            durbin_watson=1.95,
            dw_bounds=(1.5, 1.7),
            residuals=np.random.randn(100),
            fitted_curve=np.exp(-np.arange(100) / 25),
            fit_start_index=10,
            fit_end_index=110,
            background=50.0,
            num_exponentials=1,
            average_lifetime=2.5,
        )

    @pytest.fixture
    def biexp_result(self) -> FitResult:
        """Create a sample bi-exponential fit result."""
        return FitResult(
            tau=(1.5, 4.0),
            tau_std=(0.1, 0.2),
            amplitude=(0.6, 0.4),
            amplitude_std=(0.03, 0.03),
            shift=5.0,
            shift_std=0.2,
            chi_squared=0.98,
            durbin_watson=1.92,
            dw_bounds=(1.5, 1.7),
            residuals=np.random.randn(100),
            fitted_curve=np.exp(-np.arange(100) / 20),
            fit_start_index=5,
            fit_end_index=105,
            background=30.0,
            num_exponentials=2,
            average_lifetime=2.5,
        )

    def test_create_single_exponential(self, single_exp_result: FitResult) -> None:
        """Can create a single-exponential FitResult."""
        assert single_exp_result.tau == (2.5,)
        assert single_exp_result.tau_std == (0.1,)
        assert single_exp_result.amplitude == (1.0,)
        assert single_exp_result.num_exponentials == 1
        assert single_exp_result.chi_squared == 1.05
        assert single_exp_result.durbin_watson == 1.95

    def test_create_biexponential(self, biexp_result: FitResult) -> None:
        """Can create a bi-exponential FitResult."""
        assert biexp_result.tau == (1.5, 4.0)
        assert biexp_result.amplitude == (0.6, 0.4)
        assert biexp_result.num_exponentials == 2

    def test_create_triexponential(self) -> None:
        """Can create a tri-exponential FitResult."""
        result = FitResult(
            tau=(0.5, 2.0, 5.0),
            tau_std=(0.05, 0.1, 0.2),
            amplitude=(0.3, 0.5, 0.2),
            amplitude_std=(0.02, 0.03, 0.02),
            shift=4.0,
            shift_std=0.2,
            chi_squared=1.01,
            durbin_watson=1.88,
            dw_bounds=None,
            residuals=np.random.randn(100),
            fitted_curve=np.exp(-np.arange(100) / 20),
            fit_start_index=0,
            fit_end_index=100,
            background=20.0,
            num_exponentials=3,
            average_lifetime=2.0,
        )

        assert result.num_exponentials == 3
        assert len(result.tau) == 3

    def test_is_good_fit_true(self) -> None:
        """is_good_fit returns True for chi-squared near 1.0."""
        result = FitResult(
            tau=(2.5,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.05,),
            shift=5.0,
            shift_std=0.3,
            chi_squared=1.0,
            durbin_watson=1.9,
            dw_bounds=None,
            residuals=np.array([0.0]),
            fitted_curve=np.array([1.0]),
            fit_start_index=0,
            fit_end_index=1,
            background=0.0,
            num_exponentials=1,
            average_lifetime=2.5,
        )

        assert result.is_good_fit is True

    def test_is_good_fit_false_high(self) -> None:
        """is_good_fit returns False for high chi-squared."""
        result = FitResult(
            tau=(2.5,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.05,),
            shift=5.0,
            shift_std=0.3,
            chi_squared=2.5,  # Too high
            durbin_watson=1.9,
            dw_bounds=None,
            residuals=np.array([0.0]),
            fitted_curve=np.array([1.0]),
            fit_start_index=0,
            fit_end_index=1,
            background=0.0,
            num_exponentials=1,
            average_lifetime=2.5,
        )

        assert result.is_good_fit is False

    def test_is_good_fit_false_low(self) -> None:
        """is_good_fit returns False for low chi-squared."""
        result = FitResult(
            tau=(2.5,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.05,),
            shift=5.0,
            shift_std=0.3,
            chi_squared=0.5,  # Too low
            durbin_watson=1.9,
            dw_bounds=None,
            residuals=np.array([0.0]),
            fitted_curve=np.array([1.0]),
            fit_start_index=0,
            fit_end_index=1,
            background=0.0,
            num_exponentials=1,
            average_lifetime=2.5,
        )

        assert result.is_good_fit is False

    def test_dw_is_acceptable_true(self, single_exp_result: FitResult) -> None:
        """dw_is_acceptable returns True when DW > upper bound."""
        # dw_bounds=(1.5, 1.7), durbin_watson=1.95
        assert single_exp_result.dw_is_acceptable is True

    def test_dw_is_acceptable_false(self) -> None:
        """dw_is_acceptable returns False when DW < upper bound."""
        result = FitResult(
            tau=(2.5,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.05,),
            shift=5.0,
            shift_std=0.3,
            chi_squared=1.0,
            durbin_watson=1.2,  # Below upper bound
            dw_bounds=(1.5, 1.7),
            residuals=np.array([0.0]),
            fitted_curve=np.array([1.0]),
            fit_start_index=0,
            fit_end_index=1,
            background=0.0,
            num_exponentials=1,
            average_lifetime=2.5,
        )

        assert result.dw_is_acceptable is False

    def test_dw_is_acceptable_none(self) -> None:
        """dw_is_acceptable returns None when dw_bounds is None."""
        result = FitResult(
            tau=(2.5,),
            tau_std=(0.1,),
            amplitude=(1.0,),
            amplitude_std=(0.05,),
            shift=5.0,
            shift_std=0.3,
            chi_squared=1.0,
            durbin_watson=1.9,
            dw_bounds=None,
            residuals=np.array([0.0]),
            fitted_curve=np.array([1.0]),
            fit_start_index=0,
            fit_end_index=1,
            background=0.0,
            num_exponentials=1,
            average_lifetime=2.5,
        )

        assert result.dw_is_acceptable is None

    def test_fit_range(self, single_exp_result: FitResult) -> None:
        """fit_range returns (start, end) tuple."""
        assert single_exp_result.fit_range == (10, 110)

    def test_tau_length_mismatch_raises(self) -> None:
        """Raises ValueError if tau length doesn't match num_exponentials."""
        with pytest.raises(ValueError, match="tau length.*must match"):
            FitResult(
                tau=(2.5, 4.0),  # 2 values
                tau_std=(0.1,),  # 1 value
                amplitude=(1.0,),
                amplitude_std=(0.05,),
                shift=5.0,
                shift_std=0.3,
                chi_squared=1.0,
                durbin_watson=1.9,
                dw_bounds=None,
                residuals=np.array([0.0]),
                fitted_curve=np.array([1.0]),
                fit_start_index=0,
                fit_end_index=1,
                background=0.0,
                num_exponentials=1,
                average_lifetime=2.5,
            )

    def test_invalid_num_exponentials_raises(self) -> None:
        """Raises ValueError for num_exponentials outside 1-3."""
        with pytest.raises(ValueError, match="num_exponentials must be 1, 2, or 3"):
            FitResult(
                tau=(1.0, 2.0, 3.0, 4.0),
                tau_std=(0.1, 0.1, 0.1, 0.1),
                amplitude=(0.25, 0.25, 0.25, 0.25),
                amplitude_std=(0.01, 0.01, 0.01, 0.01),
                shift=5.0,
                shift_std=0.3,
                chi_squared=1.0,
                durbin_watson=1.9,
                dw_bounds=None,
                residuals=np.array([0.0]),
                fitted_curve=np.array([1.0]),
                fit_start_index=0,
                fit_end_index=1,
                background=0.0,
                num_exponentials=4,  # Invalid
                average_lifetime=2.5,
            )

    def test_negative_chi_squared_raises(self) -> None:
        """Raises ValueError for negative chi_squared."""
        with pytest.raises(ValueError, match="chi_squared must be non-negative"):
            FitResult(
                tau=(2.5,),
                tau_std=(0.1,),
                amplitude=(1.0,),
                amplitude_std=(0.05,),
                shift=5.0,
                shift_std=0.3,
                chi_squared=-0.5,
                durbin_watson=1.9,
                dw_bounds=None,
                residuals=np.array([0.0]),
                fitted_curve=np.array([1.0]),
                fit_start_index=0,
                fit_end_index=1,
                background=0.0,
                num_exponentials=1,
                average_lifetime=2.5,
            )

    def test_fit_range_inverted_raises(self) -> None:
        """Raises ValueError when fit_end_index < fit_start_index."""
        with pytest.raises(ValueError, match="fit_end_index.*must be >= fit_start_index"):
            FitResult(
                tau=(2.5,),
                tau_std=(0.1,),
                amplitude=(1.0,),
                amplitude_std=(0.05,),
                shift=5.0,
                shift_std=0.3,
                chi_squared=1.0,
                durbin_watson=1.9,
                dw_bounds=None,
                residuals=np.array([0.0]),
                fitted_curve=np.array([1.0]),
                fit_start_index=100,
                fit_end_index=50,  # Before start
                background=0.0,
                num_exponentials=1,
                average_lifetime=2.5,
            )


class TestFitResultFromFitParameters:
    """Tests for FitResult.from_fit_parameters factory method."""

    def test_from_fit_parameters_single_exp(self) -> None:
        """Can create single-exponential FitResult from parameters."""
        result = FitResult.from_fit_parameters(
            tau=[2.5],
            tau_std=[0.1],
            amplitude=[1.0],
            amplitude_std=[0.05],
            shift=5.0,
            shift_std=0.3,
            chi_squared=1.02,
            durbin_watson=1.9,
            residuals=np.zeros(50),
            fitted_curve=np.ones(50),
            fit_start_index=10,
            fit_end_index=60,
            background=25.0,
        )

        assert result.num_exponentials == 1
        assert result.tau == (2.5,)
        assert result.average_lifetime == pytest.approx(2.5)

    def test_from_fit_parameters_biexp(self) -> None:
        """Can create bi-exponential FitResult with calculated average lifetime."""
        result = FitResult.from_fit_parameters(
            tau=[1.0, 4.0],
            tau_std=[0.1, 0.2],
            amplitude=[0.5, 0.5],  # Equal amplitudes
            amplitude_std=[0.02, 0.02],
            shift=5.0,
            shift_std=0.3,
            chi_squared=0.98,
            durbin_watson=1.85,
            residuals=np.zeros(50),
            fitted_curve=np.ones(50),
            fit_start_index=5,
            fit_end_index=55,
            background=20.0,
        )

        assert result.num_exponentials == 2
        # Average lifetime = (0.5*1.0 + 0.5*4.0) / 1.0 = 2.5
        assert result.average_lifetime == pytest.approx(2.5)

    def test_from_fit_parameters_weighted_average(self) -> None:
        """Average lifetime is properly amplitude-weighted."""
        result = FitResult.from_fit_parameters(
            tau=[1.0, 5.0],
            tau_std=[0.1, 0.2],
            amplitude=[0.8, 0.2],  # 80% short, 20% long
            amplitude_std=[0.02, 0.02],
            shift=5.0,
            shift_std=0.3,
            chi_squared=1.0,
            durbin_watson=1.9,
            residuals=np.zeros(50),
            fitted_curve=np.ones(50),
            fit_start_index=0,
            fit_end_index=50,
            background=10.0,
        )

        # Average lifetime = (0.8*1.0 + 0.2*5.0) / 1.0 = 1.8
        assert result.average_lifetime == pytest.approx(1.8)

    def test_from_fit_parameters_with_dw_bounds(self) -> None:
        """Can create FitResult with Durbin-Watson bounds."""
        result = FitResult.from_fit_parameters(
            tau=[3.0],
            tau_std=[0.15],
            amplitude=[1.0],
            amplitude_std=[0.05],
            shift=4.5,
            shift_std=0.25,
            chi_squared=1.05,
            durbin_watson=1.92,
            residuals=np.zeros(100),
            fitted_curve=np.ones(100),
            fit_start_index=0,
            fit_end_index=100,
            background=15.0,
            dw_bounds=(1.5, 1.7),
        )

        assert result.dw_bounds == (1.5, 1.7)
        assert result.dw_is_acceptable is True
