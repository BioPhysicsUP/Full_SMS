"""Data models for lifetime fitting results."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FitResult:
    """Result of a fluorescence lifetime decay fit.

    Contains the fitted parameters, goodness-of-fit statistics, and
    the fitted curve data for visualization.

    Attributes:
        tau: Lifetime values in nanoseconds, one per exponential component.
        tau_std: Standard errors of tau values.
        amplitude: Relative amplitudes for each exponential component (sum to 1 if normalized).
        amplitude_std: Standard errors of amplitude values.
        shift: IRF shift in channels.
        shift_std: Standard error of IRF shift.
        chi_squared: Reduced chi-squared value (goodness of fit).
        durbin_watson: Durbin-Watson statistic for residual autocorrelation.
        dw_bounds: Durbin-Watson critical bounds (lower, upper) for significance testing.
        residuals: Weighted residuals (data - fit) / sigma.
        fitted_curve: The fitted decay curve.
        fit_start_index: Start index of fitting range in the histogram.
        fit_end_index: End index of fitting range in the histogram.
        background: Background value used in fit.
        num_exponentials: Number of exponential components (1, 2, or 3).
        average_lifetime: Amplitude-weighted average lifetime in nanoseconds.
    """

    tau: Tuple[float, ...]
    tau_std: Tuple[float, ...]
    amplitude: Tuple[float, ...]
    amplitude_std: Tuple[float, ...]
    shift: float
    shift_std: float
    chi_squared: float
    durbin_watson: float
    dw_bounds: Optional[Tuple[float, float]]
    residuals: NDArray[np.float64]
    fitted_curve: NDArray[np.float64]
    fit_start_index: int
    fit_end_index: int
    background: float
    num_exponentials: int
    average_lifetime: float

    @property
    def is_good_fit(self) -> bool:
        """Whether the fit is considered good based on chi-squared.

        A good fit typically has chi-squared close to 1.0 (between 0.8 and 1.2).
        """
        return 0.8 <= self.chi_squared <= 1.2

    @property
    def dw_is_acceptable(self) -> Optional[bool]:
        """Whether Durbin-Watson statistic indicates acceptable residuals.

        Returns None if dw_bounds is not available.
        Returns True if DW is above the upper critical bound.
        """
        if self.dw_bounds is None:
            return None
        return self.durbin_watson > self.dw_bounds[1]

    @property
    def fit_range(self) -> Tuple[int, int]:
        """Fitting range as (start_index, end_index) tuple."""
        return (self.fit_start_index, self.fit_end_index)

    def __post_init__(self) -> None:
        """Validate fit result data."""
        if len(self.tau) != self.num_exponentials:
            raise ValueError(
                f"tau length ({len(self.tau)}) must match num_exponentials ({self.num_exponentials})"
            )
        if len(self.tau_std) != self.num_exponentials:
            raise ValueError(
                f"tau_std length ({len(self.tau_std)}) must match num_exponentials ({self.num_exponentials})"
            )
        if len(self.amplitude) != self.num_exponentials:
            raise ValueError(
                f"amplitude length ({len(self.amplitude)}) must match num_exponentials ({self.num_exponentials})"
            )
        if len(self.amplitude_std) != self.num_exponentials:
            raise ValueError(
                f"amplitude_std length ({len(self.amplitude_std)}) must match num_exponentials ({self.num_exponentials})"
            )
        if self.num_exponentials not in (1, 2, 3):
            raise ValueError(f"num_exponentials must be 1, 2, or 3, got {self.num_exponentials}")
        if self.chi_squared < 0:
            raise ValueError(f"chi_squared must be non-negative, got {self.chi_squared}")
        if self.fit_start_index < 0:
            raise ValueError(f"fit_start_index must be non-negative, got {self.fit_start_index}")
        if self.fit_end_index < self.fit_start_index:
            raise ValueError(
                f"fit_end_index ({self.fit_end_index}) must be >= fit_start_index ({self.fit_start_index})"
            )

    @classmethod
    def from_fit_parameters(
        cls,
        tau: list[float],
        tau_std: list[float],
        amplitude: list[float],
        amplitude_std: list[float],
        shift: float,
        shift_std: float,
        chi_squared: float,
        durbin_watson: float,
        residuals: NDArray[np.float64],
        fitted_curve: NDArray[np.float64],
        fit_start_index: int,
        fit_end_index: int,
        background: float,
        dw_bounds: Optional[Tuple[float, float]] = None,
    ) -> "FitResult":
        """Create a FitResult from fit parameters.

        Automatically calculates the amplitude-weighted average lifetime.

        Args:
            tau: List of lifetime values in nanoseconds.
            tau_std: List of standard errors for tau values.
            amplitude: List of relative amplitudes.
            amplitude_std: List of standard errors for amplitudes.
            shift: IRF shift in channels.
            shift_std: Standard error of shift.
            chi_squared: Reduced chi-squared value.
            durbin_watson: Durbin-Watson statistic.
            residuals: Weighted residuals array.
            fitted_curve: Fitted decay curve array.
            fit_start_index: Start index of fitting range.
            fit_end_index: End index of fitting range.
            background: Background value used in fit.
            dw_bounds: Optional Durbin-Watson critical bounds.

        Returns:
            A new FitResult instance.
        """
        num_exp = len(tau)

        # Calculate amplitude-weighted average lifetime
        # <τ> = Σ(a_i * τ_i) / Σ(a_i)
        total_amp = sum(amplitude)
        if total_amp > 0:
            avg_lifetime = sum(a * t for a, t in zip(amplitude, tau)) / total_amp
        else:
            avg_lifetime = tau[0] if tau else 0.0

        return cls(
            tau=tuple(tau),
            tau_std=tuple(tau_std),
            amplitude=tuple(amplitude),
            amplitude_std=tuple(amplitude_std),
            shift=shift,
            shift_std=shift_std,
            chi_squared=chi_squared,
            durbin_watson=durbin_watson,
            dw_bounds=dw_bounds,
            residuals=residuals,
            fitted_curve=fitted_curve,
            fit_start_index=fit_start_index,
            fit_end_index=fit_end_index,
            background=background,
            num_exponentials=num_exp,
            average_lifetime=avg_lifetime,
        )
