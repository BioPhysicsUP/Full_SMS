"""Analysis algorithms: change point detection, clustering, lifetime fitting."""

from full_sms.analysis.change_point import (
    CPAParams,
    ChangePointResult,
    ConfidenceLevel,
    find_change_points,
)
from full_sms.analysis.histograms import (
    bin_photons,
    build_decay_histogram,
    compute_intensity_cps,
    rebin_histogram,
)
from full_sms.analysis.lifetime import (
    FitMethod,
    FitSettings,
    StartpointMode,
    calculate_boundaries,
    colorshift,
    durbin_watson_bounds,
    estimate_background,
    estimate_irf_background,
    fit_decay,
    simulate_irf,
)

__all__ = [
    # Change point analysis
    "find_change_points",
    "ChangePointResult",
    "ConfidenceLevel",
    "CPAParams",
    # Histograms
    "bin_photons",
    "build_decay_histogram",
    "compute_intensity_cps",
    "rebin_histogram",
    # Lifetime fitting
    "fit_decay",
    "FitMethod",
    "FitSettings",
    "StartpointMode",
    "calculate_boundaries",
    "colorshift",
    "durbin_watson_bounds",
    "estimate_background",
    "estimate_irf_background",
    "simulate_irf",
]
