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
]
