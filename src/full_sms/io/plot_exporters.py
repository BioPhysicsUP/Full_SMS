"""Plot export functions using matplotlib for publication-quality figures.

Since DearPyGui/ImPlot doesn't have direct export capabilities, this module
recreates plots using matplotlib for export to PNG, PDF, and SVG formats.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from full_sms.analysis.histograms import bin_photons
from full_sms.analysis.lifetime import compute_convolved_fit_curve
from full_sms.models.fit import FitResultData, IRFData

if TYPE_CHECKING:
    from full_sms.models.fit import FitResult
    from full_sms.models.group import ClusteringResult, GroupData
    from full_sms.models.level import LevelData
    from full_sms.models.measurement import MeasurementData


logger = logging.getLogger(__name__)


class PlotFormat(Enum):
    """Supported plot export formats."""

    PNG = "png"
    PDF = "pdf"
    SVG = "svg"


# Default color palette for levels and groups
LEVEL_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def _get_matplotlib():
    """Import and configure matplotlib with non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless operation
    import matplotlib.pyplot as plt
    return plt


def _get_level_color(index: int) -> str:
    """Get color for a level or group by index."""
    return LEVEL_COLORS[index % len(LEVEL_COLORS)]


def _ensure_directory(path: Path) -> None:
    """Ensure the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def export_intensity_plot(
    abstimes: NDArray[np.uint64],
    output_path: Path,
    bin_size_ms: float = 10.0,
    levels: list[LevelData] | None = None,
    groups: list[GroupData] | None = None,
    show_levels: bool = True,
    show_groups: bool = True,
    fmt: PlotFormat = PlotFormat.PNG,
    title: str = "",
    figsize: tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> Path:
    """Export an intensity trace plot to an image file.

    Creates a publication-quality intensity trace plot using matplotlib,
    optionally overlaying detected levels and/or group colors.

    Args:
        abstimes: Absolute photon arrival times in nanoseconds.
        output_path: Path to output file (without extension).
        bin_size_ms: Bin size in milliseconds.
        levels: Optional list of LevelData from change point analysis.
        groups: Optional list of GroupData from clustering.
        show_levels: Whether to show level boundaries.
        show_groups: Whether to color by group assignment.
        fmt: Export format (PNG, PDF, or SVG).
        title: Optional plot title.
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch (for PNG).

    Returns:
        Path to the exported file.
    """
    plt = _get_matplotlib()

    # Bin the photons
    times_ms, counts = bin_photons(abstimes.astype(np.float64), bin_size_ms)
    times_s = times_ms / 1000.0

    # Calculate intensity in kilo-counts per second
    bin_size_s = bin_size_ms / 1000.0
    intensity_kcps = (counts.astype(np.float64) / bin_size_s) / 1000.0

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot intensity trace as step function
    ax.step(times_s, intensity_kcps, where="post", color="#333333", linewidth=0.8, label="Intensity")

    # Draw level overlays
    if levels and (show_levels or show_groups):
        _draw_level_overlays(ax, levels, groups, times_s, intensity_kcps, show_groups)

    # Configure axes
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Intensity (kcps)", fontsize=11)
    ax.set_xlim(0, times_s[-1] if len(times_s) > 0 else 1)
    ax.set_ylim(0, None)

    if title:
        ax.set_title(title, fontsize=12)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    # Save
    output_file = output_path.with_suffix(f".{fmt.value}")
    _ensure_directory(output_file)

    save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
    if fmt == PlotFormat.PNG:
        save_kwargs["dpi"] = dpi

    fig.savefig(output_file, **save_kwargs)
    plt.close(fig)

    logger.info(f"Exported intensity plot to {output_file}")
    return output_file


def _draw_level_overlays(
    ax,
    levels: list[LevelData],
    groups: list[GroupData] | None,
    times_s: NDArray[np.float64],
    intensity_kcps: NDArray[np.float64],
    color_by_group: bool,
) -> None:
    """Draw level overlays on an intensity plot.

    Args:
        ax: Matplotlib axes.
        levels: List of LevelData.
        groups: Optional list of GroupData for group coloring.
        times_s: Time array in seconds.
        intensity_kcps: Intensity array in kcps.
        color_by_group: Whether to color by group assignment.
    """
    if not levels:
        return

    # Draw levels as a single red step line (matches intensity tab)
    sorted_levels = sorted(levels, key=lambda l: l.start_time_ns)

    # Build step line points
    step_times = []
    step_intensities = []

    for level in sorted_levels:
        start_s = level.start_time_ns / 1e9
        end_s = level.end_time_ns / 1e9
        intensity_kcps_val = level.intensity_cps / 1000.0

        step_times.extend([start_s, end_s])
        step_intensities.extend([intensity_kcps_val, intensity_kcps_val])

    # Draw the step line in red
    ax.plot(step_times, step_intensities, color="#d62728", linewidth=2, label="Levels", zorder=5)

    # Draw group bands and dashed lines (matches grouping tab)
    if groups and color_by_group:
        # Sort groups by average intensity to draw bands
        sorted_groups = sorted(groups, key=lambda g: g.intensity_cps)

        # Calculate band boundaries (midpoints between group intensities)
        group_intensities = [g.intensity_cps / 1000.0 for g in sorted_groups]

        # Assign alternating colors for bands
        band_colors = ["#e3f2fd", "#fff3e0"]  # Light blue and light orange

        for i, group in enumerate(sorted_groups):
            group_intensity_kcps = group.intensity_cps / 1000.0

            # Calculate band boundaries
            if i == 0:
                lower_bound = 0
            else:
                lower_bound = (group_intensities[i - 1] + group_intensities[i]) / 2

            if i == len(sorted_groups) - 1:
                upper_bound = max(intensity_kcps) * 1.1  # Extend to top
            else:
                upper_bound = (group_intensities[i] + group_intensities[i + 1]) / 2

            # Draw horizontal band
            color = band_colors[i % len(band_colors)]
            ax.axhspan(lower_bound, upper_bound, alpha=0.2, color=color, linewidth=0, zorder=1)

            # Draw dashed line at group average intensity
            ax.axhline(
                group_intensity_kcps,
                color="#666666",
                linewidth=1,
                linestyle="--",
                alpha=0.7,
                zorder=2,
            )


def export_decay_plot(
    t_ns: NDArray[np.float64],
    counts: NDArray[np.int64],
    output_path: Path,
    channelwidth: float = 0.1,
    fit_result: FitResult | None = None,
    fit_data: FitResultData | None = None,
    irf_data: IRFData | None = None,
    irf_t: NDArray[np.float64] | None = None,
    irf_counts: NDArray[np.float64] | None = None,
    log_scale: bool = True,
    show_residuals: bool = True,
    show_fit: bool = True,
    show_irf: bool = True,
    fmt: PlotFormat = PlotFormat.PNG,
    title: str = "",
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150,
) -> Path:
    """Export a fluorescence decay plot to an image file.

    Creates a publication-quality decay plot with optional fit curve,
    IRF, and residuals panel.

    For fit curve export, provide either:
    - `fit_result`: A full FitResult with pre-computed arrays (backwards compatible)
    - `fit_data` + `irf_data`: FitResultData (scalars only) + IRFData, and the
      fit curve will be recomputed using the shared computation function.

    Args:
        t_ns: Time array in nanoseconds.
        counts: Photon counts array.
        output_path: Path to output file (without extension).
        channelwidth: TCSPC channel width in nanoseconds (required for fit_data).
        fit_result: Optional FitResult for overlay (legacy, has pre-computed curve).
        fit_data: Optional FitResultData for overlay (curve will be computed).
        irf_data: Optional IRFData for computing fit curve and/or displaying IRF.
        irf_t: Optional IRF time array (legacy, for direct display).
        irf_counts: Optional IRF counts array (legacy, for direct display).
        log_scale: Whether to use logarithmic y-axis.
        show_residuals: Whether to show residuals subplot (if fit provided).
        show_fit: Whether to show the fit curve (if fit data provided).
        show_irf: Whether to show the IRF (if IRF data provided).
        fmt: Export format (PNG, PDF, or SVG).
        title: Optional plot title.
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch (for PNG).

    Returns:
        Path to the exported file.
    """
    plt = _get_matplotlib()

    # Compute fit curve from FitResultData if provided (and no FitResult)
    fitted_curve: Optional[NDArray[np.float64]] = None
    residuals: Optional[NDArray[np.float64]] = None
    fit_start_index: int = 0
    fit_end_index: int = 0
    chi_squared: float = 0.0
    durbin_watson: float = 0.0
    average_lifetime: float = 0.0

    has_fit = False

    if fit_result is not None and show_fit:
        # Use legacy FitResult with pre-computed arrays
        has_fit = True
        fitted_curve = fit_result.fitted_curve
        residuals = fit_result.residuals
        fit_start_index = fit_result.fit_start_index
        fit_end_index = fit_result.fit_end_index
        chi_squared = fit_result.chi_squared
        durbin_watson = fit_result.durbin_watson
        average_lifetime = fit_result.average_lifetime

    elif fit_data is not None and irf_data is not None and show_fit:
        # Compute fit curve from FitResultData + IRFData
        has_fit = True
        fit_start_index = fit_data.fit_start_index
        fit_end_index = fit_data.fit_end_index
        chi_squared = fit_data.chi_squared
        durbin_watson = fit_data.durbin_watson
        average_lifetime = fit_data.average_lifetime

        try:
            # Determine IRF parameters
            irf_fwhm_ns: Optional[float] = None
            irf_array: Optional[NDArray[np.float64]] = None

            if irf_data.is_simulated:
                # Use fitted FWHM if available, otherwise use stored FWHM
                if fit_data.fitted_irf_fwhm is not None:
                    irf_fwhm_ns = fit_data.fitted_irf_fwhm
                else:
                    irf_fwhm_ns = irf_data.fwhm_ns
            else:
                # Use loaded IRF array
                irf_array = irf_data.get_irf_array()

            # Compute the fit curve
            full_curve, computed_residuals = compute_convolved_fit_curve(
                t_ns=t_ns,
                counts=counts,
                channelwidth=channelwidth,
                tau=fit_data.tau,
                amplitude=fit_data.amplitude,
                shift_ns=fit_data.shift,
                background=fit_data.background,
                fit_start_index=fit_start_index,
                fit_end_index=fit_end_index,
                irf_fwhm_ns=irf_fwhm_ns,
                irf_array=irf_array,
            )

            # Extract the fit range portion of the curve
            fitted_curve = full_curve[fit_start_index:fit_end_index]
            residuals = computed_residuals

        except Exception as e:
            logger.warning(f"Failed to compute fit curve for export: {e}")
            has_fit = False

    # Determine subplot layout
    show_resid = show_residuals and has_fit and residuals is not None

    if show_resid:
        fig, (ax_decay, ax_resid) = plt.subplots(
            2, 1,
            figsize=figsize,
            height_ratios=[3, 1],
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.05)
    else:
        fig, ax_decay = plt.subplots(figsize=figsize)
        ax_resid = None

    # Plot decay data
    ax_decay.scatter(
        t_ns, counts,
        s=3,
        alpha=0.6,
        color="#333333",
        label="Data",
    )

    # Plot fit curve
    if has_fit and fitted_curve is not None:
        fit_t = t_ns[fit_start_index:fit_start_index + len(fitted_curve)]
        ax_decay.plot(
            fit_t,
            fitted_curve,
            color="#d62728",
            linewidth=1.5,
            label=f"Fit (τ={average_lifetime:.2f} ns)",
        )

    # Determine IRF display data
    display_irf_t: Optional[NDArray[np.float64]] = None
    display_irf_counts: Optional[NDArray[np.float64]] = None

    if show_irf:
        if irf_t is not None and irf_counts is not None:
            # Use legacy direct IRF arrays
            display_irf_t = irf_t
            display_irf_counts = irf_counts
        elif irf_data is not None and not irf_data.is_simulated:
            # Use loaded IRF from IRFData
            # Note: Loaded IRF may have different time axis than decay data.
            # We use the decay histogram time axis (t_ns) for proper alignment.
            if irf_data.counts is not None and len(irf_data.counts) == len(t_ns):
                # IRF has same length as decay - use decay time axis with shift
                shift_ns = fit_data.shift if fit_data else 0.0
                display_irf_t = t_ns + shift_ns
                display_irf_counts = np.array(irf_data.counts)
            elif irf_data.time_ns is not None and irf_data.counts is not None:
                # Fallback: use IRF's own time axis with shift
                shift_ns = fit_data.shift if fit_data else 0.0
                display_irf_t = np.array(irf_data.time_ns) + shift_ns
                display_irf_counts = np.array(irf_data.counts)
        elif irf_data is not None and irf_data.is_simulated and fit_data is not None:
            # Generate simulated IRF for display
            from full_sms.analysis.lifetime import simulate_irf
            fwhm = fit_data.fitted_irf_fwhm if fit_data.fitted_irf_fwhm else irf_data.fwhm_ns
            if fwhm is not None:
                sim_irf, _ = simulate_irf(channelwidth, fwhm, counts.astype(np.float64))
                # Use decay histogram time axis (t_ns) for proper alignment,
                # since simulate_irf returns time starting from 0 but decay
                # histogram may start at a non-zero offset
                shift_ns = fit_data.shift if fit_data else 0.0
                display_irf_t = t_ns + shift_ns
                display_irf_counts = sim_irf

    # Plot IRF
    if display_irf_t is not None and display_irf_counts is not None:
        # Normalize IRF to peak of data for visibility
        irf_max = np.max(display_irf_counts)
        irf_scale = np.max(counts) / irf_max if irf_max > 0 else 1
        ax_decay.plot(
            display_irf_t,
            display_irf_counts * irf_scale,
            color="#1f77b4",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
            label="IRF",
        )

    # Configure decay axes
    ax_decay.set_ylabel("Counts", fontsize=11)
    if log_scale:
        ax_decay.set_yscale("log")
        # Set reasonable y limits for log scale
        min_nonzero = counts[counts > 0].min() if np.any(counts > 0) else 1
        ax_decay.set_ylim(max(1, min_nonzero * 0.5), np.max(counts) * 1.5)
    else:
        ax_decay.set_ylim(0, None)

    ax_decay.legend(loc="upper right", fontsize=9)

    if title:
        ax_decay.set_title(title, fontsize=12)

    # Style
    ax_decay.spines["top"].set_visible(False)
    ax_decay.spines["right"].set_visible(False)

    # Plot residuals
    if show_resid and ax_resid is not None and residuals is not None:
        resid_t = t_ns[fit_start_index:fit_start_index + len(residuals)]

        ax_resid.scatter(
            resid_t,
            residuals,
            s=2,
            alpha=0.5,
            color="#333333",
        )
        ax_resid.axhline(0, color="#888888", linewidth=0.8, linestyle="-")
        ax_resid.axhline(2, color="#cccccc", linewidth=0.5, linestyle="--")
        ax_resid.axhline(-2, color="#cccccc", linewidth=0.5, linestyle="--")

        ax_resid.set_xlabel("Time (ns)", fontsize=11)
        ax_resid.set_ylabel("Residuals", fontsize=11)
        ax_resid.set_ylim(-5, 5)

        ax_resid.spines["top"].set_visible(False)
        ax_resid.spines["right"].set_visible(False)

        # Add fit statistics as text
        stats_text = f"χ² = {chi_squared:.3f}  DW = {durbin_watson:.3f}"
        ax_resid.text(
            0.02, 0.95,
            stats_text,
            transform=ax_resid.transAxes,
            fontsize=9,
            verticalalignment="top",
            color="#666666",
        )
    else:
        ax_decay.set_xlabel("Time (ns)", fontsize=11)

    plt.tight_layout()

    # Save
    output_file = output_path.with_suffix(f".{fmt.value}")
    _ensure_directory(output_file)

    save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
    if fmt == PlotFormat.PNG:
        save_kwargs["dpi"] = dpi

    fig.savefig(output_file, **save_kwargs)
    plt.close(fig)

    logger.info(f"Exported decay plot to {output_file}")
    return output_file


def export_bic_plot(
    clustering_result: ClusteringResult,
    output_path: Path,
    fmt: PlotFormat = PlotFormat.PNG,
    title: str = "BIC Optimization",
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> Path:
    """Export a BIC optimization plot to an image file.

    Creates a plot showing BIC values vs number of groups, with the
    optimal point highlighted.

    Args:
        clustering_result: ClusteringResult from AHCA.
        output_path: Path to output file (without extension).
        fmt: Export format (PNG, PDF, or SVG).
        title: Plot title.
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch (for PNG).

    Returns:
        Path to the exported file.
    """
    plt = _get_matplotlib()

    # Extract BIC values and group counts
    bic_values = list(clustering_result.all_bic_values)
    num_groups = [step.num_groups for step in clustering_result.steps]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot BIC curve
    ax.plot(
        num_groups,
        bic_values,
        "o-",
        color="#1f77b4",
        linewidth=1.5,
        markersize=6,
        label="BIC",
    )

    # Highlight optimal point
    optimal_idx = clustering_result.optimal_step_index
    optimal_groups = num_groups[optimal_idx]
    optimal_bic = bic_values[optimal_idx]

    ax.scatter(
        [optimal_groups],
        [optimal_bic],
        s=150,
        color="#d62728",
        zorder=5,
        marker="*",
        label=f"Optimal ({optimal_groups} groups)",
    )

    # Highlight selected point if different from optimal
    if clustering_result.selected_step_index != optimal_idx:
        selected_idx = clustering_result.selected_step_index
        selected_groups = num_groups[selected_idx]
        selected_bic = bic_values[selected_idx]

        ax.scatter(
            [selected_groups],
            [selected_bic],
            s=100,
            color="#2ca02c",
            zorder=4,
            marker="s",
            label=f"Selected ({selected_groups} groups)",
        )

    # Configure axes
    ax.set_xlabel("Number of Groups", fontsize=11)
    ax.set_ylabel("BIC", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Integer x-axis ticks
    ax.set_xticks(num_groups)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=9)
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    # Save
    output_file = output_path.with_suffix(f".{fmt.value}")
    _ensure_directory(output_file)

    save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
    if fmt == PlotFormat.PNG:
        save_kwargs["dpi"] = dpi

    fig.savefig(output_file, **save_kwargs)
    plt.close(fig)

    logger.info(f"Exported BIC plot to {output_file}")
    return output_file


def export_correlation_plot(
    tau: NDArray[np.float64],
    g2: NDArray[np.float64],
    output_path: Path,
    fmt: PlotFormat = PlotFormat.PNG,
    title: str = "Second-Order Correlation",
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> Path:
    """Export a g2 correlation plot to an image file.

    Creates a plot of the second-order correlation function g2(τ).

    Args:
        tau: Delay time array (typically in ns).
        g2: Correlation values.
        output_path: Path to output file (without extension).
        fmt: Export format (PNG, PDF, or SVG).
        title: Plot title.
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch (for PNG).

    Returns:
        Path to the exported file.
    """
    plt = _get_matplotlib()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot g2 curve
    ax.plot(
        tau,
        g2,
        "-",
        color="#1f77b4",
        linewidth=1.2,
    )

    # Reference line at g2 = 1
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)

    # Reference line at tau = 0
    ax.axvline(0.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)

    # Configure axes
    ax.set_xlabel("τ (ns)", fontsize=11)
    ax.set_ylabel("g²(τ)", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    # Save
    output_file = output_path.with_suffix(f".{fmt.value}")
    _ensure_directory(output_file)

    save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
    if fmt == PlotFormat.PNG:
        save_kwargs["dpi"] = dpi

    fig.savefig(output_file, **save_kwargs)
    plt.close(fig)

    logger.info(f"Exported correlation plot to {output_file}")
    return output_file


def export_all_plots(
    measurement: MeasurementData,
    channel: int,
    output_dir: Path,
    levels: list[LevelData] | None = None,
    clustering_result: ClusteringResult | None = None,
    fit_result: FitResult | None = None,
    correlation_tau: NDArray[np.float64] | None = None,
    correlation_g2: NDArray[np.float64] | None = None,
    bin_size_ms: float = 10.0,
    fmt: PlotFormat = PlotFormat.PNG,
    dpi: int = 150,
) -> list[Path]:
    """Export all available plots for a measurement.

    Args:
        measurement: The measurement data.
        channel: Channel number (1 or 2).
        output_dir: Directory to save plots.
        levels: Optional levels from CPA.
        clustering_result: Optional clustering result.
        fit_result: Optional fit result.
        correlation_tau: Optional correlation tau array.
        correlation_g2: Optional correlation g2 array.
        bin_size_ms: Bin size for intensity plot.
        fmt: Export format.
        dpi: Resolution for PNG format.

    Returns:
        List of paths to exported files.
    """
    exported: list[Path] = []
    prefix = f"measurement_{measurement.id}_ch{channel}"

    # Get channel data
    channel_data = measurement.channel1 if channel == 1 else measurement.channel2
    if channel_data is None:
        logger.warning(f"No channel {channel} data for measurement {measurement.id}")
        return exported

    # Get groups from clustering result
    groups = list(clustering_result.groups) if clustering_result else None

    # Export intensity plot
    try:
        path = export_intensity_plot(
            abstimes=channel_data.abstimes,
            output_path=output_dir / f"{prefix}_intensity",
            bin_size_ms=bin_size_ms,
            levels=levels,
            groups=groups,
            fmt=fmt,
            title=f"Measurement {measurement.id} - Channel {channel}",
            dpi=dpi,
        )
        exported.append(path)
    except Exception as e:
        logger.warning(f"Failed to export intensity plot: {e}")

    # Export decay plot if we have microtimes
    if channel_data.microtimes is not None and len(channel_data.microtimes) > 0:
        try:
            from full_sms.analysis.histograms import build_decay_histogram

            t_ns, counts = build_decay_histogram(
                channel_data.microtimes.astype(np.float64),
                measurement.channelwidth,
            )

            if len(t_ns) > 0:
                path = export_decay_plot(
                    t_ns=t_ns,
                    counts=counts,
                    output_path=output_dir / f"{prefix}_decay",
                    fit_result=fit_result,
                    fmt=fmt,
                    title=f"Measurement {measurement.id} - Channel {channel} Decay",
                    dpi=dpi,
                )
                exported.append(path)
        except Exception as e:
            logger.warning(f"Failed to export decay plot: {e}")

    # Export BIC plot if we have clustering
    if clustering_result is not None:
        try:
            path = export_bic_plot(
                clustering_result=clustering_result,
                output_path=output_dir / f"{prefix}_bic",
                fmt=fmt,
                title=f"Measurement {measurement.id} - BIC Optimization",
                dpi=dpi,
            )
            exported.append(path)
        except Exception as e:
            logger.warning(f"Failed to export BIC plot: {e}")

    # Export correlation plot if we have g2 data
    if correlation_tau is not None and correlation_g2 is not None:
        try:
            path = export_correlation_plot(
                tau=correlation_tau,
                g2=correlation_g2,
                output_path=output_dir / f"{prefix}_correlation",
                fmt=fmt,
                title=f"Measurement {measurement.id} - g²(τ)",
                dpi=dpi,
            )
            exported.append(path)
        except Exception as e:
            logger.warning(f"Failed to export correlation plot: {e}")

    return exported
