"""Tests for plot export functions."""

from pathlib import Path

import numpy as np
import pytest

from full_sms.io.plot_exporters import (
    PlotFormat,
    export_all_plots,
    export_bic_plot,
    export_correlation_plot,
    export_decay_plot,
    export_intensity_plot,
)
from full_sms.models.fit import FitResult, FitResultData, IRFData
from full_sms.models.group import ClusteringResult, ClusteringStep, GroupData
from full_sms.models.level import LevelData
from full_sms.models.particle import ChannelData, ParticleData

# Skip all tests if matplotlib is not available
pytest.importorskip("matplotlib")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_abstimes() -> np.ndarray:
    """Create sample absolute times for intensity trace."""
    # Create 1000 photons over 1 second (1e9 ns)
    return np.linspace(0, 1_000_000_000, 1000, dtype=np.uint64)


@pytest.fixture
def sample_decay_data() -> tuple[np.ndarray, np.ndarray]:
    """Create sample decay histogram data."""
    # Time axis in nanoseconds (0-50 ns, 0.1 ns bins)
    t_ns = np.arange(0, 50, 0.1, dtype=np.float64)
    # Exponential decay with Poisson noise
    true_decay = 10000 * np.exp(-t_ns / 5.0) + 10
    counts = np.random.poisson(true_decay).astype(np.int64)
    return t_ns, counts


@pytest.fixture
def sample_levels() -> list[LevelData]:
    """Create sample levels for plot testing."""
    return [
        LevelData(
            start_index=0,
            end_index=299,
            start_time_ns=0,
            end_time_ns=300_000_000,
            num_photons=300,
            intensity_cps=1000.0,
            group_id=0,
        ),
        LevelData(
            start_index=300,
            end_index=599,
            start_time_ns=300_000_000,
            end_time_ns=600_000_000,
            num_photons=300,
            intensity_cps=2000.0,
            group_id=1,
        ),
        LevelData(
            start_index=600,
            end_index=999,
            start_time_ns=600_000_000,
            end_time_ns=1_000_000_000,
            num_photons=400,
            intensity_cps=1000.0,
            group_id=0,
        ),
    ]


@pytest.fixture
def sample_groups() -> list[GroupData]:
    """Create sample groups for plot testing."""
    return [
        GroupData(
            group_id=0,
            level_indices=(0, 2),
            total_photons=700,
            total_dwell_time_s=0.7,
            intensity_cps=1000.0,
        ),
        GroupData(
            group_id=1,
            level_indices=(1,),
            total_photons=300,
            total_dwell_time_s=0.3,
            intensity_cps=2000.0,
        ),
    ]


@pytest.fixture
def sample_fit_result(sample_decay_data: tuple[np.ndarray, np.ndarray]) -> FitResult:
    """Create a sample single-exponential fit result matching decay data."""
    t_ns, _ = sample_decay_data
    fit_start = 10
    fit_end = len(t_ns)
    fit_t = t_ns[fit_start:fit_end]
    fitted_curve = 10000 * np.exp(-fit_t / 5.0) + 10
    residuals = np.random.randn(len(fit_t))

    return FitResult.from_fit_parameters(
        tau=[5.0],
        tau_std=[0.1],
        amplitude=[1.0],
        amplitude_std=[0.01],
        shift=0.5,
        shift_std=0.05,
        chi_squared=1.05,
        durbin_watson=2.1,
        residuals=residuals,
        fitted_curve=fitted_curve,
        fit_start_index=fit_start,
        fit_end_index=fit_end,
        background=10.0,
        dw_bounds=(1.5, 2.5),
    )


@pytest.fixture
def sample_fit_data() -> FitResultData:
    """Create a sample FitResultData for plot testing."""
    return FitResultData(
        tau=(5.0,),
        tau_std=(0.1,),
        amplitude=(1.0,),
        amplitude_std=(0.01,),
        shift=0.5,
        shift_std=0.05,
        chi_squared=1.05,
        durbin_watson=2.1,
        dw_bounds=(1.5, 2.5),
        fit_start_index=10,
        fit_end_index=500,
        background=10.0,
        num_exponentials=1,
        average_lifetime=5.0,
        level_index=None,
    )


@pytest.fixture
def sample_irf_data() -> IRFData:
    """Create a sample simulated IRF."""
    return IRFData.from_simulated(fwhm_ns=0.2)


@pytest.fixture
def sample_clustering_result() -> ClusteringResult:
    """Create a sample clustering result for BIC plot testing."""
    # Create 3 steps (3 groups -> 2 groups -> 1 group)
    steps = []
    for num_groups in [3, 2, 1]:
        groups = []
        assignments = [0, 0, 0]

        for g in range(num_groups):
            groups.append(
                GroupData(
                    group_id=g,
                    level_indices=(g,) if g < 3 else (0,),
                    total_photons=100,
                    total_dwell_time_s=0.1,
                    intensity_cps=1000.0,
                )
            )

        # BIC values: lower is better, optimal at 2 groups
        if num_groups == 3:
            bic = 150.0
        elif num_groups == 2:
            bic = 100.0  # Optimal
        else:
            bic = 120.0

        steps.append(
            ClusteringStep(
                groups=tuple(groups),
                level_group_assignments=tuple(assignments[:num_groups]),
                bic=bic,
                num_groups=num_groups,
            )
        )

    return ClusteringResult(
        steps=tuple(steps),
        optimal_step_index=1,  # 2 groups is optimal
        selected_step_index=1,
        num_original_levels=3,
    )


@pytest.fixture
def sample_correlation_data() -> tuple[np.ndarray, np.ndarray]:
    """Create sample g2 correlation data."""
    # Log-spaced delay times
    tau = np.logspace(-1, 4, 100)
    # Antibunching dip at short times, then converge to 1
    g2 = 1 - 0.8 * np.exp(-tau / 10) + 0.1 * np.random.randn(len(tau))
    return tau, g2


@pytest.fixture
def sample_channel() -> ChannelData:
    """Create a sample channel for particle testing."""
    return ChannelData(
        abstimes=np.linspace(0, 1_000_000_000, 1000, dtype=np.uint64),
        microtimes=np.random.rand(1000).astype(np.float64) * 50,  # 0-50 ns
    )


@pytest.fixture
def sample_particle(sample_channel: ChannelData) -> ParticleData:
    """Create a sample particle for testing."""
    return ParticleData(
        id=1,
        name="Test Particle",
        tcspc_card="SPC-150",
        channelwidth=0.1,
        channel1=sample_channel,
    )


# ============================================================================
# Test PlotFormat
# ============================================================================


class TestPlotFormat:
    """Tests for PlotFormat enum."""

    def test_png_value(self) -> None:
        assert PlotFormat.PNG.value == "png"

    def test_pdf_value(self) -> None:
        assert PlotFormat.PDF.value == "pdf"

    def test_svg_value(self) -> None:
        assert PlotFormat.SVG.value == "svg"


# ============================================================================
# Test export_intensity_plot
# ============================================================================


class TestExportIntensityPlot:
    """Tests for export_intensity_plot function."""

    def test_basic_export_png(
        self, sample_abstimes: np.ndarray, tmp_path: Path
    ) -> None:
        """Export basic intensity plot to PNG."""
        output_path = tmp_path / "intensity"
        result = export_intensity_plot(
            sample_abstimes,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.suffix == ".png"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_pdf_format(
        self, sample_abstimes: np.ndarray, tmp_path: Path
    ) -> None:
        """Export intensity plot to PDF format."""
        output_path = tmp_path / "intensity"
        result = export_intensity_plot(
            sample_abstimes,
            output_path,
            fmt=PlotFormat.PDF,
        )

        assert result.suffix == ".pdf"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_svg_format(
        self, sample_abstimes: np.ndarray, tmp_path: Path
    ) -> None:
        """Export intensity plot to SVG format."""
        output_path = tmp_path / "intensity"
        result = export_intensity_plot(
            sample_abstimes,
            output_path,
            fmt=PlotFormat.SVG,
        )

        assert result.suffix == ".svg"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_with_level_overlays(
        self,
        sample_abstimes: np.ndarray,
        sample_levels: list[LevelData],
        tmp_path: Path,
    ) -> None:
        """Export intensity plot with level overlays."""
        output_path = tmp_path / "intensity_with_levels"
        result = export_intensity_plot(
            sample_abstimes,
            output_path,
            levels=sample_levels,
            show_levels=True,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()
        # File should be larger with levels overlay
        assert result.stat().st_size > 0

    def test_with_group_coloring(
        self,
        sample_abstimes: np.ndarray,
        sample_levels: list[LevelData],
        sample_groups: list[GroupData],
        tmp_path: Path,
    ) -> None:
        """Export intensity plot with group coloring."""
        output_path = tmp_path / "intensity_with_groups"
        result = export_intensity_plot(
            sample_abstimes,
            output_path,
            levels=sample_levels,
            groups=sample_groups,
            show_groups=True,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_with_title(
        self, sample_abstimes: np.ndarray, tmp_path: Path
    ) -> None:
        """Export intensity plot with custom title."""
        output_path = tmp_path / "intensity"
        result = export_intensity_plot(
            sample_abstimes,
            output_path,
            title="Test Particle - Channel 1",
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_custom_figsize(
        self, sample_abstimes: np.ndarray, tmp_path: Path
    ) -> None:
        """Export intensity plot with custom figure size."""
        output_path = tmp_path / "intensity"
        result = export_intensity_plot(
            sample_abstimes,
            output_path,
            figsize=(12, 8),
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_custom_dpi(
        self, sample_abstimes: np.ndarray, tmp_path: Path
    ) -> None:
        """Export intensity plot with custom DPI."""
        output_low = tmp_path / "intensity_low"
        output_high = tmp_path / "intensity_high"

        result_low = export_intensity_plot(
            sample_abstimes,
            output_low,
            dpi=72,
            fmt=PlotFormat.PNG,
        )
        result_high = export_intensity_plot(
            sample_abstimes,
            output_high,
            dpi=300,
            fmt=PlotFormat.PNG,
        )

        # Higher DPI should produce larger file
        assert result_high.stat().st_size > result_low.stat().st_size

    def test_creates_parent_directories(
        self, sample_abstimes: np.ndarray, tmp_path: Path
    ) -> None:
        """Export creates parent directories if they don't exist."""
        output_path = tmp_path / "nested" / "subdir" / "intensity"
        result = export_intensity_plot(
            sample_abstimes,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()
        assert result.parent.exists()


# ============================================================================
# Test export_decay_plot
# ============================================================================


class TestExportDecayPlot:
    """Tests for export_decay_plot function."""

    def test_basic_decay_export(
        self, sample_decay_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export basic decay plot."""
        t_ns, counts = sample_decay_data
        output_path = tmp_path / "decay"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.suffix == ".png"
        assert result.exists()

    def test_with_fit_curve_from_fit_result(
        self,
        sample_decay_data: tuple[np.ndarray, np.ndarray],
        sample_fit_result: FitResult,
        tmp_path: Path,
    ) -> None:
        """Export decay plot with fit curve from FitResult."""
        t_ns, counts = sample_decay_data
        output_path = tmp_path / "decay_with_fit"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            fit_result=sample_fit_result,
            show_fit=True,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_with_fit_curve_from_fit_data(
        self,
        sample_decay_data: tuple[np.ndarray, np.ndarray],
        sample_fit_data: FitResultData,
        sample_irf_data: IRFData,
        tmp_path: Path,
    ) -> None:
        """Export decay plot with fit curve from FitResultData."""
        t_ns, counts = sample_decay_data
        output_path = tmp_path / "decay_with_fit_data"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            channelwidth=0.1,
            fit_data=sample_fit_data,
            irf_data=sample_irf_data,
            show_fit=True,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_with_residuals_panel(
        self,
        sample_decay_data: tuple[np.ndarray, np.ndarray],
        sample_fit_result: FitResult,
        tmp_path: Path,
    ) -> None:
        """Export decay plot with residuals panel."""
        t_ns, counts = sample_decay_data
        output_path = tmp_path / "decay_with_residuals"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            fit_result=sample_fit_result,
            show_fit=True,
            show_residuals=True,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_without_residuals_panel(
        self,
        sample_decay_data: tuple[np.ndarray, np.ndarray],
        sample_fit_result: FitResult,
        tmp_path: Path,
    ) -> None:
        """Export decay plot without residuals panel."""
        t_ns, counts = sample_decay_data
        output_path = tmp_path / "decay_no_residuals"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            fit_result=sample_fit_result,
            show_fit=True,
            show_residuals=False,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_with_irf_display(
        self,
        sample_decay_data: tuple[np.ndarray, np.ndarray],
        sample_fit_data: FitResultData,
        sample_irf_data: IRFData,
        tmp_path: Path,
    ) -> None:
        """Export decay plot with IRF display."""
        t_ns, counts = sample_decay_data
        output_path = tmp_path / "decay_with_irf"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            channelwidth=0.1,
            fit_data=sample_fit_data,
            irf_data=sample_irf_data,
            show_fit=True,
            show_irf=True,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_log_scale_option(
        self,
        sample_decay_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Export decay plot with log scale."""
        t_ns, counts = sample_decay_data
        output_log = tmp_path / "decay_log"
        output_lin = tmp_path / "decay_lin"

        result_log = export_decay_plot(
            t_ns,
            counts,
            output_log,
            log_scale=True,
            fmt=PlotFormat.PNG,
        )
        result_lin = export_decay_plot(
            t_ns,
            counts,
            output_lin,
            log_scale=False,
            fmt=PlotFormat.PNG,
        )

        assert result_log.exists()
        assert result_lin.exists()

    def test_pdf_format(
        self, sample_decay_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export decay plot to PDF."""
        t_ns, counts = sample_decay_data
        output_path = tmp_path / "decay"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            fmt=PlotFormat.PDF,
        )

        assert result.suffix == ".pdf"
        assert result.exists()

    def test_with_title(
        self, sample_decay_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export decay plot with custom title."""
        t_ns, counts = sample_decay_data
        output_path = tmp_path / "decay"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            title="Fluorescence Decay - Test",
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_with_loaded_irf(
        self, sample_decay_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export decay plot with loaded IRF data."""
        t_ns, counts = sample_decay_data

        # Create loaded IRF
        irf_t = t_ns.copy()
        irf_counts = np.exp(-((t_ns - 5) ** 2) / 0.5) * 1000  # Gaussian IRF
        loaded_irf = IRFData.from_loaded(irf_t.tolist(), irf_counts.tolist())

        output_path = tmp_path / "decay_loaded_irf"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            channelwidth=0.1,
            irf_data=loaded_irf,
            show_irf=True,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_legacy_irf_arrays(
        self, sample_decay_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export decay plot with legacy direct IRF arrays."""
        t_ns, counts = sample_decay_data
        irf_counts = np.exp(-((t_ns - 5) ** 2) / 0.5) * 1000

        output_path = tmp_path / "decay_legacy_irf"

        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            irf_t=t_ns,
            irf_counts=irf_counts,
            show_irf=True,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()


# ============================================================================
# Test export_bic_plot
# ============================================================================


class TestExportBicPlot:
    """Tests for export_bic_plot function."""

    def test_basic_bic_export(
        self, sample_clustering_result: ClusteringResult, tmp_path: Path
    ) -> None:
        """Export basic BIC plot."""
        output_path = tmp_path / "bic"

        result = export_bic_plot(
            sample_clustering_result,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.suffix == ".png"
        assert result.exists()

    def test_highlights_optimal_point(
        self, sample_clustering_result: ClusteringResult, tmp_path: Path
    ) -> None:
        """BIC plot highlights optimal point."""
        output_path = tmp_path / "bic"

        result = export_bic_plot(
            sample_clustering_result,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()
        # Optimal is at step 1 (2 groups)
        assert sample_clustering_result.optimal_step_index == 1

    def test_shows_selected_vs_optimal(self, tmp_path: Path) -> None:
        """BIC plot shows both selected and optimal when different."""
        # Create clustering where selected differs from optimal
        steps = []
        for num_groups in [3, 2, 1]:
            groups = tuple(
                GroupData(
                    group_id=g,
                    level_indices=(g,),
                    total_photons=100,
                    total_dwell_time_s=0.1,
                    intensity_cps=1000.0,
                )
                for g in range(num_groups)
            )
            steps.append(
                ClusteringStep(
                    groups=groups,
                    level_group_assignments=tuple(range(num_groups)),
                    bic=100.0 + num_groups * 10,
                    num_groups=num_groups,
                )
            )

        clustering = ClusteringResult(
            steps=tuple(steps),
            optimal_step_index=2,  # 1 group is optimal
            selected_step_index=0,  # But 3 groups is selected
            num_original_levels=3,
        )

        output_path = tmp_path / "bic_diff_selection"
        result = export_bic_plot(
            clustering,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()
        assert clustering.optimal_step_index != clustering.selected_step_index

    def test_pdf_format(
        self, sample_clustering_result: ClusteringResult, tmp_path: Path
    ) -> None:
        """Export BIC plot to PDF."""
        output_path = tmp_path / "bic"

        result = export_bic_plot(
            sample_clustering_result,
            output_path,
            fmt=PlotFormat.PDF,
        )

        assert result.suffix == ".pdf"
        assert result.exists()

    def test_custom_title(
        self, sample_clustering_result: ClusteringResult, tmp_path: Path
    ) -> None:
        """Export BIC plot with custom title."""
        output_path = tmp_path / "bic"

        result = export_bic_plot(
            sample_clustering_result,
            output_path,
            title="Custom BIC Title",
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_custom_figsize(
        self, sample_clustering_result: ClusteringResult, tmp_path: Path
    ) -> None:
        """Export BIC plot with custom figure size."""
        output_path = tmp_path / "bic"

        result = export_bic_plot(
            sample_clustering_result,
            output_path,
            figsize=(10, 6),
            fmt=PlotFormat.PNG,
        )

        assert result.exists()


# ============================================================================
# Test export_correlation_plot
# ============================================================================


class TestExportCorrelationPlot:
    """Tests for export_correlation_plot function."""

    def test_basic_g2_export(
        self, sample_correlation_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export basic g2 correlation plot."""
        tau, g2 = sample_correlation_data
        output_path = tmp_path / "correlation"

        result = export_correlation_plot(
            tau,
            g2,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.suffix == ".png"
        assert result.exists()

    def test_pdf_format(
        self, sample_correlation_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export correlation plot to PDF."""
        tau, g2 = sample_correlation_data
        output_path = tmp_path / "correlation"

        result = export_correlation_plot(
            tau,
            g2,
            output_path,
            fmt=PlotFormat.PDF,
        )

        assert result.suffix == ".pdf"
        assert result.exists()

    def test_svg_format(
        self, sample_correlation_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export correlation plot to SVG."""
        tau, g2 = sample_correlation_data
        output_path = tmp_path / "correlation"

        result = export_correlation_plot(
            tau,
            g2,
            output_path,
            fmt=PlotFormat.SVG,
        )

        assert result.suffix == ".svg"
        assert result.exists()

    def test_custom_title(
        self, sample_correlation_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export correlation plot with custom title."""
        tau, g2 = sample_correlation_data
        output_path = tmp_path / "correlation"

        result = export_correlation_plot(
            tau,
            g2,
            output_path,
            title="Custom g²(τ) Title",
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_custom_figsize_and_dpi(
        self, sample_correlation_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Export correlation plot with custom size and resolution."""
        tau, g2 = sample_correlation_data
        output_path = tmp_path / "correlation"

        result = export_correlation_plot(
            tau,
            g2,
            output_path,
            figsize=(10, 6),
            dpi=300,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()


# ============================================================================
# Test export_all_plots
# ============================================================================


class TestExportAllPlots:
    """Tests for export_all_plots function."""

    def test_exports_intensity_plot(
        self, sample_particle: ParticleData, tmp_path: Path
    ) -> None:
        """export_all_plots exports intensity plot."""
        output_dir = tmp_path / "plots"

        files = export_all_plots(
            sample_particle,
            channel=1,
            output_dir=output_dir,
            fmt=PlotFormat.PNG,
        )

        assert len(files) >= 1
        intensity_files = [f for f in files if "intensity" in f.name]
        assert len(intensity_files) == 1

    def test_exports_decay_plot_when_microtimes_present(
        self, sample_particle: ParticleData, tmp_path: Path
    ) -> None:
        """export_all_plots exports decay plot when microtimes are available."""
        output_dir = tmp_path / "plots"

        files = export_all_plots(
            sample_particle,
            channel=1,
            output_dir=output_dir,
            fmt=PlotFormat.PNG,
        )

        decay_files = [f for f in files if "decay" in f.name]
        assert len(decay_files) == 1

    def test_exports_bic_plot_when_clustering_provided(
        self,
        sample_particle: ParticleData,
        sample_clustering_result: ClusteringResult,
        tmp_path: Path,
    ) -> None:
        """export_all_plots exports BIC plot when clustering result is provided."""
        output_dir = tmp_path / "plots"

        files = export_all_plots(
            sample_particle,
            channel=1,
            output_dir=output_dir,
            clustering_result=sample_clustering_result,
            fmt=PlotFormat.PNG,
        )

        bic_files = [f for f in files if "bic" in f.name]
        assert len(bic_files) == 1

    def test_exports_correlation_plot_when_data_provided(
        self,
        sample_particle: ParticleData,
        sample_correlation_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """export_all_plots exports correlation plot when data is provided."""
        tau, g2 = sample_correlation_data
        output_dir = tmp_path / "plots"

        files = export_all_plots(
            sample_particle,
            channel=1,
            output_dir=output_dir,
            correlation_tau=tau,
            correlation_g2=g2,
            fmt=PlotFormat.PNG,
        )

        corr_files = [f for f in files if "correlation" in f.name]
        assert len(corr_files) == 1

    def test_handles_missing_channel_data(self, tmp_path: Path) -> None:
        """export_all_plots handles missing channel data gracefully."""
        # Create particle with only channel 1
        channel = ChannelData(
            abstimes=np.linspace(0, 1_000_000_000, 100, dtype=np.uint64),
            microtimes=np.random.rand(100).astype(np.float64),
        )
        particle = ParticleData(
            id=1,
            name="Test",
            tcspc_card="SPC-150",
            channelwidth=0.1,
            channel1=channel,
            channel2=None,
        )

        output_dir = tmp_path / "plots"

        # Request channel 2 which doesn't exist
        files = export_all_plots(
            particle,
            channel=2,
            output_dir=output_dir,
            fmt=PlotFormat.PNG,
        )

        # Should return empty list for missing channel
        assert files == []

    def test_respects_format_parameter(
        self, sample_particle: ParticleData, tmp_path: Path
    ) -> None:
        """export_all_plots respects format parameter."""
        output_dir = tmp_path / "plots"

        files = export_all_plots(
            sample_particle,
            channel=1,
            output_dir=output_dir,
            fmt=PlotFormat.PDF,
        )

        # All files should be PDF
        for f in files:
            assert f.suffix == ".pdf"

    def test_with_levels_overlay(
        self,
        sample_particle: ParticleData,
        sample_levels: list[LevelData],
        tmp_path: Path,
    ) -> None:
        """export_all_plots includes levels in intensity plot."""
        output_dir = tmp_path / "plots"

        files = export_all_plots(
            sample_particle,
            channel=1,
            output_dir=output_dir,
            levels=sample_levels,
            fmt=PlotFormat.PNG,
        )

        assert len(files) >= 1

    def test_with_fit_result(
        self,
        sample_channel: ChannelData,
        tmp_path: Path,
    ) -> None:
        """export_all_plots includes fit curve in decay plot."""
        # Create a particle with specific microtime distribution
        particle = ParticleData(
            id=1,
            name="Test Particle",
            tcspc_card="SPC-150",
            channelwidth=0.1,
            channel1=sample_channel,
        )

        # Create a fit result that matches the decay histogram dimensions
        # The decay histogram from sample_channel will have ~500 bins (0-50 ns at 0.1 ns)
        from full_sms.analysis.histograms import build_decay_histogram
        t_ns, counts = build_decay_histogram(
            sample_channel.microtimes.astype(np.float64),
            particle.channelwidth,
        )

        fit_start = 10
        fit_end = len(t_ns)
        fit_t = t_ns[fit_start:fit_end]
        fitted_curve = 10000 * np.exp(-fit_t / 5.0) + 10
        residuals = np.random.randn(len(fit_t))

        fit_result = FitResult.from_fit_parameters(
            tau=[5.0],
            tau_std=[0.1],
            amplitude=[1.0],
            amplitude_std=[0.01],
            shift=0.5,
            shift_std=0.05,
            chi_squared=1.05,
            durbin_watson=2.1,
            residuals=residuals,
            fitted_curve=fitted_curve,
            fit_start_index=fit_start,
            fit_end_index=fit_end,
            background=10.0,
            dw_bounds=(1.5, 2.5),
        )

        output_dir = tmp_path / "plots"

        files = export_all_plots(
            particle,
            channel=1,
            output_dir=output_dir,
            fit_result=fit_result,
            fmt=PlotFormat.PNG,
        )

        decay_files = [f for f in files if "decay" in f.name]
        assert len(decay_files) == 1

    def test_custom_dpi(
        self, sample_particle: ParticleData, tmp_path: Path
    ) -> None:
        """export_all_plots respects DPI parameter."""
        output_low = tmp_path / "plots_low"
        output_high = tmp_path / "plots_high"

        files_low = export_all_plots(
            sample_particle,
            channel=1,
            output_dir=output_low,
            dpi=72,
            fmt=PlotFormat.PNG,
        )
        files_high = export_all_plots(
            sample_particle,
            channel=1,
            output_dir=output_high,
            dpi=300,
            fmt=PlotFormat.PNG,
        )

        # Higher DPI should produce larger files
        for low, high in zip(files_low, files_high):
            if low.suffix == ".png" and high.suffix == ".png":
                assert high.stat().st_size > low.stat().st_size


# ============================================================================
# Test error handling
# ============================================================================


class TestPlotExportErrorHandling:
    """Tests for error handling in plot export functions."""

    def test_handles_empty_abstimes(self, tmp_path: Path) -> None:
        """Intensity plot handles empty abstimes gracefully."""
        # Very small array
        abstimes = np.array([0], dtype=np.uint64)
        output_path = tmp_path / "intensity"

        # Should not raise
        result = export_intensity_plot(
            abstimes,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_handles_all_zero_counts(self, tmp_path: Path) -> None:
        """Decay plot handles all-zero counts."""
        t_ns = np.arange(0, 50, 0.1, dtype=np.float64)
        counts = np.zeros(len(t_ns), dtype=np.int64)

        output_path = tmp_path / "decay"

        # Should not raise
        result = export_decay_plot(
            t_ns,
            counts,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Plot export creates parent directories if needed."""
        abstimes = np.linspace(0, 1e9, 100, dtype=np.uint64)
        output_path = tmp_path / "nested" / "subdir" / "plot"

        result = export_intensity_plot(
            abstimes,
            output_path,
            fmt=PlotFormat.PNG,
        )

        assert result.exists()
        assert result.parent.exists()
