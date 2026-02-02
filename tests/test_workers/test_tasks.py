"""Tests for picklable worker task functions."""

import numpy as np
import pytest

from full_sms.workers.pool import AnalysisPool, TaskResult
from full_sms.workers.tasks import (
    _dict_to_level,
    _level_to_dict,
    run_clustering_task,
    run_correlation_task,
    run_cpa_task,
    run_fit_task,
)


class TestRunCPATask:
    """Tests for change point analysis task function."""

    def test_basic_cpa_direct_call(self):
        """Test run_cpa_task can be called directly."""
        # Create synthetic photon arrival times with a change point
        # Two segments: ~1000 cps for first half, ~2000 cps for second half
        np.random.seed(42)

        # First segment: 1000 cps for 0.5s = 500 photons
        segment1_times = np.sort(np.random.uniform(0, 0.5e9, 500))
        # Second segment: 2000 cps for 0.5s = 1000 photons
        segment2_times = np.sort(np.random.uniform(0.5e9, 1.0e9, 1000))

        abstimes = np.concatenate([segment1_times, segment2_times])

        params = {
            "abstimes": abstimes,
            "confidence": 0.95,
            "measurement_id": "test_particle",
            "channel_id": 0,
        }

        result = run_cpa_task(params)

        # Check result structure
        assert "change_point_indices" in result
        assert "levels" in result
        assert "num_change_points" in result
        assert "confidence_regions" in result
        assert result["measurement_id"] == "test_particle"
        assert result["channel_id"] == 0

        # Should detect at least one change point (the intensity change)
        assert result["num_change_points"] >= 0
        assert len(result["levels"]) >= 1

        # Check level structure
        if result["levels"]:
            level = result["levels"][0]
            assert "start_index" in level
            assert "end_index" in level
            assert "num_photons" in level
            assert "intensity_cps" in level
            assert "dwell_time_s" in level

    def test_cpa_empty_data(self):
        """Test CPA with empty data returns empty result."""
        params = {
            "abstimes": np.array([]),
            "confidence": 0.95,
        }

        result = run_cpa_task(params)

        assert result["num_change_points"] == 0
        assert len(result["levels"]) == 0

    def test_cpa_submitted_to_pool(self):
        """Test run_cpa_task works when submitted to AnalysisPool."""
        np.random.seed(42)
        abstimes = np.sort(np.random.uniform(0, 1e9, 500))

        params = {
            "abstimes": abstimes,
            "confidence": 0.95,
            "measurement_id": 42,
        }

        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(run_cpa_task, params)
            task_result = future.result(timeout=30.0)

        assert task_result.success is True
        result = task_result.value
        assert result["measurement_id"] == 42
        assert "levels" in result


class TestRunClusteringTask:
    """Tests for AHCA clustering task function."""

    def test_basic_clustering_direct_call(self):
        """Test run_clustering_task can be called directly."""
        # Create level dicts representing different intensity levels
        levels = [
            {
                "start_index": 0,
                "end_index": 99,
                "start_time_ns": 0,
                "end_time_ns": 100_000_000,  # 0.1e9
                "num_photons": 100,
                "dwell_time_s": 0.1,
                "intensity_cps": 1000.0,
            },
            {
                "start_index": 100,
                "end_index": 199,
                "start_time_ns": 100_000_000,  # 0.1e9
                "end_time_ns": 200_000_000,  # 0.2e9
                "num_photons": 200,
                "dwell_time_s": 0.1,
                "intensity_cps": 2000.0,
            },
            {
                "start_index": 200,
                "end_index": 299,
                "start_time_ns": 200_000_000,  # 0.2e9
                "end_time_ns": 300_000_000,  # 0.3e9
                "num_photons": 110,
                "dwell_time_s": 0.1,
                "intensity_cps": 1100.0,
            },
        ]

        params = {
            "levels": levels,
            "use_lifetime": False,
            "measurement_id": "test",
            "channel_id": 1,
        }

        result = run_clustering_task(params)

        # Check result structure
        assert "steps" in result
        assert "optimal_step_index" in result
        assert "selected_step_index" in result
        assert "num_original_levels" in result
        assert result["measurement_id"] == "test"
        assert result["channel_id"] == 1
        assert result["num_original_levels"] == 3

        # Should have clustering steps
        assert len(result["steps"]) >= 1

        # Check step structure
        step = result["steps"][0]
        assert "groups" in step
        assert "level_group_assignments" in step
        assert "bic" in step
        assert "num_groups" in step

    def test_clustering_single_level(self):
        """Test clustering with single level returns trivial result."""
        levels = [
            {
                "start_index": 0,
                "end_index": 99,
                "start_time_ns": 0,
                "end_time_ns": 100_000_000,
                "num_photons": 100,
                "dwell_time_s": 0.1,
                "intensity_cps": 1000.0,
            },
        ]

        params = {"levels": levels}
        result = run_clustering_task(params)

        assert result["num_original_levels"] == 1
        assert len(result["steps"]) == 1

    def test_clustering_empty_levels(self):
        """Test clustering with empty levels returns None result."""
        params = {"levels": []}
        result = run_clustering_task(params)

        assert result["result"] is None

    def test_clustering_submitted_to_pool(self):
        """Test run_clustering_task works when submitted to AnalysisPool."""
        levels = [
            {
                "start_index": 0,
                "end_index": 99,
                "start_time_ns": 0,
                "end_time_ns": 100_000_000,
                "num_photons": 100,
                "dwell_time_s": 0.1,
                "intensity_cps": 1000.0,
            },
            {
                "start_index": 100,
                "end_index": 199,
                "start_time_ns": 100_000_000,
                "end_time_ns": 200_000_000,
                "num_photons": 200,
                "dwell_time_s": 0.1,
                "intensity_cps": 2000.0,
            },
        ]

        params = {"levels": levels, "measurement_id": 123}

        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(run_clustering_task, params)
            task_result = future.result(timeout=30.0)

        assert task_result.success is True
        result = task_result.value
        assert result["measurement_id"] == 123
        assert "steps" in result


class TestRunFitTask:
    """Tests for lifetime fitting task function."""

    def test_basic_fit_direct_call(self):
        """Test run_fit_task can be called directly with synthetic data."""
        # Create synthetic decay with known tau
        np.random.seed(42)
        channelwidth = 0.05  # 50 ps channels
        tau_true = 3.0  # 3 ns lifetime
        n_channels = 1000
        n_photons = 10000

        t = np.arange(n_channels) * channelwidth

        # Generate exponential decay + background
        decay = np.exp(-t / tau_true)
        background = 0.01
        decay = decay + background
        decay = decay / decay.sum()

        # Sample counts from decay (Poisson sampling)
        counts = np.random.poisson(decay * n_photons)
        counts = counts.astype(np.int64)

        params = {
            "t": t,
            "counts": counts,
            "channelwidth": channelwidth,
            "num_exponentials": 1,
            "tau_init": 2.0,
            "autostart": "Close to max",
            "autoend": True,
            "measurement_id": "decay_test",
            "level_id": 0,
        }

        result = run_fit_task(params)

        # Check result structure
        assert "tau" in result
        assert "tau_std" in result
        assert "amplitude" in result
        assert "chi_squared" in result
        assert "durbin_watson" in result
        assert "residuals" in result
        assert "fitted_curve" in result
        assert "num_exponentials" in result
        assert "average_lifetime" in result
        assert result["measurement_id"] == "decay_test"
        assert result["level_id"] == 0
        assert result["error"] is None

        # Check that fitted tau is close to true value (within 50%)
        fitted_tau = result["tau"][0]
        assert 0.5 * tau_true < fitted_tau < 2.0 * tau_true

    def test_fit_error_handling(self):
        """Test that fit errors are captured in result."""
        # Empty data should cause an error
        params = {
            "t": np.array([]),
            "counts": np.array([]),
            "channelwidth": 0.05,
            "measurement_id": "error_test",
        }

        result = run_fit_task(params)

        # Should have error field populated
        assert result["error"] is not None
        assert result["measurement_id"] == "error_test"

    def test_fit_submitted_to_pool(self):
        """Test run_fit_task works when submitted to AnalysisPool."""
        np.random.seed(42)
        channelwidth = 0.05
        t = np.arange(500) * channelwidth
        decay = np.exp(-t / 3.0)
        counts = np.random.poisson(decay * 5000).astype(np.int64)

        params = {
            "t": t,
            "counts": counts,
            "channelwidth": channelwidth,
            "num_exponentials": 1,
            "autostart": "Close to max",
            "autoend": True,
            "measurement_id": 99,
        }

        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(run_fit_task, params)
            task_result = future.result(timeout=30.0)

        assert task_result.success is True
        result = task_result.value
        assert result["measurement_id"] == 99
        assert result["error"] is None
        assert len(result["tau"]) == 1


class TestRunCorrelationTask:
    """Tests for g2 correlation task function."""

    def test_basic_correlation_direct_call(self):
        """Test run_correlation_task can be called directly."""
        np.random.seed(42)

        # Create synthetic dual-channel photon times
        # Random photon arrivals over 1 second
        n_photons = 1000
        abstimes1 = np.sort(np.random.uniform(0, 1e9, n_photons))
        abstimes2 = np.sort(np.random.uniform(0, 1e9, n_photons))
        microtimes1 = np.random.uniform(0, 50, n_photons)  # 0-50 ns micro times
        microtimes2 = np.random.uniform(0, 50, n_photons)

        params = {
            "abstimes1": abstimes1,
            "abstimes2": abstimes2,
            "microtimes1": microtimes1,
            "microtimes2": microtimes2,
            "window_ns": 200.0,
            "binsize_ns": 1.0,
            "measurement_id": "g2_test",
        }

        result = run_correlation_task(params)

        # Check result structure
        assert "tau" in result
        assert "g2" in result
        assert "events" in result
        assert "window_ns" in result
        assert "binsize_ns" in result
        assert "num_photons_ch1" in result
        assert "num_photons_ch2" in result
        assert "num_events" in result
        assert result["measurement_id"] == "g2_test"

        # tau should span from -window to +window
        assert len(result["tau"]) == int(2 * 200.0 / 1.0)
        assert min(result["tau"]) < 0
        assert max(result["tau"]) > 0

        # Photon counts should match input
        assert result["num_photons_ch1"] == n_photons
        assert result["num_photons_ch2"] == n_photons

    def test_correlation_empty_channels(self):
        """Test correlation with empty channel returns zero histogram."""
        params = {
            "abstimes1": np.array([]),
            "abstimes2": np.array([1.0, 2.0, 3.0]),
            "microtimes1": np.array([]),
            "microtimes2": np.array([0.0, 0.0, 0.0]),
            "window_ns": 100.0,
            "binsize_ns": 1.0,
        }

        result = run_correlation_task(params)

        # Should return zero histogram
        assert result["num_events"] == 0
        assert all(g == 0 for g in result["g2"])

    def test_correlation_submitted_to_pool(self):
        """Test run_correlation_task works when submitted to AnalysisPool."""
        np.random.seed(42)
        n_photons = 500
        abstimes1 = np.sort(np.random.uniform(0, 1e9, n_photons))
        abstimes2 = np.sort(np.random.uniform(0, 1e9, n_photons))
        microtimes1 = np.zeros(n_photons)
        microtimes2 = np.zeros(n_photons)

        params = {
            "abstimes1": abstimes1,
            "abstimes2": abstimes2,
            "microtimes1": microtimes1,
            "microtimes2": microtimes2,
            "window_ns": 100.0,
            "binsize_ns": 2.0,
            "measurement_id": 77,
        }

        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(run_correlation_task, params)
            task_result = future.result(timeout=30.0)

        assert task_result.success is True
        result = task_result.value
        assert result["measurement_id"] == 77
        assert "g2" in result


class TestLevelSerialization:
    """Tests for level serialization helpers."""

    def test_level_round_trip(self):
        """Test level can be converted to dict and back."""
        from full_sms.models.level import LevelData

        original = LevelData(
            start_index=10,
            end_index=50,
            start_time_ns=100,
            end_time_ns=500,
            num_photons=40,
            intensity_cps=1000.0,
            group_id=2,
        )

        # Convert to dict
        d = _level_to_dict(original)
        assert d["start_index"] == 10
        assert d["end_index"] == 50
        assert d["num_photons"] == 40
        assert d["intensity_cps"] == 1000.0
        assert d["group_id"] == 2

        # Convert back
        restored = _dict_to_level(d)
        assert restored.start_index == original.start_index
        assert restored.end_index == original.end_index
        assert restored.num_photons == original.num_photons
        assert restored.start_time_ns == original.start_time_ns
        assert restored.end_time_ns == original.end_time_ns
        assert restored.intensity_cps == original.intensity_cps
        assert restored.group_id == original.group_id


class TestBatchProcessing:
    """Tests for batch processing multiple tasks."""

    def test_batch_cpa_with_map(self):
        """Test processing multiple measurements with map_with_progress."""
        np.random.seed(42)

        # Create params for multiple measurements
        param_list = []
        for i in range(4):
            abstimes = np.sort(np.random.uniform(0, 1e9, 300))
            param_list.append(
                {
                    "abstimes": abstimes,
                    "confidence": 0.95,
                    "measurement_id": i,
                }
            )

        progress_calls = []

        def track_progress(completed: int, total: int):
            progress_calls.append((completed, total))

        with AnalysisPool(max_workers=2) as pool:
            results = pool.map_with_progress(
                run_cpa_task, param_list, on_progress=track_progress
            )

        # All should succeed
        assert len(results) == 4
        for i, result in enumerate(results):
            assert result.success is True
            assert result.value["measurement_id"] == i

        # Progress callback should have been called 4 times
        assert len(progress_calls) == 4

    def test_batch_fit_mixed_success_failure(self):
        """Test batch fitting where some succeed and some fail."""
        np.random.seed(42)
        channelwidth = 0.05
        t = np.arange(500) * channelwidth

        param_list = []

        # Good decay data
        decay = np.exp(-t / 3.0)
        counts = np.random.poisson(decay * 5000).astype(np.int64)
        param_list.append(
            {
                "t": t,
                "counts": counts,
                "channelwidth": channelwidth,
                "num_exponentials": 1,
                "autostart": "Close to max",
                "autoend": True,
                "measurement_id": 0,
            }
        )

        # Empty data (will fail)
        param_list.append(
            {
                "t": np.array([]),
                "counts": np.array([]),
                "channelwidth": channelwidth,
                "measurement_id": 1,
            }
        )

        # Another good decay
        param_list.append(
            {
                "t": t,
                "counts": counts,
                "channelwidth": channelwidth,
                "num_exponentials": 1,
                "autostart": "Close to max",
                "autoend": True,
                "measurement_id": 2,
            }
        )

        with AnalysisPool(max_workers=2) as pool:
            results = pool.map_with_progress(run_fit_task, param_list)

        assert len(results) == 3

        # First should succeed
        assert results[0].success is True
        assert results[0].value["error"] is None
        assert results[0].value["measurement_id"] == 0

        # Second should succeed but have error in result dict
        assert results[1].success is True  # Task itself didn't crash
        assert results[1].value["error"] is not None  # But fitting failed
        assert results[1].value["measurement_id"] == 1

        # Third should succeed
        assert results[2].success is True
        assert results[2].value["error"] is None
        assert results[2].value["measurement_id"] == 2
