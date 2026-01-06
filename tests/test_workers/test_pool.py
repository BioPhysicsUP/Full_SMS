"""Tests for AnalysisPool parallel processing."""

import time
from concurrent.futures import Future

import pytest

from full_sms.workers.pool import AnalysisPool, TaskResult


# Test functions must be module-level for pickling
def _add(a: int, b: int) -> int:
    """Simple add function for testing."""
    return a + b


def _slow_square(x: int) -> int:
    """Slow square function for testing parallelism."""
    time.sleep(0.1)
    return x * x


def _raise_error(x: int) -> int:
    """Function that raises an error."""
    raise ValueError(f"Error for {x}")


def _process_item(item: dict) -> dict:
    """Process a dictionary item."""
    return {"id": item["id"], "result": item["value"] * 2}


def _maybe_fail(x: int) -> int:
    """Function that fails on even numbers."""
    if x % 2 == 0:
        raise ValueError(f"Even number: {x}")
    return x * 2


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_ok_creates_success_result(self):
        result = TaskResult.ok(42)
        assert result.success is True
        assert result.value == 42
        assert result.error is None

    def test_err_creates_failure_result(self):
        error = ValueError("test error")
        result = TaskResult.err(error)
        assert result.success is False
        assert result.value is None
        assert result.error is error

    def test_generic_typing(self):
        result: TaskResult[str] = TaskResult.ok("hello")
        assert result.value == "hello"


class TestAnalysisPoolBasic:
    """Basic tests for AnalysisPool."""

    def test_init_default_workers(self):
        pool = AnalysisPool()
        assert pool.max_workers >= 1
        pool.shutdown()

    def test_init_custom_workers(self):
        pool = AnalysisPool(max_workers=2)
        assert pool.max_workers == 2
        pool.shutdown()

    def test_context_manager(self):
        with AnalysisPool(max_workers=1) as pool:
            assert pool._executor is None  # Lazy init
            future = pool.submit(_add, 1, 2)
            result = future.result()
            assert result.success is True
            assert result.value == 3
        # After context exit, executor should be None
        assert pool._executor is None


class TestAnalysisPoolSubmit:
    """Tests for submit() method."""

    def test_submit_returns_future(self):
        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(_add, 1, 2)
            assert isinstance(future, Future)

    def test_submit_simple_function(self):
        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(_add, 10, 20)
            result = future.result(timeout=5.0)
            assert result.success is True
            assert result.value == 30

    def test_submit_with_kwargs(self):
        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(_add, a=5, b=7)
            result = future.result(timeout=5.0)
            assert result.success is True
            assert result.value == 12

    def test_submit_error_captured(self):
        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(_raise_error, 42)
            result = future.result(timeout=5.0)
            assert result.success is False
            assert isinstance(result.error, ValueError)
            assert "42" in str(result.error)


class TestAnalysisPoolMapWithProgress:
    """Tests for map_with_progress() method."""

    def test_map_empty_list(self):
        with AnalysisPool(max_workers=1) as pool:
            results = pool.map_with_progress(_slow_square, [])
            assert results == []

    def test_map_single_item(self):
        with AnalysisPool(max_workers=1) as pool:
            results = pool.map_with_progress(_slow_square, [5])
            assert len(results) == 1
            assert results[0].success is True
            assert results[0].value == 25

    def test_map_multiple_items(self):
        with AnalysisPool(max_workers=2) as pool:
            items = [1, 2, 3, 4, 5]
            results = pool.map_with_progress(_slow_square, items)
            assert len(results) == 5
            # Results should be in order
            for i, result in enumerate(results):
                assert result.success is True
                assert result.value == items[i] ** 2

    def test_map_preserves_order(self):
        with AnalysisPool(max_workers=4) as pool:
            items = list(range(10))
            results = pool.map_with_progress(_slow_square, items)
            values = [r.value for r in results]
            expected = [x * x for x in items]
            assert values == expected

    def test_map_progress_callback(self):
        progress_calls = []

        def track_progress(completed: int, total: int):
            progress_calls.append((completed, total))

        with AnalysisPool(max_workers=2) as pool:
            items = [1, 2, 3, 4]
            pool.map_with_progress(_slow_square, items, on_progress=track_progress)

        # Should have been called once per item
        assert len(progress_calls) == 4
        # All should have total=4
        assert all(total == 4 for _, total in progress_calls)
        # Completed should be 1, 2, 3, 4 (in some order during execution, but final is 4)
        completed_values = sorted(c for c, _ in progress_calls)
        assert completed_values == [1, 2, 3, 4]

    def test_map_result_callback(self):
        result_calls = []

        def track_result(idx: int, result: TaskResult):
            result_calls.append((idx, result.value))

        with AnalysisPool(max_workers=2) as pool:
            items = [2, 3, 4]
            pool.map_with_progress(_slow_square, items, on_result=track_result)

        # Should have been called for each item
        assert len(result_calls) == 3
        # Check all indices were covered
        indices = [idx for idx, _ in result_calls]
        assert sorted(indices) == [0, 1, 2]
        # Check results match expected values
        for idx, value in result_calls:
            assert value == items[idx] ** 2

    def test_map_with_errors(self):
        with AnalysisPool(max_workers=2) as pool:
            items = [1, 2, 3]
            results = pool.map_with_progress(_raise_error, items)
            assert len(results) == 3
            for result in results:
                assert result.success is False
                assert isinstance(result.error, ValueError)

    def test_map_mixed_success_and_errors(self):
        with AnalysisPool(max_workers=2) as pool:
            items = [1, 2, 3, 4, 5]
            results = pool.map_with_progress(_maybe_fail, items)

            assert results[0].success is True  # 1 -> 2
            assert results[1].success is False  # 2 -> error
            assert results[2].success is True  # 3 -> 6
            assert results[3].success is False  # 4 -> error
            assert results[4].success is True  # 5 -> 10

    def test_map_with_dict_items(self):
        with AnalysisPool(max_workers=2) as pool:
            items = [
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3, "value": 30},
            ]
            results = pool.map_with_progress(_process_item, items)

            assert len(results) == 3
            assert results[0].value == {"id": 1, "result": 20}
            assert results[1].value == {"id": 2, "result": 40}
            assert results[2].value == {"id": 3, "result": 60}


class TestAnalysisPoolShutdown:
    """Tests for shutdown behavior."""

    def test_shutdown_wait(self):
        pool = AnalysisPool(max_workers=1)
        future = pool.submit(_slow_square, 5)
        pool.shutdown(wait=True)
        # Result should be available since we waited
        result = future.result(timeout=0.1)
        assert result.success is True
        assert result.value == 25

    def test_double_shutdown_safe(self):
        pool = AnalysisPool(max_workers=1)
        pool.submit(_add, 1, 1)
        pool.shutdown()
        # Second shutdown should not raise
        pool.shutdown()

    def test_shutdown_without_use(self):
        pool = AnalysisPool(max_workers=1)
        # Shutdown without ever using the pool
        pool.shutdown()
        # Should be safe


class TestAnalysisPoolParallelism:
    """Tests verifying actual parallelism."""

    def test_parallel_execution_faster(self):
        """Verify parallel execution is faster than sequential."""
        items = list(range(8))  # 8 items, each takes 0.1s

        # Parallel with 4 workers
        start = time.time()
        with AnalysisPool(max_workers=4) as pool:
            pool.map_with_progress(_slow_square, items)
        parallel_time = time.time() - start

        # Sequential would take ~0.8s, parallel with 4 workers should be ~0.2s
        # Use generous bounds to avoid flaky tests
        assert parallel_time < 0.6  # Should be much less than sequential
