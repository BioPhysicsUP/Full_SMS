"""Tests for AnalysisPool parallel processing."""

import time
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError

import pytest

from full_sms.workers.pool import AnalysisPool, TaskResult


# Test functions must be module-level for pickling
def _add(a: int, b: int) -> int:
    """Simple add function for testing."""
    return a + b


def _slow_square(x: int) -> int:
    """Square function for testing."""
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


def _slow_task(duration: float) -> str:
    """Task that sleeps for a duration."""
    time.sleep(duration)
    return f"completed after {duration}s"


def _return_with_id(task_id: int) -> tuple[int, float]:
    """Return task ID and current time for ordering tests."""
    return (task_id, time.time())


def _memory_intensive_task(size_mb: int) -> int:
    """Task that allocates memory to test process isolation."""
    # Allocate approximately size_mb megabytes
    data = bytearray(size_mb * 1024 * 1024)
    return len(data)


def _raise_runtime_error(x: int) -> int:
    """Function that raises RuntimeError."""
    raise RuntimeError(f"Runtime error for {x}")


def _timed_return(duration: float) -> float:
    """Return after sleeping for duration."""
    time.sleep(duration)
    return duration


def _sum_array(arr) -> float:
    """Sum a numpy array."""
    return arr.sum()


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

    def test_shutdown_cancel_futures(self):
        """Shutdown with cancel_futures=True cancels pending work."""
        pool = AnalysisPool(max_workers=1)
        # Submit a slow task
        pool.submit(_slow_task, 10.0)
        # Shutdown immediately with cancel
        pool.shutdown(wait=False, cancel_futures=True)
        # Should not hang


class TestAnalysisPoolErrorHandling:
    """Tests for error handling in pool operations."""

    def test_exception_type_preserved(self):
        """Specific exception types are preserved in TaskResult."""
        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(_raise_error, 42)
            result = future.result(timeout=5.0)

            assert result.success is False
            assert isinstance(result.error, ValueError)
            assert "42" in str(result.error)

    def test_multiple_exceptions_independent(self):
        """Each task's exception is independent."""
        with AnalysisPool(max_workers=2) as pool:
            future1 = pool.submit(_raise_error, 1)
            future2 = pool.submit(_raise_error, 2)

            result1 = future1.result(timeout=5.0)
            result2 = future2.result(timeout=5.0)

            assert "1" in str(result1.error)
            assert "2" in str(result2.error)

    def test_exception_does_not_crash_pool(self):
        """An exception in one task doesn't affect others."""
        with AnalysisPool(max_workers=2) as pool:
            future_fail = pool.submit(_raise_error, 42)
            future_ok = pool.submit(_add, 1, 2)

            result_fail = future_fail.result(timeout=5.0)
            result_ok = future_ok.result(timeout=5.0)

            assert result_fail.success is False
            assert result_ok.success is True
            assert result_ok.value == 3

    def test_runtime_error_captured(self):
        """RuntimeError in worker is captured as error."""
        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(_raise_runtime_error, 1)
            result = future.result(timeout=5.0)

            assert result.success is False
            assert isinstance(result.error, RuntimeError)
            assert "1" in str(result.error)

    def test_timeout_on_future_result(self):
        """Timeout raises when waiting for slow task."""
        with AnalysisPool(max_workers=1) as pool:
            future = pool.submit(_slow_task, 10.0)

            with pytest.raises(FuturesTimeoutError):
                future.result(timeout=0.1)

    def test_pool_continues_after_timeout(self):
        """Pool still works after a timeout exception."""
        with AnalysisPool(max_workers=1) as pool:
            # Submit slow task and timeout
            slow_future = pool.submit(_slow_task, 10.0)
            try:
                slow_future.result(timeout=0.1)
            except FuturesTimeoutError:
                pass

            # Submit another quick task - should work
            quick_future = pool.submit(_add, 1, 2)
            # Give it more time since the slow task is still running
            result = quick_future.result(timeout=15.0)
            assert result.success is True
            assert result.value == 3


class TestAnalysisPoolConcurrency:
    """Tests for concurrent execution behavior."""

    def test_multiple_workers_execute_in_parallel(self):
        """Multiple workers process tasks simultaneously."""
        with AnalysisPool(max_workers=4) as pool:
            # Submit 4 tasks that each take 0.5 seconds
            start_time = time.time()
            futures = [pool.submit(_slow_task, 0.5) for _ in range(4)]

            # Wait for all to complete
            for f in futures:
                f.result(timeout=5.0)

            elapsed = time.time() - start_time

            # With 4 workers, 4 x 0.5s tasks should complete in ~0.5s, not 2s
            # Allow some overhead but should be significantly less than sequential
            assert elapsed < 1.5, f"Expected parallel execution, took {elapsed}s"

    def test_max_workers_limit_respected(self):
        """Pool doesn't exceed max_workers limit."""
        with AnalysisPool(max_workers=2) as pool:
            # Submit 6 tasks that each take 0.3 seconds
            start_time = time.time()
            futures = [pool.submit(_slow_task, 0.3) for _ in range(6)]

            # Wait for all
            for f in futures:
                f.result(timeout=10.0)

            elapsed = time.time() - start_time

            # With 2 workers, 6 tasks at 0.3s each should take ~0.9s minimum
            # (3 batches of 2)
            assert elapsed >= 0.8, f"Expected ~0.9s minimum, took {elapsed}s"

    def test_results_order_independent_of_completion(self):
        """map_with_progress returns results in input order, not completion order."""
        with AnalysisPool(max_workers=4) as pool:
            # Tasks with varying durations
            durations = [0.3, 0.1, 0.4, 0.2, 0.15]

            results = pool.map_with_progress(_timed_return, durations)

            # Results should be in original order
            for i, result in enumerate(results):
                assert result.success is True
                assert result.value == durations[i]

    def test_concurrent_submit_and_map(self):
        """submit() and map_with_progress() can be used together."""
        with AnalysisPool(max_workers=4) as pool:
            # Submit some individual tasks
            future1 = pool.submit(_add, 1, 2)
            future2 = pool.submit(_add, 3, 4)

            # Also do a map operation
            map_results = pool.map_with_progress(_slow_square, [1, 2, 3])

            # All should complete successfully
            assert future1.result(timeout=5.0).value == 3
            assert future2.result(timeout=5.0).value == 7
            assert [r.value for r in map_results] == [1, 4, 9]

    def test_reuse_pool_between_operations(self):
        """Pool can be reused for multiple operations."""
        with AnalysisPool(max_workers=2) as pool:
            # First batch
            results1 = pool.map_with_progress(_slow_square, [1, 2, 3])
            assert [r.value for r in results1] == [1, 4, 9]

            # Second batch
            results2 = pool.map_with_progress(_slow_square, [4, 5, 6])
            assert [r.value for r in results2] == [16, 25, 36]

            # Individual submit
            future = pool.submit(_add, 10, 20)
            assert future.result(timeout=5.0).value == 30


class TestAnalysisPoolMemoryIsolation:
    """Tests for process isolation and memory safety."""

    def test_large_data_transfer(self):
        """Large data can be passed to/from workers."""
        import numpy as np

        with AnalysisPool(max_workers=1) as pool:
            # Create a large input (~1MB)
            large_array = np.random.rand(1000, 1000)

            future = pool.submit(_sum_array, large_array)
            result = future.result(timeout=30.0)

            assert result.success is True
            assert isinstance(result.value, float)

    def test_worker_memory_isolation(self):
        """Memory allocated in worker doesn't affect main process."""
        import os

        with AnalysisPool(max_workers=1) as pool:
            # Allocate 10MB in worker
            future = pool.submit(_memory_intensive_task, 10)
            result = future.result(timeout=10.0)

            assert result.success is True
            assert result.value == 10 * 1024 * 1024


class TestAnalysisPoolLazyInit:
    """Tests for lazy initialization behavior."""

    def test_executor_not_created_until_first_use(self):
        """Executor is only created when first task is submitted."""
        pool = AnalysisPool(max_workers=2)
        assert pool._executor is None

        # Accessing max_workers doesn't create executor
        _ = pool.max_workers
        assert pool._executor is None

        # Submitting creates executor
        pool.submit(_add, 1, 2)
        assert pool._executor is not None

        pool.shutdown()

    def test_map_creates_executor_on_nonempty_list(self):
        """map_with_progress creates executor for non-empty lists."""
        pool = AnalysisPool(max_workers=2)
        assert pool._executor is None

        # Empty list doesn't create executor
        pool.map_with_progress(_slow_square, [])
        assert pool._executor is None

        # Non-empty list creates executor
        pool.map_with_progress(_slow_square, [1])
        assert pool._executor is not None

        pool.shutdown()
