"""
Analysis pool for parallel processing of CPU-bound tasks.

Uses ProcessPoolExecutor with spawn context for GUI compatibility.
This approach was chosen over free-threading (no-GIL) and subinterpreters because:
- Numba's free-threading support is still experimental (as of 0.63.x)
- ProcessPoolExecutor is battle-tested for scientific computing
- Memory isolation prevents race conditions with NumPy arrays
- Works identically across macOS, Windows, and Linux
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskResult(Generic[T]):
    """Result wrapper for pool tasks."""

    success: bool
    value: T | None = None
    error: Exception | None = None

    @classmethod
    def ok(cls, value: T) -> TaskResult[T]:
        return cls(success=True, value=value)

    @classmethod
    def err(cls, error: Exception) -> TaskResult[T]:
        return cls(success=False, error=error)


def _get_default_workers() -> int:
    """Get default number of worker processes."""
    cpu_count = os.cpu_count() or 1
    # Leave one core free for GUI/system
    return max(1, cpu_count - 1)


class AnalysisPool:
    """
    Persistent process pool for analysis tasks.

    This pool maintains worker processes between calls for efficiency.
    Uses spawn context for compatibility with GUI frameworks (DearPyGui).

    Usage:
        pool = AnalysisPool()

        # Single task
        future = pool.submit(analyze_particle, particle_data)
        result = future.result()

        # Batch with progress
        def on_progress(completed, total):
            print(f"{completed}/{total}")

        results = pool.map_with_progress(
            analyze_particle,
            particle_list,
            on_progress
        )

        # Cleanup
        pool.shutdown()
    """

    def __init__(self, max_workers: int | None = None):
        """
        Initialize the analysis pool.

        Args:
            max_workers: Maximum worker processes. Defaults to CPU count - 1.
        """
        self._max_workers = max_workers or _get_default_workers()
        self._ctx = mp.get_context("spawn")
        self._executor: ProcessPoolExecutor | None = None

    @property
    def max_workers(self) -> int:
        """Maximum number of worker processes."""
        return self._max_workers

    def _ensure_executor(self) -> ProcessPoolExecutor:
        """Lazily create the executor on first use."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=self._ctx,
            )
        return self._executor

    def submit(
        self,
        fn: Callable[..., R],
        *args: Any,
        **kwargs: Any,
    ) -> Future[TaskResult[R]]:
        """
        Submit a single task to the pool.

        Args:
            fn: Function to call (must be picklable - module-level function)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future that resolves to TaskResult
        """
        executor = self._ensure_executor()
        return executor.submit(_run_task, fn, args, kwargs)

    def map_with_progress(
        self,
        fn: Callable[[T], R],
        items: list[T],
        on_progress: Callable[[int, int], None] | None = None,
        on_result: Callable[[int, TaskResult[R]], None] | None = None,
    ) -> list[TaskResult[R]]:
        """
        Process items in parallel with progress callbacks.

        Args:
            fn: Function to apply to each item (must be picklable)
            items: List of items to process
            on_progress: Called with (completed_count, total_count) after each completion
            on_result: Called with (index, result) after each completion

        Returns:
            List of TaskResults in same order as input items
        """
        if not items:
            return []

        executor = self._ensure_executor()
        total = len(items)

        # Submit all tasks, tracking their original indices
        future_to_index: dict[Future[TaskResult[R]], int] = {}
        for idx, item in enumerate(items):
            future = executor.submit(_run_task, fn, (item,), {})
            future_to_index[future] = idx

        # Collect results as they complete
        results: list[TaskResult[R] | None] = [None] * total
        completed = 0

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result = future.result()  # TaskResult from _run_task
            results[idx] = result
            completed += 1

            if on_result is not None:
                on_result(idx, result)

            if on_progress is not None:
                on_progress(completed, total)

        return results  # type: ignore[return-value]

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """
        Shutdown the pool and release resources.

        Args:
            wait: If True, wait for pending tasks to complete
            cancel_futures: If True, cancel pending futures
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None

    def __enter__(self) -> AnalysisPool:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown(wait=True)


def _run_task(
    fn: Callable[..., R],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> TaskResult[R]:
    """
    Execute a task and wrap the result.

    This function runs in the worker process.
    """
    try:
        result = fn(*args, **kwargs)
        return TaskResult.ok(result)
    except Exception as e:
        return TaskResult.err(e)
