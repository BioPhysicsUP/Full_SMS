"""Background processing with ProcessPoolExecutor."""

from full_sms.workers.pool import AnalysisPool, TaskResult

__all__ = ["AnalysisPool", "TaskResult"]
