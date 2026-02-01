"""
Parallel processing utilities for region analysis.

Provides thread-safe logging with buffered output per task,
and parallelization helpers using ProcessPoolExecutor.
"""
import sys
import logging
import threading
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, TypeVar, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from config import get_config

log = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# =============================================================================
# Buffered Logging for Parallel Tasks
# =============================================================================

class BufferedLogHandler(logging.Handler):
    """
    A logging handler that buffers log records.
    Used to capture logs from a task and emit them all at once.
    """
    def __init__(self):
        super().__init__()
        self.buffer: List[logging.LogRecord] = []
        self.lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        with self.lock:
            self.buffer.append(record)

    def get_records(self) -> List[logging.LogRecord]:
        with self.lock:
            records = self.buffer.copy()
            self.buffer.clear()
            return records

    def clear(self):
        with self.lock:
            self.buffer.clear()


@dataclass
class TaskResult:
    """Result from a parallel task with captured logs."""
    task_id: Any
    result: Any
    logs: List[str]
    success: bool
    error: Optional[str] = None


class LogCapture:
    """
    Context manager to capture log output from a block of code.
    Returns captured log lines as a list of strings.
    """
    def __init__(self, logger_name: Optional[str] = None):
        self.logger_name = logger_name
        self.handler = BufferedLogHandler()
        self.handler.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter("%(levelname)s | %(message)s")
        self.handler.setFormatter(self.formatter)
        self.logger: Optional[logging.Logger] = None

    def __enter__(self) -> "LogCapture":
        if self.logger_name:
            self.logger = logging.getLogger(self.logger_name)
        else:
            self.logger = logging.getLogger()
        self.logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            self.logger.removeHandler(self.handler)

    def get_logs(self) -> List[str]:
        """Get captured log lines."""
        return [self.formatter.format(r) for r in self.handler.get_records()]


# =============================================================================
# Thread-Safe Print Lock
# =============================================================================

_print_lock = threading.Lock()
_log_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    """Print with lock to prevent interleaving."""
    with _print_lock:
        print(*args, **kwargs)


def emit_logs_atomically(logs: List[str], header: Optional[str] = None):
    """
    Emit a batch of log lines atomically (all at once).
    Prevents interleaving with logs from other threads/processes.
    """
    with _log_lock:
        if header:
            print(header, flush=True)
        for line in logs:
            print(line, flush=True)
        sys.stdout.flush()


# =============================================================================
# Parallel Season Processing
# =============================================================================

# Global state for worker processes
_worker_shared_data: Dict[str, Any] = {}


def _init_season_worker(shared_data: Dict[str, Any]):
    """Initialize worker with shared data."""
    global _worker_shared_data
    _worker_shared_data = shared_data


def run_parallel_seasons(
    seasons: List[int],
    process_fn: Callable[[int, Dict[str, Any]], Tuple[List[Any], List[str]]],
    shared_data: Dict[str, Any],
    task_name: str = "Processing",
) -> Dict[int, List[Any]]:
    """
    Process multiple seasons in parallel.

    Logs are buffered per-season and emitted atomically in season order
    after all seasons complete.

    Args:
        seasons: List of season numbers to process
        process_fn: Function(season, shared_data) -> (results, log_lines)
        shared_data: Data to share with all workers
        task_name: Name for logging

    Returns:
        Dict mapping season -> results
    """
    cfg = get_config()
    n_workers = min(cfg.parallel.n_workers, len(seasons))

    all_results: Dict[int, List[Any]] = {}

    if n_workers <= 1 or len(seasons) <= 1:
        # Sequential execution
        for season in seasons:
            log.info(f"\n{'='*60}")
            log.info(f"Season {season}")
            log.info(f"{'='*60}")
            results, _ = process_fn(season, shared_data)
            all_results[season] = results
    else:
        # Parallel execution
        log.info(f"{task_name} {len(seasons)} seasons using {n_workers} workers...")

        # Collect ALL results first, then emit logs in order
        completed_results: Dict[int, TaskResult] = {}

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_season_worker,
            initargs=(shared_data,)
        ) as executor:
            # Submit all seasons
            future_to_season = {
                executor.submit(_season_worker_wrapper, season, process_fn): season
                for season in seasons
            }

            # Collect results silently as they complete
            for future in as_completed(future_to_season):
                season = future_to_season[future]
                try:
                    result = future.result()
                    completed_results[season] = result
                    # Just show progress, don't emit logs yet
                    print(f"  Completed season {season}", flush=True)
                except Exception as e:
                    completed_results[season] = TaskResult(
                        task_id=season,
                        result=[],
                        logs=[f"ERROR | Season {season} failed: {e}"],
                        success=False,
                        error=str(e),
                    )
                    print(f"  FAILED season {season}: {e}", flush=True)

        # Now emit all logs in season order, atomically per season
        print("\n" + "="*60, flush=True)
        print("DETAILED LOGS BY SEASON", flush=True)
        print("="*60, flush=True)

        for season in sorted(seasons):
            task_result = completed_results.get(season)
            if task_result:
                header = f"\n{'='*60}\nSeason {season}\n{'='*60}"
                emit_logs_atomically(task_result.logs, header)
                all_results[season] = task_result.result

    return all_results


def _season_worker_wrapper(season: int, process_fn: Callable) -> TaskResult:
    """
    Wrapper that captures logs from a season processing function.
    Called in worker process.
    """
    global _worker_shared_data

    # Capture logs
    log_buffer = []

    # Create a simple log capture by redirecting the module's logger
    class ListHandler(logging.Handler):
        def emit(self, record):
            log_buffer.append(self.format(record))

    handler = ListHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s |   %(message)s"))

    # Get the root logger and backprojection/forwardprojection loggers
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        results, extra_logs = process_fn(season, _worker_shared_data)
        log_buffer.extend(extra_logs)
        return TaskResult(
            task_id=season,
            result=results,
            logs=log_buffer,
            success=True,
        )
    except Exception as e:
        import traceback
        log_buffer.append(f"ERROR | {traceback.format_exc()}")
        return TaskResult(
            task_id=season,
            result=[],
            logs=log_buffer,
            success=False,
            error=str(e),
        )
    finally:
        root_logger.removeHandler(handler)


# =============================================================================
# Generic Parallel Map
# =============================================================================

def parallel_map(
    items: List[T],
    process_fn: Callable[[T], R],
    n_workers: Optional[int] = None,
    desc: str = "Processing",
) -> List[R]:
    """
    Apply process_fn to each item in parallel.

    Args:
        items: List of items to process
        process_fn: Function to apply to each item
        n_workers: Number of workers (default: from config)
        desc: Description for logging

    Returns:
        List of results in same order as items
    """
    cfg = get_config()
    if n_workers is None:
        n_workers = cfg.parallel.n_workers

    n_workers = min(n_workers, len(items))

    if n_workers <= 1 or len(items) <= 2:
        return [process_fn(item) for item in items]

    results = [None] * len(items)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(process_fn, item): i
            for i, item in enumerate(items)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results


def parallel_map_batched(
    items: List[T],
    process_batch_fn: Callable[[List[T]], List[R]],
    batch_size: Optional[int] = None,
    n_workers: Optional[int] = None,
    desc: str = "Processing",
) -> List[R]:
    """
    Apply process_batch_fn to batches of items in parallel.

    Args:
        items: List of items to process
        process_batch_fn: Function that processes a batch and returns list of results
        batch_size: Size of each batch (default: from config)
        n_workers: Number of workers (default: from config)
        desc: Description for logging

    Returns:
        List of results in same order as items
    """
    cfg = get_config()
    if n_workers is None:
        n_workers = cfg.parallel.n_workers
    if batch_size is None:
        batch_size = cfg.parallel.batch_size

    n_workers = min(n_workers, len(items))

    if n_workers <= 1 or len(items) <= batch_size:
        return process_batch_fn(items)

    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    # Process batches in parallel
    batch_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(process_batch_fn, batch): i
            for i, batch in enumerate(batches)
        }

        batch_results = [None] * len(batches)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            batch_results[idx] = future.result()

    # Flatten results
    return [r for batch in batch_results for r in batch]
