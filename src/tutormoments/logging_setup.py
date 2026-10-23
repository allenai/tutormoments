"""Logging configuration shared by the tutormoments and tutormoments-build CLIs.

setup_logging() attaches two sinks to the root logger:
  - a console handler (stderr), always on
  - an optional file handler; it receives the same records as the console,
    so a saved log file is a complete record of the run

The tutormoments / tutormoments_build package loggers are set to the requested
level; the root logger stays at WARNING so third-party libraries (httpx,
urllib3, botocore, ...) do not flood the output at INFO/DEBUG.

The file handler appends, so a resumed run accumulates into the same file.
"""

import argparse
import contextvars
import logging
import os
import threading
from contextlib import contextmanager

LOG_LEVEL_ENV = "TUTORMOMENTS_LOG_LEVEL"
LOG_FILE_ENV = "TUTORMOMENTS_LOG_FILE"

_CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(cell)s%(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)s %(cell)s%(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_PACKAGE_LOGGERS = ("tutormoments", "tutormoments_build")

# Marker attribute on handlers installed by setup_logging(), so repeated
# calls replace them instead of stacking duplicates.
_HANDLER_TAG = "_tutormoments_logging_setup"

# Active cell tag (e.g. "gpt-5-4/plain"); rendered as a [tag] prefix on every
# record so interleaved console lines from parallel sweep lanes stay
# attributable. contextvars do not cross thread boundaries, so each lane
# thread carries its own tag.
_cell_tag = contextvars.ContextVar("tutormoments_log_cell_tag", default="")


class _CellTagFilter(logging.Filter):
    """Stamp each record with the active cell tag for the %(cell)s field."""

    def filter(self, record):
        tag = _cell_tag.get()
        record.cell = f"[{tag}] " if tag else ""
        return True


@contextmanager
def log_context(tag: str):
    """Prefix all records logged in this context (and thread) with [tag]."""
    token = _cell_tag.set(tag)
    try:
        yield
    finally:
        _cell_tag.reset(token)


def logging_args_parent() -> argparse.ArgumentParser:
    """Parent parser carrying the shared --log-level / --log-file flags.

    Pass via parents=[...] to each subcommand parser so the flags can be
    given after the subcommand (e.g. `tutormoments run --log-file run.log`).
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--log-level",
        default=None,
        dest="log_level",
        metavar="LEVEL",
        help="Log level: DEBUG, INFO, WARNING, or ERROR "
        f"(default: INFO, or ${LOG_LEVEL_ENV})",
    )
    parent.add_argument(
        "--log-file",
        default=None,
        dest="log_file",
        metavar="FILE",
        help="Also append logs to FILE; recommended for reproducibility "
        f"(default: ${LOG_FILE_ENV} if set)",
    )
    return parent


def setup_logging(level: str | None = None, log_file: str | None = None) -> None:
    """Configure console (and optional file) logging for a CLI process.

    Args:
        level: Log level name for tutormoments/tutormoments_build loggers
            (default: $TUTORMOMENTS_LOG_LEVEL, then "INFO").
        log_file: Path to append logs to, in addition to the console
            (default: $TUTORMOMENTS_LOG_FILE, then no file).

    Raises:
        ValueError: on an unrecognized level name.
    """
    level_name = (level or os.environ.get(LOG_LEVEL_ENV) or "INFO").upper()
    resolved = logging.getLevelName(level_name)
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown log level: {level_name!r}")
    log_file = log_file or os.environ.get(LOG_FILE_ENV) or None

    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]:
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    console.addFilter(_CellTagFilter())
    setattr(console, _HANDLER_TAG, True)
    root.addHandler(console)

    if log_file:
        parent_dir = os.path.dirname(log_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
        file_handler.addFilter(_CellTagFilter())
        setattr(file_handler, _HANDLER_TAG, True)
        root.addHandler(file_handler)

    # Root stays at WARNING: third-party records must clear it to reach the
    # handlers. Our package loggers get the requested level and propagate up.
    root.setLevel(logging.WARNING)
    for name in _PACKAGE_LOGGERS:
        logging.getLogger(name).setLevel(resolved)

    if log_file:
        logging.getLogger("tutormoments").info("Logging to file: %s", log_file)


def _file_only_record(handler: logging.Handler, message: str) -> None:
    """Emit one INFO record directly to a handler, bypassing the console."""
    record = logging.LogRecord(
        name="tutormoments",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.cell = ""  # emit() bypasses filters; the formatter needs the field
    handler.emit(record)


# Run-log membership, per thread: the tokens of every per_run_log_file block
# this thread logs into. Thread-local rather than keyed on thread ids, because
# idents are recycled once a thread exits -- a new thread (e.g. the next sweep
# cell's replay worker) starts with no tokens even if it reuses the ident of
# a dead registered worker.
_run_log_membership = threading.local()


def _current_run_logs() -> frozenset:
    return getattr(_run_log_membership, "tokens", frozenset())


def _join_run_log(token: object) -> None:
    _run_log_membership.tokens = _current_run_logs() | {token}


def _leave_run_log(token: object) -> None:
    _run_log_membership.tokens = _current_run_logs() - {token}


class RunLogHandle:
    """Handle yielded by per_run_log_file for adopting worker threads.

    The run-log filter passes only threads that joined this run's log
    (initially just the thread that opened it). A worker-pool thread that
    should log into this run's file calls register_current_thread() --
    typically via bind_worker_logging() in a ThreadPoolExecutor initializer.
    """

    def __init__(self, token: object):
        self._token = token

    def register_current_thread(self) -> None:
        _join_run_log(self._token)


def bind_worker_logging(handle: "RunLogHandle | None", tag: str) -> None:
    """Adopt a run's log file and cell tag on a pool worker thread.

    Intended as (part of) a ThreadPoolExecutor initializer: registers the
    worker with the run-log thread filter so its records reach run.log, and
    sets the [tag] contextvar (contextvars don't cross thread boundaries, so
    the spawning thread's log_context() tag is otherwise lost). Both are
    per-thread state that dies with the worker, so neither needs undoing.
    """
    if handle is not None:
        handle.register_current_thread()
    _cell_tag.set(tag)


@contextmanager
def per_run_log_file(
    log_file: str,
    *,
    current_thread_only: bool = True,
    header: str | None = None,
):
    """Attach a run-scoped file handler to the root logger for the block.

    Backs the automatic per-run log (e.g. results/<run_id>/run.log). With
    current_thread_only (the default), only records emitted by the calling
    thread -- plus any worker threads registered via the yielded
    RunLogHandle (see bind_worker_logging) -- are written, so parallel
    sweep lanes in one process each keep their own run log. The file
    appends, matching resume semantics.

    Args:
        log_file: Path to the log file; parent directories are created.
        current_thread_only: Restrict capture to the calling thread and
            explicitly registered worker threads.
        header: Optional line written to the file only (not the console),
            e.g. the invoked command line.

    Yields:
        RunLogHandle for registering worker threads (None-safe to ignore).
    """
    parent_dir = os.path.dirname(log_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_CellTagFilter())
    # Filters run on the emitting thread, so this checks that thread's own
    # membership. Not record.thread: a registered replay worker exits while
    # the block stays open for scoring, and a later thread that reuses its
    # ident (e.g. the next sweep cell's worker) must not inherit it.
    token = object()
    if current_thread_only:
        handler.addFilter(lambda record: token in _current_run_logs())
    if header:
        _file_only_record(handler, header)

    # Programmatic callers may never run setup_logging(); a NOTSET package
    # logger would gate INFO records at the root's WARNING level before they
    # reach any handler, leaving the run log empty. Bump NOTSET loggers to
    # INFO while attached; explicitly configured levels are respected.
    bumped = []
    for name in _PACKAGE_LOGGERS:
        pkg_logger = logging.getLogger(name)
        if pkg_logger.level == logging.NOTSET:
            pkg_logger.setLevel(logging.INFO)
            bumped.append(pkg_logger)

    root = logging.getLogger()
    root.addHandler(handler)
    _join_run_log(token)
    try:
        yield RunLogHandle(token)
    finally:
        _leave_run_log(token)
        root.removeHandler(handler)
        handler.close()
        for pkg_logger in bumped:
            pkg_logger.setLevel(logging.NOTSET)
