"""Tests for tutorsim.logging_setup (console + optional file logging)."""

import logging
import threading

import pytest

from tutorsim.logging_setup import (
    _HANDLER_TAG,
    LOG_FILE_ENV,
    LOG_LEVEL_ENV,
    log_context,
    per_run_log_file,
    setup_logging,
)


def _our_handlers():
    root = logging.getLogger()
    return [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]


@pytest.fixture(autouse=True)
def _restore_logging_state(monkeypatch):
    """Snapshot and restore global logging state around each test."""
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    monkeypatch.delenv(LOG_FILE_ENV, raising=False)
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_root_level = root.level
    pkg_levels = {
        name: logging.getLogger(name).level for name in ("tutorsim", "tutorsim_build")
    }
    yield
    for handler in root.handlers[:]:
        if handler not in old_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(old_root_level)
    for name, level in pkg_levels.items():
        logging.getLogger(name).setLevel(level)


def test_console_only_by_default():
    setup_logging()
    handlers = _our_handlers()
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert logging.getLogger("tutorsim").getEffectiveLevel() == logging.INFO
    assert logging.getLogger("tutorsim_build").getEffectiveLevel() == logging.INFO


def test_file_and_console_both_receive(tmp_path, capsys):
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(log_file=str(log_file))

    logging.getLogger("tutorsim.cli").info("hello from the run")
    for handler in _our_handlers():
        handler.flush()

    assert log_file.read_text(encoding="utf-8").count("hello from the run") == 1
    assert "hello from the run" in capsys.readouterr().err


def test_file_appends_across_setups(tmp_path):
    log_file = tmp_path / "run.log"
    setup_logging(log_file=str(log_file))
    logging.getLogger("tutorsim.cli").info("first run")
    setup_logging(log_file=str(log_file))
    logging.getLogger("tutorsim.cli").info("second run")
    for handler in _our_handlers():
        handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "first run" in content
    assert "second run" in content


def test_repeated_setup_does_not_stack_handlers(tmp_path, capsys):
    log_file = tmp_path / "run.log"
    setup_logging(log_file=str(log_file))
    setup_logging(log_file=str(log_file))
    assert len(_our_handlers()) == 2  # one console + one file

    logging.getLogger("tutorsim.cli").info("only once")
    for handler in _our_handlers():
        handler.flush()
    assert log_file.read_text(encoding="utf-8").count("only once") == 1
    assert capsys.readouterr().err.count("only once") == 1


def test_env_var_defaults(tmp_path, monkeypatch):
    log_file = tmp_path / "env.log"
    monkeypatch.setenv(LOG_LEVEL_ENV, "DEBUG")
    monkeypatch.setenv(LOG_FILE_ENV, str(log_file))
    setup_logging()

    assert logging.getLogger("tutorsim").getEffectiveLevel() == logging.DEBUG
    assert any(isinstance(h, logging.FileHandler) for h in _our_handlers())
    assert log_file.exists()


def test_explicit_args_win_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "DEBUG")
    setup_logging(level="WARNING")
    assert logging.getLogger("tutorsim").getEffectiveLevel() == logging.WARNING


def test_third_party_info_suppressed():
    setup_logging(level="INFO")
    assert not logging.getLogger("somelib").isEnabledFor(logging.INFO)
    assert logging.getLogger("somelib").isEnabledFor(logging.WARNING)
    assert logging.getLogger("tutorsim.client").isEnabledFor(logging.INFO)


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        setup_logging(level="LOUD")


def test_per_run_log_file_captures_and_detaches(tmp_path):
    setup_logging()
    log_file = tmp_path / "run" / "run.log"
    handlers_before = logging.getLogger().handlers[:]

    with per_run_log_file(str(log_file)):
        logging.getLogger("tutorsim.cli").info("inside the run")
    logging.getLogger("tutorsim.cli").info("after the run")

    content = log_file.read_text(encoding="utf-8")
    assert "inside the run" in content
    assert "after the run" not in content
    assert logging.getLogger().handlers == handlers_before


def test_per_run_log_file_isolates_threads(tmp_path):
    """Two concurrent runs each capture only their own thread's records."""
    setup_logging()
    log = logging.getLogger("tutorsim.cli")
    barrier = threading.Barrier(2)

    def work(idx):
        with per_run_log_file(str(tmp_path / f"run{idx}.log")):
            barrier.wait()  # both handlers attached before either logs
            log.info("message from run %d", idx)
            barrier.wait()  # both logged before either detaches

    threads = [threading.Thread(target=work, args=(i,)) for i in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    run1 = (tmp_path / "run1.log").read_text(encoding="utf-8")
    run2 = (tmp_path / "run2.log").read_text(encoding="utf-8")
    assert "message from run 1" in run1 and "message from run 2" not in run1
    assert "message from run 2" in run2 and "message from run 1" not in run2


def test_per_run_header_written_to_file_only(tmp_path, capsys):
    setup_logging()
    log_file = tmp_path / "run.log"

    with per_run_log_file(str(log_file), header="Command: tutorsim-build x"):
        pass

    assert "Command: tutorsim-build x" in log_file.read_text(encoding="utf-8")
    assert "Command: tutorsim-build x" not in capsys.readouterr().err


def test_per_run_log_works_without_setup_logging(tmp_path):
    """Programmatic run_cell callers get a complete run log even when
    setup_logging() was never called (NOTSET loggers bumped to INFO)."""
    pkg_logger = logging.getLogger("tutorsim")
    pkg_logger.setLevel(logging.NOTSET)
    log_file = tmp_path / "run.log"

    with per_run_log_file(str(log_file)):
        logging.getLogger("tutorsim.cli").info("captured without setup")

    assert "captured without setup" in log_file.read_text(encoding="utf-8")
    assert pkg_logger.level == logging.NOTSET  # restored on exit


def test_log_context_prefixes_console_and_file(tmp_path, capsys):
    """Records inside log_context carry the [cell] prefix on every sink."""
    setup_logging()
    log_file = tmp_path / "run.log"

    with per_run_log_file(str(log_file)):
        with log_context("gpt-5-4/plain"):
            logging.getLogger("tutorsim.cli").info("inside cell")
        logging.getLogger("tutorsim.cli").info("outside cell")

    content = log_file.read_text(encoding="utf-8")
    assert "[gpt-5-4/plain] tutorsim.cli: inside cell" in content
    assert "[gpt-5-4/plain] tutorsim.cli: outside cell" not in content
    assert "outside cell" in content
    err = capsys.readouterr().err
    assert "[gpt-5-4/plain] inside cell" in err


def test_log_context_is_thread_local(tmp_path):
    """Each thread sees only its own cell tag."""
    setup_logging()
    log = logging.getLogger("tutorsim.cli")
    barrier = threading.Barrier(2)

    def work(tag):
        with per_run_log_file(str(tmp_path / f"{tag}.log")), log_context(tag):
            barrier.wait()
            log.info("hello")
            barrier.wait()

    threads = [threading.Thread(target=work, args=(t,)) for t in ("cellA", "cellB")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert "[cellA] " in (tmp_path / "cellA.log").read_text(encoding="utf-8")
    assert "[cellB] " in (tmp_path / "cellB.log").read_text(encoding="utf-8")


def test_cli_run_parser_accepts_log_flags():
    from tutorsim.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "run",
            "--tutors",
            "some-model",
            "--log-level",
            "DEBUG",
            "--log-file",
            "run.log",
        ]
    )
    assert args.log_level == "DEBUG"
    assert args.log_file == "run.log"

    args = _build_parser().parse_args(["report"])
    assert args.log_level is None
    assert args.log_file is None


def test_per_run_log_file_registered_worker_thread_captured(tmp_path):
    """A pool worker registered via bind_worker_logging reaches run.log with
    the cell tag; an unregistered thread stays filtered out."""
    from tutorsim.logging_setup import bind_worker_logging

    setup_logging()
    log_file = tmp_path / "run.log"
    log = logging.getLogger("tutorsim.client")

    with per_run_log_file(str(log_file)) as handle:

        def worker():
            bind_worker_logging(handle, "tutor-x/plain")
            log.warning("retry warning from worker")

        def stranger():
            log.warning("record from unregistered thread")

        for target in (worker, stranger):
            t = threading.Thread(target=target)
            t.start()
            t.join()

    content = log_file.read_text(encoding="utf-8")
    assert "retry warning from worker" in content
    assert "[tutor-x/plain]" in content
    assert "record from unregistered thread" not in content
