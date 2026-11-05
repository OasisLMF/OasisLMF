"""redirect_logging must leave the process's logging configuration as it found it."""
import io
import logging
import os
import tempfile
import warnings
from pathlib import Path
from unittest import mock

from oasislmf.pytools.gulmc.manager import run as run_gulmc
from oasislmf.pytools.utils import redirect_logging

TESTS_ASSETS_DIR = Path(__file__).parents[1].joinpath("assets")


def _run_engine():
    with tempfile.TemporaryDirectory() as t:
        rd = Path(t) / 'assets'
        os.symlink(TESTS_ASSETS_DIR.joinpath("test_model_1").resolve(), rd, target_is_directory=True)
        run_gulmc(run_dir=rd, ignore_file_type=set(), file_in=rd / 'input' / 'events.bin',
                  file_out=Path(t) / 'o.bin', sample_size=10, loss_threshold=0., alloc_rule=1,
                  debug=0, random_generator=0, ignore_correlation=True,
                  effective_damageability=False)


def test_engine_run_restores_oasislmf_logger_levels():
    """redirect_logging sets a level on every 'oasislmf.*' logger. Leaving it in place suppressed
    INFO and DEBUG from every oasislmf submodule for the rest of the process — including in
    `oasislmf model run`, which continues with output steps after the engine."""
    logger = logging.getLogger('oasislmf.preparation.gul_inputs')
    logger.setLevel(logging.NOTSET)
    _run_engine()
    assert logger.level == logging.NOTSET, "level left pinned at the engine run's log level"
    assert logger.propagate is True


def test_engine_run_leaves_a_third_party_logger_alone():
    """redirect_logging walks every logger in the process, so the reset must not force propagate
    back on for loggers whose propagate it never cleared — a host application may have set it
    deliberately."""
    third_party = logging.getLogger('some_host_app.module')
    third_party.propagate = False
    try:
        _run_engine()
        assert third_party.propagate is False, "a third-party logger's propagate was overwritten"
    finally:
        third_party.propagate = True


def _run_decorated(body=lambda: None, **kwargs):
    with tempfile.TemporaryDirectory() as log_dir:
        @redirect_logging(exec_name='test_tool', log_dir=log_dir)
        def run_func(**kw):
            body()
        run_func(**kwargs)


def test_host_logger_and_handler_levels_are_restored():
    """A host application's configured levels must survive an engine run, not come back as NOTSET."""
    host = logging.getLogger('some_host_app.levels')
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.INFO)
    host.setLevel(logging.WARNING)
    host.addHandler(handler)
    try:
        _run_decorated()
        assert host.level == logging.WARNING
        assert handler.level == logging.INFO
    finally:
        host.removeHandler(handler)
        host.setLevel(logging.NOTSET)


def test_oasislmf_root_logger_level_is_restored():
    root = logging.getLogger('oasislmf')
    previous = root.level
    root.setLevel(logging.WARNING)
    try:
        _run_decorated()
        assert root.level == logging.WARNING, "the 'oasislmf' logger was left at INFO"
    finally:
        root.setLevel(previous)


def test_handlers_attached_to_oasislmf_loggers_are_kept_and_left_open():
    """Only the handlers redirect_logging adds may be removed or closed; a host's own handlers stay."""
    logger = logging.getLogger('oasislmf.some_host_handler')
    with tempfile.TemporaryDirectory() as tmp_dir:
        handler = logging.FileHandler(Path(tmp_dir) / 'host.log', delay=False)
        logger.addHandler(handler)
        try:
            _run_decorated()
            assert handler in logger.handlers, "a host handler was removed from an 'oasislmf.*' logger"
            assert handler.stream is not None, "a host handler was closed"
        finally:
            logger.removeHandler(handler)
            handler.close()


def test_no_handler_is_left_on_py_warnings():
    warnings_logger = logging.getLogger('py.warnings')
    handlers_before = list(warnings_logger.handlers)
    _run_decorated()
    _run_decorated()
    assert warnings_logger.handlers == handlers_before


def test_capture_warnings_is_restored():
    logging.captureWarnings(True)
    try:
        _run_decorated()
        assert warnings.showwarning is logging._showwarning, "captureWarnings was switched off"
    finally:
        logging.captureWarnings(False)


def test_oasislmf_records_reach_the_log_file_when_the_logger_already_has_a_handler():
    """The engine log file must get records from an 'oasislmf.*' logger that already has a handler."""
    logger = logging.getLogger('oasislmf.some_module_with_handler')
    host_handler = logging.StreamHandler(io.StringIO())
    logger.addHandler(host_handler)
    try:
        with mock.patch.dict(os.environ), tempfile.TemporaryDirectory() as log_dir:
            os.environ.pop('OASIS_PYTEST_REDIRECT_LOGS', None)

            @redirect_logging(exec_name='test_tool', log_dir=log_dir)
            def run_func(**kw):
                logger.info('record from a module logger')
            run_func(logging_level='INFO')

            log_text = ''.join(p.read_text() for p in Path(log_dir).glob('test_tool_*.log'))
        assert 'record from a module logger' in log_text
    finally:
        logger.removeHandler(host_handler)
