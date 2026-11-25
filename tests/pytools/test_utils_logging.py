"""redirect_logging must leave the process's logging configuration as it found it."""
import logging
import os
import tempfile
from pathlib import Path

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


def test_engine_run_leaves_a_configured_oasislmf_logger_as_it_was(tmp_path):
    """A caller that configured an 'oasislmf.*' logger itself gets that configuration back, not
    the defaults: the run puts back what it found."""
    logger = logging.getLogger('oasislmf.tests.configured_by_the_caller')
    own_handler = logging.NullHandler()
    logger.addHandler(own_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        redirect_logging(exec_name='probe', log_dir=str(tmp_path))(lambda: None)()
        assert (logger.level, logger.propagate, logger.handlers) == (logging.DEBUG, False, [own_handler])
    finally:
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
