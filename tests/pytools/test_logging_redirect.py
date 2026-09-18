"""The pytools logging redirect must leave global logging exactly as it found it.

``redirect_logging`` decorates every pytools entry point (fmpy, gulpy, gulmc, plapy ...) and
retargets logging at a per-run file. Logging configuration is process-global, so anything it does
not put back leaks into whatever runs next in the same interpreter.

In a model run each tool is its own process, so a leak is invisible there. It is not invisible to
anything driving the tools in-process -- the deterministic ``exposure run`` path, an embedding
caller, or a test session -- where it silently lowered the verbosity of every ``oasislmf.*``
logger for the rest of the process.
"""
import logging
from unittest import TestCase

from oasislmf.pytools.utils import logging_set_handlers, logging_reset_handlers


def _state(name):
    logger = logging.getLogger(name)
    return logger.level, logger.propagate, list(logger.handlers)


class TestLoggingRedirectRestoresState(TestCase):

    LOGGER = 'oasislmf.tests.logging_redirect_probe'

    def tearDown(self):
        logger = logging.getLogger(self.LOGGER)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    def test_a_pristine_logger_is_returned_to_pristine(self):
        before = _state(self.LOGGER)
        previous = logging_set_handlers(self.LOGGER, logging.NullHandler(), logging.WARNING)
        logging_reset_handlers(self.LOGGER, previous)
        self.assertEqual(_state(self.LOGGER), before)

    def test_the_level_is_restored_not_left_at_the_redirect_level(self):
        """The specific regression: the level used to be reset only for non-oasislmf loggers.

        Every ``oasislmf.*`` logger was therefore left at the redirect level (WARNING by default)
        and stopped emitting INFO for the rest of the process, which is what made
        tests/computation's log assertions fail whenever they ran after tests/pytools.
        """
        previous = logging_set_handlers(self.LOGGER, logging.NullHandler(), logging.WARNING)
        self.assertEqual(logging.getLogger(self.LOGGER).level, logging.WARNING)  # during the run
        logging_reset_handlers(self.LOGGER, previous)
        self.assertEqual(logging.getLogger(self.LOGGER).level, logging.NOTSET)

    def test_a_configured_logger_keeps_its_own_configuration(self):
        """Restoring to defaults would be wrong for a caller who set the level deliberately."""
        logger = logging.getLogger(self.LOGGER)
        own_handler = logging.NullHandler()
        logger.addHandler(own_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        previous = logging_set_handlers(self.LOGGER, logging.NullHandler(), logging.WARNING)
        logging_reset_handlers(self.LOGGER, previous)

        self.assertEqual(logger.level, logging.DEBUG)
        self.assertFalse(logger.propagate)
        self.assertEqual(logger.handlers, [own_handler])

    def test_propagate_is_restored_even_with_no_handlers(self):
        """It used to be set inside the handler loop, so a handler-less logger never got it back."""
        previous = logging_set_handlers(self.LOGGER, logging.NullHandler(), logging.WARNING)
        logging.getLogger(self.LOGGER).handlers.clear()
        logging_reset_handlers(self.LOGGER, previous)
        self.assertTrue(logging.getLogger(self.LOGGER).propagate)
