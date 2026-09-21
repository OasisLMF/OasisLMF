import contextlib
import glob
import logging
import os
import tempfile
from unittest import TestCase, mock

from oasislmf.pytools.utils import redirect_logging


@contextlib.contextmanager
def _real_log_dir():
    """The session-wide conftest fixture forces every log through a session
    tmpdir that gets rmtree'd on each call's exit (OASIS_PYTEST_REDIRECT_LOGS).
    Tests that need to inspect the log file `redirect_logging` writes to
    `log_dir` itself must suspend that redirection first."""
    with mock.patch.dict(os.environ):
        os.environ.pop('OASIS_PYTEST_REDIRECT_LOGS', None)
        os.environ.pop('OASIS_TMPDIR', None)
        yield


class TestRedirectLogging(TestCase):
    def _run_decorated(self, log_dir, **kwargs):
        @redirect_logging(exec_name='test_tool', log_dir=log_dir)
        def run_func(**kw):
            return 'ok'
        return run_func(**kwargs)

    def test_writes_log_file_with_named_level(self):
        with _real_log_dir(), tempfile.TemporaryDirectory() as tmp_dir:
            result = self._run_decorated(tmp_dir, logging_level='DEBUG')
            self.assertEqual(result, 'ok')
            log_files = glob.glob(os.path.join(tmp_dir, 'test_tool_*.log'))
            self.assertEqual(len(log_files), 1)
            with open(log_files[0]) as f:
                content = f.read()
            self.assertIn('starting process', content)
            self.assertIn('finishing process', content)

    def test_invalid_named_level_falls_back_to_warning(self):
        levels_set = []
        real_set_level = logging.FileHandler.setLevel

        def spy_set_level(self, level):
            levels_set.append(level)
            return real_set_level(self, level)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(logging.FileHandler, 'setLevel', spy_set_level):
                self._run_decorated(tmp_dir, logging_level='NOT_A_REAL_LEVEL')

        # both the per-process and root file handlers get the fallback level
        self.assertIn(logging.WARNING, levels_set)

    def test_digit_string_level_is_used_directly(self):
        levels_set = []
        real_set_level = logging.FileHandler.setLevel

        def spy_set_level(self, level):
            levels_set.append(level)
            return real_set_level(self, level)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(logging.FileHandler, 'setLevel', spy_set_level):
                self._run_decorated(tmp_dir, logging_level=str(logging.DEBUG))

        self.assertIn(logging.DEBUG, levels_set)

    def test_oasis_pytools_log_dir_env_var_overrides_log_dir(self):
        with _real_log_dir(), \
                tempfile.TemporaryDirectory() as configured_dir, \
                tempfile.TemporaryDirectory() as override_dir:
            with mock.patch.dict(os.environ, {'OASIS_PYTOOLS_LOG_DIR': override_dir}):
                self._run_decorated(configured_dir)

            self.assertEqual(glob.glob(os.path.join(configured_dir, '*.log')), [])
            self.assertEqual(len(glob.glob(os.path.join(override_dir, '*.log'))), 1)

    def test_exception_in_wrapped_function_is_logged_and_reraised(self):
        with _real_log_dir(), tempfile.TemporaryDirectory() as tmp_dir:
            @redirect_logging(exec_name='test_tool', log_dir=tmp_dir)
            def failing_func():
                raise ValueError('boom')

            with self.assertRaises(ValueError):
                failing_func()

            log_files = glob.glob(os.path.join(tmp_dir, 'test_tool_*.log'))
            self.assertEqual(len(log_files), 1)
            with open(log_files[0]) as f:
                content = f.read()
            self.assertIn('boom', content)
            self.assertNotIn('finishing process', content)
