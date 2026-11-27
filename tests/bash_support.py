"""Whether the tests may run bash kernel scripts and pytools pipelines.

Tests marked ``requires_bash`` (and those using a fixture in ``REQUIRES_BASH_FIXTURES``) run bash scripts, or
pipelines of pytools connected by OS pipes, which the tools wait on with ``select``: neither works on Windows.
On macOS a tool whose output is a regular file fails: ``write_mv_to_stream`` selects on its output for
exceptional conditions, and macOS reports one on every regular file (``OSError: error with input stream``).
They are skipped on both, and anywhere ``OASIS_TEST_NO_BASH`` is set, which runs the Windows selection elsewhere.
Tests that compare with bash only when it can run check ``bash_available()`` before asking for a bash fixture.
"""
import os
import shutil
import sys

NO_BASH_ENV = 'OASIS_TEST_NO_BASH'
REQUIRES_BASH_FIXTURES = frozenset({'kernel_reference', 'full_kernel_reference', 'bash_whole_run'})
SKIP_REASON = (f'needs bash and POSIX pipes between the pytools (not on Windows; not on macOS, where a pytool writing to a '
               f'regular file fails in select; nor with {NO_BASH_ENV} set)')


def bash_available():
    """True when bash kernel scripts and pytools pipelines can run here."""
    return sys.platform not in ('win32', 'darwin') and not os.environ.get(NO_BASH_ENV) and shutil.which('bash') is not None
