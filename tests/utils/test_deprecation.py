"""The property these notices exist for: reaching the user, on the user's terms.

Python's default filter set shows a DeprecationWarning only when it is attributed to
``__main__``, and attribution follows ``stacklevel`` rather than where ``warn()`` is written.
Anything raised below the entry point was therefore dropped, silently and with no test failing --
which is how the ``--verbose`` notice went unseen for its whole life. The notices are
FutureWarnings for that reason, and they must still obey ``-W``.

These tests run a real subprocess, because that is the only place the default filter set and
any filter oasislmf installs at import time both apply; pytest resets the filters around each
test and would hide either.
"""
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

from oasislmf.utils.deprecation import warn_deprecated

MARKER = "DEPRECATED-MARKER"

NOTICE = f'''
    import oasislmf.manager  # imports every module, so any filter set at import time is in place
    from oasislmf.utils.deprecation import warn_deprecated
    def go():
        warn_deprecated("{MARKER}")
'''


def _run(tmp_path, body, *python_options):
    """Run ``body`` at the depth real code sits at, and return the finished process.

    Two module frames between __main__ and the notice, mirroring
    entry point -> computation layer -> resolve_disaggregation. One frame is not enough:
    with stacklevel=2 the warning would then be attributed to __main__ itself, which Python
    shows by default, and every form would look fine.
    """
    (tmp_path / "lib_under_test.py").write_text(textwrap.dedent(body))
    (tmp_path / "caller.py").write_text(
        "import lib_under_test\ndef run():\n    lib_under_test.go()\n")
    (tmp_path / "entry.py").write_text("import caller\ncaller.run()\n")
    return subprocess.run([sys.executable, *python_options, "entry.py"], cwd=tmp_path,
                          capture_output=True, text=True)


def test_a_notice_raised_below_main_reaches_the_user(tmp_path):
    proc = _run(tmp_path, NOTICE)
    assert proc.returncode == 0, proc.stderr
    assert MARKER in proc.stderr, "the notice was swallowed by the warning filters"


def test_a_plain_deprecation_warning_in_the_same_place_is_swallowed(tmp_path):
    """The control. Without this, the test above could pass for the wrong reason."""
    proc = _run(tmp_path, f'''
        import warnings
        def go():
            warnings.warn("{MARKER}", DeprecationWarning, stacklevel=2)
    ''')
    assert MARKER not in proc.stderr, "a plain DeprecationWarning was visible, so the test above proves nothing"


def test_a_user_can_turn_the_notice_into_an_error(tmp_path):
    """How CI catches a deprecated call before the deprecated thing is removed."""
    proc = _run(tmp_path, NOTICE, "-W", "error::FutureWarning")
    assert proc.returncode != 0
    assert MARKER in proc.stderr


def test_a_user_can_silence_the_notice(tmp_path):
    proc = _run(tmp_path, NOTICE, "-W", "ignore::FutureWarning")
    assert proc.returncode == 0, proc.stderr
    assert MARKER not in proc.stderr


def test_stacklevel_blames_the_caller_not_the_helper():
    """stacklevel is counted as if warnings.warn were called by the caller directly."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated(MARKER, stacklevel=1)
    assert len(caught) == 1
    assert Path(caught[0].filename).name == Path(__file__).name
