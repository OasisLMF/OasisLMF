"""The property these notices exist for: reaching the user.

Python's default filter set shows a DeprecationWarning only when it is attributed to
``__main__``, and attribution follows ``stacklevel`` rather than where ``warn()`` is written.
Anything raised below the entry point is therefore dropped, silently and with no test failing --
which is how the ``--verbose`` notice went unseen for its whole life. These tests run a real
subprocess with untouched filters, because that is the only place the default set applies.
"""
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

from oasislmf.utils.deprecation import warn_deprecated


def _emits_on_stderr(tmp_path, body, marker="DEPRECATED-MARKER"):
    """Run ``body`` at the depth real code sits at, and report what reached stderr.

    Two module frames between __main__ and the notice, mirroring
    entry point -> computation layer -> resolve_disaggregation. One frame is not enough:
    with stacklevel=2 the warning would then be attributed to __main__ itself, which Python
    shows by default, and every form would look fine.
    """
    (tmp_path / "lib_under_test.py").write_text(textwrap.dedent(body))
    (tmp_path / "caller.py").write_text(
        "import lib_under_test\ndef run():\n    lib_under_test.go()\n")
    (tmp_path / "entry.py").write_text("import caller\ncaller.run()\n")
    proc = subprocess.run([sys.executable, "entry.py"], cwd=tmp_path,
                          capture_output=True, text=True)
    return marker in proc.stderr


def test_a_notice_raised_below_main_reaches_the_user(tmp_path):
    assert _emits_on_stderr(tmp_path, '''
        from oasislmf.utils.deprecation import warn_deprecated
        def go():
            warn_deprecated("DEPRECATED-MARKER")
    '''), "warn_deprecated was swallowed by Python's default filters"


def test_a_plain_warning_in_the_same_place_is_swallowed(tmp_path):
    """The control. Without this, the test above could pass for the wrong reason."""
    assert not _emits_on_stderr(tmp_path, '''
        import warnings
        def go():
            warnings.warn("DEPRECATED-MARKER", DeprecationWarning, stacklevel=2)
    '''), "a plain DeprecationWarning was visible, so the test above proves nothing"


def test_it_does_not_leave_the_process_filters_changed(tmp_path):
    """simplefilter is scoped to the one call; it must not switch warnings on globally."""
    before = list(warnings.filters)
    warn_deprecated("DEPRECATED-MARKER")
    assert list(warnings.filters) == before


def test_stacklevel_blames_the_caller_not_the_helper():
    """stacklevel is counted as if warnings.warn were called by the caller directly."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated("DEPRECATED-MARKER", stacklevel=1)
    assert len(caught) == 1
    assert Path(caught[0].filename).name == Path(__file__).name
