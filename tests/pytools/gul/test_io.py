import select

from oasislmf.pytools.gul import io as gul_io
from tests.pytools.converters.helpers import TESTS_ASSETS_DIR, case_runner


def select_like_windows(*args, **kwargs):
    raise OSError(10038, "An operation was attempted on something that is not a socket")


def test_read_getmodel_stream_from_regular_file_without_select(monkeypatch):
    """On Windows select accepts only sockets, so reading the getmodel stream from a regular file must not call it."""
    monkeypatch.setattr(select, "select", select_like_windows)
    monkeypatch.setattr(gul_io, "select", select_like_windows, raising=False)
    case_runner("bintocsv", "cdf", "cdftocsv", "getmodel", run_dir=TESTS_ASSETS_DIR / "cdftocsv")
