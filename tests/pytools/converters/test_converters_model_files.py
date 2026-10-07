
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.utils.exceptions import OasisException
from tests.pytools.converters.helpers import case_runner


def test_lossfactors():
    case_runner("bintocsv", "lossfactors", "static")
    case_runner("csvtobin", "lossfactors", "static")


def test_random():
    case_runner("bintocsv", "random", "static")
    case_runner("csvtobin", "random", "static")


def test_weights():
    case_runner("bintocsv", "weights", "static")
    case_runner("csvtobin", "weights", "static")


def test_amplifications():
    case_runner("bintocsv", "amplifications", "input")
    case_runner("csvtobin", "amplifications", "input")


def test_correlations_items():
    case_runner("bintocsv", "correlations", "input")
    case_runner("csvtobin", "correlations", "input")


def test_complex_items():
    case_runner("bintocsv", "complex_items", "input")
    case_runner("csvtobin", "complex_items", "input")


def test_coverages():
    case_runner("bintocsv", "coverages", "input")
    case_runner("csvtobin", "coverages", "input")


@pytest.mark.parametrize("csv, match", [
    # unordered: TIVs would otherwise be written by position, silently landing against the wrong
    # coverage_id (coverage.bin has no id column — the engine looks up coverage n at index n - 1)
    ("coverage_id,tiv\n2,200\n1,100\n4,400\n", "coverage_id 2 at row 1 is not contiguous; expected 1"),
    ("coverage_id,tiv\n1,100\n3,300\n", "coverage_id 3 at row 2 is not contiguous; expected 2"),  # gap
    ("coverage_id,tiv\n0,100\n", "coverage_id 0 at row 1 is not contiguous; expected 1"),  # must start at 1
])
def test_coverages_rejects_non_contiguous_ids(csv, match):
    with TemporaryDirectory() as tmp:
        Path(tmp, "bad.csv").write_text(csv)
        with pytest.raises(OasisException, match=match):
            csvtobin(Path(tmp, "bad.csv"), Path(tmp, "bad.bin"), "coverages")


def test_coverages_rejects_non_contiguous_ids_across_chunk_boundary():
    rows = [f"{i},{i * 10}" for i in range(1, 8)]
    rows[5] = "999,999"  # break contiguity mid-file, past the first chunk when buffer size is small
    csv = "coverage_id,tiv\n" + "\n".join(rows) + "\n"
    with TemporaryDirectory() as tmp, mock.patch("oasislmf.pytools.converters.csvtobin.utils.coverages.DEFAULT_BUFFER_SIZE", 3):
        Path(tmp, "bad.csv").write_text(csv)
        with pytest.raises(OasisException, match="coverage_id 999 at row 6 is not contiguous; expected 6"):
            csvtobin(Path(tmp, "bad.csv"), Path(tmp, "bad.bin"), "coverages")


def test_eve():
    case_runner("bintocsv", "eve", "input")
    case_runner("csvtobin", "eve", "input")


def test_items():
    case_runner("bintocsv", "items", "input")
    case_runner("csvtobin", "items", "input")


def test_occurrence():
    case_runner("bintocsv", "occurrence", "input", "occurrence")
    case_runner("bintocsv", "occurrence", "input", "occurrence_gran")
    case_runner("bintocsv", "occurrence", "input", "occurrence_noalg")
    case_runner("csvtobin", "occurrence", "input", "occurrence", no_of_periods=9)
    case_runner("csvtobin", "occurrence", "input", "occurrence_gran", no_of_periods=9, granular=True)
    case_runner("csvtobin", "occurrence", "input", "occurrence_noalg", no_of_periods=9, no_date_alg=True)


def test_periods():
    case_runner("bintocsv", "periods", "input")
    case_runner("csvtobin", "periods", "input")


def test_quantile():
    case_runner("bintocsv", "quantile", "input")
    case_runner("csvtobin", "quantile", "input")


def test_returnperiods():
    case_runner("bintocsv", "returnperiods", "input")
    case_runner("csvtobin", "returnperiods", "input")
