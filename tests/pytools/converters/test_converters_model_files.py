
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from oasislmf.pytools.common.input_files import read_returnperiods
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.pla.structure import read_lossfactors
from oasislmf.utils.exceptions import OasisException
from tests.pytools.converters.helpers import case_runner


def test_lossfactors():
    case_runner("bintocsv", "lossfactors", "static")
    case_runner("csvtobin", "lossfactors", "static")


def _small_chunks(stack, file_in, dtype):
    return iter_csv_as_ndarray(stack, file_in, dtype, chunksize=1)


# event_id 5 appears in two non-adjacent runs: a genuinely non-contiguous sequence.
# The runtime reads (event_id, amplification_id) -> factor into a dict (pla/structure.py
# read_lossfactors), so this is valid input regardless of event_id order -- the only
# requirement is that no (event_id, amplification_id) pair is duplicated with a different
# value. ktools' own lossfactorstobin had no sort check either, grouping purely by
# contiguous runs, so this is what "correct" means for this format.
_NON_CONTIGUOUS_CSV = "event_id,amplification_id,factor\n1,1,1.1\n5,1,5.1\n5,2,5.2\n3,1,3.1\n5,3,5.3\n2,1,2.1\n"
_NON_CONTIGUOUS_EXPECTED = {
    (1, 1): pytest.approx(1.1), (2, 1): pytest.approx(2.1), (3, 1): pytest.approx(3.1),
    (5, 1): pytest.approx(5.1), (5, 2): pytest.approx(5.2), (5, 3): pytest.approx(5.3),
}


def test_lossfactors_handles_non_contiguous_event_id_single_chunk():
    # the whole file fits in one chunk, so there is no cross-chunk boundary to get wrong
    with TemporaryDirectory() as tmp:
        Path(tmp, "in.csv").write_text(_NON_CONTIGUOUS_CSV)
        csvtobin(Path(tmp, "in.csv"), Path(tmp, "out.bin"), "lossfactors")
        plafactors = read_lossfactors(run_dir=tmp, ignore_file_type={"csv"}, filename="out.bin")

    assert dict(plafactors) == _NON_CONTIGUOUS_EXPECTED


def test_lossfactors_handles_non_contiguous_event_id_across_chunk_boundary():
    # same input, forced across several single-row chunks: previously the searchsorted-based
    # chunk-boundary carry-over assumed event_ids was sorted and silently dropped/misattributed
    # rows once a partial event continued into a chunk it couldn't find by binary search
    with TemporaryDirectory() as tmp, mock.patch(
            "oasislmf.pytools.converters.csvtobin.utils.lossfactors.iter_csv_as_ndarray", _small_chunks):
        Path(tmp, "in.csv").write_text(_NON_CONTIGUOUS_CSV)
        csvtobin(Path(tmp, "in.csv"), Path(tmp, "out.bin"), "lossfactors")
        plafactors = read_lossfactors(run_dir=tmp, ignore_file_type={"csv"}, filename="out.bin")

    assert dict(plafactors) == _NON_CONTIGUOUS_EXPECTED


def test_lossfactors_accepts_sorted_event_id_across_chunk_boundary():
    csv = "event_id,amplification_id,factor\n1,1,1.1\n2,1,2.1\n3,1,3.1\n5,1,5.1\n5,2,5.2\n5,3,5.3\n"
    with TemporaryDirectory() as tmp, mock.patch(
            "oasislmf.pytools.converters.csvtobin.utils.lossfactors.iter_csv_as_ndarray", _small_chunks):
        Path(tmp, "ok.csv").write_text(csv)
        csvtobin(Path(tmp, "ok.csv"), Path(tmp, "ok.bin"), "lossfactors")
        plafactors = read_lossfactors(run_dir=tmp, ignore_file_type={"csv"}, filename="ok.bin")

    assert dict(plafactors) == {
        (1, 1): pytest.approx(1.1), (2, 1): pytest.approx(2.1), (3, 1): pytest.approx(3.1),
        (5, 1): pytest.approx(5.1), (5, 2): pytest.approx(5.2), (5, 3): pytest.approx(5.3),
    }


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


@pytest.mark.parametrize("period_no, no_date_alg", [(0, False), (-3, False), (0, True)])
def test_occurrence_rejects_period_no_below_one(period_no, no_date_alg):
    """period_no is 1-based: lecpy/aalpy index their period buffers with period_no - 1, so 0 or
    below would silently wrap to the last period's slot instead of raising."""
    with TemporaryDirectory() as tmp:
        if no_date_alg:
            csv = f"event_id,period_no,occ_date_id\n1,{period_no},730000\n"
        else:
            csv = f"event_id,period_no,occ_year,occ_month,occ_day\n1,{period_no},2000,1,1\n"
        Path(tmp, "bad.csv").write_text(csv)
        with pytest.raises(OasisException, match=f"period_no {period_no} is less than 1"):
            csvtobin(Path(tmp, "bad.csv"), Path(tmp, "bad.bin"), "occurrence", no_of_periods=5, no_date_alg=no_date_alg)


def test_occurrence_accepts_valid_period_no():
    with TemporaryDirectory() as tmp:
        Path(tmp, "ok.csv").write_text("event_id,period_no,occ_year,occ_month,occ_day\n1,3,2000,1,1\n")
        csvtobin(Path(tmp, "ok.csv"), Path(tmp, "ok.bin"), "occurrence", no_of_periods=5)


def test_periods():
    case_runner("bintocsv", "periods", "input")
    case_runner("csvtobin", "periods", "input")


def test_quantile():
    case_runner("bintocsv", "quantile", "input")
    case_runner("csvtobin", "quantile", "input")


def test_returnperiods():
    case_runner("bintocsv", "returnperiods", "input")
    case_runner("csvtobin", "returnperiods", "input")


def test_returnperiods_removes_duplicates():
    # e.g. "return_periods": [10, 100, 100] in analysis settings; lecpy rejects duplicates outright
    with TemporaryDirectory() as tmp:
        Path(tmp, "in.csv").write_text("return_period\n10\n100\n100\n")
        csvtobin(Path(tmp, "in.csv"), Path(tmp, "returnperiods.bin"), "returnperiods")
        returnperiods, use_return_period_file = read_returnperiods(True, tmp, filename="returnperiods.bin")

    assert use_return_period_file is True
    assert list(returnperiods) == [100, 10]
