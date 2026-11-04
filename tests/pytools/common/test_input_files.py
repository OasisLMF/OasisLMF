from tempfile import TemporaryDirectory
import numpy as np
import pytest
from pathlib import Path

from oasislmf.pytools.common.data import (
    oasis_int, oasis_float, coverages_dtype, COVERAGE_BUILDINGS_FILE,
    coverage_buildings_dtype, coverage_buildings_headers,
    correlations_dtype,
    periods_dtype, quantile_interval_dtype, returnperiods_dtype
)
from oasislmf.utils.exceptions import OasisException
from oasislmf.pytools.common.id_index import get_idx as id_index_get_idx, NOT_FOUND as OCC_IDX_NOT_FOUND
from oasislmf.pytools.common.input_files import (
    read_amplifications,
    read_coverage_buildings,
    read_coverages,
    read_correlations,
    read_event_rates,
    read_occurrence,
    read_occurrence_bin,
    read_periods,
    read_quantile,
    read_returnperiods,
)

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_common")


def write_items_amplifications_file(n_items, itemsamps_file, formula):
    """
    Class method to write items amplifications files, which are used in the
    tests.

    Args:
        itemsamps_file (str): items amplications file name
        n_items (int): number of unique item IDs
        formula (str): expression to evaluate when filling file
    """
    data_size = 8
    write_buffer = memoryview(bytearray(n_items * data_size))
    item_amp_dtype = np.dtype([
        ('item_id', 'i4'), ('amplification_id', 'i4')
    ])
    event_item = np.ndarray(
        n_items, buffer=write_buffer, dtype=item_amp_dtype
    )
    it = np.nditer(event_item, op_flags=['writeonly'], flags=['c_index'])
    for row in it:
        row[...] = (eval('it.index' + formula), eval('it.index' + formula))
    with open(itemsamps_file, 'wb') as f:
        f.write(np.int32(0).tobytes())   # Empty header
        f.write(write_buffer[:])


def test_structure__read_amplifications__first_item_id_not_1():
    """
    Test read_amplifications() raises SystemExit if the
    first item ID is not 1.
    """
    # Write items amplifications file with first item ID = 2
    with TemporaryDirectory() as tmpdir:
        itemsamps_file = Path(tmpdir, "amplifications.bin")
        write_items_amplifications_file(2, itemsamps_file, formula='+ 2')

        with pytest.raises(ValueError) as e:
            read_amplifications(tmpdir)


def test_structure__read_amplifications__non_contiguous_item_ids():
    """
    Test read_amplifications() raises SystemExit if the
    item IDs are not contiguous.
    """
    # Write items amplfications file where difference between item IDs is
    # not 1
    with TemporaryDirectory() as tmpdir:
        itemsamps_file = Path(tmpdir, "amplifications.bin")
        write_items_amplifications_file(
            4, itemsamps_file, formula='* 2 + 1'
        )

        with pytest.raises(ValueError) as e:
            read_amplifications(tmpdir)


def test_structure__read_amplifications__no_amplifications_file():
    """
    Test read_amplifications() raises SystemExit if the
    amplifications.bin file does not exist.
    """
    with pytest.raises(FileNotFoundError) as e:
        read_amplifications('.')


def test_read_correlations():
    """Tests read_correlations
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "correlations.csv"

    # correlations.csv carries source_item_id; the reader requires every column, so a legacy
    # 5-column file is rejected rather than upgraded
    correlations_expected = np.array([
        (1, 1, 0.700000, 123451, 0.000000, 0),
        (2, 2, 0.500000, 123451, 0.300000, 0),
        (3, 1, 0.700000, 123452, 0.000000, 0),
        (4, 2, 0.500000, 123452, 0.300000, 0),
    ], dtype=correlations_dtype)
    correlations_actual = read_correlations(run_dir, filename=filename)

    source_item_id_expected = correlations_expected["source_item_id"]
    source_item_id_actual = correlations_actual["source_item_id"]
    np.testing.assert_array_equal(source_item_id_expected, source_item_id_actual)

    item_id_expected = correlations_expected["item_id"]
    peril_correlation_group_expected = correlations_expected["peril_correlation_group"]
    damage_correlation_value_expected = correlations_expected["damage_correlation_value"]
    hazard_group_id_expected = correlations_expected["hazard_group_id"]
    hazard_correlation_value_expected = correlations_expected["hazard_correlation_value"]

    item_id_actual = correlations_actual["item_id"]
    peril_correlation_group_actual = correlations_actual["peril_correlation_group"]
    damage_correlation_value_actual = correlations_actual["damage_correlation_value"]
    hazard_group_id_actual = correlations_actual["hazard_group_id"]
    hazard_correlation_value_actual = correlations_actual["hazard_correlation_value"]

    np.testing.assert_array_almost_equal(item_id_expected, item_id_actual, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(peril_correlation_group_expected, peril_correlation_group_actual, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(damage_correlation_value_expected, damage_correlation_value_actual, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(hazard_group_id_expected, hazard_group_id_actual, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(hazard_correlation_value_expected, hazard_correlation_value_actual, decimal=3, verbose=True)


def _correlations(item_ids, damage_correlation_value=0.5):
    correlations = np.zeros(len(item_ids), dtype=correlations_dtype)
    correlations["item_id"] = item_ids
    correlations["peril_correlation_group"] = 1
    correlations["damage_correlation_value"] = damage_correlation_value
    correlations["hazard_group_id"] = 2
    correlations["hazard_correlation_value"] = 0.25
    return correlations


@pytest.mark.parametrize("item_ids", [range(1, 11), [5, 6, 7], [1, 4, 9], [100, 200, 300]],
                         ids=["dense", "offset", "gapped", "sparse"])
def test_read_correlations_bin__any_item_ids_are_read(item_ids):
    """Only the record layout is checked: a table is valid whatever ids its items carry."""
    with TemporaryDirectory() as d:
        _correlations(list(item_ids)).tofile(Path(d, "correlations.bin"))
        actual = read_correlations(d)
    np.testing.assert_array_equal(actual["item_id"], list(item_ids))
    np.testing.assert_allclose(actual["damage_correlation_value"], 0.5)


# the record layout before source_item_id was added: 20 bytes against today's 24
PRE_SOURCE_ITEM_DTYPE = np.dtype([("item_id", "<i4"), ("peril_correlation_group", "<i4"),
                                  ("damage_correlation_value", "<f4"), ("hazard_group_id", "<i4"),
                                  ("hazard_correlation_value", "<f4")])


def _older_layout(n):
    old = np.zeros(n, dtype=PRE_SOURCE_ITEM_DTYPE)
    old["item_id"] = np.arange(1, n + 1)
    old["peril_correlation_group"] = 1
    old["damage_correlation_value"] = 0.5
    old["hazard_group_id"] = 2
    old["hazard_correlation_value"] = 0.5
    return old.tobytes()


@pytest.mark.parametrize("content", [
    b"\x01" * (correlations_dtype.itemsize + 5),
    # 12 records of 20 bytes divide by 24 too, so the file maps cleanly and only the values betray it
    _older_layout(12),
    _correlations([1, 2, 3], damage_correlation_value=[0.5, 4.2, 0.1]).tobytes(),
], ids=["partial_record", "older_layout_that_divides_evenly", "correlation_out_of_range"])
def test_read_correlations_bin__a_file_of_another_layout_is_rejected(content):
    """Rather than read other fields' bytes as the current ones."""
    with TemporaryDirectory() as d:
        Path(d, "correlations.bin").write_bytes(content)
        with pytest.raises(OasisException, match="does not match the current correlations record layout"):
            read_correlations(d)


def test_read_correlations__empty_bin():
    """An empty correlations.bin (no correlations) must load without a correlations.csv"""
    with TemporaryDirectory() as tmp:
        Path(tmp, "correlations.bin").touch()
        correlations = read_correlations(tmp)
        assert correlations.dtype == correlations_dtype
        assert len(correlations) == 0


def test_read_coverages():
    """Tests read_coverages
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "coverages.csv"

    coverages_expected = np.array(
        [(1, 100.15), (2, 200.05), (3, 300.4), (4, 400.1), (5, 500.3)],
        dtype=coverages_dtype
    )
    coverages_actual = read_coverages(run_dir, filename=filename)

    np.testing.assert_array_almost_equal(coverages_expected["tiv"], coverages_actual, decimal=3, verbose=True)


def test_read_event_rates():
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "event_rates.csv"

    unique_event_ids, event_rates = read_event_rates(run_dir, filename)

    assert unique_event_ids.size == 3, f"Expected 3 event IDs, got {unique_event_ids.size}"
    assert event_rates.size == 3, f"Expected 3 event rates, got {event_rates.size}"
    assert unique_event_ids.dtype == oasis_int, f"Expected event IDs dtype {oasis_int}, got {unique_event_ids.dtype}"
    assert event_rates.dtype == oasis_float, f"Expected event rates dtype {oasis_float}, got {event_rates.dtype}"

    # Expected values
    expected_event_ids = np.array([1, 12, 23], dtype=oasis_int)
    expected_event_rates = np.array([0.001, 0.052, 0.103], dtype=oasis_float)

    # Assert that the arrays match expected values
    np.testing.assert_array_equal(unique_event_ids, expected_event_ids)
    np.testing.assert_array_almost_equal(event_rates, expected_event_rates, decimal=6)


def test_read_quantile_get_intervals():
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "quantile.bin"
    sample_size = 100

    intervals_actual = read_quantile(sample_size, run_dir, filename, False)
    intervals_expected = np.zeros(6, dtype=quantile_interval_dtype)
    intervals_expected[0] = (0.0, 1, 0.0)
    intervals_expected[1] = (0.2, 20, 0.8)
    intervals_expected[2] = (0.4, 40, 0.6)
    intervals_expected[3] = (0.5, 50, 0.5)
    intervals_expected[4] = (0.75, 75, 0.25)
    intervals_expected[5] = (1.0, 100, 0.0)

    qs_actual = intervals_actual[:]["quantile"]
    iparts_actual = intervals_actual[:]["integer_part"]
    fparts_actual = intervals_actual[:]["fractional_part"]

    qs_expected = intervals_expected[:]["quantile"]
    iparts_expected = intervals_expected[:]["integer_part"]
    fparts_expected = intervals_expected[:]["fractional_part"]

    np.testing.assert_array_almost_equal(qs_actual, qs_expected, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(iparts_actual, iparts_expected, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(fparts_actual, fparts_expected, decimal=3, verbose=True)


def test_read_occurrence():
    """Tests read_occurrence non-granular: correct metadata and CSR structure."""
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "occurrence.bin"

    occ_csr, date_algorithm, granular_date, no_of_periods = read_occurrence(run_dir, filename)
    assert date_algorithm == True
    assert granular_date == False
    assert no_of_periods == 9

    occ_arr, _, _, _ = read_occurrence_bin(run_dir, filename)
    for event_id in np.unique(occ_arr["event_id"]):
        i = id_index_get_idx(occ_csr.event_id_index, np.int32(event_id))
        assert i != OCC_IDX_NOT_FOUND, f"event_id {event_id} not found in id_index"
        csr_slice = occ_csr.occ_flat[occ_csr.occ_offsets[i]:occ_csr.occ_offsets[i + 1]]
        expected = occ_arr[occ_arr["event_id"] == event_id]["period_no"]
        np.testing.assert_array_equal(np.sort(csr_slice["period_no"]), np.sort(expected))


def test_read_occurrence_granular():
    """Tests read_occurrence granular: correct metadata and CSR structure."""
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "occurrence_gran.bin"

    occ_csr, date_algorithm, granular_date, no_of_periods = read_occurrence(run_dir, filename)
    assert date_algorithm == True
    assert granular_date == True
    assert no_of_periods == 9

    occ_arr, _, _, _ = read_occurrence_bin(run_dir, filename)
    for event_id in np.unique(occ_arr["event_id"]):
        i = id_index_get_idx(occ_csr.event_id_index, np.int32(event_id))
        assert i != OCC_IDX_NOT_FOUND, f"event_id {event_id} not found in id_index"
        csr_slice = occ_csr.occ_flat[occ_csr.occ_offsets[i]:occ_csr.occ_offsets[i + 1]]
        expected = occ_arr[occ_arr["event_id"] == event_id]["period_no"]
        np.testing.assert_array_equal(np.sort(csr_slice["period_no"]), np.sort(expected))


def test_read_periods():
    """Tests read_periods from existing binary file
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "periods.bin"
    no_of_periods = 5

    periods_expected = np.array(
        [(1, 0.15), (2, 0.05), (3, 0.4), (4, 0.1), (5, 0.3)],
        dtype=periods_dtype
    )
    periods_actual = read_periods(no_of_periods, run_dir, filename)

    period_no_expected = periods_expected[:]["period_no"]
    weighting_expected = periods_expected[:]["weighting"]
    period_no_actual = periods_actual[:]["period_no"]
    weighting_actual = periods_actual[:]["weighting"]

    np.testing.assert_array_almost_equal(period_no_expected, period_no_actual, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(weighting_expected, weighting_actual, decimal=3, verbose=True)


def test_read_periods_no_file():
    """Tests read_periods with missing periods file, generates weights reciprocal of period_no
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "periods_doesnotexist.bin"
    no_of_periods = 5

    periods_expected = np.array(
        [(1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2), (5, 0.2)],
        dtype=periods_dtype
    )
    periods_actual = read_periods(no_of_periods, run_dir, filename)

    period_no_expected = periods_expected[:]["period_no"]
    weighting_expected = periods_expected[:]["weighting"]
    period_no_actual = periods_actual[:]["period_no"]
    weighting_actual = periods_actual[:]["weighting"]

    np.testing.assert_array_almost_equal(period_no_expected, period_no_actual, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(weighting_expected, weighting_actual, decimal=3, verbose=True)


def test_read_periods_missing_period():
    """Tests read_periods with missing period_no in original binary file
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "periods_missing.bin"
    no_of_periods = 4

    with pytest.raises(RuntimeError, match="Missing period_no"):
        read_periods(no_of_periods, run_dir, filename)


def test_read_periods_wrong_period_no():
    """Tests read_periods with incorrect period_no compared to binary file
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "periods.bin"
    no_of_periods = 5

    with pytest.raises(RuntimeError, match="no_of_periods does not match total period_no"):
        read_periods(no_of_periods - 1, run_dir, filename)

    with pytest.raises(RuntimeError, match="no_of_periods does not match total period_no"):
        read_periods(no_of_periods + 1, run_dir, filename)


def test_read_return_periods():
    """Tests read_returnperiods from existing binary file
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "returnperiods.bin"
    use_return_periods = True

    returnperiods_expected = np.array(
        [5000, 1000, 500, 250, 200, 150, 100, 75, 50, 30, 25, 20, 10, 5, 2],
        dtype=returnperiods_dtype
    )["return_period"]
    returnperiods_actual, _ = read_returnperiods(use_return_periods, run_dir, filename)

    np.testing.assert_array_equal(returnperiods_expected, returnperiods_actual, verbose=True)


def test_read_return_periods_no_file():
    """Tests read_returnperiods with missing returnperiods file
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "returnperiods_notexists.bin"
    use_return_periods = True

    with pytest.raises(RuntimeError, match="ERROR: Return Periods file not found at"):
        read_returnperiods(use_return_periods, run_dir, filename)


# --- the per-coverage building count, in its own file --------------------------------------------

def test_read_coverage_buildings__absent_is_the_ordinary_case():
    """Every input set generated before building packing has no such file, and so does every
    unpacked run. The caller takes an empty result as all-ones rather than failing."""
    with TemporaryDirectory() as d:
        assert len(read_coverage_buildings(d)) == 0


@pytest.mark.parametrize("file_type", ["bin", "csv", "csv_without_header"])
def test_read_coverage_buildings__reads_a_signed_count(file_type):
    """Negative keeps the buildings separate through the financial module."""
    counts = [1, -630510, 12]
    with TemporaryDirectory() as d:
        if file_type == "bin":
            np.array(list(enumerate(counts, start=1)), dtype=coverage_buildings_dtype).tofile(
                Path(d, f"{COVERAGE_BUILDINGS_FILE}.bin"))
        else:
            lines = [f"{i},{n}" for i, n in enumerate(counts, start=1)]
            if file_type == "csv":
                lines.insert(0, ",".join(coverage_buildings_headers))
            Path(d, f"{COVERAGE_BUILDINGS_FILE}.csv").write_text("\n".join(lines) + "\n")
        actual = read_coverage_buildings(d, ignore_file_type={"csv", "bin"} - {file_type[:3]})
    np.testing.assert_array_equal(actual["coverage_id"], [1, 2, 3])
    np.testing.assert_array_equal(actual["n_building"], counts)


def _coverage_buildings(ids):
    return np.array([(i, 2) for i in ids], dtype=coverage_buildings_dtype).tobytes()


@pytest.mark.parametrize("content, message", [
    (b"\x01" * (coverage_buildings_dtype.itemsize + 3), "not a multiple of"),
    # coverage_id is stored, unlike in coverages.bin, so a truncated or concatenated file shows
    (_coverage_buildings([1, 2, 4]), "dense"),
    (_coverage_buildings([2, 3, 4]), "dense"),
    (_coverage_buildings([1, 1, 2]), "dense"),
], ids=["partial_record", "gap", "offset", "repeat"])
def test_read_coverage_buildings__a_damaged_file_is_rejected(content, message):
    """Rather than shift every coverage after the damage onto another's count."""
    with TemporaryDirectory() as d:
        Path(d, f"{COVERAGE_BUILDINGS_FILE}.bin").write_bytes(content)
        with pytest.raises(OasisException, match=message):
            read_coverage_buildings(d, ignore_file_type={"csv"})
