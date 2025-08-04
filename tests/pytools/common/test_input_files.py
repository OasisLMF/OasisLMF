from tempfile import TemporaryDirectory
import numpy as np
import pytest
from pathlib import Path

from oasislmf.pytools.common.data import (
    oasis_int, oasis_float, coverages_dtype, correlations_dtype, periods_dtype,
    quantile_interval_dtype, returnperiods_dtype
)
from oasislmf.pytools.common.input_files import (
    read_amplifications,
    read_coverages,
    read_correlations,
    read_event_rates,
    read_occurrence,
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

    correlations_expected = np.array([
        (1, 1, 0.700000, 123451, 0.000000),
        (2, 2, 0.500000, 123451, 0.300000),
        (3, 1, 0.700000, 123452, 0.000000),
        (4, 2, 0.500000, 123452, 0.300000),
    ], dtype=correlations_dtype)
    correlations_actual = read_correlations(run_dir, filename=filename)

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
    """Tests read_occurrence non granular
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "occurrence.bin"

    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(run_dir, filename)
    assert date_algorithm == True
    assert granular_date == False
    assert no_of_periods == 9


def test_read_occurrence_granular():
    """Tests read_occurrence granular
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "occurrence_gran.bin"

    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(run_dir, filename)
    assert date_algorithm == True
    assert granular_date == True
    assert no_of_periods == 9


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
