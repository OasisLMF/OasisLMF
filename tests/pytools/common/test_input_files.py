import numpy as np
import pytest
from pathlib import Path

from oasislmf.pytools.common.data import oasis_int, oasis_float, read_event_rates, read_occurrence, read_periods, read_quantile

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_common")


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
    quantile_interval_dtype = np.dtype([
        ('q', oasis_float),
        ('integer_part', oasis_int),
        ('fractional_part', oasis_float),
    ])

    intervals_actual = read_quantile(sample_size, run_dir, filename, False)
    intervals_expected = np.zeros(6, dtype=quantile_interval_dtype)
    intervals_expected[0] = (0.0, 1, 0.0)
    intervals_expected[1] = (0.2, 20, 0.8)
    intervals_expected[2] = (0.4, 40, 0.6)
    intervals_expected[3] = (0.5, 50, 0.5)
    intervals_expected[4] = (0.75, 75, 0.25)
    intervals_expected[5] = (1.0, 100, 0.0)

    qs_actual = intervals_actual[:]["q"]
    iparts_actual = intervals_actual[:]["integer_part"]
    fparts_actual = intervals_actual[:]["fractional_part"]

    qs_expected = intervals_expected[:]["q"]
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
    occ_map_dtype = np.dtype([
        ("event_id", np.int32),
        ("period_no", np.int32),
        ("occ_date_id", np.int32),
    ])

    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(run_dir, filename)
    assert isinstance(occ_map, np.ndarray)
    assert occ_map.dtype == occ_map_dtype
    assert date_algorithm == True
    assert granular_date == False
    assert no_of_periods == 9


def test_read_occurrence_granular():
    """Tests read_occurrence granular
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "occurrence_gran.bin"
    occ_map_dtype = np.dtype([
        ("event_id", np.int32),
        ("period_no", np.int32),
        ("occ_date_id", np.int64),
    ])

    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(run_dir, filename)
    assert isinstance(occ_map, np.ndarray)
    assert occ_map.dtype == occ_map_dtype
    assert date_algorithm == True
    assert granular_date == True
    assert no_of_periods == 9


def test_read_periods():
    """Tests read_periods from existing binary file
    """
    run_dir = Path(TESTS_ASSETS_DIR, "input")
    filename = "periods.bin"
    no_of_periods = 5
    period_weights_dtype = np.dtype([
        ("period_no", np.int32),
        ("weighting", np.float64),
    ])

    periods_expected = np.array(
        [(1, 0.15), (2, 0.05), (3, 0.4), (4, 0.1), (5, 0.3)],
        dtype=period_weights_dtype
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

    period_weights_dtype = np.dtype([
        ("period_no", np.int32),
        ("weighting", np.float64),
    ])

    periods_expected = np.array(
        [(1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2), (5, 0.2)],
        dtype=period_weights_dtype
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
