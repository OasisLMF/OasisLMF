
# import filecmp
import numpy as np
import pytest
# import shutil
from pathlib import Path
# from tempfile import TemporaryDirectory

# from oasislmf.pytools.common.data import (oasis_int, oasis_float)
from oasislmf.pytools.aal.manager import read_occurrence, read_periods

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_aalpy")


def test_read_occurrence():
    """Tests read_occurrence non granular
    """
    occurrence_fp = Path(TESTS_ASSETS_DIR, "input_file_tests", "occurrence.bin")
    occ_map_dtype = np.dtype([
        ("event_id", np.int32),
        ("period_no", np.int32),
        ("occ_date_id", np.int32),
    ])

    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(occurrence_fp)
    assert isinstance(occ_map, np.ndarray)
    assert occ_map.dtype == occ_map_dtype
    assert date_algorithm == True
    assert granular_date == False
    assert no_of_periods == 9


def test_read_occurrence_granular():
    """Tests read_occurrence granular
    """
    occurrence_fp = Path(TESTS_ASSETS_DIR, "input_file_tests", "occurrence_gran.bin")
    occ_map_dtype = np.dtype([
        ("event_id", np.int32),
        ("period_no", np.int32),
        ("occ_date_id", np.int64),
    ])

    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(occurrence_fp)
    assert isinstance(occ_map, np.ndarray)
    assert occ_map.dtype == occ_map_dtype
    assert date_algorithm == True
    assert granular_date == True
    assert no_of_periods == 9


def test_read_periods():
    """Tests read_periods from existing binary file
    """
    periods_fp = Path(TESTS_ASSETS_DIR, "input_file_tests", "periods.bin")
    no_of_periods = 5
    period_weights_dtype = np.dtype([
        ("period_no", np.int32),
        ("weighting", np.float64),
    ])

    periods_expected = np.array(
        [(1, 0.15), (2, 0.05), (3, 0.4), (4, 0.1), (5, 0.3)],
        dtype=period_weights_dtype
    )
    periods_actual = read_periods(periods_fp, no_of_periods)

    period_no_expected = periods_expected[:]["period_no"]
    weighting_expected = periods_expected[:]["weighting"]
    period_no_actual = periods_actual[:]["period_no"]
    weighting_actual = periods_actual[:]["weighting"]

    np.testing.assert_array_almost_equal(period_no_expected, period_no_actual, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(weighting_expected, weighting_actual, decimal=3, verbose=True)


def test_read_periods_no_file():
    """Tests read_periods with missing periods file, generates weights reciprocal of period_no
    """
    periods_fp = Path(TESTS_ASSETS_DIR, "input_file_tests", "periods_doesnotexist.bin")
    no_of_periods = 5

    period_weights_dtype = np.dtype([
        ("period_no", np.int32),
        ("weighting", np.float64),
    ])

    periods_expected = np.array(
        [(1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2), (5, 0.2)],
        dtype=period_weights_dtype
    )
    periods_actual = read_periods(periods_fp, no_of_periods)

    period_no_expected = periods_expected[:]["period_no"]
    weighting_expected = periods_expected[:]["weighting"]
    period_no_actual = periods_actual[:]["period_no"]
    weighting_actual = periods_actual[:]["weighting"]

    np.testing.assert_array_almost_equal(period_no_expected, period_no_actual, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(weighting_expected, weighting_actual, decimal=3, verbose=True)


def test_read_periods_missing_period():
    """Tests read_periods with missing period_no in original binary file
    """
    periods_fp = Path(TESTS_ASSETS_DIR, "input_file_tests", "periods_missing.bin")
    no_of_periods = 5

    with pytest.raises(RuntimeError, match="Missing period_no"):
        read_periods(periods_fp, no_of_periods)


def test_read_periods_wrong_period_no():
    """Tests read_periods with incorrect period_no compared to binary file
    """
    periods_fp = Path(TESTS_ASSETS_DIR, "input_file_tests", "periods.bin")
    no_of_periods = 5

    with pytest.raises(RuntimeError, match="no_of_periods does not match total period_no"):
        read_periods(periods_fp, no_of_periods - 1)

    with pytest.raises(RuntimeError, match="no_of_periods does not match total period_no"):
        read_periods(periods_fp, no_of_periods + 1)
