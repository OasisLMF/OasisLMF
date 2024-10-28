import filecmp
import shutil
from tempfile import TemporaryDirectory
import numpy as np
from pathlib import Path
from unittest.mock import patch

from oasislmf.pytools.elt.manager import main, read_quantile_get_intervals, quantile_interval_dtype, read_event_rate_csv
from oasislmf.pytools.common.data import (oasis_int, oasis_float)

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_eltpy")


def case_runner(test_name, with_event_rate=False):
    csv_name = f"py_{test_name}.csv"
    summary_bin_input = Path(TESTS_ASSETS_DIR, "summarypy.bin")
    if with_event_rate:
        expected_csv = Path(TESTS_ASSETS_DIR, f"py_{test_name}_er.csv")
    else:
        expected_csv = Path(TESTS_ASSETS_DIR, csv_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        actual_csv = Path(tmp_result_dir_str, csv_name)

        kwargs = {
            "run_dir": TESTS_ASSETS_DIR,
            "files_in": summary_bin_input,
        }

        if test_name in ["selt", "melt", "qelt"]:
            kwargs[f"{test_name}"] = actual_csv
        else:
            raise Exception(f"Invalid or unimplemented test case {test_name} for eltpy")

        if with_event_rate:
            eids, ers = read_event_rate_csv(Path(TESTS_ASSETS_DIR, "input", "er.csv"))
            with patch('oasislmf.pytools.elt.manager.read_event_rate_csv', return_value=(eids, ers)):
                main(**kwargs)
        else:
            with patch('oasislmf.pytools.elt.manager.read_event_rate_csv', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))):
                main(**kwargs)

        try:
            assert filecmp.cmp(expected_csv, actual_csv, shallow=False)
        except Exception as e:
            error_path = TESTS_ASSETS_DIR.joinpath('error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_csv),
                            Path(error_path, csv_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'eltpy {arg_str}' led to diff, see files at {error_path}") from e


def test_selt_output():
    case_runner("selt")


def test_melt_output():
    case_runner("melt")


def test_melt_output_with_event_rate_csv():
    case_runner("melt", with_event_rate=True)


def test_qelt_output():
    case_runner("qelt")


def test_read_quantile_get_intervals():
    fp = Path(TESTS_ASSETS_DIR, "input", "quantile.bin")
    sample_size = 100

    intervals_actual = read_quantile_get_intervals(sample_size, fp)
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


def test_read_event_rate_csv_missing_file():
    unique_event_ids, event_rates = read_event_rate_csv('nonexistent.csv')
    assert unique_event_ids.size == 0
    assert event_rates.size == 0


def test_read_event_rate_csv():
    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Create a test CSV
        test_csv_file = tmp_dir_path / 'er_test.csv'
        test_csv_content = """firs_col_eid,second_col_er
                            1,0.001
                            23,0.103
                            12,0.052
                            """
        test_csv_file.write_text(test_csv_content)

        unique_event_ids, event_rates = read_event_rate_csv(test_csv_file)

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
