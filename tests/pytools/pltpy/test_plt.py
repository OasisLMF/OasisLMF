import filecmp
import shutil
from tempfile import TemporaryDirectory
import numpy as np
from pathlib import Path

from oasislmf.pytools.plt.manager import main, read_occurrence, read_periods
import pytest

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_pltpy")


def case_runner(sub_folder, test_name):
    """Run output file correctness tests

    Args:
        sub_folder (str | os.PathLike): path to input files root
        test_name (str): test name
    """
    csv_name = f"py_{test_name}.csv"
    summary_bin_input = Path(TESTS_ASSETS_DIR, "summarypy.bin")
    expected_csv = Path(TESTS_ASSETS_DIR, sub_folder, csv_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        actual_csv = Path(tmp_result_dir_str, csv_name)

        kwargs = {
            "run_dir": Path(TESTS_ASSETS_DIR, sub_folder),
            "files_in": summary_bin_input,
        }

        if test_name in ["splt", "mplt", "qplt"]:
            kwargs[f"{test_name}"] = actual_csv
        else:
            raise Exception(f"Invalid or unimplemented test case {test_name} for pltpy")

        main(**kwargs)

        try:
            assert filecmp.cmp(expected_csv, actual_csv, shallow=False)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, sub_folder, 'error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_csv),
                            Path(error_path, csv_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'pltpy {arg_str}' led to diff, see files at {error_path}") from e


def test_splt_output():
    """Tests splt outputs
    """
    case_runner("all_files", "splt")
    case_runner("no_files", "splt")
    case_runner("occ_gran_files", "splt")


def test_mplt_output():
    """Tests mplt outputs
    """
    case_runner("all_files", "mplt")
    case_runner("no_files", "mplt")
    case_runner("occ_gran_files", "mplt")


def test_qplt_output():
    """Tests qplt outputs
    """
    # case_runner("qplt")
    pass


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
    assert occ_map.dtype == occ_map.dtype
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
    assert occ_map.dtype == occ_map.dtype
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
