import numpy as np
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.plt.manager import main

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
            expected_csv_data = np.genfromtxt(expected_csv, delimiter=',', skip_header=1)
            actual_csv_data = np.genfromtxt(actual_csv, delimiter=',', skip_header=1)
            if expected_csv_data.shape != actual_csv_data.shape:
                raise AssertionError(
                    f"Shape mismatch: {expected_csv} has shape {expected_csv_data.shape}, {actual_csv} has shape {actual_csv_data.shape}")
            np.testing.assert_allclose(expected_csv_data, actual_csv_data, rtol=1e-5, atol=1e-8)
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
    case_runner("all_files", "splt")  # All optional input files present
    case_runner("no_files", "splt")  # No optional input files present
    case_runner("occ_gran_files", "splt")  # Granular occurrence input file present


def test_mplt_output():
    """Tests mplt outputs
    """
    case_runner("all_files", "mplt")  # All optional input files present
    case_runner("no_files", "mplt")  # No optional input files present
    case_runner("occ_gran_files", "mplt")


def test_qplt_output():
    """Tests qplt outputs
    """
    case_runner("all_files", "qplt")  # All optional input files present
    case_runner("no_files", "qplt")  # No optional input files present
    case_runner("occ_gran_files", "qplt")  # Granular occurrence input file present
