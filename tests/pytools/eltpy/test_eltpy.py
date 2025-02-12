import shutil
from tempfile import TemporaryDirectory
import numpy as np
from pathlib import Path
from unittest.mock import patch

from oasislmf.pytools.common.input_files import read_event_rates
from oasislmf.pytools.elt.manager import main
from oasislmf.pytools.common.data import (oasis_int, oasis_float)

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_eltpy")


def case_runner(test_name, with_event_rate=False):
    csv_name = f"py_{test_name}.csv"
    summary_bin_input = Path(TESTS_ASSETS_DIR, "summarypy.bin")
    if with_event_rate:
        csv_name = f"py_{test_name}_er.csv"
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
            eids, ers = read_event_rates(Path(TESTS_ASSETS_DIR, "input"), filename="er.csv")
            with patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(eids, ers)):
                main(**kwargs)
        else:
            with patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))):
                main(**kwargs)

        try:
            expected_csv_data = np.genfromtxt(expected_csv, delimiter=',', skip_header=1)
            actual_csv_data = np.genfromtxt(actual_csv, delimiter=',', skip_header=1)
            if expected_csv_data.shape != actual_csv_data.shape:
                raise AssertionError(
                    f"Shape mismatch: {expected_csv} has shape {expected_csv_data.shape}, {actual_csv} has shape {actual_csv_data.shape}")
            np.testing.assert_allclose(expected_csv_data, actual_csv_data, rtol=1e-5, atol=1e-8)
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


def test_melt_output_with_event_rate():
    case_runner("melt", with_event_rate=True)


def test_qelt_output():
    case_runner("qelt")
