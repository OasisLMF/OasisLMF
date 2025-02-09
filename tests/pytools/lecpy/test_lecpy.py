import filecmp
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.lec.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_lecpy")


def case_runner(sub_folder, test_name, use_return_period):
    """Run output file correctness tests
    Args:
        sub_folder (str | os.PathLike): path to workspace sub folder
        test_name (str): test name and sub folder containing all input and work files
        use_return_period (bool): Bool to use return period flag
    """
    ept_csv_name = "py_ept.csv"
    psept_csv_name = "py_psept.csv"

    expected_ept = Path(TESTS_ASSETS_DIR, test_name, ept_csv_name)
    expected_psept = Path(TESTS_ASSETS_DIR, test_name, psept_csv_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        tmp_workspace_dir = Path(tmp_result_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, test_name), tmp_workspace_dir)

        out_dir = tmp_workspace_dir / "out"
        out_dir.mkdir()

        actual_ept = out_dir / ept_csv_name
        actual_psept = out_dir / psept_csv_name

        kwargs = {
            "run_dir": tmp_workspace_dir,
            "subfolder": sub_folder,
            "use_return_period": use_return_period,
            "agg_full_uncertainty": True,
            "agg_wheatsheaf": True,
            "agg_sample_mean": True,
            "agg_wheatsheaf_mean": True,
            "occ_full_uncertainty": True,
            "occ_wheatsheaf": True,
            "occ_sample_mean": True,
            "occ_wheatsheaf_mean": True,
            "ept": actual_ept,
            "psept": actual_psept,
        }

        if test_name not in ["lec_pw_rp", "lec_pw", "lec_rp", "lec"]:
            raise Exception(f"Invalid or unimplemented test case {test_name} for pltpy")

        main(**kwargs)

        try:
            assert filecmp.cmp(expected_ept, actual_ept, shallow=False)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, test_name, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_ept, Path(error_path, expected_ept))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'pltpy {arg_str}' led to diff, see files at {error_path}") from e

        try:
            assert filecmp.cmp(expected_psept, actual_psept, shallow=False)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, test_name, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_psept, Path(error_path, expected_psept))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'pltpy {arg_str}' led to diff, see files at {error_path}") from e


def test_lec_output_period_weights_and_return_periods():
    """Tests LEC output with period weights and return_periods flag True
    """
    case_runner("gul", "lec_pw_rp", use_return_period=True)


def test_lec_output_return_periods():
    """Tests LEC output with no period weights, and with return_periods flag True
    """
    case_runner("gul", "lec_rp", use_return_period=True)


def test_lec_output_period_weights():
    """Tests LEC output with period weights, and with return_periods flag False
    """
    case_runner("gul", "lec_pw", use_return_period=False)


def test_lec_output():
    """Tests LEC output with no period weights, and with return_periods flag False
    """
    case_runner("gul", "lec", use_return_period=False)
