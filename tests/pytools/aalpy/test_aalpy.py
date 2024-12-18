
import filecmp
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

# from oasislmf.pytools.common.data import (oasis_int, oasis_float)
from oasislmf.pytools.aal.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_aalpy")


def case_runner(sub_folder, test_name, meanonly=False):
    """Run output file correctness tests
    Args:
        sub_folder (str | os.PathLike): path to workspace sub folder
        test_name (str): test name
    """
    csv_name = f"py_{test_name}{sub_folder}.csv"
    if meanonly:
        csv_name = f"py_{test_name}meanonly{sub_folder}.csv"
    expected_csv = Path(TESTS_ASSETS_DIR, "all_files", csv_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        actual_csv = Path(tmp_result_dir_str, csv_name)

        kwargs = {
            "run_dir": Path(TESTS_ASSETS_DIR, "all_files"),
            "subfolder": sub_folder,
            "skip_idxs": True,
            "meanonly": meanonly,
        }

        if test_name in ["aal", "alct"]:
            kwargs[f"{test_name}"] = actual_csv
        else:
            raise Exception(f"Invalid or unimplemented test case {test_name} for pltpy")

        main(**kwargs)

        try:
            assert filecmp.cmp(expected_csv, actual_csv, shallow=False)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, "all_files", "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_csv),
                            Path(error_path, csv_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'pltpy {arg_str}' led to diff, see files at {error_path}") from e


def test_aal_output():
    """Tests AAL output
    """
    case_runner("gul", "aal")


def test_aalmeanonly_output():
    """Tests AAL meanonly output
    """
    case_runner("gul", "aal", meanonly=True)


def test_alct_output():
    """Tests ALCT output
    """
    case_runner("gul", "alct")
