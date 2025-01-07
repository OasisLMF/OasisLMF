import filecmp
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.aal.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_aalpy")


def case_runner(sub_folder, test_name, meanonly=False):
    """Run output file correctness tests
    Args:
        sub_folder (str | os.PathLike): path to workspace sub folder
        test_name (str): test name
    """
    print("HERE")
    csv_name = f"py_{test_name}{sub_folder}.csv"
    if meanonly:
        csv_name = f"py_{test_name}meanonly{sub_folder}.csv"
    expected_csv = Path(TESTS_ASSETS_DIR, "all_files", csv_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        tmp_workspace_dir = Path(tmp_result_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, "all_files"), tmp_workspace_dir)

        out_dir = tmp_workspace_dir / "out"
        out_dir.mkdir()

        actual_csv = out_dir / csv_name
        print("Workspace directory structure:")
        for root, dirs, files in os.walk(tmp_workspace_dir):
            print(f"Root: {root}")
            for d in dirs:
                print(f"  Dir: {d}")
            for f in files:
                print(f"  File: {f}")

        kwargs = {
            "run_dir": tmp_workspace_dir,
            "subfolder": sub_folder,
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
            shutil.copyfile(actual_csv, Path(error_path, csv_name))
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
