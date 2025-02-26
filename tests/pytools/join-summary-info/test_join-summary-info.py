import filecmp
import pytest
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.join_summary_info.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_join-summary-info")


def test_join():
    """Tests join-summary-info output with ORD file
    """
    csv_summaryinfo = Path(TESTS_ASSETS_DIR, "gul_summary-info.csv")
    csv_data = Path(TESTS_ASSETS_DIR, "aalgul_ord.csv")
    csv_expected = Path(TESTS_ASSETS_DIR, "joined_aalgul_ord.csv")
    with TemporaryDirectory() as tmp_result_dir_str:
        csv_actual = Path(tmp_result_dir_str, "joined_aalgul_ord.csv")
        kwargs = {
            "summaryinfo": csv_summaryinfo,
            "data": csv_data,
            "output": csv_actual,
        }

        main(**kwargs)
        error_path = Path(TESTS_ASSETS_DIR, "error_files")
        arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
        try:
            assert filecmp.cmp(csv_expected, csv_actual, shallow=False)
        except Exception as e:
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(csv_actual),
                            Path(error_path, "joined_aalgul_ord.csv"))
            raise Exception(f"running 'join-summary-info {arg_str}' led to diff, see files at {error_path}") from e


def test_missing_summary_col():
    """Tests join-summary-info with non-ORD file, should not generate output
    """
    csv_summaryinfo = Path(TESTS_ASSETS_DIR, "gul_summary-info.csv")
    csv_data = Path(TESTS_ASSETS_DIR, "aalgul_nonord.csv")
    with TemporaryDirectory() as tmp_result_dir_str:
        csv_actual = Path(tmp_result_dir_str, "joined_aalgul_ord.csv")
        kwargs = {
            "summaryinfo": csv_summaryinfo,
            "data": csv_data,
            "output": csv_actual,
        }

        with pytest.raises(ValueError, match="\'SummaryId\' is not in list"):
            main(**kwargs)
