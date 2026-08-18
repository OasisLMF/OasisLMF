import filecmp
import pandas as pd
import pytest
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.join_summary_info.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_join-summary-info")


def test_empty_input():
    """Test join-summary-info does not crash and produces header-only output when input CSVs have no data rows"""
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # Header-only summary info CSV
        summaryinfo_csv = tmp_dir / "gul_summary-info.csv"
        summaryinfo_csv.write_text("summary_id,PortNumber,AccNumber,LocNumber,tiv\n")

        # Header-only ORD data CSV
        data_csv = tmp_dir / "aalgul_ord.csv"
        data_csv.write_text("SummaryId,SampleType,MeanLoss,SDLoss\n")

        output_csv = tmp_dir / "joined_output.csv"
        kwargs = {
            "summaryinfo": summaryinfo_csv,
            "data": data_csv,
            "output": output_csv,
        }
        main(**kwargs)

        assert output_csv.exists(), "Output CSV was not created"
        lines = output_csv.read_text().strip().splitlines()
        assert len(lines) == 1, f"Output CSV should contain only a header line, got {len(lines)} lines"


def test_join():
    """Tests join-summary-info output with csv ORD file
    """
    csv_summaryinfo = Path(TESTS_ASSETS_DIR, "csv", "gul_summary-info.csv")
    csv_data = Path(TESTS_ASSETS_DIR, "csv", "aalgul_ord.csv")
    csv_expected = Path(TESTS_ASSETS_DIR, "csv", "joined_aalgul_ord.csv")
    with TemporaryDirectory() as tmp_result_dir_str:
        csv_actual = Path(tmp_result_dir_str, "joined_aalgul_ord.csv")
        kwargs = {
            "summaryinfo": csv_summaryinfo,
            "data": csv_data,
            "output": csv_actual,
        }

        main(**kwargs)
        error_path = Path(TESTS_ASSETS_DIR, "csv", "error_files")
        arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
        try:
            assert filecmp.cmp(csv_expected, csv_actual, shallow=False)
        except Exception as e:
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(csv_actual),
                            Path(error_path, "joined_aalgul_ord.csv"))
            raise Exception(f"running 'join-summary-info {arg_str}' led to diff, see files at {error_path}") from e


def test_missing_summary_col():
    """Tests join-summary-info with non-ORD csv file, should not generate output
    """
    csv_summaryinfo = Path(TESTS_ASSETS_DIR, "csv", "gul_summary-info.csv")
    csv_data = Path(TESTS_ASSETS_DIR, "csv", "aalgul_nonord.csv")
    with TemporaryDirectory() as tmp_result_dir_str:
        csv_actual = Path(tmp_result_dir_str, "joined_aalgul_ord.csv")
        kwargs = {
            "summaryinfo": csv_summaryinfo,
            "data": csv_data,
            "output": csv_actual,
        }

        # Python 3.14 changed the list.index() error message wording
        with pytest.raises(ValueError, match="SummaryId.*not in list|list.index.*not in list"):
            main(**kwargs)


def test_join_parquet():
    """Tests join-summary-info output with ORD parquet file
    """
    parquet_summaryinfo = Path(TESTS_ASSETS_DIR, "parquet", "gul_summary-info.parquet")
    parquet_data = Path(TESTS_ASSETS_DIR, "parquet", "aalgul_ord.parquet")
    parquet_expected = Path(TESTS_ASSETS_DIR, "parquet", "joined_aalgul_ord.parquet")
    with TemporaryDirectory() as tmp_result_dir_str:
        parquet_actual = Path(tmp_result_dir_str, "joined_aalgul_ord.parquet")
        kwargs = {
            "summaryinfo": parquet_summaryinfo,
            "data": parquet_data,
            "output": parquet_actual,
        }

        main(**kwargs)
        error_path = Path(TESTS_ASSETS_DIR, "parquet", "error_files")
        arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
        try:
            expected_df = pd.read_parquet(parquet_expected)
            actual_df = pd.read_parquet(parquet_actual)
            pd.testing.assert_frame_equal(expected_df, actual_df)
        except Exception as e:
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(parquet_actual),
                            Path(error_path, "joined_aalgul_ord.parquet"))
            raise Exception(f"running 'join-summary-info {arg_str}' led to diff, see files at {error_path}") from e


def write_parquet_pair(tmp_dir, summary_info, data):
    summaryinfo_file = Path(tmp_dir, "gul_summary-info.parquet")
    data_file = Path(tmp_dir, "aalgul_ord.parquet")
    pd.DataFrame(summary_info).to_parquet(summaryinfo_file, index=False)
    pd.DataFrame(data).to_parquet(data_file, index=False)
    output_file = Path(tmp_dir, "joined_aalgul_ord.parquet")
    main(summaryinfo=summaryinfo_file, data=data_file, output=output_file)

    return pd.read_parquet(output_file)


def test_join_parquet_with_sparse_summary_ids():
    """Tests join-summary-info fills the gaps in non contiguous summary ids
    """
    with TemporaryDirectory() as tmp_dir:
        joined = write_parquet_pair(
            tmp_dir,
            summary_info={"summary_id": [3, 1, 7], "PortNumber": ["1", "1", "2"], "tiv": [10.5, 20.0, 30.25]},
            data={"SummaryId": [7, 1, 3, 2], "MeanLoss": [1.0, 2.0, 3.0, 4.0]},
        )

        assert list(joined["PortNumber"]) == ["2", "1", "1", ""]
        assert list(joined["tiv"]) == ["30.25", "20.0", "10.5", ""]


def test_join_parquet_with_summary_id_beyond_the_summary_info():
    """Tests join-summary-info leaves the summary info blank for unknown summary ids
    """
    with TemporaryDirectory() as tmp_dir:
        joined = write_parquet_pair(
            tmp_dir,
            summary_info={"summary_id": [1, 2], "PortNumber": ["1", "2"], "tiv": [10.0, 20.0]},
            data={"SummaryId": [1, 5, 2], "MeanLoss": [1.0, 2.0, 3.0]},
        )

        # an unknown summary id joins to nothing, so only the first summary column is filled
        assert list(joined["PortNumber"]) == ["1", "", "2"]

        # the columns after it are missing rather than empty. Which value reports that is the
        # pandas string dtype's to choose, None on the object dtype and NaN on StringDtype,
        # so the test asks whether it is missing rather than which of the two it is.
        tiv = list(joined["tiv"])
        assert [tiv[0], tiv[2]] == ["10.0", "20.0"]
        assert pd.isna(tiv[1])


def test_missing_summary_col_parquet():
    """Tests join-summary-info with non-ORD parquet file, should not generate output
    """
    parquet_summaryinfo = Path(TESTS_ASSETS_DIR, "parquet", "gul_summary-info.parquet")
    parquet_data = Path(TESTS_ASSETS_DIR, "parquet", "aalgul_nonord.parquet")
    with TemporaryDirectory() as tmp_result_dir_str:
        parquet_actual = Path(tmp_result_dir_str, "joined_aalgul_ord.parquet")
        kwargs = {
            "summaryinfo": parquet_summaryinfo,
            "data": parquet_data,
            "output": parquet_actual,
        }

        with pytest.raises(ValueError, match="Missing 'SummaryId' column in data file."):
            main(**kwargs)
