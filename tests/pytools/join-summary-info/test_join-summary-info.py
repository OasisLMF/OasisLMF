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

        # an unknown summary id joins to empty summary columns, the same as a gap does
        assert list(joined["PortNumber"]) == ["1", "", "2"]
        assert list(joined["tiv"]) == ["10.0", "", "20.0"]


def test_join_parquet_with_every_summary_id_beyond_the_summary_info():
    """Tests join-summary-info joins a file whose every summary id is unknown

    Nothing to join is not an error, and the summary columns still have to be there for the
    output to have the schema the joined file is supposed to have.
    """
    with TemporaryDirectory() as tmp_dir:
        joined = write_parquet_pair(
            tmp_dir,
            summary_info={"summary_id": [1, 2], "PortNumber": ["1", "2"], "tiv": [10.0, 20.0]},
            data={"SummaryId": [7, 8], "MeanLoss": [1.0, 2.0]},
        )

        assert list(joined.columns) == ["SummaryId", "MeanLoss", "PortNumber", "tiv"]
        assert list(joined["PortNumber"]) == ["", ""]
        assert list(joined["tiv"]) == ["", ""]


def test_join_parquet_with_a_negative_summary_id():
    """Tests join-summary-info leaves a negative summary id blank rather than counting back

    A negative id is a legal index into the summary table, so it would otherwise silently join
    the row to some other summary's info.
    """
    with TemporaryDirectory() as tmp_dir:
        joined = write_parquet_pair(
            tmp_dir,
            summary_info={"summary_id": [1, 2, 3], "PortNumber": ["1", "2", "3"], "tiv": [10.0, 20.0, 30.0]},
            data={"SummaryId": [1, -1, -3, 3], "MeanLoss": [1.0, 2.0, 3.0, 4.0]},
        )

        assert list(joined["PortNumber"]) == ["1", "", "", "3"]
        assert list(joined["tiv"]) == ["10.0", "", "", "30.0"]


def test_join_matches_csv_for_a_negative_summary_id(tmp_path):
    """Tests both formats say the same thing about a negative summary id"""
    summary_info = pd.DataFrame({"summary_id": [1, 2, 3], "PortNumber": ["1", "2", "3"], "tiv": [10.0, 20.0, 30.0]})
    data = pd.DataFrame({"SummaryId": [1, -1, -3, 3], "MeanLoss": [1.0, 2.0, 3.0, 4.0]})

    summary_info.to_csv(tmp_path / "gul_summary-info.csv", index=False)
    data.to_csv(tmp_path / "aalgul_ord.csv", index=False)
    main(summaryinfo=tmp_path / "gul_summary-info.csv", data=tmp_path / "aalgul_ord.csv",
         output=tmp_path / "joined.csv")
    joined_csv = pd.read_csv(tmp_path / "joined.csv", dtype=str).fillna("")

    joined_parquet = write_parquet_pair(tmp_path, summary_info=summary_info, data=data)

    assert list(joined_csv["PortNumber"]) == list(joined_parquet["PortNumber"])
    assert list(joined_csv["tiv"]) == list(joined_parquet["tiv"])


def test_join_parquet_with_no_summary_info_columns():
    """Tests join-summary-info joins a summary info file holding nothing but summary_id

    There is no summary info to add, so every row joins to nothing and the data file comes back
    unchanged rather than the join failing.
    """
    with TemporaryDirectory() as tmp_dir:
        joined = write_parquet_pair(
            tmp_dir,
            summary_info={"summary_id": [1, 2]},
            data={"SummaryId": [1, 2, 9], "MeanLoss": [1.0, 2.0, 3.0]},
        )

        assert list(joined.columns) == ["SummaryId", "MeanLoss"]
        assert list(joined["MeanLoss"]) == [1.0, 2.0, 3.0]


def write_csv_pair(tmp_dir, summary_info, data):
    summaryinfo_file = Path(tmp_dir, "gul_summary-info.csv")
    data_file = Path(tmp_dir, "aalgul_ord.csv")
    pd.DataFrame(summary_info).to_csv(summaryinfo_file, index=False)
    pd.DataFrame(data).to_csv(data_file, index=False)
    output_file = Path(tmp_dir, "joined_aalgul_ord.csv")
    main(summaryinfo=summaryinfo_file, data=data_file, output=output_file)

    return output_file


def test_join_csv_with_no_summary_info_columns(tmp_path):
    """Tests join-summary-info joins a summary info file holding nothing but summary_id

    The csv counterpart of test_join_parquet_with_no_summary_info_columns. With no summary
    columns to add there is nothing to separate them from the data, so a joined row must not
    gain a trailing comma that a row with no summary info to join does not have -- that is a
    ragged file, and pandas reads the extra field back by shifting the first column into the
    index rather than by failing.
    """
    output_file = write_csv_pair(
        tmp_path,
        summary_info={"summary_id": [1, 2]},
        data={"SummaryId": [1, 2, 9], "MeanLoss": [1.0, 2.0, 3.0]},
    )

    lines = output_file.read_text().strip().splitlines()
    assert [line.count(",") for line in lines] == [1, 1, 1, 1]

    joined = pd.read_csv(output_file)
    assert list(joined.columns) == ["SummaryId", "MeanLoss"]
    assert list(joined["MeanLoss"]) == [1.0, 2.0, 3.0]


def test_join_matches_csv_with_no_summary_info_columns(tmp_path):
    """Tests both formats say the same thing about a summary info file with no columns to join"""
    summary_info = pd.DataFrame({"summary_id": [1, 2]})
    data = pd.DataFrame({"SummaryId": [1, 2, 9], "MeanLoss": [1.0, 2.0, 3.0]})

    joined_csv = pd.read_csv(write_csv_pair(tmp_path, summary_info=summary_info, data=data))
    joined_parquet = write_parquet_pair(tmp_path, summary_info=summary_info, data=data)

    pd.testing.assert_frame_equal(joined_csv, joined_parquet)


def test_join_parquet_matches_csv_for_unknown_summary_ids(tmp_path):
    """Tests both formats say the same thing about a summary id they cannot join"""
    summary_info = pd.DataFrame({"summary_id": [1, 2], "PortNumber": ["1", "2"], "tiv": [10.0, 20.0]})
    data = pd.DataFrame({"SummaryId": [1, 5, 2], "MeanLoss": [1.0, 2.0, 3.0]})

    summary_info.to_csv(tmp_path / "gul_summary-info.csv", index=False)
    data.to_csv(tmp_path / "aalgul_ord.csv", index=False)
    main(summaryinfo=tmp_path / "gul_summary-info.csv", data=tmp_path / "aalgul_ord.csv",
         output=tmp_path / "joined.csv")
    joined_csv = pd.read_csv(tmp_path / "joined.csv", dtype=str).fillna("")

    joined_parquet = write_parquet_pair(tmp_path, summary_info=summary_info, data=data)

    assert list(joined_csv["PortNumber"]) == list(joined_parquet["PortNumber"])
    assert list(joined_csv["tiv"]) == list(joined_parquet["tiv"])


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
