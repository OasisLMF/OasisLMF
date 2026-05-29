from pathlib import Path
import pandas as pd
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from oasislmf.pytools.kat.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_katpy")


def case_runner(dir_in, out_name, sorted):
    with TemporaryDirectory() as tmp_result_dir_str:
        dir_in = Path(TESTS_ASSETS_DIR, dir_in)
        expected_out = Path(TESTS_ASSETS_DIR, out_name)
        actual_out = Path(tmp_result_dir_str, out_name)

        kwargs = {
            "dir_in": dir_in,
            "qplt": True,
            "out": actual_out,
            "unsorted": not sorted,
        }

        main(**kwargs)

        suffix = Path(out_name).suffix

        try:
            if suffix == ".csv":
                expected_data = np.genfromtxt(expected_out, delimiter=',', skip_header=1)
                actual_data = np.genfromtxt(actual_out, delimiter=',', skip_header=1)
                if expected_data.shape != actual_data.shape:
                    raise AssertionError(
                        f"Shape mismatch: {expected_out} has shape {expected_data.shape}, {actual_out} has shape {actual_data.shape}")
                np.testing.assert_allclose(expected_data, actual_data, rtol=1e-5, atol=1e-8)
            if suffix == ".parquet":
                expected_data = pd.read_parquet(expected_out)
                actual_data = pd.read_parquet(actual_out)
                pd.testing.assert_frame_equal(expected_data, actual_data)
        except Exception as e:
            error_path = TESTS_ASSETS_DIR.joinpath('error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_out),
                            Path(error_path, out_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'katpy {arg_str}' led to diff, see files at {error_path}") from e


def test_empty_input():
    """Test katpy does not crash and produces header-only output when CSV inputs have no data rows"""
    from oasislmf.pytools.kat.manager import KAT_MAP, KAT_QPLT

    qplt_headers = KAT_MAP[KAT_QPLT]["headers"]

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        dir_in = tmp_dir / "qplt_empty"
        dir_in.mkdir()

        # Create a header-only QPLT CSV file
        (dir_in / "py_qplt1.csv").write_text(",".join(qplt_headers) + "\n")

        outfile = tmp_dir / "katpy_qplt_empty.csv"
        kwargs = {
            "dir_in": dir_in,
            "qplt": True,
            "out": outfile,
            "unsorted": False,
        }
        main(**kwargs)

        assert outfile.exists(), "Output CSV was not created"
        lines = outfile.read_text().strip().splitlines()
        assert len(lines) == 1, f"Output CSV should contain only a header line, got {len(lines)} lines"


def test_empty_input_bin_to_tabular():
    """Test katpy does not crash and produces empty output when binary inputs have no data"""
    from oasislmf.pytools.kat.manager import KAT_MAP, KAT_QPLT

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        dir_in = tmp_dir / "qplt_empty_bin"
        dir_in.mkdir()

        # 0-byte binary triggers the empty-input path in bin_concat_sort_by_headers
        (dir_in / "py_qplt1.bin").write_bytes(b"")

        for ext in ["csv", "parquet"]:
            outfile = tmp_dir / f"katpy_qplt_empty.{ext}"
            main(dir_in=dir_in, qplt=True, out=outfile, unsorted=False)

            assert outfile.exists(), f"Output {ext} was not created"
            if ext == "csv":
                lines = outfile.read_text().strip().splitlines()
                assert len(lines) == 1, f"Output CSV should contain only a header line, got {len(lines)} lines"
                assert lines[0] == ",".join(KAT_MAP[KAT_QPLT]["headers"])
            else:
                df = pd.read_parquet(outfile)
                assert len(df) == 0, f"Output parquet should have no rows, got {len(df)}"
                assert list(df.columns) == list(KAT_MAP[KAT_QPLT]["dtype"].names)


def test_katpy_csv_sorted():
    """Test katpy with csv inputs (using QPLT) sorted"""
    case_runner("qplt", "katpy_qplt.csv", True)


def test_katpy_bin_sorted():
    """Test katpy with bin inputs (using QPLT) sorted"""
    case_runner("bqplt", "bkatpy_qplt.csv", True)


def test_katpy_parquet_sorted():
    """Test katpy with parquet inputs (using QPLT) sorted"""
    case_runner("pqplt", "pkatpy_qplt.parquet", True)


def test_katpy_csv_unsorted():
    """Test katpy with csv inputs (using QPLT) unsorted"""
    case_runner("qplt", "ukatpy_qplt.csv", False)


def test_katpy_bin_unsorted():
    """Test katpy with bin inputs (using QPLT) unsorted"""
    case_runner("bqplt", "ubkatpy_qplt.csv", False)


def test_katpy_parquet_unsorted():
    """Test katpy with parquet inputs (using QPLT) unsorted"""
    case_runner("pqplt", "upkatpy_qplt.parquet", False)
