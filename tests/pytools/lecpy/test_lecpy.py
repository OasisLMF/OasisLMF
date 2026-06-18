import shutil
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from oasislmf.pytools.common.data import occurrence_dtype
from oasislmf.pytools.common.event_stream import SUMMARY_STREAM_ID, stream_info_to_bytes
from oasislmf.pytools.lec.data import EPT_dtype, PSEPT_dtype
from oasislmf.pytools.lec.manager import main
from tests.pytools.utils import make_idx_from_bin

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_lecpy")


def case_runner(sub_folder, test_name, out_ext="csv", use_return_period=False):
    """Run output file correctness tests
    Args:
        sub_folder (str | os.PathLike): path to workspace sub folder
        test_name (str): test name and sub folder containing all input and work files
        use_return_period (bool): Bool to use return period flag
    """
    ept_outfile_name = f"py_ept.{out_ext}"
    psept_outfile_name = f"py_psept.{out_ext}"

    with TemporaryDirectory() as tmp_result_dir_str:
        tmp_workspace_dir = Path(tmp_result_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, test_name), tmp_workspace_dir)

        expected_ept = tmp_workspace_dir / ept_outfile_name
        expected_psept = tmp_workspace_dir / psept_outfile_name

        out_dir = tmp_workspace_dir / "out"
        out_dir.mkdir()

        actual_ept = out_dir / ept_outfile_name
        actual_psept = out_dir / psept_outfile_name

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
            "ext": out_ext,
        }

        if test_name not in ["lec_pw_rp", "lec_pw", "lec_rp", "lec"]:
            raise Exception(f"Invalid or unimplemented test case {test_name} for lecpy")

        main(**kwargs)

        error_path = Path(TESTS_ASSETS_DIR, test_name, "error_files")
        arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])

        try:
            if out_ext == "csv":
                expected_ept_data = np.genfromtxt(expected_ept, delimiter=',', skip_header=1)
                actual_ept_data = np.genfromtxt(actual_ept, delimiter=',', skip_header=1)
                if expected_ept_data.shape != actual_ept_data.shape:
                    raise AssertionError(
                        f"Shape mismatch: {expected_ept} has shape {expected_ept_data.shape}, {actual_ept} has shape {actual_ept_data.shape}")
                np.testing.assert_allclose(expected_ept_data, actual_ept_data, rtol=1e-5, atol=1e-8)
            if out_ext == "parquet":
                expected_ept_data = pd.read_parquet(expected_ept)
                actual_ept_data = pd.read_parquet(actual_ept)
                pd.testing.assert_frame_equal(expected_ept_data, actual_ept_data)
            if out_ext == "bin":
                expected_ept_data = pd.DataFrame(np.fromfile(expected_ept, dtype=EPT_dtype))
                actual_ept_data = pd.DataFrame(np.fromfile(actual_ept, dtype=EPT_dtype))
                pd.testing.assert_frame_equal(expected_ept_data, actual_ept_data)
        except Exception as e:
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_ept, Path(error_path, ept_outfile_name))
            raise Exception(f"running 'lecpy {arg_str}' led to diff, see files at {error_path}") from e

        try:
            if out_ext == "csv":
                expected_psept_data = np.genfromtxt(expected_psept, delimiter=',', skip_header=1)
                actual_psept_data = np.genfromtxt(actual_psept, delimiter=',', skip_header=1)
                if expected_psept_data.shape != actual_psept_data.shape:
                    raise AssertionError(
                        f"Shape mismatch: {expected_psept} has shape {expected_psept_data.shape}, {actual_psept} has shape {actual_psept_data.shape}")
                np.testing.assert_allclose(expected_psept_data, actual_psept_data, rtol=1e-5, atol=1e-8)
            if out_ext == "parquet":
                expected_psept_data = pd.read_parquet(expected_psept)
                actual_psept_data = pd.read_parquet(actual_psept)
                pd.testing.assert_frame_equal(expected_psept_data, actual_psept_data)
            if out_ext == "bin":
                expected_psept_data = pd.DataFrame(np.fromfile(expected_psept, dtype=PSEPT_dtype))
                actual_psept_data = pd.DataFrame(np.fromfile(actual_psept, dtype=PSEPT_dtype))
                pd.testing.assert_frame_equal(expected_psept_data, actual_psept_data)
        except Exception as e:
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_psept, Path(error_path, psept_outfile_name))
            raise Exception(f"running 'lecpy {arg_str}' led to diff, see files at {error_path}") from e


def test_empty_input():
    """Test LEC does not crash and produces no output when summary binary has no loss records"""
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        work_dir = tmp_dir / "work" / "gul"
        work_dir.mkdir(parents=True)

        # 12-byte header-only summary binary (no loss records)
        # Occurrence files are read after the early return so are not needed here
        stream_header_int32 = np.frombuffer(stream_info_to_bytes(SUMMARY_STREAM_ID, 1), dtype=np.int32)[0]
        np.array([stream_header_int32, 10, 1], dtype=np.int32).tofile(work_dir / "summarypy1.bin")

        out_dir = tmp_dir / "out"
        out_dir.mkdir()
        ept_outfile = out_dir / "ept.csv"
        psept_outfile = out_dir / "psept.csv"

        kwargs = {
            "run_dir": tmp_dir,
            "subfolder": "gul",
            "agg_full_uncertainty": True,
            "agg_wheatsheaf": True,
            "agg_sample_mean": True,
            "agg_wheatsheaf_mean": True,
            "occ_full_uncertainty": True,
            "occ_wheatsheaf": True,
            "occ_sample_mean": True,
            "occ_wheatsheaf_mean": True,
            "ept": ept_outfile,
            "psept": psept_outfile,
            "ext": "csv",
        }
        main(**kwargs)

        # Early return when no loss records: output files should not be created
        assert not ept_outfile.exists(), "ept.csv should not be created when there are no loss records"
        assert not psept_outfile.exists(), "psept.csv should not be created when there are no loss records"


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


def test_lec_output_period_weights_and_return_periods_bin():
    """Tests LEC output with period weights and return_periods flag True
    """
    case_runner("gul", "lec_pw_rp", "bin", use_return_period=True)


def test_lec_output_return_periods_bin():
    """Tests LEC output with no period weights, and with return_periods flag True
    """
    case_runner("gul", "lec_rp", "bin", use_return_period=True)


def test_lec_output_period_weights_bin():
    """Tests LEC output with period weights, and with return_periods flag False
    """
    case_runner("gul", "lec_pw", "bin", use_return_period=False)


def test_lec_output_bin():
    """Tests LEC output with no period weights, and with return_periods flag False
    """
    case_runner("gul", "lec", "bin", use_return_period=False)


def test_lec_output_period_weights_and_return_periods_parquet():
    """Tests LEC output with period weights and return_periods flag True
    """
    case_runner("gul", "lec_pw_rp", "parquet", use_return_period=True)


def test_lec_output_return_periods_parquet():
    """Tests LEC output with no period weights, and with return_periods flag True
    """
    case_runner("gul", "lec_rp", "parquet", use_return_period=True)


def test_lec_output_period_weights_parquet():
    """Tests LEC output with period weights, and with return_periods flag False
    """
    case_runner("gul", "lec_pw", "parquet", use_return_period=False)


def test_lec_output_parquet():
    """Tests LEC output with no period weights, and with return_periods flag False
    """
    case_runner("gul", "lec", "parquet", use_return_period=False)


# ---------------------------------------------------------------------------
# .idx path tests
# ---------------------------------------------------------------------------

_ALL_OUTPUT_FLAGS = dict(
    agg_full_uncertainty=True, agg_wheatsheaf=True,
    agg_sample_mean=True, agg_wheatsheaf_mean=True,
    occ_full_uncertainty=True, occ_wheatsheaf=True,
    occ_sample_mean=True, occ_wheatsheaf_mean=True,
)


def test_lec_idx_output_matches_sequential():
    """LEC .idx path produces identical EPT/PSEPT output to the sequential path."""
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, "lec"), tmp_dir)

        work_dir = tmp_dir / "work" / "gul"
        for bin_path in sorted(work_dir.glob("*.bin")):
            make_idx_from_bin(bin_path, bin_path.with_suffix(".idx"))

        out_dir = tmp_dir / "out"
        out_dir.mkdir()
        actual_ept = out_dir / "py_ept.csv"
        actual_psept = out_dir / "py_psept.csv"

        main(
            run_dir=tmp_dir, subfolder="gul", use_return_period=False,
            ept=actual_ept, psept=actual_psept, ext="csv",
            **_ALL_OUTPUT_FLAGS,
        )

        for out_file, golden_name in [(actual_ept, "py_ept.csv"), (actual_psept, "py_psept.csv")]:
            expected = np.genfromtxt(tmp_dir / golden_name, delimiter=',', skip_header=1)
            actual = np.genfromtxt(out_file, delimiter=',', skip_header=1)
            assert expected.shape == actual.shape, f"Shape mismatch for {golden_name}: {expected.shape} vs {actual.shape}"
            np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-8,
                                       err_msg=f".idx path output differs from sequential for {golden_name}")


def test_lec_idx_empty_file_no_crash():
    """A 0-byte .idx for an empty partition does not raise ValueError (crash regression).

    summarypy --low-memory creates 0-byte .idx files for partitions that received no
    events. Before the fix np.memmap raised 'ValueError: cannot mmap an empty file'.
    """
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, "lec"), tmp_dir)

        work_dir = tmp_dir / "work" / "gul"
        for bin_path in sorted(work_dir.glob("*.bin")):
            make_idx_from_bin(bin_path, bin_path.with_suffix(".idx"))

        # Add an empty partition: header-only .bin + 0-byte .idx
        stream_hdr = np.frombuffer(stream_info_to_bytes(SUMMARY_STREAM_ID, 1), dtype=np.int32)[0]
        np.array([stream_hdr, 100, 1], dtype=np.int32).tofile(work_dir / "summarypy9.bin")
        (work_dir / "summarypy9.idx").touch()

        out_dir = tmp_dir / "out"
        out_dir.mkdir()

        main(
            run_dir=tmp_dir, subfolder="gul", use_return_period=False,
            ept=out_dir / "py_ept.csv", psept=out_dir / "py_psept.csv", ext="csv",
            **_ALL_OUTPUT_FLAGS,
        )
        assert (out_dir / "py_ept.csv").exists(), "EPT output should be created when populated bins are present"
        assert (out_dir / "py_psept.csv").exists(), "PSEPT output should be created when populated bins are present"


def test_lec_sequential_raises_on_insufficient_disk():
    """Sequential path raises RuntimeError before allocating .bdat files when disk is full."""
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, "lec"), tmp_dir)

        # No .idx files → sequential path is taken
        out_dir = tmp_dir / "out"
        out_dir.mkdir()

        real = shutil.disk_usage("/")
        fake_usage = type(real)(total=real.total, used=real.used, free=0)

        with patch("oasislmf.pytools.lec.manager.shutil.disk_usage", return_value=fake_usage):
            with pytest.raises(RuntimeError, match="Insufficient disk space"):
                main(
                    run_dir=tmp_dir, subfolder="gul", use_return_period=False,
                    ept=out_dir / "py_ept.csv", psept=out_dir / "py_psept.csv", ext="csv",
                    **_ALL_OUTPUT_FLAGS,
                )


def test_lec_idx_partial_coverage_falls_back_to_sequential():
    """Partial .idx coverage (not all .bin files have .idx) triggers sequential fallback with correct output."""
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, "lec"), tmp_dir)

        work_dir = tmp_dir / "work" / "gul"
        bins = sorted(work_dir.glob("*.bin"))
        # Only provide idx for the first bin → triggers warning + sequential fallback
        make_idx_from_bin(bins[0], bins[0].with_suffix(".idx"))

        out_dir = tmp_dir / "out"
        out_dir.mkdir()
        actual_ept = out_dir / "py_ept.csv"
        actual_psept = out_dir / "py_psept.csv"

        main(
            run_dir=tmp_dir, subfolder="gul", use_return_period=False,
            ept=actual_ept, psept=actual_psept, ext="csv",
            **_ALL_OUTPUT_FLAGS,
        )

        for out_file, golden_name in [(actual_ept, "py_ept.csv"), (actual_psept, "py_psept.csv")]:
            expected = np.genfromtxt(tmp_dir / golden_name, delimiter=',', skip_header=1)
            actual = np.genfromtxt(out_file, delimiter=',', skip_header=1)
            assert expected.shape == actual.shape, f"Shape mismatch for {golden_name}: {expected.shape} vs {actual.shape}"
            np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-8)


def _build_multi_summary_run(tmp_dir, n_summaries, n_events=60, sample_size=20, n_periods=80, n_files=3, seed=11):
    """Build a multi-summary, partitioned summary stream + occurrence/returnperiods for buffer tests.

    Events are split across n_files (partition-by-event); every summary_id appears in each
    file for the events that landed there — mirroring a real partitioned run. A paired .idx
    is generated for each .bin so the per-summary-id path is taken.
    """
    rng = np.random.default_rng(seed)
    input_dir = tmp_dir / "input"
    work_dir = tmp_dir / "work" / "gul"
    input_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)

    event_ids = np.arange(1, n_events + 1, dtype=np.int32)
    recs = np.zeros(n_events, dtype=occurrence_dtype)
    recs["event_id"] = event_ids
    recs["period_no"] = (event_ids % n_periods) + 1
    with open(input_dir / "occurrence.bin", "wb") as f:
        f.write(np.array([1, n_periods], dtype=np.int32).tobytes())
        f.write(recs.tobytes())
    np.array([1000, 500, 250, 100, 50, 25, 10, 5, 2], dtype=np.int32).tofile(input_dir / "returnperiods.bin")

    stream_hdr = np.frombuffer(stream_info_to_bytes(SUMMARY_STREAM_ID, sample_size), dtype=np.int32)[0]
    sidxs = np.array([-1] + list(range(1, sample_size + 1)), dtype=np.int32)  # mean + samples
    for fi, events in enumerate(np.array_split(event_ids, n_files), start=1):
        rows = [np.array([stream_hdr, sample_size, 1], dtype=np.int32)]
        for ev in events:
            for sid in range(1, n_summaries + 1):
                block = np.empty(3 + 2 * (len(sidxs) + 1), dtype=np.int32)
                block[0] = ev
                block[1] = sid
                block[2] = np.float32(0.0).view(np.int32)  # expval
                pos = 3
                for s, loss in zip(sidxs, rng.uniform(1.0, 1000.0, len(sidxs)).astype(np.float32)):
                    block[pos] = s
                    block[pos + 1] = loss.view(np.int32)
                    pos += 2
                block[pos] = 0                               # terminating sidx
                block[pos + 1] = np.float32(0.0).view(np.int32)
                rows.append(block)
        np.concatenate(rows).tofile(work_dir / f"summarypy{fi}.bin")
    for b in sorted(work_dir.glob("*.bin")):
        make_idx_from_bin(b, b.with_suffix(".idx"))


def test_lec_idx_reused_buffer_matches_sequential_multi_summary():
    """Per-summary .idx path matches the dense path across many summaries (reused-buffer guard).

    The write_* generators share one output buffer across all per-summary calls. This builds
    a many-summary run (so the buffer is reused hundreds of times) and asserts the .idx output
    is byte-identical to the dense path — catching any stale data leaking between calls.
    use_return_period=True keeps the dense wheatsheaf-mean path off its empty-mean_map edge.
    """
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str) / "workspace"
        _build_multi_summary_run(tmp_dir, n_summaries=120)

        seq_dir = tmp_dir / "out_seq"
        idx_dir = tmp_dir / "out_idx"
        seq_dir.mkdir()
        idx_dir.mkdir()

        # Sequential (dense) path: remove .idx files
        work_dir = tmp_dir / "work" / "gul"
        for idx in work_dir.glob("*.idx"):
            idx.unlink()
        main(run_dir=tmp_dir, subfolder="gul", use_return_period=True,
             ept=seq_dir / "py_ept.csv", psept=seq_dir / "py_psept.csv", ext="csv", **_ALL_OUTPUT_FLAGS)

        # Per-summary .idx path: regenerate .idx files
        for b in sorted(work_dir.glob("*.bin")):
            make_idx_from_bin(b, b.with_suffix(".idx"))
        main(run_dir=tmp_dir, subfolder="gul", use_return_period=True,
             ept=idx_dir / "py_ept.csv", psept=idx_dir / "py_psept.csv", ext="csv", **_ALL_OUTPUT_FLAGS)

        for name in ("py_ept.csv", "py_psept.csv"):
            a = np.genfromtxt(seq_dir / name, delimiter=',', skip_header=1)
            b = np.genfromtxt(idx_dir / name, delimiter=',', skip_header=1)
            a = a[np.lexsort(a.T[::-1])]
            b = b[np.lexsort(b.T[::-1])]
            assert a.shape == b.shape, f"Shape mismatch for {name}: {a.shape} vs {b.shape}"
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8,
                                       err_msg=f".idx (reused buffer) differs from sequential for {name}")
