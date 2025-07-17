import filecmp
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.converters.cdftocsv import cdftocsv
from oasislmf.pytools.converters.data import TYPE_MAP

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_converters")


def case_runner(converter, file_type, sub_dir, filename=None, **kwargs):
    if converter == "bintocsv":
        in_ext = ".bin"
        out_ext = ".csv"
        converter = bintocsv
    elif converter == "csvtobin":
        in_ext = ".csv"
        out_ext = ".bin"
        converter = csvtobin
    else:
        raise RuntimeError(f"Unknown test type {file_type}")

    if filename == None:
        filename = file_type
    with TemporaryDirectory() as tmp_result_dir_str:
        infile_name = f"{filename}{in_ext}"
        outfile_name = f"{filename}{out_ext}"
        infile = Path(TESTS_ASSETS_DIR, sub_dir, infile_name)
        expected_outfile = Path(TESTS_ASSETS_DIR, sub_dir, outfile_name)
        actual_outfile = Path(tmp_result_dir_str, outfile_name)

        converter_args = {
            "file_in": infile,
            "file_out": actual_outfile,
            "file_type": file_type,
            **kwargs,
        }
        converter(**converter_args)

        try:
            if converter == "bintocsv":
                expected_outfile_data = np.genfromtxt(expected_outfile, delimiter=',', skip_header=1)
                actual_outfile_data = np.genfromtxt(actual_outfile, delimiter=',', skip_header=1)
                if expected_outfile_data.shape != actual_outfile_data.shape:
                    raise AssertionError(
                        f"Shape mismatch: {expected_outfile} has shape {expected_outfile_data.shape}, {actual_outfile} has shape {actual_outfile_data.shape}"
                    )
                np.testing.assert_allclose(expected_outfile_data, actual_outfile_data, rtol=1e-5, atol=1e-8)
            if converter == "csvtobin":
                expected_outfile_data = pd.DataFrame(np.fromfile(expected_outfile, dtype=TYPE_MAP[file_type]))
                actual_outfile_data = pd.DataFrame(np.fromfile(actual_outfile, dtype=TYPE_MAP[file_type]))
                pd.testing.assert_frame_equal(expected_outfile_data, actual_outfile_data)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, sub_dir, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in converter_args.items()])
            raise Exception(f"running '{converter} {arg_str}' led to diff, see files at {error_path}") from e


def test_aggregatevulnerability():
    case_runner("bintocsv", "aggregatevulnerability", "static")
    case_runner("csvtobin", "aggregatevulnerability", "static")


def test_damagebin():
    case_runner("bintocsv", "damagebin", "static")
    case_runner("csvtobin", "damagebin", "static")


def test_lossfactors():
    case_runner("bintocsv", "lossfactors", "static")
    case_runner("csvtobin", "lossfactors", "static")


def test_random():
    case_runner("bintocsv", "random", "static")
    case_runner("csvtobin", "random", "static")


def test_weights():
    case_runner("bintocsv", "weights", "static")
    case_runner("csvtobin", "weights", "static")


def test_amplifications():
    case_runner("bintocsv", "amplifications", "input")
    case_runner("csvtobin", "amplifications", "input")


def test_correlations_items():
    case_runner("bintocsv", "correlations", "input")
    case_runner("csvtobin", "correlations", "input")


def test_complex_items():
    case_runner("bintocsv", "complex_items", "input")
    case_runner("csvtobin", "complex_items", "input")


def test_coverages():
    case_runner("bintocsv", "coverages", "input")
    case_runner("csvtobin", "coverages", "input")


def test_eve():
    case_runner("bintocsv", "eve", "input")
    case_runner("csvtobin", "eve", "input")


def test_fm_policytc():
    case_runner("bintocsv", "fm_policytc", "input")
    case_runner("csvtobin", "fm_policytc", "input")


def test_fm_profile():
    case_runner("bintocsv", "fm_profile", "input")
    case_runner("csvtobin", "fm_profile", "input")


def test_fm_profile_step():
    case_runner("bintocsv", "fm_profile_step", "input")
    case_runner("csvtobin", "fm_profile_step", "input")


def test_fm_programme():
    case_runner("bintocsv", "fm_programme", "input")
    case_runner("csvtobin", "fm_programme", "input")


def test_fm_summary_xref():
    case_runner("bintocsv", "fm_summary_xref", "input")
    case_runner("csvtobin", "fm_summary_xref", "input")


def test_gul_summary_xref():
    case_runner("bintocsv", "gul_summary_xref", "input")
    case_runner("csvtobin", "gul_summary_xref", "input")


def test_fm_xref():
    case_runner("bintocsv", "fm_xref", "input")
    case_runner("csvtobin", "fm_xref", "input")


def test_items():
    case_runner("bintocsv", "items", "input")
    case_runner("csvtobin", "items", "input")


def test_occurrence():
    case_runner("bintocsv", "occurrence", "input", "occurrence")
    case_runner("bintocsv", "occurrence", "input", "occurrence_gran")
    case_runner("bintocsv", "occurrence", "input", "occurrence_noalg")
    case_runner("csvtobin", "occurrence", "input", "occurrence", no_of_periods=9)
    case_runner("csvtobin", "occurrence", "input", "occurrence_gran", no_of_periods=9, granular=True)
    case_runner("csvtobin", "occurrence", "input", "occurrence_noalg", no_of_periods=9, no_date_alg=True)


def test_periods():
    case_runner("bintocsv", "periods", "input")
    case_runner("csvtobin", "periods", "input")


def test_quantile():
    case_runner("bintocsv", "quantile", "input")
    case_runner("csvtobin", "quantile", "input")


def test_returnperiods():
    case_runner("bintocsv", "returnperiods", "input")
    case_runner("csvtobin", "returnperiods", "input")


def test_aal():
    case_runner("bintocsv", "aal", "output")
    case_runner("csvtobin", "aal", "output")
    case_runner("bintocsv", "aalmeanonly", "output")
    case_runner("csvtobin", "aalmeanonly", "output")
    case_runner("bintocsv", "alct", "output")
    case_runner("csvtobin", "alct", "output")


def test_elt():
    case_runner("bintocsv", "selt", "output")
    case_runner("csvtobin", "selt", "output")
    case_runner("bintocsv", "melt", "output")
    case_runner("csvtobin", "melt", "output")
    case_runner("bintocsv", "qelt", "output")
    case_runner("csvtobin", "qelt", "output")


def test_lec():
    case_runner("bintocsv", "ept", "output")
    case_runner("csvtobin", "ept", "output")
    case_runner("bintocsv", "psept", "output")
    case_runner("csvtobin", "psept", "output")


def test_plt():
    case_runner("bintocsv", "splt", "output")
    case_runner("csvtobin", "splt", "output")
    case_runner("bintocsv", "mplt", "output")
    case_runner("csvtobin", "mplt", "output")
    case_runner("bintocsv", "qplt", "output")
    case_runner("csvtobin", "qplt", "output")


def test_fm():
    case_runner("bintocsv", "fm", "misc", "raw_ils")
    case_runner("csvtobin", "fm", "misc", "raw_ils", stream_type=2, max_sample_index=1)


def test_gul():
    case_runner("bintocsv", "gul", "misc", "raw_guls")
    case_runner("csvtobin", "gul", "misc", "raw_guls", stream_type=2, max_sample_index=1)


def test_summarycalc():
    case_runner("csvtobin", "summarycalc", "misc", "summary", summary_set_id=1, max_sample_index=100)


def test_cdftocsv():
    with TemporaryDirectory() as tmp_result_dir_str:
        infile_name = "getmodel.bin"
        outfile_name = "getmodel.csv"
        run_dir = Path(TESTS_ASSETS_DIR, "cdftocsv")
        infile = Path(run_dir, infile_name)
        expected_outfile = Path(run_dir, outfile_name)
        actual_outfile = Path(tmp_result_dir_str, outfile_name)

        kwargs = {
            "file_in": infile,
            "file_out": actual_outfile,
            "run_dir": run_dir,
        }
        cdftocsv(**kwargs)

        try:
            assert filecmp.cmp(expected_outfile, actual_outfile, shallow=False)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'cdftocsv {arg_str}' led to diff, see files at {error_path}") from e
