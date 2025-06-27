import filecmp
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path

from oasislmf.pytools.converters.bintocsv import bintocsv
from oasislmf.pytools.converters.csvtobin import csvtobin
from oasislmf.pytools.converters.cdftocsv import cdftocsv

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_converters")


def case_runner(converter, type):
    if converter == "bintocsv":
        in_ext = ".bin"
        out_ext = ".csv"
        converter = bintocsv
    elif converter == "csvtobin":
        in_ext = ".csv"
        out_ext = ".bin"
        converter = csvtobin
    else:
        raise RuntimeError(f"Unknown test type {type}")

    with TemporaryDirectory() as tmp_result_dir_str:
        infile_name = f"{type}{in_ext}"
        outfile_name = f"{type}{out_ext}"
        infile = Path(TESTS_ASSETS_DIR, infile_name)
        expected_outfile = Path(TESTS_ASSETS_DIR, outfile_name)
        actual_outfile = Path(tmp_result_dir_str, outfile_name)

        kwargs = {
            "file_in": infile,
            "file_out": actual_outfile,
            "type": type,
        }
        converter(**kwargs)

        try:
            assert filecmp.cmp(expected_outfile, actual_outfile, shallow=False)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running '{converter} {arg_str}' led to diff, see files at {error_path}") from e


def test_aggregatevulnerability():
    case_runner("bintocsv", "aggregatevulnerability")
    case_runner("csvtobin", "aggregatevulnerability")


def test_amplifications():
    case_runner("bintocsv", "amplifications")
    case_runner("csvtobin", "amplifications")


def test_complex_items():
    case_runner("bintocsv", "complex_items")
    case_runner("csvtobin", "complex_items")


def test_coverages():
    case_runner("bintocsv", "coverages")
    case_runner("csvtobin", "coverages")


def test_damagebin():
    case_runner("bintocsv", "damagebin")
    case_runner("csvtobin", "damagebin")


def test_eve():
    case_runner("bintocsv", "eve")
    case_runner("csvtobin", "eve")


def test_fm_policytc():
    case_runner("bintocsv", "fm_policytc")
    case_runner("csvtobin", "fm_policytc")


def test_fm_profile():
    case_runner("bintocsv", "fm_profile")
    case_runner("csvtobin", "fm_profile")


def test_fm_profile_step():
    case_runner("bintocsv", "fm_profile_step")
    case_runner("csvtobin", "fm_profile_step")


def test_fm_programme():
    case_runner("bintocsv", "fm_programme")
    case_runner("csvtobin", "fm_programme")


def test_fm_summary_xref():
    case_runner("bintocsv", "fm_summary_xref")
    case_runner("csvtobin", "fm_summary_xref")


def test_gul_summary_xref():
    case_runner("bintocsv", "gul_summary_xref")
    case_runner("csvtobin", "gul_summary_xref")


def test_fm_xref():
    case_runner("bintocsv", "fm_xref")
    case_runner("csvtobin", "fm_xref")


def test_items():
    case_runner("bintocsv", "items")
    case_runner("csvtobin", "items")


def test_periods():
    case_runner("bintocsv", "periods")
    case_runner("csvtobin", "periods")


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
