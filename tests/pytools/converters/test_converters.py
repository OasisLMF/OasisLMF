import filecmp
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path

from oasislmf.pytools.converters.bintocsv import bintocsv
from oasislmf.pytools.converters.csvtobin import csvtobin

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


def test_coverages():
    case_runner("bintocsv", "coverages")
    case_runner("csvtobin", "coverages")


def test_damagebin():
    case_runner("bintocsv", "damagebin")
    case_runner("csvtobin", "damagebin")


def test_eve():
    case_runner("bintocsv", "eve")
    case_runner("csvtobin", "eve")


def test_fmpolicytc():
    case_runner("bintocsv", "fmpolicytc")
    case_runner("csvtobin", "fmpolicytc")
