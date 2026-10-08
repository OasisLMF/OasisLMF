from pathlib import Path

from tests.pytools.converters.helpers import TESTS_ASSETS_DIR, case_runner


def test_aal():
    # Test bin/csv
    case_runner("bintocsv", "aal", "output")
    case_runner("csvtobin", "aal", "output")
    case_runner("bintocsv", "aalmeanonly", "output")
    case_runner("csvtobin", "aalmeanonly", "output")
    case_runner("bintocsv", "alct", "output")
    case_runner("csvtobin", "alct", "output")

    # Test bin/parquet
    case_runner("bintoparquet", "aal", "output")
    case_runner("parquettobin", "aal", "output")
    case_runner("bintoparquet", "aalmeanonly", "output")
    case_runner("parquettobin", "aalmeanonly", "output")
    case_runner("bintoparquet", "alct", "output")
    case_runner("parquettobin", "alct", "output")


def test_elt():
    # Test bin/csv
    case_runner("bintocsv", "selt", "output")
    case_runner("csvtobin", "selt", "output")
    case_runner("bintocsv", "melt", "output")
    case_runner("csvtobin", "melt", "output")
    case_runner("bintocsv", "qelt", "output")
    case_runner("csvtobin", "qelt", "output")

    # Test bin/parquet
    case_runner("bintoparquet", "selt", "output")
    case_runner("parquettobin", "selt", "output")
    case_runner("bintoparquet", "melt", "output")
    case_runner("parquettobin", "melt", "output")
    case_runner("bintoparquet", "qelt", "output")
    case_runner("parquettobin", "qelt", "output")


def test_lec():
    # Test bin/csv
    case_runner("bintocsv", "ept", "output")
    case_runner("csvtobin", "ept", "output")
    case_runner("bintocsv", "psept", "output")
    case_runner("csvtobin", "psept", "output")

    # Test bin/parquet
    case_runner("bintoparquet", "ept", "output")
    case_runner("parquettobin", "ept", "output")
    case_runner("bintoparquet", "psept", "output")
    case_runner("parquettobin", "psept", "output")


def test_plt():
    # Test bin/csv
    case_runner("bintocsv", "splt", "output")
    case_runner("csvtobin", "splt", "output")
    case_runner("bintocsv", "mplt", "output")
    case_runner("csvtobin", "mplt", "output")
    case_runner("bintocsv", "qplt", "output")
    case_runner("csvtobin", "qplt", "output")

    # Test bin/parquet
    case_runner("bintoparquet", "splt", "output")
    case_runner("parquettobin", "splt", "output")
    case_runner("bintoparquet", "mplt", "output")
    case_runner("parquettobin", "mplt", "output")
    case_runner("bintoparquet", "qplt", "output")
    case_runner("parquettobin", "qplt", "output")


def test_fm():
    case_runner("bintocsv", "fm", "misc", "raw_ils")
    case_runner("csvtobin", "fm", "misc", "raw_ils", abnormal_dtype=True, stream_type=2, max_sample_index=1)


def test_gul():
    case_runner("bintocsv", "gul", "misc", "raw_guls")
    case_runner("csvtobin", "gul", "misc", "raw_guls", abnormal_dtype=True, stream_type=2, max_sample_index=1)


def test_summarycalc():
    case_runner("csvtobin", "summarycalc", "misc", "summary", abnormal_dtype=True, summary_set_id=1, max_sample_index=100)


def test_cdf():
    case_runner("bintocsv", "cdf", "cdftocsv", "getmodel", run_dir=Path(TESTS_ASSETS_DIR, "cdftocsv"))
