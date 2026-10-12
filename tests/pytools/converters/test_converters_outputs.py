import os
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import numpy as np

from oasislmf.pytools.converters.bintoparquet.manager import bintoparquet
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.converters.parquettobin.manager import parquettobin
from oasislmf.utils.exceptions import OasisException
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


def test_parquettobin_rejects_null_in_int_column():
    # a null in a non-float column casts through NaN into the target dtype's bit pattern
    # (e.g. INT_MIN for an int column) instead of raising
    with TemporaryDirectory() as tmp:
        table = pa.table({"event_id": pa.array([1, None, 3], type=pa.int32())})
        pq.write_table(table, Path(tmp, "in.parquet"))
        with pytest.raises(OasisException, match="null value"):
            parquettobin(Path(tmp, "in.parquet"), Path(tmp, "out.bin"), "eve")


def test_bintoparquet_removes_output_on_truncated_input():
    # parquet is self-validating per row group, so a conversion that fails partway through
    # closes into a structurally valid, readable file with silently fewer rows -- unlike a
    # truncated flat binary file, which is visibly wrong on the next read.
    with TemporaryDirectory() as tmp:
        dtype = TOOL_INFO["eve"]["dtype"]
        with open(Path(tmp, "in.bin"), "wb") as f:
            f.write(np.array([1, 2, 3], dtype=dtype).tobytes())
            f.write(b"\x01\x02")  # trailing partial record
        with pytest.raises(ValueError, match="multiple of element size"):
            bintoparquet(Path(tmp, "in.bin"), Path(tmp, "out.parquet"), "eve")

        assert not Path(tmp, "out.parquet").exists()


def test_parquettobin_gives_clear_error_on_piped_stdin():
    # the parquet footer is read by seeking to the end of the file, which a pipe cannot support;
    # this used to surface as a bare "OSError: [Errno 29] Illegal seek"
    with TemporaryDirectory() as tmp:
        dtype = TOOL_INFO["selt"]["dtype"]
        table = pa.table({name: pa.array([1], type=pa.from_numpy_dtype(dtype[name])) for name in dtype.names})
        pq.write_table(table, Path(tmp, "in.parquet"))
        parquet_bytes = Path(tmp, "in.parquet").read_bytes()
        out_path = Path(tmp, "out.bin")

        read_fd, write_fd = os.pipe()
        writer = threading.Thread(target=lambda: (os.write(write_fd, parquet_bytes), os.close(write_fd)))
        writer.start()
        try:
            with pytest.raises(OasisException, match="must be a seekable file"):
                parquettobin(f"/dev/fd/{read_fd}", out_path, "selt")
        finally:
            writer.join()
            os.close(read_fd)


def test_parquettobin_allows_null_in_float_column():
    # a null in a float column legitimately becomes NaN; must not be rejected
    with TemporaryDirectory() as tmp:
        table = pa.table({
            "event_id": pa.array([1, 1, 1], type=pa.int32()),
            "amplification_id": pa.array([1, 2, 3], type=pa.int32()),
            "factor": pa.array([1.5, None, 3.5], type=pa.float32()),
        })
        pq.write_table(table, Path(tmp, "in.parquet"))
        parquettobin(Path(tmp, "in.parquet"), Path(tmp, "out.bin"), "lossfactors")
