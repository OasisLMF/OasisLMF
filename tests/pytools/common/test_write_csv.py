"""Tests for write_ndarray_to_fmt_csv — correctness, output parity, and performance."""
import io
import timeit

import numpy as np
import pytest

from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.elt.data import (
    MELT_headers, MELT_dtype, MELT_fmt,
    SELT_headers, SELT_dtype, SELT_fmt,
)
from oasislmf.pytools.plt.data import MPLT_headers, MPLT_dtype, MPLT_fmt


def _write(data, headers, fmt, use_cython=False):
    buf = io.StringIO()
    write_ndarray_to_fmt_csv(buf, data, headers, fmt, use_cython=use_cython)
    return buf.getvalue()


def _make_data(dtype, n, seed=42):
    rng = np.random.default_rng(seed)
    data = np.empty(n, dtype=dtype)
    for col in dtype.names:
        dt = dtype.fields[col][0]
        if np.issubdtype(dt, np.integer):
            hi = min(int(np.iinfo(dt).max), 10_000)
            data[col] = rng.integers(1, hi, size=n, dtype=dt)
        else:
            data[col] = rng.uniform(0.0, 100_000.0, size=n).astype(dt)
    return data


# Simple two-column dtype used in correctness tests.
# All values are chosen to be exactly representable in float32.
_SIMPLE_DTYPE = np.dtype([('X', np.int32), ('Y', np.float32)])
_SIMPLE_HEADERS = ['X', 'Y']
_SIMPLE_FMT = '%d,%.2f'


class TestCorrectOutput:
    """Both implementations produce valid, exactly correct CSV content."""

    @pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
    def test_zeros(self, use_cython):
        data = np.array([(0, 0.0)], dtype=_SIMPLE_DTYPE)
        assert _write(data, _SIMPLE_HEADERS, _SIMPLE_FMT, use_cython) == '0,0.00\n'

    @pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
    def test_single_row(self, use_cython):
        data = np.array([(7, 1.5)], dtype=_SIMPLE_DTYPE)
        assert _write(data, _SIMPLE_HEADERS, _SIMPLE_FMT, use_cython) == '7,1.50\n'

    @pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
    def test_multiple_rows(self, use_cython):
        data = np.array([(1, 0.0), (2, 1.5), (3, 100.25)], dtype=_SIMPLE_DTYPE)
        assert _write(data, _SIMPLE_HEADERS, _SIMPLE_FMT, use_cython) == '1,0.00\n2,1.50\n3,100.25\n'

    @pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
    def test_negative_values(self, use_cython):
        data = np.array([(-1, -1.5), (-100, -0.25)], dtype=_SIMPLE_DTYPE)
        assert _write(data, _SIMPLE_HEADERS, _SIMPLE_FMT, use_cython) == '-1,-1.50\n-100,-0.25\n'

    @pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
    def test_large_integer(self, use_cython):
        data = np.array([(9999, 0.5)], dtype=_SIMPLE_DTYPE)
        assert _write(data, _SIMPLE_HEADERS, _SIMPLE_FMT, use_cython) == '9999,0.50\n'

    @pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
    def test_trailing_newline(self, use_cython):
        data = np.array([(1, 1.0)], dtype=_SIMPLE_DTYPE)
        assert _write(data, _SIMPLE_HEADERS, _SIMPLE_FMT, use_cython).endswith('\n')

    @pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
    def test_high_precision_float(self, use_cython):
        """%.6f precision: 0.125 = 1/8 is exactly representable in float32."""
        dtype = np.dtype([('A', np.int32), ('B', np.float32)])
        data = np.array([(1, 0.125)], dtype=dtype)
        assert _write(data, ['A', 'B'], '%d,%.6f', use_cython) == '1,0.125000\n'

    @pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
    def test_mixed_precisions(self, use_cython):
        """Multiple float precisions in a single format string."""
        dtype = np.dtype([('A', np.int32), ('B', np.float32), ('C', np.float32)])
        data = np.array([(5, 0.25, 0.5)], dtype=dtype)
        assert _write(data, ['A', 'B', 'C'], '%d,%.4f,%.2f', use_cython) == '5,0.2500,0.50\n'


class TestErrorHandling:
    def test_header_fmt_mismatch_python(self):
        data = np.array([(1, 2.0)], dtype=_SIMPLE_DTYPE)
        with pytest.raises(RuntimeError):
            _write(data, _SIMPLE_HEADERS, '%d', use_cython=False)

    def test_header_fmt_mismatch_cython(self):
        data = np.array([(1, 2.0)], dtype=_SIMPLE_DTYPE)
        with pytest.raises(RuntimeError):
            _write(data, _SIMPLE_HEADERS, '%d', use_cython=True)


@pytest.mark.parametrize('label,headers,dtype,fmt', [
    ('MELT', MELT_headers, MELT_dtype, MELT_fmt),   # 3 ints + 8 floats @ %.6f
    ('SELT', SELT_headers, SELT_dtype, SELT_fmt),   # 3 ints + 2 floats @ %.2f
    ('MPLT', MPLT_headers, MPLT_dtype, MPLT_fmt),   # mixed %.6f / %.4f / %.2f
])
def test_python_cython_parity(label, headers, dtype, fmt):
    """Cython and Python produce byte-identical output on real-world dtypes."""
    data = _make_data(dtype, 500)
    python_out = _write(data, headers, fmt, use_cython=False)
    cython_out = _write(data, headers, fmt, use_cython=True)
    assert python_out == cython_out, f"{label}: Python and Cython outputs differ"


@pytest.mark.parametrize('label,headers,dtype,fmt', [
    ('MELT', MELT_headers, MELT_dtype, MELT_fmt),
    ('MPLT', MPLT_headers, MPLT_dtype, MPLT_fmt),
])
def test_cython_speedup(label, headers, dtype, fmt):
    """Cython implementation must be at least 2× faster than Python at 50k rows."""
    N = 50_000
    data = _make_data(dtype, N)

    def run(use_cython):
        buf = io.StringIO()
        write_ndarray_to_fmt_csv(buf, data, headers, fmt, use_cython=use_cython)

    # Warmup
    run(False)
    run(True)

    python_t = min(timeit.repeat(lambda: run(False), repeat=5, number=1))
    cython_t = min(timeit.repeat(lambda: run(True), repeat=5, number=1))
    speedup = python_t / cython_t

    assert speedup >= 2.0, (
        f"{label}: expected >=2x speedup, got {speedup:.2f}x "
        f"(python={python_t * 1000:.1f}ms, cython={cython_t * 1000:.1f}ms)"
    )
