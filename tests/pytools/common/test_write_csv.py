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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(data, headers, fmt, use_cython=False):
    buf = io.StringIO()
    write_ndarray_to_fmt_csv(buf, data, headers, fmt, use_cython=use_cython)
    return buf.getvalue()


def _parity(fmt, value):
    """Assert Cython output == Python output for a single float64 cell."""
    data = np.array([(value,)], dtype=np.dtype([('V', np.float64)]))
    py_out = _write(data, ['V'], fmt, use_cython=False)
    cy_out = _write(data, ['V'], fmt, use_cython=True)
    assert cy_out == py_out, (
        f"fmt={fmt!r} value={value!r}: cython={cy_out!r} != python={py_out!r}"
    )


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


# ---------------------------------------------------------------------------
# Section 1: Basic output structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
def test_basic_output(use_cython):
    """Correct CSV structure: values, commas, row-separating newlines, trailing newline."""
    dtype = np.dtype([('X', np.int32), ('Y', np.float32)])
    data = np.array([(1, 0.0), (2, 1.5), (-3, 100.25)], dtype=dtype)
    assert _write(data, ['X', 'Y'], '%d,%.2f', use_cython) == '1,0.00\n2,1.50\n-3,100.25\n'


@pytest.mark.parametrize('use_cython', [False, True], ids=['python', 'cython'])
def test_empty_data(use_cython):
    """Zero-row input produces only a trailing newline."""
    dtype = np.dtype([('X', np.int32), ('Y', np.float32)])
    data = np.empty(0, dtype=dtype)
    assert _write(data, ['X', 'Y'], '%d,%.2f', use_cython) == '\n'


# ---------------------------------------------------------------------------
# Section 2: COL_INT fast path (%d / %i / %u)
# ---------------------------------------------------------------------------

_COL_INT_FMTS = ['%d', '%i', '%u']

_INT_VALS = [
    0.0,
    1.0, -1.0,
    42.0, -42.0,
    9999.0, -9999.0,
    1_000_000.0,
    3.7, -3.7,    # truncation toward zero → '3' / '-3'
    -0.9,         # truncation → '0'
]


@pytest.mark.parametrize('fmt', _COL_INT_FMTS)
@pytest.mark.parametrize('value', _INT_VALS)
def test_col_int_parity(fmt, value):
    _parity(fmt, value)


@pytest.mark.parametrize('fmt', _COL_INT_FMTS)
@pytest.mark.parametrize('special', [np.nan, np.inf, -np.inf])
def test_col_int_special_same_exception(fmt, special):
    """Cython raises the same exception type as Python for NaN/Inf (was: C UB)."""
    data = np.array([(special,)], dtype=np.dtype([('V', np.float64)]))
    py_exc = cy_exc = None
    try:
        _write(data, ['V'], fmt, use_cython=False)
    except Exception as e:
        py_exc = type(e)
    try:
        _write(data, ['V'], fmt, use_cython=True)
    except Exception as e:
        cy_exc = type(e)
    assert py_exc is not None, "Python path should raise"
    assert py_exc == cy_exc, f"Python raised {py_exc.__name__}, Cython raised {getattr(cy_exc, '__name__', cy_exc)}"


@pytest.mark.parametrize('fmt', _COL_INT_FMTS)
def test_col_int_large_value_parity(fmt):
    """Values above LLONG_MAX fall through to Python's arbitrary-precision %d (was: C UB)."""
    _parity(fmt, 1e19)
    _parity(fmt, -1e19)


# ---------------------------------------------------------------------------
# Section 3: COL_FIXED fast path (%.1f .. %.9f)
# ---------------------------------------------------------------------------

_COL_FIXED_PRECISIONS = list(range(1, 10))

_FLOAT_VALS = [
    0.0,
    1.0, -1.0,
    0.5, -0.5,
    1.5, -1.5,
    2.5, -2.5,          # round-half-to-even
    42.0, -42.0,
    0.1, -0.1,
    0.123456789,
    100.25, -100.25,
    1e6, -1e6,
    1e9, -1e9,
    np.float64('nan'),
    np.float64('inf'),
    np.float64('-inf'),
]


@pytest.mark.parametrize('prec', _COL_FIXED_PRECISIONS)
@pytest.mark.parametrize('value', _FLOAT_VALS)
def test_col_fixed_parity(prec, value):
    """COL_FIXED matches Python's %.Xf including nan/inf handling."""
    _parity(f'%.{prec}f', value)


@pytest.mark.parametrize('prec', _COL_FIXED_PRECISIONS)
def test_col_fixed_negative_zero(prec):
    """-0.0 produces '-0.xxx' on both paths (copysign fix)."""
    data = np.array([(-0.0,)], dtype=np.dtype([('V', np.float64)]))
    py_out = _write(data, ['V'], f'%.{prec}f', use_cython=False)
    cy_out = _write(data, ['V'], f'%.{prec}f', use_cython=True)
    assert py_out.startswith('-0.'), f"Python produced {py_out!r}"
    assert cy_out == py_out, f"cython={cy_out!r} != python={py_out!r}"


@pytest.mark.parametrize('prec', _COL_FIXED_PRECISIONS)
def test_col_fixed_overflow_threshold_parity(prec):
    """Values above 9.2e18/10^prec fall back to Python % (was: C UB via long long overflow)."""
    threshold = 9.2e18 / (10 ** prec)
    overflow_val = threshold * 1.01
    _parity(f'%.{prec}f', overflow_val)
    _parity(f'%.{prec}f', -overflow_val)


@pytest.mark.parametrize('value', [0.5, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5])
def test_col_fixed_rounding_half_parity(value):
    """Rounding at .5 boundaries matches Python (IEEE 754 round-half-to-even)."""
    for prec in _COL_FIXED_PRECISIONS:
        _parity(f'%.{prec}f', value)


# ---------------------------------------------------------------------------
# Section 4: COL_PYTHON fallback
# ---------------------------------------------------------------------------

_PYTHON_FLOAT_FMTS = [
    '%f',
    '%.0f',         # 0 d.p. — below COL_FIXED range
    '%.10f',        # above COL_FIXED range
    '%.15f',
    '%e', '%E',
    '%.2e', '%.2E',
    '%g', '%G',
    '%.4g', '%.4G',
    '%10.2f',       # width
    '%-10.2f',      # left-aligned
    '%010.2f',      # zero-padded
    '%+.2f',        # always sign
    '% .2f',        # space for positive
    '%0.9lf',       # length modifier (period_weighting in data.py)
]

_PYTHON_FLOAT_VALS = [
    0.0, 1.0, -1.0, 42.0, -42.0,
    0.123456789, 100.25, -100.25,
    1e6, 1e-6, 1e9, -1e9,
    np.float64('nan'), np.float64('inf'), np.float64('-inf'),
]


@pytest.mark.parametrize('fmt', _PYTHON_FLOAT_FMTS)
@pytest.mark.parametrize('value', _PYTHON_FLOAT_VALS)
def test_python_fallback_float_parity(fmt, value):
    _parity(fmt, value)


_PYTHON_INT_FMTS = ['%10d', '%-10d', '%010d', '%+d', '% d', '%+i', '%-8i', '%8u', '%-8u']
_PYTHON_INT_VALS = [0.0, 1.0, -1.0, 42.0, -42.0, 9999.0, -9999.0, 3.7, -3.7]


@pytest.mark.parametrize('fmt', _PYTHON_INT_FMTS)
@pytest.mark.parametrize('value', _PYTHON_INT_VALS)
def test_python_fallback_int_parity(fmt, value):
    _parity(fmt, value)


# ---------------------------------------------------------------------------
# Section 5: Multi-column / integration
# ---------------------------------------------------------------------------

def test_multirow_all_format_classes():
    """All three format classes in one dataset produce identical output."""
    n = 200
    rng = np.random.default_rng(1)
    dtype = np.dtype([
        ('a', np.float64),  # %d      → COL_INT
        ('b', np.float64),  # %.3f    → COL_FIXED
        ('c', np.float64),  # %f      → COL_PYTHON
        ('d', np.float64),  # %e      → COL_PYTHON
        ('e', np.float64),  # %g      → COL_PYTHON
        ('f', np.float64),  # %10.4f  → COL_PYTHON (has width)
    ])
    data = np.empty(n, dtype=dtype)
    data['a'] = np.floor(rng.uniform(-9999.0, 9999.0, n))
    for col in ('b', 'c', 'd', 'e', 'f'):
        data[col] = rng.uniform(-1000.0, 1000.0, n)
    fmt = '%d,%.3f,%f,%e,%g,%10.4f'
    assert _write(data, list(dtype.names), fmt, False) == _write(data, list(dtype.names), fmt, True)


# ---------------------------------------------------------------------------
# Section 6: No column-count limit (was MAX_COLS = 64)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_cols', [65, 100, 128])
def test_more_than_64_columns_parity(n_cols):
    """Column counts above the old MAX_COLS=64 hard limit work correctly."""
    dtype = np.dtype([(f'c{i}', np.float64) for i in range(n_cols)])
    headers = [f'c{i}' for i in range(n_cols)]
    fmt = ','.join(['%.2f'] * n_cols)
    data = np.ones(10, dtype=dtype)
    assert _write(data, headers, fmt, False) == _write(data, headers, fmt, True)


# ---------------------------------------------------------------------------
# Section 7: Error handling
# ---------------------------------------------------------------------------

def test_header_fmt_mismatch():
    dtype = np.dtype([('X', np.int32), ('Y', np.float32)])
    data = np.array([(1, 2.0)], dtype=dtype)
    with pytest.raises(RuntimeError):
        _write(data, ['X', 'Y'], '%d', use_cython=False)
    with pytest.raises(RuntimeError):
        _write(data, ['X', 'Y'], '%d', use_cython=True)


def test_col_python_buffer_overflow_raises():
    """COL_PYTHON output exceeding the 128-char per-cell estimate raises RuntimeError.

    %.100f on 1e100 produces ~202 chars, exceeding the 128-char budget.
    Python path is unaffected (no buffer; pure string ops).
    """
    data = np.array([(1e100,)], dtype=np.dtype([('V', np.float64)]))
    with pytest.raises(RuntimeError, match="buffer overflow"):
        _write(data, ['V'], '%.100f', use_cython=True)
    # Python path handles it fine
    assert len(_write(data, ['V'], '%.100f', use_cython=False)) > 100


# ---------------------------------------------------------------------------
# Section 8: Real-world dtype parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label,headers,dtype,fmt', [
    ('MELT', MELT_headers, MELT_dtype, MELT_fmt),
    ('SELT', SELT_headers, SELT_dtype, SELT_fmt),
    ('MPLT', MPLT_headers, MPLT_dtype, MPLT_fmt),
])
def test_realworld_dtype_parity(label, headers, dtype, fmt):
    data = _make_data(dtype, 500)
    py_out = _write(data, headers, fmt, use_cython=False)
    cy_out = _write(data, headers, fmt, use_cython=True)
    assert py_out == cy_out, f"{label}: Python and Cython outputs differ"


# ---------------------------------------------------------------------------
# Section 9: Performance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label,headers,dtype,fmt', [
    ('MELT', MELT_headers, MELT_dtype, MELT_fmt),
    ('MPLT', MPLT_headers, MPLT_dtype, MPLT_fmt),
])
def test_cython_speedup(label, headers, dtype, fmt):
    """Cython must be at least 2× faster than Python at 50k rows."""
    N = 50_000
    data = _make_data(dtype, N)

    def run(use_cython):
        buf = io.StringIO()
        write_ndarray_to_fmt_csv(buf, data, headers, fmt, use_cython=use_cython)

    run(False)
    run(True)  # warmup

    python_t = min(timeit.repeat(lambda: run(False), repeat=5, number=1))
    cython_t = min(timeit.repeat(lambda: run(True), repeat=5, number=1))
    speedup = python_t / cython_t

    assert speedup >= 2.0, (
        f"{label}: expected >=2x speedup, got {speedup:.2f}x "
        f"(python={python_t * 1000:.1f}ms, cython={cython_t * 1000:.1f}ms)"
    )
