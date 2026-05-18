import io
import logging
import timeit
from unittest.mock import patch

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


def _parity(fmt, value, dtype=np.float64):
    """Assert Cython output == Python output for a single cell of the given dtype."""
    data = np.array([(value,)], dtype=np.dtype([('V', dtype)]))
    py_out = _write(data, ['V'], fmt, use_cython=False)
    cy_out = _write(data, ['V'], fmt, use_cython=True)
    assert cy_out == py_out, (
        f"fmt={fmt!r} value={value!r} dtype={np.dtype(dtype)}: cython={cy_out!r} != python={py_out!r}"
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
_UINT_VALS = [v for v in _INT_VALS if v >= 0]


@pytest.mark.parametrize('dtype,values', [
    (np.float64, _INT_VALS),
    (np.float32, _INT_VALS),
    (np.int32, _INT_VALS),
    (np.uint32, _UINT_VALS),   # uint32 cannot represent negative values
    (np.int64, _INT_VALS),
], ids=['f64', 'f32', 'i32', 'u32', 'i64'])
@pytest.mark.parametrize('fmt', _COL_INT_FMTS)
def test_col_int_parity(fmt, dtype, values):
    """COL_INT fast path across all native dtypes matches Python output."""
    for v in values:
        _parity(fmt, v, dtype)


@pytest.mark.parametrize('fmt', _COL_INT_FMTS)
@pytest.mark.parametrize('special', [np.nan, np.inf, -np.inf])
def test_col_int_special_same_exception(fmt, special):
    """COL_INT raises the same exception type as Python for NaN/Inf input."""
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
    """Values above LLONG_MAX fall through to Python's arbitrary-precision % operator."""
    _parity(fmt, 1e19)
    _parity(fmt, -1e19)


def test_col_int_llong_min_parity():
    """LLONG_MIN (-9223372036854775808) is special-cased in write_int.

    -LLONG_MIN is C signed-integer overflow; Cython detects it before the digit
    loop and emits the precomputed literal string directly.
    """
    val = np.iinfo(np.int64).min  # -9223372036854775808
    dtype = np.dtype([('V', np.int64)])
    data = np.array([(val,)], dtype=dtype)
    py_out = _write(data, ['V'], '%d', use_cython=False)
    cy_out = _write(data, ['V'], '%d', use_cython=True)
    assert cy_out == py_out, f"cython={cy_out!r} != python={py_out!r}"
    assert py_out.strip() == '-9223372036854775808'


def test_col_int_u_format_negative_parity():
    """%u with negative signed-int columns matches Python 3.

    Python 3 treats %u as equivalent to %d (both output the signed decimal).
    Verifies Cython does the same rather than treating the value as unsigned.
    """
    dtype = np.dtype([('V', np.int32)])
    data = np.array([(-1,), (-42,), (0,), (2147483647,)], dtype=dtype)
    assert _write(data, ['V'], '%u', use_cython=False) == _write(data, ['V'], '%u', use_cython=True)


# ---------------------------------------------------------------------------
# Section 3: COL_FIXED fast path (%.Xf, precision 0–15)
# ---------------------------------------------------------------------------

_COL_FIXED_PRECISIONS = list(range(0, 16))  # 0 through 15 inclusive

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


@pytest.mark.parametrize('dtype,values', [
    (np.float64, _FLOAT_VALS),
    (np.float32, _FLOAT_VALS),
], ids=['f64', 'f32'])
@pytest.mark.parametrize('prec', _COL_FIXED_PRECISIONS)
def test_col_fixed_parity(prec, dtype, values):
    """COL_FIXED matches Python's %.Xf including nan/inf handling."""
    for v in values:
        _parity(f'%.{prec}f', v, dtype)


@pytest.mark.parametrize('prec', _COL_FIXED_PRECISIONS)
def test_col_fixed_negative_zero(prec):
    """-0.0 produces '-0' or '-0.xxx' on both paths.

    prec=0 produces '-0' (no dot); prec>=1 produces '-0.000...'.
    Cython uses copysign to detect the sign bit, matching Python's output.
    """
    data = np.array([(-0.0,)], dtype=np.dtype([('V', np.float64)]))
    py_out = _write(data, ['V'], f'%.{prec}f', use_cython=False)
    cy_out = _write(data, ['V'], f'%.{prec}f', use_cython=True)
    assert py_out.rstrip('\n').startswith('-0'), f"Python produced {py_out!r}"
    assert cy_out == py_out, f"cython={cy_out!r} != python={py_out!r}"


@pytest.mark.parametrize('prec', _COL_FIXED_PRECISIONS)
def test_col_fixed_overflow_threshold_parity(prec):
    """Values above 9.2e18/10^prec exceed the long long range and fall back to Python %."""
    threshold = 9.2e18 / (10 ** prec)
    overflow_val = threshold * 1.01
    _parity(f'%.{prec}f', overflow_val)
    _parity(f'%.{prec}f', -overflow_val)


@pytest.mark.parametrize('dtype', [np.float64, np.float32], ids=['f64', 'f32'])
@pytest.mark.parametrize('value', [0.5, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5])
def test_col_fixed_rounding_half_parity(value, dtype):
    """Rounding at .5 boundaries matches Python (IEEE 754 round-half-to-even)."""
    for prec in _COL_FIXED_PRECISIONS:
        _parity(f'%.{prec}f', value, dtype)


@pytest.mark.parametrize('value,prec', [
    (-0.4, 0),   # |val| < 0.5 → rounds to ±0; IEEE 754 rint(-0.4) = -0.0 → '-0'
    (-0.49, 0),
    (-0.004, 2),   # rint(-0.004*100) = rint(-0.4) = -0.0 → '-0.00'
    (-0.0049, 2),
    (-4e-7, 6),   # rint(-4e-7 * 1e6) = rint(-0.4) = -0.0 → '-0.000000'
])
def test_col_fixed_negative_rounds_to_zero_parity(value, prec):
    """Negative values whose magnitude rounds to zero match Python output.

    IEEE 754 rint(-x) = -0.0 for 0 < x < 0.5, so printf('%.0f', -0.4) = '-0'
    on glibc.  The Cython copysign approach produces the same result.
    """
    _parity(f'%.{prec}f', value)


@pytest.mark.parametrize('int_dtype,values', [
    (np.int32, [100, -1, 0, 2147483647, -2147483648]),
    (np.uint32, [0, 1, 200, 4294967295]),
    (np.int64, [100, -1, 0, 9223372036854775807, -9223372036854775808]),
])
def test_col_fixed_integer_dtype_parity(int_dtype, values):
    """COL_FIXED (%.Xf) on integer-dtype columns reads the correct 4/8-byte value.

    Each integer dtype has a dedicated pointer cast in the hot loop; falling through
    to the float64 branch would read the wrong number of bytes from the struct field.
    """
    dtype = np.dtype([('V', int_dtype)])
    data = np.array([(v,) for v in values], dtype=dtype)
    for prec in (0, 2, 6):
        fmt = f'%.{prec}f'
        py_out = _write(data, ['V'], fmt, use_cython=False)
        cy_out = _write(data, ['V'], fmt, use_cython=True)
        assert cy_out == py_out, (
            f"dtype={int_dtype.__name__} fmt={fmt!r}: cython={cy_out!r} != python={py_out!r}"
        )


# ---------------------------------------------------------------------------
# Section 4a: COL_FIXED additional formats (bare %f, sign flags, length modifiers)
# ---------------------------------------------------------------------------

_COL_FIXED_EXTENDED_FMTS = [
    '%f',       # bare %f → prec=6
    '%.0f',     # prec=0: no decimal point
    '%.10f',    # prec=10: high precision, narrow overflow threshold
    '%.15f',    # prec=15: maximum supported precision
    '%0.9lf',   # zero-flag + length modifier stripped → prec=9
    '%+.2f',    # sign flag: always '+'
    '% .2f',    # sign flag: space for positive
]

_COL_FIXED_EXTENDED_VALS = [
    0.0, 1.0, -1.0, 42.0, -42.0,
    0.123456789, 100.25, -100.25,
    1e6, 1e-6, 1e9, -1e9,
    np.float64('nan'), np.float64('inf'), np.float64('-inf'),
]


@pytest.mark.parametrize('dtype,values', [
    (np.float64, _COL_FIXED_EXTENDED_VALS),
    (np.float32, _COL_FIXED_EXTENDED_VALS),
], ids=['f64', 'f32'])
@pytest.mark.parametrize('fmt', _COL_FIXED_EXTENDED_FMTS)
def test_col_fixed_extended_parity(fmt, dtype, values):
    """Extended COL_FIXED formats match Python output."""
    for v in values:
        _parity(fmt, v, dtype)


@pytest.mark.parametrize('fmt,canonical', [
    ('%lf', '%f'),
    ('%hf', '%f'),
    ('%Lf', '%f'),
    ('%llf', '%f'),
    ('%.2lf', '%.2f'),
    ('%.4llf', '%.4f'),
])
def test_col_fixed_length_modifier_parity(fmt, canonical):
    """C length modifiers (l/h/L/ll) before f are stripped → same Cython output as canonical.

    Python's % operator doesn't accept %llf/%hf/%Lf, so this tests Cython-only:
    the modifier is stripped during column classification, giving the same COL_FIXED
    fast path and identical output as the unmodified canonical format.
    """
    vals = [0.0, 1.0, -1.0, 42.5, np.float64('nan'), np.float64('inf')]
    for v in vals:
        data = np.array([(v,)], dtype=np.dtype([('V', np.float64)]))
        cy_out = _write(data, ['V'], fmt, use_cython=True)
        canon_out = _write(data, ['V'], canonical, use_cython=True)
        assert cy_out == canon_out, (
            f"fmt={fmt!r} value={v!r}: Cython output {cy_out!r} differs from canonical {canon_out!r}"
        )


# ---------------------------------------------------------------------------
# Section 4b: COL_INT with sign flags
# ---------------------------------------------------------------------------

_COL_INT_SIGN_FMTS = ['%+d', '% d', '%+i', '% i', '%+u', '% u']

_COL_INT_SIGN_VALS = [0.0, 1.0, -1.0, 42.0, -42.0, 9999.0, -9999.0, 3.7, -3.7]
_UINT_SIGN_VALS = [v for v in _COL_INT_SIGN_VALS if v >= 0]


@pytest.mark.parametrize('dtype,values', [
    (np.float64, _COL_INT_SIGN_VALS),
    (np.float32, _COL_INT_SIGN_VALS),
    (np.int32, _COL_INT_SIGN_VALS),
    (np.uint32, _UINT_SIGN_VALS),   # uint32 cannot represent negative values
    (np.int64, _COL_INT_SIGN_VALS),
], ids=['f64', 'f32', 'i32', 'u32', 'i64'])
@pytest.mark.parametrize('fmt', _COL_INT_SIGN_FMTS)
def test_col_int_sign_flag_parity(fmt, dtype, values):
    """Sign-flag COL_INT matches Python across all native dtypes."""
    for v in values:
        _parity(fmt, v, dtype)


@pytest.mark.parametrize('fmt', _COL_INT_SIGN_FMTS)
@pytest.mark.parametrize('special', [np.nan, np.inf, -np.inf])
def test_col_int_sign_flag_special_same_exception(fmt, special):
    """Sign-flag COL_INT raises the same exception type as Python for NaN/Inf input."""
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


def test_col_int_sign_flag_neg_zero():
    """%+d / % d with -0.0: IEEE 754 -0.0 == 0.0, so gets '+0' / ' 0' (not '-0').

    COL_INT uses >= 0.0 (not copysign) so -0.0 is treated as non-negative,
    matching Python's behaviour: '%+d' % -0.0 == '%+d' % 0 == '+0'.
    """
    data = np.array([(-0.0,)], dtype=np.dtype([('V', np.float64)]))
    for fmt in ('%+d', '% d'):
        _parity(fmt, -0.0)
        cy_out = _write(data, ['V'], fmt, use_cython=True).rstrip('\n')
        assert cy_out[0] in ('+', ' '), f"{fmt}: expected sign prefix, got {cy_out!r}"


# ---------------------------------------------------------------------------
# Section 4c: Sign flag edge cases for COL_FIXED
# ---------------------------------------------------------------------------

def test_col_fixed_sign_flag_neg_zero():
    """%+.2f / % .2f with -0.0 → '-0.00', not '+0.00' / ' 0.00'.

    COL_FIXED uses copysign to check sign, so -0.0 (negative sign bit) is
    NOT treated as positive — write_float emits '-0.xx', matching Python.
    Compare with COL_INT where -0.0 >= 0.0 is True and gets a sign prefix.
    """
    data = np.array([(-0.0,)], dtype=np.dtype([('V', np.float64)]))
    for fmt in ('%+.2f', '% .2f'):
        py_out = _write(data, ['V'], fmt, use_cython=False)
        cy_out = _write(data, ['V'], fmt, use_cython=True)
        assert cy_out == py_out, f"{fmt}: cython={cy_out!r} != python={py_out!r}"
        # Python emits '-0.00' (not '+0.00' or ' 0.00')
        assert py_out.rstrip('\n') == '-0.00', f"{fmt}: expected '-0.00', got {py_out!r}"


@pytest.mark.parametrize('fmt', ['%+.2f', '% .2f', '%+f', '% f'])
def test_col_fixed_sign_flag_special_parity(fmt):
    """COL_FIXED sign flag with nan/inf matches Python output.

    nan and positive inf trigger the sign-prefix branch (isnan or copysign > 0);
    -inf has a negative copysign result and is written without a prefix, both
    matching Python's behaviour.
    """
    for special in (np.float64('nan'), np.float64('inf'), np.float64('-inf')):
        _parity(fmt, special)


# ---------------------------------------------------------------------------
# Section 4d: Remaining COL_PYTHON fallback formats (width/alignment, %e, %g)
# ---------------------------------------------------------------------------

_PYTHON_FLOAT_FMTS = [
    '%e', '%E',
    '%.2e', '%.2E',
    '%g', '%G',
    '%.4g', '%.4G',
    '%10.2f',       # width field → COL_PYTHON
    '%-10.2f',      # left-aligned
    '%010.2f',      # zero-padded width
]

_PYTHON_FLOAT_VALS = [
    0.0, 1.0, -1.0, 42.0, -42.0,
    0.123456789, 100.25, -100.25,
    1e6, 1e-6, 1e9, -1e9,
    np.float64('nan'), np.float64('inf'), np.float64('-inf'),
]


@pytest.mark.parametrize('dtype,values', [
    (np.float64, _PYTHON_FLOAT_VALS),
    (np.float32, _PYTHON_FLOAT_VALS),
], ids=['f64', 'f32'])
@pytest.mark.parametrize('fmt', _PYTHON_FLOAT_FMTS)
def test_python_fallback_float_parity(fmt, dtype, values):
    """COL_PYTHON float formats match Python output for both float64 and float32."""
    for v in values:
        _parity(fmt, v, dtype)


_PYTHON_INT_FMTS = ['%10d', '%-10d', '%010d', '%-8i', '%8u', '%-8u']
_PYTHON_INT_VALS = [0.0, 1.0, -1.0, 42.0, -42.0, 9999.0, -9999.0, 3.7, -3.7]
_UINT_INT_VALS = [v for v in _PYTHON_INT_VALS if v >= 0]


@pytest.mark.parametrize('dtype,values', [
    (np.float64, _PYTHON_INT_VALS),
    (np.int32, _PYTHON_INT_VALS),
    (np.uint32, _UINT_INT_VALS),   # uint32 cannot represent negative values
    (np.int64, _PYTHON_INT_VALS),
], ids=['f64', 'i32', 'u32', 'i64'])
@pytest.mark.parametrize('fmt', _PYTHON_INT_FMTS)
def test_python_fallback_int_parity(fmt, dtype, values):
    """COL_PYTHON integer formats (with width/alignment) match Python output across dtypes."""
    for v in values:
        _parity(fmt, v, dtype)


# ---------------------------------------------------------------------------
# Section 4e: Unusual dtype coercion to float64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.uint8, np.uint16, np.float16],
                         ids=['i8', 'i16', 'u8', 'u16', 'f16'])
@pytest.mark.parametrize('fmt', ['%d', '%.2f'])
def test_unusual_dtype_coerced_parity(fmt, dtype):
    """Dtypes not in {int32,uint32,int64,float32,float64} are coerced to float64 before the hot loop.

    Both COL_INT (%d) and COL_FIXED (%.2f) paths must read the coerced value correctly.
    """
    if np.issubdtype(dtype, np.unsignedinteger):
        values = [0, 1, 42, 100]
    else:
        values = [0, 1, -1, 42, -42]
    data = np.array([(v,) for v in values], dtype=np.dtype([('V', dtype)]))
    py_out = _write(data, ['V'], fmt, use_cython=False)
    cy_out = _write(data, ['V'], fmt, use_cython=True)
    assert cy_out == py_out, f"dtype={np.dtype(dtype)} fmt={fmt!r}: cython={cy_out!r} != python={py_out!r}"


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
        ('c', np.float64),  # %f      → COL_FIXED (bare %f = %.6f)
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
# Section 6: Column-count limit (MAX_COLS = 64)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_cols', [1, 10, 64])
def test_within_max_cols_parity(n_cols):
    """Column counts up to MAX_COLS=64 produce identical Cython and Python output."""
    dtype = np.dtype([(f'c{i}', np.float64) for i in range(n_cols)])
    headers = [f'c{i}' for i in range(n_cols)]
    fmt = ','.join(['%.2f'] * n_cols)
    data = np.ones(10, dtype=dtype)
    assert _write(data, headers, fmt, False) == _write(data, headers, fmt, True)


@pytest.mark.parametrize('n_cols', [65, 100, 128])
def test_above_max_cols_falls_back(n_cols, caplog):
    """Column counts above MAX_COLS=64: Cython raises ValueError and falls back to Python.

    The fallback produces correct output and logs a warning containing 'MAX_COLS'.
    """
    dtype = np.dtype([(f'c{i}', np.float64) for i in range(n_cols)])
    headers = [f'c{i}' for i in range(n_cols)]
    fmt = ','.join(['%.2f'] * n_cols)
    data = np.ones(10, dtype=dtype)
    expected = _write(data, headers, fmt, use_cython=False)
    with caplog.at_level(logging.WARNING, logger='oasislmf.pytools.common.data'):
        result = _write(data, headers, fmt, use_cython=True)
    assert result == expected
    assert any('MAX_COLS' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Section 7: Cython fallback on runtime error
# ---------------------------------------------------------------------------

_FALLBACK_DATA = np.array([(3, 1.5)], dtype=np.dtype([('X', np.int32), ('Y', np.float32)]))
_FALLBACK_HDRS = ['X', 'Y']
_FALLBACK_FMT = '%d,%.2f'


@pytest.mark.parametrize('exc_type', [RuntimeError, ValueError, MemoryError])
def test_cython_fallback_produces_correct_output(exc_type):
    """When the Cython writer raises, Python fallback produces the correct output."""
    expected = _write(_FALLBACK_DATA, _FALLBACK_HDRS, _FALLBACK_FMT, use_cython=False)

    with patch('oasislmf.pytools.common.data.cython_write_csv', side_effect=exc_type("boom")):
        result = _write(_FALLBACK_DATA, _FALLBACK_HDRS, _FALLBACK_FMT, use_cython=True)

    assert result == expected


def test_cython_fallback_logs_warning(caplog):
    """A logger.warning is emitted when the Cython writer falls back."""
    with patch('oasislmf.pytools.common.data.cython_write_csv', side_effect=RuntimeError("test error")):
        with caplog.at_level(logging.WARNING, logger='oasislmf.pytools.common.data'):
            _write(_FALLBACK_DATA, _FALLBACK_HDRS, _FALLBACK_FMT, use_cython=True)

    assert any(
        'RuntimeError' in r.message and 'falling back' in r.message
        for r in caplog.records
    ), f"Expected fallback warning, got: {[r.message for r in caplog.records]}"


def test_cython_fallback_no_partial_write():
    """Output file is empty if Cython raises before its single write() call.

    The Cython path buffers all output then calls output_file.write() exactly
    once at the end, so any earlier exception leaves the file untouched.
    """
    buf = io.StringIO()
    original_write = buf.write
    write_calls = []
    buf.write = lambda s: (write_calls.append(s), original_write(s))[1]

    with patch('oasislmf.pytools.common.data.cython_write_csv', side_effect=RuntimeError("early fail")):
        write_ndarray_to_fmt_csv(buf, _FALLBACK_DATA, _FALLBACK_HDRS, _FALLBACK_FMT, use_cython=True)

    # The only write call should be from the Python fallback, not a partial Cython write
    assert len(write_calls) <= 2  # Python path does at most 2 writes (data + '\n')
    assert buf.getvalue() != ''   # Python fallback did write something


def test_header_fmt_mismatch():
    """Mismatched header count and format field count raises RuntimeError on both paths."""
    dtype = np.dtype([('X', np.int32), ('Y', np.float32)])
    data = np.array([(1, 2.0)], dtype=dtype)
    with pytest.raises(RuntimeError):
        _write(data, ['X', 'Y'], '%d', use_cython=False)
    with pytest.raises(RuntimeError):
        _write(data, ['X', 'Y'], '%d', use_cython=True)


def test_col_python_buffer_overflow_falls_back(caplog):
    """A format that produces more bytes than the Cython buffer estimate triggers the fallback.

    The per-cell buffer estimate is generous but finite; a pathological format like
    %.100f on a large value exceeds it.  Cython raises RuntimeError, which the
    write_ndarray_to_fmt_csv wrapper catches and retries via the Python path.
    """
    data = np.array([(1e100,)], dtype=np.dtype([('V', np.float64)]))
    expected = _write(data, ['V'], '%.100f', use_cython=False)
    assert len(expected) > 100  # sanity: Python produces a long string

    with caplog.at_level(logging.WARNING, logger='oasislmf.pytools.common.data'):
        result = _write(data, ['V'], '%.100f', use_cython=True)

    assert result == expected
    assert any('falling back' in r.message for r in caplog.records)


def test_tls_buffer_reuse():
    """Thread-local buffer is reused (not reallocated) when already large enough.

    The reuse path: `if not hasattr(_tls, 'buf') or len(_tls.buf) < buf_size` is False,
    so _tls.buf is used as-is.  pos is a local variable always reset to 0, so stale
    bytes from the previous call cannot leak into the new output.
    """
    import oasis_writecsv as _mod  # type: ignore[import-not-found]

    # Prime with a large call to guarantee _tls.buf exists and is at least 256 KB.
    large_data = np.ones(5000, dtype=np.dtype([('X', np.float64)]))
    _write(large_data, ['X'], '%.6f', use_cython=True)
    buf_id = id(_mod._tls.buf)

    # Smaller call — buf_size < existing capacity, must reuse the same bytearray.
    small_data = np.ones(10, dtype=np.dtype([('X', np.float64)]))
    out = _write(small_data, ['X'], '%.6f', use_cython=True)

    assert id(_mod._tls.buf) == buf_id, "Buffer was reallocated when reuse was expected"
    assert out == '1.000000\n' * 10


# ---------------------------------------------------------------------------
# Section 8: Real-world dtype parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label,headers,dtype,fmt', [
    ('MELT', MELT_headers, MELT_dtype, MELT_fmt),
    ('SELT', SELT_headers, SELT_dtype, SELT_fmt),
    ('MPLT', MPLT_headers, MPLT_dtype, MPLT_fmt),
])
def test_realworld_dtype_parity(label, headers, dtype, fmt):
    """Production schemas (MELT, SELT, MPLT) produce identical Cython and Python output."""
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
    """Cython must be at least 4× faster than Python at 50k rows."""
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

    assert speedup >= 4.0, (
        f"{label}: expected >=4x speedup, got {speedup:.2f}x "
        f"(python={python_t * 1000:.1f}ms, cython={cython_t * 1000:.1f}ms)"
    )
