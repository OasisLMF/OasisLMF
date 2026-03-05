# cython: language_level=3, boundscheck=False, wraparound=False
"""
CSV row writing using custom C-level formatters for common format specifiers,
falling back to Python's % operator for full format compatibility.

Fast paths (no format-string parsing per row):
  - %d / %i / %u  ->  write_int:   tmp-buffer LSB-fill + reverse copy
  - %.Xf (X=1..9) ->  write_float: val * 10^X as integer, tmp-buffer for fractional part

All other format specifiers (e.g. %f, %e, %g, %10.2f, %0.9lf, %x) fall back
to Python's own % operator, producing output identical to the non-Cython path
in write_ndarray_to_fmt_csv.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport rint as c_rint, isnan, isinf, copysign
from libc.stdlib cimport malloc, free

cnp.import_array()

DEF COL_INT    = 0   # %d / %i / %u  — fast C path
DEF COL_FIXED  = 1   # %.Xf (X=1..9) — fast C path
DEF COL_PYTHON = 2   # everything else — Python's % operator


# ---------------------------------------------------------------------------
# Custom formatters  (cdef = C-level, not callable from Python)
#
# write_int:   fills a tmp[22] buffer LSB-first (% 10), then copies in reverse
#              using a while loop — avoids range(..., -1, -1) which triggers
#              a false-positive wraparound=False warning.
# write_float: same idea for the fractional part via ftmp[10].
# ---------------------------------------------------------------------------

cdef int write_int(char* buf, long long val) noexcept nogil:
    """Write decimal integer to buf, return chars written."""
    cdef char tmp[22]
    cdef int i = 0, n = 0
    if val == 0:
        buf[0] = 48  # '0'
        return 1
    cdef bint neg = val < 0
    cdef long long v = -val if neg else val
    # Extract digits LSB-first into tmp
    while v > 0:
        tmp[i] = <char>(48 + v % 10)
        i += 1
        v //= 10
    if neg:
        buf[n] = 45  # '-'
        n += 1
    # Copy in reverse via while loop (range(i-1,-1,-1) triggers wraparound warning)
    cdef int k = i - 1
    while k >= 0:
        buf[n] = tmp[k]
        k -= 1
        n += 1
    return n


cdef int write_float(char* buf, double val, int prec) noexcept nogil:
    """
    Write val as %.{prec}f to buf (prec 1..9), return chars written.

    Uses integer arithmetic: multiply by 10^prec, split into integer and
    fractional parts, write both via write_int and a tmp fractional buffer.
    IEEE 754 round-half-to-even via c_rint matches C printf / Python %.
    """
    # Match Python's '%.Xf' % nan/inf behaviour
    if isnan(val):
        buf[0] = 110; buf[1] = 97; buf[2] = 110  # 'nan'
        return 3
    if isinf(val):
        if val > 0.0:
            buf[0] = 105; buf[1] = 110; buf[2] = 102  # 'inf'
            return 3
        else:
            buf[0] = 45; buf[1] = 105; buf[2] = 110; buf[3] = 102  # '-inf'
            return 4

    cdef long long scale = 1
    cdef int p
    for p in range(prec):
        scale *= 10

    cdef bint neg = copysign(1.0, val) < 0.0
    if neg:
        val = -val

    # Round using IEEE 754 round-half-to-even (same mode as C printf / Python %)
    cdef long long ival      = <long long>c_rint(val * scale)
    cdef long long int_part  = ival // scale
    cdef long long frac_part = ival % scale

    cdef int n = 0
    if neg:
        buf[n] = 45  # '-'
        n += 1

    n += write_int(buf + n, int_part)

    buf[n] = 46  # '.'
    n += 1

    # Fill fractional digits right-to-left into ftmp, then copy forward.
    # while loop avoids range(prec-1, -1, -1) wraparound=False warning.
    cdef char ftmp[10]
    cdef long long fp = frac_part
    cdef int k = prec - 1
    while k >= 0:
        ftmp[k] = <char>(48 + fp % 10)
        fp //= 10
        k -= 1
    for p in range(prec):
        buf[n] = ftmp[p]
        n += 1

    return n


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_rows(output_file, object data, list headers, str row_fmt):
    cdef:
        Py_ssize_t n_rows = data.shape[0]
        Py_ssize_t n_cols = len(headers)
        Py_ssize_t i, j, pos = 0
        Py_ssize_t chars_per_row, buf_size
        int written
        int*    col_type     = <int*>    malloc(n_cols * sizeof(int))
        int*    col_prec     = <int*>    malloc(n_cols * sizeof(int))
        double* col_overflow = <double*> malloc(n_cols * sizeof(double))
        cnp.ndarray[cnp.float64_t, ndim=2] data_cpy
        double[:, :] vals
        bytearray out
        char[:] buf
        bytes py_str             # reused for COL_PYTHON fallback

    if col_type is NULL or col_prec is NULL or col_overflow is NULL:
        free(col_type); free(col_prec); free(col_overflow)
        raise MemoryError("failed to allocate column metadata arrays")

    cdef list col_fmts = row_fmt.split(',')

    try:
        # Classify each column once before the row loop.
        #
        # Fast paths are only taken for the exact canonical forms:
        #   COL_INT:   %d  %i  %u      (no flags, no width)
        #   COL_FIXED: %.Xf  where X is 1..9  (no flags, no width, no length modifier)
        #
        # Everything else goes to COL_PYTHON, which delegates to Python's own %
        # operator — identical output to the non-Cython path for any format string.
        for j in range(n_cols):
            fmt = col_fmts[j]
            if fmt in ('%d', '%i', '%u'):
                col_type[j] = COL_INT
                col_prec[j] = 0
                col_overflow[j] = 9.2e18   # safe upper bound for <long long> cast
            elif fmt.startswith('%.') and fmt.endswith('f') and fmt[2:-1].isdigit():
                prec = int(fmt[2:-1])
                if 1 <= prec <= 9:
                    col_type[j] = COL_FIXED
                    col_prec[j] = prec
                    scale = 1
                    for _ in range(prec):
                        scale *= 10
                    col_overflow[j] = 9.2e18 / scale  # c_rint(val*scale) overflows above this
                else:
                    col_type[j] = COL_PYTHON
                    col_prec[j] = 0
                    col_overflow[j] = 0.0
            else:
                col_type[j] = COL_PYTHON
                col_prec[j] = 0
                col_overflow[j] = 0.0

        # Copy columns to contiguous float64 (same as the Python fallback path)
        data_cpy = np.empty((n_rows, n_cols), dtype=np.float64)
        for j in range(n_cols):
            data_cpy[:, j] = data[headers[j]]
        vals = data_cpy

        # Pre-allocate output buffer with per-column estimates:
        #   COL_INT:    22  (sign + up to 20 decimal digits)
        #   COL_FIXED:  32  (sign + digits + dot + up to 9 frac digits)
        #   COL_PYTHON: 128 (generous for any format specifier, e.g. %.15f on large values)
        # +n_cols for commas, +1 per row for newline.
        chars_per_row = n_cols  # commas + newline
        for j in range(n_cols):
            if col_type[j] == COL_INT:
                chars_per_row += 22
            elif col_type[j] == COL_FIXED:
                chars_per_row += 32
            else:
                chars_per_row += 128
        buf_size = n_rows * chars_per_row + 64
        out = bytearray(buf_size)
        buf = out

        # -----------------------------------------------------------------------
        # Hot loop
        #   COL_INT / COL_FIXED: pure C, no Python objects
        #   COL_PYTHON:          Python's % — correct for any format specifier
        # -----------------------------------------------------------------------
        for i in range(n_rows):
            if i > 0:
                buf[pos] = 10   # '\n'
                pos += 1
            for j in range(n_cols):
                if j > 0:
                    buf[pos] = 44  # ','
                    pos += 1
                if col_type[j] == COL_INT:
                    # Guard: nan/inf → Python's % raises ValueError/OverflowError to match
                    # the pure-Python path. Large finite values → Python's arbitrary-precision
                    # int avoids <long long> C UB above LLONG_MAX (~9.22e18).
                    if isnan(vals[i, j]) or isinf(vals[i, j]) or vals[i, j] >= col_overflow[j] or vals[i, j] <= -col_overflow[j]:
                        py_str = (col_fmts[j] % vals[i, j]).encode('ascii')
                        written = len(py_str)
                        out[pos:pos + written] = py_str
                    else:
                        written = write_int(&buf[pos], <long long>vals[i, j])
                elif col_type[j] == COL_FIXED:
                    # Guard: finite values where val*scale would overflow long long in c_rint.
                    # nan comparisons return False, so nan still reaches write_float (handles it).
                    # inf comparisons return True, so inf/−inf fall to Python % (same output).
                    if vals[i, j] >= col_overflow[j] or vals[i, j] <= -col_overflow[j]:
                        py_str = (col_fmts[j] % vals[i, j]).encode('ascii')
                        written = len(py_str)
                        out[pos:pos + written] = py_str
                    else:
                        written = write_float(&buf[pos], vals[i, j], col_prec[j])
                else:  # COL_PYTHON
                    # vals[i, j] is a C double; Cython boxes it to Python float,
                    # matching exactly what the Python fallback passes to %.
                    py_str = (col_fmts[j] % vals[i, j]).encode('ascii')
                    written = len(py_str)
                    if pos + written > buf_size - 2:
                        raise RuntimeError(
                            f"output buffer overflow at row {i}, col {j}: "
                            f"format '{col_fmts[j]}' produced {written} chars "
                            f"but only {buf_size - pos - 2} remain"
                        )
                    out[pos:pos + written] = py_str
                pos += written

        buf[pos] = 10   # trailing '\n'
        pos += 1

        output_file.write(out[:pos].decode('ascii'))

    finally:
        free(col_type)
        free(col_prec)
        free(col_overflow)
