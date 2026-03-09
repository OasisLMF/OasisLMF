# cython: language_level=3, boundscheck=False, wraparound=False
"""
CSV row writing using custom C-level formatters for common format specifiers,
falling back to Python's % operator for full format compatibility.

Fast paths (no format-string parsing per row):
  - %d / %i / %u                    ->  write_int:   tmp-buffer LSB-fill + reverse copy
  - %[+| ]d / %[+| ]i / %[+| ]u    ->  write_int:   with leading '+' or ' ' for non-negative
  - %.Xf (X=0..15), %f, %0.Xlf     ->  write_float: val * 10^X as integer, tmp-buffer for frac
  - %[+| ].Xf / %[+| ]f            ->  write_float: with leading '+' or ' ' for non-negative

All other format specifiers (e.g. %e, %g, %10.2f, %x) fall back to Python's
own % operator, producing output identical to the non-Cython path in
write_ndarray_to_fmt_csv.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport rint as c_rint, isnan, isinf, copysign
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cpython.unicode cimport PyUnicode_DecodeASCII

cnp.import_array()

# Two-digit ASCII lookup table: _DIGITS2[n*2] and _DIGITS2[n*2+1] give the
# two-character decimal representation of n (00..99).  Declared as a C static
# so it lives in read-only data and is shared across calls with zero overhead.
cdef extern from *:
    """
    static const char _DIGITS2[200] =
        "00010203040506070809"
        "10111213141516171819"
        "20212223242526272829"
        "30313233343536373839"
        "40414243444546474849"
        "50515253545556575859"
        "60616263646566676869"
        "70717273747576777879"
        "80818283848586878889"
        "90919293949596979899";
    """
    const char* _DIGITS2

DEF COL_INT    = 0   # %d / %i / %u              — fast C path
DEF COL_FIXED  = 1   # %.Xf (X=0..15) / %f       — fast C path
DEF COL_PYTHON = 2   # everything else            — Python's % operator

DEF SIGN_PLUS  = 1   # col_flags bit: prepend '+' for non-negative values
DEF SIGN_SPACE = 2   # col_flags bit: prepend ' ' for non-negative values


# ---------------------------------------------------------------------------
# Custom formatters  (cdef = C-level, not callable from Python)
#
# write_int:   fills a tmp[22] buffer LSB-first (% 10), then copies in reverse
#              using a while loop — avoids range(..., -1, -1) which triggers
#              a false-positive wraparound=False warning.
# write_float: same idea for the fractional part via ftmp[16].
# ---------------------------------------------------------------------------

cdef int write_int(char* buf, long long val) noexcept nogil:
    """Write decimal integer to buf, return chars written.

    Processes two digits per iteration using _DIGITS2, filling a local buffer
    right-to-left (MSB order) to avoid a reversal pass, then memcpy to buf.
    Halves the number of integer divisions vs the single-digit approach.
    """
    cdef char[22] tmp
    cdef int pos = 22, n = 0, digit_count
    cdef bint neg = val < 0
    cdef long long v = -val if neg else val
    cdef long long q
    cdef int r

    if val == 0:
        buf[0] = 48  # '0'
        return 1

    # Two digits at a time, filling tmp from the right so digits end up in
    # MSB-first order — no reversal needed.
    while v >= 100:
        q = v // 100
        r = <int>(v - q * 100)   # v % 100, avoiding a second idiv
        pos -= 2
        tmp[pos]     = _DIGITS2[r * 2]
        tmp[pos + 1] = _DIGITS2[r * 2 + 1]
        v = q

    if v >= 10:
        pos -= 2
        tmp[pos]     = _DIGITS2[<int>v * 2]
        tmp[pos + 1] = _DIGITS2[<int>v * 2 + 1]
    else:
        pos -= 1
        tmp[pos] = <char>(48 + <int>v)

    if neg:
        buf[n] = 45  # '-'
        n += 1

    digit_count = 22 - pos
    memcpy(buf + n, tmp + pos, digit_count)
    return n + digit_count


cdef int write_float(char* buf, double val, int prec, long long scale) noexcept nogil:
    """
    Write val as %.{prec}f to buf (prec 0..15), return chars written.

    scale must equal 10^prec and is precomputed once per column by the caller,
    eliminating the per-call multiply loop.  The fractional digits are filled
    right-to-left into ftmp then copied forward with memcpy rather than a loop.
    IEEE 754 round-half-to-even via c_rint matches C printf / Python %.
    prec=0 omits the decimal point entirely (e.g. '%.0f' % 3.7 -> '4').
    """
    cdef int n = 0, k
    cdef bint neg
    cdef long long ival, int_part, frac_part, fp
    cdef char[16] ftmp  # 16 covers prec up to 15

    # Match Python's nan/inf behaviour
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

    # Use copysign to detect negative zero (avoids -0.0 printing as 0.00)
    neg = copysign(1.0, val) < 0.0
    if neg:
        val = -val

    # Round using IEEE 754 round-half-to-even (same mode as C printf / Python %)
    ival      = <long long>c_rint(val * scale)
    int_part  = ival // scale
    frac_part = ival % scale

    if neg:
        buf[n] = 45  # '-'
        n += 1
    n += write_int(buf + n, int_part)

    if prec > 0:
        buf[n] = 46  # '.'
        n += 1
        # Fill fractional digits right-to-left into ftmp, then copy forward
        # with memcpy (replaces the per-byte loop).
        # while loop avoids range(prec-1, -1, -1) wraparound=False warning.
        fp = frac_part
        k = prec - 1
        while k >= 0:
            ftmp[k] = <char>(48 + fp % 10)
            fp //= 10
            k -= 1
        memcpy(buf + n, ftmp, prec)
        n += prec

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
        char*   col_flags    = <char*>   malloc(n_cols * sizeof(char))
        long long* col_scale = <long long*> malloc(n_cols * sizeof(long long))
        cnp.ndarray[cnp.float64_t, ndim=2] data_cpy
        double[:, :] vals
        bytearray out
        char[:] buf
        bytes py_str             # reused for COL_PYTHON fallback

    if col_type is NULL or col_prec is NULL or col_overflow is NULL or col_flags is NULL or col_scale is NULL:
        free(col_type); free(col_prec); free(col_overflow); free(col_flags); free(col_scale)
        raise MemoryError("failed to allocate column metadata arrays")

    cdef list col_fmts = row_fmt.split(',')

    try:
        # Classify each column once before the row loop.
        #
        # Fast paths (COL_INT / COL_FIXED) are taken for:
        #   COL_INT:   %[+| ]?d|i|u          (optional sign flag, no width)
        #   COL_FIXED: %[+| ]?[0]?[.X]?[l|h|ll|L]?f
        #              where X is 0..15; bare %f treated as %.6f;
        #              length modifiers (l/h/ll/L) are ignored (no-op in Python too);
        #              leading '0' flag without a width field is also a no-op.
        #
        # Everything else → COL_PYTHON (Python's % operator, identical to non-Cython path).
        for j in range(n_cols):
            raw_fmt = col_fmts[j]
            col_flags[j] = 0

            if not raw_fmt.startswith('%'):
                col_type[j] = COL_PYTHON; col_prec[j] = 0; col_overflow[j] = 0.0
                continue

            s = raw_fmt[1:]  # strip leading '%'

            # Detect and strip sign flag (+ or space)
            sign_flag = 0
            if s.startswith('+'):
                sign_flag = SIGN_PLUS; s = s[1:]
            elif s.startswith(' '):
                sign_flag = SIGN_SPACE; s = s[1:]

            # --- Integer types ---
            if s in ('d', 'i', 'u'):
                col_type[j] = COL_INT
                col_prec[j] = 0
                col_overflow[j] = 9.2e18   # safe upper bound for <long long> cast
                col_flags[j] = sign_flag
                continue

            # --- Float type: strip no-op flags/modifiers, then parse ---
            t = s

            # Strip leading '0' flag that immediately precedes '.' (zero-pad without
            # a width field is a no-op: '%0.9f' == '%.9f' in Python)
            if len(t) > 1 and t[0] == '0' and t[1] == '.':
                t = t[1:]

            # Strip C length modifiers before the 'f' type char (Python ignores them)
            if t.endswith('llf'):
                t = t[:-3] + 'f'
            elif t.endswith('lf') or t.endswith('hf') or t.endswith('Lf'):
                t = t[:-2] + 'f'

            # Resolve precision
            if t == 'f':
                prec = 6  # bare %f — Python default is 6 decimal places
            elif t.startswith('.') and t.endswith('f') and t[1:-1].isdigit():
                prec = int(t[1:-1])
            else:
                col_type[j] = COL_PYTHON; col_prec[j] = 0; col_overflow[j] = 0.0
                continue

            if 0 <= prec <= 15:
                col_type[j] = COL_FIXED
                col_prec[j] = prec
                col_flags[j] = sign_flag
                scale = 1
                for _ in range(prec):
                    scale *= 10
                col_overflow[j] = 9.2e18 / scale if prec > 0 else 9.2e18
                col_scale[j]    = <long long>scale
            else:
                col_type[j] = COL_PYTHON; col_prec[j] = 0; col_overflow[j] = 0.0

        # Copy columns to contiguous float64 (same as the Python fallback path)
        data_cpy = np.empty((n_rows, n_cols), dtype=np.float64)
        for j in range(n_cols):
            data_cpy[:, j] = data[headers[j]]
        vals = data_cpy

        # Pre-allocate output buffer with per-column estimates:
        #   COL_INT:    22  (sign_flag + sign + up to 20 decimal digits)
        #   COL_FIXED:  32  (sign_flag + sign + digits + dot + up to 15 frac digits)
        #   COL_PYTHON: 128 (generous for any format specifier)
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
        #
        # '\n' is written after each row (not before rows 1+) to remove the
        # per-row branch.  PyUnicode_DecodeASCII builds the output str directly
        # from the buffer, eliminating the intermediate bytearray slice.
        # -----------------------------------------------------------------------
        for i in range(n_rows):
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
                    elif col_flags[j] and vals[i, j] >= 0.0:
                        # Sign flag: '+' or ' ' for non-negative values.
                        # Use >= 0.0 (not copysign) so that -0.0 compares equal to 0.0
                        # and gets '+0' / ' 0' to match Python's '%+d' % -0.0 = '+0'.
                        buf[pos] = 43 if (col_flags[j] & SIGN_PLUS) else 32  # '+' or ' '
                        written = 1 + write_int(&buf[pos + 1], <long long>vals[i, j])
                    else:
                        written = write_int(&buf[pos], <long long>vals[i, j])
                elif col_type[j] == COL_FIXED:
                    # Guard: finite values where val*scale would overflow long long in c_rint.
                    # nan comparisons return False, so nan still reaches write_float (handles it).
                    # inf comparisons return True, so inf/−inf fall to Python % (same output).
                    if vals[i, j] >= col_overflow[j] or vals[i, j] <= -col_overflow[j]:
                        py_str = (col_fmts[j] % vals[i, j]).encode('ascii')
                        written = len(py_str)
                        if pos + written > buf_size - 2:
                            raise RuntimeError(
                                f"output buffer overflow at row {i}, col {j}: "
                                f"format '{col_fmts[j]}' produced {written} chars "
                                f"but only {buf_size - pos - 2} remain"
                            )
                        out[pos:pos + written] = py_str
                    elif col_flags[j] and (isnan(vals[i, j]) or copysign(1.0, vals[i, j]) > 0.0):
                        # Sign flag: prepend '+' or ' ' for nan and positive values.
                        # Python applies sign flags to nan: '%+.2f' % nan == '+nan'.
                        # copysign correctly excludes -0.0 (sign bit negative → no prefix),
                        # so write_float renders -0.0 as '-0.XX' matching Python.
                        buf[pos] = 43 if (col_flags[j] & SIGN_PLUS) else 32  # '+' or ' '
                        written = 1 + write_float(&buf[pos + 1], vals[i, j], col_prec[j], col_scale[j])
                    else:
                        written = write_float(&buf[pos], vals[i, j], col_prec[j], col_scale[j])
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
            buf[pos] = 10   # '\n' — written after each row, no per-row branch needed
            pos += 1

        if n_rows == 0:
            buf[pos] = 10   # '\n' — preserve empty-input behaviour (trailing newline)
            pos += 1

        output_file.write(PyUnicode_DecodeASCII(&buf[0], pos, NULL))

    finally:
        free(col_type)
        free(col_prec)
        free(col_overflow)
        free(col_flags)
        free(col_scale)
