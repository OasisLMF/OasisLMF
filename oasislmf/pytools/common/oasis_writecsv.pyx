# cython: language_level=3, boundscheck=False, wraparound=False
"""
CSV row writing using custom C-level formatters for common format specifiers,
falling back to Python's % operator for full format compatibility.

Fast paths (no format-string parsing per row):
  - %d / %i / %u                    ->  write_int:   two-digit table, fill right-to-left + memcpy
  - %[+ ]d / %[+ ]i / %[+ ]u        ->  write_int:   with leading '+' or ' ' for non-negative
  - %.Xf (X=0..15), %f, %0.Xlf     ->  write_float: val * 10^X as integer, two-digit frac pairs
  - %[+ ].Xf / %[+ ]f              ->  write_float: with leading '+' or ' ' for non-negative

Column data is read directly from the structured numpy array via typed C pointers
and per-column strides, eliminating the intermediate float64 copy.  All column
metadata is stack-allocated (no malloc/free).

All other format specifiers (e.g. %e, %g, %10.2f, %x) fall back to Python's
own % operator, producing output identical to the non-Cython path in
write_ndarray_to_fmt_csv.
"""

import threading

import numpy as np
cimport numpy as cnp
from libc.math cimport rint as c_rint, isnan, isinf, copysign, rintl
from libc.string cimport memcpy
from cpython.unicode cimport PyUnicode_DecodeASCII

cnp.import_array()

# Thread-local output buffer — reused across calls (avoids malloc/zero) and
# safe for concurrent use: each thread writes to its own buffer.
_tls = threading.local()

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
    /* LLONG_MIN literal — negating LLONG_MIN is C signed-integer overflow (UB),
       so we memcpy this constant rather than computing digits at runtime. */
    static const char _LLONG_MIN_STR[] = "-9223372036854775808";
    """
    const char* _DIGITS2
    const char* _LLONG_MIN_STR

DEF COL_INT    = 0   # %d / %i / %u              — fast C path
DEF COL_FIXED  = 1   # %.Xf (X=0..15) / %f       — fast C path
DEF COL_PYTHON = 2   # everything else            — Python's % operator

DEF SIGN_PLUS  = 1   # col_flags bit: prepend '+' for non-negative values
DEF SIGN_SPACE = 2   # col_flags bit: prepend ' ' for non-negative values

# Maximum column count.  All per-column metadata is stack-allocated up to this
# limit, eliminating malloc/free overhead.  All production schemas are well below
# this (MELT=11, MPLT=17, ALCT=14).
DEF MAX_COLS = 64

# Native dtype codes used to dispatch the typed pointer read in the hot loop.
DEF NATIVE_INT32   = 0
DEF NATIVE_UINT32  = 1
DEF NATIVE_INT64   = 2
DEF NATIVE_FLOAT32 = 3
DEF NATIVE_FLOAT64 = 4


# ---------------------------------------------------------------------------
# Custom formatters  (cdef = C-level, not callable from Python)
# ---------------------------------------------------------------------------

cdef int write_int(char* buf, long long val) noexcept nogil:
    """Write decimal integer to buf, return chars written.

    Processes two digits per iteration using _DIGITS2, filling a local buffer
    right-to-left (MSB order) to avoid a reversal pass, then memcpy to buf.
    Halves the number of integer divisions vs the single-digit approach.
    """
    cdef char[22] tmp
    cdef int pos = 22, digit_count
    cdef bint neg = val < 0
    cdef long long v = -val if neg else val
    cdef long long q
    cdef int r

    if val == 0:
        buf[0] = 48  # '0'
        return 1

    # LLONG_MIN = -9223372036854775808: negating it is C signed-integer overflow (UB).
    # Two's-complement wraparound leaves v < 0, breaking the digit loop.
    # Detect it before the loop and emit the precomputed literal directly.
    if neg and v < 0:
        memcpy(buf, _LLONG_MIN_STR, 20)
        return 20

    # Short-circuit for 1- and 2-digit values: eliminates the loop, tmp stack
    # frame, and memcpy for the vast majority of production values.
    if v < 10:
        if neg:
            buf[0] = 45                      # '-'
            buf[1] = <char>(48 + <int>v)
            return 2
        buf[0] = <char>(48 + <int>v)
        return 1
    if v < 100:
        if neg:
            buf[0] = 45                      # '-'
            buf[1] = _DIGITS2[<int>v * 2]
            buf[2] = _DIGITS2[<int>v * 2 + 1]
            return 3
        buf[0] = _DIGITS2[<int>v * 2]
        buf[1] = _DIGITS2[<int>v * 2 + 1]
        return 2

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

    digit_count = 22 - pos
    if neg:
        buf[0] = 45  # '-'
        memcpy(buf + 1, tmp + pos, digit_count)
        return 1 + digit_count
    memcpy(buf, tmp + pos, digit_count)
    return digit_count


cdef int write_float(char* buf, double val, int prec, long long scale) noexcept nogil:
    """
    Write val as %.{prec}f to buf (prec 0..15), return chars written.

    scale must equal 10^prec and is precomputed once per column by the caller,
    eliminating the per-call multiply loop.  Fractional digits are extracted two
    at a time via _DIGITS2 (right-to-left fill into ftmp, then memcpy), halving
    integer divisions vs the per-digit approach.
    IEEE 754 round-half-to-even via c_rint matches C printf / Python %.
    prec=0 omits the decimal point entirely (e.g. '%.0f' % 3.7 -> '4').
    """
    cdef int n = 0, k, r
    cdef bint neg
    cdef long long ival, int_part, frac_part, fp, q2
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
    # For prec >= 9, val * scale can exceed double precision — use long double
    # to avoid a half-ULP error in the last digit.  For prec < 9 the 64-bit
    # multiply is exact enough, and c_rint (SSE2) is ~12% faster than rintl.
    if prec >= 9:
        ival = <long long>rintl(<long double>val * <long double>scale)
    else:
        ival  = <long long>c_rint(val * scale)
    int_part  = ival // scale
    frac_part = ival % scale

    if neg:
        buf[n] = 45  # '-'
        n += 1
    n += write_int(buf + n, int_part)

    if prec > 0:
        buf[n] = 46  # '.'
        n += 1
        fp = frac_part
        k  = prec
        # Two digits at a time: fills ftmp right-to-left (LSB pair first at
        # ftmp[k-2..k-1]), ending with the MSB pair at ftmp[0..1], so the buffer
        # is already in output order and can be memcpy'd directly.
        # For %.2f (the most common production format) one iteration covers
        # both digits with a single table lookup.
        while k >= 2:
            q2 = fp // 100
            r  = <int>(fp - q2 * 100)   # fp % 100, avoiding a second idiv
            fp = q2
            ftmp[k - 2] = _DIGITS2[r * 2]
            ftmp[k - 1] = _DIGITS2[r * 2 + 1]
            k -= 2
        if k == 1:   # odd prec: one digit remaining
            ftmp[0] = <char>(48 + <int>fp)
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
        # Stack-allocated column metadata — no malloc/free, always cache-warm.
        int[MAX_COLS]        col_type
        int[MAX_COLS]        col_prec
        double[MAX_COLS]     col_overflow
        char[MAX_COLS]       col_flags
        long long[MAX_COLS]  col_scale
        # Per-column typed C pointers directly into the structured array.
        # col_ptr[j]    = pointer to element 0 of column j in data's memory.
        # col_stride[j] = byte distance between consecutive rows (= data.itemsize).
        # Eliminates the O(n_rows * n_cols) float64 intermediate copy.
        Py_ssize_t[MAX_COLS] col_ptr
        Py_ssize_t[MAX_COLS] col_stride
        int[MAX_COLS]        col_native
        # Hot-loop working variables
        char*     raw_ptr
        long long ll_val
        double    dbl_val
        cnp.ndarray col_arr_nd
        bytearray out
        char[::1] buf   # C-contiguous: required for nogil memoryview access
        bytes py_str
        list col_fmts
        list coerced_cols

    if n_cols > MAX_COLS:
        raise ValueError(
            f"write_rows: {n_cols} columns exceeds MAX_COLS ({MAX_COLS}). "
            "Increase MAX_COLS and recompile oasis_writecsv."
        )

    col_fmts     = row_fmt.split(',')
    coerced_cols = []   # keeps float64 copies alive for unusual (non-standard) dtypes

    # -----------------------------------------------------------------------
    # Column classification: format specifier → COL_INT / COL_FIXED / COL_PYTHON
    # Column access setup:   dtype → col_native, col_ptr, col_stride
    #
    # Fast paths (COL_INT / COL_FIXED) are taken for:
    #   COL_INT:   %[+ ]?d|i|u           (optional sign flag, no width)
    #   COL_FIXED: %[+ ]?[0]?[.X]?[l|h|ll|L]?f
    #              where X is 0..15; bare %f treated as %.6f;
    #              length modifiers (l/h/ll/L) are ignored (no-op in Python too);
    #              leading '0' flag without a width field is also a no-op.
    #
    # Everything else → COL_PYTHON (Python's % operator, identical output).
    # -----------------------------------------------------------------------
    for j in range(n_cols):
        raw_fmt = col_fmts[j]
        col_flags[j] = 0

        if not raw_fmt.startswith('%'):
            col_type[j] = COL_PYTHON; col_prec[j] = 0; col_overflow[j] = 0.0; col_scale[j] = 0  # prec/overflow/scale unused for COL_PYTHON
        else:
            s = raw_fmt[1:]  # strip leading '%'

            # Detect and strip sign flag (+ or space)
            sign_flag = 0
            if s.startswith('+'):
                sign_flag = SIGN_PLUS; s = s[1:]
            elif s.startswith(' '):
                sign_flag = SIGN_SPACE; s = s[1:]

            if s in ('d', 'i', 'u'):
                col_type[j] = COL_INT
                col_prec[j] = 0        # unused for COL_INT
                col_scale[j] = 0       # unused for COL_INT
                # overflow guard only applies to float-typed columns (%d on a float field)
                col_overflow[j] = 9.2e18
                col_flags[j] = sign_flag
            else:
                t = s

                # Strip leading '0' flag that immediately precedes '.'
                # (zero-pad without a width field is a no-op: '%0.9f' == '%.9f')
                if len(t) > 1 and t[0] == '0' and t[1] == '.':
                    t = t[1:]

                # Strip C length modifiers before the 'f' type char
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
                    prec = 16  # sentinel: out of [0..15], forces COL_PYTHON below

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
                    col_type[j] = COL_PYTHON; col_prec[j] = 0; col_overflow[j] = 0.0; col_scale[j] = 0  # prec/overflow/scale unused for COL_PYTHON

        # Set up typed C pointer and row stride for this column.
        # np.asarray on a structured array field returns a strided view with no
        # copy; its memory is owned by `data` (a live function argument).
        col_arr_nd = np.asarray(data[headers[j]])
        dt = col_arr_nd.dtype
        if dt.kind == 'i' and dt.itemsize == 4:
            col_native[j] = NATIVE_INT32
        elif dt.kind == 'u' and dt.itemsize == 4:
            col_native[j] = NATIVE_UINT32
        elif dt.kind == 'i' and dt.itemsize == 8:
            col_native[j] = NATIVE_INT64
        elif dt.kind == 'f' and dt.itemsize == 4:
            col_native[j] = NATIVE_FLOAT32
        elif dt.kind == 'f' and dt.itemsize == 8:
            col_native[j] = NATIVE_FLOAT64
        else:
            # Unusual dtype: coerce to float64 and keep the copy alive in
            # coerced_cols so its pointer remains valid through the hot loop.
            col_native[j] = NATIVE_FLOAT64
            col_arr_nd = col_arr_nd.astype(np.float64)
            coerced_cols.append(col_arr_nd)

        col_ptr[j]    = <Py_ssize_t>col_arr_nd.data
        col_stride[j] = col_arr_nd.strides[0]

    # Pre-allocate output buffer with per-column estimates:
    #   COL_INT:    22  (sign + up to 20 decimal digits)
    #   COL_FIXED:  32  (sign + digits + dot + up to 15 frac digits)
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

    # Reuse the thread-local buffer — avoids malloc/zero on every call.
    if not hasattr(_tls, 'buf') or len(_tls.buf) < buf_size:
        _tls.buf = bytearray(max(buf_size, 262144))  # 256 KB minimum
    out = <bytearray>_tls.buf
    buf = out

    # -----------------------------------------------------------------------
    # Hot loop — GIL released for the full duration.
    #   COL_INT / COL_FIXED: pure C, no Python objects touched.
    #   COL_PYTHON + COL_FIXED overflow: GIL reacquired via 'with gil:' for
    #     the Python % call only; in production those paths are never taken.
    #   write_float handles inf/nan at C level, matching Python's output.
    #   '\n' appended after each row; PyUnicode_DecodeASCII builds the result
    #   string directly from the buffer without an intermediate copy.
    # -----------------------------------------------------------------------
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                if j > 0:
                    buf[pos] = 44  # ','
                    pos += 1

                raw_ptr = <char*>col_ptr[j] + i * col_stride[j]

                if col_type[j] == COL_INT:
                    if col_native[j] == NATIVE_INT32:
                        # i4: always fits in long long, no guard needed.
                        ll_val = <long long>((<int*>raw_ptr)[0])
                        if col_flags[j] and ll_val >= 0:
                            buf[pos] = 43 if (col_flags[j] & SIGN_PLUS) else 32
                            written = 1 + write_int(&buf[pos + 1], ll_val)
                        else:
                            written = write_int(&buf[pos], ll_val)
                    elif col_native[j] == NATIVE_UINT32:
                        # u4: always fits in long long; cast to long long is always >= 0.
                        ll_val = <long long>((<unsigned int*>raw_ptr)[0])
                        if col_flags[j]:
                            buf[pos] = 43 if (col_flags[j] & SIGN_PLUS) else 32
                            written = 1 + write_int(&buf[pos + 1], ll_val)
                        else:
                            written = write_int(&buf[pos], ll_val)
                    elif col_native[j] == NATIVE_INT64:
                        # i8: exact long long, no guard needed.
                        ll_val = (<long long*>raw_ptr)[0]
                        if col_flags[j] and ll_val >= 0:
                            buf[pos] = 43 if (col_flags[j] & SIGN_PLUS) else 32
                            written = 1 + write_int(&buf[pos + 1], ll_val)
                        else:
                            written = write_int(&buf[pos], ll_val)
                    else:
                        # Float-typed column with an integer format specifier.
                        # Read as double and apply the NaN/Inf/overflow guard,
                        # matching Python's truncation-toward-zero behaviour.
                        # dbl_val >= 0.0 treats -0.0 as non-negative, matching
                        # Python: '%+d' % -0.0 == '+0'.
                        if col_native[j] == NATIVE_FLOAT32:
                            dbl_val = <double>((<float*>raw_ptr)[0])
                        else:  # NATIVE_FLOAT64
                            dbl_val = (<double*>raw_ptr)[0]
                        if isnan(dbl_val) or isinf(dbl_val) or dbl_val >= col_overflow[j] or dbl_val <= -col_overflow[j]:
                            with gil:
                                py_str = (col_fmts[j] % dbl_val).encode('ascii')
                                written = len(py_str)
                                out[pos:pos + written] = py_str
                        elif col_flags[j] and dbl_val >= 0.0:
                            buf[pos] = 43 if (col_flags[j] & SIGN_PLUS) else 32
                            written = 1 + write_int(&buf[pos + 1], <long long>dbl_val)
                        else:
                            written = write_int(&buf[pos], <long long>dbl_val)

                elif col_type[j] == COL_FIXED:
                    if col_native[j] == NATIVE_FLOAT32:
                        dbl_val = <double>((<float*>raw_ptr)[0])
                    elif col_native[j] == NATIVE_INT32:
                        dbl_val = <double>((<int*>raw_ptr)[0])
                    elif col_native[j] == NATIVE_UINT32:
                        dbl_val = <double>((<unsigned int*>raw_ptr)[0])
                    elif col_native[j] == NATIVE_INT64:
                        dbl_val = <double>((<long long*>raw_ptr)[0])
                    else:  # NATIVE_FLOAT64
                        dbl_val = (<double*>raw_ptr)[0]

                    # Only finite values that overflow long long (val*scale > 9.2e18)
                    # need the Python % fallback; inf/nan go directly to write_float.
                    if not isinf(dbl_val) and (dbl_val >= col_overflow[j] or dbl_val <= -col_overflow[j]):
                        with gil:
                            py_str = (col_fmts[j] % dbl_val).encode('ascii')
                            written = len(py_str)
                            if pos + written > buf_size - 2:
                                raise RuntimeError(
                                    f"output buffer overflow at row {i}, col {j}: "
                                    f"format '{col_fmts[j]}' produced {written} chars "
                                    f"but only {buf_size - pos - 2} remain"
                                )
                            out[pos:pos + written] = py_str
                    elif col_flags[j] and (isnan(dbl_val) or copysign(1.0, dbl_val) > 0.0):
                        # Prepend '+' or ' ' for nan and positive values.
                        # copysign excludes -0.0 so write_float renders it as '-0.XX',
                        # matching Python ('%+.2f' % -0.0 == '-0.00').
                        buf[pos] = 43 if (col_flags[j] & SIGN_PLUS) else 32  # '+' or ' '
                        written = 1 + write_float(&buf[pos + 1], dbl_val, col_prec[j], col_scale[j])
                    else:
                        written = write_float(&buf[pos], dbl_val, col_prec[j], col_scale[j])

                else:  # COL_PYTHON
                    # Read value at C level (nogil-safe), then format with Python's %
                    # inside a brief GIL reacquisition.
                    if col_native[j] == NATIVE_INT32:
                        dbl_val = <double>((<int*>raw_ptr)[0])
                    elif col_native[j] == NATIVE_UINT32:
                        dbl_val = <double>((<unsigned int*>raw_ptr)[0])
                    elif col_native[j] == NATIVE_INT64:
                        dbl_val = <double>((<long long*>raw_ptr)[0])
                    elif col_native[j] == NATIVE_FLOAT32:
                        dbl_val = <double>((<float*>raw_ptr)[0])
                    else:  # NATIVE_FLOAT64
                        dbl_val = (<double*>raw_ptr)[0]
                    with gil:
                        py_str = (col_fmts[j] % dbl_val).encode('ascii')
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
