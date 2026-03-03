# cython: language_level=3, boundscheck=False, wraparound=False
"""
CSV row writing using custom C-level formatters, avoiding snprintf format-string
parsing on every call.

snprintf("%.6f", val) re-parses the format string on every invocation.
With N*C calls per chunk this is the dominant overhead (~6x vs Python's single
% call, which parses the combined format string once).

Instead:
  - %d / %i / %u  ->  write_int:   tmp-buffer LSB-fill + reverse copy, no format parsing
  - %.Xf          ->  write_float: val * 10^X as integer, tmp-buffer for fractional
                                   part, no format parsing
  - anything else ->  snprintf fallback (rare/unusual formats)

The hot double-loop contains only C array reads/writes, integer arithmetic,
and calls to the two cdef helpers — no Python objects, no format-string parsing.
"""

import numpy as np
cimport numpy as cnp
from libc.stdio cimport snprintf
from libc.math cimport rint as c_rint, isnan, isinf

cnp.import_array()

DEF MAX_COLS  = 64
DEF COL_INT   = 0   # %d / %i / %u
DEF COL_FIXED = 1   # %.Xf  — handled by write_float
DEF COL_OTHER = 2   # fallback to snprintf


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
    Write val as %.{prec}f to buf (prec <= 9), return chars written.

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

    cdef bint neg = val < 0.0
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
        int written
        int col_type[MAX_COLS]
        int col_prec[MAX_COLS]          # precision for COL_FIXED
        const char* fmt_ptrs[MAX_COLS]  # format ptr for COL_OTHER fallback

    if n_cols > MAX_COLS:
        raise ValueError(f"Too many columns: {n_cols} > {MAX_COLS}")

    cdef list col_fmts = row_fmt.split(',')

    # Classify each column and parse precision once, before the row loop.
    # fmts_b keeps bytes objects alive so fmt_ptrs remain valid.
    cdef list fmts_b = []
    cdef bytes _fb
    for j in range(n_cols):
        fmt = col_fmts[j]
        if fmt.endswith(('d', 'i', 'u')):
            col_type[j] = COL_INT
            col_prec[j] = 0
            fmts_b.append(b'')
        elif fmt.endswith('f') and '.' in fmt:
            dot = fmt.index('.')
            try:
                prec = int(fmt[dot + 1: dot + 4].rstrip('f'))
                col_type[j] = COL_FIXED
                col_prec[j] = prec
                fmts_b.append(b'')
            except (ValueError, IndexError):
                col_type[j] = COL_OTHER
                col_prec[j] = 0
                fmts_b.append(fmt.encode('ascii'))
        else:
            col_type[j] = COL_OTHER
            col_prec[j] = 0
            fmts_b.append(fmt.encode('ascii'))

    # Populate fmt_ptrs for fallback columns (fmts_b keeps them alive)
    for j in range(n_cols):
        if col_type[j] == COL_OTHER:
            _fb = fmts_b[j]
            fmt_ptrs[j] = _fb
        # else: fmt_ptrs[j] is never read

    # Copy columns to contiguous float64 (same as original)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] data_cpy = np.empty(
        (n_rows, n_cols), dtype=np.float64
    )
    for j in range(n_cols):
        data_cpy[:, j] = data[headers[j]]
    cdef double[:, :] vals = data_cpy

    # Pre-allocate output buffer (30 chars/value is generous)
    cdef Py_ssize_t buf_size = n_rows * (n_cols * 30 + n_cols + 1) + 64
    cdef bytearray out = bytearray(buf_size)
    cdef char[:] buf = out

    # -----------------------------------------------------------------------
    # Hot loop — pure C: no Python objects, no format-string parsing
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
                written = write_int(&buf[pos], <long long>vals[i, j])
            elif col_type[j] == COL_FIXED:
                written = write_float(&buf[pos], vals[i, j], col_prec[j])
            else:
                written = snprintf(
                    &buf[pos], buf_size - pos,
                    fmt_ptrs[j], vals[i, j],
                )
            pos += written

    buf[pos] = 10   # trailing '\n'
    pos += 1

    output_file.write(out[:pos].decode('ascii'))
