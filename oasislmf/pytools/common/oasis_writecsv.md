# oasis_writecsv — How it works

Cython implementation of a fast CSV row writer used by `write_ndarray_to_fmt_csv` in
[data.py](data.py).  Entry point: `write_rows(output_file, data, headers, row_fmt)`.

The key idea is to classify each column's format specifier once at call time (not per
row), then dispatch to a pure-C formatter in a GIL-free hot loop, building one large
ASCII buffer that is written to the output file in a single call.

---

## High-level call flow

```
write_rows(output_file, data, headers, row_fmt)
    │
    ├─ 1. Classify columns & set up C pointers   (Python, GIL held, one pass)
    │      parse row_fmt → COL_INT / COL_FIXED / COL_PYTHON per column
    │      resolve dtype  → NATIVE_INT32 / UINT32 / INT64 / FLOAT32 / FLOAT64
    │      store metadata in stack arrays (col_type, col_prec, col_scale, …)
    │      col_ptr[j]    = pointer to element 0 of column j in data's memory
    │      col_stride[j] = col_arr_nd.strides[0]  (= parent struct's itemsize)
    │
    ├─ 3. Size & acquire buffer      (Python, GIL held)
    │      estimate chars_per_row; reuse / grow thread-local bytearray
    │
    └─ 4. Hot loop (nogil)           ← most CPU time spent here
           for each row i:
               for each col j:
                   raw_ptr = col_ptr[j] + i * col_stride[j]
                   dispatch → write_int / write_float / (with gil: Python %)
               buf[pos++] = '\n'
           output_file.write(PyUnicode_DecodeASCII(buf, pos))
```

---

## Column classification

`row_fmt` is a comma-separated format string, e.g. `"%d,%.2f,%e"`.  It is split once
and each specifier is classified into one of three types:

| Type | Constant | Matched specifiers | Formatter |
|------|----------|--------------------|-----------|
| Integer | `COL_INT` | `%d`, `%i`, `%u`, `%+d`, `% d` | `write_int` |
| Fixed-point | `COL_FIXED` | `%.Xf` (X=0..15), `%f`, `%0.Xf`, `%lf`, `%Lf`, `%llf` | `write_float` |
| Python fallback | `COL_PYTHON` | everything else (`%e`, `%g`, `%10.2f`, `%x`, …) | Python `%` |

Sign flags `+` and ` ` (space) are recognised and stored in `col_flags[j]`; they are
prepended by the hot loop before calling the formatter.

Length modifiers (`l`, `ll`, `h`, `L`) and a leading `0` flag without a width field
are stripped as no-ops — Python ignores them too.

```
raw_fmt = "%+.2f"
           │└──┘└─ prec = 2
           └─ sign flag '+' → SIGN_PLUS stored in col_flags[j]
              → col_type = COL_FIXED, col_prec = 2, col_scale = 100
```

---

## Direct pointer access (no float64 copy)

A structured NumPy array holds all columns interleaved in memory.  Rather than
extracting a column into a separate float64 array (the naive approach), the code
stores one raw pointer and one stride per column:

```
data memory layout (structured array, e.g. dtype=[('a', i4), ('b', f4)]):

  row 0: [ a0 (4 bytes) | b0 (4 bytes) ]
  row 1: [ a1 (4 bytes) | b1 (4 bytes) ]
  row 2: [ a2 (4 bytes) | b2 (4 bytes) ]
          ^                ^
          col_ptr[0]       col_ptr[1]
          stride = 8       stride = 8

  element (i, j) = *(col_ptr[j] + i * col_stride[j])
```

The dtype is mapped to one of five native codes so the hot loop knows the cast:

| NumPy kind | itemsize | Code |
|------------|----------|------|
| `i` (signed int) | 4 | `NATIVE_INT32` |
| `u` (unsigned int) | 4 | `NATIVE_UINT32` |
| `i` (signed int) | 8 | `NATIVE_INT64` |
| `f` (float) | 4 | `NATIVE_FLOAT32` |
| `f` (float) | 8 | `NATIVE_FLOAT64` |

Any other dtype (e.g. `int16`, `float16`) is coerced to `float64` once up front,
and the copy is kept alive in `coerced_cols` until the hot loop finishes.

---

## The two-digit lookup table

Rather than one division per digit, both `write_int` and `write_float` use a
200-byte static table `_DIGITS2` that encodes all two-character pairs `"00"` through
`"99"`.  Each loop iteration divides by 100 and looks up two digits at once:

```
v = 12345

iteration 1:  q = 12345 // 100 = 123,  r = 12345 - 123*100 = 45
              → _DIGITS2[45*2]   = '4'
              → _DIGITS2[45*2+1] = '5'    written to tmp[pos..pos+1]
              v = 123

iteration 2:  q = 123 // 100 = 1,  r = 123 - 1*100 = 23
              → _DIGITS2[23*2]   = '2'
              → _DIGITS2[23*2+1] = '3'    written to tmp[pos-2..pos-1]
              v = 1   (< 100 → exit loop)

remainder:    v = 1 < 10 → single digit '1'

result in tmp (right-to-left fill, MSB at tmp[pos]):  "12345"
→ memcpy to buf (no reversal needed)
```

This halves the integer divisions compared to the per-digit approach.

---

## `write_int` — integer formatter

```
write_int(buf, val: long long) → chars written

  val == 0          → write '0', return 1
  val == LLONG_MIN  → memcpy precomputed "-9223372036854775808", return 20
  |val| < 10        → 1 digit directly (no table, no loop)
  |val| < 100       → 2 digits via _DIGITS2 (no loop)
  |val| >= 100      → two-digit loop → memcpy from tmp
```

`LLONG_MIN` (`-9223372036854775808`) is special-cased because negating it is C
undefined behaviour (signed integer overflow); the precomputed string `_LLONG_MIN_STR`
is emitted directly via `memcpy`.

---

## `write_float` — fixed-point formatter

Implements `%.{prec}f` for `prec` 0..15.

```
write_float(buf, val: double, prec: int, scale: long long) → chars written

  scale = 10^prec  (precomputed once per column, passed in)

  1. nan → "nan"; +inf → "inf"; -inf → "-inf"
  2. neg = copysign(1.0, val) < 0.0    ← detects -0.0 correctly
  3. round:  ival = rint(val * scale)   [rintl for prec >= 9]
  4. int_part  = ival // scale
     frac_part = ival % scale
  5. emit '-' if neg
  6. emit int_part  via write_int
  7. if prec > 0: emit '.' then frac digits via two-digit loop into ftmp
```

Worked example for `%.2f` on `val = 3.456`:

```
scale = 100
ival  = rint(3.456 * 100) = rint(345.6) = 346
int_part  = 346 // 100 = 3
frac_part = 346 % 100  = 46

output: "3" + "." + _DIGITS2[46*2.."46*2+1"] = "3.46"
```

### Rounding

Matching Python's `%` operator requires IEEE 754 round-half-to-even.  Two rounding
functions are used depending on precision:

```
prec 0..8:   ival = (long long) c_rint(val * scale)
prec 9..15:  ival = (long long) rintl((long double)val * (long double)scale)
```

`c_rint` maps to the SSE2 `roundsd` instruction — fast (~1 cycle), but operates in
64-bit double precision.  For high precisions the multiply `val * scale` can consume
all 53 mantissa bits, leaving the result off by one in the last decimal place.

Example: `%.9f` on `val = 1.0000000005`

```
scale = 1_000_000_000

c_rint path (double):
  val * scale = 1000000000.5  (rounds at the edge of double precision)
  c_rint(...)  may give  1000000000  → "1.000000000"   ← wrong

rintl path (80-bit long double, x86):
  val * scale has 64 mantissa bits of precision
  rintl(...)   gives  1000000001  → "1.000000001"   ← matches Python
```

The threshold prec >= 9 is chosen because at that point `scale >= 1e9`, and a value
near `1e9` times a typical input starts to exhaust the 53-bit mantissa.  Below the
threshold `c_rint` is used because it is ~12% faster on benchmarks.

### Negative zero

`val < 0.0` returns `False` for `-0.0`, which would cause it to be printed as
`"0.00"` instead of `"-0.00"`.  `copysign(1.0, val) < 0.0` reads the raw sign bit
and correctly identifies `-0.0`, so the output matches Python:

```
val = -0.0

val < 0.0                     → False   (arithmetic: -0.0 == 0.0)
copysign(1.0, -0.0) < 0.0    → True    (sign bit is set)

→ neg = True → '-' prepended → "-0.00"   matches Python '%.2f' % -0.0
```

---

## Hot loop dispatch

The GIL is released for the entire hot loop.  For each `(row i, col j)`:

```
raw_ptr = col_ptr[j] + i * col_stride[j]
```

The column type then determines the path.  Each is shown separately below.

---

### COL_INT — `%d` / `%i` / `%u`

Integer dtypes (`int32`, `uint32`, `int64`) fit directly in `long long` — no overflow
guard needed.  Float dtypes need a guard because `nan`/`inf`/large values can't be
truncated to `long long`.

```
                         COL_INT
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
      int32/int64         uint32          float32/float64
          │                 │             (shown separately below)
    ll_val = cast     ll_val = cast
          │                 │
     sign flag?        sign flag?
     ll_val >= 0?      (unconditional:    ← uint32 is always >= 0
       │       │        uint32 always
      yes      no       gets sign)
       │       │          │       │
    buf[pos] write_int  buf[pos] write_int
    = '+'/   (buf+pos,  = '+'/   (buf+pos,
      ' '     ll_val)    ' '     ll_val)
    write_int           write_int
    (buf+1,             (buf+1,
     ll_val)             ll_val)

          float32 / float64
          (float dtype, %d fmt)
                       │
               dbl_val = cast
                       │
          nan / inf / |val| >= 9.2e18?
                  │            │
                 yes            no
                  │             │
           ┌─with gil─┐    truncate:
           │ fmt%val  │    ll_val = (long long)dbl_val
           │ .encode()│         │
           └──────────┘    sign flag?
                           dbl_val >= 0.0?   ← -0.0 treated as +0, matching Python
                             │       │
                            yes      no
                             │       │
                         buf[pos]  write_int
                         = '+'/    (buf+pos,
                           ' '      ll_val)
                         write_int
                         (buf+1, ll_val)
```

---

### COL_FIXED — `%.Xf`

All dtypes are cast to `double` first.  The overflow guard fires only for finite values
too large to represent as `long long` after scaling.  `inf` and `nan` bypass the guard
and go directly to `write_float`, which handles them at C level.

```
                   COL_FIXED
                       │
          any dtype → dbl_val = (double)cast
                       │
          finite AND |val| >= 9.2e18 / scale?
          (i.e. val * scale would overflow long long)
                  │            │
                 yes            no  (includes inf and nan)
                  │             │
           ┌─with gil─┐    sign flag?
           │ fmt%val  │    nan OR copysign(1,val) > 0?   ← excludes -0.0
           │ .encode()│         │             │
           └──────────┘        yes             no
                                │              │
                            buf[pos]       write_float
                            = '+'/         (buf+pos,
                              ' '           dbl_val,
                            write_float     prec,
                            (buf+1, …)      scale)
```

The sign flag condition uses `copysign` to exclude `-0.0`: Python produces `"-0.00"`
for `'%+.2f' % -0.0`, so `-0.0` must not get a `'+'` prepended — it falls to the
`else` branch where `write_float` handles the sign itself.

---

### COL_PYTHON — `%e`, `%g`, `%x`, width fields, etc.

```
                  COL_PYTHON
                      │
          any dtype → dbl_val = (double)cast   ← done nogil
                      │
                 ┌─with gil─┐
                 │ fmt%val  │
                 │ .encode()│
                 └──────────┘
```

The value is read at C level (nogil-safe), then the GIL is reacquired only for the
Python `%` call.  In production schemas this path is never taken; it exists as a
correctness fallback for any specifier the fast paths don't cover.

---

The `with gil:` reacquisitions are the only points where the hot loop touches Python
objects.  Everything else — pointer arithmetic, dtype dispatch, `write_int`,
`write_float`, comma and newline insertion — is pure C.

---

## Output buffer

All formatted bytes are written into a single flat buffer, then handed to the output
file in one call.  This avoids repeated small `file.write()` calls, which would each
involve Python overhead and a system call.

```
call 1 (1000 rows, %d,%.2f)          call 2 (500 rows, same schema)

  _tls.buf (bytearray, 256 KB)
  ┌─────────────────────────────┐
  │ 1,2.50\n3,4.75\n … (12 KB) │  ← pos = 12000 after hot loop
  └─────────────────────────────┘
          │
          └─ PyUnicode_DecodeASCII(&buf[0], pos)
             → Python str (no copy: reads directly from buf)
             → output_file.write(str)

  call 2: buf is reused (pos reset to 0), no malloc/zero
```

The buffer lives in `threading.local()` so each thread has its own copy — concurrent
calls are safe without any locking.  The buffer grows when needed but never shrinks:
the largest dataset seen so far sets the high-water mark for that thread.

Size before the loop is estimated conservatively per column type:

```
col_type    reserved per cell   rationale
─────────   ─────────────────   ──────────────────────────────────────────
COL_INT          22 bytes       sign + up to 20 decimal digits (LLONG_MIN)
COL_FIXED        32 bytes       sign + digits + '.' + up to 15 frac digits
COL_PYTHON      128 bytes       generous headroom for any format specifier

+ n_cols bytes for commas and the trailing '\n'
+ 64 bytes of slack

total = n_rows * chars_per_row + 64
floor = 262144 bytes (256 KB minimum, even for tiny inputs)
```

At the end, `PyUnicode_DecodeASCII(&buf[0], pos)` constructs a Python `str` directly
from the C buffer — there is no intermediate `bytearray` → `bytes` → `str` conversion.

---

## Limitations

- **MAX_COLS = 64**: All per-column metadata is stack-allocated up to this limit.
  Schemas with more than 64 columns will raise `ValueError`.  To lift the limit,
  increase `MAX_COLS` and recompile.

- **ASCII output only**: `PyUnicode_DecodeASCII` will raise if any COL_PYTHON
  formatter produces non-ASCII characters (e.g. locale-dependent decimal separators).

- **COL_FIXED range**: Only precisions 0..15 are handled natively.  `%.16f` and
  beyond fall through to COL_PYTHON.

- **No width field**: Format specifiers with a minimum width (e.g. `%10d`, `%8.2f`)
  fall through to COL_PYTHON.  The fast path handles sign flags (`+`, ` `) only.

- **Float columns with %d**: Values that are `nan`, `inf`, or outside ±9.2×10¹⁸
  (i.e. would overflow `long long`) fall through to Python `%`.  This matches Python's
  own `'%d' % float('inf')` behaviour (raises `OverflowError`).

- **Integer overflow in COL_FIXED**: If `val * scale > 9.2e18` (e.g. a very large
  float with many decimal places requested), the Python `%` fallback is used.

- **Unusual dtypes**: Any dtype not in {`int32`, `uint32`, `int64`, `float32`,
  `float64`} is coerced to `float64` before the hot loop — a one-time copy per column,
  not per row.

- **Thread safety**: Safe for concurrent use only if each thread calls `write_rows`
  independently.  The buffer is per-thread (`threading.local`), but `output_file` is
  not protected — callers must not share a file handle across threads.
