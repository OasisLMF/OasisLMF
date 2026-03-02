#!/usr/bin/env python3
"""
Benchmark: write_ndarray_to_fmt_csv vs charmod alternative.

Assumes data is already chunked by the caller.

Run from the OasisLMF repo root:
    python benchmark_write_csv.py
"""

import io
import timeit
import tracemalloc
import sys

from line_profiler import profile
import numpy as np

sys.path.insert(0, ".")
from oasislmf.pytools.elt.data import MELT_headers, MELT_dtype, MELT_fmt
from oasislmf.pytools.plt.data import MPLT_headers, MPLT_dtype, MPLT_fmt
from oasislmf.pytools.aal.data import ALCT_headers, ALCT_dtype, ALCT_fmt
from oasislmf.pytools.lec.data import PSEPT_headers, PSEPT_dtype, PSEPT_fmt

# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

@profile
def original(output_file, data, headers, row_fmt):
    """Current production implementation."""
    data_cpy = np.empty((data.shape[0], len(headers)))
    for i in range(len(headers)):
        data_cpy[:, i] = data[headers[i]]

    final_fmt = "\n".join([row_fmt] * data_cpy.shape[0])
    flat     = tuple(np.ravel(data_cpy))
    str_data = final_fmt % flat
    output_file.write(str_data)
    output_file.write("\n")



IMPLEMENTATIONS = [
    ("original",   original),
]

# ---------------------------------------------------------------------------
# Cython snprintf implementation (compiled on first run via pyximport)
# ---------------------------------------------------------------------------
try:
    from oasislmf.pytools.common._write_csv_cython import write_rows as _cython_write_rows

    @profile
    def cython_snprintf(output_file, data, headers, row_fmt):
        _cython_write_rows(output_file, data, headers, row_fmt)

    IMPLEMENTATIONS.append(("cython", cython_snprintf))
    print("Cython extension compiled and loaded.\n")
except Exception as _e:
    print(f"Cython extension unavailable ({_e}), skipping.\n")


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def make_data(dtype, n, seed=42):
    rng = np.random.default_rng(seed)
    data = np.empty(n, dtype=dtype)
    for col, (dt, _) in dtype.fields.items():
        if np.issubdtype(dt, np.integer):
            hi = min(int(np.iinfo(dt).max), 10_000)
            data[col] = rng.integers(1, hi, size=n, dtype=dt)
        else:
            data[col] = rng.uniform(0.0, 100_000.0, size=n).astype(dt)
    return data


def bench_time(fn, data, headers, fmt, repeats=5, number=3):
    """Return the best-of-repeats average wall time in seconds."""
    def run():
        buf = io.StringIO()
        fn(buf, data, headers, fmt)

    t = timeit.Timer(run)
    t.timeit(1)  # warmup
    return min(t.repeat(repeat=repeats, number=number)) / number


def bench_memory(fn, data, headers, fmt):
    """Return (peak_bytes, output_bytes) measured by tracemalloc."""
    tracemalloc.start()
    buf = io.StringIO()
    fn(buf, data, headers, fmt)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out_bytes = buf.tell() * 2  # StringIO stores UTF-16 internally; approx via len
    return peak, buf.tell()


def verify_equal(data, headers, fmt):
    """Sanity-check: both implementations must produce identical output."""
    bufs = {}
    for name, fn in IMPLEMENTATIONS:
        buf = io.StringIO()
        fn(buf, data, headers, fmt)
        bufs[name] = buf.getvalue()

    names = list(bufs)
    for i in range(1, len(names)):
        if bufs[names[0]] != bufs[names[i]]:
            print(f"  [MISMATCH] {names[0]} vs {names[i]}")
            print(f"  first 200 chars of {names[0]}: {bufs[names[0]][:200]}")
            print(f"  first 200 chars of {names[i]}: {bufs[names[i]][:200]}")
            return False
    return True


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    # (label, headers, dtype, fmt)
    ("MELT", MELT_headers, MELT_dtype, MELT_fmt),
    ("MPLT", MPLT_headers, MPLT_dtype, MPLT_fmt),
    ("ALCT", ALCT_headers, ALCT_dtype, ALCT_fmt),
    ("PSEPT", PSEPT_headers, PSEPT_dtype, PSEPT_fmt),
]

SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

COL_W = dict(case=44, n=10, impl=12, time_ms=12, peak_mb=10, out_kb=10, speedup=12)

def header_line():
    return (
        f"{'dtype / shape':<{COL_W['case']}}"
        f"{'N':>{COL_W['n']}}"
        f"{'impl':>{COL_W['impl']}}"
        f"{'time (ms)':>{COL_W['time_ms']}}"
        f"{'peak MB':>{COL_W['peak_mb']}}"
        f"{'output KB':>{COL_W['out_kb']}}"
        f"{'speedup':>{COL_W['speedup']}}"
    )

def result_line(case, n, impl, time_s, peak_b, out_b, speedup=None):
    sp = f"{speedup:.2f}x" if speedup is not None else "baseline"
    return (
        f"{case:<{COL_W['case']}}"
        f"{n:>{COL_W['n']},}"
        f"{impl:>{COL_W['impl']}}"
        f"{time_s * 1000:>{COL_W['time_ms']}.2f}"
        f"{peak_b / 1e6:>{COL_W['peak_mb']}.1f}"
        f"{out_b / 1e3:>{COL_W['out_kb']}.1f}"
        f"{sp:>{COL_W['speedup']}}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep = "-" * sum(COL_W.values())
    print(header_line())
    print(sep)

    for case_label, headers, dtype, fmt in CASES:
        for n in SIZES:
            data = make_data(dtype, n)

            ok = verify_equal(data, headers, fmt)
            if not ok:
                print(f"  Output mismatch for {case_label} n={n}, skipping.")
                continue

            baseline_time = None
            for impl_name, fn in IMPLEMENTATIONS:
                t = bench_time(fn, data, headers, fmt)
                peak, out_bytes = bench_memory(fn, data, headers, fmt)

                speedup = None if impl_name == "original" else (baseline_time / t if baseline_time else None)
                if impl_name == "original":
                    baseline_time = t

                print(result_line(case_label, n, impl_name, t, peak, out_bytes, speedup))

        print()
