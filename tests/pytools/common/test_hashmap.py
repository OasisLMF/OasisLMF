"""Unit tests for oasislmf.pytools.common.hashmap.

Mirrors the original __main__ sanity test of hashmap.py, converted into
pytest cases with explicit assertions.
"""
import numpy as np

from oasislmf.pytools.common.hashmap import jit_factorize


def _reference_factorize(arr):
    """Pure-Python reference: 1-based id per row, identical rows share an id."""
    ref = []
    seen = {}
    agg_id = 0
    for val in arr:
        key = tuple(int(val[name]) for name in arr.dtype.names)
        if key not in seen:
            agg_id += 1
            seen[key] = agg_id
        ref.append(seen[key])
    return ref


def test_jit_factorize_matches_python_reference():
    dtype = np.dtype([('a', 'int32'), ('b', 'uint8')])
    rng = np.random.RandomState(1)
    arr = np.empty(1000, dtype=dtype)
    arr['a'] = rng.randint(1, 1000, size=1000)
    arr['b'] = rng.randint(0, 256, size=1000)

    ref = _reference_factorize(arr)
    res = list(jit_factorize(arr))

    assert res == ref
    assert len(set(ref)) > 1  # sanity: at least some unique keys were collapsed


def test_jit_factorize_all_unique():
    """When every row is unique, agg_ids run 1..N in row order."""
    dtype = np.dtype([('a', 'int32'), ('b', 'uint8')])
    n = 50
    arr = np.empty(n, dtype=dtype)
    arr['a'] = np.arange(n, dtype=np.int32)
    arr['b'] = np.zeros(n, dtype=np.uint8)

    res = list(jit_factorize(arr))
    assert res == list(range(1, n + 1))


def test_jit_factorize_all_identical():
    """When every row is the same, every agg_id is 1."""
    dtype = np.dtype([('a', 'int32'), ('b', 'uint8')])
    n = 25
    arr = np.empty(n, dtype=dtype)
    arr['a'] = np.full(n, 7, dtype=np.int32)
    arr['b'] = np.full(n, 3, dtype=np.uint8)

    res = list(jit_factorize(arr))
    assert res == [1] * n
