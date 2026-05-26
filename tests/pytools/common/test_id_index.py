"""Unit tests for oasislmf.pytools.common.id_index.

Mirrors the original __main__ sanity tests of id_index.py, converted into
pytest cases with explicit assertions.
"""
import numpy as np
import pytest

from oasislmf.pytools.common.id_index import (
    HEADER,
    NOT_FOUND,
    MODE_FLAT,
    MODE_SORTED_UNIFORM,
    MODE_SORTED_NONUNIFORM,
    MODE_EMPTY,
    build,
    get_idx,
    get_idx_batch,
    get_idx_sorted_batch,
    match,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_lookup(keys, missing):
    """Verify single, batch (unsorted), and sorted-batch lookups against `keys`."""
    arr = build(keys)

    # All present keys must round-trip via single get_idx.
    for i in range(keys.shape[0]):
        idx = get_idx(arr, keys[i])
        assert idx != NOT_FOUND, f"single key={keys[i]} missing"
        assert keys[idx] == keys[i]

    # All missing keys must return NOT_FOUND.
    for k in missing:
        assert get_idx(arr, keys.dtype.type(k)) == NOT_FOUND

    # Batch (unsorted) lookup.
    lookup = np.concatenate([keys, np.array(missing, dtype=keys.dtype)])
    out = np.empty(lookup.shape[0], dtype=np.uint32)
    get_idx_batch(arr, lookup, out)
    for i in range(keys.shape[0]):
        assert out[i] != NOT_FOUND
        assert lookup[i] == keys[out[i]]
    for i in range(keys.shape[0], lookup.shape[0]):
        assert out[i] == NOT_FOUND

    # Sorted batch lookup — must match single-key lookup for every position.
    sorted_lookup = np.sort(lookup)
    out_s = np.empty(sorted_lookup.shape[0], dtype=np.uint32)
    get_idx_sorted_batch(arr, sorted_lookup, out_s)
    for i in range(sorted_lookup.shape[0]):
        assert out_s[i] == get_idx(arr, sorted_lookup[i])


# ---------------------------------------------------------------------------
# build() mode selection
# ---------------------------------------------------------------------------

def test_build_empty():
    arr = build(np.array([], dtype=np.uint32))
    assert arr[0] == MODE_EMPTY
    assert arr.shape[0] == HEADER


def test_build_flat_mode_dense():
    keys = np.arange(100, 200, dtype=np.uint32)
    arr = build(keys)
    assert arr[0] == MODE_FLAT
    assert arr[1] == keys.shape[0]
    assert arr[2] == 100
    assert arr[3] == 199


def test_build_sorted_uniform_mode():
    rng = np.random.RandomState(42)
    keys = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    arr = build(keys)
    assert arr[0] == MODE_SORTED_UNIFORM


def test_build_sorted_nonuniform_mode():
    rng = np.random.RandomState(42)
    keys = np.unique(((rng.pareto(1.5, size=20000) + 1) * 100).astype(np.uint32))[:5000]
    arr = build(keys)
    assert arr[0] == MODE_SORTED_NONUNIFORM


# ---------------------------------------------------------------------------
# Lookups across dtypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.uint32, np.int32, np.uint64, np.int64])
def test_flat_lookups_all_dtypes(dtype):
    keys = np.arange(100, 200, dtype=dtype)
    _check_lookup(keys, [50, 99, 200, 999])


@pytest.mark.parametrize("dtype", [np.uint32, np.int32, np.int64])
def test_sorted_uniform_lookups(dtype):
    rng = np.random.RandomState(42)
    keys = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(dtype)
    _check_lookup(keys, [1_000_001])


def test_sorted_nonuniform_lookups():
    rng = np.random.RandomState(42)
    keys = np.unique(((rng.pareto(1.5, size=20000) + 1) * 100).astype(np.uint32))[:5000]
    _check_lookup(keys, [999_999_999])


# ---------------------------------------------------------------------------
# Sorted batch with wide query range outside the flat structure
# ---------------------------------------------------------------------------

def test_flat_offset_wide_sorted_batch():
    rng = np.random.RandomState(42)
    keys = np.arange(4500, 5500, dtype=np.uint32)
    arr = build(keys)
    assert arr[0] == MODE_FLAT

    # Lookups span 0..10_000 — 10x wider than the structure's 4500..5500 range.
    # get_idx_sorted_batch requires unique sorted keys.
    lk = np.unique(rng.randint(0, 10_000, size=5000).astype(np.uint32))
    out = np.empty(lk.shape[0], dtype=np.uint32)
    get_idx_sorted_batch(arr, lk, out)
    for i in range(lk.shape[0]):
        assert out[i] == get_idx(arr, lk[i])


def test_flat_offset_wide_unsorted_batch():
    rng = np.random.RandomState(42)
    keys = np.arange(4500, 5500, dtype=np.uint32)
    arr = build(keys)

    lk = rng.randint(0, 10_000, size=5000).astype(np.uint32)
    out = np.empty(lk.shape[0], dtype=np.uint32)
    get_idx_batch(arr, lk, out)
    for i in range(lk.shape[0]):
        assert out[i] == get_idx(arr, lk[i])


# ---------------------------------------------------------------------------
# Large correctness
# ---------------------------------------------------------------------------

def test_large_batch_correctness_100k():
    rng = np.random.RandomState(42)
    keys = np.unique(rng.randint(50_000, 5_000_000, size=100_000)).astype(np.uint32)
    arr = build(keys)

    lookup = keys[rng.permutation(keys.shape[0])]
    out = np.empty(lookup.shape[0], dtype=np.uint32)
    get_idx_batch(arr, lookup, out)
    for i in range(lookup.shape[0]):
        assert out[i] != NOT_FOUND
        assert lookup[i] == keys[out[i]]

    extra = rng.randint(50_000, 5_000_000, size=50_000).astype(np.uint32)
    mixed = np.sort(np.concatenate([keys, extra]))
    out_mixed = np.empty(mixed.shape[0], dtype=np.uint32)
    get_idx_sorted_batch(arr, mixed, out_mixed)
    for i in range(mixed.shape[0]):
        assert out_mixed[i] == get_idx(arr, mixed[i])


# ---------------------------------------------------------------------------
# match() — query keys inside target
# ---------------------------------------------------------------------------

def _check_match(query_keys, target_keys):
    arr_q = build(query_keys)
    arr_t = build(target_keys)
    n_q = int(arr_q[1])

    out = np.empty(n_q, dtype=np.uint32)
    match(arr_t, arr_q, out)

    for i in range(n_q):
        expected = get_idx(arr_t, query_keys[i])
        assert out[i] == expected, f"i={i} key={query_keys[i]} got={out[i]} expected={expected}"


def test_match_sorted_x_sorted_overlap():
    rng = np.random.RandomState(42)
    kq = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    kt = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    _check_match(kq, kt)


def test_match_sorted_x_sorted_disjoint():
    kq = np.arange(0, 1000, dtype=np.uint32)
    kt = np.arange(2000, 3000, dtype=np.uint32)
    _check_match(kq, kt)


def test_match_flat_x_flat_partial_overlap():
    kq = np.arange(100, 200, dtype=np.uint32)
    kt = np.arange(150, 250, dtype=np.uint32)
    _check_match(kq, kt)


def test_match_flat_x_flat_identical():
    kq = np.arange(0, 100, dtype=np.uint32)
    kt = np.arange(0, 100, dtype=np.uint32)
    _check_match(kq, kt)


def test_match_flat_x_sorted():
    rng = np.random.RandomState(42)
    kq = np.arange(0, 1000, dtype=np.uint32)
    kt = np.unique(rng.randint(0, 10_000_000, size=50000)).astype(np.uint32)
    _check_match(kq, kt)


def test_match_sorted_x_flat():
    rng = np.random.RandomState(42)
    kq = np.unique(rng.randint(0, 10_000_000, size=50000)).astype(np.uint32)
    kt = np.arange(0, 1000, dtype=np.uint32)
    _check_match(kq, kt)


def test_match_flat_x_sorted_query_inside_target():
    rng = np.random.RandomState(42)
    kq = np.arange(500, 600, dtype=np.uint32)
    kt = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    _check_match(kq, kt)


def test_match_nonuniform_x_sorted():
    rng = np.random.RandomState(42)
    kq = np.unique(((rng.pareto(1.5, size=20000) + 1) * 100).astype(np.uint32))[:5000]
    kt = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    _check_match(kq, kt)


def test_match_large_sorted_x_sorted_100k():
    rng = np.random.RandomState(42)
    kq = np.unique(rng.randint(0, 5_000_000, size=100_000)).astype(np.uint32)
    kt = np.unique(rng.randint(0, 5_000_000, size=100_000)).astype(np.uint32)
    _check_match(kq, kt)


def test_match_empty_x_full():
    kq = np.array([], dtype=np.uint32)
    kt = np.arange(100, dtype=np.uint32)
    _check_match(kq, kt)


def test_match_full_x_empty():
    kq = np.arange(100, dtype=np.uint32)
    kt = np.array([], dtype=np.uint32)
    _check_match(kq, kt)
