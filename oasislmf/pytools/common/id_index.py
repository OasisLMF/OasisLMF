"""
Unified integer key → index lookup.

Auto-selects strategy at build time:
    mode 0: flat   (density >= 25%, shifted by min_id)
    mode 1: sorted uniform   (avg_err / n < 0.02, use interpolation)
    mode 2: sorted non-uniform (use binary search)

Keys must be sorted on input. Return array has same dtype as input keys.

Header (first HEADER elements, same dtype as keys):
    arr[0] = mode (0=flat, 1=sorted_uniform, 2=sorted_nonuniform)
    arr[1] = n_keys
    arr[2] = min_id
    arr[3] = max_id

Data (arr[HEADER:]):
    Flat:   arr[HEADER + (key - min_id)] = index  (NOT_FOUND = empty)
    Sorted: arr[HEADER + i] = keys[i], position i = index

Batch output:
    out[i] = index into keys passed to build(), or NOT_FOUND (0xFFFFFFFF).
"""
import numpy as np
import numba as nb
from numba import types

HEADER = 4
HEADER_N_KEYS = 1
NOT_FOUND = np.uint32(0xFFFFFFFF)

MODE_FLAT = 0
MODE_SORTED_UNIFORM = 1
MODE_SORTED_NONUNIFORM = 2
MODE_EMPTY = 3

# Build-time mode selection thresholds.
FLAT_DENSITY_THRESHOLD = 0.25       # density >= this → MODE_FLAT
# Scale-invariant: _avg_err / n is the average fraction of n by which a
# linear interpolation mispredicts a key's position. Empirical bench shows
# the interp-vs-binary crossover sits around 0.02–0.075 depending on the
# distribution shape; 0.02 is the tightest value that classifies every
# distribution tested correctly. See tmp/id_index_bench/bench_heuristic.py.
SORTED_UNIFORM_REL_ERR = 0.02       # _avg_err / n < this → MODE_SORTED_UNIFORM

# dtype shorthand (used for hot-path helper signatures)
_u32 = types.uint32
_i32 = types.int32
_u64 = types.uint64
_i64 = types.int64
_u32_1d = _u32[::1]
_i32_1d = _i32[::1]
_u64_1d = _u64[::1]
_i64_1d = _i64[::1]


# ──────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _avg_err(sorted_keys):
    """Average interpolation estimation error (sampled)."""
    if len(sorted_keys) < 2:
        return 0.0
    lo_val = np.float64(sorted_keys[0])
    hi_val = np.float64(sorted_keys[len(sorted_keys) - 1])
    if hi_val == lo_val:
        return 0.0
    total = 0.0
    step = max(1, len(sorted_keys) // 1000)
    count = 0
    for i in range(0, len(sorted_keys), step):
        frac = (np.float64(sorted_keys[i]) - lo_val) / (hi_val - lo_val)
        total += abs(frac * (len(sorted_keys) - 1) - i)
        count += 1
    return total / count


@nb.njit(cache=True)
def build(keys):
    """Build a self-describing lookup structure.

    Args:
        keys: sorted array of unique keys (uint32, int32, uint64, or int64).

    Returns:
        arr: array of same dtype as keys, with header + data.
    """
    n = keys.shape[0]
    dtype = keys.dtype

    if n == 0:
        arr = np.zeros(HEADER, dtype=dtype)
        arr[0] = MODE_EMPTY
        return arr

    min_id = keys[0]
    max_id = keys[-1]

    id_range = np.int64(max_id) - np.int64(min_id) + 1
    density = n / id_range

    # ── FLAT MODE ──
    if density >= FLAT_DENSITY_THRESHOLD:
        arr = np.full(HEADER + id_range, NOT_FOUND, dtype=dtype)
        arr[0] = MODE_FLAT
        arr[1] = dtype.type(n)
        arr[2] = min_id
        arr[3] = max_id
        data = arr[HEADER:]
        for i in range(n):
            data[np.int64(keys[i] - min_id)] = dtype.type(i)
        return arr

    # ── SORTED MODE (keys already sorted, store directly) ──
    rel_error = _avg_err(keys) / n
    if rel_error < SORTED_UNIFORM_REL_ERR:
        mode = MODE_SORTED_UNIFORM
    else:
        mode = MODE_SORTED_NONUNIFORM

    arr = np.empty(HEADER + n, dtype=dtype)
    arr[0] = dtype.type(mode)
    arr[1] = dtype.type(n)
    arr[2] = min_id
    arr[3] = max_id
    arr[HEADER:] = keys[:]

    return arr


# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────

@nb.njit([_u32(_u32_1d, _u32, _u32),
          _u32(_i32_1d, _i32, _i32),
          _u32(_u64_1d, _u64, _u64),
          _u32(_i64_1d, _i64, _i64)], cache=True, inline='always')
def _flat_lookup_unsafe(flat_data, key, min_id):
    """O(1) flat lookup. No bounds check — caller must ensure min_id <= key <= max_id."""
    return flat_data[np.int64(key - min_id)]


@nb.njit([_u32(_u32_1d, _u32, _u32, _u32),
          _u32(_i32_1d, _i32, _i32, _i32),
          _u32(_u64_1d, _u64, _u64, _u64),
          _u32(_i64_1d, _i64, _i64, _i64)], cache=True, inline='always')
def _flat_lookup(flat_data, key, min_id, max_id):
    """O(1) flat lookup. Returns index or NOT_FOUND."""
    return NOT_FOUND if key < min_id or key > max_id else flat_data[np.int64(key - min_id)]


@nb.njit([_i64(_u32_1d, _u32, _i64),
          _i64(_i32_1d, _i32, _i64),
          _i64(_u64_1d, _u64, _i64),
          _i64(_i64_1d, _i64, _i64)], cache=True, inline='always')
def _interp_find_pos(data, key, lo):
    """Interpolation search returning position.
    If data[pos] == key: found at pos. Otherwise: not found, pos is resume point."""
    hi = len(data) - 1
    lo_val = data[lo]
    hi_val = data[hi]
    if key <= lo_val:
        return lo
    if key >= hi_val:
        return hi
    while lo < hi:
        frac = np.float64(key - lo_val) / np.float64(hi_val - lo_val)
        mid = lo + np.int64(frac * (hi - lo))
        mid_val = data[mid]
        if key < mid_val:
            hi = mid - 1
            hi_val = data[hi]
            if key >= hi_val:
                return hi
        else:
            lo = mid + 1
            lo_val = data[lo]
            if key < lo_val:
                return mid
    return hi


@nb.njit([_u32(_u32_1d, _u32),
          _u32(_i32_1d, _i32),
          _u32(_u64_1d, _u64),
          _u32(_i64_1d, _i64)], cache=True, inline='always')
def _interp_search(data, key):
    """Interpolation search. Returns sorted position or NOT_FOUND."""
    pos = _interp_find_pos(data, key, np.int64(0))
    if data[pos] == key:
        return np.uint32(pos)
    return NOT_FOUND


@nb.njit([_i64(_u32_1d, _u32, _i64),
          _i64(_i32_1d, _i32, _i64),
          _i64(_u64_1d, _u64, _i64),
          _i64(_i64_1d, _i64, _i64)], cache=True, inline='always')
def _binary_find_pos(data, key, lo):
    """Binary search returning position.
    If data[pos] == key: found at pos. Otherwise: not found, pos is resume point."""
    hi = len(data) - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        mid_val = data[mid]
        if key == mid_val:
            return mid
        elif key < mid_val:
            hi = mid - 1
        else:
            lo = mid + 1
    return hi


@nb.njit([_u32(_u32_1d, _u32),
          _u32(_i32_1d, _i32),
          _u32(_u64_1d, _u64),
          _u32(_i64_1d, _i64)], cache=True, inline='always')
def _binary_search(data, key):
    """Binary search. Returns sorted position or NOT_FOUND."""
    pos = _binary_find_pos(data, key, np.int64(0))
    if data[pos] == key:
        return np.uint32(pos)
    return NOT_FOUND


@nb.njit(cache=True)
def _merge_scan(data, keys, out):
    """Merge scan: for each key in keys, find key in data and set out to its index.
    Writes NOT_FOUND for misses. Both data and keys must be sorted.
    """
    if len(data) == 0 or len(keys) == 0:
        return
    data_i = np.int64(0)
    keys_i = np.int64(0)
    while data_i < len(data) and keys_i < len(keys):
        if data[data_i] == keys[keys_i]:
            out[keys_i] = data_i
            keys_i += 1
        elif data[data_i] < keys[keys_i]:
            data_i += 1
        else:
            out[keys_i] = NOT_FOUND
            keys_i += 1
    out[keys_i:] = NOT_FOUND

# ──────────────────────────────────────────────────────────
# Single lookup
# ──────────────────────────────────────────────────────────


@nb.njit(cache=True)
def get_idx(arr, key):
    """Single lookup. Returns uint32 index or NOT_FOUND.

    Note: explicit np.uint32() casts ensure the NOT_FOUND sentinel (0xFFFFFFFF)
    is returned correctly even when the underlying array has a signed dtype
    (e.g. int32 flat arrays store NOT_FOUND as -1, which must be cast to uint32).
    """
    mode = arr[0]
    data = arr[HEADER:]
    if mode == MODE_FLAT:
        return np.uint32(_flat_lookup(data, key, arr[2], arr[3]))
    if mode == MODE_SORTED_UNIFORM:
        return np.uint32(_interp_search(data, key))
    if mode == MODE_SORTED_NONUNIFORM:
        return np.uint32(_binary_search(data, key))
    return NOT_FOUND


# ──────────────────────────────────────────────────────────
# Batch lookup (unsorted keys)
# ──────────────────────────────────────────────────────────

@nb.njit(cache=True)
def get_idx_batch(arr, keys, out):
    """Batch lookup, keys may be unsorted.

    Args:
        arr: structure from build().
        keys: array of lookup keys (same dtype as build input).
        out: uint32 array of length len(keys).
    """
    mode = arr[0]
    data = arr[HEADER:]
    if mode == MODE_FLAT:
        for i in range(keys.shape[0]):
            out[i] = _flat_lookup(data, keys[i], arr[2], arr[3])
    elif mode == MODE_SORTED_UNIFORM:
        for i in range(keys.shape[0]):
            out[i] = _interp_search(data, keys[i])
    elif mode == MODE_SORTED_NONUNIFORM:
        for i in range(keys.shape[0]):
            out[i] = _binary_search(data, keys[i])
    else:
        out[:len(keys)] = NOT_FOUND


# ──────────────────────────────────────────────────────────
# Sorted batch lookup
# ──────────────────────────────────────────────────────────

@nb.njit(cache=True)
def get_idx_sorted_batch(arr, sorted_keys, out):
    """Batch lookup for sorted keys.

    Precondition: sorted_keys must be unique. In the FLAT branch, duplicates equal to
    arr[3] (the structure's max_id) are silently dropped past the first occurrence
    because the truncation uses `_interp_find_pos`, which returns only one position.
    Callers in this codebase pass deduplicated event/key arrays, so this is fine in
    practice — but if you add a new caller, dedupe first.

    Args:
        arr: structure from build().
        sorted_keys: sorted array (same dtype as build input).
        out: uint32 array of length len(sorted_keys).
    """
    if len(sorted_keys) == 0:
        return

    mode = arr[0]
    data = arr[HEADER:]

    # ── FLAT ──
    if mode == MODE_FLAT:
        if sorted_keys[0] >= arr[2]:
            lo = 0
        else:
            lo = _interp_find_pos(sorted_keys, arr[2], 0)
            if sorted_keys[lo] < arr[2]:
                lo += 1
            out[:lo] = NOT_FOUND

        if sorted_keys[-1] <= arr[3]:
            hi = len(sorted_keys)
        else:
            hi = _interp_find_pos(sorted_keys, arr[3], lo) + 1
            out[hi:] = NOT_FOUND

        for i in range(lo, hi):
            out[i] = _flat_lookup_unsafe(data, sorted_keys[i], arr[2])

    # ── SORTED: merge scan if batch large enough ──
    elif len(sorted_keys) > len(arr) // 10:
        # find first value bigger or equal to min_id
        lo = _interp_find_pos(sorted_keys, arr[2], 0)
        # fill head (keys < min_id)
        out[:lo] = NOT_FOUND
        _merge_scan(data, sorted_keys[lo:], out[lo:])

    # ── SORTED: interpolation with narrowing lo_base ──
    elif mode == MODE_SORTED_UNIFORM:
        pos = np.int64(0)
        for i in range(len(sorted_keys)):
            pos = _interp_find_pos(data, sorted_keys[i], pos)

            if data[pos] == sorted_keys[i]:
                out[i] = np.uint32(pos)
            elif pos == len(data):
                out[i:] = NOT_FOUND
                break
            else:
                out[i] = NOT_FOUND

    elif mode == MODE_SORTED_NONUNIFORM:
        # ── SORTED: binary search with narrowing lo_base ──
        pos = np.int64(0)
        for i in range(len(sorted_keys)):
            pos = _binary_find_pos(data, sorted_keys[i], pos)
            if data[pos] == sorted_keys[i]:
                out[i] = np.uint32(pos)
            elif pos == len(data):
                out[i:] = NOT_FOUND
                break
            else:
                out[i] = NOT_FOUND

    else:
        out[:len(sorted_keys)] = NOT_FOUND

# ──────────────────────────────────────────────────────────
# Match: find query keys in target
# ──────────────────────────────────────────────────────────


@nb.njit(cache=True)
def match(target, query, out):
    """For each key in query, find its index in target.

    Args:
        target: structure from build() (the keys to search in).
        query:  structure from build() (the keys to look up).
        out:    uint32 array of length n_query_keys.
                out[i] = index of query's i-th key in target, or NOT_FOUND.

    Auto-selects strategy:
        query sorted           -> delegates to get_idx_sorted_batch
        both flat              -> iterate overlap range, O(1) per key
        query flat, target sorted -> iterate target overlap, flat check in query
    """
    mode_q = np.int64(query[0])
    n_q = np.int64(query[1])
    mode_t = np.int64(target[0])
    n_t = np.int64(target[1])

    out[:n_q] = NOT_FOUND

    if n_q == 0 or n_t == 0:
        return

    min_q = query[2]
    max_q = query[3]
    min_t = target[2]
    max_t = target[3]

    # ── both sorted: merge scan ──
    if mode_q != MODE_FLAT:
        get_idx_sorted_batch(target, query[HEADER:], out)
        return

    # ── both flat: iterate overlap range ──
    if mode_t == MODE_FLAT:
        flat_t = target[HEADER:]
        # both flat: iterate overlap range
        overlap_lo = np.int64(min_q) if np.int64(min_q) > np.int64(min_t) else np.int64(min_t)
        overlap_hi = np.int64(max_q) if np.int64(max_q) < np.int64(max_t) else np.int64(max_t)
        if overlap_lo > overlap_hi:
            return
        flat_q = query[HEADER:]
        min_q_i64 = np.int64(min_q)
        min_t_i64 = np.int64(min_t)
        for key_i64 in range(overlap_lo, overlap_hi + 1):
            val_q = flat_q[key_i64 - min_q_i64]
            if val_q == NOT_FOUND:  # no value to find in query
                continue
            val_t = flat_t[key_i64 - min_t_i64]
            if val_t != NOT_FOUND:  # value found in target and query
                out[val_q] = val_t
        return

    # ── query flat, target sorted: iterate through target, flat check in query ──
    flat_q = query[HEADER:]
    data_t = target[HEADER:]
    min_q_i64 = np.int64(min_q)

    # find first target key >= min_q
    if mode_t == MODE_SORTED_UNIFORM:
        j_start = _interp_find_pos(data_t, min_q, 0)
    else:
        j_start = _binary_find_pos(data_t, min_q, 0)
    if data_t[j_start] < min_q:
        j_start += 1

    for j in range(j_start, n_t):
        key = data_t[j]  # look key in target
        if key > max_q:  # target key is out of query range
            return
        val_q = flat_q[np.int64(key) - min_q_i64]  # check corresponding key index in flat query
        if val_q != NOT_FOUND:  # query has a value
            out[val_q] = np.uint32(j)
