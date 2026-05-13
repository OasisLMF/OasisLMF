"""
Unified integer key → index lookup.

Auto-selects strategy at build time:
    mode 0: flat   (density >= 25%, shifted by min_id)
    mode 1: sorted uniform   (avg_err < 500, use interpolation)
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
    if density >= 0.25:
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
    avg_error = _avg_err(keys)
    if avg_error < 500.0:
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
    mode = arr[0]
    data = arr[HEADER:]

    # ── FLAT ──
    if mode == MODE_FLAT:
        if len(sorted_keys) == 0 or sorted_keys[0] >= arr[2]:
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


# ──────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':

    def run_test(label, keys, values, missing):
        print(f"\n=== {label} ===")
        arr = build(keys)
        mode = ['flat', 'sorted_uniform', 'sorted_nonuniform'][int(arr[0])]
        dtype_name = keys.dtype.name
        data_bytes = arr.shape[0] * arr.itemsize
        print(f"  mode={mode}  dtype={dtype_name}  n={arr[1]}  min_id={arr[2]}  max_id={arr[3]}"
              f"  arr={arr.shape[0]} ({data_bytes:,} bytes)")

        # single
        ok = 0
        for i in range(keys.shape[0]):
            idx = get_idx(arr, keys[i])
            if idx != NOT_FOUND and values[idx] == values[i]:
                ok += 1
            else:
                print(f"  FAIL single key={keys[i]} idx={idx}")
        print(f"  single hits: {ok}/{keys.shape[0]} OK")

        miss_ok = sum(1 for k in missing if get_idx(arr, keys.dtype.type(k)) == NOT_FOUND)
        print(f"  single misses: {miss_ok}/{len(missing)} OK")

        # batch (unsorted)
        lookup = np.concatenate([keys, np.array(missing, dtype=keys.dtype)])
        out = np.empty(lookup.shape[0], dtype=np.uint32)
        get_idx_batch(arr, lookup, out)
        batch_ok = sum(1 for i in range(keys.shape[0]) if out[i] != NOT_FOUND and lookup[i] == keys[out[i]])
        miss_batch = sum(1 for i in range(keys.shape[0], lookup.shape[0]) if out[i] == NOT_FOUND)
        print(f"  batch: {batch_ok} hits, {miss_batch} misses"
              f"  {'OK' if batch_ok == keys.shape[0] and miss_batch == len(missing) else 'FAIL'}")

        # sorted batch
        sorted_lookup = np.sort(lookup)
        out_s = np.empty(sorted_lookup.shape[0], dtype=np.uint32)
        get_idx_sorted_batch(arr, sorted_lookup, out_s)
        mismatch = sum(1 for i in range(sorted_lookup.shape[0])
                       if out_s[i] != get_idx(arr, sorted_lookup[i]))
        found_s = sum(1 for i in range(sorted_lookup.shape[0]) if out_s[i] != NOT_FOUND)
        print(f"  sorted batch: {found_s} found, {mismatch} mismatches"
              f"  {'OK' if mismatch == 0 else 'FAIL'}")

    rng = np.random.RandomState(42)

    # uint32
    keys1 = np.arange(100, 200, dtype=np.uint32)
    vals1 = (keys1 * 10).astype(np.int32)
    run_test("uint32 dense flat", keys1, vals1, [50, 99, 200, 999])

    keys2 = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    vals2 = np.arange(keys2.shape[0], dtype=np.int32)
    run_test("uint32 sorted_uniform", keys2, vals2, [1_000_001])

    # int32
    keys3 = np.arange(100, 200, dtype=np.int32)
    vals3 = (keys3 * 10).astype(np.int32)
    run_test("int32 dense flat", keys3, vals3, [50, 99, 200, 999])

    keys4 = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.int32)
    vals4 = np.arange(keys4.shape[0], dtype=np.int32)
    run_test("int32 sorted_uniform", keys4, vals4, [1_000_001])

    # uint64
    keys5 = np.arange(100, 200, dtype=np.uint64)
    vals5 = (keys5 * 10).astype(np.int32)
    run_test("uint64 dense flat", keys5, vals5, [50, 99, 200, 999])

    # int64
    keys6 = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.int64)
    vals6 = np.arange(keys6.shape[0], dtype=np.int32)
    run_test("int64 sorted_uniform", keys6, vals6, [1_000_001])

    # flat offset with wide query range (sorted batch clipping test)
    print("\n=== Flat offset, wide query (sorted batch clipping) ===")
    keys_off = np.arange(4500, 5500, dtype=np.uint32)
    arr_off = build(keys_off)
    mode_off = ['flat', 'sorted_uniform', 'sorted_nonuniform'][int(arr_off[0])]
    print(f"  mode={mode_off}  n={arr_off[1]}  min={arr_off[2]}  max={arr_off[3]}")
    # lookups span 0-10000 (10x wider than target 4500-5500)
    lk_wide = np.sort(rng.randint(0, 10_000, size=5000).astype(np.uint32))
    out_wide = np.empty(lk_wide.shape[0], dtype=np.uint32)
    get_idx_sorted_batch(arr_off, lk_wide, out_wide)
    mismatch_off = sum(1 for i in range(len(lk_wide))
                       if out_wide[i] != get_idx(arr_off, lk_wide[i]))
    found_off = sum(1 for x in out_wide if x != NOT_FOUND)
    print(f"  sorted batch: {found_off} found, {mismatch_off} mismatches"
          f"  {'OK' if mismatch_off == 0 else 'FAIL'}")
    # also test unsorted batch
    lk_wide_unsorted = rng.randint(0, 10_000, size=5000).astype(np.uint32)
    out_wide2 = np.empty(lk_wide_unsorted.shape[0], dtype=np.uint32)
    get_idx_batch(arr_off, lk_wide_unsorted, out_wide2)
    mismatch_off2 = sum(1 for i in range(len(lk_wide_unsorted))
                        if out_wide2[i] != get_idx(arr_off, lk_wide_unsorted[i]))
    print(f"  batch: {mismatch_off2} mismatches  {'OK' if mismatch_off2 == 0 else 'FAIL'}")

    # non-uniform
    keys7 = np.unique(((rng.pareto(1.5, size=20000) + 1) * 100).astype(np.uint32))[:5000]
    vals7 = np.arange(keys7.shape[0], dtype=np.int32)
    run_test("uint32 sorted_nonuniform", keys7, vals7, [999_999_999])

    # large
    print("\n=== Large correctness (100K uint32) ===")
    keys_big = np.unique(rng.randint(50_000, 5_000_000, size=100_000)).astype(np.uint32)
    arr_big = build(keys_big)
    mode = ['flat', 'sorted_uniform', 'sorted_nonuniform'][int(arr_big[0])]
    print(f"  mode={mode}  n={arr_big[1]}  arr={arr_big.shape[0]} ({arr_big.shape[0]*4:,} bytes)")

    lookup_big = keys_big[rng.permutation(keys_big.shape[0])]
    out_big = np.empty(lookup_big.shape[0], dtype=np.uint32)
    get_idx_batch(arr_big, lookup_big, out_big)
    ok_big = sum(1 for i in range(len(lookup_big))
                 if out_big[i] != NOT_FOUND and lookup_big[i] == keys_big[out_big[i]])
    print(f"  batch: {ok_big}/{keys_big.shape[0]} ({'OK' if ok_big == keys_big.shape[0] else 'FAIL'})")

    extra = rng.randint(50_000, 5_000_000, size=50_000).astype(np.uint32)
    mixed = np.sort(np.concatenate([keys_big, extra]))
    out_mixed = np.empty(mixed.shape[0], dtype=np.uint32)
    get_idx_sorted_batch(arr_big, mixed, out_mixed)
    mismatch = sum(1 for i in range(len(mixed))
                   if out_mixed[i] != get_idx(arr_big, mixed[i]))
    found_m = sum(1 for x in out_mixed if x != NOT_FOUND)
    print(f"  sorted: {found_m} found, {mismatch} mismatches ({'OK' if mismatch == 0 else 'FAIL'})")

    # ── match() tests ──
    def check_match(label, query_keys, target_keys):
        """Verify match(target, query, out) against brute-force."""
        arr_q = build(query_keys)
        arr_t = build(target_keys)
        mode_q = ['flat', 'interp', 'binary', 'empty'][int(arr_q[0])]
        mode_t = ['flat', 'interp', 'binary', 'empty'][int(arr_t[0])]
        n_q = int(arr_q[1])

        out_m = np.empty(n_q, dtype=np.uint32)
        match(arr_t, arr_q, out_m)

        # brute force: for each query key, search in target
        errors = 0
        for i in range(n_q):
            expected = get_idx(arr_t, query_keys[i])
            if out_m[i] != expected:
                errors += 1
                if errors <= 3:
                    print(f"    MISMATCH i={i} key={query_keys[i]} got={out_m[i]} expected={expected}")

        found = sum(1 for x in out_m if x != NOT_FOUND)
        status = "OK" if errors == 0 else f"FAIL({errors})"
        print(f"  {label}: {mode_q}x{mode_t}  n_q={n_q}  found={found}  {status}")

    print("\n=== match() tests ===")

    # sorted x sorted (full overlap)
    kq = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    kt = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    check_match("sorted x sorted (overlap)", kq, kt)

    # sorted x sorted (no overlap)
    kq2 = np.arange(0, 1000, dtype=np.uint32)
    kt2 = np.arange(2000, 3000, dtype=np.uint32)
    check_match("sorted x sorted (disjoint)", kq2, kt2)

    # flat x flat (same range)
    kq3 = np.arange(100, 200, dtype=np.uint32)
    kt3 = np.arange(150, 250, dtype=np.uint32)
    check_match("flat x flat (partial overlap)", kq3, kt3)

    # flat x flat (full overlap)
    kq4 = np.arange(0, 100, dtype=np.uint32)
    kt4 = np.arange(0, 100, dtype=np.uint32)
    check_match("flat x flat (identical)", kq4, kt4)

    # flat query x sorted target
    kq5 = np.arange(0, 1000, dtype=np.uint32)
    kt5 = np.unique(rng.randint(0, 10_000_000, size=50000)).astype(np.uint32)
    check_match("flat x sorted", kq5, kt5)

    # sorted query x flat target
    kq6 = np.unique(rng.randint(0, 10_000_000, size=50000)).astype(np.uint32)
    kt6 = np.arange(0, 1000, dtype=np.uint32)
    check_match("sorted x flat", kq6, kt6)

    # flat query x sorted target, query range inside target
    kq7 = np.arange(500, 600, dtype=np.uint32)
    kt7 = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    check_match("flat x sorted (query inside target)", kq7, kt7)

    # non-uniform query x sorted target
    kq8 = np.unique(((rng.pareto(1.5, size=20000) + 1) * 100).astype(np.uint32))[:5000]
    kt8 = np.unique(rng.randint(0, 1_000_000, size=5000)).astype(np.uint32)
    check_match("nonuniform x sorted", kq8, kt8)

    # large match
    kq_big = np.unique(rng.randint(0, 5_000_000, size=100_000)).astype(np.uint32)
    kt_big = np.unique(rng.randint(0, 5_000_000, size=100_000)).astype(np.uint32)
    check_match("large sorted x sorted (100K each)", kq_big, kt_big)

    # empty
    kq_empty = np.array([], dtype=np.uint32)
    kt_full = np.arange(100, dtype=np.uint32)
    check_match("empty x full", kq_empty, kt_full)
    check_match("full x empty", kt_full, kq_empty)
