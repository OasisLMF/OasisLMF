"""Unit tests for oasislmf.pytools.common.hashmap.

Tests target the user-facing API documented in the module docstring:
``init_dict``, ``unpack``, ``try_add_key``, ``find_key``, ``rehash``,
``jit_factorize``, ``factorize``.

Strategy: a few high-level tests, each chosen to maximize coverage:
  1. ``jit_factorize`` vs ``pd.factorize`` on a structured array — exercises
     the bulk of the internal Robin Hood machinery (init_dict, unpack,
     _try_add_key by-position, rehash via load factor, structured-record
     fnv1a + key_eq overloads, _move_key).
  2. ``try_add_key`` (by-value, ``i_item=None``) + ``find_key`` parametrized
     over scalar dtypes incl. float32/float64 bit-cast — covers scalar
     overloads, NOT_FOUND path, ``new_slot_bit`` semantics, +0.0/-0.0
     distinguishability for both float widths.
  3. ``try_add_key`` (by-position, ``i_item=int``) with structured records —
     covers the other ``try_add_key`` mode and verifies key_storage is left
     untouched.
  4. Collision-triggered rehash — brute-forces a colliding key set so that
     ``try_add_key`` returns ``i_add_key_fail``, then calls ``rehash`` and
     retries. Covers the ``i_add_key_fail`` return and explicit rehash path.
  5. ``factorize`` (DataFrame wrapper) over mixed dtypes — nullable Int,
     regular numeric, and string columns — vs ``pd.factorize``.
"""
import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.hashmap import (
    init_dict, unpack, try_add_key, find_key, rehash, jit_factorize, factorize,
    i_add_key_fail, new_slot_bit, slot_mask, NOT_FOUND,
    init_hash, FNV_PRIME, M3_C1, M3_C2, hash_key_shift, HM_INFO_N_VALID,
)


def test_jit_factorize_matches_pandas():
    rng = np.random.RandomState(42)
    n = 5_000
    dtype = np.dtype([('a', 'int32'), ('b', 'uint8'), ('c', 'int64')])
    arr = np.empty(n, dtype=dtype)
    arr['a'] = rng.randint(0, 100, size=n, dtype=np.int32)
    arr['b'] = rng.randint(0, 5, size=n).astype('uint8')
    arr['c'] = rng.randint(-1000, 1000, size=n, dtype=np.int64)

    res = np.asarray(jit_factorize(arr))

    # pd.factorize on a MultiIndex returns 0-based codes; jit_factorize is 1-based
    expected, _ = pd.factorize(
        pd.MultiIndex.from_arrays([arr['a'], arr['b'], arr['c']])
    )
    expected = expected + 1

    assert np.array_equal(res, expected)
    # sanity: hash actually collapsed near-duplicates
    assert res.max() < n


@pytest.mark.parametrize(
    ("dtype", "keys_in", "absent"),
    [
        pytest.param(
            np.int32,
            np.array([5, 7, 100, 7, 5, 0], dtype=np.int32),
            np.int32(99999),
            id="int32",
        ),
        pytest.param(
            np.uint64,
            np.array([1, 2**40, 1, 0, 2**40], dtype=np.uint64),
            np.uint64(99999),
            id="uint64",
        ),
        # +0.0 / -0.0 bit-distinct ⇒ distinct slots (float64 bitwidth=64 path)
        pytest.param(
            np.float64,
            np.array([0.0, -0.0, 1.5, 0.0, np.inf, -0.0], dtype=np.float64),
            np.float64(3.14159),
            id="float64",
        ),
        # float32: exercises the bitwidth=32 branch of the fnv1a / key_eq overloads
        # (1-element float32 buffer viewed as uint32). +0.0 / -0.0 still bit-distinct.
        pytest.param(
            np.float32,
            np.array([0.0, -0.0, 1.5, 0.0, np.inf, -0.0], dtype=np.float32),
            np.float32(3.14159),
            id="float32",
        ),
    ],
)
def test_try_add_key_find_key_roundtrip_scalars(dtype, keys_in, absent):
    """Public try_add_key (by-value, i_item=None) + find_key on scalar dtypes.
    Covers scalar fnv1a / key_eq overloads, the NOT_FOUND path, and the
    new_slot_bit semantics on first-vs-repeat insert.

    The float32 and float64 cases exercise bit-cast hashing AND bit-compare
    equality for scalar floats: +0.0 and -0.0 are bit-distinct and must
    therefore hash and compare as distinct keys. The two widths take
    different overload branches (uint32 vs uint64 bit-view)."""
    key_storage = np.empty(len(keys_in), dtype=dtype)
    table = init_dict(len(keys_in))

    seen_first_storage_idx = {}
    for k in keys_in:
        result = try_add_key(table, key_storage, k)
        assert result != i_add_key_fail
        slot = result & slot_mask
        _, _, index = unpack(table)
        key_bytes = k.tobytes()  # bit-identity (separates +0.0 / -0.0)

        if key_bytes in seen_first_storage_idx:
            assert not (result & new_slot_bit), f"dup of {k!r} should reuse slot"
            assert index[slot] == seen_first_storage_idx[key_bytes]
        else:
            assert (result & new_slot_bit), f"first {k!r} should set new_slot_bit"
            seen_first_storage_idx[key_bytes] = index[slot]
            # by-value mode wrote the key to key_storage at the dense index
            assert key_storage[index[slot]].tobytes() == key_bytes

    for k in keys_in:
        slot = find_key(table, key_storage, k)
        assert slot != NOT_FOUND
        _, _, index = unpack(table)
        assert key_storage[index[slot]].tobytes() == k.tobytes()

    assert find_key(table, key_storage, absent) == NOT_FOUND


def test_try_add_key_find_key_roundtrip_unichr_scalar():
    """Scalar UnicodeCharSeq keys (e.g. dtype 'U3') via try_add_key + find_key.
    Covers fnv1a_overload_unichr / key_eq_overload_unichr, which together
    treat the key as a fixed-width unicode string with NUL-trim semantics."""
    keys_in = np.array(['ab', 'cd', 'ab', 'ef', 'cd', 'gh'], dtype='U3')
    key_storage = np.empty(len(keys_in), dtype='U3')
    table = init_dict(len(keys_in))

    seen_first_slot = {}
    for k in keys_in:
        result = try_add_key(table, key_storage, k)
        assert result != i_add_key_fail
        slot = result & slot_mask
        _, _, index = unpack(table)
        s = str(k)

        if s in seen_first_slot:
            assert not (result & new_slot_bit), f"dup of {s!r} should reuse slot"
            assert index[slot] == seen_first_slot[s]
        else:
            assert (result & new_slot_bit), f"first {s!r} should set new_slot_bit"
            seen_first_slot[s] = index[slot]
            assert str(key_storage[index[slot]]) == s

    for k in keys_in:
        slot = find_key(table, key_storage, k)
        assert slot != NOT_FOUND
        _, _, index = unpack(table)
        assert str(key_storage[index[slot]]) == str(k)

    assert find_key(table, key_storage, np.str_('zz')) == NOT_FOUND


def test_try_add_key_by_position_mode():
    """Public try_add_key with explicit i_item. Caller pre-populates
    key_storage; hashmap stores i_item at the slot and must NOT write
    to key_storage. Also exercises the float-field bit-cast path: +0.0
    and -0.0 must be treated as distinct keys."""
    dtype = np.dtype([('areaperil_id', 'int32'), ('vuln_id', 'int32'),
                      ('correlation', 'float64')])
    key_storage = np.array(
        [(1, 10, 0.5), (2, 20, 0.5), (1, 10, 0.5),
         (3, 30, 0.0), (2, 20, 0.5), (3, 30, -0.0)],
        dtype=dtype,
    )
    snapshot = key_storage.copy()
    table = init_dict(len(key_storage))

    slots = []
    for i, k in enumerate(key_storage):
        result = try_add_key(table, key_storage, k, i)
        assert result != i_add_key_fail
        slot = int(result & slot_mask)
        slots.append(slot)
        _, _, index = unpack(table)
        if result & new_slot_bit:
            assert index[slot] == i  # first occurrence: stored i_item
        else:
            assert index[slot] < i   # repeat: earlier i_item still recorded

    # by-position mode must NOT have written key_storage
    assert np.array_equal(key_storage, snapshot)

    # Verify identity via find_key (returns the CURRENT slot, robust to
    # Robin-Hood displacement). The insertion-time slots[] array records
    # the slot at insert time, which may move under later displacement —
    # so re-query for every assertion that compares slots across rows.
    info, _, _ = unpack(table)
    assert info[HM_INFO_N_VALID] == 4  # 4 unique rows out of 6 inputs

    # Duplicate rows must resolve to the same slot under find_key.
    assert find_key(table, key_storage, key_storage[0]) == \
        find_key(table, key_storage, key_storage[2])
    assert find_key(table, key_storage, key_storage[1]) == \
        find_key(table, key_storage, key_storage[4])

    # (3, 30, +0.0) and (3, 30, -0.0) bit-differ ⇒ they're distinct keys.
    slot_pos = find_key(table, key_storage, key_storage[3])
    slot_neg = find_key(table, key_storage, key_storage[5])
    assert slot_pos != NOT_FOUND and slot_neg != NOT_FOUND
    assert slot_pos != slot_neg


def _find_colliding_int32_keys(n_keys, mask):
    """Brute-force int32 keys whose hashes share both ``hash & mask`` and
    ``hash >> hash_key_shift`` — i.e. they produce identical lookup_vals
    and all probe from the same bucket. Used to deterministically force
    Robin Hood displacement to exhaust max_rh.

    Mirrors the production hash: FNV-1a per-field accumulation + Murmur3
    fmix64 finalize. With fmix64 the avalanche is strong, so collisions
    are very rare — the candidate space needs to be tens of millions to
    find ~17 colliding keys. Runs in ~5-15 s; that's why
    ``test_collision_triggers_rehash`` uses a pre-mined hardcoded set by
    default and only re-mines on ``--remine-hash-collisions``."""
    cap = 40_000_000
    chunk = 5_000_000
    bits_for_high = 64 - hash_key_shift
    cls_to_keys = {}
    target_class = None
    for start in range(1, cap, chunk):
        cands = np.arange(start, min(start + chunk, cap + 1), dtype=np.int64)
        h = (np.uint64(init_hash) ^ cands.astype(np.uint64)) * np.uint64(FNV_PRIME)
        h ^= h >> np.uint64(33)
        h = h * np.uint64(M3_C1)
        h ^= h >> np.uint64(33)
        h = h * np.uint64(M3_C2)
        h ^= h >> np.uint64(33)
        bucket = (h & np.uint64(mask)).astype(np.int64)
        high_bits = (h >> np.uint64(hash_key_shift)).astype(np.int64)
        classes = (bucket << bits_for_high) | high_bits
        # Accumulate per-class keys until we hit n_keys for some class
        for c, k in zip(classes.tolist(), cands.tolist()):
            bucket_list = cls_to_keys.setdefault(c, [])
            bucket_list.append(k)
            if len(bucket_list) >= n_keys:
                target_class = c
                break
        if target_class is not None:
            break
    if target_class is None:
        max_seen = max(len(v) for v in cls_to_keys.values()) if cls_to_keys else 0
        raise RuntimeError(
            f"only {max_seen} colliding keys found in {cap}; raise cap")
    return np.array(cls_to_keys[target_class][:n_keys], dtype=np.int32)


# Pre-mined int32 keys that collide under the current hash (FNV per-field +
# Murmur3 fmix64) on mask=31: each one hashes to the same bucket AND has the
# same high hash bits, so a probe-chain of 17 inserts trips ``i_add_key_fail``
# inside ``init_dict(20)``'s 32-slot table.
#
# Hardcoded to keep the test fast — re-mining via _find_colliding_int32_keys()
# takes ~5-15 s. If the hash function (init_hash, FNV_PRIME, fmix64
# constants, or any of the overloads) ever changes, this list becomes stale
# and ``test_collision_triggers_rehash`` will fail with
# "no collision-induced fail; encoding may have changed".
#
# To regenerate:
#     pytest tests/pytools/common/test_hashmap.py::test_collision_triggers_rehash --remine-hash-collisions -s
# That run will brute-force a fresh set, print it, and use it for the test.
# Paste the printed list back here.
_COLLIDING_INT32_KEYS_MASK31 = np.array([
    31468, 46035, 83767, 106906, 121400, 123238, 127812, 142470, 145320,
    220523, 259791, 274517, 281610, 286860, 294983, 311790, 334897,
], dtype=np.int32)


def test_collision_triggers_rehash(request):
    """Force ``i_add_key_fail`` by inserting keys whose lookup_vals all collide.
    Caller must call rehash and retry. Verify every key is still findable."""
    # init_dict(20) → table size 32 → mask 31 → ~30 entries before load-factor rehash
    mask = 31
    if request.config.getoption('--remine-hash-collisions'):
        colliding = _find_colliding_int32_keys(17, mask)
        print(
            "\n[--remine-hash-collisions] re-mined colliding keys for mask=31:\n"
            "    np.array([" + ", ".join(str(int(k)) for k in colliding) + "], dtype=np.int32)\n"
            "Paste this into _COLLIDING_INT32_KEYS_MASK31 in tests/pytools/common/test_hashmap.py."
        )
    else:
        colliding = _COLLIDING_INT32_KEYS_MASK31

    key_storage = np.empty(64, dtype=np.int32)
    table = init_dict(20)

    failed_at = None
    for i, k in enumerate(colliding):
        result = try_add_key(table, key_storage, k)
        if result == i_add_key_fail:
            failed_at = i
            break
    assert failed_at is not None, "no collision-induced fail; encoding may have changed"

    # explicit rehash, then retry the failed key
    table = rehash(table, key_storage)
    result = try_add_key(table, key_storage, colliding[failed_at])
    assert result != i_add_key_fail

    # finish the remaining keys (may need more rehashes)
    for k in colliding[failed_at + 1:]:
        result = try_add_key(table, key_storage, k)
        while result == i_add_key_fail:
            table = rehash(table, key_storage)
            result = try_add_key(table, key_storage, k)

    for k in colliding:
        assert find_key(table, key_storage, k) != NOT_FOUND


def test_factorize_mixed_dtypes():
    """factorize(df) DataFrame wrapper: covers all three column-handling
    branches — nullable Int (coerced to float32), regular numeric, and
    non-numeric (coerced to fixed-width unicode 'U<n>'). Compares the
    resulting grouping against pd.factorize on the equivalent MultiIndex."""
    df = pd.DataFrame({
        'nullable_int': pd.array([1, 2, 1, 3, 2, 1], dtype='Int32'),
        'numeric': np.array([10.5, 20.0, 10.5, 30.0, 20.0, 10.5], dtype='float64'),
        'label': ['foo', 'bar', 'foo', 'baz', 'bar', 'foo'],
    })

    res = np.asarray(factorize(df))

    # pd.factorize on a MultiIndex returns 0-based codes; jit_factorize is 1-based
    expected, _ = pd.factorize(
        pd.MultiIndex.from_arrays([df['nullable_int'], df['numeric'], df['label']])
    )
    expected = expected + 1

    assert np.array_equal(res, expected)
    # sanity: three distinct row-tuples → max agg_id is 3
    assert res.max() == 3


def test_factorize_unicode_trailing_null_padding():
    """fnv1a's unichr arm trims trailing NUL padding via str(field), so the
    same logical string stored in different fixed-widths must hash and
    compare equal. Build two parallel arrays, U2 and U4, with the same
    content, factorize each, and verify groupings match."""
    rows_u2 = np.array([('ab',), ('cd',), ('ab',), ('cd',), ('ab',)],
                       dtype=[('s', 'U2')])
    rows_u4 = np.array([('ab',), ('cd',), ('ab',), ('cd',), ('ab',)],
                       dtype=[('s', 'U4')])

    res_u2 = np.asarray(jit_factorize(rows_u2))
    res_u4 = np.asarray(jit_factorize(rows_u4))

    expected = np.array([1, 2, 1, 2, 1], dtype=np.uint64)
    assert np.array_equal(res_u2, expected)
    assert np.array_equal(res_u4, expected)
