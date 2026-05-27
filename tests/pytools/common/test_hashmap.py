"""Unit tests for oasislmf.pytools.common.hashmap.

Tests target the user-facing API documented in the module docstring:
``init_dict``, ``unpack``, ``try_add_key``, ``find_key``, ``rehash``,
``jit_factorize``.

Strategy: a few high-level tests, each chosen to maximize coverage:
  1. ``jit_factorize`` vs ``pd.factorize`` on a structured array — exercises
     the bulk of the internal Robin Hood machinery (init_dict, unpack,
     _try_add_key by-position, rehash via load factor, structured-record
     fnv1a + key_eq overloads, _move_key).
  2. ``try_add_key`` (by-value, ``i_item=None``) + ``find_key`` over scalar
     dtypes incl. float bit-cast — covers scalar overloads, NOT_FOUND path,
     ``new_slot_bit`` semantics, +0.0/-0.0 distinguishability.
  3. ``try_add_key`` (by-position, ``i_item=int``) with structured records —
     covers the other ``try_add_key`` mode and verifies key_storage is left
     untouched.
  4. Collision-triggered rehash — brute-forces a colliding key set so that
     ``try_add_key`` returns ``i_add_key_fail``, then calls ``rehash`` and
     retries. Covers the ``i_add_key_fail`` return and explicit rehash path.
"""
import numpy as np
import pandas as pd

from oasislmf.pytools.common.hashmap import (
    init_dict, unpack, try_add_key, find_key, rehash, jit_factorize,
    i_add_key_fail, new_slot_bit, slot_mask, NOT_FOUND,
    init_hash, FNV_PRIME, hash_key_shift,
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


def test_try_add_key_find_key_roundtrip_scalars():
    """Public try_add_key (by-value, i_item=None) + find_key on scalar dtypes.
    Covers scalar fnv1a / key_eq overloads, the NOT_FOUND path, and the
    new_slot_bit semantics on first-vs-repeat insert.

    The float64 case exercises bit-cast hashing AND bit-compare equality
    for scalar floats: +0.0 and -0.0 are bit-distinct and must therefore
    hash and compare as distinct keys."""
    cases = [
        (np.int32, np.array([5, 7, 100, 7, 5, 0], dtype=np.int32)),
        (np.uint64, np.array([1, 2**40, 1, 0, 2**40], dtype=np.uint64)),
        # +0.0 / -0.0 bit-distinct ⇒ distinct slots
        (np.float64, np.array([0.0, -0.0, 1.5, 0.0, np.inf, -0.0], dtype=np.float64)),
    ]
    for dtype, keys_in in cases:
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

        absent = dtype(99999) if dtype is not np.float64 else np.float64(3.14159)
        assert find_key(table, key_storage, absent) == NOT_FOUND


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

    # duplicate rows collapse to the same slot
    assert slots[0] == slots[2]
    assert slots[1] == slots[4]
    # (3, 30, +0.0) and (3, 30, -0.0) bit-differ ⇒ distinct slots
    assert slots[3] != slots[5]
    assert slots[3] != slots[0] and slots[3] != slots[1]


def _find_colliding_int32_keys(n_keys, mask):
    """Brute-force int32 keys whose fnv1a hashes share both
    ``hash & mask`` and ``hash >> hash_key_shift`` — i.e. they produce
    identical lookup_vals and all probe from the same bucket. Used to
    deterministically force Robin Hood displacement to exhaust max_rh."""
    cap = 4_000_000
    candidates = np.arange(1, cap, dtype=np.int64)
    hashes = ((np.uint64(init_hash) ^ candidates.astype(np.uint64))
              * np.uint64(FNV_PRIME))
    bucket = (hashes & np.uint64(mask)).astype(np.int64)
    high_bits = (hashes >> np.uint64(hash_key_shift)).astype(np.int64)
    bits_for_high = 64 - hash_key_shift
    classes = (bucket << bits_for_high) | high_bits
    counts = np.bincount(classes)
    target = int(np.argmax(counts))
    if counts[target] < n_keys:
        raise RuntimeError(
            f"only {counts[target]} colliding keys found in {cap}; raise cap")
    return candidates[classes == target][:n_keys].astype(np.int32)


def test_collision_triggers_rehash():
    """Force ``i_add_key_fail`` by inserting keys whose lookup_vals all collide.
    Caller must call rehash and retry. Verify every key is still findable."""
    # init_dict(20) → table size 32 → mask 31 → ~30 entries before load-factor rehash
    mask = 31
    colliding = _find_colliding_int32_keys(17, mask)

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
