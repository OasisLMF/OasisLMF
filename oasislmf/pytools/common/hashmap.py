"""
Robin Hood hash table for numba JIT code.

Supports both structured-record keys (e.g. np.dtype([('a','i4'),('b','u1')]))
and scalar numeric keys (int32, uint64, float64, ...). Hash and equality are
synthesized per dtype at compile time via @numba.extending.overload, so each
call site does typed field accesses only — no byte view, no buffer allocation.

All hashmap state is stored in a single packed uint8 buffer ("table") holding
[info | lookup_table | index_table]. Use unpack(table) to get views.

Public API (convenience — unpacks per call):
    init_dict(hint_size)                                -> table
    try_add_key(table, key_storage, key, i_item=None)   -> slot | new_slot_bit, or i_add_key_fail
    find_key(table, key_storage, key)                   -> slot, or NOT_FOUND
    rehash(table, key_storage)                          -> new_table
    unpack(table)                                       -> (info, lookup, index) views

Internal API (hot-path — caller unpacks once):
    _try_add_key(info, lookup, index, key_storage, key, i_item=None)
    _find_key(info, lookup, index, key_storage, key)

Performance (1M ops, numba 0.64, int32 keys):
    Public  try_add_key  : ~2.3 M ops/s   (unpack overhead per call)
    Public  find_key     : ~5.3 M ops/s
    Internal _try_add_key: ~4.6 M ops/s   (2x faster)
    Internal _find_key   : ~21  M ops/s   (4x faster)

Rule of thumb: use the public API for one-off or low-volume calls. For tight
loops (inserting/looking up thousands of keys), unpack once and call the
internal _impl functions directly.

try_add_key has two modes selected via `i_item` (compile-time literal branch —
both modes specialize with no runtime overhead):

    i_item is None (by-value mode):
        On insertion, the hashmap writes `key` to key_storage[info[HM_INFO_N_VALID]] and
        stores info[HM_INFO_N_VALID] in the index. The caller doesn't track positions;
        key_storage[:info[HM_INFO_N_VALID]] ends up holding the unique keys in insertion
        order.

    i_item is an int (by-position mode):
        On insertion, stores `i_item` in the index. The hashmap does NOT
        write to key_storage — the caller is asserting that
        key_storage[i_item] already holds `key`.

Sizing and rehash:

    init_dict(hint_size) pre-allocates the table so that `hint_size` unique
    keys fit within the load factor (n_full_factor = 0.95). When the caller
    knows an upper bound on the number of unique keys and passes it as
    hint_size, the load-factor guard (info[HM_INFO_N_VALID] >= info[HM_INFO_N_FULL])
    is guaranteed never to fire and can be omitted from the insert loop.

    The `while result == i_add_key_fail` rehash loop must still be kept even
    with a correctly pre-sized table. i_add_key_fail signals that the Robin
    Hood displacement chain exceeded max_rh due to hash collisions — this is
    independent of the load factor and can (rarely) occur at any occupancy.

    When hint_size is not known up front (e.g. streaming inserts), the caller
    must check the load-factor guard before each insertion:

        if info[HM_INFO_N_VALID] >= info[HM_INFO_N_FULL]:
            table = rehash(table, key_storage)
            info, lookup, index = unpack(table)

Example — by-value insert, size known (pre-sized table, no load-factor check)::

    from oasislmf.pytools.common.hashmap import (
        init_dict, unpack, rehash, _try_add_key,
        i_add_key_fail, new_slot_bit, slot_mask,
    )

    @nb.jit(cache=True)
    def build_vuln_dict(vuln_ids):
        key_storage = np.empty(len(vuln_ids), dtype=np.int32)
        table = init_dict(len(vuln_ids))  # pre-sized: no load-factor rehash needed
        info, lookup, index = unpack(table)

        for i in range(len(vuln_ids)):
            result = _try_add_key(info, lookup, index, key_storage, vuln_ids[i])
            while result == i_add_key_fail:  # Robin Hood collision — still possible
                table = rehash(table, key_storage)
                info, lookup, index = unpack(table)
                result = _try_add_key(info, lookup, index, key_storage, vuln_ids[i])

            dense_idx = index[result & slot_mask]  # works for both new and existing

        return table, key_storage[:info[HM_INFO_N_VALID]]

Example — by-value insert, size unknown (must check load factor)::

    from oasislmf.pytools.common.hashmap import (
        init_dict, unpack, rehash, _try_add_key,
        i_add_key_fail, new_slot_bit, slot_mask,
        HM_INFO_N_VALID, HM_INFO_N_FULL,
    )

    @nb.jit(cache=True)
    def build_vuln_dict(vuln_ids):
        key_storage = np.empty(len(vuln_ids), dtype=np.int32)
        table = init_dict()
        info, lookup, index = unpack(table)

        for i in range(len(vuln_ids)):
            if info[HM_INFO_N_VALID] >= info[HM_INFO_N_FULL]:
                table = rehash(table, key_storage)
                info, lookup, index = unpack(table)

            result = _try_add_key(info, lookup, index, key_storage, vuln_ids[i])
            while result == i_add_key_fail:
                table = rehash(table, key_storage)
                info, lookup, index = unpack(table)
                result = _try_add_key(info, lookup, index, key_storage, vuln_ids[i])

            dense_idx = index[result & slot_mask]  # works for both new and existing

        return table, key_storage[:info[HM_INFO_N_VALID]]

Example — lookup with internal API::

    from oasislmf.pytools.common.hashmap import (
        unpack, _find_key, NOT_FOUND,
    )

    @nb.jit(cache=True)
    def lookup_vuln_ids(table, key_storage, query_ids, out):
        info, lookup, index = unpack(table)
        for i in range(len(query_ids)):
            slot = _find_key(info, lookup, index, key_storage, query_ids[i])
            if slot != NOT_FOUND:
                out[i] = index[slot]   # dense index
            else:
                out[i] = -1

Notes:
    - The hash function (fnv1a) is deterministic with no random seed. Both
      `table` and `key_storage` can be saved/loaded from binary files and
      used directly with find_key — no rebuild needed.
    - Float fields are hashed via bit-cast (IEEE 754 bytes reinterpreted as
      uint64), so distinct float bit patterns always produce distinct hashes.
"""
import numpy as np
import numba as nb
from numba.extending import overload
from numba.core import types as nbt
from pandas.api.types import is_numeric_dtype


# initial val for fnv1a hashing function
init_hash = np.uint64(14695981039346656037)
FNV_PRIME = np.uint64(1099511628211)

# lookup bit definition
# 1 bit - 0 empty / 1 full | 4 bit - robinhood index 0 to 15 | left over n bits - first left n bit of hash

lookup_dtype = np.uint16
nb_lookup_dtype = nb.from_dtype(lookup_dtype)
lookup_bitsize = nb_lookup_dtype(0).itemsize * 8
lookup_hash_bitsize = lookup_bitsize - 5
hash_key_shift = 64 - lookup_hash_bitsize

full_bit = lookup_dtype(1 << (lookup_bitsize - 1))  # for 8 bit          0b10000000
hash_mask = lookup_dtype((2**lookup_bitsize - 1) >> 5)  # for 8 bit      0b00000111
max_rh = lookup_dtype(1 << (lookup_bitsize - 1))  # for 8 bit            0b10000000
i_rh_mask = lookup_dtype(0b1111 << lookup_hash_bitsize)  # for 8 bit   0b01111000
i_rh_increment = lookup_dtype(0b1 << lookup_hash_bitsize)  # for 8 bit  0b00001000
full_rh = lookup_dtype(0b11111 << lookup_hash_bitsize)  # lookval is full and i_rh = 15
n_full_factor = 0.95
inverse_n_full_factor = 1 / n_full_factor

index_dtype = np.uint32
nb_index_dtype = nb.from_dtype(index_dtype)

# Named offsets into the 3-element info array returned by unpack().
HM_INFO_MASK = 0      # bitmask for slot indexing (table_size - 1)
HM_INFO_N_VALID = 1   # number of unique keys currently stored
HM_INFO_N_FULL = 2    # max occupancy before rehash (table_size * 0.95)

i_add_key_fail = index_dtype(np.iinfo(index_dtype).max)

# try_add_key returns (slot index) | (new_slot_bit if the key was just inserted).
# Table sizes are bounded well below half the index_dtype range, so the top bit
# is always free to carry the flag. `i_add_key_fail` is all-ones and is checked
# *before* the new-bit test, so there is no ambiguity with a new-slot encoding.
index_bitsize = np.dtype(index_dtype).itemsize * 8
new_slot_bit = index_dtype(1 << (index_bitsize - 1))
slot_mask = index_dtype(~new_slot_bit)

# find_key returns NOT_FOUND when the key is not in the table.
NOT_FOUND = index_dtype(np.iinfo(index_dtype).max)

# ---------------------------------------------------------------------------
# Packed table layout: a single uint8 buffer holding [info | lookup | index].
#
#   info:   3 × index_dtype values (HM_INFO_MASK, HM_INFO_N_VALID, HM_INFO_N_FULL)
#   lookup: table_size × lookup_dtype values
#   index:  table_size × index_dtype values
#
# table_size is derived from mask (= info[HM_INFO_MASK]) + 1. All three sections are
# accessed via views into the buffer, so mutations go through to the same
# backing memory.
# ---------------------------------------------------------------------------
LOOKUP_ITEMSIZE = nb.int64(np.dtype(lookup_dtype).itemsize)
INDEX_ITEMSIZE = nb.int64(np.dtype(index_dtype).itemsize)
INFO_N = nb.int64(3)
INFO_BYTES = INFO_N * INDEX_ITEMSIZE


# ---------------------------------------------------------------------------
# Bit helpers (unchanged from hashmap.py)
# ---------------------------------------------------------------------------

@nb.jit(nb.uint8(nb.uint8), cache=True)
def extract_i_rh(lookup_val):
    return lookup_val & i_rh_mask


# @nb.jit(nb.uint8(nb.uint64), cache=True)
# def extract_hash_bit(hash_key):
#     return hash_key >> np.uint64(61)
#
#
# @nb.jit(nb.uint8(nb.uint8, nb.uint8, nb.uint8), cache=True)
# def make_lookup_val(is_full, i_rh, hash_lookup_bit):
#     return is_full | i_rh | hash_lookup_bit

@nb.jit(nb_lookup_dtype(nb.uint64), cache=True)
def extract_hash_bit(hash_key):
    return hash_key >> np.uint64(hash_key_shift)


@nb.jit(nb_lookup_dtype(nb_lookup_dtype, nb_lookup_dtype, nb_lookup_dtype), cache=True)
def make_lookup_val(is_full, i_rh, hash_lookup_bit):
    return is_full | i_rh | hash_lookup_bit


# ---------------------------------------------------------------------------
# Per-dtype specialized hash and equality.
# Both stubs raise NotImplementedError; @overload replaces them at compile
# time with a body that does typed field accesses for the actual dtype.
# ---------------------------------------------------------------------------

def fnv1a(record, h=init_hash):
    """FNV-1a hash of a key. Under JIT, @overload synthesizes a typed impl.
    This Python fallback handles both structured records and scalars."""
    if hasattr(record, 'dtype') and record.dtype.names:
        for fname in record.dtype.names:
            h = (h ^ np.uint64(record[fname])) * FNV_PRIME
        return h
    return (h ^ np.uint64(record)) * FNV_PRIME


@overload(fnv1a)
def fnv1a_overload_record(record, h=init_hash):
    if not isinstance(record, nbt.Record):
        return None
    field_names = list(record.fields)

    src = ["def impl(record, h=init_hash):"]
    for fname in field_names:
        ftype = record.fields[fname][0]
        if isinstance(ftype, nbt.Float):
            # bit-cast: reinterpret IEEE 754 bytes as uint64 via a 1-element buffer.
            # The buffer is stack-allocated by LLVM (no heap alloc) since its
            # size is known at compile time and it doesn't escape the function.
            if ftype.bitwidth == 64:
                src.append("    _buf = np.empty(1, dtype=np.float64)")
                src.append(f"    _buf[0] = record['{fname}']")
                src.append(f"    h = (h ^ _buf.view(np.uint64)[0]) * {FNV_PRIME}")
            else:  # float32 → zero-extend to 64 bits after bit-cast to uint32
                src.append("    _buf = np.empty(1, dtype=np.float32)")
                src.append(f"    _buf[0] = record['{fname}']")
                src.append(f"    h = (h ^ np.uint64(_buf.view(np.uint32)[0])) * {FNV_PRIME}")
        else:
            # int/uint/bool: np.uint64() is already a zero-extension, equivalent to bitcast
            src.append(
                f"    h = (h ^ np.uint64(record['{fname}'])) * {FNV_PRIME}"
            )
    src.append("    return h")

    ns = {'np': np, 'init_hash': init_hash}
    exec("\n".join(src), ns)
    return ns['impl']


@overload(fnv1a)
def fnv1a_overload_scalar(key, h=init_hash):
    """Handles any numeric scalar: int8..int64, uint8..uint64, float32, float64, bool."""
    if not isinstance(key, nbt.Number):
        return None

    def impl(key, h=init_hash):
        return (h ^ np.uint64(key)) * np.uint64(1099511628211)
    return impl


def key_eq(a, b):
    """Equality of two keys. Under JIT, @overload synthesizes a typed impl.
    This Python fallback handles both structured records and scalars."""
    if hasattr(a, 'dtype') and a.dtype.names:
        return all(a[fname] == b[fname] for fname in a.dtype.names)
    return a == b


@overload(key_eq)
def key_eq_overload_record(a, b):
    if not (isinstance(a, nbt.Record) and isinstance(b, nbt.Record)):
        return None
    field_names = list(a.fields)

    src = ["def impl(a, b):"]
    compare = ' and '.join(f"a['{fname}'] == b['{fname}']" for fname in field_names)
    src.append(f"    return {compare}")

    ns = {}
    exec("\n".join(src), ns)
    return ns['impl']


@overload(key_eq)
def key_eq_overload_scalar(a, b):
    """Handles any numeric scalar."""
    if not (isinstance(a, nbt.Number) and isinstance(b, nbt.Number)):
        return None

    def impl(a, b):
        return a == b
    return impl


# ---------------------------------------------------------------------------
# Packed table: unpack and creation
# ---------------------------------------------------------------------------

@nb.jit(cache=True, inline='always')
def unpack(table):
    """Extract (info, lookup_table, index_table) views from a packed buffer.
    info:   uint8 view → .view(index_dtype) → 3-element array [HM_INFO_MASK, HM_INFO_N_VALID, HM_INFO_N_FULL]
    lookup: uint8 view → .view(lookup_dtype) → table_size elements
    index:  uint8 view → .view(index_dtype)  → table_size elements
    """
    info = table[:INFO_BYTES].view(index_dtype)
    table_size = nb.int64(info[HM_INFO_MASK]) + nb.int64(1)
    lookup_end = INFO_BYTES + table_size * LOOKUP_ITEMSIZE
    lookup = table[INFO_BYTES:lookup_end].view(lookup_dtype)
    index_end = lookup_end + table_size * INDEX_ITEMSIZE
    index = table[lookup_end:index_end].view(index_dtype)
    return info, lookup, index


@nb.jit(cache=True)
def init_dict(hint_size=15):
    """Create a packed table buffer. Returns a single uint8 array."""
    init_size = nb.int64(16)
    mask = index_dtype(0b1111)
    while init_size < (hint_size * inverse_n_full_factor):
        mask = index_dtype((mask << index_dtype(1)) + index_dtype(1))
        init_size *= 2
    total_bytes = INFO_BYTES + init_size * LOOKUP_ITEMSIZE + init_size * INDEX_ITEMSIZE
    table = np.zeros(total_bytes, dtype=np.uint8)
    info = table[:INFO_BYTES].view(index_dtype)
    info[HM_INFO_MASK] = mask
    info[HM_INFO_N_VALID] = index_dtype(0)
    info[HM_INFO_N_FULL] = index_dtype(init_size * n_full_factor)
    return table


# ---------------------------------------------------------------------------
# Robin Hood hashmap operations
# ---------------------------------------------------------------------------

@nb.jit(cache=True)
def _move_key(lookup_table, index_table, mask, i_lookup):
    """Find the next empty slot to the right of i_lookup and shift entries up
    by one slot to make room. Returns False if a key would become 'too poor'
    to fit (signals a rehash is needed)."""
    i_lookup_start = i_lookup
    while full_bit <= lookup_table[i_lookup & mask] < full_rh:
        i_lookup += index_dtype(1)
    if lookup_table[i_lookup & mask] >= full_rh:
        return False

    while i_lookup > i_lookup_start:
        lookup_table[i_lookup & mask] = (
            (lookup_table[(i_lookup - index_dtype(1)) & mask]) + i_rh_increment
        )
        index_table[i_lookup & mask] = (
            index_table[(i_lookup - index_dtype(1)) & mask]
        )
        i_lookup -= index_dtype(1)
    return True


@nb.jit(cache=True)
def _try_add_key(info, lookup_table, index_table, key_storage, key, i_item=None):
    """Internal: insert-or-find on unpacked views. Hot-path callers use this
    directly to avoid per-call unpack overhead.

    Two modes selected via `i_item`:

    i_item is None (by-value mode):
        On insertion, writes `key` to key_storage[info[HM_INFO_N_VALID]] and stores info[HM_INFO_N_VALID]
        in the index. The caller doesn't track positions; `key_storage[:info[HM_INFO_N_VALID]]`
        ends up with the unique keys in insertion order.

    i_item is an int (by-position mode):
        On insertion, stores `i_item` in the index. Skips the storage write —
        the caller is asserting that key_storage[i_item] already holds `key`
        (typically because key_storage IS the input array and
        key = key_storage[i_item]).

    The `i_item is None` check is a numba compile-time literal branch — both
    modes compile to specialized code with no runtime overhead.

    Args:
        info, lookup_table, index_table: unpacked views.
        key_storage: 1D array. Read for equality checks against stored keys;
                     also written to in by-value mode.
        key: the key value to insert or find.
        i_item: None (by-value) or int (by-position).

    Returns:
        i_add_key_fail                         — rehash needed
        slot                                   — key already present at `slot`
        slot | new_slot_bit                    — key was just inserted at `slot`
    """
    mask = info[HM_INFO_MASK]
    hash_key = np.uint64(fnv1a(key))

    i_lookup = index_dtype(hash_key & mask)
    hash_lookup_bit = extract_hash_bit(hash_key)
    i_rh = nb_lookup_dtype(0)

    i_lookup_res = i_add_key_fail
    while i_rh < max_rh:
        masked_i_lookup = i_lookup & mask
        lookup_val = make_lookup_val(full_bit, i_rh, hash_lookup_bit)
        if lookup_val < lookup_table[masked_i_lookup]:  # poorer, keep probing
            i_rh += i_rh_increment
            i_lookup += index_dtype(1)
        elif lookup_val == lookup_table[masked_i_lookup]:
            if key_eq(key, key_storage[index_table[masked_i_lookup]]):
                i_lookup_res = masked_i_lookup  # found existing
                break
            # hash bits collide but keys differ — keep probing
            i_rh += i_rh_increment
            i_lookup += index_dtype(1)
        else:  # we're richer — displace and insert here
            moved = _move_key(lookup_table, index_table, mask, masked_i_lookup)
            if moved:
                if i_item is None:
                    # by-value: write key to storage, use info[HM_INFO_N_VALID] as index
                    key_storage[info[HM_INFO_N_VALID]] = key
                    index_table[masked_i_lookup] = info[HM_INFO_N_VALID]
                else:
                    # by-position: caller already has key at key_storage[i_item]
                    index_table[masked_i_lookup] = i_item
                lookup_table[masked_i_lookup] = lookup_val
                info[HM_INFO_N_VALID] += index_dtype(1)
                i_lookup_res = masked_i_lookup | new_slot_bit  # just inserted
            break

    return i_lookup_res


@nb.jit(cache=True)
def try_add_key(table, key_storage, key, i_item=None):
    """Try to insert `key`. Caller must check for i_add_key_fail first, then
    mask off `new_slot_bit` to recover the slot index. See `_try_add_key` for
    the `i_item` mode semantics.

    Returns:
        i_add_key_fail                         — rehash needed
        slot                                   — key already present at `slot`
        slot | new_slot_bit                    — key was just inserted at `slot`
    """
    info, lookup_table, index_table = unpack(table)
    return _try_add_key(info, lookup_table, index_table, key_storage, key, i_item)


@nb.jit(cache=True)
def _find_key(info, lookup_table, index_table, key_table, key):
    """Internal: lookup on unpacked views."""
    mask = info[HM_INFO_MASK]
    hash_key = np.uint64(fnv1a(key))
    hash_lookup_bit = extract_hash_bit(hash_key)
    i_lookup = index_dtype(hash_key & mask)
    i_rh = nb_lookup_dtype(0)

    while i_rh < max_rh:
        masked_i_lookup = i_lookup & mask
        lookup_val = make_lookup_val(full_bit, i_rh, hash_lookup_bit)
        if lookup_val < lookup_table[masked_i_lookup]:  # poorer, keep probing
            i_rh += i_rh_increment
            i_lookup += index_dtype(1)
        elif lookup_val == lookup_table[masked_i_lookup]:
            if key_eq(key, key_table[index_table[masked_i_lookup]]):
                return masked_i_lookup
            # hash bits collide but keys differ — keep probing
            i_rh += i_rh_increment
            i_lookup += index_dtype(1)
        else:  # we're richer — key can't exist past this point in Robin Hood
            return NOT_FOUND
    return NOT_FOUND


@nb.jit(cache=True)
def find_key(table, key_table, key):
    """Look up `key` in the table without inserting. Returns the slot index
    on hit, or NOT_FOUND on miss.

    Note: takes a key *value* (not an index into key_table), since the
    caller typically has a raw key in hand rather than a position in
    key_table.
    """
    info, lookup_table, index_table = unpack(table)
    return _find_key(info, lookup_table, index_table, key_table, key)


@nb.jit(cache=True)
def rehash(table, key_table):
    """Double the table size and re-insert every live key. Returns a new
    packed table buffer (the old one becomes stale)."""
    info, lookup_table, index_table = unpack(table)

    while info[HM_INFO_N_VALID] > (info[HM_INFO_N_FULL] >> np.uint8(3)):
        new_table_size = nb.int64(lookup_table.shape[0]) * 2
        new_table = np.zeros(
            INFO_BYTES + new_table_size * LOOKUP_ITEMSIZE + new_table_size * INDEX_ITEMSIZE,
            dtype=np.uint8
        )
        # Write info BEFORE unpack — unpack reads mask to compute view sizes
        new_mask = (info[HM_INFO_MASK] << index_dtype(1)) + index_dtype(1)
        new_table[:INFO_BYTES].view(index_dtype)[HM_INFO_MASK] = new_mask
        new_table[:INFO_BYTES].view(index_dtype)[HM_INFO_N_VALID] = index_dtype(0)
        new_table[:INFO_BYTES].view(index_dtype)[HM_INFO_N_FULL] = index_dtype(new_table_size * n_full_factor)
        new_info, new_lookup, new_index = unpack(new_table)

        for i_lookup in range(lookup_table.shape[0]):
            masked_i_lookup = index_dtype(i_lookup) & info[HM_INFO_MASK]
            if lookup_table[masked_i_lookup] >= full_bit:
                i_item = index_table[masked_i_lookup]
                added = _try_add_key(new_info, new_lookup, new_index, key_table, key_table[i_item], i_item)
                if added == i_add_key_fail:
                    break
        else:
            break
        # retry with doubled table — update refs for next iteration
        info = new_info
        lookup_table = new_lookup
        index_table = new_index
        table = new_table
    else:
        raise Exception("rehashed too many times")

    return new_table


@nb.jit(cache=True)
def jit_factorize(key_table):
    """Return a 1-based id per row of `key_table`; identical rows share an id.

    Uses _try_add_key in by-position mode (i_item passed): no extra storage
    allocated, no input modification. index[slot] points to the i_item of
    the first occurrence; agg_id is tracked via info[HM_INFO_N_VALID] (which increments
    to match the insertion order)."""
    table = init_dict()
    info, lookup_table, index_table = unpack(table)
    res = np.empty(key_table.shape[0], dtype=np.uint64)

    for i_item_int in range(key_table.shape[0]):
        i_item = index_dtype(i_item_int)
        # cold path: rehash if full
        if info[HM_INFO_N_VALID] >= info[HM_INFO_N_FULL]:
            table = rehash(table, key_table)
            info, lookup_table, index_table = unpack(table)

        result = _try_add_key(
            info, lookup_table, index_table, key_table, key_table[i_item], i_item
        )

        # cold path: insertion failed (max_rh exceeded) — rehash and retry
        while result == i_add_key_fail:
            if info[HM_INFO_N_VALID] < (info[HM_INFO_N_FULL] >> np.uint8(3)):
                raise Exception("rehashed too many times")
            table = rehash(table, key_table)
            info, lookup_table, index_table = unpack(table)
            result = _try_add_key(
                info, lookup_table, index_table, key_table, key_table[i_item], i_item
            )

        # by-position: index[slot] is i_item of first occurrence.
        # For new keys, that's the current i_item; for existing keys, an earlier i_item.
        # Either way, info[HM_INFO_N_VALID] (post-increment) is the next agg_id, and res of
        # the first occurrence holds the assigned agg_id.
        if result & new_slot_bit:
            # new: just inserted, info[HM_INFO_N_VALID] post-increment == this key's agg_id
            res[i_item] = np.uint64(info[HM_INFO_N_VALID])
        else:
            # existing: copy agg_id from the first occurrence
            res[i_item] = res[index_table[result]]

    return res


def factorize(df):
    """pd.factorize-equivalent driver. Same surface as hashmap.factorize."""
    np_arrays = {}
    np_dtypes = []
    for _name, _dtype in df.dtypes.items():
        if _dtype.name.startswith('Int'):
            np_dtypes.append((_name, 'f'))  # convert int-with-nan to float
            np_arrays[_name] = df[_name].to_numpy(dtype='f')
        elif is_numeric_dtype(_dtype):
            np_dtypes.append((_name, _dtype.name))
            np_arrays[_name] = df[_name].to_numpy(dtype=_dtype.name)
        else:
            _serie = df[_name].astype(str)
            max_str_len = int(np.max(_serie.str.len()))
            np_dtypes.append((_name, f'U{max_str_len}'))
            np_arrays[_name] = _serie.to_numpy(dtype=f'U{max_str_len}')
    arr = np.empty(df.shape[0], dtype=np_dtypes)
    for _name, _dtype in np_dtypes:
        arr[_name] = np_arrays[_name]

    return jit_factorize(arr)


if __name__ == "__main__":
    # Sanity test mirroring hashmap.py's __main__: factorize by all fields and
    # compare against a pure-Python reference.
    _dtype = np.dtype([('a', 'int32'), ('b', 'uint8')])
    np.random.seed(seed=1)
    arr = np.empty(1000, dtype=_dtype)
    arr['a'] = np.random.randint(1, 1000, size=1000)
    arr['b'] = np.random.randint(0, 256, size=1000)

    ref = []
    _dict = {}
    agg_id = 0
    for val in arr:
        key = (int(val['a']), int(val['b']))
        if key not in _dict:
            agg_id += 1
            _dict[key] = agg_id
        ref.append(_dict[key])

    res = list(jit_factorize(arr))
    for i in range(len(res)):
        if res[i] != ref[i]:
            print("error for ", arr[i], i, ref[i], res[i])
            break
    else:
        print(f"ok! {len(set(ref))} unique keys / {len(ref)} rows")
