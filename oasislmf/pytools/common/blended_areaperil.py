"""Blended areaperil table: model data saying where in a location's extent a building may stand.

A blended areaperil is an ``AreaPerilID`` the key server returns for a location it cannot pin to
one hazard cell. It names a set of candidate positions with weights (its *placement scheme*) and,
for each position, the real cell that covers it for this peril. Every blended id of one location
names the same scheme, so a position chosen once per building and sample is read consistently by
all its perils and coverage types.

The table is model data, ``blended_areaperil_id.bin`` or ``.csv`` beside the footprint, with one
row per (blended id, position)::

    blended_areaperil_id, placement_scheme_id, placement_id, [weight], areaperil_id

Weight types
------------
A model writes weights in whichever form it holds, and all three select through the same rule:
the uniform is scaled by ``max(1, total weight)`` and searched in the running weights
(``placement_scale`` and ``select_placement``):

``WEIGHT_FLOAT``
    probabilities. They may sum to less than 1, and the shortfall is the chance the building is
    outside every modelled position. They total at most 1, so the scale is **1.0**, never the sum.
``WEIGHT_COUNT``
    non-negative integer counts (address points, pixels, buildings). Scaled by their total, so
    there is no shortfall; "not at risk" is a row with ``areaperil_id = 0``.
``WEIGHT_EQUAL``
    no weight column. Every position has weight 1, scaled by the number of positions.

Damage bin 0
------------
A sample whose placement has no hazard behind it -- an areaperil with no footprint for the event,
``areaperil_id = 0``, a float weight shortfall, or (dynamic footprint) an areaperil where the event
is within the item's standard of protection -- loses nothing. Where that has to be expressed as a
damage distribution (the analytic mean and std in gulmc, the cdf getmodel streams to gulpy), the
mass goes to **damage bin 0, which is assumed to be the zero-loss bin** of the damage bin
dictionary. See ``mix_effective_damage_cdfs``.

The trap is the float scale. One routine that divides every mode by its total looks tidier, and
silently turns "10% of this postcode is outside the model" into "all of it is inside".

Binary layout
-------------
A single int32 header, ``record_type``, then records of that type's layout (``record_dtype``).
The record types are the weight types: ``WEIGHT_EQUAL`` (0), ``WEIGHT_COUNT`` (1) and
``WEIGHT_FLOAT`` (2).

The record type is in the header rather than inferred from the file size: a record count can be
divisible by more than one record size, which is how a stale ``correlations.bin`` once parsed
cleanly into the wrong fields. It is also how the layout evolves, in place of a version number: a
changed record layout gets a new code, which an older oasislmf rejects as unknown rather than
misreading.

As in the other model files, the areaperil id fields follow ``OASIS_AREAPERIL_TYPE``, and the file
does not record which width it was written with.

In the CSV form the weight type is read from the file: no ``weight`` column is equal weights, an
integer column is counts and a decimal column is probabilities. Unknown columns are rejected,
because a misspelt ``weight`` header would otherwise read as equal weights without complaint.
"""
import logging
from typing import NamedTuple

import numba as nb
import numpy as np
import pandas as pd

from oasislmf.pytools.common.data import (
    areaperil_int, oasis_int,
    blended_areaperil_count_dtype, blended_areaperil_count_fmt, blended_areaperil_count_headers,
    blended_areaperil_equal_dtype, blended_areaperil_equal_fmt, blended_areaperil_equal_headers,
    blended_areaperil_float_dtype, blended_areaperil_float_fmt, blended_areaperil_float_headers,
)
from oasislmf.utils.exceptions import OasisException

logger = logging.getLogger(__name__)

BLENDED_AREAPERIL_FILE = 'blended_areaperil_id'

WEIGHT_EQUAL = 0
WEIGHT_COUNT = 1
WEIGHT_FLOAT = 2
WEIGHT_TYPE_NAMES = {WEIGHT_EQUAL: 'equal', WEIGHT_COUNT: 'count', WEIGHT_FLOAT: 'float'}

blend_header_dtype = np.dtype([('record_type', np.int32)])

# Float weights may sum to slightly over 1 from decimal rounding in the file.
FLOAT_SUM_TOLERANCE = 1e-6

_RECORD = {
    WEIGHT_EQUAL: (blended_areaperil_equal_dtype, blended_areaperil_equal_headers, blended_areaperil_equal_fmt),
    WEIGHT_COUNT: (blended_areaperil_count_dtype, blended_areaperil_count_headers, blended_areaperil_count_fmt),
    WEIGHT_FLOAT: (blended_areaperil_float_dtype, blended_areaperil_float_headers, blended_areaperil_float_fmt),
}

# In memory every weight type is held in one layout, with equal weights stored as 1.0. Counts
# fit f8 exactly up to 2**53, so the count type keeps its exactness.
blend_table_dtype = blended_areaperil_float_dtype


class BlendTable(NamedTuple):
    """A loaded blended areaperil table.

    ``table`` is in ``blend_table_dtype``, sorted by (blended_areaperil_id, placement_id), so each
    blended id's positions are one contiguous run in placement order.
    """
    table: np.ndarray
    weight_type: int


def record_dtype(weight_type):
    """On-disk record layout, CSV headers and CSV row format for a weight type."""
    if weight_type not in _RECORD:
        raise OasisException(
            f"Unknown blended areaperil record type {weight_type}; this oasislmf reads {sorted(WEIGHT_TYPE_NAMES)}. "
            f"The file may have been written by a newer oasislmf.")
    return _RECORD[weight_type]


def to_blend_table(records, weight_type):
    """Widen on-disk records of any weight type to ``blend_table_dtype``, in file order."""
    table = np.empty(len(records), dtype=blend_table_dtype)
    for name in blend_table_dtype.names:
        if name == 'weight':
            table[name] = 1.0 if weight_type == WEIGHT_EQUAL else records[name]
        else:
            table[name] = records[name]
    return table


def to_records(table, weight_type):
    """Narrow a ``blend_table_dtype`` array to the on-disk layout of ``weight_type``."""
    dtype, _, _ = record_dtype(weight_type)
    records = np.empty(len(table), dtype=dtype)
    for name in dtype.names:
        records[name] = table[name]
    return records


# ---------------------------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------------------------

def write_bin(file_out, table, weight_type):
    """Write the header and records to an open binary file."""
    header = np.zeros(1, dtype=blend_header_dtype)
    header['record_type'] = weight_type
    file_out.write(header.tobytes())
    file_out.write(to_records(table, weight_type).tobytes())


def read_bin(file_in, label='blended areaperil file'):
    """Read a binary blended areaperil table from an open binary file.

    Returns:
        BlendTable: rows in file order (not sorted; ``validate`` sorts).
    """
    raw = file_in.read()
    if len(raw) < blend_header_dtype.itemsize:
        raise OasisException(f"{label} is too short to hold its {blend_header_dtype.itemsize}-byte header")
    header = np.frombuffer(raw, dtype=blend_header_dtype, count=1)[0]
    weight_type = int(header['record_type'])
    dtype, _, _ = record_dtype(weight_type)
    body = len(raw) - blend_header_dtype.itemsize
    if body % dtype.itemsize:
        raise OasisException(
            f"{label} holds {body} bytes of records, not a whole number of {dtype.itemsize}-byte "
            f"{WEIGHT_TYPE_NAMES[weight_type]}-weight records; the file is truncated or corrupt")
    records = np.frombuffer(raw, dtype=dtype, offset=blend_header_dtype.itemsize)
    return BlendTable(to_blend_table(records, weight_type), weight_type)


# ---------------------------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------------------------

def read_csv(file_in, label='blended areaperil csv'):
    """Read a CSV blended areaperil table, taking the weight type from its columns.

    Args:
        file_in: path or open text file, as accepted by ``pandas.read_csv``.
        label (str): name used in error messages.

    Returns:
        BlendTable: rows in file order (not sorted; ``validate`` sorts).
    """
    id_cols = {name: blended_areaperil_equal_dtype[name] for name in blended_areaperil_equal_headers}
    try:
        df = pd.read_csv(file_in, dtype=id_cols)
    except pd.errors.EmptyDataError:
        raise OasisException(f"{label} is empty; it needs at least a header row")
    except ValueError as e:  # an id column that does not parse as its integer type
        raise OasisException(f"{label}: {e}") from e
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in id_cols if c not in df.columns]
    unknown = [c for c in df.columns if c not in id_cols and c != 'weight']
    if missing or unknown:
        raise OasisException(
            f"{label} has columns {list(df.columns)}: "
            + (f"missing {missing}" if missing else "")
            + ("; " if missing and unknown else "")
            + (f"unknown {unknown}" if unknown else "")
            + f". Expected {blended_areaperil_equal_headers} with an optional 'weight' column.")

    if 'weight' not in df.columns:
        weight_type = WEIGHT_EQUAL
    else:
        kind = df['weight'].dtype.kind
        if kind in 'iu':
            weight_type = WEIGHT_COUNT
            info = np.iinfo(blended_areaperil_count_dtype['weight'])
            if len(df) and (df['weight'].min() < info.min or df['weight'].max() > info.max):
                raise OasisException(f"{label}: integer weights must fit {blended_areaperil_count_dtype['weight']}")
        elif kind == 'f':
            weight_type = WEIGHT_FLOAT
            if df['weight'].isna().any():
                rows = (np.flatnonzero(df['weight'].isna().to_numpy())[:5] + 2).tolist()
                raise OasisException(f"{label}: weight is empty on line(s) {rows}")
        else:
            raise OasisException(f"{label}: weight column must hold integers or decimals, found {df['weight'].dtype}")

    dtype, _, _ = record_dtype(weight_type)
    records = np.empty(len(df), dtype=dtype)
    for name in dtype.names:
        records[name] = df[name].to_numpy()
    return BlendTable(to_blend_table(records, weight_type), weight_type)


# ---------------------------------------------------------------------------------------------
# Model storage
# ---------------------------------------------------------------------------------------------

def read_blended_areaperil(storage, ignore_file_type=set()):
    """Load and validate the model's blended areaperil table, if it has one.

    The binary form is preferred when both are present, as for the other model files.

    Args:
        storage (BaseStorage): model data storage.
        ignore_file_type (Set[str]): file extensions not to load.

    Returns:
        BlendTable | None: the validated, sorted table; None when the model has no blended
        areaperils, which is the ordinary case.
    """
    files = set(storage.listdir())
    for ext in ('bin', 'csv'):
        name = f'{BLENDED_AREAPERIL_FILE}.{ext}'
        if name not in files or ext in ignore_file_type:
            continue
        logger.debug(f"loading {storage.get_storage_url(name, encode_params=False)}")
        if ext == 'bin':
            with storage.open(name, 'rb') as f:
                blend = read_bin(f, label=name)
        else:
            with storage.open(name) as f:
                blend = read_csv(f, label=name)
        return validate(blend, label=name)
    return None


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------

def _ids(values, limit=5):
    values = np.unique(values)
    shown = ', '.join(str(v) for v in values[:limit])
    return shown + (f' ... ({len(values)} in all)' if len(values) > limit else '')


def validate(blend, label='blended areaperil table'):
    """Check the table's internal consistency and return it sorted by (blended id, placement).

    Checked, raising with the offending ids named:

    - every blended id names one ``placement_scheme_id``;
    - a blended id's ``placement_id`` values are exactly 0..n-1;
    - within a scheme every blended id lists the same number of positions, and a position has
      one weight whichever blended id carries it -- the weights describe where the building is,
      not which grid is looking at it;
    - weights are finite and non-negative, and a count or equal scheme has a positive total;
    - float weights sum to at most 1 (within ``FLOAT_SUM_TOLERANCE``). Less than 1 is legal and
      means the building may be outside every position; more than 1 cannot mean anything, since
      positions past a cumulative weight of 1 would never be selected;
    - no blended id is also used as an ``areaperil_id``.

    Not checked, deliberately: that an ``areaperil_id`` exists in the model. A cell the footprint
    does not provide behaves as no hazard, which a model may use on purpose, and the engine only
    loads the footprint rows a run needs. Nor that a location's blended ids share a scheme: the
    table has no locations in it.

    Args:
        blend (BlendTable): table as read, in any row order.
        label (str): name used in messages.

    Returns:
        BlendTable: the same rows, sorted.
    """
    table, weight_type = blend
    record_dtype(weight_type)
    table = table[np.lexsort((table['placement_id'], table['blended_areaperil_id']))]
    n = len(table)
    if n == 0:
        logger.warning(f"{label} has no rows")
        return BlendTable(table, weight_type)

    blended = table['blended_areaperil_id']
    scheme = table['placement_scheme_id']
    placement = table['placement_id'].astype(np.int64)
    weight = table['weight']

    group_start = np.flatnonzero(np.r_[True, blended[1:] != blended[:-1]])
    group_size = np.diff(np.r_[group_start, n])
    first = np.repeat(group_start, group_size)        # row index of each row's group start

    bad = scheme != scheme[first]
    if bad.any():
        raise OasisException(f"{label}: blended_areaperil_id {_ids(blended[bad])} name more than one placement_scheme_id")

    bad = placement != np.arange(n) - first
    if bad.any():
        raise OasisException(
            f"{label}: placement_id of blended_areaperil_id {_ids(blended[bad])} is not 0..n-1 "
            f"(each position listed once, numbered from 0 with no gaps)")

    group_scheme = scheme[group_start]
    order = np.lexsort((group_size, group_scheme))
    gs, gn = group_scheme[order], group_size[order]
    bad = (gs[1:] == gs[:-1]) & (gn[1:] != gn[:-1])
    if bad.any():
        raise OasisException(
            f"{label}: blended ids of placement_scheme_id {_ids(gs[1:][bad])} list different numbers of positions; "
            f"every blended id of a scheme must list the same positions")

    if weight_type != WEIGHT_EQUAL:
        bad = ~np.isfinite(weight) | (weight < 0)
        if bad.any():
            raise OasisException(f"{label}: blended_areaperil_id {_ids(blended[bad])} have negative or non-finite weights")
        order = np.lexsort((placement, scheme))
        s, p, w = scheme[order], placement[order], weight[order]
        bad = (s[1:] == s[:-1]) & (p[1:] == p[:-1]) & (w[1:] != w[:-1])
        if bad.any():
            raise OasisException(
                f"{label}: placement_scheme_id {_ids(s[1:][bad])} give one placement_id different weights in "
                f"different blended ids; a position's weight must be the same for every peril reading it")

    totals = np.add.reduceat(weight, group_start)
    if weight_type == WEIGHT_FLOAT:
        over = totals > 1.0 + FLOAT_SUM_TOLERANCE
        if over.any():
            raise OasisException(
                f"{label}: float weights of placement_scheme_id {_ids(group_scheme[over])} sum to more than 1 "
                f"(up to {totals[over].max():.6g}); positions past a cumulative weight of 1 would never be "
                f"selected. Use integer counts for relative weights, or rescale the weights to probabilities")
    else:
        bad = totals <= 0
        if bad.any():
            raise OasisException(
                f"{label}: placement_scheme_id {_ids(group_scheme[bad])} have a total weight of 0; with "
                f"{WEIGHT_TYPE_NAMES[weight_type]} weights no position could ever be selected")

    nested = np.isin(blended[group_start], table['areaperil_id'])
    if nested.any():
        raise OasisException(
            f"{label}: blended_areaperil_id {_ids(blended[group_start][nested])} also appear as areaperil_id; "
            f"blended ids and real cell ids must not overlap")

    return BlendTable(table, weight_type)


# ---------------------------------------------------------------------------------------------
# Blended areaperils of a run
# ---------------------------------------------------------------------------------------------

def build_blended_areaperils(items, blend=None):
    """Give every distinct item ``areaperil_id`` its blended areaperil.

    A blended areaperil is what an item's ``areaperil_id`` names: weighted placements, each read
    through one footprint areaperil. An id listed in the blend table takes its placements from
    there. Any other id is the size-1 case -- one placement of weight 1, read through the id
    itself -- so ordinary items take the same path as blended ones with K = 1.

    The K footprint areaperils of a blended areaperil are its distinct non-zero ``areaperil_id``
    values, in order of first placement, and ``areaperil_j`` maps each placement to one of them
    (-1 for ``areaperil_id = 0``, a placement this peril does not cover). A blended areaperil whose
    placements are all uncovered has K = 0: its items read no hazard and never produce a loss.

    Args:
        items (np.ndarray): items table with 'areaperil_id'.
        blend (BlendTable, optional): the validated blend table, sorted by (blended id,
            placement), or None when the model has none.

    Returns:
        tuple: ``items_blended_areaperil_idx`` (dense blended areaperil index per item), then the
        blended areaperil arrays: ``blended_areaperil_ja_offsets``,
        ``blended_areaperil_ja_areaperil_ids``, ``blended_areaperil_ja_probabilities``,
        ``blended_areaperil_placement_ja_offsets``, ``blended_areaperil_placement_ja_cum_weights``,
        and ``blended_areaperil_placement_ja_areaperil_j``. The ids are needed only while the
        structures are built; the event loop works with areaperil_j.
    """
    blended_ids, items_blended_areaperil_idx = np.unique(items['areaperil_id'], return_inverse=True)
    n_blended = blended_ids.shape[0]

    # rows of each id in the blend table; an id with none is a base areaperil
    if blend is None or blend.table.shape[0] == 0:
        row_start = row_end = np.zeros(n_blended, dtype=np.int64)
        table = None
    else:
        table = blend.table
        row_start = np.searchsorted(table['blended_areaperil_id'], blended_ids, side='left')
        row_end = np.searchsorted(table['blended_areaperil_id'], blended_ids, side='right')
        n_listed = len(np.unique(table['blended_areaperil_id']))
        n_used = int(np.count_nonzero(row_end > row_start))
        logger.info(f"blended areaperils: {n_used} of the {n_listed} in the blend table are used by items")
    is_blended = row_end > row_start

    # per blended areaperil: its areaperils and placements, built as lists for the blended ones
    k = np.ones(n_blended, dtype=np.int64)
    n_placements = np.ones(n_blended, dtype=np.int64)
    blended_parts = {}
    for b in np.flatnonzero(is_blended):
        rows = table[row_start[b]:row_end[b]]
        areaperil_ids = rows['areaperil_id']
        covered = areaperil_ids != 0
        _, first_seen = np.unique(areaperil_ids[covered], return_index=True)
        distinct = areaperil_ids[covered][np.sort(first_seen)]
        areaperil_j = np.full(rows.shape[0], -1, dtype=np.int32)
        for j, areaperil_id in enumerate(distinct):
            areaperil_j[areaperil_ids == areaperil_id] = j
        cum_weights = np.cumsum(rows['weight'], dtype=np.float64)
        scale = placement_scale(cum_weights)
        probabilities = np.array([rows['weight'][areaperil_j == j].sum() for j in range(distinct.shape[0])],
                                 dtype=np.float64) / scale if scale > 0 else np.zeros(distinct.shape[0])
        blended_parts[b] = (distinct, probabilities, cum_weights, areaperil_j, scale)
        k[b] = distinct.shape[0]
        n_placements[b] = rows.shape[0]

    blended_areaperil_ja_offsets = np.zeros(n_blended + 1, dtype=oasis_int)
    np.cumsum(k, out=blended_areaperil_ja_offsets[1:])
    blended_areaperil_placement_ja_offsets = np.zeros(n_blended + 1, dtype=oasis_int)
    np.cumsum(n_placements, out=blended_areaperil_placement_ja_offsets[1:])

    # base areaperils first, all at once: one areaperil (the id itself), one placement of weight 1
    blended_areaperil_ja_areaperil_ids = np.empty(blended_areaperil_ja_offsets[-1], dtype=areaperil_int)
    blended_areaperil_ja_probabilities = np.ones(blended_areaperil_ja_offsets[-1], dtype=np.float64)
    blended_areaperil_placement_ja_cum_weights = np.ones(blended_areaperil_placement_ja_offsets[-1], dtype=np.float64)
    blended_areaperil_placement_ja_areaperil_j = np.zeros(blended_areaperil_placement_ja_offsets[-1], dtype=np.int32)
    base = ~is_blended
    blended_areaperil_ja_areaperil_ids[blended_areaperil_ja_offsets[:-1][base]] = blended_ids[base]

    for b, (distinct, probabilities, cum_weights, areaperil_j, scale) in blended_parts.items():
        ap = slice(blended_areaperil_ja_offsets[b], blended_areaperil_ja_offsets[b + 1])
        pl = slice(blended_areaperil_placement_ja_offsets[b], blended_areaperil_placement_ja_offsets[b + 1])
        blended_areaperil_ja_areaperil_ids[ap] = distinct
        blended_areaperil_ja_probabilities[ap] = probabilities
        blended_areaperil_placement_ja_cum_weights[pl] = cum_weights
        blended_areaperil_placement_ja_areaperil_j[pl] = areaperil_j

    return (items_blended_areaperil_idx.astype(oasis_int),
            blended_areaperil_ja_offsets,
            blended_areaperil_ja_areaperil_ids,
            blended_areaperil_ja_probabilities,
            blended_areaperil_placement_ja_offsets,
            blended_areaperil_placement_ja_cum_weights,
            blended_areaperil_placement_ja_areaperil_j)


# ---------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------

@nb.njit(cache=True)
def placement_scale(cumulative):
    """What the placement uniform is multiplied by: the total weight, but never less than 1.

    That one rule serves every weight type. Counts and equal weights total at least 1, so they
    scale by their total. Float weights sum to at most 1 (the load validation refuses more, within
    its tolerance), so they scale by 1: a sum below 1 then leaves that share of samples unplaced,
    which is how a model says part of a location is outside every modelled position. Scaling float
    weights by their sum instead would silently normalise that shortfall away.
    """
    return max(1.0, cumulative[-1])


@nb.njit(cache=True)
def select_placement(cumulative, u):
    """Position selected by uniform ``u`` in [0, 1], or ``len(cumulative)`` when unplaced.

    ``cumulative`` is the running weight of the positions in placement order, scaled as
    ``placement_scale`` says. Unplaced happens only for float weights summing to less than 1, and
    means the building is outside every modelled position for that sample. Positions of zero weight
    are never selected.
    """
    scale = placement_scale(cumulative)
    x = u * scale
    if x >= scale:
        # The Latin Hypercube generator returns values in (0, 1], so u can be exactly 1; and u just
        # below 1 can round u * scale up to the total. Either way it belongs to the top of the
        # range, which is the last position, or unplaced when float weights fall short of 1.
        x = np.nextafter(scale, 0.0)
    return np.searchsorted(cumulative, x, side='right')


# ---------------------------------------------------------------------------------------------
# Mixture
# ---------------------------------------------------------------------------------------------

@nb.njit(cache=True)
def mix_effective_damage_cdfs(eff_cdfs, eff_lens, probabilities, live, out):
    """Effective damage cdf of a blended areaperil: its areaperils' cdfs weighted by probability.

    ``out[k] = sum over live j of probabilities[j] * cdf_j[k] + (1 - sum over live j of probabilities[j])``,
    where a cdf shorter than ``k`` reads as complete (1). The mass that no live areaperil carries --
    dead areaperils, placements no areaperil covers, and a float weight shortfall -- loses nothing,
    so it is put on damage bin 0. **Damage bin 0 must therefore be the zero-loss bin** of the damage
    bin dictionary, which is the convention every Oasis model follows.

    This is the per-item marginal of a blended item: what getmodel streams to gulpy, and what gulmc
    reports as the analytic mean, standard deviation, chance of loss and max. gulmc never samples
    from it: each of its samples reads the areaperil its placement falls on.

    Args:
        eff_cdfs (np.ndarray[float, 2d]): row j is areaperil j's effective damage cdf.
        eff_lens (np.ndarray[int]): valid length of each row.
        probabilities (np.ndarray[float64]): probability the building stands on each areaperil.
        live (np.ndarray[bool]): whether each areaperil has hazard (and no protection) here.
        out (np.ndarray[float]): output buffer, at least as long as the longest live row.

    Returns:
        int: the length of the mixed cdf, the longest live row; 0 when nothing is live.
    """
    n = 0
    live_probability = 0.
    for j in range(eff_lens.shape[0]):
        if live[j]:
            n = max(n, eff_lens[j])
            live_probability += probabilities[j]
    for k in range(n):
        acc = 1. - live_probability
        for j in range(eff_lens.shape[0]):
            if live[j]:
                acc += probabilities[j] * eff_cdfs[j, min(k, eff_lens[j] - 1)]
        out[k] = acc
    return n
