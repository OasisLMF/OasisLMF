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
A model writes weights in whichever form it holds, and all three select through the same rule
(``placement_cumulative`` and ``select_placement``):

``WEIGHT_FLOAT``
    probabilities. They may sum to less than 1, and the shortfall is the chance the building is
    outside every modelled position: the uniform is scaled by **1.0**, never by the sum.
``WEIGHT_COUNT``
    non-negative integer counts (address points, pixels, buildings). Scaled by their total, so
    there is no shortfall; "not at risk" is a row with ``areaperil_id = 0``.
``WEIGHT_EQUAL``
    no weight column. Every position has weight 1, scaled by the number of positions.

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
# Selection
# ---------------------------------------------------------------------------------------------

def placement_cumulative(weights, weight_type):
    """Cumulative weights of one scheme's positions and the scale to apply to the uniform.

    Returns:
        tuple(np.ndarray[f8], float): the running sum in placement order, and the scale --
        1.0 for float weights, so a sum below 1 leaves samples unplaced; the total otherwise.
    """
    cumulative = np.cumsum(weights, dtype=np.float64)
    return cumulative, placement_scale(cumulative, weight_type)


def placement_scale(cumulative, weight_type):
    """What the placement uniform is multiplied by: 1.0 for float weights, the total otherwise.

    Float weights keep 1.0 even when they sum below 1, so the shortfall leaves samples unplaced;
    scaling by their total would silently normalise it away.
    """
    return 1.0 if weight_type == WEIGHT_FLOAT else float(cumulative[-1])


@nb.njit(cache=True)
def select_placement(cumulative, scale, u):
    """Position selected by uniform ``u`` in [0, 1), or ``len(cumulative)`` when unplaced.

    Unplaced happens only for float weights summing to less than 1, and means the building is
    outside every modelled position for that sample. Positions of zero weight are never selected.
    """
    x = u * scale
    if x >= scale:
        # u just below 1 can round u * scale up to the total. Unreachable for float weights,
        # where scale is 1.0 and u * 1.0 is exact.
        x = np.nextafter(scale, 0.0)
    return np.searchsorted(cumulative, x, side='right')
