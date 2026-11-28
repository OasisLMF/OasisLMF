"""Narrow integer data to fixed-width dtypes without silently wrapping (#2197).

``Series.astype('u4')``, assigning a wider column into a structured numpy field and, on some
pandas versions, ``pd.read_csv(dtype={'x': 'u4'})`` all wrap a value that doesn't fit the target
dtype: areaperil_id 4294967298 is stored as 2 and vulnerability_id 3000000000 as -1294967296.
The wrapped value usually looks like a valid id, so the run completes with wrong losses.

The helpers here check the range before narrowing and raise an OasisException naming the source,
the column and the offending value. Data that is already of a dtype that casts safely to the
target, such as ids oasislmf generates itself, is not scanned.
"""
import numpy as np
import pandas as pd

from oasislmf.utils.exceptions import OasisException

__all__ = [
    'raise_if_out_of_range',
    'astype_in_range',
    'widen_int_dtype',
    'widen_int_dtypes',
]


def _int_target(dtype):
    """The numpy integer dtype of `dtype`, or None if it isn't a fixed-width numpy integer dtype."""
    try:
        target = np.dtype(dtype)
    except TypeError:  # pandas extension dtypes ('category', 'Int64', ...)
        return None
    return target if target.kind in 'iu' else None


def _as_numpy(values):
    """`values` as a numpy array plus the labels to report a row by (None for plain arrays)."""
    if isinstance(values, (pd.Series, pd.Index)):
        labels = values.index if isinstance(values, pd.Series) else None
        dtype = values.dtype
        if isinstance(dtype, pd.CategoricalDtype):  # rare for integers; checked value by value
            values = values.astype(object)
            dtype = values.dtype
        numpy_dtype = getattr(dtype, 'numpy_dtype', None)  # nullable and arrow integer dtypes
        if numpy_dtype is not None and not isinstance(dtype, np.dtype):
            present = values.notna().to_numpy()
            if not present.all():
                values = values[present]
                labels = labels[present] if labels is not None else None
            return values.to_numpy(numpy_dtype), labels
        return values.to_numpy(), labels
    return np.asarray(values), None


def _object_out_of_range(value, info):
    """Whether a single Python object is a number outside `info`'s range (missing values are not)."""
    try:
        if value is None or value != value:
            return False
        return not (info.min <= value <= info.max)
    except TypeError:  # not a number, left to the cast itself
        return False


def _out_of_range_mask(arr, target):
    """Boolean mask of the values of `arr` that `target` can't hold, or None if all fit."""
    info = np.iinfo(target)
    if arr.dtype.kind in 'iu':
        if np.can_cast(arr.dtype, target, 'safe') or arr.size == 0:
            return None
        # min/max first: a single pass each, and nothing more to do in the usual case
        if int(arr.min()) >= info.min and int(arr.max()) <= info.max:
            return None
        # compare only against the bounds the source dtype can cross, as Python ints outside the
        # source range don't compare reliably on every numpy version
        source_info = np.iinfo(arr.dtype)
        bad = np.zeros(arr.shape, dtype=bool)
        if info.min > source_info.min:
            bad |= arr < info.min
        if info.max < source_info.max:
            bad |= arr > info.max
        return bad
    if arr.dtype.kind == 'f':
        # bounds as exact powers of two, upper bound exclusive: float(info.max) rounds up to
        # 2**63 for int64, which would let 2.0**63 through and wrap to the int64 minimum
        low, high = float(info.min), float(info.max + 1)
        values = arr.astype(np.float64, copy=False)
        bad = (values < low) | (values >= high)  # NaN compares False, it isn't a range problem
        return bad if bad.any() else None
    if arr.dtype.kind == 'b':
        return None
    if arr.dtype.kind == 'O':  # e.g. integers beyond 64 bits kept as Python ints
        bad = np.fromiter((_object_out_of_range(value, info) for value in arr), dtype=bool, count=arr.size)
        return bad if bad.any() else None
    return None  # strings and other kinds are left to the cast itself


def raise_if_out_of_range(values, dtype, column, source=None):
    """Raise OasisException if any of `values` would not fit the integer `dtype`.

    Values that are missing (NaN/NA) are not range errors and are ignored here. A `dtype` that
    isn't a fixed-width numpy integer dtype is not checked.

    Args:
        values (pd.Series | pd.Index | np.ndarray): the values about to be narrowed
        dtype: the target dtype
        column (str): column or field name, for the error message
        source (str): file or table the values come from, for the error message

    Raises:
        OasisException: naming the source, column, first offending value and its row
    """
    target = _int_target(dtype)
    if target is None:
        return
    arr, labels = _as_numpy(values)
    if arr.ndim != 1:
        arr = arr.ravel()
        labels = None
    bad = _out_of_range_mask(arr, target)
    if bad is None:
        return
    first = int(np.flatnonzero(bad)[0])
    info = np.iinfo(target)
    where = f"{source}: " if source else ""
    row = f" at index {labels[first]}" if labels is not None else f" at position {first}"
    hint = ""
    if 'areaperil' in str(column).lower():
        hint = " Models with larger areaperil ids need OASIS_AREAPERIL_TYPE=u8."
    raise OasisException(
        f"{where}{column} value {arr[first]}{row} is out of range for {target} "
        f"({info.min} to {info.max}) and would wrap to a different value; "
        f"{int(bad.sum())} value(s) out of range.{hint}")


def astype_in_range(data, dtypes, source=None):
    """``data.astype(dtypes)`` that raises instead of wrapping integer values that don't fit.

    Args:
        data (pd.DataFrame | pd.Series): the data to cast
        dtypes (dict | dtype): for a DataFrame, a dict of column -> dtype; for a Series, a dtype
        source (str): file or table the data comes from, for the error message

    Returns:
        the cast data, as ``astype`` would return it
    """
    if isinstance(data, pd.Series):
        raise_if_out_of_range(data, dtypes, data.name, source)
        return data.astype(dtypes)
    for column, dtype in dtypes.items():
        if column in data.columns:
            raise_if_out_of_range(data[column], dtype, column, source)
    return data.astype(dtypes)


def widen_int_dtype(dtype):
    """The dtype to read a column with before narrowing it to `dtype` with a range check.

    Integer dtypes narrower than 64 bits become int64 (so a negative value for an unsigned target
    is still read), uint64 stays uint64. Any other dtype is returned unchanged.
    """
    target = _int_target(dtype)
    if target is None or target == np.uint64:
        return dtype
    return np.dtype(np.int64)


def widen_int_dtypes(dtypes):
    """``{column: widen_int_dtype(dtype)}`` for a dict or a structured numpy dtype."""
    if isinstance(dtypes, np.dtype) and dtypes.names:
        return {name: widen_int_dtype(dtypes[name]) for name in dtypes.names}
    return {column: widen_int_dtype(dtype) for column, dtype in dtypes.items()}
