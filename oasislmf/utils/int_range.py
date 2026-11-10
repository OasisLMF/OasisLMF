"""Range-checked narrowing of integer data.

numpy and pandas wrap integers that do not fit a narrower type without any error when converting
with ``astype``, assigning into a structured array field, or casting float to int. An id such as
areaperil 4294967298 then becomes 2, a valid looking id, and the run completes with wrong losses.
These helpers check the range first and raise instead (#2197).
"""
import numpy as np
import pandas as pd

from oasislmf.utils.exceptions import OasisException

__all__ = [
    'astype_checked',
    'check_int_range',
    'narrow_int',
    'widen_int_dtypes',
]


def _as_numpy(values):
    """Return ``values`` as a numpy array to range check, and whether its positions match those of
    ``values``. The array is None if numpy and pandas raise on out of range values of this kind
    themselves (object, str) or it has no numpy equivalent.
    """
    if isinstance(values, (pd.Series, pd.Index)):
        if isinstance(values.dtype, pd.CategoricalDtype):
            return _as_numpy(values.cat.categories)[0], False
        if isinstance(values.dtype, pd.api.extensions.ExtensionDtype):
            numpy_dtype = getattr(values.dtype, 'numpy_dtype', None)
            if numpy_dtype is None or numpy_dtype.kind not in 'iuf':
                return None, False
            # 0 fits every integer type, so missing values pass and positions are kept
            return values.to_numpy(dtype=numpy_dtype, na_value=0), True
        values = values.to_numpy()
    arr = np.asarray(values)
    return (arr if arr.dtype.kind in 'iuf' else None), True


def _out_of_range(arr, info):
    """Boolean mask of the values in ``arr`` outside the range ``info`` (an ``np.iinfo``)."""
    if arr.dtype.kind == 'f':
        # info.max (2**n - 1) is not exactly representable in floats this wide: 2**63 - 1 rounds up to
        # 2**63 in float64, so `arr > info.max` would let 2.0**63 through to wrap. The bounds that are
        # powers of two are exact, so compare against those, with an exclusive upper bound. Narrower
        # floats are widened first (exactly) so the bounds neither round nor overflow to inf.
        if arr.dtype.itemsize < 8:
            arr = arr.astype(np.float64)
        signed = info.min < 0
        low = -np.ldexp(1.0, info.bits - 1) if signed else 0.0
        high = np.ldexp(1.0, info.bits - 1 if signed else info.bits)
        return (arr < low) | (arr >= high)  # NaN compares False either way
    # Only compare against a bound the source type can go past; that bound is then representable
    # in the source type, so the comparison is exact on every numpy version.
    src = np.iinfo(arr.dtype)
    mask = np.zeros(arr.shape, dtype=bool)
    if src.min < info.min:
        mask |= arr < info.min
    if src.max > info.max:
        mask |= arr > info.max
    return mask


def check_int_range(values, dtype, name):
    """Raise if any of ``values`` would not fit the integer type ``dtype``.

    Source data whose type always fits ``dtype`` is accepted without scanning it, so the check
    costs nothing for ids that are already the target type.

    Args:
        values (pandas.Series | pandas.Index | numpy.ndarray): the data about to be converted.
        dtype (numpy.dtype | str | type): the target type; non-integer targets are not checked.
        name (str): what the data is, for the error message, e.g. ``'keys AreaPerilID'``.

    Raises:
        OasisException: naming the first value out of range, where it is and the valid range.
    """
    try:
        dtype = np.dtype(dtype)
    except TypeError:
        return  # a pandas-only type such as 'category'
    if dtype.kind not in 'iu':
        return
    arr, aligned = _as_numpy(values)
    if arr is None or arr.size == 0:
        return
    if arr.dtype.kind in 'iu' and np.can_cast(arr.dtype, dtype, casting='safe'):
        return
    info = np.iinfo(dtype)
    mask = _out_of_range(arr, info)
    if not mask.any():
        return
    pos = int(np.flatnonzero(mask)[0])
    if not aligned:
        where = ' (a category)'
    elif isinstance(values, pd.Series):
        where = f' at row {values.index[pos]}'
    else:
        where = f' at position {pos}'
    msg = (f'{name}: value {arr[pos]}{where} does not fit {dtype.name} '
           f'(range {info.min} to {info.max}); converting it would silently change it to another value.')
    if 'areaperil' in name.lower() and dtype.itemsize < 8:
        msg += ' Use OASIS_AREAPERIL_TYPE=u8 for areaperil ids above 4294967295.'
    raise OasisException(msg)


def narrow_int(values, dtype, name):
    """Convert ``values`` to ``dtype`` with ``astype``, raising rather than wrapping out of range values.

    Args:
        values (pandas.Series | numpy.ndarray): the data to convert.
        dtype (numpy.dtype | str | type): the target type.
        name (str): what the data is, for the error message.

    Returns:
        pandas.Series | numpy.ndarray: ``values`` converted to ``dtype``.
    """
    check_int_range(values, dtype, name)
    return values.astype(dtype)


def astype_checked(df, dtypes, source):
    """``df.astype(dtypes)``, raising rather than wrapping integers that do not fit their new type.

    Args:
        df (pandas.DataFrame): the frame to convert.
        dtypes (dict): column name to target type, as for ``DataFrame.astype``.
        source (str): where the data came from, e.g. a file name, for the error message.

    Returns:
        pandas.DataFrame: the converted frame.
    """
    for col, dtype in dtypes.items():
        if col in df.columns:
            check_int_range(df[col], dtype, f'{source} {col}')
    return df.astype(dtypes)


def widen_int_dtypes(dtypes):
    """Return a copy of a ``read_csv`` dtype mapping with integer types narrower than 64 bits
    replaced by ``int64``.

    Reading at full width and then narrowing with :func:`astype_checked` or :func:`check_int_range`
    turns an out of range value into a clear error. Reading straight into the narrow type wraps it
    on older pandas, and fails with an error that names no column or value on pandas >= 3.1.

    Args:
        dtypes (dict): column name to type, as passed to ``pandas.read_csv``.

    Returns:
        dict: the widened mapping; non-integer and 64-bit types are unchanged.
    """
    widened = {}
    for col, dtype in dtypes.items():
        try:
            np_dtype = np.dtype(dtype)
        except TypeError:
            np_dtype = None
        narrow = np_dtype is not None and np_dtype.kind in 'iu' and np_dtype.itemsize < 8
        widened[col] = np.dtype('int64') if narrow else dtype
    return widened
