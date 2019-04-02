# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import int, str

from future import standard_library
standard_library.install_aliases()

__all__ = [
    'factorize_array',
    'factorize_dataframe',
    'factorize_ndarray',
    'fast_zip_arrays',
    'fast_zip_dataframe_columns',
    'get_dataframe',
    'get_json',
    'get_timestamp',
    'get_utctimestamp',
    'merge_dataframes',
    'PANDAS_BASIC_DTYPES',
    'set_dataframe_column_dtypes'
]

import builtins
import io
import json
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime
from future.utils import viewitems

try:
    from json import JSONDecodeError
except ImportError:
    from builtins import ValueError as JSONDecodeError

from tabulate import tabulate

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import pytz

from .exceptions import OasisException


PANDAS_BASIC_DTYPES = {
    'int32': np.int32,
    'int64': np.int64,
    builtins.int: np.int64,
    'float32': np.float32,
    'float64': np.float64,
    builtins.float: np.float64,
    'bool': np.bool,
    builtins.bool: np.bool,
    'str': np.object,
    builtins.str: np.object
}


def factorize_array(arr):
    """
    Groups a 1D Numpy array by item value, and optionally enumerates the
    groups, starting from 1. The default or assumed type is a Nunpy
    array, although a Python list, tuple or Pandas series will work too.

    :param arr: 1D Numpy array (or list, tuple, or Pandas series)
    :type arr: numpy.ndarray

    :return: A 2-tuple consisting of the enumeration and the value groups
    :rtype: tuple
    """
    enum, groups = pd.factorize(arr)

    return enum + 1, groups


def factorize_ndarray(ndarr, row_idxs=[], col_idxs=[]):
    """
    Groups an n-D Numpy array by item value, and optionally enumerates the
    groups, starting from 1. The default or assumed type is a Nunpy
    array, although a Python list, tuple or Pandas series will work too.

    :param ndarr: n-D Numpy array (or appropriate Python structure or Pandas dataframe)
    :type ndarr: numpy.ndarray

    :param row_idxs: A list of row indices to use for factorization (optional)
    :type row_idxs: list

    :param col_idxs: A list of column indices to use for factorization (optional)
    :type col_idxs: list

    :return: A 2-tuple consisting of the enumeration and the value groups
    :rtype: tuple
    """
    if not (row_idxs or col_idxs):
        raise OasisException('A list of row indices or column indices must be provided')

    _ndarr = ndarr[:, col_idxs].transpose() if col_idxs else ndarr[row_idxs, :]
    rows, _ = _ndarr.shape

    if rows == 1:
        return factorize_array(_ndarr[0])

    enum, groups = pd.factorize(fast_zip_arrays(*(arr for arr in _ndarr)))

    return enum + 1, groups


def factorize_dataframe(
        df,
        by_row_labels=None,
        by_row_indices=None,
        by_col_labels=None,
        by_col_indices=None
    ):
    """
    Groups a selection of rows or columns of a Pandas DataFrame array by value,
    and optionally enumerates the groups, starting from 1.

    :param df: Pandas DataFrame
    :type: pandas.DataFrame

    :param by_row_labels: A list or tuple of row labels
    :type by_row_labels: list, tuple

    :param by_row_indices: A list or tuple of row indices
    :type by_row_indices: list, tuple

    :param by_col_labels: A list or tuple of column labels
    :type by_col_labels: list, tuple

    :param by_col_indices: A list or tuple of column indices
    :type by_col_indices: list, tuple

    :return: A 2-tuple consisting of the enumeration and the value groups
    :rtype: tuple
    """
    by_row_indices = by_row_indices or (None if not by_row_labels else [df.index.get_loc(label) for label in by_row_labels])
    by_col_indices = by_col_indices or (None if not by_col_labels else [df.columns.get_loc(label) for label in by_col_labels])

    return factorize_ndarray(
        df.values,
        row_idxs=by_row_indices,
        col_idxs=by_col_indices
    )


def fast_zip_arrays(*arrays):
    """
    Speedy zip of a sequence or ordered iterable of Numpy arrays (Python
    iterables with ordered elements such as lists and tuples, or iterators
    or generators of these, will also work).

    :param arrays: An iterable or iterator or generator of Numpy arrays
    :type arrays: list, tuple, collections.Iterator, types.GeneratorType

    :return: A Numpy 1D array of ``n``-tuples of the zipped ``n`` sequences
    :rtype: np.array
    """
    return pd._libs.lib.fast_zip([arr for arr in arrays])


def fast_zip_dataframe_columns(df, cols):
    """
    Speedy zip of a sequence or ordered iterable of Pandas DataFrame columns
    (Python iterables with ordered elements such as lists and tuples, or
    iterators or generators of these, will also work).

    :param df: Pandas DataFrame
    :type df: pandas.DataFrame

    :cols: An iterable or iterator or generator of Pandas DataFrame columns
    :type cols: list, tuple, collections.Iterator, types.GeneratorType
    """
    return fast_zip_arrays(*(df[col].values for col in cols))


def get_dataframe(
    src_fp=None,
    src_type='csv',
    src_buf=None,
    src_data=None,
    col_dtypes={},
    subset_cols=None,
    float_precision='high',
    empty_data_error_msg=None,
    lowercase_cols=True,
    required_cols=(),
    col_defaults={},
    non_na_cols=(),
    sort_cols=None,
    sort_ascending=None,
    memory_map=False
):
    if not (src_fp or src_buf or src_data is not None):
        raise OasisException(
            'A CSV or JSON file path or a string buffer of such a file or an '
            'appropriate data structure or Pandas DataFrame must be provided'
        )

    _col_dtypes = {
        k: getattr(builtins, v) if v in ('int', 'float', 'str', 'bool',) else PANDAS_BASIC_DTYPES[v]
        for k, v in viewitems(col_dtypes)
    }

    df = None


    if src_fp and src_type == 'csv':
        df = pd.read_csv(src_fp,  float_precision=float_precision, usecols=subset_cols, memory_map=memory_map)
        col_dtypes_set = True
    elif src_buf and src_type == 'csv':
        df = pd.read_csv(io.StringIO(src_buf),  float_precision=float_precision, usecols=subset_cols, memory_map=memory_map)
        col_dtypes_set = True
    elif src_fp and src_type == 'json':
        df = pd.read_json(src_fp,  precise_float=(True if float_precision == 'high' else False))
        col_dtypes_set = True
    elif src_buf and src_type == 'json':
        df = pd.read_json(io.StringIO(src_buf),  precise_float=(True if float_precision == 'high' else False))
        col_dtypes_set = True
    elif src_data and isinstance(src_data, list):
        df = pd.DataFrame(data=src_data)
    elif src_data and  isinstance(src_data, pd.DataFrame):
        df = pd.DataFrame(src_data)

    if len(df) == 0:
        raise OasisException(empty_data_error_msg)

    if lowercase_cols:
        df.columns = df.columns.str.lower()

    if required_cols:
        _required_cols = [c.lower() for c in required_cols] if lowercase_cols else required_cols
        missing = {col for col in sorted(_required_cols)}.difference(df.columns)
        if missing:
            raise OasisException('Missing required columns: {}'.format(missing))

    # Defaulting of column values is best done via the source data and not the
    # code, i.e. if a column 'X' in a frame is supposed to have 0s everywhere
    # the simplest way of achieving this is for the source data (whether it is
    # CSV or JSON file, or a list of dicts or tuples) to have an 'X' column
    # with 0 values, so that when it is loaded into the frame the 'X' will have
    # 0 values, as expected.
    #
    # In this sense, defaulting of column values via the `col_defaults`
    # optional argument is redundant - but there may be some cases where it is
    # convenient to have this feature at the code level.

    if col_defaults:
        _col_defaults = {k.lower(): v for k, v in viewitems(col_defaults)} if lowercase_cols else col_defaults
        for col, val in viewitems(_col_defaults):
            df.loc[:, col] = df.loc[:, col].fillna(val) if col in df else val

    if non_na_cols:
        _non_na_cols = tuple(col.lower() for col in non_na_cols) if lowercase_cols else non_na_cols
        df.dropna(subset=_non_na_cols, inplace=True)

    if col_dtypes:
        _col_dtypes = {k.lower(): v for k, v in viewitems(col_dtypes)} if lowercase_cols else col_dtypes
        set_dataframe_column_dtypes(df, _col_dtypes)

    if sort_cols:
        _sort_cols = (
            [(col.lower() if lowercase_cols else col) for col in sort_cols] if (isinstance(sort_cols, list) or isinstance(sort_cols, tuple) or isinstance(sort_cols, set))
            else (sort_cols.lower() if lowercase_cols else sort_cols)
        )
        sort_ascending = sort_ascending if sort_ascending is not None else True
        df.sort_values(_sort_cols, axis=0, ascending=sort_ascending, inplace=True)

    return df


def get_json(src_fp):
    """
    Loads JSON from file.

    :param src_fp: Source JSON file path
    :type src_fp: str

    :return: dict
    :rtype: dict
    """
    try:
        with io_open(src_fp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, JSONDecodeError, OSError, TypeError) as e:
        raise OasisException('Error trying to load JSON from {}'.format(src_fp))


def get_timestamp(thedate=datetime.now(), fmt='%Y%m%d%H%M%S'):
    """
    Get a timestamp string from a ``datetime.datetime`` object

    :param thedate: ``datetime.datetime`` object
    :type thedate: datetime.datetime

    :param fmt: Timestamp format string
    :type fmt: str

    :return: Timestamp string
    :rtype: str
    """
    return thedate.strftime(fmt)


def get_utctimestamp(thedate=datetime.utcnow(), fmt='%Y-%b-%d %H:%M:%S'):
    """
    Get a UTC timestamp string from a ``datetime.datetime`` object

    :param thedate: ``datetime.datetime`` object
    :type thedate: datetime.datetime

    :param fmt: Timestamp format string, default is "%Y-%b-%d %H:%M:%S"
    :type fmt: str

    :return: UTC timestamp string
    :rtype: str
    """
    return thedate.astimezone(pytz.utc).strftime(fmt)


def merge_dataframes(left, right, **kwargs):
    """
    Merges two dataframes by ensuring there is no duplication of columns.
    """
    _left = left.copy(deep=True)
    _right = right.copy(deep=True)

    left_keys = kwargs.get('left_on') or kwargs.get('on') or []
    left_keys = [left_keys] if isinstance(left_keys, str) else left_keys

    drop_cols = [
        k for k in set(_left.columns).intersection(_right.columns)
        if k and k not in left_keys
    ]

    drop_duplicates = kwargs.get('drop_duplicates', True)
    kwargs.pop('drop_duplicates') if 'drop_duplicates' in kwargs else None

    merge = pd.merge(
        _left.drop(drop_cols, axis=1),
        _right,
        **kwargs
    )

    return merge if not drop_duplicates else merge.drop_duplicates()


def print_dataframe(
    df,
    objectify_cols=[],
    show_index=False,
    table_header=None,
    column_headers='keys',
    tablefmt='psql',
    floatfmt=".2f",
    sep=' ',
    end='\n',
    file=sys.stdout,
    flush=False,
    **tabulate_kwargs
):
    _df = df.copy(deep=True)

    for col in objectify_cols:
        _df[col] = _df[col].astype(object)

    if table_header:
        print('\n{}'.format(table_header))

    if tabulate_kwargs:
        tabulate_kwargs.pop('headers') if 'headers' in tabulate_kwargs else None
        tabulate_kwargs.pop('tablefmt') if 'tablefmt' in tabulate_kwargs else None
        tabulate_kwargs.pop('floatfmt') if 'floatfmt' in tabulate_kwargs else None
        tabulate_kwargs.pop('showindex') if 'showindex' in tabulate_kwargs else None

    print(tabulate(_df, headers=column_headers, tablefmt=tablefmt, showindex=show_index, floatfmt=floatfmt, **tabulate_kwargs), sep=sep, end=end, file=file, flush=flush)


def set_dataframe_column_dtypes(df, col_dtypes):
    for col, dtype in viewitems(col_dtypes):
        if dtype in ('int', 'bool', 'float', 'object', 'str',):
            dtype = getattr(builtins, dtype)
        if col in df:
            df[col] = df[col].astype(PANDAS_BASIC_DTYPES[dtype])
