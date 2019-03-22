# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

__all__ = [
    'factorize_dataframe',
    'fast_zip_dataframe_cols',
    'get_dataframe',
    'get_json',
    'get_timestamp',
    'get_utctimestamp',
    'merge_dataframes',
    'PANDAS_BASIC_DTYPES',
    'set_col_dtypes'
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
    'object': np.object,
    'str': np.object,
    builtins.str: np.object
}


def factorize_dataframe(
        df,
        by_cols,
        enumerate_only=False
    ):
        factors = pd.factorize(fast_zip_dataframe_cols(df, by_cols))
        return factors[0] + 1 if enumerate_only else factors


def fast_zip_dataframe_cols(df, cols):
    return pd._libs.lib.fast_zip(
        [df[t].values for t in cols]
    )

def get_dataframe(
    src_fp=None,
    src_type='csv',
    src_buf=None,
    src_data=None,
    csv_cols=None,
    float_precision='high',
    empty_data_error_msg=None,
    lowercase_cols=True,
    replace_nans_by_none=True,
    required_cols=(),
    defaulted_cols={},
    non_na_cols=(),
    col_dtypes={},
    index=None,
    sort_col=None,
    sort_ascending=None,
    memory_map=False
):
    if not (src_fp or src_buf or src_data is not None):
        raise OasisException(
            'A CSV or JSON file path or a string buffer of such a file or an '
            'appropriate data structure or dataframe must be provided'
        )

    df = None

    if src_fp and src_type == 'csv':
        df = pd.read_csv(src_fp, float_precision=float_precision, usecols=csv_cols, memory_map=memory_map)
    elif src_buf and src_type == 'csv':
        df = pd.read_csv(io.StringIO(src_buf), float_precision=float_precision, usecols=csv_cols, memory_map=memory_map)
    elif src_fp and src_type == 'json':
        df = pd.read_json(src_fp, precise_float=(True if float_precision == 'high' else False))
    elif src_buf and src_type == 'json':
        df = pd.read_json(io.StringIO(src_buf), precise_float=(True if float_precision == 'high' else False))
    elif src_data and isinstance(src_data, list):
        df = pd.DataFrame(data=src_data)
    elif src_data and  isinstance(src_data, pd.DataFrame):
        df = pd.DataFrame(src_data)

    if len(df) == 0:
        raise OasisException(empty_data_error_msg)

    if index:
        df.index = index

    if lowercase_cols:
        df.columns = df.columns.str.lower()

    if replace_nans_by_none:
        df = df.where(df.notnull(), None)

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
    # In this sense, defaulting of column values via the `defaulted_cols`
    # optional argument is redundant - but there may be some cases where it is
    # convenient to have this feature at the code level.

    if defaulted_cols:
        _defaulted_cols = {k.lower(): v for k, v in viewitems(defaulted_cols)} if lowercase_cols else defaulted_cols
        defaults = {c for c in _defaulted_cols}.difference({c for c in df.columns})
        for col in defaults:
            df[col] = _defaulted_cols[col]

    if non_na_cols:
        _non_na_cols = tuple(col.lower() for col in non_na_cols) if lowercase_cols else non_na_cols
        df.dropna(subset=_non_na_cols, inplace=True)

    if col_dtypes:
        set_col_dtypes(df, col_dtypes)

    if sort_col:
        _sort_col = sort_col.lower() if lowercase_cols else sort_col
        sort_ascending = sort_ascending if sort_ascending is not None else True
        df.sort_values(_sort_col, axis=0, ascending=sort_ascending, inplace=True)

    return df


def set_col_dtypes(df, col_dtypes):
    for col, dtype in viewitems(col_dtypes):
        if dtype in ('int', 'bool', 'float', 'object', 'str',):
            dtype = getattr(builtins, dtype)
        if col in df:
            df[col] = df[col].astype(PANDAS_BASIC_DTYPES[dtype])


def get_json(src_fp, key_transform=None):
    di = None
    try:
        with io_open(src_fp, 'r', encoding='utf-8') as f:
            di = json.load(f)
    except (IOError, JSONDecodeError, OSError, TypeError) as e:
        return

    return di if not key_transform else {key_transform(k): v for k, v in viewitems(di)}


def get_timestamp(thedate=None, fmt='%Y%m%d%H%M%S'):
    """ Get a timestamp """
    d = thedate if thedate else datetime.now()
    return d.strftime(fmt)


def get_utctimestamp(thedate=None, fmt='%Y-%b-%d %H:%M:%S'):
    """
    Returns a UTC timestamp for a given ``datetime.datetime`` in the
    specified string format - the default format is::

        YYYY-MMM-DD HH:MM:SS
    """
    d = thedate.astimezone(pytz.utc) if thedate else datetime.utcnow()
    return d.strftime(fmt)


def merge_dataframes(left, right, **kwargs):
    """
    Merges two dataframes by ensuring there is no duplication of columns.
    """
    _left = left.copy(deep=True)
    _left['index'] = _left.get('index', _left.index)
    _right = right.copy(deep=True)
    _right['index'] = _right.get('index', _right.index)

    left_keys = kwargs.get('left_on') or kwargs.get('on') or []
    left_keys = [left_keys] if isinstance(left_keys, str) else left_keys

    drop_cols = [
        k for k in set(_left.columns).intersection(_right.columns)
        if k and k not in left_keys + ['index']
    ]

    drop_duplicates = kwargs.get('drop_duplicates', True)
    kwargs.pop('drop_duplicates') if 'drop_duplicates' in kwargs else None

    merge = pd.merge(
        _left.drop(drop_cols, axis=1),
        _right,
        **kwargs
    )
    merge['index'] = merge.index

    merge.drop(['index_x', 'index_y'], axis=1, inplace=True)

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
