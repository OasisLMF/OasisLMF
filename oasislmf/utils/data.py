# -*- coding: utf-8 -*-

__all__ = [
    'get_dataframe'
]

import builtins
import io

import pandas as pd

import six

from .exceptions import OasisException


def get_dataframe(
    src_fp=None,
    src_type='csv',
    src_buf=None,
    src_data=None,
    float_precision='high',
    lowercase_cols=True,
    required_cols=[],
    defaulted_cols={},
    non_na_cols=(),
    col_dtypes={},
    index_col=True,
    sort_col=None,
    sort_ascending=None,
):
    if not (src_fp or src_buf or src_data is not None):
        raise OasisException(
            'A CSV or JSON file path or a string buffer of such a file or an '
            'appropriate data structure or dataframe must be provided'
        )

    df = None

    if src_fp and src_type == 'csv':
        df = pd.read_csv(src_fp, float_precision=float_precision)
    elif src_buf and src_type == 'csv':
        df = pd.read_csv(io.StringIO(src_buf), float_precision=float_precision)
    elif src_fp and src_type == 'json':
        df = pd.read_json(src_fp, precise_float=(True if float_precision == 'high' else False))
    elif src_buf and src_type == 'json':
        df = pd.read_json(io.StringIO(src_buf), precise_float=(True if float_precision == 'high' else False))
    elif src_data and (isinstance(src_data, list) or isinstance(src_data, pd.DataFrame)):
        df = pd.DataFrame(data=src_data, dtype=object)

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
    # In this sense, defaulting of column values via the `defaulted_cols`
    # optional argument is redundant - but there may be some cases where it is
    # convenient to have this feature at the code level.

    if defaulted_cols:
        _defaulted_cols = {k.lower(): v for k, v in defaulted_cols.items()} if lowercase_cols else defaulted_cols
        defaults = {c for c in _defaulted_cols}.difference({c for c in df.columns})
        for col in defaults:
            df[col] = _defaulted_cols[col]

    if non_na_cols:
        _non_na_cols = tuple(col.lower() for col in non_na_cols) if lowercase_cols else non_na_cols
        df.dropna(subset=_non_na_cols, inplace=True)

    if index_col:
        df['index'] = range(len(df))

    if col_dtypes:
        _col_dtypes = {
            (k.lower() if lowercase_cols else k): (getattr(builtins, v) if v in ('int', 'bool', 'float', 'str',) else v) for k, v in six.iteritems(col_dtypes)
        }
        for col, dtype in six.iteritems(_col_dtypes):
            df[col] = df[col].astype(dtype) if dtype != int else df[col].astype(object)

    if sort_col:
        _sort_col = sort_col.lower() if lowercase_cols else sort_col
        sort_ascending = sort_ascending if sort_ascending is not None else True
        df.sort_values(_sort_col, axis=0, ascending=sort_ascending, inplace=True)

    return df
