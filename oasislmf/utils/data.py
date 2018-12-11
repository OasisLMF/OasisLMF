# -*- coding: utf-8 -*-

__all__ = [
    'get_dataframe'
]

import builtins

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
    index_col=True,
    non_na_cols=(),
    col_dtypes={},
    sort_col=None,
    sort_ascending=None,
    required_cols = [],
    default_values = {}
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

    if required_cols is not None:
        if lowercase_cols:
            required_cols = [s.lower() for s in required_cols]
        missing_required_cols = []
        for required_col in required_cols:
            if required_col not in df.columns:
                missing_required_cols.append(required_col)
        if len(missing_required_cols) > 0:
            raise OasisException(
                "Missing required columns: {}".format(
                    ','.join(missing_required_cols)
                )                    
            )

    if default_values is not None:
        for default_col in default_values:
            if default_col not in df.columns:
                df[default_col] = default_values[default_col]

    if index_col:
        df['index'] = list(range(len(df)))

    if non_na_cols:
        _non_na_cols = tuple(col.lower() for col in non_na_cols) if lowercase_cols else non_na_cols
        df.dropna(subset=_non_na_cols, inplace=True)

    if col_dtypes:
        _col_dtypes = {
            (k.lower() if lowercase_cols else k):(getattr(builtins, v) if v in ('int', 'bool', 'float', 'str',) else v) for k, v in six.iteritems(col_dtypes)
        }
        for col, dtype in six.iteritems(_col_dtypes):
            df[col] = df[col].astype(dtype) if dtype != int else df[col].astype(object)

    if sort_col:
        _sort_col = sort_col.lower() if lowercase_cols else sort_col
        sort_ascending = sort_ascending if sort_ascending is not None else True
        df.sort_values(_sort_col, axis=0, ascending=sort_ascending, inplace=True)

    return df
