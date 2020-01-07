__all__ = [
    'factorize_array',
    'factorize_dataframe',
    'factorize_ndarray',
    'fast_zip_arrays',
    'fast_zip_dataframe_columns',
    'get_dataframe',
    'get_dtypes_and_required_cols',
    'get_ids',
    'get_json',
    'get_timestamp',
    'get_utctimestamp',
    'merge_check',
    'merge_dataframes',
    'PANDAS_BASIC_DTYPES',
    'PANDAS_DEFAULT_NULL_VALUES',
    'set_dataframe_column_dtypes'
]

import builtins
import io
import json
import re
import warnings

from datetime import datetime
from itertools import groupby

try:
    from json import JSONDecodeError
except ImportError:
    from builtins import ValueError as JSONDecodeError

from tabulate import tabulate

import numpy as np
import pandas as pd
import pytz

from .exceptions import OasisException

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

PANDAS_BASIC_DTYPES = {
    'int8': np.int8,
    'uint8': np.uint8,
    'int16': np.int16,
    'uint16': np.uint16,
    'int32': np.int32,
    'uint32': np.uint32,
    'int64': np.int64,
    'uint64': np.uint64,
    builtins.int: np.int64,
    'float32': np.float32,
    'float64': np.float64,
    builtins.float: np.float64,
    'bool': np.bool,
    builtins.bool: np.bool,
    'str': np.object,
    builtins.str: np.object
}

PANDAS_DEFAULT_NULL_VALUES = {
    '-1.#IND',
    '1.#QNAN',
    '1.#IND',
    '-1.#QNAN',
    '#N/A N/A',
    '#N/A',
    'N/A',
    'n/a',
    'NA',
    '#NA',
    'NULL',
    'null',
    'NaN',
    '-NaN',
    'nan',
    '-nan',
    ''
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

    :return: A Numpy 1D array of n-tuples of the zipped sequences
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

    :param cols: An iterable or iterator or generator of Pandas DataFrame columns
    :type cols: list, tuple, collections.Iterator, types.GeneratorType

    :return: A Numpy 1D array of n-tuples of the dataframe columns to be zipped
    :rtype: np.array
    """
    return fast_zip_arrays(*(df[col].values for col in cols))


def get_dataframe(
    src_fp=None,
    src_type='csv',
    src_buf=None,
    src_data=None,
    float_precision='high',
    empty_data_error_msg=None,
    lowercase_cols=True,
    required_cols=(),
    col_defaults={},
    non_na_cols=(),
    col_dtypes={},
    sort_cols=None,
    sort_ascending=None,
    memory_map=False
):
    """
    Loads a Pandas dataframe from a source CSV or JSON file, or a text buffer
    of such a file (``io.StringIO``), or another Pandas dataframe.

    :param src_fp: Source CSV or JSON file path (optional)
    :type src_fp: str

    :param src_type: Type of source file -CSV or JSON (optional; default is csv)
    :param src_type: str

    :param src_buf: Text buffer of a source CSV or JSON file (optional)
    :type src_buf: io.StringIO

    :param float_precision: Indicates whether to support high-precision numbers
                            present in the data (optional; default is high)
    :type float_precision: str

    :param empty_data_error_msg: The message of the exception that is thrown
                                there is no data content, i.e no rows
                                (optional)
    :type empty_data_error_msg: str

    :param lowercase_cols: Whether to convert the dataframe columns to lowercase
                           (optional; default is True)
    :type lowercase_cols: bool

    :param required_cols: An iterable of columns required to be present in the
                          source data (optional)
    :type required_cols: list, tuple, collections.Iterable

    :param col_defaults: A dict of column names and their default values. This
                         can include both existing columns and new columns -
                         defaults for existing columns are set row-wise using
                         pd.DataFrame.fillna, while defaults for non-existent
                         columns are set column-wise using assignment (optional)
    :type col_defaults: dict

    :param non_na_cols: An iterable of names of columns which must be dropped
                        if they contain any null values (optional)
    :type non_na_cols: list, tuple, collections.Iterable

    :param col_dtypes: A dict of column names and corresponding data types -
                       Python built-in datatypes are accepted but are mapped
                       to the corresponding Numpy datatypes (optional)
    :type col_dtypes: dict

    :param sort_cols: An iterable of column names by which to sort the frame
                      rows (optional)
    :type sort_cols: list, tuple, collections.Iterable

    :param sort_ascending: Whether to perform an ascending or descending sort -
                           is used only in conjunction with the sort_cols
                           option (optional)
    :type sort_ascending: bool

    :param memory_map: Memory-efficient option used when loading a frame from
                       a file or text buffer - is a direct optional argument
                       for the pd.read_csv method
    :type memory_map: bool

    :return: A Pandas dataframe
    :rtype: pd.DataFrame
    """
    if not (src_fp or src_buf or src_data is not None):
        raise OasisException(
            'A CSV or JSON file path or a string buffer of such a file or an '
            'appropriate data structure or Pandas DataFrame must be provided'
        )

    df = None

    # Use a custom list of null values without the string "`NA`" when using
    # pandas.read_csv because the default list contains the string "`NA`",
    # which can appear in the `CountryCode` column in the loc. file
    na_values = list(PANDAS_DEFAULT_NULL_VALUES.difference(['NA']))

    if src_fp and src_type == 'csv':
        # Find flexible fields in loc file and set their data types to that of
        # FlexiLocZZZ
        if 'FlexiLocZZZ' in col_dtypes.keys():
            headers = list(pd.read_csv(src_fp).head(0))
            for flexiloc_col in filter(re.compile('^FlexiLoc').match, headers):
                col_dtypes[flexiloc_col] = col_dtypes['FlexiLocZZZ']
        df = pd.read_csv(src_fp, float_precision=float_precision, memory_map=memory_map, keep_default_na=False, na_values=na_values, dtype=col_dtypes)
    elif src_buf and src_type == 'csv':
        df = pd.read_csv(io.StringIO(src_buf), float_precision=float_precision, memory_map=memory_map, keep_default_na=False, na_values=na_values)
    elif src_fp and src_type == 'json':
        df = pd.read_json(src_fp, precise_float=(True if float_precision == 'high' else False))
    elif src_buf and src_type == 'json':
        df = pd.read_json(io.StringIO(src_buf), precise_float=(True if float_precision == 'high' else False))
    elif isinstance(src_data, list) and src_data:
        df = pd.DataFrame(data=src_data)
    elif isinstance(src_data, pd.DataFrame):
        df = src_data.copy(deep=True)

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
        # Lowercase the keys in the defaults dict depending on whether the `lowercase_cols`
        # option was passed
        _col_defaults = {k.lower(): v for k, v in col_defaults.items()} if lowercase_cols else col_defaults

        # Use the defaults dict to set defaults for existing columns
        df.fillna(value=_col_defaults, inplace=True)

        # A separate step to set as yet non-existent columns with default values
        # in the frame
        new = {k: _col_defaults[k] for k in set(_col_defaults).difference(df.columns)}
        df = df.join(pd.DataFrame(data=new, index=df.index))

    if non_na_cols:
        _non_na_cols = tuple(col.lower() for col in non_na_cols) if lowercase_cols else non_na_cols
        df.dropna(subset=_non_na_cols, inplace=True)

    if col_dtypes:
        _col_dtypes = {k.lower(): v for k, v in col_dtypes.items()} if lowercase_cols else col_dtypes
        df = set_dataframe_column_dtypes(df, _col_dtypes)

    if sort_cols:
        _sort_cols = (
            [(col.lower() if lowercase_cols else col) for col in sort_cols] if (isinstance(sort_cols, list) or isinstance(sort_cols, tuple) or isinstance(sort_cols, set))
            else (sort_cols.lower() if lowercase_cols else sort_cols)
        )
        sort_ascending = sort_ascending if sort_ascending is not None else True
        df.sort_values(_sort_cols, axis=0, ascending=sort_ascending, inplace=True)

    return df


def get_dtypes_and_required_cols(get_dtypes):
    """
    Get OED column data types and required column names from JSON.

    :param get_dtypes: method to get dict from JSON
    :type get_dtypes: function
    """
    dtypes = get_dtypes()
    col_dtypes = {
        k: v['py_dtype']
        for k, v in dtypes.items()
        if v['py_dtype'] == 'str'
    }
    required_cols = [
        k for k, v in dtypes.items()
        if v['py_dtype'] == 'str'
        if v['require_field'] == 'R'
    ]

    return col_dtypes, required_cols


def get_ids(df, usecols, group_by=[]):
    """
    Enumerates (counts) the rows of a given dataframe in a given subset
    of dataframe columns, and optionally does the enumeration with
    respect to subgroups of the column subset.

    :param df: Input dataframe
    :type df: pandas.DataFrame

    :param usecols: The column subset
    :param usecols: list

    :param group_by: A subset of the column subset to use a subgroup key
    :param group_by: list

    :return: The enumeration
    :rtype: numpy.ndarray
    """
    _usecols = group_by + list(set(usecols).difference(group_by))

    if not group_by:
        return factorize_ndarray(df.loc[:, usecols].values, col_idxs=range(len(_usecols)))[0]
    else:
        return (df[usecols].groupby(group_by).cumcount()) + 1


def get_json(src_fp):
    """
    Loads JSON from file.

    :param src_fp: Source JSON file path
    :type src_fp: str

    :return: dict
    :rtype: dict
    """
    try:
        with io.open(src_fp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, JSONDecodeError, OSError, TypeError):
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


def merge_check(left, right, on=[], raise_error=True):
    """
    Check two dataframes for keys intersection, use before performing a merge

    :param left: The first of two dataframes to be merged
    :type left: pd.DataFrame

    :param right: The second of two dataframes to be merged
    :type left: pd.DataFrame

    :param on: column keys to test
    :type on: list

    :return: A dict of booleans, True for an intersection between left/right
    :rtype: dict

    {'portnumber': False, 'accnumber': True, 'layer_id': True, 'condnumber': True}
    """
    keys_checked = {}
    for key in on:
        key_intersect = set(left[key].unique()).intersection(right[key].unique())
        keys_checked[key] = bool(key_intersect)

    if raise_error and not all(keys_checked.values()):
        err_msg = "Error: Merge mismatch on column(s) {}".format(
            [k for k in keys_checked if not keys_checked[k]]
        )
        raise OasisException(err_msg)


def merge_dataframes(left, right, join_on=None, **kwargs):
    """
    Merges two dataframes by ensuring there is no duplication of columns.

    :param left: The first of two dataframes to be merged
    :type left: pd.DataFrame

    :param right: The second of two dataframes to be merged
    :type left: pd.DataFrame

    :param kwargs: Optional keyword arguments passed directly to the underlying
                   pd.merge method that is called, including options for the
                   join keys, join type, etc. - please see the pd.merge
                   documentation for details of these optional arguments
    :type kwargs: dict

    :return: A merged dataframe
    :rtype: pd.DataFrame
    """
    merge = None

    if not join_on:
        left_keys = kwargs.get('left_on') or kwargs.get('on') or []
        left_keys = [left_keys] if isinstance(left_keys, str) else left_keys

        drop_cols = [
            k for k in set(left.columns).intersection(right.columns)
            if k and k not in left_keys
        ]
        drop_duplicates = kwargs.get('drop_duplicates', True)
        kwargs.pop('drop_duplicates') if 'drop_duplicates' in kwargs else None

        merge = pd.merge(
            left.drop(drop_cols, axis=1),
            right,
            **kwargs
        )

        return merge if not drop_duplicates else merge.drop_duplicates()
    else:
        _join_on = [join_on] if isinstance(join_on, str) else join_on.copy()
        drop_cols = list(set(left.columns).intersection(right.columns).difference(_join_on))
        _left = left.set_index(_join_on)
        _right = right.drop(drop_cols, axis=1).set_index(_join_on)

        join = _left.join(_right, how=(kwargs.get('how') or 'left')).reset_index()
        del _left, _right

        return join


def print_dataframe(
    df,
    string_cols=[],
    show_index=False,
    frame_header=None,
    column_headers='keys',
    tablefmt='psql',
    floatfmt=".2f",
    end='\n',
    **tabulate_kwargs
):
    """
    A method to pretty-print a Pandas dataframe - calls on the ``tabulate``
    package

    :param df: The dataframe to pretty-print
    :type df: pd.DataFrame

    :param string_cols: An iterable of names of columns whose values should
                           be treated as strings (optional)
    :type string_cols: list, tuple, collections.Iterable

    :param show_index: Whether to display the index column in the printout
                       (optional; default is False)
    :type show_index: bool

    :param frame_header: Header string to display on top of the printed
                         dataframe (optional)
    :type frame_header: str

    :param column_headers: Column header format - see the tabulate.tabulate
                        method documentation (optional, default is 'keys')
    :type column_headers: list, str

    :param tablefmt: Table format - see the tabulate.tabulate method
                     documentation (optional; default is 'psql')
    :type tablefmt: str, list, tuple

    :param floatfmt: Floating point format - see the tabulate.tabulate
                    method documnetation (optional; default is ".2f")
    :type floatfmt: str

    :param end: String to append after printing the dataframe
                (optional; default is newline)
    :type end: str

    :param tabulate_kwargs: Additional optional arguments passed directly to
                            the underlying tabulate.tabulate method - see the
                            method documentation for more details
    :param tabulate_kwargs: dict
    """
    _df = df.copy(deep=True)

    for col in string_cols:
        _df[col] = _df[col].astype(object)

    if frame_header:
        print('\n{}'.format(frame_header))

    if tabulate_kwargs:
        tabulate_kwargs.pop('headers') if 'headers' in tabulate_kwargs else None
        tabulate_kwargs.pop('tablefmt') if 'tablefmt' in tabulate_kwargs else None
        tabulate_kwargs.pop('floatfmt') if 'floatfmt' in tabulate_kwargs else None
        tabulate_kwargs.pop('showindex') if 'showindex' in tabulate_kwargs else None

    print(tabulate(_df, headers=column_headers, tablefmt=tablefmt, showindex=show_index, floatfmt=floatfmt, **tabulate_kwargs), end=end)


def set_dataframe_column_dtypes(df, dtypes):
    """
    A method to set column datatypes for a Pandas dataframe

    :param df: The dataframe to process
    :type df: pd.DataFrame

    :param dtypes: A dict of column names and corresponding Numpy datatypes -
                   Python built-in datatypes can be passed in but they will be
                   mapped to the corresponding Numpy datatypes
    :type dtypes: dict

    :return: The processed dataframe with column datatypes set
    :rtype: pandas.DataFrame
    """
    existing_cols = list(set(dtypes).intersection(df.columns))
    _dtypes = {
        col: PANDAS_BASIC_DTYPES[getattr(builtins, dtype) if dtype in ('int', 'bool', 'float', 'object', 'str',) else dtype]
        for col, dtype in [(_col, dtypes[_col]) for _col in existing_cols]
    }
    df = df.astype(_dtypes)

    return df
