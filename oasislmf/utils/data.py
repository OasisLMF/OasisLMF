__all__ = [
    'factorize_array',
    'factorize_dataframe',
    'factorize_ndarray',
    'fast_zip_arrays',
    'fast_zip_dataframe_columns',
    'fill_na_with_categoricals',
    'get_dataframe',
    'get_exposure_data',
    'get_dtypes_and_required_cols',
    'get_ids',
    'get_json',
    'analysis_settings_loader',
    'model_settings_loader',
    'settings_loader',
    'prepare_location_df',
    'prepare_account_df',
    'prepare_reinsurance_df',
    'get_timestamp',
    'get_utctimestamp',
    'detect_encoding',
    'merge_check',
    'merge_dataframes',
    'print_dataframe',
    'PANDAS_BASIC_DTYPES',
    'PANDAS_DEFAULT_NULL_VALUES',
    'set_dataframe_column_dtypes',
    'RI_SCOPE_DEFAULTS',
    'RI_INFO_DEFAULTS',
    'validate_vuln_csv_contents',
    'validate_vulnerability_replacements',
]

import builtins
import io
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

from ods_tools.oed import fill_empty, OedExposure, OdsException, AnalysisSettingHandler, ModelSettingHandler

try:
    from json import JSONDecodeError
except ImportError:
    from builtins import ValueError as JSONDecodeError

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz
from chardet.universaldetector import UniversalDetector
from tabulate import tabulate

from oasislmf.utils.defaults import SOURCE_IDX, SAR_ID
from oasislmf.utils.exceptions import OasisException


logger = logging.getLogger(__name__)

analysis_settings_loader = AnalysisSettingHandler.make().load
model_settings_loader = ModelSettingHandler.make().load


def settings_loader(name, settings_json, loader, required=False, **kwargs):
    try:
        return loader(settings_json, **kwargs)
    except OdsException as e:
        if required:
            raise
        else:
            logger.debug(f"error loading {name}: {repr(e)}")
            return {}


pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

SUPPORTED_SRC_TYPE = ['parquet', 'csv', 'json']

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
    'bool': 'bool',
    builtins.bool: 'bool',
    'str': 'object',
    builtins.str: 'object',
    'category': 'category'
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
    '',
}

# Load schema json dir
SCHEMA_DATA_FP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'schema')

RI_INFO_DEFAULTS = {
    'CededPercent': 1.0,
    'RiskLimit': 0.0,
    'RiskAttachment': 0.0,
    'OccLimit': 0.0,
    'OccAttachment': 0.0,
    'TreatyShare': 1.0,
    'AttachmentBasis': 'LO',
    'ReinsInceptionDate': '',
    'ReinsExpiryDate': '',
}

RI_SCOPE_DEFAULTS = {
    'PortNumber': '',
    'AccNumber': '',
    'PolNumber': '',
    'LocGroup': '',
    'LocNumber': '',
    'CedantName': '',
    'ProducerName': '',
    'LOB': '',
    'CountryCode': '',
    'ReinsTag': '',
    'CededPercent': 1.0
}


def factorize_array(arr, sort_opt=False):
    """
    Groups a 1D Numpy array by item value, and optionally enumerates the
    groups, starting from 1. The default or assumed type is a Nunpy
    array, although a Python list, tuple or Pandas series will work too.

    :param arr: 1D Numpy array (or list, tuple, or Pandas series)
    :type arr: numpy.ndarray

    :return: A 2-tuple consisting of the enumeration and the value groups
    :rtype: tuple
    """
    enum, groups = pd.factorize(arr, sort=sort_opt)
    return enum + 1, groups


def factorize_ndarray(ndarr, row_idxs=[], col_idxs=[], sort_opt=False):
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

    enum, groups = pd.factorize(fast_zip_arrays(*(arr for arr in _ndarr)), sort=sort_opt)

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


def establish_correlations(model_settings: dict) -> bool:
    """
    Checks the model settings to see if correlations are present.

    Args:
        model_settings: (dict) the model settings that are going to be checked

    Returns: (bool) True if correlations, False if not
    """
    key = 'correlation_settings'
    correlations_legacy: Optional[List[dict]] = model_settings.get(key, [])
    correlations: Optional[List[dict]] = model_settings.get("model_settings", {}).get(key, correlations_legacy)

    if correlations is None:
        return False
    if not isinstance(correlations, list):
        return False
    if len(correlations) == 0:
        return False
    return True


def detect_encoding(filepath):
    """
    Given a path to a CSV of unknown encoding
    read lines to detects its encoding type

    :param filepath: Filepath to check
    :type  filepath: str

    :return: Example `{'encoding': 'ISO-8859-1', 'confidence': 0.73, 'language': ''}`
    :rtype: dict
    """

    detector = UniversalDetector()
    with io.open(filepath, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result


def get_dataframe(
        src_fp=None,
        src_type=None,
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
        memory_map=False,
        low_memory=False,
        encoding=None
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

    :param low_memory: Internally process the file in chunks, resulting in lower memory use
                       while parsing, but possibly mixed type inference.
                       To ensure no mixed types either set False,
    :type low_memory: bool

    :param encoding: Try to read CSV of JSON data with the given encoding type,
                     if 'None' will try to auto-detect on UnicodeDecodeError
    :type  encoding: str



    :return: A Pandas dataframe
    :rtype: pd.DataFrame
    """
    if not (src_fp or src_buf or src_data is not None):
        raise OasisException(
            'A CSV or JSON file path or a string buffer of such a file or an '
            'appropriate data structure or Pandas DataFrame must be provided'
        )

    if src_fp and src_type is None:
        try:
            src_type = src_fp.rsplit('.', 1)[1]
            if src_type not in SUPPORTED_SRC_TYPE:
                src_type = 'csv'
        except IndexError:
            src_type = 'csv'

    df = None

    # Use a custom list of null values without the string "`NA`" when using
    # pandas.read_csv because the default list contains the string "`NA`",
    # which can appear in the `CountryCode` column in the loc. file
    na_values = list(PANDAS_DEFAULT_NULL_VALUES.difference(['NA']))

    try:
        # memory map causes encoding errors with non-standard formats
        use_encoding = encoding if encoding else 'utf-8'
        memory_map = memory_map and (use_encoding == 'utf-8')

        if src_fp or src_buf:
            try:
                if src_type == 'csv':
                    # Find flexible fields in loc file and set their data types to that of
                    # FlexiLocZZZ
                    if 'FlexiLocZZZ' in col_dtypes.keys():
                        headers = list(pd.read_csv(src_fp, encoding=use_encoding, low_memory=low_memory).head(0))
                        for flexiloc_col in filter(re.compile('^FlexiLoc').match, headers):
                            col_dtypes[flexiloc_col] = col_dtypes['FlexiLocZZZ']
                    df = pd.read_csv(
                        src_fp or src_buf,
                        float_precision=float_precision,
                        memory_map=memory_map,
                        low_memory=low_memory,
                        keep_default_na=False,
                        na_values=na_values,
                        dtype=col_dtypes,
                        encoding=use_encoding,
                        quotechar='"',
                        skipinitialspace=True,
                    )
                elif src_type == 'parquet':
                    df = pd.read_parquet(src_fp or src_buf)
                elif src_type == 'json':
                    df = pd.read_json(
                        src_fp or src_buf,
                        precise_float=(True if float_precision == 'high' else False),
                        encoding=use_encoding
                    )
            except UnicodeDecodeError as e:
                raise e

            except (ValueError, OSError) as e:
                error_msg = f'Failed to load "{src_fp}", ' if src_fp else f'Failed to load "{src_buf}", '
                if empty_data_error_msg:
                    error_msg += empty_data_error_msg
                raise OasisException(error_msg, e)

        elif isinstance(src_data, list) and src_data:
            df = pd.DataFrame(data=src_data)
        elif isinstance(src_data, pd.DataFrame):
            df = src_data.copy(deep=True)

    # On DecodeError try to auto-detect the encoding and retry once
    except UnicodeDecodeError as e:
        detected_encoding = detect_encoding(src_fp)['encoding']
        if encoding is None and detected_encoding:
            return get_dataframe(
                src_fp=src_fp, src_type=src_type, src_buf=src_buf, src_data=src_data,
                float_precision=float_precision, empty_data_error_msg=empty_data_error_msg,
                lowercase_cols=lowercase_cols, required_cols=required_cols, col_defaults=col_defaults,
                non_na_cols=non_na_cols, col_dtypes=col_dtypes, sort_cols=sort_cols,
                sort_ascending=sort_ascending, memory_map=memory_map, low_memory=low_memory,
                encoding=detected_encoding)
        else:
            raise OasisException('Failed to load DataFrame due to Encoding error', e)

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

        fill_na_with_categoricals(df, _col_defaults)

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
            [(col.lower() if lowercase_cols else col) for col in sort_cols] if (
                isinstance(sort_cols, list) or isinstance(sort_cols, tuple) or isinstance(sort_cols, set))
            else (sort_cols.lower() if lowercase_cols else sort_cols)
        )
        sort_ascending = sort_ascending if sort_ascending is not None else True
        df.sort_values(_sort_cols, axis=0, ascending=sort_ascending, inplace=True, kind='stable')
    return df


def get_dtypes_and_required_cols(get_dtypes, all_dtypes=False):
    """
    Get OED column data types and required column names from JSON.

    :param all_dtypes: If true return every dtype field, otherwise only categoricals
    :type all_dtypes: boolean

    :param get_dtypes: method to get dict from JSON
    :type get_dtypes: function
    """
    dtypes = get_dtypes()

    if all_dtypes:
        col_dtypes = {k: v['py_dtype'].lower() for k, v in dtypes.items()}
    else:
        col_dtypes = {
            k: v['py_dtype'].lower() for k, v in dtypes.items() if v['py_dtype'] == 'category'
        }

    required_cols = [
        k for k, v in dtypes.items()
        if v['require_field'] == 'R'
    ]

    return col_dtypes, required_cols


def get_ids(df, usecols, group_by=[], sort_keys=True):
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

    :param sort_keys: Sort keys by value before assigning ids
    :param sort_keys: Boolean

        Example if sort_keys=True:
        -----------------
        index  PortNumber AccNumber    locnumbera  id (returned)
            0           1    A11111  10002082049    3
            1           1    A11111  10002082050    4
            2           1    A11111  10002082051    5
            3           1    A11111  10002082053    7
            4           1    A11111  10002082054    8
            5           1    A11111  10002082052    6
            6           1    A11111  10002082046    1
            7           1    A11111  10002082046    1
            8           1    A11111  10002082048    2
            9           1    A11111  10002082055    9

    :return: The enumeration
    :rtype: numpy.ndarray
    """
    _usecols = group_by + list(set(usecols).difference(group_by))

    if not group_by:
        if sort_keys:
            sorted_df = df.loc[:, usecols].sort_values(by=usecols, kind='stable')
            sorted_df['ids'] = factorize_ndarray(sorted_df.values, col_idxs=range(len(_usecols)))[0]
            return sorted_df.sort_index()['ids'].to_list()
        else:
            return factorize_ndarray(df.loc[:, usecols].values, col_idxs=range(len(_usecols)))[0]
    else:
        return (df[usecols].groupby(group_by, observed=True).cumcount()) + 1


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

    {'PortNumber': False, 'AccNumber': True, 'layer_id': True, 'condnumber': True}
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
        left = left.set_index(_join_on)
        right = right.drop(drop_cols, axis=1).set_index(_join_on)

        join = left.join(right, how=(kwargs.get('how') or 'left')).reset_index()

        return join


def prepare_oed_exposure(exposure_data):
    sar_df = exposure_data.get_subject_at_risk_source().dataframe
    if sar_df is not None:
        if SAR_ID not in sar_df.columns:
            sar_df[SAR_ID] = get_ids(sar_df, exposure_data.class_of_business_info['subject_at_risk_id_fields'])
        else:
            sar_df[SAR_ID] = sar_df[SAR_ID].astype(int)

    if exposure_data.location:
        exposure_data.location.dataframe = prepare_location_df(exposure_data.location.dataframe)
    if exposure_data.account:
        exposure_data.account.dataframe = prepare_account_df(exposure_data.account.dataframe)
    if exposure_data.ri_info and exposure_data.ri_scope:
        exposure_data.ri_info.dataframe, exposure_data.ri_scope.dataframe = prepare_reinsurance_df(
            exposure_data.ri_info.dataframe,
            exposure_data.ri_scope.dataframe)


def prepare_location_df(location_df):
    # Add file Index column to extract OED columns for summary grouping
    location_df[SOURCE_IDX['loc']] = location_df.index
    return location_df


def prepare_account_df(accounts_df):
    if SOURCE_IDX['acc'] not in accounts_df.columns:
        accounts_df[SOURCE_IDX['acc']] = accounts_df.index
    else:
        accounts_df[SOURCE_IDX['acc']] = accounts_df[SOURCE_IDX['acc']].astype(int)
    if 'LayerNumber' not in accounts_df:
        accounts_df['LayerNumber'] = 1
    accounts_df['LayerNumber'] = accounts_df['LayerNumber'].fillna(1)

    # Determine whether step policies are listed, are not full of nans and step
    # numbers are greater than zero
    step_policies_present = False
    if 'StepTriggerType' in accounts_df and 'StepNumber' in accounts_df:
        if accounts_df['StepTriggerType'].notnull().any():
            if accounts_df[accounts_df['StepTriggerType'].notnull()]['StepNumber'].gt(0).any():
                step_policies_present = True

    # Determine whether layer num. column exists in the accounts dataframe and
    # create it if needed, filling it with default value. The layer num. field
    # is used to identify unique layers in cases where layers share the same
    # policy num.
    # Create `layer_id` column, which is simply an enumeration of the unique
    # (portfolio_num., acc. num., policy num., layer num.) combinations in the
    # accounts file.
    # If step policies are listed use `StepNumber` column in combination
    layers_cols = ['PortNumber', 'AccNumber']
    if step_policies_present:
        layers_cols += ['StepNumber']
        accounts_df['StepNumber'] = accounts_df['StepNumber'].fillna(0)
    if 'layer_id' not in accounts_df.columns:
        id_df = accounts_df[layers_cols + ['PolNumber', 'LayerNumber']].drop_duplicates(keep='first')
        id_df['layer_id'] = get_ids(id_df,
                                    layers_cols + ['PolNumber', 'LayerNumber'], group_by=layers_cols,
                                    ).astype('uint32')
        accounts_df = merge_dataframes(accounts_df, id_df, join_on=layers_cols + ['PolNumber', 'LayerNumber'])
    else:
        accounts_df['layer_id'] = accounts_df['layer_id'].astype('uint32')
    return accounts_df


def prepare_reinsurance_df(ri_info, ri_scope):
    fill_empty(ri_info, 'RiskLevel', 'SEL')

    # add default column if not present in the RI files
    fill_na_with_categoricals(ri_info, RI_INFO_DEFAULTS)
    for column in set(RI_INFO_DEFAULTS).difference(ri_info.columns):
        ri_info[column] = RI_INFO_DEFAULTS[column]

    fill_na_with_categoricals(ri_scope, RI_SCOPE_DEFAULTS)
    for column in set(RI_SCOPE_DEFAULTS).difference(ri_scope.columns):
        ri_scope[column] = RI_SCOPE_DEFAULTS[column]

    return ri_info, ri_scope


def get_exposure_data(computation_step, add_internal_col=False):
    try:
        if 'exposure_data' in computation_step.kwargs:
            logger.debug("Exposure data found in `exposure_data` key of computation step kwargs")
            exposure_data = computation_step.kwargs['exposure_data']
        else:
            if hasattr(computation_step, 'oasis_files_dir') and Path(computation_step.oasis_files_dir, OedExposure.DEFAULT_EXPOSURE_CONFIG_NAME).is_file():
                logger.debug(f"Exposure data is read from {Path(computation_step.oasis_files_dir, OedExposure.DEFAULT_EXPOSURE_CONFIG_NAME)}")
                exposure_data = OedExposure.from_config(Path(computation_step.oasis_files_dir, OedExposure.DEFAULT_EXPOSURE_CONFIG_NAME))
            elif hasattr(computation_step, 'get_exposure_data_config'):  # if computation step input specify ExposureData config
                logger.debug("Exposure data is generated from `get_exposure_data_config` key of computation kwargs")
                exposure_data = OedExposure(**computation_step.get_exposure_data_config())
            else:
                logger.debug("ExposureData info was not created, oed input file must have default name (location, account, ...)")
                exposure_data = OedExposure.from_dir(
                    computation_step.oasis_files_dir,
                    oed_schema_info=getattr(computation_step, 'oed_schema_info', None),
                    currency_conversion=getattr(computation_step, 'currency_conversion_json', None),
                    reporting_currency=getattr(computation_step, 'reporting_currency', None),
                    check_oed=computation_step.check_oed,
                    use_field=True)

            if add_internal_col:
                prepare_oed_exposure(exposure_data)
        return exposure_data
    except OdsException as ods_error:
        raise OasisException("Failed to load OED exposure files", ods_error)


def print_dataframe(
        df,
        cols=[],
        string_cols=[],
        show_index=False,
        frame_header=None,
        column_headers='keys',
        tablefmt='psql',
        floatfmt=",.2f",
        end='\n',
        **tabulate_kwargs
):
    """
    A method to pretty-print a Pandas dataframe - calls on the ``tabulate``
    package

    :param df: The dataframe to pretty-print
    :type df: pd.DataFrame

    :param cols: An iterable of names of columns whose values should
                           be printed (optional). If unset, all columns will be printed.
    :type cols: list, tuple, collections.Iterable

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

    if cols is not None and len(cols) > 0:
        _df = _df[cols]

    for col in string_cols:
        _df[col] = _df[col].astype(object)

    if frame_header:
        print('\n{}'.format(frame_header))

    if tabulate_kwargs:
        tabulate_kwargs.pop('headers') if 'headers' in tabulate_kwargs else None
        tabulate_kwargs.pop('tablefmt') if 'tablefmt' in tabulate_kwargs else None
        tabulate_kwargs.pop('floatfmt') if 'floatfmt' in tabulate_kwargs else None
        tabulate_kwargs.pop('showindex') if 'showindex' in tabulate_kwargs else None

    print(
        tabulate(
            _df, headers=column_headers, tablefmt=tablefmt,
            showindex=show_index, floatfmt=floatfmt, **tabulate_kwargs),
        end=end)


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


def validate_vuln_csv_contents(file_path):
    """
    Validate the contents of the CSV file for vulnerability replacements.

    Args:
        file_path (str): Path to the vulnerability CSV file

    Returns:
        bool: True if the file is valid, False otherwise
    """
    expected_columns = ['vulnerability_id', 'intensity_bin_id', 'damage_bin_id', 'probability']
    try:
        vuln_df = pd.read_csv(file_path)
        if list(vuln_df.columns) != expected_columns:
            logger.warning(f"CSV file {file_path} does not have the expected columns.")
            return False

        # Check data types and constraints
        if not (
            np.issubdtype(vuln_df['vulnerability_id'].dtype, np.integer)
            and np.issubdtype(vuln_df['intensity_bin_id'].dtype, np.integer)
            and np.issubdtype(vuln_df['damage_bin_id'].dtype, np.integer)
        ):
            logger.warning("vulnerability_id, intensity_bin_id, and damage_bin_id columns must contain integer values.")
            return False
        if vuln_df['probability'].apply(lambda x: isinstance(x, (int, float))).all():
            if not (vuln_df['probability'].between(0, 1).all()):
                logger.warning("probability column must contain values between 0 and 1.")
                return False
        else:
            logger.warning("probability column must contain numeric values.")
            return False
        return True
    except Exception as e:
        # No fail if the file is not valid, just warn the user
        logger.warning(f"Error occurred while validating CSV file: {e}")
        return False


def validate_vulnerability_replacements(analysis_settings_json):
    """
    Validate vulnerability replacements in analysis settings file.
    If vulnerability replacements are specified as a file path, check that the file exists.
    This way the user will be warned early if the vulnerability option selected is not valid.

    Args:
        analysis_settings_json (str): JSON file path to analysis settings file

    Returns:
        bool: True if the vulnerability replacements are present and valid, False otherwise

    """
    if analysis_settings_json is None:
        return False

    vulnerability_adjustments_key = analysis_settings_loader(analysis_settings_json).get('vulnerability_adjustments')
    if vulnerability_adjustments_key is None:
        return False

    vulnerability_replacements = vulnerability_adjustments_key.get('replace_data', None)
    if vulnerability_replacements is None:
        vulnerability_replacements = vulnerability_adjustments_key.get('replace_file', None)
    if vulnerability_replacements is None:
        return False
    if isinstance(vulnerability_replacements, dict):
        logger.info('Vulnerability replacements are specified in the analysis settings file')
        return True
    if isinstance(vulnerability_replacements, str):
        abs_path = os.path.abspath(vulnerability_replacements)
        if not os.path.isfile(abs_path):
            logger.warning('Vulnerability replacements file does not exist: {}'.format(abs_path))
            return False

        if not validate_vuln_csv_contents(abs_path):
            logger.warning('Vulnerability replacements file is not valid: {}'.format(abs_path))
            return False

        logger.info('Vulnerability replacements found in file: {}'.format(abs_path))
        return True
    logger.warning('Vulnerability replacements must be a dict or a file path, got: {}'.format(vulnerability_replacements))
    return False


def fill_na_with_categoricals(df, fill_value):
    """
    Fill NA values in a Pandas DataFrame, with handling for Categorical dtype columns.

    The input dataframe is modified inplace.

    :param df: The dataframe to process
    :type df: pd.DataFrame

    :param fill_value: A single value to use in all columns, or a dict of column names and
                       corresponding values to fill.
    :type fill_value: int, float, str, dict

    """
    if not isinstance(fill_value, dict):
        fill_value = {col_name: fill_value for col_name in df.columns}

    for col_name, value in fill_value.items():
        if col_name not in df:
            continue

        col = df[col_name]
        if pd.api.types.is_categorical_dtype(col):
            # Force to be a string - using categorical for string columns
            value = str(value)
            fill_value[col_name] = value
            if value not in col.cat.categories:
                df[col_name] = col.cat.add_categories([value])

    # Note that the following lines do not work properly with Pandas 1.1.0/1.1.1, due to a bug
    # related to fillna and categorical dtypes. This bug should be fixed in >1.1.2.
    # https://github.com/pandas-dev/pandas/issues/35731
    df.fillna(value=fill_value, inplace=True)
