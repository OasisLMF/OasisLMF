# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

import copy
import io
import itertools
import json
import os
import shutil
import string
import sys

from collections import OrderedDict
from datetime import datetime
from future.utils import viewitems, viewvalues
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
import pytz

from backports.tempfile import TemporaryDirectory
from hypothesis import (
    given,
    HealthCheck,
    reproduce_failure,
    settings,
)
from hypothesis.strategies import (
    characters,
    data,
    datetimes,
    integers,
    fixed_dictionaries,
    floats,
    just,
    lists,
    sampled_from,
    text,
    tuples,
)
from pandas.util.testing import assert_frame_equal
from tempfile import NamedTemporaryFile

from oasislmf.utils.data import (
    factorize_array,
    factorize_ndarray,
    fast_zip_arrays,
    get_dataframe,
    get_timestamp,
    get_utctimestamp,
)
from oasislmf.utils.exceptions import OasisException


def arrays_are_identical(expected, result):
    try:
        np.testing.assert_array_equal(expected, result)
    except AssertionError:
        raise

    return True


class TestFactorizeArrays(TestCase):

    @settings(max_examples=10)
    @given(
        num_chars=integers(min_value=2, max_value=len(string.ascii_lowercase + string.digits)),
        str_len=integers(min_value=2, max_value=100),
        num_strs=integers(min_value=10, max_value=100)
    )
    def test_factorize_1darray(self, num_chars, str_len, num_strs):
        alphabet = np.random.choice(list(string.ascii_lowercase + string.digits), size=num_chars)
        strings = [''.join([np.random.choice(alphabet) for i in range(str_len)]) for j in range(num_strs)]

        expected_groups = list(OrderedDict({s: s for s in strings}))
        expected_enum = np.array([expected_groups.index(s) + 1 for s in strings])

        result_enum, result_groups  = factorize_array(strings)

        self.assertTrue(arrays_are_identical(expected_groups, result_groups))
        self.assertTrue(arrays_are_identical(expected_enum, result_enum))

    @settings(max_examples=1)
    @given(
        num_chars=integers(min_value=2, max_value=len(string.ascii_lowercase + string.digits)),
        str_len=integers(min_value=2, max_value=100),
        rows=integers(min_value=10, max_value=100),
        cols=integers(min_value=10, max_value=100)
    )
    def test_factorize_ndarray__no_row_or_col_indices_provided__raises_oasis_exception(self, num_chars, str_len, rows, cols):
        alphabet = np.random.choice(list(string.ascii_lowercase + string.digits), size=num_chars)
        strings = [''.join([np.random.choice(alphabet) for i in range(str_len)]) for j in range(rows * cols)]

        ndarr = np.random.choice(strings, (rows, cols))

        with self.assertRaises(OasisException):
            factorize_ndarray(ndarr)

    @settings(max_examples=10, deadline=None)
    @given(
        num_chars=integers(min_value=2, max_value=len(string.ascii_lowercase + string.digits)),
        str_len=integers(min_value=2, max_value=100),
        rows=integers(min_value=10, max_value=100),
        cols=integers(min_value=10, max_value=100),
        num_row_idxs=integers(min_value=2, max_value=10)
    )
    def test_factorize_ndarray__by_row_idxs(self, num_chars, str_len, rows, cols, num_row_idxs):
        alphabet = np.random.choice(list(string.ascii_lowercase + string.digits), size=num_chars)
        strings = [''.join([np.random.choice(alphabet) for i in range(str_len)]) for j in range(rows * cols)]

        ndarr = np.random.choice(strings, (rows, cols))

        row_idxs = np.random.choice(range(rows), num_row_idxs, replace=False).tolist()

        zipped = list(zip(*(ndarr[i, :] for i in row_idxs)))
        groups = list(OrderedDict({x: x for x in zipped}))
        expected_groups = np.empty(len(groups), dtype=object)
        expected_groups[:] = groups
        expected_enum = np.array([groups.index(x) + 1 for x in zipped])

        result_enum, result_groups  = factorize_ndarray(ndarr, row_idxs=row_idxs)

        self.assertTrue(arrays_are_identical(expected_groups, result_groups))
        self.assertTrue(arrays_are_identical(expected_enum, result_enum))

    @settings(max_examples=10, deadline=None)
    @given(
        num_chars=integers(min_value=2, max_value=len(string.ascii_lowercase + string.digits)),
        str_len=integers(min_value=2, max_value=100),
        rows=integers(min_value=10, max_value=100),
        cols=integers(min_value=10, max_value=100),
        num_col_idxs=integers(min_value=2, max_value=10)
    )
    def test_factorize_ndarray__by_col_idxs(self, num_chars, str_len, rows, cols, num_col_idxs):
        alphabet = np.random.choice(list(string.ascii_lowercase + string.digits), size=num_chars)
        strings = [''.join([np.random.choice(alphabet) for i in range(str_len)]) for j in range(rows * cols)]

        ndarr = np.random.choice(strings, (rows, cols))

        col_idxs = np.random.choice(range(cols), num_col_idxs, replace=False).tolist()

        zipped = list(zip(*(ndarr[:, i] for i in col_idxs)))
        groups = list(OrderedDict({x: x for x in zipped}))
        expected_groups = np.empty(len(groups), dtype=object)
        expected_groups[:] = groups
        expected_enum = np.array([groups.index(x) + 1 for x in zipped])

        result_enum, result_groups = factorize_ndarray(ndarr, col_idxs=col_idxs)

        self.assertTrue(arrays_are_identical(expected_groups, result_groups))
        self.assertTrue(arrays_are_identical(expected_enum, result_enum))


class TestFastZipArrays(TestCase):

    @settings(max_examples=10)
    @given(
        array_len=integers(min_value=10, max_value=100),
        num_arrays=integers(2, 100)
    )
    def test_fast_zip_arrays(self, array_len, num_arrays):
        arrays = np.random.randint(1, 10**6, (num_arrays, array_len))

        li = list(zip(*arrays))
        zipped = np.empty(len(li), dtype=object)
        zipped[:] = li

        result = fast_zip_arrays(*arrays)

        self.assertTrue(arrays_are_identical(zipped, result))


def dataframes_are_identical(df1, df2):
    try:
        assert_frame_equal(df1, df2)
    except:
        return False

    return True


def set_method_name(name):
    def wrapper(f):
        f.__name__ = name
        f.__qualname__ = ''.join(f.__qualname__.split('.')[:-1] + ['.', name])
        return f
    return wrapper


class TestGetDataframe(TestCase):

    def test_get_dataframe__no_src_fp_or_buf_or_data_provided__oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            get_dataframe(src_fp=None, src_buf=None, src_data=None)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file__use_default_options(self, data):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)

            result = get_dataframe(src_fp=fp.name)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__use_default_options(self, data):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)
            expected.columns = expected.columns.str.lower()

            result = get_dataframe(src_fp=fp.name)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        dtypes=fixed_dictionaries({
            'int_col': sampled_from(['int32', 'int64']),
            'float_col': sampled_from(['float32', 'float64'])
        })
    )
    def test_get_dataframe__from_csv_file__set_col_dtypes_option_and_use_defaults_for_all_other_options(self, data, dtypes):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            for col, dtype in viewitems(dtypes):
                df[col] = df[col].astype(dtype)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = pd.read_csv(fp.name, dtype=dtypes)

            result = get_dataframe(src_fp=fp.name, col_dtypes=dtypes)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'INT_COL': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        dtypes=fixed_dictionaries({
            'INT_COL': sampled_from(['int32', 'int64']),
            'FloatCol': sampled_from(['float32', 'float64'])
        })
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_col_dtypes_option_and_use_defaults_for_all_other_options(self, data, dtypes):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            for col, dtype in viewitems(dtypes):
                df[col] = df[col].astype(dtype)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = pd.read_csv(fp.name, dtype=dtypes)
            expected.columns = expected.columns.str.lower()

            result = get_dataframe(src_fp=fp.name, col_dtypes=dtypes)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        subset_cols=just(
            np.random.choice(
                ['str_col','int_col','float_col','bool_col','null_col'],
                np.random.choice(range(1, 6)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file__set_subset_cols_option_and_use_defaults_for_all_other_options(self, data, subset_cols):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.drop([col for col in df.columns if col not in subset_cols], axis=1)

            result = get_dataframe(
                src_fp=fp.name,
                subset_cols=subset_cols
            )

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        subset_cols=just(
            np.random.choice(
                ['STR_COL','int_col','FloatCol','boolCol','null_col'],
                np.random.choice(range(1, 6)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_subset_cols_option_and_use_defaults_for_all_other_options(self, data, subset_cols):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.drop([col for col in df.columns if col not in subset_cols], axis=1)
            expected.columns = expected.columns.str.lower()

            result = get_dataframe(
                src_fp=fp.name,
                subset_cols=subset_cols
            )

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(empty_data_err_msg=text(min_size=1, max_size=10, alphabet=string.ascii_lowercase))
    def test_get_dataframe__from_empty_csv_file__set_empty_data_err_msg_and_defaults_for_all_other_options__oasis_exception_is_raised_with_empty_data_err_msg(self, empty_data_err_msg):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame()
            df.to_csv(path_or_buf=fp)
            fp.close()

            with self.assertRaises(OasisException):
                try:
                    get_dataframe(src_fp=fp.name, empty_data_error_msg=empty_data_err_msg)
                except OasisException as e:
                    self.assertEqual(str(e), empty_data_err_msg)
                    raise e
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        required_cols=just(
            np.random.choice(
                ['str_col','int_col','float_col','bool_col','null_col'],
                np.random.choice(range(1, 6)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file__set_required_cols_option_and_use_defaults_for_all_other_options(self, data, required_cols):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)

            result = get_dataframe(
                src_fp=fp.name,
                required_cols=required_cols
            )

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        required_cols=just(
            np.random.choice(
                ['STR_COL','int_col','FloatCol','boolCol','null_col'],
                np.random.choice(range(1, 6)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_required_cols_option_and_use_defaults_for_all_other_options(self, data, required_cols):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)
            expected.columns = expected.columns.str.lower()

            result = get_dataframe(
                src_fp=fp.name,
                required_cols=required_cols
            )

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        missing_cols=just(
            np.random.choice(
                ['str_col','int_col','float_col','bool_col','null_col'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_missing_some_required_cols__set_required_cols_option_and_use_defaults_for_all_other_options__oasis_exception_is_raised(self, data, missing_cols):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.drop(missing_cols, axis=1).to_csv(path_or_buf=fp, encoding='utf-8', index=False)
            fp.close()

            with self.assertRaises(OasisException):
                get_dataframe(
                    src_fp=fp.name,
                    required_cols=df.columns.tolist()
                )
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        missing_cols=just(
            np.random.choice(
                ['STR_COL','int_col','FloatCol','boolCol','null_col'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols_and_missing_some_required_cols__set_required_cols_option_and_use_defaults_for_all_other_options__oasis_exception_is_raised(self, data, missing_cols):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.drop(missing_cols, axis=1).to_csv(path_or_buf=fp, encoding='utf-8', index=False)
            fp.close()

            with self.assertRaises(OasisException):
                get_dataframe(
                    src_fp=fp.name,
                    required_cols=df.columns.tolist()
                )
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        defaults=fixed_dictionaries({
            'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_uppercase),
            'int_col': integers(min_value=0, max_value=10),
            'float_col': floats(min_value=1.0, allow_infinity=False)
        })
    )
    def test_get_dataframe__from_csv_file__set_col_defaults_option_and_use_defaults_for_all_other_options(self, data, defaults):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)
            for col, default in viewitems(defaults):
                expected.loc[:, col].fillna(defaults[col], inplace=True)

            result = get_dataframe(src_fp=fp.name, col_defaults=defaults)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        defaults=fixed_dictionaries({
            'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_uppercase),
            'int_col': integers(min_value=0, max_value=10),
            'FloatCol': floats(min_value=1.0, allow_infinity=False)
        })
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_col_defaults_option_and_use_defaults_for_all_other_options(self, data, defaults):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)
            expected.columns = expected.columns.str.lower()
            for col, default in viewitems(defaults):
                expected.loc[:, col.lower()].fillna(defaults[col], inplace=True)

            result = get_dataframe(src_fp=fp.name, col_defaults=defaults)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False])
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file_with_nulls_in_some_columns__set_non_na_cols_option_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            data[-1]['int_col'] = np.nan
            data[-2]['str_col'] = np.nan
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            non_na_cols = ['int_col', 'str_col']
            expected = df.dropna(subset=non_na_cols, axis=0)

            result = get_dataframe(src_fp=fp.name, non_na_cols=non_na_cols)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False])
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols_and_nulls_in_some_columns__set_non_na_cols_option_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            data[-1]['int_col'] = np.nan
            data[-2]['STR_COL'] = np.nan
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            non_na_cols = ['int_col', 'STR_COL']
            expected = df.dropna(subset=non_na_cols, axis=0)
            expected.columns = expected.columns.str.lower()

            result = get_dataframe(src_fp=fp.name, non_na_cols=non_na_cols)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file__set_sort_cols_option_on_single_col_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            data = [{k: (v if k != 'int_col' else np.random.choice(range(10))) for k, v in viewitems(it)} for it in data]
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            sort_cols = ['int_col']
            expected = df.sort_values(sort_cols, axis=0)

            result = get_dataframe(src_fp=fp.name, sort_cols=sort_cols)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'IntCol': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_sort_cols_option_on_single_col_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            data = [{k: (v if k != 'IntCol' else np.random.choice(range(10))) for k, v in viewitems(it)} for it in data]
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            sort_cols = ['IntCol']
            expected = df.sort_values(sort_cols, axis=0)
            expected.columns = expected.columns.str.lower()

            result = get_dataframe(src_fp=fp.name, sort_cols=sort_cols)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file__set_sort_cols_option_on_two_cols_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            data = [
                {k: (v if k not in ('int_col', 'str_col') else (np.random.choice(range(10)) if k == 'int_col' else np.random.choice(list(string.ascii_lowercase)))) for k, v in viewitems(it)}
                for it in data
            ]
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            sort_cols = ['int_col', 'str_col']
            expected = df.sort_values(sort_cols, axis=0)

            result = get_dataframe(src_fp=fp.name, sort_cols=sort_cols)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'IntCol': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_sort_cols_option_on_two_cols_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            data = [
                {k: (v if k not in ('IntCol', 'STR_COL') else (np.random.choice(range(10)) if k == 'IntCol' else np.random.choice(list(string.ascii_lowercase)))) for k, v in viewitems(it)}
                for it in data
            ]
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            sort_cols = ['IntCol', 'STR_COL']
            expected = df.sort_values(sort_cols, axis=0)
            expected.columns = expected.columns.str.lower()

            result = get_dataframe(src_fp=fp.name, sort_cols=sort_cols)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'Float_Col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'NullCol': just(np.nan)
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_columns___set_lowercase_cols_option_to_false_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)

            result = get_dataframe(src_fp=fp.name, lowercase_cols=False)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'Int_Col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'NullCol': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        dtypes=fixed_dictionaries({
            'Int_Col': sampled_from(['int32', 'int64']),
            'FloatCol': sampled_from(['float32', 'float64'])
        })
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_columns__set_lowercase_col_option_to_false_and_col_dtypes_option_and_use_defaults_for_all_other_options(self, data, dtypes):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            for col, dtype in viewitems(dtypes):
                df[col] = df[col].astype(dtype)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = pd.read_csv(fp.name, dtype=dtypes)

            result = get_dataframe(src_fp=fp.name, col_dtypes=dtypes, lowercase_cols=False)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        subset_cols=just(
            np.random.choice(
                ['STR_COL','int_col','FloatCol','boolCol','null_col'],
                np.random.choice(range(1, 6)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_lowercase_cols_option_to_false_and_subset_cols_option_and_use_defaults_for_all_other_options(self, data, subset_cols):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.drop([col for col in df.columns if col not in subset_cols], axis=1)

            result = get_dataframe(
                src_fp=fp.name,
                subset_cols=subset_cols,
                lowercase_cols=False
            )

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        required_cols=just(
            np.random.choice(
                ['STR_COL','int_col','FloatCol','boolCol','null_col'],
                np.random.choice(range(1, 6)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_lowercase_cols_option_to_false_and_required_cols_option_and_use_defaults_for_all_other_options(self, data, required_cols):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)

            result = get_dataframe(
                src_fp=fp.name,
                required_cols=required_cols,
                lowercase_cols=False
            )

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        missing_cols=just(
            np.random.choice(
                ['STR_COL','int_col','FloatCol','boolCol','null_col'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols_and_missing_some_required_cols__set_lowercase_cols_option_to_false_and_required_cols_option_and_use_defaults_for_all_other_options__oasis_exception_is_raised(self, data, missing_cols):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.drop(missing_cols, axis=1).to_csv(path_or_buf=fp, encoding='utf-8', index=False)
            fp.close()

            with self.assertRaises(OasisException):
                get_dataframe(
                    src_fp=fp.name,
                    required_cols=df.columns.tolist(),
                    lowercase_cols=False
                )
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        defaults=fixed_dictionaries({
            'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_uppercase),
            'int_col': integers(min_value=0, max_value=10),
            'FloatCol': floats(min_value=1.0, allow_infinity=False)
        })
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_lowercase_cols_option_to_false_and_col_defaults_option_and_use_defaults_for_all_other_options(self, data, defaults):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)
            for col, default in viewitems(defaults):
                expected.loc[:, col].fillna(defaults[col], inplace=True)

            result = get_dataframe(src_fp=fp.name, col_defaults=defaults, lowercase_cols=False)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=1, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False])
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols_and_nulls_in_some_columns__set_lowercase_cols_option_to_false_and_non_na_cols_option_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            data[-1]['int_col'] = np.nan
            data[-2]['STR_COL'] = np.nan
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            non_na_cols = ['int_col', 'STR_COL']
            expected = df.dropna(subset=non_na_cols, axis=0)

            result = get_dataframe(src_fp=fp.name, non_na_cols=non_na_cols, lowercase_cols=False)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'IntCol': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_lowercase_cols_option_to_false_and_sort_cols_option_on_single_col_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            data = [{k: (v if k != 'IntCol' else np.random.choice(range(10))) for k, v in viewitems(it)} for it in data]
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            sort_cols = ['IntCol']
            expected = df.sort_values(sort_cols, axis=0)

            result = get_dataframe(src_fp=fp.name, sort_cols=sort_cols, lowercase_cols=False)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'IntCol': integers(min_value=1, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_lowercase_cols_option_to_false_and_sort_cols_option_on_two_cols_and_use_defaults_for_all_other_options(self, data):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            data = [
                {k: (v if k not in ('IntCol', 'STR_COL') else (np.random.choice(range(10)) if k == 'IntCol' else np.random.choice(list(string.ascii_lowercase)))) for k, v in viewitems(it)}
                for it in data
            ]
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            sort_cols = ['IntCol', 'STR_COL']
            expected = df.sort_values(sort_cols, axis=0)

            result = get_dataframe(src_fp=fp.name, sort_cols=sort_cols, lowercase_cols=False)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

class TestGetJson(TestCase):

    @settings(max_examples=10)
    @given(
        data=fixed_dictionaries({
            'str': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
            'int': integers(min_value=1, max_value=10),
            'float': floats(min_value=0.0, max_value=10.0),
            'bool': sampled_from([True, False]),
            'null': just(None),
            'list': lists(integers(min_value=1, max_value=10), min_size=1, max_size=10),
            'subdict': fixed_dictionaries({
                'str': text(min_size=1, max_size=1, alphabet=string.ascii_lowercase),
                'int': integers(min_value=1, max_value=10),
                'float': floats(min_value=0.0, max_value=10.0),
                'bool': sampled_from([True, False]),
                'list': lists(integers(min_value=1, max_value=10), min_size=1, max_size=10),
                'null': just(None)
            })
        })
    )
    def test_get_json__with_nesting_depth_of_1(self, data):
        expected = copy.deepcopy(data)
        f1 = NamedTemporaryFile("w", delete=False)
        try:
            f1.write(json.dumps(expected, indent=4, sort_keys=True))
            f1.close()

            with io_open(f1.name, 'r', encoding='utf-8') as f2:
                result = json.load(f2)
                self.assertEqual(result, expected)
        finally:
            os.remove(f1.name)


class TestGetTimestamp(TestCase):

    @settings(max_examples=10)
    @given(
        dt=datetimes(min_value=datetime.now()),
        fmt=just('%Y%m%d%H%M%S')
    )
    def test_get_timestamp(self, dt, fmt):
        expected = dt.strftime(fmt)
        result = get_timestamp(dt, fmt=fmt)

        self.assertEqual(result, expected)

    # Windows does not support converting datetimes after 3001-01-01T20:59:59
    @settings(max_examples=10, deadline=None)
    @given(
        dt=datetimes(min_value=datetime.now(),
                     max_value=datetime(3001, 1, 1)),
        fmt=just('%Y-%b-%d %H:%M:%S')
    )
    def test_get_utctimestamp(self, dt, fmt):
        expected = dt.astimezone(pytz.utc).strftime(fmt)
        result = get_utctimestamp(dt, fmt=fmt)

        self.assertEqual(result, expected)
