import copy
import io
import json
import os
import string

from collections import OrderedDict
from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd
import pytz

from hypothesis import (
    given,
    settings,
)
from hypothesis.strategies import (
    datetimes,
    integers,
    fixed_dictionaries,
    floats,
    just,
    lists,
    sampled_from,
    text,
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
    get_location_df,
)

from oasislmf.utils.defaults import (
    get_loc_dtypes,
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

        result_enum, result_groups = factorize_array(strings)

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

        result_enum, result_groups = factorize_ndarray(ndarr, row_idxs=row_idxs)

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
    except AssertionError:
        return False

    return True


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
            for col, dtype in dtypes.items():
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
            for col, dtype in dtypes.items():
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
        required=just(
            np.random.choice(
                ['str_col', 'int_col', 'float_col', 'bool_col', 'null_col'],
                np.random.choice(range(1, 6)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file__set_required_cols_option_and_use_defaults_for_all_other_options(self, data, required):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)

            result = get_dataframe(
                src_fp=fp.name,
                required_cols=required
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
        required=just(
            np.random.choice(
                ['STR_COL', 'int_col', 'FloatCol', 'boolCol', 'null_col'],
                np.random.choice(range(1, 6)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_required_cols_option_and_use_defaults_for_all_other_options(self, data, required):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)
            expected.columns = expected.columns.str.lower()

            result = get_dataframe(
                src_fp=fp.name,
                required_cols=required
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
                ['str_col', 'int_col', 'float_col', 'bool_col', 'null_col'],
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
        missing=just(
            np.random.choice(
                ['STR_COL', 'int_col', 'FloatCol', 'boolCol', 'null_col'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols_and_missing_some_required_cols__set_required_cols_option_and_use_defaults_for_all_other_options__oasis_exception_is_raised(self, data, missing):
        fp = NamedTemporaryFile('w', delete=False)
        try:
            df = pd.DataFrame(data)
            df.drop(missing, axis=1).to_csv(path_or_buf=fp, encoding='utf-8', index=False)
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
            for col, default in defaults.items():
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
            for col, default in defaults.items():
                expected.loc[:, col.lower()].fillna(defaults[col], inplace=True)

            result = get_dataframe(src_fp=fp.name, col_defaults=defaults)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=10, max_size=15, alphabet=string.ascii_lowercase),
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
            data = [{k: (v if k != 'int_col' else np.random.choice(range(10))) for k, v in it.items()} for it in data]
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
            data = [{k: (v if k != 'IntCol' else np.random.choice(range(10))) for k, v in it.items()} for it in data]
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
                {k: (v if k not in ('int_col', 'str_col') else (np.random.choice(range(10)) if k == 'int_col' else np.random.choice(list(string.ascii_lowercase)))) for k, v in it.items()}
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
                {k: (v if k not in ('IntCol', 'STR_COL') else (np.random.choice(range(10)) if k == 'IntCol' else np.random.choice(list(string.ascii_lowercase)))) for k, v in it.items()}
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
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=0, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        required=just(
            np.random.choice(
                ['str_col', 'int_col', 'float_col', 'bool_col'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        ),
        defaults=fixed_dictionaries({
            'str_col': just('s'),
            'int_col': just(1),
            'float_col': just(1.0),
            'bool_col': just(False)
        })
    )
    def test_get_dataframe__from_csv_file__set_required_cols_and_col_defaults_options_and_use_defaults_for_all_other_options(self, data, required, defaults):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)
            for col, default in defaults.items():
                expected.loc[:, col].fillna(defaults[col], inplace=True)

            result = get_dataframe(src_fp=fp.name, required_cols=required, col_defaults=defaults)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'IntCol': integers(min_value=0, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        required=just(
            np.random.choice(
                ['STR_COL', 'IntCol', 'float_col', 'boolCol'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        ),
        defaults=fixed_dictionaries({
            'STR_COL': just('s'),
            'IntCol': just(1),
            'float_col': just(1.0),
            'boolCol': just(False)
        })
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_required_cols_and_col_defaults_options_and_use_defaults_for_all_other_options(self, data, required, defaults):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)
            for col, default in defaults.items():
                expected.loc[:, col].fillna(defaults[col], inplace=True)
            expected.columns = expected.columns.str.lower()

            result = get_dataframe(src_fp=fp.name, required_cols=required, col_defaults=defaults)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'str_col': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=0, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'bool_col': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        missing=just(
            np.random.choice(
                ['str_col', 'int_col', 'float_col', 'bool_col', 'null_col'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        ),
        defaults=fixed_dictionaries({
            'str_col': just('s'),
            'int_col': just(1),
            'float_col': just(1.0),
            'bool_col': just(False)
        })
    )
    def test_get_dataframe__from_csv_file_missing_some_required_cols__set_required_cols_and_col_defaults_options_and_use_defaults_for_all_other_options__oasis_exception_is_raised(self, data, missing, defaults):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.drop(missing, axis=1).to_csv(path_or_buf=fp, encoding='utf-8', index=False)
            fp.close()

            with self.assertRaises(OasisException):
                get_dataframe(src_fp=fp.name, required_cols=list(df.columns), col_defaults=defaults)

        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'int_col': integers(min_value=0, max_value=10),
                'FloatCol': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        missing=just(
            np.random.choice(
                ['STR_COL', 'int_col', 'FloatCol', 'boolCol', 'null_col'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        ),
        defaults=fixed_dictionaries({
            'STR_COL': just('s'),
            'int_col': just(1),
            'FloatCol': just(1.0),
            'boolCol': just(False)
        })
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols_and_missing_some_required_cols__set_required_cols_and_col_defaults_options_and_use_defaults_for_all_other_options__oasis_exception_is_raised(self, data, missing, defaults):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.drop(missing, axis=1).to_csv(path_or_buf=fp, encoding='utf-8', index=False)
            fp.close()

            with self.assertRaises(OasisException):
                get_dataframe(src_fp=fp.name, required_cols=list(df.columns), col_defaults=defaults)

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
            for col, dtype in dtypes.items():
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
        required=just(
            np.random.choice(
                ['STR_COL', 'int_col', 'FloatCol', 'boolCol', 'null_col'],
                np.random.choice(range(1, 6)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_lowercase_cols_option_to_false_and_required_cols_option_and_use_defaults_for_all_other_options(self, data, required):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, columns=df.columns, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)

            result = get_dataframe(
                src_fp=fp.name,
                required_cols=required,
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
        missing=just(
            np.random.choice(
                ['STR_COL', 'int_col', 'FloatCol', 'boolCol', 'null_col'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        )
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols_and_missing_some_required_cols__set_lowercase_cols_option_to_false_and_required_cols_option_and_use_defaults_for_all_other_options__oasis_exception_is_raised(self, data, missing):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.drop(missing, axis=1).to_csv(path_or_buf=fp, encoding='utf-8', index=False)
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
            for col, default in defaults.items():
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
            data = [{k: (v if k != 'IntCol' else np.random.choice(range(10))) for k, v in it.items()} for it in data]
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
                {k: (v if k not in ('IntCol', 'STR_COL') else (np.random.choice(range(10)) if k == 'IntCol' else np.random.choice(list(string.ascii_lowercase)))) for k, v in it.items()}
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

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'IntCol': integers(min_value=0, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        required=just(
            np.random.choice(
                ['STR_COL', 'IntCol', 'float_col', 'boolCol'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        ),
        defaults=fixed_dictionaries({
            'STR_COL': just('s'),
            'IntCol': just(1),
            'float_col': just(1.0),
            'boolCol': just(False)
        })
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols__set_lowercase_cols_option_to_false__set_required_cols_and_col_defaults_options_and_use_defaults_for_all_other_options(self, data, required, defaults):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.to_csv(path_or_buf=fp, encoding='utf-8', index=False)
            fp.close()

            expected = df.copy(deep=True)
            for col, default in defaults.items():
                expected.loc[:, col].fillna(defaults[col], inplace=True)

            result = get_dataframe(src_fp=fp.name, lowercase_cols=False, required_cols=required, col_defaults=defaults)

            self.assertTrue(dataframes_are_identical(result, expected))
        finally:
            os.remove(fp.name)

    @settings(max_examples=10)
    @given(
        data=lists(
            fixed_dictionaries({
                'STR_COL': text(min_size=1, max_size=10, alphabet=string.ascii_lowercase),
                'IntCol': integers(min_value=0, max_value=10),
                'float_col': floats(min_value=0.0, max_value=10.0),
                'boolCol': sampled_from([True, False]),
                'null_col': just(np.nan)
            }),
            min_size=10,
            max_size=10
        ),
        missing=just(
            np.random.choice(
                ['STR_COL', 'IntCol', 'float_col', 'boolCol', 'null_col'],
                np.random.choice(range(1, 5)),
                replace=False
            ).tolist()
        ),
        defaults=fixed_dictionaries({
            'STR_COL': just('s'),
            'IntCol': just(1),
            'float_col': just(1.0),
            'boolCol': just(False)
        })
    )
    def test_get_dataframe__from_csv_file_with_mixed_case_cols_and_missing_some_required_cols__set_lowercase_cols_option_to_false__set_required_cols_and_col_defaults_options_and_use_defaults_for_all_other_options__oasis_exception_is_raised(self, data, missing, defaults):
        fp = NamedTemporaryFile("w", delete=False)
        try:
            df = pd.DataFrame(data)
            df.drop(missing, axis=1).to_csv(path_or_buf=fp, encoding='utf-8', index=False)
            fp.close()

            with self.assertRaises(OasisException):
                get_dataframe(src_fp=fp.name, required_cols=list(df.columns), col_defaults=defaults)

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

            with io.open(f1.name, 'r', encoding='utf-8') as f2:
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



class TestGetLocationDf(TestCase):

    @settings(max_examples=10, deadline=None)
    @given(
        data=fixed_dictionaries({
             "PortNumber": sampled_from([str(1), int(1), float(1)]),
             "AccNumber": sampled_from([str(1), int(1), float(1)]),
             "LocNumber": sampled_from([str(1), int(1), float(1)]),
             "LocName": sampled_from([str(1), int(1), float(1)]),
             "LocGroup": sampled_from([str(1), int(1), float(1)]),
             "CorrelationGroup": sampled_from([str(1), int(1), float(1)]),
             "IsPrimary": sampled_from([str(1), int(1), float(1)]),
             "IsTenant": sampled_from([str(1), int(1), float(1)]),
             "BuildingID": sampled_from([str(1), int(1), float(1)]),
             "LocInceptionDate": sampled_from([str(1), int(1), float(1)]),
             "LocExpiryDate": sampled_from([str(1), int(1), float(1)]),
             "PercentComplete": sampled_from([str(1), int(1), float(1)]),
             "CompletionDate": sampled_from([str(1), int(1), float(1)]),
             "CountryCode": sampled_from([str(1), int(1), float(1)]),
             "Latitude": sampled_from([str(1), int(1), float(1)]),
             "Longitude": sampled_from([str(1), int(1), float(1)]),
             "StreetAddress": sampled_from([str(1), int(1), float(1)]),
             "PostalCode": sampled_from([str(1), int(1), float(1)]),
             "City": sampled_from([str(1), int(1), float(1)]),
             "AreaCode": sampled_from([str(1), int(1), float(1)]),
             "AreaName": sampled_from([str(1), int(1), float(1)]),
             "GeogScheme1": sampled_from([str(1), int(1), float(1)]),
             "GeogName1": sampled_from([str(1), int(1), float(1)]),
             "GeogScheme2": sampled_from([str(1), int(1), float(1)]),
             "GeogName2": sampled_from([str(1), int(1), float(1)]),
             "GeogScheme3": sampled_from([str(1), int(1), float(1)]),
             "GeogName3": sampled_from([str(1), int(1), float(1)]),
             "GeogScheme4": sampled_from([str(1), int(1), float(1)]),
             "GeogName4": sampled_from([str(1), int(1), float(1)]),
             "GeogScheme5": sampled_from([str(1), int(1), float(1)]),
             "GeogName5": sampled_from([str(1), int(1), float(1)]),
             "AddressMatch": sampled_from([str(1), int(1), float(1)]),
             "GeocodeQuality": sampled_from([str(1), int(1), float(1)]),
             "Geocoder": sampled_from([str(1), int(1), float(1)]),
             "OrgOccupancyScheme": sampled_from([str(1), int(1), float(1)]),
             "OrgOccupancyCode": sampled_from([str(1), int(1), float(1)]),
             "OrgConstructionScheme": sampled_from([str(1), int(1), float(1)]),
             "OrgConstructionCode": sampled_from([str(1), int(1), float(1)]),
             "OccupancyCode": sampled_from([str(1), int(1), float(1)]),
             "ConstructionCode": sampled_from([str(1), int(1), float(1)]),
             "YearBuilt": sampled_from([str(1), int(1), float(1)]),
             "NumberOfStoreys": sampled_from([str(1), int(1), float(1)]),
             "NumberOfBuildings": sampled_from([str(1), int(1), float(1)]),
             "FloorArea": sampled_from([str(1), int(1), float(1)]),
             "FloorAreaUnit": sampled_from([str(1), int(1), float(1)]),
             "LocUserDef1": sampled_from([str(1), int(1), float(1)]),
             "LocUserDef2": sampled_from([str(1), int(1), float(1)]),
             "LocUserDef3": sampled_from([str(1), int(1), float(1)]),
             "LocUserDef4": sampled_from([str(1), int(1), float(1)]),
             "LocUserDef5": sampled_from([str(1), int(1), float(1)]),
             "FlexiLocZZZ": sampled_from([str(1), int(1), float(1)]),
             "LocPerilsCovered": sampled_from([str(1), int(1), float(1)]),
             "BuildingTIV": sampled_from([str(1), int(1), float(1)]),
             "OtherTIV": sampled_from([str(1), int(1), float(1)]),
             "ContentsTIV": sampled_from([str(1), int(1), float(1)]),
             "BITIV": sampled_from([str(1), int(1), float(1)]),
             "BIPOI": sampled_from([str(1), int(1), float(1)]),
             "LocCurrency": sampled_from([str(1), int(1), float(1)]),
             "LocGrossPremium": sampled_from([str(1), int(1), float(1)]),
             "LocTax": sampled_from([str(1), int(1), float(1)]),
             "LocBrokerage": sampled_from([str(1), int(1), float(1)]),
             "LocNetPremium": sampled_from([str(1), int(1), float(1)]),
             "NonCatGroundUpLoss": sampled_from([str(1), int(1), float(1)]),
             "LocParticipation": sampled_from([str(1), int(1), float(1)]),
             "PayoutBasis": sampled_from([str(1), int(1), float(1)]),
             "ReinsTag": sampled_from([str(1), int(1), float(1)]),
             "CondNumber": sampled_from([str(1), int(1), float(1)]),
             "CondPriority": sampled_from([str(1), int(1), float(1)]),
             "LocDedCode1Building": sampled_from([str(1), int(1), float(1)]),
             "LocDedType1Building": sampled_from([str(1), int(1), float(1)]),
             "LocDed1Building": sampled_from([str(1), int(1), float(1)]),
             "LocMinDed1Building": sampled_from([str(1), int(1), float(1)]),
             "LocMaxDed1Building": sampled_from([str(1), int(1), float(1)]),
             "LocDedCode2Other": sampled_from([str(1), int(1), float(1)]),
             "LocDedType2Other": sampled_from([str(1), int(1), float(1)]),
             "LocDed2Other": sampled_from([str(1), int(1), float(1)]),
             "LocMinDed2Other": sampled_from([str(1), int(1), float(1)]),
             "LocMaxDed2Other": sampled_from([str(1), int(1), float(1)]),
             "LocDedCode3Contents": sampled_from([str(1), int(1), float(1)]),
             "LocDedType3Contents": sampled_from([str(1), int(1), float(1)]),
             "LocDed3Contents": sampled_from([str(1), int(1), float(1)]),
             "LocMinDed3Contents": sampled_from([str(1), int(1), float(1)]),
             "LocMaxDed3Contents": sampled_from([str(1), int(1), float(1)]),
             "LocDedCode4BI": sampled_from([str(1), int(1), float(1)]),
             "LocDedType4BI": sampled_from([str(1), int(1), float(1)]),
             "LocDed4BI": sampled_from([str(1), int(1), float(1)]),
             "LocMinDed4BI": sampled_from([str(1), int(1), float(1)]),
             "LocMaxDed4BI": sampled_from([str(1), int(1), float(1)]),
             "LocDedCode5PD": sampled_from([str(1), int(1), float(1)]),
             "LocDedType5PD": sampled_from([str(1), int(1), float(1)]),
             "LocDed5PD": sampled_from([str(1), int(1), float(1)]),
             "LocMinDed5PD": sampled_from([str(1), int(1), float(1)]),
             "LocMaxDed5PD": sampled_from([str(1), int(1), float(1)]),
             "LocDedCode6All": sampled_from([str(1), int(1), float(1)]),
             "LocDedType6All": sampled_from([str(1), int(1), float(1)]),
             "LocDed6All": sampled_from([str(1), int(1), float(1)]),
             "LocMinDed6All": sampled_from([str(1), int(1), float(1)]),
             "LocMaxDed6All": sampled_from([str(1), int(1), float(1)]),
             "LocLimitCode1Building": sampled_from([str(1), int(1), float(1)]),
             "LocLimitType1Building": sampled_from([str(1), int(1), float(1)]),
             "LocLimit1Building": sampled_from([str(1), int(1), float(1)]),
             "LocLimitCode2Other": sampled_from([str(1), int(1), float(1)]),
             "LocLimitType2Other": sampled_from([str(1), int(1), float(1)]),
             "LocLimit2Other": sampled_from([str(1), int(1), float(1)]),
             "LocLimitCode3Contents": sampled_from([str(1), int(1), float(1)]),
             "LocLimitType3Contents": sampled_from([str(1), int(1), float(1)]),
             "LocLimit3Contents": sampled_from([str(1), int(1), float(1)]),
             "LocLimitCode4BI": sampled_from([str(1), int(1), float(1)]),
             "LocLimitType4BI": sampled_from([str(1), int(1), float(1)]),
             "LocLimit4BI": sampled_from([str(1), int(1), float(1)]),
             "LocLimitCode5PD": sampled_from([str(1), int(1), float(1)]),
             "LocLimitType5PD": sampled_from([str(1), int(1), float(1)]),
             "LocLimit5PD": sampled_from([str(1), int(1), float(1)]),
             "LocLimitCode6All": sampled_from([str(1), int(1), float(1)]),
             "LocLimitType6All": sampled_from([str(1), int(1), float(1)]),
             "LocLimit6All": sampled_from([str(1), int(1), float(1)]),
             "BIWaitingPeriod": sampled_from([str(1), int(1), float(1)]),
             "LocPeril": sampled_from([str(1), int(1), float(1)]),
             "YearUpgraded": sampled_from([str(1), int(1), float(1)]),
             "SurgeLeakage": sampled_from([str(1), int(1), float(1)]),
             "SprinklerType": sampled_from([str(1), int(1), float(1)]),
             "PercentSprinklered": sampled_from([str(1), int(1), float(1)]),
             "RoofCover": sampled_from([str(1), int(1), float(1)]),
             "RoofYearBuilt": sampled_from([str(1), int(1), float(1)]),
             "RoofGeometry": sampled_from([str(1), int(1), float(1)]),
             "RoofEquipment": sampled_from([str(1), int(1), float(1)]),
             "RoofFrame": sampled_from([str(1), int(1), float(1)]),
             "RoofMaintenance": sampled_from([str(1), int(1), float(1)]),
             "BuildingCondition": sampled_from([str(1), int(1), float(1)]),
             "RoofAttachedStructures": sampled_from([str(1), int(1), float(1)]),
             "RoofDeck": sampled_from([str(1), int(1), float(1)]),
             "RoofPitch": sampled_from([str(1), int(1), float(1)]),
             "RoofAnchorage": sampled_from([str(1), int(1), float(1)]),
             "RoofDeckAttachment": sampled_from([str(1), int(1), float(1)]),
             "RoofCoverAttachment": sampled_from([str(1), int(1), float(1)]),
             "GlassType": sampled_from([str(1), int(1), float(1)]),
             "LatticeType": sampled_from([str(1), int(1), float(1)]),
             "FloodZone": sampled_from([str(1), int(1), float(1)]),
             "SoftStory": sampled_from([str(1), int(1), float(1)]),
             "Basement": sampled_from([str(1), int(1), float(1)]),
             "BasementLevelCount": sampled_from([str(1), int(1), float(1)]),
             "WindowProtection": sampled_from([str(1), int(1), float(1)]),
             "FoundationType": sampled_from([str(1), int(1), float(1)]),
             "WallAttachedStructure": sampled_from([str(1), int(1), float(1)]),
             "AppurtenantStructure": sampled_from([str(1), int(1), float(1)]),
             "ConstructionQuality": sampled_from([str(1), int(1), float(1)]),
             "GroundEquipment": sampled_from([str(1), int(1), float(1)]),
             "EquipmentBracing": sampled_from([str(1), int(1), float(1)]),
             "Flashing": sampled_from([str(1), int(1), float(1)]),
             "BuildingShape": sampled_from([str(1), int(1), float(1)]),
             "ShapeIrregularity": sampled_from([str(1), int(1), float(1)]),
             "Pounding": sampled_from([str(1), int(1), float(1)]),
             "Ornamentation": sampled_from([str(1), int(1), float(1)]),
             "SpecialEQConstruction": sampled_from([str(1), int(1), float(1)]),
             "Retrofit": sampled_from([str(1), int(1), float(1)]),
             "CrippleWall": sampled_from([str(1), int(1), float(1)]),
             "FoundationConnection": sampled_from([str(1), int(1), float(1)]),
             "ShortColumn": sampled_from([str(1), int(1), float(1)]),
             "Fatigue": sampled_from([str(1), int(1), float(1)]),
             "Cladding": sampled_from([str(1), int(1), float(1)]),
             "BIPreparedness": sampled_from([str(1), int(1), float(1)]),
             "BIRedundancy": sampled_from([str(1), int(1), float(1)]),
             "FirstFloorHeight": sampled_from([str(1), int(1), float(1)]),
             "FirstFloorHeightUnit": sampled_from([str(1), int(1), float(1)]),
             "Datum": sampled_from([str(1), int(1), float(1)]),
             "GroundElevation": sampled_from([str(1), int(1), float(1)]),
             "GroundElevationUnit": sampled_from([str(1), int(1), float(1)]),
             "Tank": sampled_from([str(1), int(1), float(1)]),
             "Redundancy": sampled_from([str(1), int(1), float(1)]),
             "InternalPartition": sampled_from([str(1), int(1), float(1)]),
             "ExternalDoors": sampled_from([str(1), int(1), float(1)]),
             "Torsion": sampled_from([str(1), int(1), float(1)]),
             "MechanicalEquipmentSide": sampled_from([str(1), int(1), float(1)]),
             "ContentsWindVuln": sampled_from([str(1), int(1), float(1)]),
             "ContentsFloodVuln": sampled_from([str(1), int(1), float(1)]),
             "ContentsQuakeVuln": sampled_from([str(1), int(1), float(1)]),
             "SmallDebris": sampled_from([str(1), int(1), float(1)]),
             "FloorsOccupied": sampled_from([str(1), int(1), float(1)]),
             "FloodDefenseHeight": sampled_from([str(1), int(1), float(1)]),
             "FloodDefenseHeightUnit": sampled_from([str(1), int(1), float(1)]),
             "FloodDebrisResilience": sampled_from([str(1), int(1), float(1)]),
             "BaseFloodElevation": sampled_from([str(1), int(1), float(1)]),
             "BaseFloodElevationUnit": sampled_from([str(1), int(1), float(1)]),
             "BuildingHeight": sampled_from([str(1), int(1), float(1)]),
             "BuildingHeightUnit": sampled_from([str(1), int(1), float(1)]),
             "BuildingValuation": sampled_from([str(1), int(1), float(1)]),
             "TreeExposure": sampled_from([str(1), int(1), float(1)]),
             "Chimney": sampled_from([str(1), int(1), float(1)]),
             "BuildingType": sampled_from([str(1), int(1), float(1)]),
             "Packaging": sampled_from([str(1), int(1), float(1)]),
             "Protection": sampled_from([str(1), int(1), float(1)]),
             "SalvageProtection": sampled_from([str(1), int(1), float(1)]),
             "ValuablesStorage": sampled_from([str(1), int(1), float(1)]),
             "DaysHeld": sampled_from([str(1), int(1), float(1)]),
             "BrickVeneer": sampled_from([str(1), int(1), float(1)]),
             "FEMACompliance": sampled_from([str(1), int(1), float(1)]),
             "CustomFloodSOP": sampled_from([str(1), int(1), float(1)]),
             "CustomFloodZone": sampled_from([str(1), int(1), float(1)]),
             "MultiStoryHall": sampled_from([str(1), int(1), float(1)]),
             "BuildingExteriorOpening": sampled_from([str(1), int(1), float(1)]),
             "ServiceEquipmentProtection": sampled_from([str(1), int(1), float(1)]),
             "TallOneStory": sampled_from([str(1), int(1), float(1)]),
             "TerrainRoughness": sampled_from([str(1), int(1), float(1)]),
             "NumberOfEmployees": sampled_from([str(1), int(1), float(1)]),
             "Payroll": sampled_from([str(1), int(1), float(1)]),
        })
    )
    def test_csv_dtypes_loaded_correctly(self, data):
        loc_expected_dtypes = {k.lower(): v for k, v in get_loc_dtypes().items()}
        loc_sample_file = NamedTemporaryFile("w", delete=False)

        valid_str_types = (
            str
        )
        valid_int_types = (
            int,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        )
        valid_float_types = (
            float,
            np.single,
            np.double,
            np.longdouble,
            np.float32,
            np.float64
        )

        try:
            df = pd.DataFrame(data, index=[0])
            df.to_csv(path_or_buf=loc_sample_file, encoding='utf-8', index=False)
            loc_sample_file.close()

            df_result = get_location_df(loc_sample_file.name)
            for col in df_result.columns:
                if col in loc_expected_dtypes:
                    dtype_expected = loc_expected_dtypes[col]['py_dtype']
                    dtype_found = type(df_result[col][0])
                    print(f'{col} - Expected: {dtype_expected}, Found: {dtype_found}')

                    if dtype_expected == 'str':
                        self.assertTrue(isinstance(df_result[col][0], valid_str_types))
                    elif dtype_expected == 'int':
                        self.assertTrue(isinstance(df_result[col][0], valid_int_types))
                    elif dtype_expected == 'float':
                        self.assertTrue(isinstance(df_result[col][0], valid_float_types))

        finally:
            os.remove(loc_sample_file.name)
