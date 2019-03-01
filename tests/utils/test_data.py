# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from unittest import TestCase
from tempfile import NamedTemporaryFile
from oasislmf.utils.data import get_dataframe, set_col_dtypes
from oasislmf.utils.exceptions import OasisException
from pandas.util.testing import assert_frame_equal
import pandas.api.types as pd_types

class GetDataFrame(TestCase):

    def test_basic_read_csv(self):
        with NamedTemporaryFile('w') as f:
            f.writelines([
                'a,b\n1,2\n3,4',
            ])
            f.flush()
            df = get_dataframe(f.name, index_col=False)
            
            ref_data = {
                'a': [1,3],
                'b': [2,4]
            }

            ref_df = pd.DataFrame.from_dict(ref_data)

        assert_frame_equal(df, ref_df)

    def test_all_required_cols_present_in_csv(self):
        with NamedTemporaryFile('w') as f:
            f.writelines([
                'A,b\n1,2\n3,4',
            ])
            f.flush()
            df = get_dataframe(
                f.name, lowercase_cols=False, index_col=False, required_cols=['A','b'])
            
            ref_data = {
                'A': [1,3],
                'b': [2,4]
            }

            ref_df = pd.DataFrame.from_dict(ref_data)

        assert_frame_equal(df, ref_df)

    def test_all_required_cols_present_in_csv_case_insensitive(self):
        with NamedTemporaryFile('w') as f:
            f.writelines([
                'a,b\n1,2\n3,4',
            ])
            f.flush()
            df = get_dataframe(
                f.name, index_col=False, lowercase_cols=True, required_cols=['A','B'])
            
            ref_data = {
                'a': [1,3],
                'b': [2,4]
            }

            ref_df = pd.DataFrame.from_dict(ref_data)

        assert_frame_equal(df, ref_df)

    def test_missing_required_cols_in_csv_throws_exception(self):
        with self.assertRaises(OasisException):
            with NamedTemporaryFile('w') as f:
                f.writelines([
                    'a,b\n1,2\n3,4',
                ])
                f.flush()
                df = get_dataframe(f.name, index_col=False, required_cols=['a','b','c'])

    def test_all_default_cols_present_in_csv(self):
        with NamedTemporaryFile('w') as f:
            f.writelines([
                'a,b\n1,2\n3,4',
            ])
            f.flush()
            df = get_dataframe(
                f.name, index_col=False, defaulted_cols={'a': 1, 'b': 2})

            ref_data = {
                'a': [1,3],
                'b': [2,4],
            }

            ref_df = pd.DataFrame.from_dict(ref_data)

        assert_frame_equal(df, ref_df)

    def test_all_add_default_str_in_csv(self):
        with NamedTemporaryFile('w') as f:
            f.writelines([
                'a,b\n1,2\n3,4',
            ])
            f.flush()
            df = get_dataframe(f.name, index_col=False, defaulted_cols={'c': 'abc'})

            ref_data = {
                'a': [1,3],
                'b': [2,4],
                'c': ['abc', 'abc']
            }

            ref_df = pd.DataFrame.from_dict(ref_data)

        assert_frame_equal(df, ref_df)

    def test_all_add_default_number_in_csv(self):
        with NamedTemporaryFile('w') as f:
            f.writelines([
                'a,b\n1,2\n3,4',
            ])
            f.flush()
            df = get_dataframe(f.name, index_col=False, defaulted_cols={'c': 9.99})

            ref_data = {
                'a': [1,3],
                'b': [2,4],
                'c': [9.99, 9.99]
            }

            ref_df = pd.DataFrame.from_dict(ref_data)

        assert_frame_equal(df, ref_df)

    def test_set_col_dtypes(self):
        data = {
                'a': [1,3]
        }
        df = pd.DataFrame.from_dict(data)
        assert pd_types.is_integer_dtype(df.a.dtype)
        
    def test_set_col_dtypes_change_to_float(self):
        data = {
                'a': [1,3]
        }
        df = pd.DataFrame.from_dict(data)
        set_col_dtypes(df, {"a": "float"})
        print(df.a.dtype)
        assert pd_types.is_float_dtype(df.a.dtype)

    def test_set_col_dtypes_change_to_string(self):
        data = {
                'a': [1,3]
        }
        df = pd.DataFrame.from_dict(data)
        set_col_dtypes(df, {"a": "str"})
        print(df.a.dtype)
        assert pd_types.is_string_dtype(df.a.dtype)

    def test_set_col_dtypes_ignore_extra_columns(self):
        data = {
                'a': [1,3]
        }
        df = pd.DataFrame.from_dict(data)
        set_col_dtypes(df, {"a": "str", "b": "int"})
        print(df.a.dtype)
        assert pd_types.is_string_dtype(df.a.dtype)
