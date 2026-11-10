"""Tests for range-checked integer narrowing (#2197)."""
import io

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from oasislmf.pytools.converters.csvtobin.utils.common import df_to_ndarray, iter_csv_as_ndarray
from oasislmf.pytools.getmodel.footprint import df_to_numpy
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.int_range import astype_checked, check_int_range, narrow_int, widen_int_dtypes

INT_TYPES = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8']


@pytest.mark.parametrize('values, dtype', [
    (pd.Series([1, 4294967298]), 'u4'),             # the issue's areaperil example
    (pd.Series([3000000000]), np.int32),            # the issue's vulnerability example
    (pd.Series([5000000000.0]), 'u4'),              # float to int wraps too
    (pd.Series([2.0 ** 63]), 'i8'),                 # i8 max rounds up to 2**63 as a float
    (pd.Series([2.0 ** 64]), 'u8'),
    (np.array([2.0 ** 31], dtype='f4'), 'i4'),      # i4 max rounds up to 2**31 in float32
    (np.array([np.inf]), 'i8'),
    (np.array([-np.inf], dtype='f2'), 'i4'),
    (np.array([-1]), 'u4'),
    (np.array([2 ** 63], dtype='u8'), 'i8'),
    (pd.Series([None, 2 ** 40], dtype='Int64'), 'u4'),
    (pd.Series([2 ** 40]).astype('category'), 'u4'),
])
def test_check_int_range__out_of_range__raises(values, dtype):
    with pytest.raises(OasisException, match='does not fit'):
        check_int_range(values, dtype, 'x')


@pytest.mark.parametrize('values, dtype', [
    (pd.Series([0, 4294967295]), 'u4'),
    (pd.Series([5.0, np.nan]), 'u4'),               # NaN is not a range problem
    (pd.Series([-2.0 ** 63, 2.0 ** 63 - 1024]), 'i8'),  # the extreme floats that do fit
    (pd.Series([2.0 ** 64 - 2048]), 'u8'),
    (np.array([-2.0 ** 31, 2.0 ** 31 - 128], dtype='f4'), 'i4'),
    (pd.Series([None, 1], dtype='Int64'), 'u4'),
    (np.array([5]), 'u8'),
    (np.arange(3, dtype='i4'), 'i4'),
    (pd.Series(['a']), 'u4'),                       # pandas raises on str itself
    (pd.Series([2 ** 40]), 'f4'),                   # not an integer target
    (pd.Series([2 ** 40]), 'category'),
    (np.array([], dtype='i8'), 'u1'),
])
def test_check_int_range__in_range_or_not_checked__passes(values, dtype):
    check_int_range(values, dtype, 'x')


def test_check_int_range__message_names_data_value_row_and_range():
    with pytest.raises(OasisException) as e:
        check_int_range(pd.Series([1, 4294967298], index=[10, 11]), 'u4', 'keys AreaPerilID')
    msg = str(e.value)
    assert 'keys AreaPerilID' in msg and '4294967298' in msg and 'row 11' in msg and '0 to 4294967295' in msg
    assert 'OASIS_AREAPERIL_TYPE=u8' in msg


@given(st.sampled_from(INT_TYPES), st.sampled_from(INT_TYPES), st.data())
def test_narrow_int__matches_astype_when_in_range_and_raises_otherwise(src, dst, data):
    src_info, dst_info = np.iinfo(src), np.iinfo(dst)
    values = np.array(data.draw(st.lists(st.integers(int(src_info.min), int(src_info.max)), max_size=20)), dtype=src)
    fits = all(dst_info.min <= int(v) <= dst_info.max for v in values)
    if fits:
        np.testing.assert_array_equal(narrow_int(values, dst, 'x'), values.astype(dst))
    else:
        with pytest.raises(OasisException):
            narrow_int(values, dst, 'x')


def test_astype_checked__converts_like_astype_and_raises_on_overflow():
    df = pd.DataFrame({'a': [1, 2], 'b': [0.5, 1.5]})
    out = astype_checked(df, {'a': 'u1', 'b': 'f4'}, 'f.csv')
    pd.testing.assert_frame_equal(out, df.astype({'a': 'u1', 'b': 'f4'}))
    with pytest.raises(KeyError):  # same as DataFrame.astype
        astype_checked(df, {'missing': 'i4'}, 'f.csv')
    with pytest.raises(OasisException, match='f.csv a'):
        astype_checked(pd.DataFrame({'a': [256]}), {'a': 'u1'}, 'f.csv')


def test_widen_int_dtypes():
    widened = widen_int_dtypes({'a': np.int32, 'b': 'category', 'c': 'uint64', 'd': 'u4', 'e': 'Int64', 'f': str, 'g': 'float32'})
    assert widened == {'a': np.dtype('int64'), 'b': 'category', 'c': 'uint64', 'd': np.dtype('int64'), 'e': 'Int64', 'f': str, 'g': 'float32'}


def test_df_to_ndarray__out_of_range__raises():
    dtype = np.dtype([('areaperil_id', 'u4'), ('vulnerability_id', 'i4')])
    with pytest.raises(OasisException, match='areaperil_id'):
        df_to_ndarray(pd.DataFrame({'areaperil_id': [4294967298], 'vulnerability_id': [1]}), dtype)
    with pytest.raises(OasisException, match='vulnerability_id'):
        df_to_ndarray(pd.DataFrame({'areaperil_id': [1], 'vulnerability_id': [3000000000]}), dtype)


def test_csvtobin_reader__out_of_range__raises_with_column_and_value():
    dtype = np.dtype([('event_id', 'i4'), ('areaperil_id', 'u4')])
    csv = io.StringIO('event_id,areaperil_id\n1,1\n2,4294967298\n')
    with pytest.raises(OasisException, match='areaperil_id: value 4294967298 at row 1'):
        list(iter_csv_as_ndarray(None, csv, dtype))


def test_footprint_df_to_numpy__out_of_range__raises():
    dtype = np.dtype([('areaperil_id', 'u4'), ('intensity_bin_id', 'i4')])
    with pytest.raises(OasisException, match='footprint areaperil_id'):
        df_to_numpy(pd.DataFrame({'areaperil_id': [4294967298], 'intensity_bin_id': [1]}), dtype)
