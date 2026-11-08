from tempfile import TemporaryDirectory
import os

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.data import load_as_ndarray, oasis_int
from oasislmf.utils.exceptions import OasisException

summary_map_dtype = np.dtype([('loc_id', oasis_int), ('item_id', oasis_int), ('peril_id', 'U3')])


def test_load_as_ndarray__parquet_matches_csv():
    df = pd.DataFrame({'loc_id': [1, 1, 2], 'output_id': [1, 2, 3], 'peril_id': ['WTC', 'WSS', 'WTC'], 'tiv': [1., 2., 3.]})
    with TemporaryDirectory() as tmp:
        df.to_csv(os.path.join(tmp, 'map_csv.csv'), index=False)
        df.to_parquet(os.path.join(tmp, 'map_pq.parquet'))
        from_csv = load_as_ndarray(tmp, 'map_csv', summary_map_dtype, col_map={'item_id': 'output_id'})
        from_parquet = load_as_ndarray(tmp, 'map_pq', summary_map_dtype, col_map={'item_id': 'output_id'})
    np.testing.assert_array_equal(from_csv, from_parquet)
    assert from_parquet.dtype == summary_map_dtype


def test_load_as_ndarray__parquet_categorical_column():
    df = pd.DataFrame({'loc_id': [1, 2], 'item_id': [1, 2], 'peril_id': pd.Categorical(['WTC', 'WSS'])})
    with TemporaryDirectory() as tmp:
        df.to_parquet(os.path.join(tmp, 'map.parquet'))
        result = load_as_ndarray(tmp, 'map', summary_map_dtype)
    assert list(result['peril_id']) == ['WTC', 'WSS']


def test_load_as_ndarray__parquet_preferred_over_csv():
    with TemporaryDirectory() as tmp:
        pd.DataFrame({'loc_id': [1], 'item_id': [1], 'peril_id': ['WTC']}).to_parquet(os.path.join(tmp, 'map.parquet'))
        pd.DataFrame({'loc_id': [9], 'item_id': [9], 'peril_id': ['WSS']}).to_csv(os.path.join(tmp, 'map.csv'), index=False)
        result = load_as_ndarray(tmp, 'map', summary_map_dtype)
    assert result['loc_id'][0] == 1


def test_load_as_ndarray__parquet_missing_column():
    with TemporaryDirectory() as tmp:
        pd.DataFrame({'loc_id': [1], 'peril_id': ['WTC']}).to_parquet(os.path.join(tmp, 'map.parquet'))
        with pytest.raises(OasisException, match='item_id'):
            load_as_ndarray(tmp, 'map', summary_map_dtype)


def _load_csv(text, dtype):
    with TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'f.csv'), 'w') as f:
            f.write(text)
        return load_as_ndarray(tmp, 'f', dtype)


def test_load_as_ndarray__csv_whole_float_ints__issue_2217():
    from oasislmf.pytools.common.input_files import KEYS_DTYPE
    keys = _load_csv("LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID\n"
                     "1,OSF,1,367923277.0,20001\n"
                     "2,OSF,3,367923278,20002.00\n", KEYS_DTYPE)
    assert keys.dtype == KEYS_DTYPE
    assert keys['AreaPerilID'].tolist() == [367923277, 367923278]
    assert keys['VulnerabilityID'].tolist() == [20001, 20002]
    assert keys['PerilID'].tolist() == ['OSF', 'OSF']
    assert keys['LocID'].tolist() == [1, 2]


def test_load_as_ndarray__csv_whole_float_ints__exact_beyond_float_precision():
    dtype = np.dtype([('a', 'u8'), ('b', 'i8'), ('x', 'f4')])
    result = _load_csv("a,b,x\n18446744073709551615.0,1e+16,0.5\n9007199254740993.0,-3.0,1\n", dtype)
    assert result['a'].tolist() == [18446744073709551615, 9007199254740993]
    assert result['b'].tolist() == [10 ** 16, -3]
    assert result['x'].tolist() == [0.5, 1.0]


@pytest.mark.parametrize('value, match', [
    ('1.5', r"f.csv: a value '1.5' on line 3 is not a whole number"),
    ('nan', r"value 'nan' on line 3 is not a whole number"),
    ('4294967296.0', r"value '4294967296.0' on line 3 does not fit uint32 \(0 to 4294967295\)"),
    ('-1.0', r"value '-1.0' on line 3 does not fit uint32"),
])
def test_load_as_ndarray__csv_bad_int__raises_with_value_and_line(value, match):
    with pytest.raises(OasisException, match=match):
        _load_csv(f"a\n1.0\n{value}\n", np.dtype([('a', 'u4')]))


def test_load_as_ndarray__csv_out_of_range_int_without_float__still_raises():
    with pytest.raises((OasisException, ValueError)):
        _load_csv("a\n4294967296\n", np.dtype([('a', 'u4')]))
