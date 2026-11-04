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
