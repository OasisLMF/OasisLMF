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


# --- whole numbers written as floats in integer fields (#2217) ---

from oasislmf.pytools.common.input_files import KEYS_DTYPE  # noqa: E402

KEYS_HEADER = "LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID\n"


def _load_keys(tmp_path, rows):
    (tmp_path / "keys.csv").write_text(KEYS_HEADER + "".join(row + "\n" for row in rows))
    return load_as_ndarray(str(tmp_path), "keys", KEYS_DTYPE)


def test_load_as_ndarray__csv_whole_number_floats_in_integer_fields(tmp_path):
    keys = _load_keys(tmp_path, ["1,OSF,1,367923277.0,20001", "2,OSF,1,3.67923278e8,20002.00", "3,OSF,3,7,5"])
    assert keys.dtype == KEYS_DTYPE
    assert keys["AreaPerilID"].tolist() == [367923277, 367923278, 7]
    assert keys["VulnerabilityID"].tolist() == [20001, 20002, 5]
    assert keys["PerilID"].tolist() == ["OSF"] * 3


def test_load_as_ndarray__csv_integers_unchanged(tmp_path):
    keys = _load_keys(tmp_path, ["1,WTC,1,10,3", "2,WSS,4,11,4"])
    assert keys["AreaPerilID"].tolist() == [10, 11]
    assert keys["CoverageTypeID"].tolist() == [1, 4]


def test_load_as_ndarray__csv_single_float_row(tmp_path):
    keys = _load_keys(tmp_path, ["1,OSF,1,367923277.0,20001"])
    assert keys.shape == (1,)
    assert keys["AreaPerilID"][0] == 367923277


@pytest.mark.parametrize("value, match", [
    ("1.5", "not a whole number"),
    ("abc", "could not read integer fields"),
    ("-1.0", "out of range"),
    ("inf", "not a whole number"),
])
def test_load_as_ndarray__csv_bad_integer_values_raise(tmp_path, value, match):
    with pytest.raises(OasisException, match=match):
        _load_keys(tmp_path, ["1,OSF,1,7,3", f"2,OSF,1,{value},3"])


def test_load_as_ndarray__csv_whole_number_floats_beyond_float64_precision(tmp_path):
    """uint64 ids above 2**53 are read as text, so float64 rounding can't change them."""
    dtype = np.dtype([('LocID', np.int32), ('AreaPerilID', np.uint64)])
    (tmp_path / "big.csv").write_text("LocID,AreaPerilID\n1,18446744073709551615.0\n2,9007199254740993\n")
    result = load_as_ndarray(str(tmp_path), "big", dtype)
    assert result["AreaPerilID"].tolist() == [18446744073709551615, 9007199254740993]
