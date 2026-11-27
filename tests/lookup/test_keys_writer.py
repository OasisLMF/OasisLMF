"""Keys files keep integer ids integer when a lookup returns NaN ids for failed rows (#2217)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oasislmf.lookup.factory import BasicKeyServer, ids_as_int
from oasislmf.pytools.common.data import load_as_ndarray
from oasislmf.pytools.common.input_files import KEYS_DTYPE
from oasislmf.utils.status import OASIS_KEYS_STATUS

SUCCESS = OASIS_KEYS_STATUS['success']['id']
FAIL = OASIS_KEYS_STATUS['fail']['id']


def _lookup_result():
    return pd.DataFrame({
        'loc_id': [1, 2, 3], 'peril_id': ['OSF'] * 3, 'coverage_type': [1, 1, 3],
        'area_peril_id': [367923277, np.nan, 18], 'vulnerability_id': [20001, np.nan, 7],
        'status': [SUCCESS, FAIL, SUCCESS], 'message': ['', 'no match', ''],
    })


def test_ids_as_int__casts_whole_float_ids_and_leaves_others():
    df = pd.DataFrame({'area_peril_id': [1.0, 2.0], 'vulnerability_id': [1.0, np.nan], 'amplification_id': [1.5, 2.0],
                       'tiv': [1.0, 2.0]})
    out = ids_as_int(df)
    assert out['area_peril_id'].dtype == np.int64
    assert out['vulnerability_id'].dtype == np.float64
    assert out['amplification_id'].dtype == np.float64
    assert out['tiv'].dtype == np.float64
    assert df['area_peril_id'].dtype == np.float64  # input not modified


def test_ids_as_int__uint64_range():
    out = ids_as_int(pd.DataFrame({'area_peril_id': [2.0 ** 63]}))
    assert out['area_peril_id'].dtype == np.uint64
    assert out['area_peril_id'][0] == 2 ** 63


def test_write_oasis_keys_file__success_ids_are_ints(tmp_path):
    server = object.__new__(BasicKeyServer)
    successes_fp, errors_fp = tmp_path / 'keys.csv', tmp_path / 'keys-errors.csv'
    assert server.write_oasis_keys_file([_lookup_result()], False, successes_fp, errors_fp) == (2, 1)
    lines = successes_fp.read_text().splitlines()
    assert lines[1:] == ['1,OSF,1,367923277,20001', '3,OSF,3,18,7']
    keys = load_as_ndarray(str(tmp_path), 'keys', KEYS_DTYPE)
    assert keys['AreaPerilID'].tolist() == [367923277, 18]


def test_write_parquet_keys_file__success_ids_are_ints(tmp_path):
    server = object.__new__(BasicKeyServer)
    successes_fp, errors_fp = tmp_path / 'keys.parquet', tmp_path / 'keys-errors.parquet'
    assert server.write_parquet_keys_file([_lookup_result()], False, successes_fp, errors_fp) == (2, 1)
    schema = pq.read_schema(successes_fp)
    assert schema.field('AreaPerilID').type == 'int64'
    assert schema.field('VulnerabilityID').type == 'int64'
