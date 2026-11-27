"""Keys writers cast float ids of success rows back to integers (#2217)."""
import numpy as np
import pandas as pd
import pytest

from oasislmf.lookup.factory import BasicKeyServer
from oasislmf.utils.status import OASIS_KEYS_STATUS


@pytest.mark.parametrize("output_format, ext", [("oasis", "csv"), ("parquet", "parquet")])
def test_success_ids_written_as_integers_when_failed_rows_are_nan(tmp_path, output_format, ext):
    result = pd.DataFrame({
        'loc_id': [1, 2, 3], 'peril_id': ['OSF'] * 3, 'coverage_type': [1, 1, 1],
        'area_peril_id': [367923277, np.nan, 5],  # NaN on the failed row makes the column float
        'vulnerability_id': [20001, np.nan, 7],
        'status': [OASIS_KEYS_STATUS['success']['id'], OASIS_KEYS_STATUS['fail']['id'], OASIS_KEYS_STATUS['success']['id']],
        'message': ['', 'no match', ''],
    })
    assert result['area_peril_id'].dtype == float
    keys_fp = tmp_path / f'keys.{ext}'
    BasicKeyServer({'builtin_lookup_type': 'peril_covered_deterministic'}).write_keys_file(
        [result], successes_fp=str(keys_fp), errors_fp=str(tmp_path / f'keys-errors.{ext}'),
        output_format=output_format, keys_success_msg=False)

    if output_format == 'oasis':
        lines = keys_fp.read_text().splitlines()
        assert lines[1:] == ['1,OSF,1,367923277,20001', '3,OSF,1,5,7']
    else:
        keys = pd.read_parquet(keys_fp)
        assert pd.api.types.is_integer_dtype(keys['AreaPerilID'])
        assert keys['AreaPerilID'].tolist() == [367923277, 5]
