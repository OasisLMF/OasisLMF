"""A lookup may return `areaperil_id`, the spelling used by dynamic footprint files, for `area_peril_id` (#2216)."""
import pandas as pd
import pytest

from oasislmf.lookup.base import alias_area_peril_id
from oasislmf.lookup.builtin import Lookup
from oasislmf.lookup.factory import BasicKeyServer
from oasislmf.utils.status import OASIS_KEYS_STATUS

SUCCESS = OASIS_KEYS_STATUS['success']['id']
FAIL = OASIS_KEYS_STATUS['fail']['id']


def test_alias_renames_areaperil_id():
    df = pd.DataFrame({'loc_id': [1], 'areaperil_id': [7]})
    assert list(alias_area_peril_id(df).columns) == ['loc_id', 'area_peril_id']
    assert list(df.columns) == ['loc_id', 'areaperil_id']  # input not modified


def test_alias_keeps_area_peril_id_when_both_present():
    df = pd.DataFrame({'area_peril_id': [7], 'areaperil_id': [99]})
    assert alias_area_peril_id(df) is df


def test_alias_leaves_other_frames_alone():
    df = pd.DataFrame({'loc_id': [1], 'area_peril_id': [7]})
    assert alias_area_peril_id(df) is df


def test_builtin_lookup_accepts_areaperil_id():
    config = {
        "step_definition": {
            "pivot": {
                "type": "simple_pivot",
                "parameters": {"pivots": [{"new_cols": {
                    "peril_id": "WTC", "coverage_type": 1, "areaperil_id": 5, "vulnerability_id": 3}}]},
            },
        },
        "strategy": ["pivot"],
    }
    keys = Lookup(config=config).process_locations(pd.DataFrame({"loc_id": [1, 2]}))
    assert 'areaperil_id' not in keys.columns
    assert keys['area_peril_id'].tolist() == [5, 5]
    assert (keys['status'] == SUCCESS).all()


@pytest.mark.parametrize("output_format, read", [
    ("oasis", pd.read_csv),
    ("parquet", pd.read_parquet),
])
def test_keys_writer_accepts_areaperil_id_from_custom_lookups(tmp_path, output_format, read):
    ext = 'csv' if output_format == 'oasis' else 'parquet'
    result = pd.DataFrame({
        'loc_id': [1, 2], 'peril_id': ['WTC', 'WTC'], 'coverage_type': [1, 1],
        'areaperil_id': [11, -1], 'vulnerability_id': [3, -1],
        'status': [SUCCESS, FAIL], 'message': ['', 'no match'],
    })
    _, n_success, _, n_errors = BasicKeyServer({'builtin_lookup_type': 'peril_covered_deterministic'}).write_keys_file(
        [result], successes_fp=str(tmp_path / f'keys.{ext}'), errors_fp=str(tmp_path / f'keys-errors.{ext}'),
        output_format=output_format, keys_success_msg=False)

    assert (n_success, n_errors) == (1, 1)
    keys = read(tmp_path / f'keys.{ext}')
    assert keys['AreaPerilID'].tolist() == [11]
