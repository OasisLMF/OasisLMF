"""Integer values that don't fit their target dtype raise instead of silently wrapping (#2197).

Each site test passes a value that, before the fix, was stored as a different, valid-looking
integer: areaperil_id 4294967298 as uint32 2, vulnerability_id 3000000000 as int32 -1294967296.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from oasis_data_manager.filestore.backends.local import LocalStorage

from oasislmf.pytools.common.data import correlations_dtype, item_adjustment_dtype, items_dtype
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.converters.csvtobin.utils.common import df_to_ndarray
from oasislmf.pytools.getmodel.common import Event_dtype
from oasislmf.pytools.getmodel.footprint import (FootprintCsv, OasisFootPrintError, df_to_numpy, validate_event_definition,
                                                 validate_hazard_case)
from oasislmf.pytools.getmodel.structure import build_structures
from oasislmf.preparation.gul_inputs import write_file
from oasislmf.utils.data import get_dataframe, prepare_account_df
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.integer_narrowing import astype_in_range, raise_if_out_of_range, widen_int_dtype, widen_int_dtypes

AREAPERIL_TOO_BIG = 4294967298      # uint32 2 when wrapped
VULNERABILITY_TOO_BIG = 3000000000  # int32 -1294967296 when wrapped


# --- the check itself ------------------------------------------------------------------------

@pytest.mark.parametrize('values, dtype', [
    (pd.Series([1, AREAPERIL_TOO_BIG]), 'u4'),
    (pd.Series([1, -1]), 'u4'),
    (pd.Series([VULNERABILITY_TOO_BIG]), 'i4'),
    (pd.Series([2 ** 64 - 1], dtype='u8'), 'i8'),
    (pd.Series([-1], dtype='i8'), 'u8'),
    (pd.Series([5000000000, None], dtype='Int64'), 'u4'),
    (pd.Series([2 ** 70], dtype=object), 'u8'),
    (np.array([2.0 ** 32]), 'u4'),
    (np.array([2.0 ** 32], dtype='f4'), 'u4'),
    (np.array([2.0 ** 31]), 'i4'),
    (np.array([-2.0 ** 31 - 1]), 'i4'),
    # float(int64 max) rounds up to 2**63: an inclusive float upper bound would let it through
    (pd.Series([2.0 ** 63]), 'i8'),
    (np.array([np.inf]), 'i4'),
])
def test_out_of_range_raises(values, dtype):
    with pytest.raises(OasisException, match='out of range'):
        raise_if_out_of_range(values, dtype, 'col', 'file.csv')


@pytest.mark.parametrize('values, dtype', [
    (pd.Series([0, 4294967295]), 'u4'),
    (pd.Series([-2 ** 31, 2 ** 31 - 1]), 'i4'),
    (pd.Series([1, None], dtype='Int64'), 'u4'),
    (np.array([2.0 ** 63 - 1024]), 'i8'),
    (np.array([-2.0 ** 63]), 'i8'),
    (np.array([1.0, np.nan]), 'i4'),  # NaN is not a range problem
    (pd.Series([3, None], dtype='category'), 'i4'),
    (pd.Series(['a', 'b']), 'i4'),    # left to the cast itself
    (pd.Series([2 ** 40]), 'f8'),     # not an integer target
    (pd.Series([2 ** 40]), 'category'),
    (np.array([], dtype='i8'), 'u4'),
])
def test_in_range_or_unchecked_passes(values, dtype):
    raise_if_out_of_range(values, dtype, 'col', 'file.csv')


def test_error_names_source_column_value_and_row():
    with pytest.raises(OasisException) as e:
        raise_if_out_of_range(pd.Series([1, 2, AREAPERIL_TOO_BIG], index=[10, 11, 12]), 'u4', 'areaperil_id', 'keys.csv')
    message = str(e.value)
    assert 'keys.csv' in message and 'areaperil_id' in message and str(AREAPERIL_TOO_BIG) in message
    assert 'index 12' in message and 'OASIS_AREAPERIL_TYPE=u8' in message


def test_astype_in_range():
    df = pd.DataFrame({'a': [1, AREAPERIL_TOO_BIG], 'b': ['x', 'y']})
    with pytest.raises(OasisException, match='a value 4294967298'):
        astype_in_range(df, {'a': 'u4', 'b': 'category'})
    assert astype_in_range(df, {'a': 'u8'})['a'].dtype == np.uint64
    assert astype_in_range(pd.Series([1, 2]), 'u1').tolist() == [1, 2]


def test_widen_int_dtypes():
    assert widen_int_dtype('u4') == np.int64
    assert widen_int_dtype(np.uint64) == np.uint64
    assert widen_int_dtype('category') == 'category'
    assert widen_int_dtype('str') == 'str'
    assert widen_int_dtypes(np.dtype([('a', 'i4'), ('b', 'f4')])) == {'a': np.int64, 'b': np.dtype('f4')}


# --- preparation: items.bin, correlations.bin, csv files ----------------------------------------

def items_df(areaperil_id=1, vulnerability_id=1):
    return pd.DataFrame({
        'item_id': np.array([1], dtype='i8'), 'coverage_id': np.array([1], dtype='i8'),
        'areaperil_id': np.array([areaperil_id], dtype='u8'),
        'vulnerability_id': np.array([vulnerability_id], dtype='i8'), 'group_id': np.array([1], dtype='i8'),
    })


@pytest.mark.parametrize('column, kwargs', [
    ('areaperil_id', {'areaperil_id': AREAPERIL_TOO_BIG}),
    ('vulnerability_id', {'vulnerability_id': VULNERABILITY_TOO_BIG}),
])
def test_items_bin_out_of_range(column, kwargs):
    with pytest.raises(OasisException, match=f'items.bin: {column} value'):
        df_to_ndarray(items_df(**kwargs), items_dtype, source='items.bin')


def test_items_bin_in_range():
    assert df_to_ndarray(items_df(4294967295, 2 ** 31 - 1), items_dtype)[0].tolist() == (1, 1, 4294967295, 2 ** 31 - 1, 1)


def test_correlations_bin_peril_correlation_group_out_of_range():
    df = pd.DataFrame({'item_id': [1], 'peril_correlation_group': [4294967297], 'damage_correlation_value': [0.5],
                       'hazard_group_id': [1], 'hazard_correlation_value': [0.5], 'source_item_id': [0]})
    with pytest.raises(OasisException, match='peril_correlation_group value 4294967297'):
        df_to_ndarray(df, correlations_dtype)


def test_item_adjustments_csv_out_of_range(tmp_path):
    df = pd.DataFrame({'item_id': [1], 'intensity_adjustment': [VULNERABILITY_TOO_BIG], 'return_period': [0]})
    file_path = str(tmp_path / 'item_adjustments.csv')
    with pytest.raises(OasisException, match='intensity_adjustment value 3000000000'):
        write_file(df, file_path, {name: item_adjustment_dtype[name] for name in item_adjustment_dtype.names})


# --- keys and account files -------------------------------------------------------------------

def test_get_dataframe_keys_vulnerability_id_out_of_range(tmp_path):
    keys_fp = tmp_path / 'keys.csv'
    keys_fp.write_text('LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID\n'
                       '1,WTC,1,5,1\n1,WTC,3,5,5000000000\n')
    with pytest.raises(OasisException, match='vulnerabilityid value 5000000000'):
        get_dataframe(src_fp=str(keys_fp), col_dtypes={'locid': 'int64', 'perilid': 'category',
                                                       'areaperilid': 'uint64', 'vulnerabilityid': 'uint32'})


def test_get_dataframe_matching_header_case_out_of_range(tmp_path):
    # with a header that matches the dtype keys, pandas reads straight into the narrow dtype
    fp = tmp_path / 'data.csv'
    fp.write_text('vulnerabilityid\n1\n5000000000\n')
    with pytest.raises(OasisException, match='vulnerabilityid value 5000000000'):
        get_dataframe(src_fp=str(fp), col_dtypes={'vulnerabilityid': 'uint32'})


def test_get_dataframe_in_range(tmp_path):
    fp = tmp_path / 'data.csv'
    fp.write_text('VulnerabilityID,PerilID\n4294967295,WTC\n')
    df = get_dataframe(src_fp=str(fp), col_dtypes={'vulnerabilityid': 'uint32', 'perilid': 'category'})
    assert df['vulnerabilityid'].dtype == np.uint32 and df['vulnerabilityid'].tolist() == [4294967295]


def test_account_layer_id_out_of_range():
    accounts_df = pd.DataFrame({'PortNumber': ['1', '1'], 'AccNumber': ['1', '1'], 'PolNumber': ['1', '2'],
                                'LayerNumber': [1, 1], 'layer_id': [1, 4294967297]})
    with pytest.raises(OasisException, match='layer_id value 4294967297'):
        prepare_account_df(accounts_df)


def test_build_structures_keys_out_of_range(tmp_path):
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'keys.csv').write_text(
        f'LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID\n1,WTC,1,{AREAPERIL_TOO_BIG},1\n')
    with pytest.raises(OasisException, match=f'AreaPerilID value {AREAPERIL_TOO_BIG}'):
        build_structures(str(tmp_path), [], ['WTC'])


# --- csvtobin ---------------------------------------------------------------------------------

@pytest.mark.parametrize('file_type, csv_text, message', [
    ('items', f'item_id,coverage_id,areaperil_id,vulnerability_id,group_id\n1,1,{AREAPERIL_TOO_BIG},1,1\n',
     f'areaperil_id value {AREAPERIL_TOO_BIG}'),
    ('items', f'item_id,coverage_id,areaperil_id,vulnerability_id,group_id\n1,1,1,{VULNERABILITY_TOO_BIG},1\n',
     f'vulnerability_id value {VULNERABILITY_TOO_BIG}'),
    ('items', 'item_id,coverage_id,areaperil_id,vulnerability_id,group_id\n1,1,-1,1,1\n', 'areaperil_id value -1'),
    ('amplifications', f'item_id,amplification_id\n1,{VULNERABILITY_TOO_BIG}\n', 'amplification_id value'),
    ('returnperiods', f'return_period\n{VULNERABILITY_TOO_BIG}\n', 'return_period value'),  # read_csv_as_ndarray
    ('items', 'item_id,coverage_id,areaperil_id,vulnerability_id,group_id\n1,1,18446744073709551617,1,1\n',
     'out of range for a 64-bit integer'),
])
def test_csvtobin_out_of_range(tmp_path, file_type, csv_text, message):
    Path(tmp_path, 'in.csv').write_text(csv_text)
    with pytest.raises(OasisException, match=message):
        csvtobin(Path(tmp_path, 'in.csv'), Path(tmp_path, 'out.bin'), file_type)


def test_csvtobin_items_in_range(tmp_path):
    Path(tmp_path, 'in.csv').write_text('item_id,coverage_id,areaperil_id,vulnerability_id,group_id\n1,1,4294967295,7,1\n')
    csvtobin(Path(tmp_path, 'in.csv'), Path(tmp_path, 'out.bin'), 'items')
    items = np.fromfile(Path(tmp_path, 'out.bin'), dtype=items_dtype)
    assert items['areaperil_id'].tolist() == [4294967295] and items['vulnerability_id'].tolist() == [7]


# --- footprint --------------------------------------------------------------------------------

def test_footprint_df_to_numpy_out_of_range():
    df = pd.DataFrame({'areaperil_id': [AREAPERIL_TOO_BIG], 'intensity_bin_id': [1], 'probability': [1.0]})
    with pytest.raises(OasisException, match=f'areaperil_id value {AREAPERIL_TOO_BIG}'):
        df_to_numpy(df, Event_dtype, source='footprint')


def test_footprint_csv_out_of_range(tmp_path):
    # areaperil 4294967298 used to be read as 2 and so matched item areaperil 2
    Path(tmp_path, 'footprint.csv').write_text(
        f'event_id,areaperil_id,intensity_bin_id,probability\n1,2,1,1.0\n1,{AREAPERIL_TOO_BIG},2,1.0\n')
    with FootprintCsv(LocalStorage(root_dir=str(tmp_path), cache_dir=None)) as footprint:
        with pytest.raises(OasisException, match=f'areaperil_id value {AREAPERIL_TOO_BIG}'):
            footprint.get_event(1)


def test_footprint_csv_in_range(tmp_path):
    Path(tmp_path, 'footprint.csv').write_text('event_id,areaperil_id,intensity_bin_id,probability\n1,4294967295,1,1.0\n')
    with FootprintCsv(LocalStorage(root_dir=str(tmp_path), cache_dir=None)) as footprint:
        assert footprint.get_event(1)['areaperil_id'].tolist() == [4294967295]


def test_dynamic_footprint_return_period_out_of_range():
    hazard_case = pd.DataFrame({'section_id': [1], 'areaperil_id': [1], 'return_period': [VULNERABILITY_TOO_BIG],
                                'intensity': [10]})
    with pytest.raises(OasisFootPrintError, match='return_period outside the int32 range'):
        validate_hazard_case(hazard_case)
    event_definition = pd.DataFrame({'event_id': [1], 'section_id': [1], 'rp_from': [10],
                                     'rp_to': [VULNERABILITY_TOO_BIG], 'interpolation': [0.5]})
    with pytest.raises(OasisFootPrintError, match='rp_to outside the int32 range'):
        validate_event_definition(event_definition)
