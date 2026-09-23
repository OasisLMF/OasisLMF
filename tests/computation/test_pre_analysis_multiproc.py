import pandas as pd
import pytest

from oasislmf.computation.hooks.pre_analysis_multiproc import (
    _make_chunk_processor,
    _split_chunks,
    run_pre_analysis_multiproc,
)
from oasislmf.utils.exceptions import OasisException


class DummyLocation:
    def __init__(self, dataframe):
        self.dataframe = dataframe


class DummyAccount:
    def __init__(self, dataframe):
        self.dataframe = dataframe


class DummyReinsSource:
    def __init__(self, dataframe):
        self.dataframe = dataframe


class DummyExposureData:
    def __init__(self, location_df, account_df=None, ri_info_df=None, ri_scope_df=None):
        self.location = DummyLocation(location_df)
        self.account = DummyAccount(account_df) if account_df is not None else None
        self.ri_info = DummyReinsSource(ri_info_df) if ri_info_df is not None else None
        self.ri_scope = DummyReinsSource(ri_scope_df) if ri_scope_df is not None else None


class DummyHook:
    def __init__(self, exposure_data, multiplier=1, **kwargs):
        self.exposure_data = exposure_data
        self.multiplier = multiplier

    def run(self):
        self.exposure_data.location.dataframe['BuildingTIV'] *= self.multiplier
        if self.exposure_data.account is not None:
            self.exposure_data.account.dataframe['LayerLimit'] *= self.multiplier
        return 'ok'


def _make_loc_acc_df():
    loc_df = pd.DataFrame({
        'PortNumber': [1, 1, 1, 1],
        'AccNumber': ['A1', 'A1', 'A2', 'A2'],
        'loc_id': [1, 2, 3, 4],
        'BuildingTIV': [1, 2, 3, 4],
    })
    acc_df = pd.DataFrame({
        'PortNumber': [1, 1],
        'AccNumber': ['A1', 'A2'],
        'LayerLimit': [10, 20],
    })
    return loc_df, acc_df


def test_split_chunks_splits_by_group_cols():
    loc_df, acc_df = _make_loc_acc_df()

    parts = list(_split_chunks(loc_df, acc_df, 2, ['PortNumber', 'AccNumber']))

    assert len(parts) == 2
    seen_accounts = set()
    for loc_part, acc_part in parts:
        assert set(loc_part['AccNumber'].unique()) == set(acc_part['AccNumber'].unique())
        seen_accounts.update(acc_part['AccNumber'].tolist())
    assert seen_accounts == {'A1', 'A2'}


def test_split_chunks_includes_account_only_groups():
    """An account row whose (PortNumber, AccNumber) doesn't appear in loc_df at all (e.g. a
    genuinely locationless account) must still be assigned to a chunk, not silently dropped."""
    loc_df = pd.DataFrame({
        'PortNumber': [1, 1],
        'AccNumber': ['A1', 'A1'],
        'loc_id': [1, 2],
        'BuildingTIV': [1, 2],
    })
    acc_df = pd.DataFrame({
        'PortNumber': [1, 1],
        'AccNumber': ['A1', 'A2'],
        'LayerLimit': [10, 20],
    })

    parts = list(_split_chunks(loc_df, acc_df, 2, ['PortNumber', 'AccNumber']))

    seen_accounts = set()
    for _, acc_part in parts:
        seen_accounts.update(acc_part['AccNumber'].tolist())
    assert seen_accounts == {'A1', 'A2'}


def test_split_chunks_splits_by_loc_id_when_no_group_cols():
    loc_df = pd.DataFrame({'loc_id': [1, 2, 3, 4], 'BuildingTIV': [1, 2, 3, 4]})

    parts = list(_split_chunks(loc_df, None, 2, None))

    assert len(parts) == 2
    all_loc_ids = sorted(sum((p[0]['loc_id'].tolist() for p in parts), []))
    assert all_loc_ids == [1, 2, 3, 4]
    assert all(acc_part is None for _, acc_part in parts)


def test_chunk_processor_processes_chunk_with_account():
    loc_df = pd.DataFrame({'BuildingTIV': [1.0, 2.0]})
    acc_df = pd.DataFrame({'LayerLimit': [10]})
    exposure_data = DummyExposureData(loc_df, acc_df)

    process_chunk = _make_chunk_processor(exposure_data, DummyHook, {'multiplier': 2})
    loc_result, acc_result, class_return = process_chunk((loc_df.copy(), acc_df.copy()))

    assert loc_result['BuildingTIV'].tolist() == [2.0, 4.0]
    assert acc_result['LayerLimit'].tolist() == [20]
    assert class_return == 'ok'


def test_chunk_processor_processes_chunk_without_account():
    loc_df = pd.DataFrame({'BuildingTIV': [1.0]})
    exposure_data = DummyExposureData(loc_df, account_df=None)

    process_chunk = _make_chunk_processor(exposure_data, DummyHook, {'multiplier': 3})
    loc_result, acc_result, class_return = process_chunk((loc_df.copy(), None))

    assert loc_result['BuildingTIV'].tolist() == [3.0]
    assert acc_result is None


class MutatingRiInfoHook:
    def __init__(self, exposure_data, **kwargs):
        self.exposure_data = exposure_data

    def run(self):
        self.exposure_data.ri_info.dataframe['ReinsPeril'] = 'WTC'


class MutatingRiScopeHook:
    def __init__(self, exposure_data, **kwargs):
        self.exposure_data = exposure_data

    def run(self):
        self.exposure_data.ri_scope.dataframe['CededPercent'] = 1.0


def test_chunk_processor_raises_if_hook_mutates_ri_info():
    """ri_info isn't chunked or merged back, so a hook mutating it would otherwise have that
    change silently discarded - process_chunk should fail loudly instead."""
    loc_df = pd.DataFrame({'BuildingTIV': [1.0]})
    ri_info_df = pd.DataFrame({'ReinsNumber': [1]})
    exposure_data = DummyExposureData(loc_df, ri_info_df=ri_info_df)

    process_chunk = _make_chunk_processor(exposure_data, MutatingRiInfoHook, {})

    with pytest.raises(OasisException, match='ri_info'):
        process_chunk((loc_df.copy(), None))


def test_chunk_processor_raises_if_hook_mutates_ri_scope():
    loc_df = pd.DataFrame({'BuildingTIV': [1.0]})
    ri_scope_df = pd.DataFrame({'ScopeNumber': [1]})
    exposure_data = DummyExposureData(loc_df, ri_scope_df=ri_scope_df)

    process_chunk = _make_chunk_processor(exposure_data, MutatingRiScopeHook, {})

    with pytest.raises(OasisException, match='ri_scope'):
        process_chunk((loc_df.copy(), None))


def test_chunk_processor_does_not_raise_when_ri_info_untouched():
    """Sanity check: the guard should only trigger on an actual mutation, not just because
    ri_info/ri_scope are present."""
    loc_df = pd.DataFrame({'BuildingTIV': [1.0]})
    ri_info_df = pd.DataFrame({'ReinsNumber': [1]})
    ri_scope_df = pd.DataFrame({'ScopeNumber': [1]})
    exposure_data = DummyExposureData(loc_df, ri_info_df=ri_info_df, ri_scope_df=ri_scope_df)

    process_chunk = _make_chunk_processor(exposure_data, DummyHook, {'multiplier': 1})

    # must not raise
    process_chunk((loc_df.copy(), None))


@pytest.mark.parametrize('pool_count', [0, 1])
def test_run_pre_analysis_multiproc_rejects_pool_count_of_one_or_fewer(pool_count):
    """Guards against a caller forgetting to check pool_count > 1 before calling in - it should
    fail loudly rather than silently spin up a full process pool for a single chunk."""
    loc_df = pd.DataFrame({'PortNumber': [1], 'AccNumber': ['A1'], 'BuildingTIV': [1.0]})
    exposure_data = DummyExposureData(loc_df)

    with pytest.raises(ValueError, match='pool_count'):
        run_pre_analysis_multiproc(exposure_data, DummyHook, {}, pool_count, 1, ['PortNumber', 'AccNumber'])


def test_run_pre_analysis_multiproc_end_to_end():
    """A real (small) multiprocess run across group-based chunks, merged back together."""
    loc_df, acc_df = _make_loc_acc_df()
    exposure_data = DummyExposureData(loc_df, acc_df)

    location_df, account_df, class_returns = run_pre_analysis_multiproc(
        exposure_data, DummyHook, {'multiplier': 2}, pool_count=2, part_count=2,
        group_cols=['PortNumber', 'AccNumber'])

    assert sorted(location_df['BuildingTIV'].tolist()) == [2, 4, 6, 8]
    assert sorted(account_df['LayerLimit'].tolist()) == [20, 40]
    assert class_returns == ['ok', 'ok']
