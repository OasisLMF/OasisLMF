import io
import json
import os
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from ods_tools.oed import OedExposure

from oasislmf.manager import OasisManager
from oasislmf.utils.defaults import SOURCE_FILENAMES
from oasislmf.utils.exceptions import OasisException

input_oed_location = """PortNumber,AccNumber,LocNumber,BuildingTIV,CountryCode,LocPerilsCovered,LocCurrency
1,A11111,10002082046,1,UK,AA1,GBP
1,A11111,10002082047,2,UK,AA1,GBP
1,A11111,10002082048,3,UK,AA1,GBP
1,A11111,10002082049,4,UK,AA1,GBP
1,A11111,10002082050,5,UK,AA1,GBP
"""

output_oed_location = """PortNumber,AccNumber,LocNumber,BuildingTIV,CountryCode,LocPerilsCovered,LocCurrency
1,A11111,10002082046,2.0,UK,AA1,GBP
1,A11111,10002082047,4.0,UK,AA1,GBP
1,A11111,10002082048,6.0,UK,AA1,GBP
1,A11111,10002082049,8.0,UK,AA1,GBP
1,A11111,10002082050,10.0,UK,AA1,GBP
"""


def write_simple_epa_module(module_path, class_name='ExposurePreAnalysis'):
    with open(module_path, 'w') as f:
        f.write(f'''
class {class_name}:
    """
    Example of custum module called by oasislmf/model_preparation/ExposurePreAnalysis.py
    """

    def __init__(self, exposure_data, exposure_pre_analysis_setting, **kwargs):
        self.exposure_data = exposure_data
        self.exposure_pre_analysis_setting = exposure_pre_analysis_setting

    def run(self):
        self.exposure_data.location.dataframe['BuildingTIV'] = (self.exposure_data.location.dataframe['BuildingTIV']
                                                                * self.exposure_pre_analysis_setting['BuildingTIV_multiplyer'])
''')


def write_oed_location(oed_location_csv):
    with open(oed_location_csv, 'w') as f:
        f.write(input_oed_location)


def write_exposure_pre_analysis_setting_json(exposure_pre_analysis_setting_json):
    with open(exposure_pre_analysis_setting_json, 'w') as f:
        f.write('{"BuildingTIV_multiplyer":  2}')


def assert_location_snapshot_matches(d, expected_csv):
    # a csv source is upgraded to parquet by default (see get_source_compression),
    # so the adjusted snapshot is no longer written as location.csv
    assert not os.path.isfile(os.path.join(d, SOURCE_FILENAMES['oed_location_csv']))
    new_oed_location = pd.read_parquet(os.path.join(d, 'location.parquet'))
    expected_oed_location = pd.read_csv(io.StringIO(expected_csv))
    # OED fields are loaded as categoricals, so compare values as strings rather than dtypes
    pd.testing.assert_frame_equal(
        new_oed_location[expected_oed_location.columns].astype(str).reset_index(drop=True),
        expected_oed_location.astype(str))


def test_exposure_pre_analysis_simple_example():
    with TemporaryDirectory() as d:
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple.py'),
                  'oed_location_csv': os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_location_csv'])),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'])
        write_oed_location(kwargs['oed_location_csv'])
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        assert_location_snapshot_matches(d, output_oed_location)


def test_exposure_pre_analysis_class_name():
    with TemporaryDirectory() as d:
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_class_name': 'foobar',
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple_foobar.py'),
                  'oed_location_csv': os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_location_csv'])),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'], kwargs['exposure_pre_analysis_class_name'])
        write_oed_location(kwargs['oed_location_csv'])
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        assert_location_snapshot_matches(d, output_oed_location)


def write_oed_location_parquet(oed_location_parquet):
    pd.DataFrame({
        'PortNumber': [1, 1, 1, 1, 1],
        'AccNumber': ['A11111'] * 5,
        'LocNumber': [10002082046 + i for i in range(5)],
        'BuildingTIV': [1, 2, 3, 4, 5],
        'CountryCode': ['UK'] * 5,
        'LocPerilsCovered': ['AA1'] * 5,
        'LocCurrency': ['GBP'] * 5,
    }).to_parquet(oed_location_parquet, index=False)


def test_exposure_pre_analysis_preserves_parquet_source_format():
    """
    Regression test: when the original location source is parquet, the
    pre-analysis raw/adjusted exposure snapshots should also be written as
    parquet rather than being silently downgraded to csv, which is much
    slower for large portfolios (see get_source_compression).
    """
    with TemporaryDirectory() as d:
        oed_location_parquet = os.path.join(d, 'input_location.parquet')
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple.py'),
                  'oed_location_csv': oed_location_parquet,
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'])
        write_oed_location_parquet(oed_location_parquet)
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        assert os.path.isfile(os.path.join(d, 'raw_location.parquet')), \
            'expected raw exposure snapshot to be saved as parquet, not csv'
        assert os.path.isfile(os.path.join(d, 'location.parquet')), \
            'expected adjusted exposure snapshot to be saved as parquet, not csv'
        assert not os.path.isfile(os.path.join(d, 'raw_location.csv'))
        assert not os.path.isfile(os.path.join(d, 'location.csv'))

        new_oed_location = pd.read_parquet(os.path.join(d, 'location.parquet'))
        assert list(new_oed_location['BuildingTIV']) == [2.0, 4.0, 6.0, 8.0, 10.0]


def write_oed_account_csv(oed_account_csv, compression=None):
    df = pd.DataFrame({
        'PortNumber': [1],
        'AccNumber': ['A11111'],
        'PolNumber': ['P1'],
        'AccCurrency': ['GBP'],
        'PolPerilsCovered': ['AA1'],
    })
    df.to_csv(oed_account_csv, index=False, compression=compression)


def test_exposure_pre_analysis_preserves_per_source_format():
    """
    Regression test: each OED source should keep its own original file
    format, resolved independently. A csv location file is upgraded to
    parquet (see get_source_compression), but that shouldn't force-convert
    an account file that was given as gzip-compressed csv to parquet, or to
    plain csv, too (see save_exposure_data).
    """
    with TemporaryDirectory() as d:
        oed_location_csv = os.path.join(d, 'input_location.csv')
        oed_account_gz = os.path.join(d, 'input_account.csv.gz')
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple.py'),
                  'oed_location_csv': oed_location_csv,
                  'oed_accounts_csv': oed_account_gz,
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'])
        write_oed_location(oed_location_csv)
        write_oed_account_csv(oed_account_gz, compression='gzip')
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        assert os.path.isfile(os.path.join(d, 'raw_location.parquet')), \
            'expected csv location snapshot to be upgraded to parquet'
        assert os.path.isfile(os.path.join(d, 'location.parquet')), \
            'expected csv location snapshot to be upgraded to parquet'
        assert os.path.isfile(os.path.join(d, 'raw_account.gz')), \
            'expected gzip account snapshot to stay gzip, not be force-converted to parquet or csv'
        assert os.path.isfile(os.path.join(d, 'account.gz')), \
            'expected gzip account snapshot to stay gzip, not be force-converted to parquet or csv'
        assert not os.path.isfile(os.path.join(d, 'raw_account.parquet'))
        assert not os.path.isfile(os.path.join(d, 'account.parquet'))
        assert not os.path.isfile(os.path.join(d, 'raw_account.csv'))
        assert not os.path.isfile(os.path.join(d, 'account.csv'))


def test_exposure_pre_analysis_config_uses_relative_filepaths():
    """
    Regression test: the saved exposure config should record each source's
    filepath relative to the config file, as OedExposure.save() does when
    save_config is True, so the input dir can be moved or remounted and still
    reloaded. Saving the sources group by group (see save_exposure_data)
    must not turn these into absolute paths.
    """
    with TemporaryDirectory() as d:
        oed_location_csv = os.path.join(d, 'input_location.csv')
        oed_account_gz = os.path.join(d, 'input_account.csv.gz')
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple.py'),
                  'oed_location_csv': oed_location_csv,
                  'oed_accounts_csv': oed_account_gz,
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'])
        write_oed_location(oed_location_csv)
        write_oed_account_csv(oed_account_gz, compression='gzip')
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        with open(os.path.join(d, OedExposure.DEFAULT_EXPOSURE_CONFIG_NAME)) as config_file:
            config = json.load(config_file)

        assert set(config) >= {'location', 'account'}
        for oed_name in ('location', 'account'):
            filepath = config[oed_name]['sources'][config[oed_name]['cur_version_name']]['filepath']
            assert not os.path.isabs(filepath), f'expected {oed_name} filepath to be relative, got {filepath}'
            assert os.path.isfile(os.path.join(d, filepath))


def test_missing_module():
    with pytest.raises(OasisException, match="parameter exposure_pre_analysis_module is required for Computation Step ExposurePreAnalysis"):
        OasisManager().exposure_pre_analysis()


multi_account_oed_location = """PortNumber,AccNumber,LocNumber,BuildingTIV,CountryCode,LocPerilsCovered,LocCurrency
1,A11111,1,1,UK,AA1,GBP
1,A11111,2,2,UK,AA1,GBP
1,A22222,3,3,UK,AA1,GBP
1,A22222,4,4,UK,AA1,GBP
1,A33333,5,5,UK,AA1,GBP
1,A33333,6,6,UK,AA1,GBP
"""

multi_account_oed_account = """PortNumber,AccNumber,PolNumber,PolPerilsCovered,AccCurrency,LayerLimit
1,A11111,P1,AA1,GBP,10
1,A22222,P2,AA1,GBP,20
1,A33333,P3,AA1,GBP,30
"""


def write_account_aware_epa_module(module_path):
    with open(module_path, 'w') as f:
        f.write('''
class ExposurePreAnalysis:
    """
    Pre-analysis hook that touches both the location and account dataframes, so a
    chunked run can be checked for keeping each account's locations/account row together.
    """
    multiproc_enabled = True

    def __init__(self, exposure_data, exposure_pre_analysis_setting, **kwargs):
        self.exposure_data = exposure_data
        self.exposure_pre_analysis_setting = exposure_pre_analysis_setting

    def run(self):
        mult = self.exposure_pre_analysis_setting['BuildingTIV_multiplyer']
        loc_df = self.exposure_data.location.dataframe
        acc_df = self.exposure_data.account.dataframe

        # Every location in this chunk must share exactly one account.
        assert loc_df[['PortNumber', 'AccNumber']].drop_duplicates().shape[0] == acc_df.shape[0]

        loc_df['BuildingTIV'] = loc_df['BuildingTIV'] * mult
        acc_df['LayerLimit'] = acc_df['LayerLimit'] * mult
''')


def _write_multi_account_inputs(d, exposure_pre_analysis_setting_json):
    oed_location_csv = os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_location_csv']))
    oed_accounts_csv = os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_accounts_csv']))
    with open(oed_location_csv, 'w') as f:
        f.write(multi_account_oed_location)
    with open(oed_accounts_csv, 'w') as f:
        f.write(multi_account_oed_account)
    write_exposure_pre_analysis_setting_json(exposure_pre_analysis_setting_json)
    return oed_location_csv, oed_accounts_csv


@pytest.mark.parametrize('num_chunks', [1, 2, 3])
def test_exposure_pre_analysis_multiproc_chunking(num_chunks):
    """Chunked (num_chunks > 1) and single-process (num_chunks == 1) runs of an
    account-aware hook must produce equivalent merged location/account output."""
    with TemporaryDirectory() as d:
        exposure_pre_analysis_module = os.path.join(d, 'exposure_pre_analysis_account_aware.py')
        exposure_pre_analysis_setting_json = os.path.join(d, 'exposure_pre_analysis_setting.json')
        oed_location_csv, oed_accounts_csv = _write_multi_account_inputs(d, exposure_pre_analysis_setting_json)
        write_account_aware_epa_module(exposure_pre_analysis_module)

        kwargs = {
            'oasis_files_dir': d,
            'exposure_pre_analysis_module': exposure_pre_analysis_module,
            'oed_location_csv': oed_location_csv,
            'oed_accounts_csv': oed_accounts_csv,
            'exposure_pre_analysis_setting_json': exposure_pre_analysis_setting_json,
            'lookup_multiprocessing': True,
            'lookup_num_chunks': num_chunks,
            'lookup_num_processes': num_chunks,
            'check_oed': False,
        }

        OasisManager().exposure_pre_analysis(**kwargs)

        location_df = pd.read_parquet(os.path.join(d, 'location.parquet')).sort_values('LocNumber')
        account_df = pd.read_parquet(os.path.join(d, 'account.parquet')).sort_values('AccNumber')

        assert location_df['BuildingTIV'].tolist() == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        assert account_df['LayerLimit'].tolist() == [20, 40, 60]


def test_exposure_pre_analysis_multiproc_disabled_matches_singleproc():
    with TemporaryDirectory() as d:
        exposure_pre_analysis_module = os.path.join(d, 'exposure_pre_analysis_account_aware.py')
        exposure_pre_analysis_setting_json = os.path.join(d, 'exposure_pre_analysis_setting.json')
        oed_location_csv, oed_accounts_csv = _write_multi_account_inputs(d, exposure_pre_analysis_setting_json)
        write_account_aware_epa_module(exposure_pre_analysis_module)

        kwargs = {
            'oasis_files_dir': d,
            'exposure_pre_analysis_module': exposure_pre_analysis_module,
            'oed_location_csv': oed_location_csv,
            'oed_accounts_csv': oed_accounts_csv,
            'exposure_pre_analysis_setting_json': exposure_pre_analysis_setting_json,
            'lookup_multiprocessing': False,
            'check_oed': False,
        }

        OasisManager().exposure_pre_analysis(**kwargs)

        location_df = pd.read_parquet(os.path.join(d, 'location.parquet')).sort_values('LocNumber')
        account_df = pd.read_parquet(os.path.join(d, 'account.parquet')).sort_values('AccNumber')

        assert location_df['BuildingTIV'].tolist() == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        assert account_df['LayerLimit'].tolist() == [20, 40, 60]


def write_non_empty_account_aware_epa_module(module_path):
    with open(module_path, 'w') as f:
        f.write('''
class ExposurePreAnalysis:
    """
    Like the account-aware hook, but also fails loudly if it's ever invoked with an empty
    chunk - regression check for requesting more chunks than there are account groups.
    """
    multiproc_enabled = True

    def __init__(self, exposure_data, exposure_pre_analysis_setting, **kwargs):
        self.exposure_data = exposure_data
        self.exposure_pre_analysis_setting = exposure_pre_analysis_setting

    def run(self):
        mult = self.exposure_pre_analysis_setting['BuildingTIV_multiplyer']
        loc_df = self.exposure_data.location.dataframe
        acc_df = self.exposure_data.account.dataframe

        assert acc_df.shape[0] > 0, 'hook was dispatched an empty chunk'
        assert loc_df[['PortNumber', 'AccNumber']].drop_duplicates().shape[0] == acc_df.shape[0]

        loc_df['BuildingTIV'] = loc_df['BuildingTIV'] * mult
        acc_df['LayerLimit'] = acc_df['LayerLimit'] * mult
''')


def test_exposure_pre_analysis_num_chunks_clamped_to_group_count():
    """Requesting more chunks than there are (PortNumber, AccNumber) groups must not dispatch
    empty chunks to the hook (see the num_groups clamp in ExposurePreAnalysis.run)."""
    with TemporaryDirectory() as d:
        exposure_pre_analysis_module = os.path.join(d, 'exposure_pre_analysis_non_empty.py')
        exposure_pre_analysis_setting_json = os.path.join(d, 'exposure_pre_analysis_setting.json')
        oed_location_csv, oed_accounts_csv = _write_multi_account_inputs(d, exposure_pre_analysis_setting_json)
        write_non_empty_account_aware_epa_module(exposure_pre_analysis_module)

        kwargs = {
            'oasis_files_dir': d,
            'exposure_pre_analysis_module': exposure_pre_analysis_module,
            'oed_location_csv': oed_location_csv,
            'oed_accounts_csv': oed_accounts_csv,
            'exposure_pre_analysis_setting_json': exposure_pre_analysis_setting_json,
            'lookup_multiprocessing': True,
            'lookup_num_chunks': 10,  # only 3 (PortNumber, AccNumber) groups exist
            'lookup_num_processes': 10,
            'check_oed': False,
        }

        OasisManager().exposure_pre_analysis(**kwargs)

        location_df = pd.read_parquet(os.path.join(d, 'location.parquet')).sort_values('LocNumber')
        account_df = pd.read_parquet(os.path.join(d, 'account.parquet')).sort_values('AccNumber')

        assert location_df['BuildingTIV'].tolist() == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        assert account_df['LayerLimit'].tolist() == [20, 40, 60]


def test_exposure_pre_analysis_result_class_is_always_a_list():
    """result['class'] must have a consistent shape regardless of whether multiprocessing
    ran, so callers don't need to special-case a single value vs a list of values."""
    with TemporaryDirectory() as d:
        exposure_pre_analysis_module = os.path.join(d, 'exposure_pre_analysis_account_aware.py')
        exposure_pre_analysis_setting_json = os.path.join(d, 'exposure_pre_analysis_setting.json')
        oed_location_csv, oed_accounts_csv = _write_multi_account_inputs(d, exposure_pre_analysis_setting_json)
        write_account_aware_epa_module(exposure_pre_analysis_module)

        kwargs = {
            'exposure_pre_analysis_module': exposure_pre_analysis_module,
            'oed_location_csv': oed_location_csv,
            'oed_accounts_csv': oed_accounts_csv,
            'exposure_pre_analysis_setting_json': exposure_pre_analysis_setting_json,
            'check_oed': False,
        }

        result_singleproc = OasisManager().exposure_pre_analysis(
            oasis_files_dir=os.path.join(d, 'singleproc'), lookup_multiprocessing=False, **kwargs)
        assert isinstance(result_singleproc['class'], list)
        assert len(result_singleproc['class']) == 1

        result_multiproc = OasisManager().exposure_pre_analysis(
            oasis_files_dir=os.path.join(d, 'multiproc'), lookup_multiprocessing=True,
            lookup_num_chunks=3, lookup_num_processes=3, **kwargs)
        assert isinstance(result_multiproc['class'], list)
        assert len(result_multiproc['class']) == 3


def write_not_opted_in_epa_module(module_path):
    with open(module_path, 'w') as f:
        f.write('''
class ExposurePreAnalysis:
    """Hook that doesn't set multiproc_enabled, so must never be chunked."""

    def __init__(self, exposure_data, **kwargs):
        self.exposure_data = exposure_data

    def run(self):
        return self.exposure_data.location.dataframe.shape[0]
''')


def test_exposure_pre_analysis_hook_without_opt_in_runs_single_process():
    """A hook must set multiproc_enabled = True to be chunked, like a lookup class - otherwise
    it runs once on the whole portfolio even with lookup_multiprocessing enabled."""
    with TemporaryDirectory() as d:
        exposure_pre_analysis_module = os.path.join(d, 'exposure_pre_analysis_not_opted_in.py')
        exposure_pre_analysis_setting_json = os.path.join(d, 'exposure_pre_analysis_setting.json')
        oed_location_csv, oed_accounts_csv = _write_multi_account_inputs(d, exposure_pre_analysis_setting_json)
        write_not_opted_in_epa_module(exposure_pre_analysis_module)

        result = OasisManager().exposure_pre_analysis(
            oasis_files_dir=d,
            exposure_pre_analysis_module=exposure_pre_analysis_module,
            oed_location_csv=oed_location_csv,
            oed_accounts_csv=oed_accounts_csv,
            lookup_multiprocessing=True,
            lookup_num_chunks=3,
            lookup_num_processes=3,
            check_oed=False,
        )

        assert result['class'] == [6]


def write_slow_first_chunk_epa_module(module_path):
    with open(module_path, 'w') as f:
        f.write('''
import time


class ExposurePreAnalysis:
    """Opted-in hook whose first account finishes last, so chunk results arrive out of order."""
    multiproc_enabled = True

    def __init__(self, exposure_data, **kwargs):
        self.exposure_data = exposure_data

    def run(self):
        loc_df = self.exposure_data.location.dataframe
        if (loc_df['AccNumber'] == 'A11111').any():
            time.sleep(1)
        loc_df['BuildingTIV'] = loc_df['BuildingTIV'] * 2
''')


def test_exposure_pre_analysis_multiproc_output_order_matches_singleproc():
    """Chunk results are merged back in chunk order, not arrival order, so the saved location
    file is the same however the worker processes happen to be scheduled."""
    with TemporaryDirectory() as d:
        exposure_pre_analysis_module = os.path.join(d, 'exposure_pre_analysis_slow_first_chunk.py')
        exposure_pre_analysis_setting_json = os.path.join(d, 'exposure_pre_analysis_setting.json')
        oed_location_csv, oed_accounts_csv = _write_multi_account_inputs(d, exposure_pre_analysis_setting_json)
        write_slow_first_chunk_epa_module(exposure_pre_analysis_module)

        kwargs = {
            'exposure_pre_analysis_module': exposure_pre_analysis_module,
            'oed_location_csv': oed_location_csv,
            'oed_accounts_csv': oed_accounts_csv,
            'check_oed': False,
        }
        OasisManager().exposure_pre_analysis(
            oasis_files_dir=os.path.join(d, 'singleproc'), lookup_multiprocessing=False, **kwargs)
        OasisManager().exposure_pre_analysis(
            oasis_files_dir=os.path.join(d, 'multiproc'), lookup_multiprocessing=True,
            lookup_num_chunks=3, lookup_num_processes=3, **kwargs)

        singleproc_df = pd.read_parquet(os.path.join(d, 'singleproc', 'location.parquet'))
        multiproc_df = pd.read_parquet(os.path.join(d, 'multiproc', 'location.parquet'))
        assert multiproc_df['LocNumber'].astype(str).tolist() == ['1', '2', '3', '4', '5', '6']
        pd.testing.assert_frame_equal(multiproc_df, singleproc_df)


def write_noop_epa_module(module_path):
    with open(module_path, 'w') as f:
        f.write('''
class ExposurePreAnalysis:
    def __init__(self, exposure_data, **kwargs):
        self.exposure_data = exposure_data

    def run(self):
        pass
''')


def test_exposure_pre_analysis_sizes_partitions_from_location_row_count(monkeypatch):
    """Partition sizing must track the number of location rows (the real per-hook workload),
    not the number of account groups - a portfolio with few accounts but many locations per
    account should still be sized for chunking (see resolve_partition_count call in run)."""
    import oasislmf.computation.hooks.pre_analysis as pre_analysis_module

    captured = {}
    original_resolve = pre_analysis_module.resolve_partition_count

    def spy(row_count, num_cores, num_partitions, *args, **kwargs):
        captured['row_count'] = row_count
        return original_resolve(row_count, num_cores, num_partitions, *args, **kwargs)

    monkeypatch.setattr(pre_analysis_module, 'resolve_partition_count', spy)

    with TemporaryDirectory() as d:
        exposure_pre_analysis_module = os.path.join(d, 'exposure_pre_analysis_noop.py')
        write_noop_epa_module(exposure_pre_analysis_module)

        n_accounts = 5
        locs_per_account = 4
        oed_location_csv = os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_location_csv']))
        rows = [
            {
                'PortNumber': 1, 'AccNumber': f'A{acc_idx}',
                'LocNumber': acc_idx * locs_per_account + loc_idx + 1,
                'BuildingTIV': 1, 'CountryCode': 'UK',
                'LocPerilsCovered': 'AA1', 'LocCurrency': 'GBP',
            }
            for acc_idx in range(n_accounts)
            for loc_idx in range(locs_per_account)
        ]
        pd.DataFrame(rows).to_csv(oed_location_csv, index=False)

        kwargs = {
            'oasis_files_dir': d,
            'exposure_pre_analysis_module': exposure_pre_analysis_module,
            'oed_location_csv': oed_location_csv,
            'lookup_multiprocessing': True,
            'check_oed': False,
        }
        OasisManager().exposure_pre_analysis(**kwargs)

    assert captured.get('row_count') == n_accounts * locs_per_account


def write_always_raising_epa_module(module_path):
    with open(module_path, 'w') as f:
        f.write('''
class ExposurePreAnalysis:
    multiproc_enabled = True

    def __init__(self, exposure_data, exposure_pre_analysis_setting, **kwargs):
        self.exposure_data = exposure_data

    def run(self):
        raise ValueError('boom-for-test')
''')


def test_exposure_pre_analysis_multiproc_propagates_worker_exception():
    """If every chunked worker raises, the original exception must propagate out of
    run_pre_analysis_multiproc rather than being swallowed."""
    with TemporaryDirectory() as d:
        exposure_pre_analysis_module = os.path.join(d, 'exposure_pre_analysis_raising.py')
        exposure_pre_analysis_setting_json = os.path.join(d, 'exposure_pre_analysis_setting.json')
        oed_location_csv, oed_accounts_csv = _write_multi_account_inputs(d, exposure_pre_analysis_setting_json)
        write_always_raising_epa_module(exposure_pre_analysis_module)

        kwargs = {
            'oasis_files_dir': d,
            'exposure_pre_analysis_module': exposure_pre_analysis_module,
            'oed_location_csv': oed_location_csv,
            'oed_accounts_csv': oed_accounts_csv,
            'exposure_pre_analysis_setting_json': exposure_pre_analysis_setting_json,
            'lookup_multiprocessing': True,
            'lookup_num_chunks': 3,
            'lookup_num_processes': 3,
            'check_oed': False,
        }

        with pytest.raises(ValueError, match='boom-for-test'):
            OasisManager().exposure_pre_analysis(**kwargs)


def test_wrong_class():
    with TemporaryDirectory() as d:
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_class_name': 'foobar',
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple.py'),
                  'oed_location_csv': os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_location_csv'])),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'])
        write_oed_location(kwargs['oed_location_csv'])
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        with pytest.raises(OasisException, match=f"class {kwargs['exposure_pre_analysis_class_name']} "
                           f"is not defined in module {kwargs['exposure_pre_analysis_module']}"):
            OasisManager().exposure_pre_analysis(**kwargs)


def write_counting_epa_module(module_path, counter_path):
    with open(module_path, 'w') as f:
        f.write(f'''
class ExposurePreAnalysis:
    """
    Records every instantiation so the caller can assert it is built once.
    """

    def __init__(self, exposure_data, exposure_pre_analysis_setting, **kwargs):
        self.exposure_data = exposure_data
        self.exposure_pre_analysis_setting = exposure_pre_analysis_setting
        with open({counter_path!r}, 'a') as counter:
            counter.write('init\\n')

    def run(self):
        self.exposure_data.location.dataframe['BuildingTIV'] = (self.exposure_data.location.dataframe['BuildingTIV']
                                                                * self.exposure_pre_analysis_setting['BuildingTIV_multiplyer'])
''')


def test_exposure_pre_analysis_class_is_built_once(capsys):
    with TemporaryDirectory() as d:
        counter_path = os.path.join(d, 'init_count.txt')
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_counting.py'),
                  'oed_location_csv': os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_location_csv'])),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_counting_epa_module(kwargs['exposure_pre_analysis_module'], counter_path)
        write_oed_location(kwargs['oed_location_csv'])
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        with open(counter_path) as counter:
            assert counter.read() == 'init\n'

        assert_location_snapshot_matches(d, output_oed_location)

        assert 'exposure_pre_analysis_setting' not in capsys.readouterr().out
