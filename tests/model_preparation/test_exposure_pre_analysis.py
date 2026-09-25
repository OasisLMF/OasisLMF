import os
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

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

        with open(os.path.join(d, SOURCE_FILENAMES['oed_location_csv'])) as new_oed_location_csv:
            new_oed_location_csv_data = new_oed_location_csv.read()
            assert new_oed_location_csv_data == output_oed_location


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

        with open(os.path.join(d, SOURCE_FILENAMES['oed_location_csv'])) as new_oed_location_csv:
            new_oed_location_csv_data = new_oed_location_csv.read()
            assert new_oed_location_csv_data == output_oed_location


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

        location_df = pd.read_csv(os.path.join(d, SOURCE_FILENAMES['oed_location_csv'])).sort_values('LocNumber')
        account_df = pd.read_csv(os.path.join(d, SOURCE_FILENAMES['oed_accounts_csv'])).sort_values('AccNumber')

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

        location_df = pd.read_csv(os.path.join(d, SOURCE_FILENAMES['oed_location_csv'])).sort_values('LocNumber')
        account_df = pd.read_csv(os.path.join(d, SOURCE_FILENAMES['oed_accounts_csv'])).sort_values('AccNumber')

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

        location_df = pd.read_csv(os.path.join(d, SOURCE_FILENAMES['oed_location_csv'])).sort_values('LocNumber')
        account_df = pd.read_csv(os.path.join(d, SOURCE_FILENAMES['oed_accounts_csv'])).sort_values('AccNumber')

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

        with open(os.path.join(d, SOURCE_FILENAMES['oed_location_csv'])) as new_oed_location_csv:
            assert new_oed_location_csv.read() == output_oed_location

        assert 'exposure_pre_analysis_setting' not in capsys.readouterr().out
