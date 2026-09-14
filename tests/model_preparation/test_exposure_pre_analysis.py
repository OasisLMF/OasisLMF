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
            'exposure_pre_analysis_multiprocessing': True,
            'exposure_pre_analysis_num_chunks': num_chunks,
            'exposure_pre_analysis_num_processes': num_chunks,
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
            'exposure_pre_analysis_multiprocessing': False,
            'check_oed': False,
        }

        OasisManager().exposure_pre_analysis(**kwargs)

        location_df = pd.read_csv(os.path.join(d, SOURCE_FILENAMES['oed_location_csv'])).sort_values('LocNumber')
        account_df = pd.read_csv(os.path.join(d, SOURCE_FILENAMES['oed_accounts_csv'])).sort_values('AccNumber')

        assert location_df['BuildingTIV'].tolist() == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        assert account_df['LayerLimit'].tolist() == [20, 40, 60]


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
