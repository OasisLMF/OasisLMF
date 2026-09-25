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
