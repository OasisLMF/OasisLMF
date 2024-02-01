__all__ = [
    'ExposurePreAnalysis'
]

import json
import pathlib
from ods_tools.oed import UnknownColumnSaveOption

from ..base import ComputationStep
from ...utils.data import get_exposure_data, prepare_location_df
from ...utils.inputs import str2bool
from ...utils.path import get_custom_module
from ...utils.exceptions import OasisException


class ExposurePreAnalysis(ComputationStep):
    """
    Computation step that will be call before the gulcalc.
    Add the ability to specify a model specific pre-analysis hook for exposure modification,
    Allows OED to be processed by some custom code.
    Example of usage include geo-coding, exposure enhancement, or dis-aggregation...

    when the run method is call it will :
    - load the module specified at exposure_pre_analysis_module
    - init the class named exposure_pre_analysis_class_name with all the non null args in step_params as key arguments
    - call the method run of the object
    - return the output of the method

    you can find an example of such custom module in OasisPyWind/custom_module/exposure_pre_analysis.py

    """
    step_params = [{'name': 'exposure_pre_analysis_module', 'required': True, 'is_path': True, 'pre_exist': True,
                    'help': 'Exposure Pre-Analysis lookup module path'},
                   {'name': 'exposure_pre_analysis_class_name', 'default': 'ExposurePreAnalysis',
                    'help': 'Name of the class to use for the exposure_pre_analysis'},
                   {'name': 'exposure_pre_analysis_setting_json', 'is_path': True, 'pre_exist': True,
                    'help': 'Exposure Pre-Analysis config JSON file path'},
                   {'name': 'oed_schema_info', 'is_path': True, 'pre_exist': True, 'help': 'path to custom oed_schema'},
                   {'name': 'oed_location_csv', 'is_path': True, 'pre_exist': True, 'help': 'Source location CSV file path'},
                   {'name': 'oed_accounts_csv', 'is_path': True, 'pre_exist': True, 'help': 'Source accounts CSV file path'},
                   {'name': 'oed_info_csv', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance info. CSV file path'},
                   {'name': 'oed_scope_csv', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance scope CSV file path'},
                   {'name': 'check_oed', 'type': str2bool, 'const': True, 'nargs': '?', 'default': True, 'help': 'if True check input oed files'},
                   {'name': 'oasis_files_dir', 'flag': '-o', 'is_path': True, 'pre_exist': False,
                    'help': 'Path to the directory in which to generate the Oasis files'},
                   {'name': 'location', 'type': str, 'nargs': '+', 'help': 'A set of locations to include in the files'},
                   {'name': 'portfolio', 'type': str, 'nargs': '+', 'help': 'A set of portfolios to include in the files'},
                   {'name': 'account', 'type': str, 'nargs': '+', 'help': 'A set of locations to include in the files'},
                   {'name': 'base_df_engine', 'type': str, 'default': 'oasis_data_manager.df_reader.reader.OasisPandasReader',
                    'help': 'The default dataframe reading engine to use when loading files'},
                   {'name': 'exposure_df_engine', 'type': str, 'default': None,
                    'help': 'The dataframe reading engine to use when loading exposure files'},
                   {'name': 'model_df_engine', 'type': str, 'default': None, 'help': 'The dataframe reading engine to use when loading model files'},
                   {'name': 'model_data_dir', 'flag': '-d', 'is_path': True, 'pre_exist': True, 'help': 'Model data directory path'},
                   {'name': 'analysis_settings_json', 'flag': '-a', 'is_path': True, 'pre_exist': True,
                    'help': 'Analysis settings JSON file path'},
                   ]

    run_dir_key = 'pre-analysis'

    def get_exposure_data_config(self):
        return {
            'location': self.oed_location_csv,
            'account': self.oed_accounts_csv,
            'ri_info': self.oed_info_csv,
            'ri_scope': self.oed_scope_csv,
            'oed_schema_info': self.oed_schema_info,
            'check_oed': self.check_oed,
            'use_field': True,
            'location_numbers': self.location,
            'portfolio_numbers': self.portfolio,
            'account_numbers': self.account,
            'base_df_engine': self.base_df_engine,
            'exposure_df_engine': self.exposure_df_engine,
        }

    def run(self):
        """
        import exposure_pre_analysis_module and call the run method
        """
        exposure_data = get_exposure_data(self, add_internal_col=True)
        kwargs = dict()

        # If given a value for 'oasis_files_dir' then use that directly
        if self.oasis_files_dir:
            input_dir = self.oasis_files_dir
        else:
            input_dir = self.get_default_run_dir()
            pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)

        ids_option = {'loc_id': UnknownColumnSaveOption.DELETE,
                      'loc_idx': UnknownColumnSaveOption.DELETE}
        exposure_data.save(path=input_dir, version_name='raw', save_config=True, unknown_columns=ids_option)
        kwargs['exposure_data'] = exposure_data
        kwargs['input_dir'] = input_dir
        kwargs['model_data_dir'] = self.model_data_dir
        kwargs['analysis_settings_json'] = self.analysis_settings_json

        if self.exposure_pre_analysis_setting_json:
            with open(self.exposure_pre_analysis_setting_json) as exposure_pre_analysis_setting_file:
                kwargs['exposure_pre_analysis_setting'] = json.load(exposure_pre_analysis_setting_file)

        _module = get_custom_module(self.exposure_pre_analysis_module, 'Exposure Pre-Analysis lookup module path')

        try:
            _class = getattr(_module, self.exposure_pre_analysis_class_name)
        except AttributeError as e:
            raise OasisException(f"class {self.exposure_pre_analysis_class_name} "
                                 f"is not defined in module {self.exposure_pre_analysis_module}") from e.__cause__

        original_files = {oed_source.oed_name: str(oed_source.current_source['filepath']) for oed_source in exposure_data.get_oed_sources()}
        self.logger.info('\nPre-analysis original files: {}'.format(
            json.dumps(original_files, indent=4)))

        print(kwargs)
        print(_class(**kwargs))
        _class_return = _class(**kwargs).run()

        exposure_data.save(path=input_dir, version_name='', save_config=True, unknown_columns=ids_option)
        # regenerate ids
        exposure_data.location.dataframe = exposure_data.location.dataframe.drop(columns=['loc_id', 'loc_idx'])
        exposure_data.location.dataframe = prepare_location_df(exposure_data.location.dataframe)

        modified_files = {oed_source.oed_name: str(oed_source.current_source['filepath']) for oed_source in exposure_data.get_oed_sources()}
        self.logger.info('\nPre-analysis modified files: {}'.format(
            json.dumps(modified_files, indent=4)))
        return {
            "class": _class_return,
            "modified": modified_files,
            "original": original_files,
        }
