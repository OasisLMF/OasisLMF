__all__ = [
    'PostFileGen'
]

import json
import pathlib

from ..base import ComputationStep
from ...utils.data import get_exposure_data
from ...utils.inputs import str2bool
from ...utils.path import get_custom_module
from ...utils.exceptions import OasisException


class PostFileGen(ComputationStep):
    """
    Computation step that will be call just after oasis file generation.
    On the platform it will be called on a single machine before the files are copied on the several worker for the loss calculation
    Add the ability to specify a model specific step that will modify or expand on the loss calculation input file
    """
    step_params = [{'name': 'post_file_gen_module', 'required': True, 'is_path': True, 'pre_exist': True,
                    'help': 'post file generation lookup module path'},
                   {'name': 'post_file_gen_class_name', 'default': 'PostFileGen',
                    'help': 'Name of the class to use for the pre loss calculation'},
                   {'name': 'post_file_gen_setting_json', 'is_path': True, 'pre_exist': True,
                    'help': 'post file generation config JSON file path'},
                   {'name': 'oed_schema_info', 'is_path': True, 'pre_exist': True, 'help': 'path to custom oed_schema'},
                   {'name': 'oed_location_csv', 'flag': '-x', 'is_path': True, 'pre_exist': True, 'help': 'Source location CSV file path'},
                   {'name': 'oed_accounts_csv', 'flag': '-y', 'is_path': True, 'pre_exist': True, 'help': 'Source accounts CSV file path'},
                   {'name': 'oed_info_csv', 'flag': '-i', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance info. CSV file path'},
                   {'name': 'oed_scope_csv', 'flag': '-s', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance scope CSV file path'},
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
                   {'name': 'user_data_dir', 'flag': '-D', 'is_path': True, 'pre_exist': False,
                    'help': 'Directory containing additional model data files which varies between analysis runs'},
                   ]

    run_dir_key = 'pre-loss'

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
        import post_file_gen_module and call the run method
        """
        exposure_data = get_exposure_data(self, add_internal_col=True)
        kwargs = dict()

        # If given a value for 'oasis_files_dir' then use that directly
        if self.oasis_files_dir:
            input_dir = self.oasis_files_dir
        else:
            input_dir = self.get_default_run_dir()
            pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)

        kwargs['exposure_data'] = exposure_data
        kwargs['input_dir'] = input_dir
        kwargs['model_data_dir'] = self.model_data_dir
        kwargs['analysis_settings_json'] = self.analysis_settings_json
        kwargs['user_data_dir'] = self.user_data_dir
        kwargs['logger'] = self.logger

        if self.post_file_gen_setting_json:
            with open(self.post_file_gen_setting_json) as post_file_gen_setting_file:
                kwargs['post_file_gen_setting'] = json.load(post_file_gen_setting_file)
        else:
            kwargs['post_file_gen_setting'] = {}

        _module = get_custom_module(self.post_file_gen_module, 'post file gen module path')

        try:
            _class = getattr(_module, self.post_file_gen_class_name)
        except AttributeError as e:
            raise OasisException(f"class {self.post_file_gen_class_name} "
                                 f"is not defined in module {self.post_file_gen_module}") from e.__cause__

        _class_return = _class(**kwargs).run()

        return {
            "class": _class_return
        }
