__all__ = [
    'RunModel'
]

import os
from tqdm import tqdm

from ods_tools.oed.setting_schema import ModelSettingSchema, AnalysisSettingSchema

from ..base import ComputationStep

from ..generate.files import GenerateFiles
from ..generate.losses import GenerateLosses
from ..hooks.pre_analysis import ExposurePreAnalysis
from ..hooks.post_analysis import PostAnalysis

from ...utils.data import get_exposure_data
from ...utils.path import empty_dir


class RunModel(ComputationStep):
    """
    Run models end to end.
    """

    # Override params
    step_params = [
        {'name': 'oasis_files_dir', 'flag': '-o', 'is_path': True, 'pre_exist': False,
            'help': 'Path to the directory in which to generate the Oasis files'},
        {'name': 'exposure_pre_analysis_module', 'required': False, 'is_path': True,
            'pre_exist': True, 'help': 'Exposure Pre-Analysis lookup module path'},
        {'name': 'post_analysis_module', 'required': False, 'is_path': True, 'pre_exist': True,
         'help': 'Post-Analysis module path'},
    ]
    # Add params from each sub command not in 'step_params'
    chained_commands = [
        GenerateLosses,
        GenerateFiles,
        ExposurePreAnalysis,
        PostAnalysis,
    ]

    def get_exposure_data_config(self):
        return {
            'location': self.oed_location_csv,
            'account': self.oed_accounts_csv,
            'ri_info': self.oed_info_csv,
            'ri_scope': self.oed_scope_csv,
            'oed_schema_info': self.oed_schema_info,
            'currency_conversion': self.currency_conversion_json,
            'check_oed': self.check_oed,
            'use_field': True
        }

    def run(self):

        # setup output dir
        if not self.model_run_dir:
            self.model_run_dir = GenerateLosses._get_output_dir(self)
        if os.path.exists(self.model_run_dir):
            empty_dir(self.model_run_dir)
        os.makedirs(os.path.join(self.model_run_dir, 'input'))

        self.kwargs['model_run_dir'] = self.model_run_dir
        # TODO: input oasis_files_dir is actually not use in the code
        self.kwargs['oasis_files_dir'] = os.path.join(self.model_run_dir, 'input')
        self.oasis_files_dir = self.kwargs['oasis_files_dir']

        # Validate JSON files (Fail at entry point not after input generation)
        AnalysisSettingSchema().validate_file(self.analysis_settings_json)
        if self.model_settings_json:
            ModelSettingSchema().validate_file(self.model_settings_json)

        self.kwargs['exposure_data'] = get_exposure_data(self, add_internal_col=True)

        # Run chain
        cmds = []
        if self.exposure_pre_analysis_module:
            cmds += [(ExposurePreAnalysis, self.kwargs)]
        cmds += [(GenerateFiles, self.kwargs), (GenerateLosses, self.kwargs)]
        if self.post_analysis_module:
            cmds += [(PostAnalysis, self.kwargs)]

        with tqdm(total=len(cmds)) as pbar:
            for cmd in cmds:
                cmd[0](**cmd[1]).run()
                pbar.update(1)

        self.logger.info('\nModel run completed successfully in {}'.format(self.model_run_dir))
