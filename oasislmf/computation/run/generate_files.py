
__all__ = [
    'GenerateOasisFiles'
]

import os

from tqdm import tqdm

from ..base import ComputationStep
from ..generate.files import GenerateFiles
from ..hooks.pre_analysis import ExposurePreAnalysis
from ..hooks.post_file_gen import PostFileGen
from ...utils.data import get_exposure_data


class GenerateOasisFiles(ComputationStep):
    """
    Run Oasis file geneartion with optional PreAnalysis hook.
    """

    # Override params
    step_params = [
        {'name': 'exposure_pre_analysis_module', 'required': False, 'is_path': True,
            'pre_exist': True, 'help': 'Exposure Pre-Analysis lookup module path'},
        {'name': 'post_file_gen_module', 'required': False, 'is_path': True,
         'pre_exist': True, 'help': 'post-file gen hook module path'},
    ]
    # Add params from each sub command not in 'step_params'
    chained_commands = [
        PostFileGen,
        GenerateFiles,
        ExposurePreAnalysis,
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
            'use_field': True,
            'location_numbers': self.location,
            'portfolio_numbers': self.portfolio,
            'account_numbers': self.account,
            'base_df_engine': self.base_df_engine,
            'exposure_df_engine': self.exposure_df_engine or self.base_df_engine,
        }

    def run(self):
        # setup input dir
        if not self.oasis_files_dir:
            self.oasis_files_dir = GenerateFiles._get_output_dir(self)

        # create input dir
        if not os.path.exists(self.oasis_files_dir):
            os.makedirs(self.oasis_files_dir)

        self.kwargs['oasis_files_dir'] = self.oasis_files_dir
        self.kwargs['exposure_data'] = get_exposure_data(self, add_internal_col=True)

        # Run chain
        if self.exposure_pre_analysis_module:
            cmds = [(ExposurePreAnalysis, self.kwargs), (GenerateFiles, self.kwargs)]
        else:
            cmds = [(GenerateFiles, self.kwargs)]

        if self.post_file_gen_module:
            cmds += [(PostFileGen, self.kwargs)]

        with tqdm(total=len(cmds)) as pbar:
            for cmd in cmds:
                cmd[0](**cmd[1]).run()
                pbar.update(1)

        self.logger.info('\nGenerate Files completed successfully in {}'.format(self.oasis_files_dir))
