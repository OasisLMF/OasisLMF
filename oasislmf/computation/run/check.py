__all__ = [
    'CheckModel',
]

import json
import os
import shutil
import tempfile

from oasis_data_manager.filestore.config import get_storage_from_config_path

from ..base import ComputationStep
from ..generate.files import GenerateFiles
from ..hooks.pre_analysis import ExposurePreAnalysis
from ...utils.data import analysis_settings_loader, get_exposure_data, model_settings_loader
from ...utils.exceptions import OasisException
from ...utils.inputs import str2bool
from ...validation.model_check import CheckReport, run_model_check

RUN_INPUT_FILES = {f'{name}.{ext}' for name in ('events', 'event_rates', 'occurrence', 'periods', 'quantile', 'returnperiods')
                   for ext in ('bin', 'csv')}


class CheckModel(ComputationStep):
    """Check that a portfolio, the model data and the analysis settings work together before running losses.

    Generates the Oasis files (or uses ``check_inputs_dir``) in a scratch run directory, resolves the event, occurrence and model data files
    exactly as a loss run would, then cross-checks keys, items and coverages against the vulnerability, footprint,
    damage bin, event and loss factor files, and the lookup dictionaries against the model data.
    Raises if any ERROR level finding is reported.
    """
    step_params = [
        {'name': 'model_data_dir', 'flag': '-d', 'is_path': True, 'pre_exist': True, 'help': 'Model data directory path'},
        {'name': 'model_storage_json', 'is_path': True, 'pre_exist': True, 'required': False,
         'help': 'Model data storage settings JSON file path'},
        {'name': 'user_data_dir', 'flag': '-D', 'is_path': True, 'pre_exist': False,
         'help': 'Directory containing additional model data files which varies between analysis runs'},
        {'name': 'exposure_pre_analysis_module', 'required': False, 'is_path': True,
         'pre_exist': True, 'help': 'Exposure Pre-Analysis lookup module path'},
        {'name': 'check_inputs_dir', 'is_path': True, 'pre_exist': True,
         'help': 'Check these existing Oasis files instead of generating them from the exposure'},
        {'name': 'check_dir', 'is_path': True, 'pre_exist': False,
         'help': 'Directory for the generated files and report (default: temporary directory, removed afterwards)'},
        {'name': 'check_report_json', 'is_path': True, 'pre_exist': False, 'help': 'Write the findings to this JSON file'},
        {'name': 'check_max_events', 'type': int, 'default': 1000,
         'help': 'Scan this many footprint events, evenly spaced over the selected event set (0 = all events)'},
        {'name': 'full_model_check', 'type': str2bool, 'const': True, 'nargs': '?', 'default': False,
         'help': 'Check every vulnerability function, not just those referenced by the portfolio'},
        {'name': 'dynamic_footprint', 'type': str2bool, 'const': True, 'nargs': '?', 'default': False,
         'help': 'The model uses a dynamic footprint (intensity bins are computed at run time)'},
        {'name': 'check_warnings_as_errors', 'type': str2bool, 'const': True, 'nargs': '?', 'default': False,
         'help': 'Fail on WARNING level findings as well as ERROR'},
    ]
    chained_commands = [
        GenerateFiles,
        ExposurePreAnalysis,
    ]

    def get_exposure_data_config(self):
        return {
            'location': self.oed_location_csv,
            'account': self.oed_accounts_csv,
            'ri_info': self.oed_info_csv,
            'ri_scope': self.oed_scope_csv,
            'oed_schema_info': self.oed_schema_info if self.oed_schema_info is not None else self.settings.get('oed_version', None),
            'currency_conversion': self.currency_conversion_json,
            'check_oed': self.check_oed,
            'use_field': True,
            'location_numbers': self.location,
            'portfolio_numbers': self.portfolio,
            'account_numbers': self.account,
            'base_df_engine': self.base_df_engine,
            'exposure_df_engine': self.exposure_df_engine or self.base_df_engine,
            'backend_dtype': self.oed_backend_dtype,
            'supported_oed_versions': self.settings.get('data_settings', {}).get('supported_oed_versions'),
            'disable_oed_version_update': self.disable_oed_version_update,
        }

    @staticmethod
    def _link_dir(src_dir, dst_dir, copy_names=()):
        os.makedirs(dst_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            src = os.path.abspath(os.path.join(src_dir, name))
            dst = os.path.join(dst_dir, name)
            if os.path.lexists(dst):
                os.remove(dst)
            if name in copy_names:
                shutil.copy(src, dst)
            else:
                os.symlink(src, dst)

    def _prepare_static(self, run_dir):
        static_dir = os.path.join(run_dir, 'static')
        os.makedirs(static_dir)
        if self.model_storage_json:
            shutil.copy(self.model_storage_json, os.path.join(run_dir, 'model_storage.json'))
        else:
            self._link_dir(self.model_data_dir, static_dir)
        if self.user_data_dir and os.path.exists(self.user_data_dir):
            self._link_dir(self.user_data_dir, static_dir)
        return get_storage_from_config_path(self.model_storage_json, static_dir)

    def _generate_inputs(self, report, input_dir):
        os.makedirs(input_dir, exist_ok=True)
        self.oasis_files_dir = self.kwargs['oasis_files_dir'] = input_dir
        try:
            self.kwargs['exposure_data'] = get_exposure_data(self, add_internal_col=True)
            if self.exposure_pre_analysis_module:
                ExposurePreAnalysis(**self.kwargs).run()
            GenerateFiles(**self.kwargs).run()
            report.ok('inputs.generate')
            return True
        except Exception as e:
            self.logger.exception('oasis file generation failed')
            report.error('inputs.generate', f'oasis file generation failed, portfolio checks skipped: {type(e).__name__}: {e}')
            return False

    def run(self):
        if not self.model_data_dir and not self.model_storage_json:
            raise OasisException('model check needs model_data_dir or model_storage_json')
        run_dir = self.check_dir or tempfile.mkdtemp(prefix='oasis-check-')
        if os.path.exists(os.path.join(run_dir, 'input')) or os.path.exists(os.path.join(run_dir, 'static')):
            raise OasisException(f'check_dir {run_dir} already contains input/ or static/, use an empty directory')
        report = CheckReport()
        try:
            if self.check_inputs_dir:
                self._link_dir(self.check_inputs_dir, os.path.join(run_dir, 'input'), copy_names=RUN_INPUT_FILES)
                inputs_generated = True
            else:
                inputs_generated = self._generate_inputs(report, os.path.join(run_dir, 'input'))
            model_storage = self._prepare_static(run_dir)

            analysis_settings = analysis_settings_loader(self.analysis_settings_json) if self.analysis_settings_json else {}
            model_settings = model_settings_loader(self.model_settings_json) if self.model_settings_json else {}
            run_model_check(report, run_dir, model_storage, analysis_settings, model_settings, self.settings,
                            lookup_config_json=self.lookup_config_json, dynamic_footprint=self.dynamic_footprint,
                            max_events=self.check_max_events, full_model=self.full_model_check, check_portfolio=inputs_generated)
        finally:
            if not self.check_dir:
                shutil.rmtree(run_dir, ignore_errors=True)

        self.logger.info('\n' + report.format())
        if self.check_report_json:
            with open(self.check_report_json, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)

        if report.errors or (self.check_warnings_as_errors and report.warnings):
            raise OasisException(f'Model check failed: {len(report.errors)} errors, {len(report.warnings)} warnings')
        return report
