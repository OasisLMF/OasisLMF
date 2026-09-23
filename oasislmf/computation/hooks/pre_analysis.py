__all__ = [
    'ExposurePreAnalysis'
]

import json
import pathlib
from collections import defaultdict
from ods_tools.oed import OED_TYPE_TO_NAME, PANDAS_COMPRESSION_MAP, UnknownColumnSaveOption

from ..base import ComputationStep
from ...utils.data import get_exposure_data, prepare_oed_exposure, analysis_settings_loader, model_settings_loader
from ...utils.inputs import str2bool
from ...utils.path import get_custom_module
from ...utils.exceptions import OasisException


def get_source_compression(oed_source):
    """Derive the compression/format to use when persisting a pre-analysis
    exposure snapshot, based on the oed_source's original source file extension.

    Exposure.save() defaults to csv whenever no explicit compression is given
    and the current source version has no recorded 'extension' (which is
    always true for a freshly loaded source - see ods_tools
    OedSource.from_filepath). Without this, the raw/adjusted exposure
    snapshots below are silently written as csv even when the original input
    was e.g. parquet, which can be drastically slower for large portfolios.

    A csv source is upgraded to parquet, since parquet is much more
    efficient to read/write for large portfolios and there is no reason to
    keep a snapshot in the slower format just because the original input
    happened to be csv. Any other recognized format is preserved as-is.

    Args:
        oed_source (OedSource): a single OED source of the loaded exposure data

    Returns:
        str or None: a key of ods_tools.oed.common.PANDAS_COMPRESSION_MAP
                      to save the source as, or None if the source's format
                      can't be determined (Exposure.save() then falls back
                      to its default of csv, unchanged from current
                      behaviour).
    """
    source = oed_source.current_source
    if source.get('source_type') != 'filepath':
        return None
    suffix = pathlib.Path(source['filepath']).suffix.lstrip('.').lower()
    for compression, mapped_suffix in PANDAS_COMPRESSION_MAP.items():
        if mapped_suffix.lstrip('.') == suffix:
            return 'parquet' if compression == 'csv' else compression
    return None


def save_exposure_data(exposure_data, path, version_name, save_config, unknown_columns):
    """Save each OED source of exposure_data preserving its own original file
    format, rather than forcing every source to the same one.

    OedExposure.save() only accepts a single 'compression' value which it
    applies to every source it saves, so a compression derived from one
    source (e.g. location) would silently force-convert the other sources
    (account, ri_info, ri_scope) to that same format. To avoid this, sources
    are grouped by their own derived compression and saved in separate
    calls, temporarily hiding the other sources from exposure_data so each
    call only saves its group.

    Args:
        exposure_data (OedExposure): the loaded exposure data
        path (str): output folder, passed through to OedExposure.save()
        version_name (str): passed through to OedExposure.save()
        save_config (bool): if true save the Exposure config as json, once
                             all sources have been saved
        unknown_columns (UnknownColumnSaveOption or Dict): passed through to
                                                             OedExposure.save()
    """
    original_sources = {oed_name: getattr(exposure_data, oed_name) for oed_name in OED_TYPE_TO_NAME.values()}

    sources_by_compression = defaultdict(list)
    for oed_name, oed_source in original_sources.items():
        if oed_source:
            sources_by_compression[get_source_compression(oed_source)].append(oed_name)

    try:
        for compression, oed_names in sources_by_compression.items():
            for oed_name, oed_source in original_sources.items():
                setattr(exposure_data, oed_name, oed_source if oed_name in oed_names else None)
            exposure_data.save(path=path, version_name=version_name, compression=compression,
                               save_config=False, unknown_columns=unknown_columns)
    finally:
        for oed_name, oed_source in original_sources.items():
            setattr(exposure_data, oed_name, oed_source)

    if save_config:
        exposure_data.save_config(pathlib.Path(path, exposure_data.DEFAULT_EXPOSURE_CONFIG_NAME))


class ExposurePreAnalysis(ComputationStep):
    """Computation step that will be call before the gulcalc.
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
    settings_params = [{'name': 'analysis_settings_json', 'loader': analysis_settings_loader, 'user_role': 'user'},
                       {'name': 'model_settings_json', 'loader': model_settings_loader}]

    step_params = [{'name': 'exposure_pre_analysis_module', 'required': True, 'is_path': True, 'pre_exist': True,
                    'help': 'Exposure Pre-Analysis lookup module path'},
                   {'name': 'exposure_pre_analysis_class_name', 'default': 'ExposurePreAnalysis',
                    'help': 'Name of the class to use for the exposure_pre_analysis'},
                   {'name': 'exposure_pre_analysis_setting_json', 'is_path': True, 'pre_exist': True,
                    'help': 'Exposure Pre-Analysis config JSON file path'},
                   {'name': 'oed_schema_info', 'help': 'Takes a version of OED schema to use in the form "v1.2.3" or a path to an OED schema json'},
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
                   {'name': 'oed_backend_dtype', 'type': str, 'default': 'pd_dtype',
                    'help': "define what type dtype the oed column will be (pd_dtype or pa_dtype)"},
                   {'name': 'disable_oed_version_update', 'type': str2bool, 'const': True, 'nargs': '?', 'default': False,
                    'help': 'Flag to disable automatic conversion of exposure data to the latest compatible OED version.'},
                   ]

    run_dir_key = 'pre-analysis'

    def get_exposure_data_config(self):
        return {
            'location': self.oed_location_csv,
            'account': self.oed_accounts_csv,
            'ri_info': self.oed_info_csv,
            'ri_scope': self.oed_scope_csv,
            'oed_schema_info': self.oed_schema_info if self.oed_schema_info is not None else self.settings.get('oed_version', None),
            'check_oed': self.check_oed,
            'use_field': True,
            'location_numbers': self.location,
            'portfolio_numbers': self.portfolio,
            'account_numbers': self.account,
            'base_df_engine': self.base_df_engine,
            'exposure_df_engine': self.exposure_df_engine,
            'backend_dtype': self.oed_backend_dtype,
            'supported_oed_versions': self.settings.get('data_settings', {}).get('supported_oed_versions'),
            'disable_oed_version_update': self.disable_oed_version_update,
        }

    def run(self):
        """Import exposure_pre_analysis_module and call the run method"""
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
        save_exposure_data(exposure_data, path=input_dir, version_name='raw', save_config=True, unknown_columns=ids_option)
        kwargs['exposure_data'] = exposure_data
        kwargs['input_dir'] = input_dir
        kwargs['model_data_dir'] = self.model_data_dir
        kwargs['user_data_dir'] = self.user_data_dir
        kwargs['analysis_settings_json'] = self.analysis_settings_json
        kwargs['settings'] = self.settings

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

        save_exposure_data(exposure_data, path=input_dir, version_name='', save_config=True, unknown_columns=ids_option)
        # regenerate ids
        exposure_data.location.dataframe = exposure_data.location.dataframe.drop(columns=['loc_id', 'loc_idx'])
        prepare_oed_exposure(exposure_data)

        modified_files = {oed_source.oed_name: str(oed_source.current_source['filepath']) for oed_source in exposure_data.get_oed_sources()}
        self.logger.info('\nPre-analysis modified files: {}'.format(
            json.dumps(modified_files, indent=4)))
        return {
            "class": _class_return,
            "modified": modified_files,
            "original": original_files,
        }
