__all__ = [
    'GenerateFiles',
    'GenerateDummyModelFiles',
    'GenerateDummyOasisFiles'
]

import io
import json
import os
from pathlib import Path
from typing import List
import pandas as pd

from oasislmf.computation.base import ComputationStep
from oasislmf.computation.data.dummy_model.generate import (AmplificationsFile,
                                                            CoveragesFile,
                                                            DamageBinDictFile,
                                                            EventsFile,
                                                            FMPolicyTCFile,
                                                            FMProfileFile,
                                                            FMProgrammeFile,
                                                            FMSummaryXrefFile,
                                                            FMXrefFile,
                                                            FootprintBinFile,
                                                            GULSummaryXrefFile,
                                                            ItemsFile,
                                                            LossFactorsFile,
                                                            OccurrenceFile,
                                                            RandomFile,
                                                            VulnerabilityFile)
from oasislmf.computation.generate.keys import GenerateKeys
from oasislmf.preparation.correlations import map_data
from oasislmf.preparation.dir_inputs import (create_target_directory,
                                             prepare_input_files_directory)
from oasislmf.preparation.gul_inputs import (get_gul_input_items,
                                             process_group_id_cols,
                                             write_gul_input_files)
from oasislmf.preparation.il_inputs import (get_il_input_items,
                                            get_oed_hierarchy,
                                            write_il_input_files)
from oasislmf.preparation.reinsurance_layer import write_files_for_reinsurance
from oasislmf.preparation.summaries import (get_summary_mapping,
                                            merge_oed_to_mapping,
                                            write_exposure_summary,
                                            write_mapping_file,
                                            write_summary_levels)
from oasislmf.pytools.data_layer.oasis_files.correlations import \
    CorrelationsData
from oasislmf.utils.data import (establish_correlations, get_dataframe,
                                 get_exposure_data, get_json, get_utctimestamp,
                                 prepare_account_df,
                                 prepare_reinsurance_df, validate_vulnerability_replacements,
                                 analysis_settings_loader, model_settings_loader)

from oasislmf.utils.defaults import (DAMAGE_GROUP_ID_COLS,
                                     HAZARD_GROUP_ID_COLS,
                                     OASIS_FILES_PREFIXES, WRITE_CHUNKSIZE,
                                     get_default_accounts_profile,
                                     get_default_exposure_profile,
                                     get_default_fm_aggregation_profile)
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.inputs import str2bool


class GenerateFiles(ComputationStep):
    """
    Generates the standard Oasis GUL input files + optionally the IL/FM input
    files and the RI input files.
    """
    settings_params = [{'name': 'analysis_settings_json', 'loader': analysis_settings_loader, 'user_role': 'user'},
                       {'name': 'model_settings_json', 'loader': model_settings_loader}]

    step_params = [
        # Command line options
        {'name': 'oasis_files_dir', 'flag': '-o', 'is_path': True, 'pre_exist': False,
         'help': 'Path to the directory in which to generate the Oasis files'},
        {'name': 'keys_data_csv', 'flag': '-z', 'is_path': True, 'pre_exist': True, 'help': 'Pre-generated keys CSV file path'},
        {'name': 'analysis_settings_json', 'flag': '-a', 'is_path': True, 'pre_exist': True, 'required': False,
         'help': 'Analysis settings JSON file path'},
        {'name': 'keys_errors_csv', 'is_path': True, 'pre_exist': True, 'help': 'Pre-generated keys errors CSV file path'},
        {'name': 'profile_loc_json', 'is_path': True, 'pre_exist': True, 'help': 'Source (OED) exposure profile JSON path'},
        {'name': 'profile_acc_json', 'flag': '-b', 'is_path': True, 'pre_exist': True, 'help': 'Source (OED) accounts profile JSON path'},
        {'name': 'profile_fm_agg_json', 'is_path': True, 'pre_exist': True, 'help': 'FM (OED) aggregation profile path'},
        {'name': 'oed_schema_info', 'is_path': True, 'pre_exist': True, 'help': 'path to custom oed_schema'},
        {'name': 'currency_conversion_json', 'is_path': True, 'pre_exist': True, 'help': 'settings to perform currency conversion of oed files'},
        {'name': 'reporting_currency', 'help': 'currency to use in the results reported'},
        {'name': 'oed_location_csv', 'flag': '-x', 'is_path': True, 'pre_exist': True, 'help': 'Source location CSV file path'},
        {'name': 'oed_accounts_csv', 'flag': '-y', 'is_path': True, 'pre_exist': True, 'help': 'Source accounts CSV file path'},
        {'name': 'oed_info_csv', 'flag': '-i', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance info. CSV file path'},
        {'name': 'oed_scope_csv', 'flag': '-s', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance scope CSV file path'},
        {'name': 'check_oed', 'type': str2bool, 'const': True, 'nargs': '?', 'default': True, 'help': 'if True check input oed files'},
        {'name': 'disable_summarise_exposure', 'flag': '-S', 'default': False, 'type': str2bool, 'const': True, 'nargs': '?',
         'help': 'Disables creation of an exposure summary report'},
        {'name': 'damage_group_id_cols', 'flag': '-G', 'nargs': '+', 'help': 'Columns from loc file to set group_id', 'default': DAMAGE_GROUP_ID_COLS},
        {'name': 'hazard_group_id_cols', 'flag': '-H', 'nargs': '+', 'help': 'Columns from loc file to set hazard_group_id', 'default': HAZARD_GROUP_ID_COLS},
        {'name': 'lookup_multiprocessing', 'type': str2bool, 'const': False, 'nargs': '?', 'default': False,
         'help': 'Flag to enable/disable lookup multiprocessing'},
        {'name': 'do_disaggregation', 'type': str2bool, 'const': True, 'nargs': '?', 'default': True, 'help': 'if True run the oasis disaggregation.'},

        # Manager only options (pass data directy instead of filepaths)
        {'name': 'lookup_config'},
        {'name': 'lookup_complex_config'},
        {'name': 'write_ri_tree', 'default': False},
        {'name': 'verbose', 'default': False},
        {'name': 'write_chunksize', 'type': int, 'default': WRITE_CHUNKSIZE},
        {'name': 'oasis_files_prefixes', 'default': OASIS_FILES_PREFIXES},
        {'name': 'profile_loc', 'default': get_default_exposure_profile()},
        {'name': 'profile_acc', 'default': get_default_accounts_profile()},
        {'name': 'profile_fm_agg', 'default': get_default_fm_aggregation_profile()},
        {'name': 'location', 'type': str, 'nargs': '+', 'help': 'A set of locations to include in the files'},
        {'name': 'portfolio', 'type': str, 'nargs': '+', 'help': 'A set of portfolios to include in the files'},
        {'name': 'account', 'type': str, 'nargs': '+', 'help': 'A set of locations to include in the files'},
        {'name': 'base_df_engine', 'type': str, 'default': 'oasis_data_manager.df_reader.reader.OasisPandasReader',
         'help': 'The default dataframe reading engine to use when loading files'},
        {'name': 'exposure_df_engine', 'type': str, 'default': None,
         'help': 'The dataframe reading engine to use when loading exposure files'},
    ]

    chained_commands = [
        GenerateKeys
    ]

    def _get_output_dir(self):
        if self.oasis_files_dir:
            return self.oasis_files_dir
        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        return os.path.join(os.getcwd(), 'runs', 'files-{}'.format(utcnow))

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
            'exposure_df_engine': self.exposure_df_engine,
        }

    def run(self):
        self.logger.info('\nProcessing arguments - Creating Oasis Files')

        if not (self.keys_data_csv or self.lookup_config_json or (self.lookup_data_dir and self.model_version_csv and self.lookup_module_path)):
            raise OasisException(
                'No pre-generated keys file provided, and no lookup assets '
                'provided to generate a keys file - if you do not have a '
                'pre-generated keys file then lookup assets must be provided - '
                'for a built-in lookup the lookup config. JSON file path must '
                'be provided, or for custom lookups the keys data path + model '
                'version file path + lookup package path must be provided'
            )
        self.oasis_files_dir = self._get_output_dir()
        exposure_data = get_exposure_data(self, add_internal_col=True)
        self.kwargs['exposure_data'] = exposure_data

        il = bool(exposure_data.account)
        ri = exposure_data.ri_info and exposure_data.ri_scope and il
        self.logger.info('\nGenerating Oasis files (GUL=True, IL={}, RIL={})'.format(il, ri))
        summarise_exposure = not self.disable_summarise_exposure

        validate_vulnerability_replacements(self.analysis_settings_json)

        # Prepare the target directory and copy the source files, profiles and
        # model version into it
        target_dir = prepare_input_files_directory(
            target_dir=self.oasis_files_dir,
            exposure_data=exposure_data,
            exposure_profile_fp=self.profile_loc_json,
            keys_fp=self.keys_data_csv,
            keys_errors_fp=self.keys_errors_csv,
            lookup_config_fp=self.lookup_config_json,
            model_version_fp=self.model_version_csv,
            complex_lookup_config_fp=self.lookup_complex_config_json,
            accounts_profile_fp=self.profile_acc_json,
            fm_aggregation_profile_fp=self.profile_fm_agg_json,
        )
        # Get the profiles defining the exposure and accounts files, ID related
        # terms in these files, and FM aggregation hierarchy
        location_profile = get_json(src_fp=self.profile_loc_json) if self.profile_loc_json else self.profile_loc
        accounts_profile = get_json(src_fp=self.profile_acc_json) if self.profile_acc_json else self.profile_acc
        oed_hierarchy = get_oed_hierarchy(location_profile, accounts_profile)

        fm_aggregation_profile = get_json(src_fp=self.profile_fm_agg_json) if self.profile_fm_agg_json else self.profile_fm_agg

        # force fm_agg level keys to type int:
        if any(isinstance(lvl, str) for lvl in fm_aggregation_profile.keys()):
            fm_aggregation_profile = {int(k): v for k, v in fm_aggregation_profile.items()}

        if self.reporting_currency:
            exposure_data.reporting_currency = self.reporting_currency
            exposure_data.save(target_dir, self.reporting_currency, save_config=True)

        location_df = exposure_data.get_subject_at_risk_source().dataframe

        if il:
            exposure_data.account.dataframe = prepare_account_df(exposure_data.account.dataframe)
            account_df = exposure_data.account.dataframe
        else:
            account_df = None

        # If a pre-generated keys file path has not been provided,
        # then it is asssumed some model lookup assets have been provided, so
        # as to allow the lookup to be instantiated and called to generated
        # the keys file.
        _keys_fp = _keys_errors_fp = None
        if not self.keys_data_csv:
            _keys_fp = self.kwargs['keys_data_csv'] = os.path.join(target_dir, 'keys.csv')
            _keys_errors_fp = self.kwargs['keys_errors_csv'] = os.path.join(target_dir, 'keys-errors.csv')
            GenerateKeys(**self.kwargs).run()
        else:
            _keys_fp = os.path.join(target_dir, os.path.basename(self.keys_data_csv))
            if self.keys_errors_csv:
                _keys_errors_fp = os.path.join(target_dir, os.path.basename(self.keys_errors_csv))

        # Load keys file  **** WARNING - REFACTOR THIS ****
        dtypes = {
            'locid': 'int64',
            'perilid': 'category',
            'coveragetypeid': 'uint8',
            'areaperilid': 'uint64',
            'vulnerabilityid': 'uint32',
            'modeldata': 'str'
        }
        keys_error_fp = os.path.join(os.path.dirname(_keys_fp), 'keys-errors.csv') if _keys_fp else 'Missing'
        missing_keys_msg = 'No successful lookup results found in the keys file - '
        missing_keys_msg += 'Check the `keys-errors.csv` file for details. \n File path: {}'.format(keys_error_fp)
        keys_df = get_dataframe(
            src_fp=_keys_fp,
            col_dtypes=dtypes,
            empty_data_error_msg=missing_keys_msg,
            memory_map=True
        )
        # ************************************************

        # check that all loc_ids have been returned from keys lookup
        try:
            if self.keys_errors_csv:
                keys_errors_df = get_dataframe(src_fp=self.keys_errors_csv, memory_map=True)
            else:
                keys_errors_df = get_dataframe(src_fp=_keys_errors_fp, memory_map=True)
        except OasisException:
            # Assume empty file on read error.
            keys_errors_df = pd.DataFrame(columns=['locid'])

        returned_locid = set(keys_errors_df['locid']).union(set(keys_df['locid']))
        del keys_errors_df

        missing_ids = set(location_df['loc_id']).difference(returned_locid)
        if len(missing_ids) > 0:
            raise OasisException(f'Lookup error: missing "loc_id" values from keys return: {list(missing_ids)}')

        # Columns from loc file to assign group_id
        model_damage_group_fields = []
        model_hazard_group_fields = []

        # If analysis settings file contains correlation settings, they will overwrite the ones in model settings
        correlations_analysis_settings = self.settings.get('model_settings', {}).get('correlation_settings', None)

        model_settings = self.settings
        if correlations_analysis_settings is not None:
            model_settings['correlation_settings'] = correlations_analysis_settings
        correlations = establish_correlations(model_settings=model_settings)
        try:
            model_damage_group_fields = model_settings["data_settings"].get("damage_group_fields")
        except (KeyError, AttributeError, OasisException) as e:
            self.logger.warning(f'WARNING: Failed to load "damage_group_fields", file: {self.model_settings_json}, error: {e}')
        try:
            model_hazard_group_fields = model_settings["data_settings"].get("hazard_group_fields")
        except (KeyError, AttributeError, OasisException) as e:
            self.logger.warning(f'WARNING: Failed to load "hazard_group_fields", file: {self.model_settings_json}, error: {e}')

        # load group columns from model_settings.json if not set in kwargs (CLI)
        if model_damage_group_fields and not self.kwargs.get('group_id_cols'):
            damage_group_id_cols = model_damage_group_fields
        # otherwise load group cols from args
        else:
            damage_group_id_cols = self.damage_group_id_cols

        # load hazard group columns from model_settings.json if not set in kwargs (CLI)
        if model_hazard_group_fields and not self.kwargs.get('hazard_group_id_cols'):
            hazard_group_id_cols = model_hazard_group_fields
        # otherwise load group cols from args
        else:
            hazard_group_id_cols = self.hazard_group_id_cols

        damage_group_id_cols: List[str] = process_group_id_cols(group_id_cols=damage_group_id_cols,
                                                                exposure_df_columns=location_df,
                                                                has_correlation_groups=correlations)
        hazard_group_id_cols: List[str] = process_group_id_cols(group_id_cols=hazard_group_id_cols,
                                                                exposure_df_columns=location_df,
                                                                has_correlation_groups=correlations)
        gul_inputs_df = get_gul_input_items(
            location_df,
            keys_df,
            peril_correlation_group_df=map_data(data=model_settings, logger=self.logger),
            correlations=correlations,
            exposure_profile=location_profile,
            damage_group_id_cols=damage_group_id_cols,
            hazard_group_id_cols=hazard_group_id_cols,
            do_disaggregation=self.do_disaggregation
        )

        # If not in det. loss gen. scenario, write exposure summary file
        if summarise_exposure:
            write_exposure_summary(
                target_dir,
                location_df,
                keys_fp=_keys_fp,
                keys_errors_fp=_keys_errors_fp,
                exposure_profile=location_profile,
                additional_fields=(model_settings.get('model_settings', {})
                                   .get('summary_report_fields', []))
            )

        # If exposure summary set, write valid columns for summary levels to file
        if summarise_exposure:
            write_summary_levels(location_df, account_df, exposure_data, target_dir)

        # Write the GUL input files
        files_prefixes = self.oasis_files_prefixes

        gul_input_files = write_gul_input_files(
            gul_inputs_df,
            target_dir,
            correlations_df=gul_inputs_df[CorrelationsData.COLUMNS],
            output_dir=self._get_output_dir(),
            oasis_files_prefixes=files_prefixes['gul'],
            chunksize=self.write_chunksize,
        )
        gul_summary_mapping = get_summary_mapping(gul_inputs_df, oed_hierarchy)
        write_mapping_file(gul_summary_mapping, target_dir)
        del gul_summary_mapping
        # If no source accounts file path has been provided assume that IL
        # input files, and therefore also RI input files, are not needed
        if not il:
            # Write `summary_map.csv` for GUL only
            self.logger.info('\nOasis files generated: {}'.format(json.dumps(gul_input_files, indent=4)))
            return gul_input_files

        # Get the IL input items
        il_inputs_df = get_il_input_items(
            gul_inputs_df=gul_inputs_df.copy(),
            locations_df=exposure_data.location.dataframe if exposure_data.location is not None else None,
            accounts_df=exposure_data.account.dataframe,
            oed_schema=exposure_data.oed_schema,
            exposure_profile=location_profile,
            accounts_profile=accounts_profile,
            fm_aggregation_profile=fm_aggregation_profile,
            do_disaggregation=self.do_disaggregation,
        )

        # Write the IL/FM input files
        il_input_files = write_il_input_files(
            il_inputs_df,
            target_dir,
            oasis_files_prefixes=files_prefixes['il'],
            chunksize=self.write_chunksize
        )

        fm_summary_mapping = get_summary_mapping(il_inputs_df, oed_hierarchy, is_fm_summary=True)
        write_mapping_file(fm_summary_mapping, target_dir, is_fm_summary=True)

        # Combine the GUL and IL input file paths into a single dict (for convenience)
        oasis_files = {**gul_input_files, **il_input_files}

        # If no RI input file paths (info. and scope) have been provided then
        # no RI input files are needed, just return the GUL and IL Oasis files
        if not ri:
            self.logger.info('\nOasis files generated: {}'.format(json.dumps(oasis_files, indent=4)))
            return oasis_files

        exposure_data.ri_info.dataframe, exposure_data.ri_scope.dataframe = prepare_reinsurance_df(exposure_data.ri_info.dataframe,
                                                                                                   exposure_data.ri_scope.dataframe)

        # Write the RI input files, and write the returned RI layer info. as a
        # file, which can be reused by the model runner (in the model execution
        # stage) to set the number of RI iterations
        fm_summary_mapping['loc_id'] = fm_summary_mapping['loc_id'].astype(exposure_data.location.dataframe['loc_id'].dtype)
        xref_descriptions_df = merge_oed_to_mapping(
            fm_summary_mapping,
            exposure_data.location.dataframe,
            ['loc_id'], {'LocGroup': '', 'ReinsTag': '', 'CountryCode': ''})
        xref_descriptions_df[['PortNumber', 'AccNumber', 'PolNumber']] = xref_descriptions_df[['PortNumber', 'AccNumber', 'PolNumber']].astype(str)
        xref_descriptions_df = merge_oed_to_mapping(
            xref_descriptions_df,
            exposure_data.account.dataframe,
            ['PortNumber', 'AccNumber', 'PolNumber'],
            {
                'CedantName': '',
                'ProducerName': '',
                'LOB': '',
                'PolInceptionDate': '',
                'PolExpiryDate': '',
            }
        )
        xref_descriptions_df = xref_descriptions_df.sort_values(by='agg_id', kind='stable')

        del fm_summary_mapping
        self.kwargs['oed_info_csv'] = exposure_data.ri_info

        ri_layers = write_files_for_reinsurance(
            exposure_data.ri_info.dataframe,
            exposure_data.ri_scope.dataframe,
            xref_descriptions_df,
            target_dir,
            oasis_files['fm_xref'],
            self.logger
        )

        with io.open(os.path.join(target_dir, 'ri_layers.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(ri_layers, ensure_ascii=False, indent=4))
            oasis_files['ri_layers'] = os.path.abspath(f.name)
            for layer, layer_info in ri_layers.items():
                oasis_files['RI_{}'.format(layer)] = layer_info['directory']

        self.logger.info('\nOasis files generated: {}'.format(json.dumps(oasis_files, indent=4)))

        return oasis_files


class GenerateDummyModelFiles(ComputationStep):
    """
    Generates dummy model files.
    """

    # Command line options
    step_params = [
        {'name': 'target_dir', 'flag': '-o', 'is_path': True, 'pre_exist': False,
         'help': 'Path to the directory in which to generate the Model files'},
        {'name': 'num_vulnerabilities', 'flag': '-v', 'required': True, 'type': int, 'help': 'Number of vulnerabilities'},
        {'name': 'num_intensity_bins', 'flag': '-i', 'required': True, 'type': int, 'help': 'Number of intensity bins'},
        {'name': 'num_damage_bins', 'flag': '-d', 'required': True, 'type': int, 'help': 'Number of damage bins'},
        {'name': 'vulnerability_sparseness', 'flag': '-s', 'required': False, 'type': float, 'default': 1.0,
         'help': 'Percentage of bins normalised to range [0,1] impacted for a vulnerability at an intensity level'},
        {'name': 'num_events', 'flag': '-e', 'required': True, 'type': int, 'help': 'Number of events'},
        {'name': 'num_areaperils', 'flag': '-a', 'required': True, 'type': int, 'help': 'Number of areaperils'},
        {'name': 'areaperils_per_event', 'flag': '-A', 'required': False, 'type': int, 'default': None,
         'help': 'Number of areaperils impacted per event'},
        {'name': 'intensity_sparseness', 'flag': '-S', 'required': False, 'type': float, 'default': 1.0,
         'help': 'Percentage of bins normalised to range [0,1] impacted for an event and areaperil'},
        {'name': 'no_intensity_uncertainty', 'flag': '-u', 'required': False,
         'default': False, 'action': 'store_true', 'help': 'No intensity uncertainty flag'},
        {'name': 'num_periods', 'flag': '-p', 'required': True, 'type': int, 'help': 'Number of periods'},
        {'name': 'periods_per_event_mean', 'flag': '-P', 'required': False, 'type': int, 'default': 1,
         'help': 'Mean of truncated normal distribution sampled to determine number of periods per event'},
        {'name': 'periods_per_event_stddev', 'flag': '-Q', 'required': False, 'type': float, 'default': 0.0,
         'help': 'Standard deviation of truncated normal distribution sampled to determine number of periods per event'},
        {'name': 'num_amplifications', 'flag': '-m', 'required': False, 'type': int, 'default': 0, 'help': 'Number of amplifications'},
        {'name': 'min_pla_factor', 'flag': '-f', 'required': False, 'type': float, 'default': 0.875,
         'help': 'Minimum Post Loss Amplification Factor'},
        {'name': 'max_pla_factor', 'flag': '-F', 'required': False, 'type': float, 'default': 1.5,
         'help': 'Maximum Post Loss Amplification Factor'},
        {'name': 'num_randoms', 'flag': '-r', 'required': False, 'type': int, 'default': 0, 'help': 'Number of random numbers'},
        {'name': 'random_seed', 'flag': '-R', 'required': False, 'type': int, 'default': -
         1, 'help': 'Random seed (-1 for 1234 (default), 0 for current system time'}
    ]

    def _validate_input_arguments(self):
        if self.vulnerability_sparseness > 1.0 or self.vulnerability_sparseness < 0.0:
            raise OasisException('Invalid value for --vulnerability-sparseness')
        if self.intensity_sparseness > 1.0 or self.intensity_sparseness < 0.0:
            raise OasisException('Invalid value for --intensity-sparseness')
        if not self.areaperils_per_event:
            self.areaperils_per_event = self.num_areaperils
        if self.areaperils_per_event > self.num_areaperils:
            raise OasisException('Number of areaperils per event exceeds total number of areaperils')
        if self.num_amplifications < 0.0:
            raise OasisException('Invalid value for --num-amplifications')
        if self.max_pla_factor < self.min_pla_factor:
            raise OasisException('Value for --max-pla-factor must be greater than that for --min-pla-factor')
        if self.min_pla_factor < 0:
            raise OasisException('Invalid value for --min-pla-factor')
        if self.random_seed < -1:
            raise OasisException('Invalid random seed')

    def _create_target_directory(self, label):
        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        if not self.target_dir:
            self.target_dir = os.path.join(os.getcwd(), 'runs', f'test-{label}-{utcnow}')

        self.target_dir = create_target_directory(
            self.target_dir, 'target test model files directory'
        )

    def _prepare_run_directory(self):
        self.input_dir = os.path.join(self.target_dir, 'input')
        self.static_dir = os.path.join(self.target_dir, 'static')
        directories = [
            self.input_dir, self.static_dir
        ]
        for directory in directories:
            if not os.path.exists(directory):
                Path(directory).mkdir(parents=True, exist_ok=True)

    def _set_footprint_files_inputs(self):
        self.footprint_files_inputs = {
            'num_events': self.num_events,
            'num_areaperils': self.num_areaperils,
            'areaperils_per_event': self.areaperils_per_event,
            'num_intensity_bins': self.num_intensity_bins,
            'intensity_sparseness': self.intensity_sparseness,
            'no_intensity_uncertainty': self.no_intensity_uncertainty,
            'directory': self.static_dir
        }

    def _set_periods_per_event_parameters(self):
        self.periods_per_event_parameters = {
            'mean': self.periods_per_event_mean,
            'stddev': self.periods_per_event_stddev
        }

    def _get_model_file_objects(self):

        # vulnerability.bin, events.bin, footprint.bin, footprint.idx,
        # damage_bin_dict.bin and occurrence.bin
        self._set_footprint_files_inputs()
        self._set_periods_per_event_parameters()
        self.model_files = [
            VulnerabilityFile(
                self.num_vulnerabilities, self.num_intensity_bins,
                self.num_damage_bins, self.vulnerability_sparseness,
                self.random_seed, self.static_dir
            ),
            EventsFile(self.num_events, self.input_dir),
            FootprintBinFile(
                **self.footprint_files_inputs, random_seed=self.random_seed
            ),
            DamageBinDictFile(self.num_damage_bins, self.static_dir),
            OccurrenceFile(
                self.num_events, self.num_periods, self.random_seed,
                self.input_dir, **self.periods_per_event_parameters
            )
        ]
        if self.num_amplifications > 0:
            self.model_files += [
                LossFactorsFile(
                    self.num_events, self.num_amplifications,
                    self.min_pla_factor, self.max_pla_factor, self.random_seed,
                    self.static_dir
                )
            ]
        if self.num_randoms > 0:
            self.model_files += [
                RandomFile(self.num_randoms, self.random_seed, self.static_dir)
            ]

    def run(self):
        self.logger.info('\nProcessing arguments - Creating Dummy Model Files')

        self._validate_input_arguments()
        self._create_target_directory(label='files')
        self._prepare_run_directory()
        self._get_model_file_objects()

        for model_file in self.model_files:
            self.logger.info(f'Writing {model_file.file_name}')
            model_file.write_file()

        self.logger.info(f'\nDummy Model files generated in {self.target_dir}')


class GenerateDummyOasisFiles(GenerateDummyModelFiles):
    """
    Generates dummy model and Oasis GUL input files + optionally the IL/FM
    input files.
    """

    step_params = [
        {'name': 'num_locations', 'flag': '-l', 'required': True, 'type': int, 'help': 'Number of locations'},
        {'name': 'coverages_per_location', 'flag': '-c', 'required': True, 'type': int, 'help': 'Number of coverage types per location'},
        {'name': 'num_layers', 'required': False, 'type': int, 'default': 1, 'help': 'Number of layers'}
    ]
    chained_commands = [GenerateDummyModelFiles]

    def _validate_input_arguments(self):
        super()._validate_input_arguments()
        if self.coverages_per_location > 4 or self.coverages_per_location < 1:
            raise OasisException('Number of supported coverage types is 1 to 4')

    def _get_gul_file_objects(self):

        # coverages.bin, items.bin and gulsummaryxref.bin
        self.gul_files = [
            CoveragesFile(
                self.num_locations, self.coverages_per_location,
                self.random_seed, self.input_dir
            ),
            ItemsFile(
                self.num_locations, self.coverages_per_location,
                self.num_areaperils, self.num_vulnerabilities,
                self.random_seed, self.input_dir
            ),
            GULSummaryXrefFile(
                self.num_locations, self.coverages_per_location, self.input_dir
            )
        ]
        if self.num_amplifications > 0:
            self.gul_files += [
                AmplificationsFile(
                    self.num_locations, self.coverages_per_location,
                    self.num_amplifications, self.random_seed, self.input_dir
                )
            ]

    def _get_fm_file_objects(self):

        # fm_programme.bin, fm_policytc.bin, fm_profile.bin, fm_xref.bin and
        # fmsummaryxref.bin
        self.fm_files = [
            FMProgrammeFile(
                self.num_locations, self.coverages_per_location, self.input_dir
            ),
            FMPolicyTCFile(
                self.num_locations, self.coverages_per_location,
                self.num_layers, self.input_dir
            ),
            FMProfileFile(self.num_layers, self.input_dir),
            FMXrefFile(
                self.num_locations, self.coverages_per_location,
                self.num_layers, self.input_dir
            ),
            FMSummaryXrefFile(
                self.num_locations, self.coverages_per_location,
                self.num_layers, self.input_dir
            )
        ]

    def run(self):
        self.logger.info('\nProcessing arguments - Creating Model & Test Oasis Files')

        self._validate_input_arguments()
        self._create_target_directory(label='files')
        self._prepare_run_directory()
        self._get_model_file_objects()
        self._get_gul_file_objects()
        self._get_fm_file_objects()

        output_files = self.model_files + self.gul_files + self.fm_files
        for output_file in output_files:
            self.logger.info(f'Writing {output_file.file_name}')
            output_file.write_file()

        self.logger.info(f'\nDummy Model and Oasis files generated in {self.target_dir}')
