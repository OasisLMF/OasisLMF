__all__ = [
    'GenerateFiles',
    'GenerateDummyModelFiles',
    'GenerateDummyOasisFiles'
]

import io
import json
import os

from .keys import GenerateKeys, GenerateKeysDeterministic
from ..base import ComputationStep

#from ...utils.coverages import SUPPORTED_COVERAGE_TYPES
from ...preparation.oed import load_oed_dfs
from ...preparation.dir_inputs import (
    create_target_directory,
    prepare_input_files_directory
)
from ...preparation.reinsurance_layer import write_files_for_reinsurance
from ...utils.exceptions import OasisException
from ...utils.inputs import str2bool

from ...utils.data import (
    get_model_settings,
    get_location_df,
    get_dataframe,
    get_json,
    get_utctimestamp,
)
from ...utils.defaults import (
    get_default_accounts_profile,
    get_default_exposure_profile,
    get_default_fm_aggregation_profile,
    GROUP_ID_COLS,
    OASIS_FILES_PREFIXES,
    WRITE_CHUNKSIZE,
)
from ...preparation.gul_inputs import (
    get_gul_input_items,
    write_gul_input_files,
)
from ...preparation.il_inputs import (
    get_account_df,
    get_il_input_items,
    get_oed_hierarchy,
    write_il_input_files,
)
from ...preparation.summaries import (
    get_summary_mapping,
    merge_oed_to_mapping,
    write_mapping_file,
    write_exposure_summary,
    write_summary_levels,
)

from ..data.dummy_model.generate import (
    VulnerabilityFile,
    EventsFile,
    FootprintBinFile,
    FootprintIdxFile,
    DamageBinDictFile,
    OccurrenceFile,
    RandomFile,
    CoveragesFile,
    ItemsFile,
    FMProgrammeFile,
    FMPolicyTCFile,
    FMProfileFile,
    FMXrefFile,
    GULSummaryXrefFile,
    FMSummaryXrefFile
)


class GenerateFiles(ComputationStep):
    """
    Generates the standard Oasis GUL input files + optionally the IL/FM input
    files and the RI input files.
    """
    step_params = [
        # Command line options
        {'name': 'oasis_files_dir',            'flag':'-o', 'is_path': True, 'pre_exist': False, 'help': 'Path to the directory in which to generate the Oasis files'},
        {'name': 'keys_data_csv',              'flag':'-z', 'is_path': True, 'pre_exist': True,  'help': 'Pre-generated keys CSV file path'},
        {'name': 'keys_errors_csv',                         'is_path': True, 'pre_exist': True,  'help': 'Pre-generated keys errors CSV file path'},
        {'name': 'lookup_config_json',         'flag':'-m', 'is_path': True, 'pre_exist': False, 'help': 'Lookup config JSON file path'},
        {'name': 'lookup_data_dir',            'flag':'-k', 'is_path': True, 'pre_exist': True,  'help': 'Model lookup/keys data directory path'},
        {'name': 'lookup_module_path',         'flag':'-l', 'is_path': True, 'pre_exist': False, 'help': 'Model lookup module path'},
        {'name': 'lookup_complex_config_json', 'flag':'-L', 'is_path': True, 'pre_exist': False, 'help': 'Complex lookup config JSON file path'},
        {'name': 'lookup_num_processes',       'type':int,  'default': -1,                       'help': 'Number of workers in multiprocess pools'},
        {'name': 'lookup_num_chunks',          'type':int,  'default': -1,                       'help': 'Number of chunks to split the location file into for multiprocessing'},
        {'name': 'model_version_csv',          'flag':'-v', 'is_path': True, 'pre_exist': False, 'help': 'Model version CSV file path'},
        {'name': 'model_settings_json',        'flag':'-M', 'is_path': True, 'pre_exist': True,  'help': 'Model settings JSON file path'},
        {'name': 'user_data_dir',              'flag':'-D', 'is_path': True, 'pre_exist': False, 'help': 'Directory containing additional model data files which varies between analysis runs'},
        {'name': 'profile_loc_json',           'flag':'-e', 'is_path': True, 'pre_exist': True,  'help': 'Source (OED) exposure profile JSON path'},
        {'name': 'profile_acc_json',           'flag':'-b', 'is_path': True, 'pre_exist': True,  'help': 'Source (OED) accounts profile JSON path'},
        {'name': 'profile_fm_agg_json',        'flag':'-g', 'is_path': True, 'pre_exist': True,  'help': 'FM (OED) aggregation profile path'},
        {'name': 'oed_location_csv',           'flag':'-x', 'is_path': True, 'pre_exist': True,  'help': 'Source location CSV file path', 'required': True},
        {'name': 'oed_accounts_csv',           'flag':'-y', 'is_path': True, 'pre_exist': True,  'help': 'Source accounts CSV file path'},
        {'name': 'oed_info_csv',               'flag':'-i', 'is_path': True, 'pre_exist': True,  'help': 'Reinsurance info. CSV file path'},
        {'name': 'oed_scope_csv',              'flag':'-s', 'is_path': True, 'pre_exist': True,  'help': 'Reinsurance scope CSV file path'},
        {'name': 'disable_summarise_exposure', 'flag':'-S', 'default': False, 'type': str2bool, 'const':True, 'nargs':'?', 'help': 'Disables creation of an exposure summary report'},
        {'name': 'group_id_cols',              'flag':'-G', 'nargs':'+',                         'help': 'Columns from loc file to set group_id', 'default': GROUP_ID_COLS},

        # Manager only options (pass data directy instead of filepaths)
        {'name': 'lookup_config'},
        {'name': 'lookup_complex_config'},
        {'name': 'lookup_multiprocessing',        'default': True},
        {'name': 'write_ri_tree',                 'default': False},
        {'name': 'verbose',                       'default': False},
        {'name': 'write_chunksize', 'type':int,   'default': WRITE_CHUNKSIZE},
        {'name': 'oasis_files_prefixes',          'default': OASIS_FILES_PREFIXES},
        {'name': 'profile_loc',                   'default': get_default_exposure_profile()},
        {'name': 'profile_acc',                   'default': get_default_accounts_profile()},
        {'name': 'profile_fm_agg',                'default': get_default_fm_aggregation_profile()},
    ]

    def _get_output_dir(self):
        if self.oasis_files_dir:
            return self.oasis_files_dir
        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        return os.path.join(os.getcwd(), 'runs', 'files-{}'.format(utcnow))


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

        il = True if self.oed_accounts_csv else False
        ri = all([self.oed_info_csv, self.oed_scope_csv]) and il
        self.logger.info('\nGenerating Oasis files (GUL=True, IL={}, RIL={})'.format(il, ri))
        summarise_exposure = not self.disable_summarise_exposure

        # Prepare the target directory and copy the source files, profiles and
        # model version into it
        target_dir = prepare_input_files_directory(
            self._get_output_dir(),
            self.oed_location_csv,
            exposure_profile_fp=self.profile_loc_json,
            keys_fp=self.keys_data_csv,
            keys_errors_fp=self.keys_errors_csv,
            lookup_config_fp=self.lookup_config_json,
            model_version_fp=self.model_version_csv,
            complex_lookup_config_fp=self.lookup_complex_config_json,
            accounts_fp=self.oed_accounts_csv,
            accounts_profile_fp=self.profile_acc_json,
            fm_aggregation_profile_fp=self.profile_fm_agg_json,
            ri_info_fp=self.oed_info_csv,
            ri_scope_fp=self.oed_scope_csv
        )

        # Get the profiles defining the exposure and accounts files, ID related
        # terms in these files, and FM aggregation hierarchy
        location_profile = get_json(src_fp=self.profile_loc_json) if self.profile_loc_json else self.profile_loc
        accounts_profile = get_json(src_fp=self.profile_acc_json) if self.profile_acc_json else self.profile_acc
        oed_hierarchy = get_oed_hierarchy(location_profile, accounts_profile)
        loc_grp = oed_hierarchy['locgrp']['ProfileElementName'].lower()

        fm_aggregation_profile = (
            {int(k): v for k, v in get_json(src_fp=self.profile_fm_agg_json).items()} if self.profile_fm_agg_json  else
            self.profile_fm_agg
        )

        # Load Location file at a single point in the Generate files cmd
        location_df = get_location_df(self.oed_location_csv, location_profile)

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
            'locid': 'str',
            'perilid': 'str',
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


        # Columns from loc file to assign group_id
        model_group_fields = None
        if self.model_settings_json:
            try:
                model_group_fields = get_model_settings(
                    self.model_settings_json, key='data_settings'
                ).get('group_fields')
            except (KeyError, AttributeError, OasisException) as e:
                self.logger.warn('WARNING: Failed to load {} - {}'.format(self.model_settings_json, e))

        # load group columns from model_settings.json if not set in kwargs (CLI)
        if model_group_fields and not self.kwargs.get('group_id_cols'):
            group_id_cols = model_group_fields
        # otherwise load group cols from args
        else:
            group_id_cols = self.group_id_cols
        group_id_cols = list(map(lambda col: col.lower(), group_id_cols))

        # Get the GUL input items and exposure dataframes
        gul_inputs_df = get_gul_input_items(
            location_df,
            keys_df,
            exposure_profile=location_profile,
            group_id_cols=group_id_cols
        )

        # If not in det. loss gen. scenario, write exposure summary file
        if summarise_exposure:
            write_exposure_summary(
                target_dir,
                location_df,
                keys_fp=_keys_fp,
                keys_errors_fp=_keys_errors_fp,
                exposure_profile=location_profile
            )

        # If exposure summary set, write valid columns for summary levels to file
        if summarise_exposure:
            write_summary_levels(location_df, self.oed_location_csv, target_dir)

        # Write the GUL input files
        files_prefixes = self.oasis_files_prefixes
        gul_input_files = write_gul_input_files(
            gul_inputs_df,
            target_dir,
            oasis_files_prefixes=files_prefixes['gul'],
            chunksize=self.write_chunksize
        )
        gul_summary_mapping = get_summary_mapping(gul_inputs_df, oed_hierarchy)
        write_mapping_file(gul_summary_mapping, target_dir)

        # If no source accounts file path has been provided assume that IL
        # input files, and therefore also RI input files, are not needed
        if not il:
            # Write `summary_map.csv` for GUL only
            self.logger.info('\nOasis files generated: {}'.format(json.dumps(gul_input_files, indent=4)))
            return gul_input_files

        account_df = get_account_df(self.oed_accounts_csv, accounts_profile)

        # Get the IL input items
        il_inputs_df = get_il_input_items(
            location_df,
            gul_inputs_df,
            account_df,
            exposure_profile=location_profile,
            accounts_profile=accounts_profile,
            fm_aggregation_profile=fm_aggregation_profile
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

        # Write the RI input files, and write the returned RI layer info. as a
        # file, which can be reused by the model runner (in the model execution
        # stage) to set the number of RI iterations

        xref_descriptions_df = merge_oed_to_mapping(
            fm_summary_mapping,
            location_df,
            oed_column_set=[loc_grp],
            defaults={loc_grp: 1}
        ).sort_values(by='agg_id')

        ri_info_df, ri_scope_df, _ = load_oed_dfs(self.oed_info_csv, self.oed_scope_csv)
        ri_layers = write_files_for_reinsurance(
            gul_inputs_df,
            xref_descriptions_df,
            ri_info_df,
            ri_scope_df,
            oasis_files['fm_xref'],
            target_dir,
            self.write_ri_tree
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
        {'name': 'num_vulnerabilities',      'flag': '-v', 'required': True,  'type': int,                              'help': 'Number of vulnerabilities'},
        {'name': 'num_intensity_bins',       'flag': '-i', 'required': True,  'type': int,                              'help': 'Number of intensity bins'},
        {'name': 'num_damage_bins',          'flag': '-d', 'required': True,  'type': int,                              'help': 'Number of damage bins'},
        {'name': 'vulnerability_sparseness', 'flag': '-s', 'required': False, 'type': float,    'default': 1.0,         'help': 'Percentage of bins normalised to range [0,1] impacted for a vulnerability at an intensity level'},
        {'name': 'num_events',               'flag': '-e', 'required': True,  'type': int,                              'help': 'Number of events'},
        {'name': 'num_areaperils',           'flag': '-a', 'required': True,  'type': int,                              'help': 'Number of areaperils'},
        {'name': 'areaperils_per_event',     'flag': '-A', 'required': False, 'type': int,      'default': None,        'help': 'Number of areaperils impacted per event'},
        {'name': 'intensity_sparseness',     'flag': '-S', 'required': False, 'type': float,    'default': 1.0,         'help': 'Percentage of bins normalised to range [0,1] impacted for an event and areaperil'},
        {'name': 'no_intensity_uncertainty', 'flag': '-u', 'required': False, 'default': False, 'action': 'store_true', 'help': 'No intensity uncertainty flag'},
        {'name': 'num_periods',              'flag': '-p', 'required': True,  'type': int,                              'help': 'Number of periods'},
        {'name': 'num_randoms',              'flag': '-r', 'required': False, 'type': int,      'default': 0,           'help': 'Number of random numbers'},
        {'name': 'random_seed',              'flag': '-R', 'required': False, 'type': int,      'default': -1,          'help': 'Random seed (-1 for 1234 (default), 0 for current system time'}
    ]

    def _validate_input_arguments(self):
        if self.vulnerability_sparseness > 1.0 or self.vulnerability_sparseness < 0.0:
            raise OasisException('Invalid value for --vulnerability-sparseness')
        if self.intensity_sparseness > 1.0 or self.intensity_sparseness < 0.0:
            raise OasisException('Invlid value for --intensity-sparseness')
        if not self.areaperils_per_event:
            self.areaperils_per_event = self.num_areaperils
        if self.areaperils_per_event > self.num_areaperils:
            raise OasisException('Number of areaperils per event exceeds total number of areaperils')
        if self.random_seed < -1:
            raise OasisException('Invalid random seed')

    def _create_target_directory(self, label):
        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        target_dir =  os.path.join(os.getcwd(), 'runs', f'test-{label}-{utcnow}')
        self.target_dir = create_target_directory(
            target_dir, 'target test model files directory'
        )
        self.input_dir = self.target_dir
        self.static_dir = self.target_dir

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

    def _get_model_file_objects(self):

        # vulnerability.bin, events.bin, footprint.bin, footprint.idx,
        # damage_bin_dict.bin and occurrence.bin
        self._set_footprint_files_inputs()
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
            FootprintIdxFile(**self.footprint_files_inputs),
            DamageBinDictFile(self.num_damage_bins, self.static_dir),
            OccurrenceFile(
                self.num_events, self.num_periods, self.random_seed,
                self.input_dir
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
        self._get_model_file_objects()

        for model_file in self.model_files:
            model_file.write_file()

        self.logger.info(f'\nDummy Model files generated in {self.target_dir}')


class GenerateDummyOasisFiles(GenerateDummyModelFiles):
    """
    Generates dummy model and Oasis GUL input files + optionally the IL/FM
    input files.
    """

    step_params = [
        {'name': 'num_locations',          'flag': '-l', 'required': True,  'type': int,               'help': 'Number of locations'},
        {'name': 'coverages_per_location', 'flag': '-c', 'required': True,  'type': int,               'help': 'Number of coverage types per location'},
        {'name': 'num_layers',             'flag': '-L', 'required': False, 'type': int, 'default': 1, 'help': 'Number of layers'}
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
        self._get_model_file_objects()
        self._get_gul_file_objects()

        il = True   # Assume that FM files should be generated too
        if il is True:
            self._get_fm_file_objects()
        else:
            self.fm_files = []

        output_files = self.model_files + self.gul_files + self.fm_files
        for output_file in output_files:
            output_file.write_file()

        self.logger.info(f'\nDummy Model and Oasis files generated in {self.target_dir}')
