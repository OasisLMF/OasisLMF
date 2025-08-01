__all__ = [
    'GenerateKeys',
    'GenerateKeysDeterministic'
]

import os
from packaging import version

from ..base import ComputationStep
from ...lookup.factory import KeyServerFactory
from ...utils.exceptions import OasisException

from ...utils.inputs import str2bool
from ...utils.data import get_utctimestamp, get_exposure_data, analysis_settings_loader, model_settings_loader


class KeyComputationStep(ComputationStep):
    def get_exposure_data_config(self):
        return {
            'location': self.oed_location_csv,
            'oed_schema_info': self.oed_schema_info,
            'check_oed': self.check_oed,
            'use_field': True
        }


class GenerateKeys(KeyComputationStep):
    """
    Generates keys from a model lookup, and write Oasis keys and keys error files.

    The model lookup, which is normally independently implemented by the model
    supplier, should generate keys as dicts with the following format
    ::

        {
            "id": <loc. ID>,
            "peril_id": <OED sub-peril ID>,
            "coverage_type": <OED coverage type ID>,
            "area_peril_id": <area peril ID>,
            "vulnerability_id": <vulnerability ID>,
            "message": <loc. lookup status message>,
            "status": <loc. lookup status flag indicating success, failure or no-match>
        }

    The keys generation command can generate these dicts, and write them to
    file. It can also be used to write these to an Oasis keys file (which is a
    requirement for model execution), which has the following format.::

        LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID
        ..
        ..
    This file only lists the locations for which there has been a successful
    lookup. The keys errors file lists all the locations with failing or
    non-matching lookups and has the following format::

        LocID,PerilID,CoverageTypeID,Message
        ..
        ..
    """
    settings_params = [{'name': 'lookup_complex_config_json', 'loader': analysis_settings_loader, 'user_role': 'user'},
                       {'name': 'model_settings_json', 'loader': model_settings_loader}]

    step_params = [
        {'name': 'oed_location_csv', 'flag': '-x', 'is_path': True, 'pre_exist': True, 'help': 'Source location CSV file path'},
        {'name': 'oed_schema_info', 'is_path': True, 'pre_exist': True, 'help': 'path to custom oed_schema'},
        {'name': 'check_oed', 'type': str2bool, 'const': True, 'nargs': '?', 'default': True, 'help': 'if True check input oed files'},
        {'name': 'keys_data_csv', 'flag': '-k', 'is_path': True, 'pre_exist': False, 'help': 'Generated keys CSV output path'},
        {'name': 'keys_errors_csv', 'flag': '-e', 'is_path': True, 'pre_exist': False, 'help': 'Generated keys errors CSV output path'},
        {'name': 'keys_format', 'flag': '-f', 'help': 'Keys files output format', 'choices': ['oasis', 'json'], 'default': 'oasis'},
        {'name': 'lookup_config_json', 'flag': '-g', 'is_path': True, 'pre_exist': False, 'help': 'Lookup config JSON file path'},
        {'name': 'lookup_data_dir', 'is_path': True, 'pre_exist': True, 'help': 'Model lookup/keys data directory path'},
        {'name': 'lookup_module_path', 'flag': '-l', 'is_path': True, 'pre_exist': False, 'help': 'Model lookup module path'},
        {'name': 'lookup_complex_config_json', 'is_path': True, 'pre_exist': False, 'help': 'Complex lookup config JSON file path'},
        {'name': 'lookup_num_processes', 'type': int, 'default': -1, 'help': 'Number of workers in multiprocess pools'},
        {'name': 'lookup_num_chunks', 'type': int, 'default': -1, 'help': 'Number of chunks to split the location file into for multiprocessing'},
        {'name': 'model_version_csv', 'flag': '-v', 'is_path': True, 'pre_exist': False, 'help': 'Model version CSV file path'},
        {'name': 'model_settings_json', 'flag': '-M', 'is_path': True, 'pre_exist': True, 'help': 'Model settings JSON file path'},
        {'name': 'user_data_dir', 'flag': '-D', 'is_path': True, 'pre_exist': False,
         'help': 'Directory containing additional model data files which varies between analysis runs'},
        {'name': 'lookup_multiprocessing', 'type': str2bool, 'const': True, 'nargs': '?', 'default': True,
         'help': 'Flag to enable/disable lookup multiprocessing'},
        {'name': 'disable_oed_version_update', 'type': str2bool, 'const': True, 'nargs': '?', 'default': False,
         'help': 'Flag to enable/disable conversion to latest compatible OED version. Must be present in model settings.'},

        # Manager only options
        {'name': 'verbose', 'default': False},
    ]

    def _get_output_dir(self):
        if self.keys_data_csv:
            return os.path.dirname(self.keys_data_csv)
        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        return os.path.join(os.getcwd(), 'runs', 'keys-{}'.format(utcnow))

    def run(self):
        if not (self.lookup_config_json or (self.lookup_data_dir and self.model_version_csv and self.lookup_module_path)):
            raise OasisException(
                'No pre-generated keys file provided, and no lookup assets '
                'provided to generate a keys file - if you do not have a '
                'pre-generated keys file then lookup assets must be provided - '
                'for a built-in lookup the lookup config. JSON file path must '
                'be provided, or for custom lookups the keys data path + model '
                'version file path + lookup package path must be provided'
            )

        output_dir = self._get_output_dir()
        output_type = 'json' if self.keys_format.lower() == 'json' else 'csv'

        exposure_data = get_exposure_data(self, add_internal_col=True)

        if not self.disable_oed_version_update:
            supported_versions = self.settings.get('data_settings', {}).get('supported_oed_versions', None)
            if supported_versions:
                if isinstance(supported_versions, str):
                    self.logger.info(f"Converting to OED version {supported_versions}")
                    exposure_data.to_version(supported_versions)
                elif isinstance(supported_versions, list) and supported_versions:
                    # If 'supported_oed_versions' is a list and is not empty
                    # Sort the versions in descending order
                    supported_versions = sorted(supported_versions, key=version.parse, reverse=True)
                    self.logger.info(f"Converting to OED version {supported_versions[0]}")
                    exposure_data.to_version(supported_versions[0])
                else:
                    # If 'supported_oed_versions' is neither a string nor a non-empty list
                    self.logger.warning("Invalid OED version information in model settings.")
            else:
                # If 'supported_oed_versions' is missing or empty
                self.logger.debug("No OED version information in model settings.")

        keys_fp = self.keys_data_csv or os.path.join(output_dir, f'keys.{output_type}')
        keys_errors_fp = self.keys_errors_csv or os.path.join(output_dir, f'keys-errors.{output_type}')
        os.makedirs(os.path.dirname(keys_fp), exist_ok=True)
        os.makedirs(os.path.dirname(keys_errors_fp), exist_ok=True)
        keys_success_msg = True if self.lookup_complex_config_json else False

        model_info, key_server = KeyServerFactory.create(
            lookup_config_fp=self.lookup_config_json,
            model_keys_data_path=self.lookup_data_dir,
            model_version_file_path=self.model_version_csv,
            lookup_module_path=self.lookup_module_path,
            complex_lookup_config_fp=self.lookup_complex_config_json,
            user_data_dir=self.user_data_dir,
            output_directory=output_dir
        )

        res = key_server.generate_key_files(
            location_df=exposure_data.get_subject_at_risk_source().dataframe,
            successes_fp=keys_fp,
            errors_fp=keys_errors_fp,
            format=self.keys_format,
            keys_success_msg=keys_success_msg,
            multiproc_enabled=self.lookup_multiprocessing,
            multiproc_num_cores=self.lookup_num_processes,
            multiproc_num_partitions=self.lookup_num_chunks,
        )

        self.logger.debug(f"key generated used model {model_info}")
        self.logger.info('\nKeys successful: {} generated with {} items'.format(res[0], res[1]))
        if len(res) == 4:
            self.logger.info('Keys errors: {} generated with {} items'.format(res[2], res[3]))
        return res


class GenerateKeysDeterministic(KeyComputationStep):
    step_params = [
        {'name': 'oed_location_csv', 'flag': '-x', 'is_path': True, 'pre_exist': True, 'help': 'Source location CSV file path'},
        {'name': 'oed_schema_info', 'is_path': True, 'pre_exist': True, 'help': 'path to custom oed_schema'},
        {'name': 'check_oed', 'type': str2bool, 'const': True, 'nargs': '?', 'default': True, 'help': 'if True check input oed files'},
        {'name': 'keys_data_csv', 'flag': '-k', 'is_path': True, 'pre_exist': False, 'help': 'Generated keys CSV output path'},
        {'name': 'supported_oed_coverage_types', 'type': int, 'nargs': '+', 'help': 'Select List of supported coverage_types [1, .. ,15]'},
        {'name': 'model_perils_covered', 'nargs': '+', 'default': ['AA1'],
         'help': 'List of peril covered by the model'}
    ]

    def _get_output_dir(self):
        if self.keys_data_csv:
            return os.path.basename(self.keys_data_csv)
        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        return os.path.join(os.getcwd(), 'runs', 'keys-{}'.format(utcnow))

    def run(self):
        output_dir = self._get_output_dir()
        keys_fp = self.keys_data_csv or os.path.join(output_dir, 'keys.csv')
        os.makedirs(os.path.dirname(keys_fp), exist_ok=True)

        exposure_data = get_exposure_data(self, add_internal_col=True)

        if self.supported_oed_coverage_types is None:
            coverage_values = exposure_data.oed_schema.schema['CoverageValues']
            cob_coverage = list(
                coverage_info['CoverageID'] for coverage_info in coverage_values.values()
                if not coverage_info['SubCoverages']
                and coverage_info['Type'] == exposure_data.class_of_business_info['name'])
            self.supported_oed_coverage_types = cob_coverage

        config = {'builtin_lookup_type': 'peril_covered_deterministic',
                  'model': {"supplier_id": "OasisLMF",
                            "model_id": "Deterministic",
                            "model_version": "1"},
                  'supported_oed_coverage_types': self.supported_oed_coverage_types,
                  'model_perils_covered': self.model_perils_covered}

        model_info, lookup = KeyServerFactory.create(
            lookup_config=config,
            output_directory=output_dir
        )

        return lookup.generate_key_files(
            location_df=exposure_data.get_subject_at_risk_source().dataframe,
            successes_fp=keys_fp,
            format='oasis',
        )
