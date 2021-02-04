__all__ = [
    'GenerateKeys',
    'GenerateKeysDeterministic'
]

import os

from itertools import (
    product,
)

from ..base import ComputationStep
from ...lookup.factory import OasisLookupFactory as olf
from ...utils.exceptions import OasisException
from ...utils.coverages import SUPPORTED_COVERAGE_TYPES

from ...utils.data import (
    get_location_df,
    get_utctimestamp,
)


class GenerateKeys(ComputationStep):
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

    step_params = [
        {'name': 'oed_location_csv',           'flag':'-x', 'is_path': True, 'pre_exist': True,  'help': 'Source location CSV file path', 'required': True},
        {'name': 'keys_data_csv',              'flag':'-k', 'is_path': True, 'pre_exist': False, 'help': 'Generated keys CSV output path'},
        {'name': 'keys_errors_csv',            'flag':'-e', 'is_path': True, 'pre_exist': False, 'help': 'Generated keys errors CSV output path'},
        {'name': 'keys_format',                'flag':'-f',  'help': 'Keys files output format', 'choices':['oasis', 'json'], 'default':'oasis'},
        {'name': 'lookup_config_json',         'flag':'-g', 'is_path': True, 'pre_exist': False, 'help': 'Lookup config JSON file path'},
        {'name': 'lookup_data_dir',            'flag':'-d', 'is_path': True, 'pre_exist': True,  'help': 'Model lookup/keys data directory path'},
        {'name': 'lookup_module_path',         'flag':'-l', 'is_path': True, 'pre_exist': False, 'help': 'Model lookup module path'},
        {'name': 'lookup_complex_config_json', 'flag':'-L', 'is_path': True, 'pre_exist': False, 'help': 'Complex lookup config JSON file path'},
        {'name': 'lookup_num_processes',       'type':int,  'default': -1,                       'help': 'Number of workers in multiprocess pools'},
        {'name': 'lookup_num_chunks',          'type':int,  'default': -1,                       'help': 'Number of chunks to split the location file into for multiprocessing'},
        {'name': 'model_version_csv',          'flag':'-v', 'is_path': True, 'pre_exist': False, 'help': 'Model version CSV file path'},
        {'name': 'user_data_dir',              'flag':'-D', 'is_path': True, 'pre_exist': False, 'help': 'Directory containing additional model data files which varies between analysis runs'},

        # Manager only options
        {'name': 'verbose',                'default': False},
        {'name': 'lookup_multiprocessing', 'default': True},            # Enable/disable multiprocessing
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
        location_df = get_location_df(self.oed_location_csv)

        keys_fp = self.keys_data_csv or os.path.join(output_dir, f'keys.{output_type}')
        keys_errors_fp = self.keys_errors_csv or os.path.join(output_dir, f'keys-errors.{output_type}')
        os.makedirs(os.path.dirname(keys_fp), exist_ok=True)
        os.makedirs(os.path.dirname(keys_errors_fp), exist_ok=True)
        keys_success_msg = True if self.lookup_complex_config_json else False

        model_info, lookup = olf.create(
            lookup_config_fp=self.lookup_config_json,
            model_keys_data_path=self.lookup_data_dir,
            model_version_file_path=self.model_version_csv,
            lookup_module_path=self.lookup_module_path,
            complex_lookup_config_fp=self.lookup_complex_config_json,
            user_data_dir=self.user_data_dir,
            output_directory=output_dir
        )

        f1, n1, f2, n2 = olf.save_results(
            lookup,
            location_df=location_df,
            successes_fp=keys_fp,
            errors_fp=keys_errors_fp,
            format=self.keys_format,
            keys_success_msg=keys_success_msg,
            multiproc_enabled=self.lookup_multiprocessing,
            multiproc_num_cores=self.lookup_num_processes,
            multiproc_num_partitions=self.lookup_num_chunks,
        )
        self.logger.info('\nKeys successful: {} generated with {} items'.format(f1, n1))
        self.logger.info('Keys errors: {} generated with {} items'.format(f2, n2))
        return (f1, n1, f2, n2)


class GenerateKeysDeterministic(ComputationStep):

    step_params = [
        {'name': 'oed_location_csv',           'flag':'-x', 'is_path': True, 'pre_exist': True,  'help': 'Source location CSV file path', 'required': True},
        {'name': 'keys_data_csv',              'flag':'-k', 'is_path': True, 'pre_exist': False,  'help': 'Generated keys CSV output path'},
        {'name': 'num_subperils',               'flag':'-p', 'default': 1,  'type':int,          'help': 'Set the number of subperils returned by deterministic key generator'},
        {'name': 'supported_oed_coverage_types', 'type' :int, 'nargs':'+', 'default': list(v['id'] for v in SUPPORTED_COVERAGE_TYPES.values()), 'help': 'Select List of supported coverage_types [1, .. ,4]'},
    ]

    def _get_output_dir(self):
        if self.keys_data_csv:
            return os.path.basename(self.keys_data_csv)
        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        return os.path.join(os.getcwd(), 'runs', 'keys-{}'.format(utcnow))

    def run(self):
        output_dir = self._get_output_dir()
        keys_fp = self.keys_data_csv or os.path.join(output_dir, 'keys.csv')
        location_df = get_location_df(self.oed_location_csv)

        loc_ids = (loc_it['loc_id'] for _, loc_it in location_df.loc[:, ['loc_id']].sort_values('loc_id').iterrows())
        keys = [
            {'loc_id': _loc_id, 'peril_id': peril, 'coverage_type': cov_type, 'area_peril_id': i + 1, 'vulnerability_id': i + 1}
            for i, (_loc_id, peril, cov_type) in enumerate(product(loc_ids, range(1, 1 + self.num_subperils), self.supported_oed_coverage_types))
        ]
        return  olf.write_oasis_keys_file(keys, keys_fp)
