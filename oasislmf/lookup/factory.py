__all__ = [
    'KeyServerFactory',
    'BasicKeyServer'
]

import copy
import csv
import json
import os
import sys
import types
import warnings

from collections import OrderedDict
from contextlib import ExitStack

import math
import numpy as np
import pandas as pd

from ..utils.data import get_json, get_location_df
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import import_from_string, get_custom_module, as_path
from ..utils.status import OASIS_KEYS_STATUS

from .builtin import DeterministicLookup
from .builtin import Lookup as NewLookup

from multiprocessing import cpu_count,  Queue, Process
from queue import Empty, Full

# add pickling support for traceback object
import tblib.pickling_support
tblib.pickling_support.install()


def with_error_queue(fct):
    def wrapped_fct(error_queue, *args, **kwargs):
        try:
            return fct(error_queue, *args, **kwargs)
        except Exception:
            error_queue.put(sys.exc_info())
    return wrapped_fct


class KeyServerFactory(object):
    """
    A factory class to create the Keys Server that will be use to generate the keys files
    All Key Server must implement the interface defined in lookup.interface.KeyServerInterface

    Oasis provides a built-in Key Server that manage the generation of the key files from the key provided by
    a built-in or a custom Key Lookup.

    The factory now return a KeyServer object and not a KeyLookup.
    The parameter to pass has also been simplified
    usage of all the below parameter are now deprecated
      - complex_lookup_config_fp => pass the path to your complex lookup config directly in lookup_config_fg
      - lookup_module_path => set as key 'lookup_module_path' in the lookup config
      - model_keys_data_path => set as key 'keys_data_path' in the lookup config
      - model_version_file_path => set the model information ('supplier_id', 'model_id', 'model_version') directly
        into the config
    """

    @classmethod
    def get_config(cls, config_fp):
        return as_path(os.path.dirname(config_fp), 'config_fp'), get_json(config_fp)

    @classmethod
    def get_model_info(cls, model_version_file_path):
        """
        Get model information from the model version file.
        """
        model_version_file_path = as_path(model_version_file_path, 'model_version_file_path', preexists=True, null_is_valid=False)

        with open(model_version_file_path, 'r', encoding='utf-8') as f:
            return next(csv.DictReader(
                f, fieldnames=['supplier_id', 'model_id', 'model_version']
            ))

    @classmethod
    def update_deprecated_args(cls, config_dir, config,
                               complex_lookup_config_fp, model_keys_data_path, model_version_file_path, lookup_module_path):

        if (complex_lookup_config_fp
                or model_keys_data_path
                or model_version_file_path
                or lookup_module_path):
            warnings.warn('usage of complex_lookup_config_fp, model_keys_data_path, '
                          'model_version_file_path and lookup_module_path is now deprecated'
                          'those variables now need to be set in lookup config see (key server documentation)')

        if complex_lookup_config_fp:
            config_dir, config = cls.get_config(complex_lookup_config_fp)

        if model_keys_data_path:
            config['keys_data_path'] = as_path(model_keys_data_path, 'model_keys_data_path', preexists=True)

        if model_version_file_path:
            config['model'] = cls.get_model_info(model_version_file_path)

        if lookup_module_path:
            config['lookup_module_path'] = lookup_module_path

        return config_dir, config

    @classmethod
    def create(
        cls,
        model_keys_data_path=None,
        model_version_file_path=None,
        lookup_module_path=None,
        lookup_config=None,
        lookup_config_json=None,
        lookup_config_fp=None,
        complex_lookup_config_fp=None,
        user_data_dir=None,
        output_directory=None,
    ):
        """
        Creates a keys lookup class instance for the given model and supplier -
        local file paths are required for the model keys data folder, the model
        version file and the Git repository for the model keys server. Returns a
        pair ``(model_info, klc)``, where ``model_info`` is a dictionary holding
        model information from the model version file and `klc` is the lookup
        service class instance for the model.
        """
        if lookup_config:
            config_dir = '.'
            config = lookup_config
        elif lookup_config_json:
            config_dir = '.'
            config = json.loads(lookup_config_json)
        elif lookup_config_fp:
            config_dir, config = cls.get_config(lookup_config_fp)
        else: # no config
            config_dir, config = '.', {}

        if not config:
            config_dir, config = cls.update_deprecated_args(config_dir, config,
                                                            complex_lookup_config_fp, model_keys_data_path,
                                                            model_version_file_path, lookup_module_path)
        else: # reproduce lookup_config overwrite complex_lookup_config_fp
            complex_lookup_config_fp = None

        if config.get('key_server_module_path'):
            _KeyServer = get_custom_module(config.get('key_server_module_path'), 'key_server_module_path')
        else:
            _KeyServer = BasicKeyServer

        if _KeyServer.interface_version == '1':
            key_server = _KeyServer(config,
                                    config_dir=config_dir,
                                    user_data_dir=user_data_dir,
                                    output_dir=output_directory)
        else:
            raise OasisException(f"KeyServer interface version {_KeyServer.interface_version} not implemented")

        if complex_lookup_config_fp:
            key_server.complex_lookup_config_fp = complex_lookup_config_fp

        return config['model'], key_server


class BasicKeyServer:
    """
    A basic implementation of the KeyServerInterface
    will load the KeyLookup class from config['lookup_module_path'] if present or used the built-in KeyLookup
    KeyLookup must implement the KeyLookupInterface

    will provide a multiprocess solution if KeyLoopup implement the process_locations_multiproc method

    both single and multiprocess solutions will use low amount of memory
    as they process the key by chunk of limited size.

    This class implement all the file writing method that were previously handled by the lookup factory
    """
    interface_version = "1"

    valid_format = ['oasis', 'json']

    error_heading_row = OrderedDict([
        ('loc_id', 'LocID'),
        ('peril_id', 'PerilID'),
        ('coverage_type', 'CoverageTypeID'),
        ('status', 'Status'),
        ('message', 'Message'),
    ])

    model_data_heading_row = OrderedDict([
        ('loc_id', 'LocID'),
        ('peril_id', 'PerilID'),
        ('coverage_type', 'CoverageTypeID'),
        ('model_data', 'ModelData'),
    ])

    key_success_heading_row = OrderedDict([
        ('loc_id', 'LocID'),
        ('peril_id', 'PerilID'),
        ('coverage_type', 'CoverageTypeID'),
        ('area_peril_id', 'AreaPerilID'),
        ('vulnerability_id', 'VulnerabilityID'),
    ])

    key_success_with_message_heading_row = OrderedDict([
        ('loc_id', 'LocID'),
        ('peril_id', 'PerilID'),
        ('coverage_type', 'CoverageTypeID'),
        ('area_peril_id', 'AreaPerilID'),
        ('vulnerability_id', 'VulnerabilityID'),
        ('message', 'Message')
    ])

    min_bloc_size = 1000
    max_bloc_size = 10000

    def __init__(self, config, config_dir=None, user_data_dir=None, output_dir=None):
        self.config = config
        self.config_dir = config_dir or '.'
        self.user_data_dir = user_data_dir
        self.output_dir = output_dir
        self.lookup_cls = self.get_lookup_cls()

    def get_lookup_cls(self):
        if self.config.get('lookup_class'):
            lookup_cls = import_from_string(self.config.get('lookup_class'))

        elif self.config.get('lookup_module'):
            lookup_module = import_from_string(self.config.get('lookup_module'))
            lookup_cls = getattr(lookup_module, '{}KeysLookup'.format(self.config['model']['model_id']))

        elif self.config.get('lookup_module_path'):
            lookup_module_path = self.config.get('lookup_module_path')
            if not os.path.isabs(lookup_module_path):
                lookup_module_path = os.path.join(self.config_dir, lookup_module_path)
            lookup_module = get_custom_module(lookup_module_path, 'lookup_module_path')
            lookup_cls = getattr(lookup_module, '{}KeysLookup'.format(self.config['model']['model_id']))
        else: # built-in lookup
            if self.config.get('builtin_lookup_type') == 'deterministic':
                lookup_cls = DeterministicLookup
            elif self.config.get('builtin_lookup_type') == 'new_lookup':
                lookup_cls = NewLookup
            else:
                raise OasisException(f"Unrecognised lookup config file, or config file is from deprecated built in lookup module 'oasislmf<=1.16.0' ")

        return lookup_cls

    @staticmethod
    def create_lookup(lookup_cls, config, config_dir, user_data_dir, output_dir, lookup_id):
        lookup_config = copy.deepcopy(config)
        lookup_config['lookup_id'] = lookup_id
        lookup_interface_version = getattr(lookup_cls, 'interface_version', '0')
        if lookup_interface_version == '1':
            return lookup_cls(config,
                              config_dir=config_dir,
                              user_data_dir=user_data_dir,
                              output_dir=output_dir)
        elif lookup_interface_version == '0':
            warnings.warn('OasisLookupInterface (or OasisBaseKeysLookup) is now deprecated'
                          ' Interface for lookup is now lookup.interface.LookupInterface'
                          ' for similar functionality use lookup.base.AbstractBasicKeyLookup'
                          ' for multiprocess implementation add lookup.base.MultiprocLookupMixin')
            if not (config and output_dir):
                return lookup_cls(
                    keys_data_directory=config.get('keys_data_path'),
                    supplier=config['model']['supplier_id'],
                    model_name=config['model']['model_id'],
                    model_version=config['model']['model_version'],
                )
            elif not user_data_dir:
                return lookup_cls(
                    keys_data_directory=config.get('keys_data_path'),
                    supplier=config['model']['supplier_id'],
                    model_name=config['model']['model_id'],
                    model_version=config['model']['model_version'],
                    complex_lookup_config_fp=config_dir,
                    output_directory=output_dir
                )
            else:
                return lookup_cls(
                    keys_data_directory=config.get('keys_data_path'),
                    supplier=config['model']['supplier_id'],
                    model_name=config['model']['model_id'],
                    model_version=config['model']['model_version'],
                    complex_lookup_config_fp=config_dir,
                    user_data_dir=user_data_dir,
                    output_directory=output_dir
                )
        else:
            raise OasisException(f"lookup interface version {lookup_interface_version} not implemented")

    def get_locations(self, location_fp):
        """load exposure data from location_fp and return the exposure dataframe"""
        return get_location_df(location_fp)

    @staticmethod
    @with_error_queue
    def location_producer(error_queue, loc_df, part_count, loc_queue):
        loc_ids_parts = np.array_split(np.unique(loc_df['loc_id']), part_count)
        loc_df_parts = (loc_df[loc_df['loc_id'].isin(loc_ids_parts[i])] for i in range(part_count))
        loc_df_part = True
        while loc_df_part is not None:
            loc_df_part = next(loc_df_parts, None)
            while error_queue.empty():
                try:
                    loc_queue.put(loc_df_part, timeout=5)
                    break
                except Full:
                    pass
            else:
                return

    @staticmethod
    @with_error_queue
    def lookup_multiproc_worker(error_queue, lookup_cls, config, config_dir, user_data_dir, output_dir, lookup_id, loc_queue, key_queue):
        lookup = BasicKeyServer.create_lookup(lookup_cls, config, config_dir, user_data_dir, output_dir, lookup_id)
        while True:
            while error_queue.empty():
                try:
                    loc_df_part = loc_queue.get(timeout=5)

                    break
                except Empty:
                    pass
            else:
                return

            if loc_df_part is None:
                loc_queue.put(None)
                key_queue.put(None)
                break

            while error_queue.empty():
                try:
                    key_queue.put(lookup.process_locations_multiproc(loc_df_part), timeout=5)
                    break
                except Full:
                    pass
            else:
                return

    @staticmethod
    def key_producer(key_queue, error_queue, worker_count):
        finished_workers = 0
        while finished_workers < worker_count and error_queue.empty():
            while error_queue.empty():
                try:
                    res = key_queue.get(timeout=5)
                    break
                except Empty:
                    pass
            else:
                break

            if res is None:
                finished_workers+=1
            else:
                yield res

    def get_success_heading_row(self, keys, keys_success_msg):
        if 'model_data' in keys:
            return self.model_data_heading_row
        elif keys_success_msg:
            return self.key_success_with_message_heading_row
        else:
            return self.key_success_heading_row

    def write_json_keys_file(self, results, keys_success_msg, successes_fp, errors_fp):
        # no streaming implementation for json format
        results = pd.concat((r for r in results if not r.empty))

        success = results['status'] == OASIS_KEYS_STATUS['success']['id']
        success_df = results[success]
        success_df.to_json(successes_fp, orient='records', indent=4, force_ascii=False)
        successes_count = success_df.shape[0]
        if errors_fp:
            errors_df = results[~success]
            errors_df.to_json(errors_fp, orient='records', indent=4, force_ascii=False)
            error_count = errors_df.shape[0]
        else:
            error_count = 0
        return successes_count, error_count

    def write_oasis_keys_file(self, results, keys_success_msg, successes_fp, errors_fp):
        with ExitStack() as stack:
            successes_file = stack.enter_context(open(successes_fp, 'w', encoding='utf-8'))
            if errors_fp:
                errors_file = stack.enter_context(open(errors_fp, 'w', encoding='utf-8'))
                errors_file.write(','.join(self.error_heading_row.values()) + '\n')
            else:
                errors_file = None
            success_heading_row = None
            successes_count = 0
            error_count = 0
            for i, result in enumerate(results):
                success = result['status'] == OASIS_KEYS_STATUS['success']['id']
                success_df = result[success]
                if success_heading_row is None:
                    success_heading_row = self.get_success_heading_row(result.columns, keys_success_msg)
                success_df[success_heading_row.keys()].rename(columns=success_heading_row
                                                                ).to_csv(successes_file, index=False, header=not i)
                successes_count += success_df.shape[0]
                if errors_file:
                    errors_df = result[~success]
                    errors_df[self.error_heading_row.keys()].rename(columns=self.error_heading_row
                                                                    ).to_csv(errors_file, index=False, header=False)
                    error_count += errors_df.shape[0]
            return successes_count, error_count

    def write_keys_file(self, results, successes_fp, errors_fp, output_format, keys_success_msg):
        if output_format not in self.valid_format:
            raise OasisException(f"Unrecognised lookup file output format {output_format} - valid formats are {self.valid_format}")

        write = getattr(self, f'write_{output_format}_keys_file')
        successes_count, error_count = write(results, keys_success_msg, successes_fp, errors_fp)

        if errors_fp:
            return successes_fp, successes_count, errors_fp, error_count
        else:
            return successes_fp, successes_count

    def generate_key_files_singleproc(self, loc_df, successes_fp, errors_fp, output_format, keys_success_msg, **kwargs):
        if getattr(self, 'complex_lookup_config_fp', None):  # backward compatibility 1.15 hack
            config_dir = getattr(self, 'complex_lookup_config_fp', None)
        else:
            config_dir = self.config_dir
        lookup = self.create_lookup(self.lookup_cls, self.config, config_dir, self.user_data_dir, self.output_dir,
                                    lookup_id=None)

        key_results = lookup.process_locations(loc_df)

        def gen_results(results):
            if isinstance(results, pd.DataFrame):
                yield results
            elif isinstance(results, (list, tuple)):
                yield pd.DataFrame(results)
            elif isinstance(results, types.GeneratorType):
                results_part = pd.DataFrame.from_records(results, nrows=self.max_bloc_size)
                while not results_part.empty:
                    yield results_part
                    results_part = pd.DataFrame.from_records(results, nrows=self.max_bloc_size)
            else:
                raise OasisException("Unrecognised type for results: {type(results)}. expected ")

        return self.write_keys_file(gen_results(key_results),
                                    successes_fp=successes_fp,
                                    errors_fp=errors_fp,
                                    output_format=output_format,
                                    keys_success_msg=keys_success_msg,)

    def generate_key_files_multiproc(self, loc_df, successes_fp, errors_fp, output_format, keys_success_msg,
                                     num_cores, num_partitions, **kwargs):
        """
        Process and return the lookup results a location row
        Used in multiprocessing based query

        location_row is of type <class 'pandas.core.series.Series'>

        """
        if getattr(self, 'complex_lookup_config_fp', None):  # backward compatibility 1.15 hack
            config_dir = getattr(self, 'complex_lookup_config_fp', None)
        else:
            config_dir = self.config_dir

        pool_count = num_cores if num_cores > 0 else cpu_count()
        if num_partitions > 0:
            part_count = num_partitions
        else:
            bloc_size = min(max(math.ceil(loc_df.shape[0] / pool_count), self.min_bloc_size), self.max_bloc_size)
            part_count = math.ceil(loc_df.shape[0] / bloc_size)
            pool_count = min(pool_count, part_count)
        if pool_count <= 1:
            return self.generate_key_files_singleproc(loc_df, successes_fp, errors_fp, output_format, keys_success_msg)

        loc_queue = Queue(maxsize=pool_count)
        key_queue = Queue(maxsize=pool_count)
        error_queue = Queue()

        location_producer = Process(target=self.location_producer, args=(error_queue, loc_df, part_count, loc_queue))

        workers = [Process(target=self.lookup_multiproc_worker,
                           args=(error_queue, self.lookup_cls, self.config, config_dir,
                                 self.user_data_dir, self.output_dir,
                                 lookup_id, loc_queue, key_queue))
                   for lookup_id in range(pool_count)]

        location_producer.start()
        [worker.start() for worker in workers]

        try:
            return self.write_keys_file(self.key_producer(key_queue, error_queue, worker_count= pool_count),
                                        successes_fp=successes_fp,
                                        errors_fp=errors_fp,
                                        output_format=output_format,
                                        keys_success_msg=keys_success_msg,)
        except Exception:
            error_queue.put(sys.exc_info())
        finally:
            for process in [location_producer] + workers:
                if process.is_alive():
                    process.terminate()
                    process.join()
            loc_queue.close()
            key_queue.close()
            if not error_queue.empty():
                exc_info = error_queue.get()
                raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    @oasis_log()
    def generate_key_files(
        self,
        location_fp,
        successes_fp,
        errors_fp=None,
        output_format='oasis',
        keys_success_msg=False,
        multiproc_enabled=True,
        multiproc_num_cores=-1,
        multiproc_num_partitions=-1,
        location_df=None,
        **kwargs
    ):
        """
        generate key files by calling:
        1. get_locations to get a location object from the location_fp
        2. process_locations or process_locations_multiproc to get results object from the locations object
        3. write_keys_file to writes the relevant files from the results object
        """
        successes_fp = as_path(successes_fp, 'successes_fp', preexists=False)
        errors_fp = as_path(errors_fp, 'errors_fp', preexists=False)

        if location_df is not None:
            locations = location_df
        else:    
            locations = self.get_locations(location_fp)

        if multiproc_enabled and hasattr(self.lookup_cls, 'process_locations_multiproc'):
            return self.generate_key_files_multiproc(locations,
                                                     successes_fp=successes_fp,
                                                     errors_fp=errors_fp,
                                                     output_format=output_format,
                                                     keys_success_msg=keys_success_msg,
                                                     num_cores = multiproc_num_cores,
                                                     num_partitions = multiproc_num_partitions)
        else:
            return self.generate_key_files_singleproc(locations,
                                                      successes_fp=successes_fp,
                                                      errors_fp=errors_fp,
                                                      output_format=output_format,
                                                      keys_success_msg=keys_success_msg,
                                                      )
