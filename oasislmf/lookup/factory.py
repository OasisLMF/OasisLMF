__all__ = [
    'OasisLookupFactory'
]

import csv
import io
import json
import itertools

from collections import OrderedDict

from multiprocessing import cpu_count
from billiard import Pool

import numpy as np
import pandas as pd

from .base import OasisBaseLookup
from .rtree import RTreeLookup
from .interface import OasisLookupInterface

from ..utils.exceptions import OasisException
from ..utils.status import OASIS_KEYS_STATUS
from ..utils.path import get_custom_module, as_path


class OasisLookupFactory(object):
    """
    A factory class to load and run keys lookup services for different
    models/suppliers.
    """
    @classmethod
    def get_model_info(cls, model_version_file_path):
        """
        Get model information from the model version file.
        """
        model_version_file_path = as_path(model_version_file_path, 'model_version_file_path', preexists=True, null_is_valid=False)

        with io.open(model_version_file_path, 'r', encoding='utf-8') as f:
            return next(csv.DictReader(
                f, fieldnames=['supplier_id', 'model_id', 'model_version']
            ))

    @classmethod
    def get_custom_lookup(
        cls,
        lookup_module,
        keys_data_path,
        model_info,
        complex_lookup_config_fp=None,
        user_data_dir=None,
        output_directory=None
    ):
        """
        Get the keys lookup class instance.
        """
        klc = getattr(lookup_module, '{}KeysLookup'.format(model_info['model_id']))

        if not (complex_lookup_config_fp and output_directory):
            return klc(
                keys_data_directory=keys_data_path,
                supplier=model_info['supplier_id'],
                model_name=model_info['model_id'],
                model_version=model_info['model_version']
            )
        elif not user_data_dir:
            return klc(
                keys_data_directory=keys_data_path,
                supplier=model_info['supplier_id'],
                model_name=model_info['model_id'],
                model_version=model_info['model_version'],
                complex_lookup_config_fp=complex_lookup_config_fp,
                output_directory=output_directory
            )
        else:
            return klc(
                keys_data_directory=keys_data_path,
                supplier=model_info['supplier_id'],
                model_name=model_info['model_id'],
                model_version=model_info['model_version'],
                complex_lookup_config_fp=complex_lookup_config_fp,
                user_data_dir=user_data_dir,
                output_directory=output_directory
            )

    @classmethod
    def write_oasis_keys_file(cls, records, output_file_path, output_success_msg=False):
        """
        Writes an Oasis keys file from an iterable of keys records.
        """

        if len(records) > 0 and 'model_data' in records[0]:
            heading_row = OrderedDict([
                ('loc_id', 'LocID'),
                ('peril_id', 'PerilID'),
                ('coverage_type', 'CoverageTypeID'),
                ('model_data', 'ModelData'),
            ])
        else:
            heading_row = OrderedDict([
                ('loc_id', 'LocID'),
                ('peril_id', 'PerilID'),
                ('coverage_type', 'CoverageTypeID'),
                ('area_peril_id', 'AreaPerilID'),
                ('vulnerability_id', 'VulnerabilityID'),
            ])
            if output_success_msg:
                heading_row.update({'message': 'Message'})

        pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row] + records,
        ).to_csv(
            output_file_path,
            index=False,
            encoding='utf-8',
            header=False,
        )

        return output_file_path, len(records)

    @classmethod
    def write_oasis_keys_errors_file(cls, records, output_file_path):
        """
        Writes an Oasis keys errors file from an iterable of keys records.
        """
        heading_row = OrderedDict([
            ('loc_id', 'LocID'),
            ('peril_id', 'PerilID'),
            ('coverage_type', 'CoverageTypeID'),
            ('status', 'Status'),
            ('message', 'Message'),
        ])

        pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row] + records,
        ).to_csv(
            output_file_path,
            index=False,
            encoding='utf-8',
            header=False,
        )

        return output_file_path, len(records)

    @classmethod
    def write_json_keys_file(cls, records, output_file_path):
        """
        Writes the keys records as a simple list to file.
        """
        with io.open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(u'{}'.format(json.dumps(records, sort_keys=True, indent=4, ensure_ascii=False)))

            return output_file_path, len(records)

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
        builtin_lookup_type='combined'
    ):
        """
        Creates a keys lookup class instance for the given model and supplier -
        local file paths are required for the model keys data folder, the model
        version file and the Git repository for the model keys server. Returns a
        pair ``(model_info, klc)``, where ``model_info`` is a dictionary holding
        model information from the model version file and `klc` is the lookup
        service class instance for the model.
        """
        is_builtin = lookup_config or lookup_config_json or lookup_config_fp
        is_complex = complex_lookup_config_fp and output_directory

        if is_builtin:
            lookup = RTreeLookup(
                config=lookup_config,
                config_json=lookup_config_json,
                config_fp=lookup_config_fp
            )
            model_info = lookup.config.get('model')
            if builtin_lookup_type == 'base':
                lookup = OasisBuiltinBaseLookup(
                    config=lookup_config,
                    config_json=lookup_config_json,
                    config_fp=lookup_config_fp
                )
                return lookup.config.get('model'), lookup
            elif builtin_lookup_type == 'combined':
                return model_info, lookup
            elif builtin_lookup_type == 'peril':
                return model_info, lookup.peril_lookup
            elif builtin_lookup_type == 'vulnerability':
                return model_info, lookup.vulnerability_lookup
        else:
            _model_keys_data_path = as_path(model_keys_data_path, 'model_keys_data_path', preexists=True)

            model_info = cls.get_model_info(model_version_file_path)
            lookup_module = get_custom_module(lookup_module_path, 'lookup_module_path')

            if not is_complex:
                lookup = cls.get_custom_lookup(
                    lookup_module=lookup_module,
                    keys_data_path=_model_keys_data_path,
                    model_info=model_info
                )
                return model_info, lookup

            _complex_lookup_config_fp = as_path(complex_lookup_config_fp, 'complex_lookup_config_fp', preexists=True)
            _output_directory = as_path(output_directory, 'output_directory', preexists=True)

            lookup = cls.get_custom_lookup(
                lookup_module=lookup_module,
                keys_data_path=_model_keys_data_path,
                model_info=model_info,
                complex_lookup_config_fp=_complex_lookup_config_fp,
                user_data_dir=user_data_dir,
                output_directory=_output_directory
            )

            return model_info, lookup

    @classmethod
    def get_keys_base(
        cls,
        lookup,
        loc_df,
        success_only=False
    ):
        """
        Used when lookup is an instances of `OasisBaseKeysLookup(object)`

        Generates keys records (JSON) for the given model and supplier -
        requires an instance of the lookup service (which can be created using
        the `create` method in this factory class), and either the model
        location file path or the string contents of such a file.

        The optional keyword argument ``success_only`` indicates whether only
        records with successful lookups should be returned (default), or all
        records.
        """
        for record in lookup.process_locations(loc_df):
            if success_only:
                if record['status'].lower() == OASIS_KEYS_STATUS['success']['id']:
                    yield record
            else:
                yield record

    @classmethod
    def get_keys_builtin(
        cls,
        lookup,
        loc_df,
        success_only=False
    ):
        """
        Used when lookup is an instances of `OasisBuiltinBaseLookup(object)`

        Generates lookup results (dicts) for the given model and supplier -
        requires a lookup instance (which can be created using the `create2`
        method in this factory class), and the source exposures/locations
        dataframe.

        The optional keyword argument ``success_only`` indicates whether only
        results with successful lookup status should be returned (default),
        or all results.
        """
        locations = (loc for _, loc in loc_df.iterrows())
        for result in lookup.bulk_lookup(locations):
            if success_only:
                if result['status'].lower() == OASIS_KEYS_STATUS['success']['id']:
                    yield result
            else:
                yield result

    @classmethod
    def get_keys_multiproc(
        cls,
        lookup,
        loc_df,
        success_only=False,
        num_cores=-1,
        num_partitions=-1
    ):
        """
        Used for CPU bound lookup operations, Depends on a method

        `process_locations_multiproc(dataframe)`

        where single_row is a pandas series from a location Pandas DataFrame
        and returns a list of dicts holding the lookup results for that single row
        """
        pool_count = num_cores if num_cores > 0 else cpu_count()
        part_count = num_partitions if num_partitions > 0 else min(pool_count * 2, len(loc_df))
        locations = np.array_split(loc_df, part_count)

        pool = Pool(pool_count)
        results = pool.map(lookup.process_locations_multiproc, locations)
        lookup_results = sum([r for r in results if r], [])
        pool.terminate()
        return lookup_results

    @classmethod
    def save_keys(
        cls,
        keys_data,
        keys_file_path=None,
        keys_errors_file_path=None,
        keys_format='oasis',
        keys_success_msg=False
    ):
        """
        Writes a keys file, and optionally a keys error file, for the keys
        generated by the lookup service for the given model, supplier and
        exposure sfile - requires a lookup service instance (which can be
        created using the `create` method in this factory class), the path of
        the model location file, the path of the keys file, and the format of
        the output file which can be an Oasis keys file (``oasis``) or a
        simple listing of the records to file (``json``).

        The optional keyword argument ``keys_error_file_path`` if present
        indicates that all keys records, whether for locations with successful
        or unsuccessful lookups, will be generated and written to separate
        files. A keys record with a successful lookup will have a `status`
        field whose value will be `success`, otherwise the record will have
        a `status` field value of `failure` or `nomatch`.

        If ``keys_errors_file_path`` is not present then the method returns a
        pair ``(p, n)`` where ``p`` is the keys file path and ``n`` is the
        number of "successful" keys records written to the keys file, otherwise
        it returns a quadruple ``(p1, n1, p2, n2)`` where ``p1`` is the keys
        file path, ``n1`` is the number of "successful" keys records written to
        the keys file, ``p2`` is the keys errors file path and ``n2`` is the
        number of "unsuccessful" keys records written to keys errors file.
        """

        _keys_file_path = as_path(keys_file_path, 'keys_file_path', preexists=False)
        _keys_errors_file_path = as_path(keys_errors_file_path, 'keys_errors_file_path', preexists=False)

        successes = []
        nonsuccesses = []
        for k in keys_data:
            successes.append(k) if k['status'] == OASIS_KEYS_STATUS['success']['id'] else nonsuccesses.append(k)

        if keys_format == 'json':
            if _keys_errors_file_path:
                fp1, n1 = cls.write_json_keys_file(successes, _keys_file_path)
                fp2, n2 = cls.write_json_keys_file(nonsuccesses, _keys_errors_file_path)

                return fp1, n1, fp2, n2
            return cls.write_json_keys_file(successes, _keys_file_path)
        elif keys_format == 'oasis':
            if _keys_errors_file_path:
                fp1, n1 = cls.write_oasis_keys_file(successes, _keys_file_path, keys_success_msg)
                fp2, n2 = cls.write_oasis_keys_errors_file(nonsuccesses, _keys_errors_file_path)

                return fp1, n1, fp2, n2
            return cls.write_oasis_keys_file(successes, _keys_file_path, keys_success_msg)
        else:
            raise OasisException("Unrecognised keys file output format - valid formats are 'oasis' or 'json'")

    @classmethod
    def check_results(
        cls,
        key_results
    ):
        """
        checks that the keys return data is populated
        """
        if isinstance(key_results, list):
            if len(key_results) == 0:
                return True, None
            else:
                return False, key_results
        else:
            try:
                first = next(key_results)
            except StopIteration:
                return True, None
            return False, itertools.chain([first], key_results)

    @classmethod
    def save_results(
        cls,
        lookup,
        location_df,
        successes_fp=None,
        errors_fp=None,
        format='oasis',
        keys_success_msg=False,
        multiproc_enabled=True,
        multiproc_num_cores=-1,
        multiproc_num_partitions=-1,

    ):
        """
        Writes a keys file, and optionally a keys error file, for the keys
        generated by the lookup service for the given model, supplier and
        exposure sfile - requires a lookup service instance (which can be
        created using the `create` method in this factory class), the path of
        the model location file, the path of the keys file, and the format of
        the output file which can be an Oasis keys file (``oasis``) or a
        simple listing of the records to file (``json``).

        The optional keyword argument ``keys_error_file_path`` if present
        indicates that all keys records, whether for locations with successful
        or unsuccessful lookups, will be generated and written to separate
        files. A keys record with a successful lookup will have a `status`
        field whose value will be `success`, otherwise the record will have
        a `status` field value of `failure` or `nomatch`.

        If ``keys_errors_file_path`` is not present then the method returns a
        pair ``(p, n)`` where ``p`` is the keys file path and ``n`` is the
        number of "successful" keys records written to the keys file, otherwise
        it returns a quadruple ``(p1, n1, p2, n2)`` where ``p1`` is the keys
        file path, ``n1`` is the number of "successful" keys records written to
        the keys file, ``p2`` is the keys errors file path and ``n2`` is the
        number of "unsuccessful" keys records written to keys errors file.
        """
        sfp = as_path(successes_fp, 'successes_fp', preexists=False)
        efp = as_path(errors_fp, 'errors_fp', preexists=False)
        kwargs = {
            "lookup": lookup,
            "loc_df": location_df,
            "success_only": (False if efp else True)
        }

        # Return the multiproccessed generated (both lookup classes have this method)
        if (multiproc_enabled and hasattr(lookup, 'process_locations_multiproc')):
            keys_generator = cls.get_keys_multiproc
            kwargs['num_cores'] = multiproc_num_cores
            kwargs['num_partitions'] = multiproc_num_partitions
        # Return Rtree based method
        elif isinstance(lookup, (RTreeLookup, OasisBaseLookup)):
            keys_generator = cls.get_keys_builtin
        # Return Interface method, same as 'OasisBaseKeysLookup' before refactor
        elif isinstance(lookup, OasisLookupInterface):
            keys_generator = cls.get_keys_base
        # Fallback to trying 'get_keys_base', if that fails raise a error
        else:
            try:
                keys_generator = cls.get_keys_base
            except AttributeError:
                raise OasisException('Unknown lookup class {}, missing default method "cls.get_keys_base"'.format(type(lookup)))

        results_empty, results = cls.check_results(keys_generator(**kwargs))
        successes = []
        nonsuccesses = []

        # Todo: Move the inside the keys_generators?  and return a tuple of (successes, nonsuccesses)
        if results_empty == False:
            for r in results:
                successes.append(r) if r['status'] == OASIS_KEYS_STATUS['success']['id'] else nonsuccesses.append(r)

            if format == 'json':
                if efp:
                    fp1, n1 = cls.write_json_keys_file(successes, sfp)
                    fp2, n2 = cls.write_json_keys_file(nonsuccesses, efp)
                    return fp1, n1, fp2, n2
                return cls.write_json_keys_file(successes, sfp)
            elif format == 'oasis':
                if efp:
                    fp1, n1 = cls.write_oasis_keys_file(successes, sfp, keys_success_msg)
                    fp2, n2 = cls.write_oasis_keys_errors_file(nonsuccesses, efp)
                    return fp1, n1, fp2, n2
                return cls.write_oasis_keys_file(successes, sfp, keys_success_msg)
            else:
                raise OasisException("Unrecognised lookup file output format - valid formats are 'oasis' or 'json'")
        else: 
            raise OasisException("No data returned from keys service")
                    
