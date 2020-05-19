__all__ = [
    'OasisBuiltinBaseLookup',
    'OasisBaseKeysLookup',
    'OasisLookup',
    'OasisPerilLookup',
    'OasisVulnerabilityLookup',
    'OasisLookupFactory'
]

import builtins
import csv
import importlib
import io
import itertools
import json
import os
import re
import sys
import types
import uuid

from collections import OrderedDict

from multiprocessing import cpu_count
from billiard import Pool

import pandas as pd

from shapely.geometry import (
    box,
    Point,
)

from shapely import speedups as shapely_speedups

from rtree.core import RTreeError

from ..utils.data import (
    get_dataframe,
    get_dtypes_and_required_cols,
    get_ids,
)
from ..utils.defaults import get_loc_dtypes
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.peril import (
    DEFAULT_RTREE_INDEX_PROPS,
    PerilAreasIndex,
)
from ..utils.profiles import get_oed_hierarchy
from ..utils.status import OASIS_KEYS_STATUS
from ..utils.path import get_custom_module, as_path

if shapely_speedups.available:
    shapely_speedups.enable()
UNKNOWN_ID = -1


class OasisBuiltinBaseLookup(object):

    @oasis_log()
    def __init__(self, config=None, config_json=None, config_fp=None, config_dir=None):
        if config:
            self._config = config
            self.config_dir = config_dir or '.'
        elif config_json:
            self._config = json.loads(config_json)
            self.config_dir = config_dir or '.'
        elif config_fp:
            self.config_dir = config_dir or os.path.dirname(config_fp)
            _config_fp = as_path(config_fp, 'config_fp')
            with io.open(_config_fp, 'r', encoding='utf-8') as f:
                self._config = json.load(f)

        keys_data_path = self._config.get('keys_data_path')
        keys_data_path = os.path.join(self.config_dir, keys_data_path) if keys_data_path else ''

        self._config['keys_data_path'] = as_path(keys_data_path, 'keys_data_path', preexists=(True if keys_data_path else False))

        peril_config = self._config.get('peril') or {}

        self._peril_ids = tuple(peril_config.get('peril_ids') or ())

        self._peril_id_col = peril_config.get('peril_id_col') or 'peril_id'

        coverage_config = self._config.get('coverage') or {}

        self._coverage_types = tuple(coverage_config.get('coverage_types') or ())

        self._coverage_type_col = peril_config.get('coverage_type_col') or 'coverage_type'

        self._config.setdefault('exposure', self._config.get('exposure') or self._config.get('locations') or {})

        self.__tweak_config_data__()

    def __tweak_config_data__(self):
        for section in ('exposure', 'peril', 'vulnerability',):
            section_config = self._config.get(section) or {}
            for k, v in section_config.items():
                if isinstance(v, str) and '%%KEYS_DATA_PATH%%' in v:
                    self._config[section][k] = v.replace('%%KEYS_DATA_PATH%%', self._config['keys_data_path'])
                elif type(v) == list:
                    self._config[section][k] = tuple(v)
                elif isinstance(v, dict):
                    for _k, _v in v.items():
                        if isinstance(_v, str) and '%%KEYS_DATA_PATH%%' in _v:
                            self._config[section][k][_k] = _v.replace('%%KEYS_DATA_PATH%%', self._config['keys_data_path'])

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, c):
        self._config = c
        self.__tweak_config_data__()

    @property
    def peril_ids(self):
        return self._peril_ids

    @property
    def peril_id_col(self):
        return self._peril_id_col

    @property
    def coverage_types(self):
        return self._coverage_types

    @property
    def coverage_type_col(self):
        return self._coverage_type_col

    def lookup(self, loc, peril_id, coverage_type, **kwargs):
        """
        Lookup for an individual location item, which could be a dict or a
        Pandas series object.
        """
        pass

    def process_locations_multiproc(self, loc_df):
        """
        Process and return the lookup results a location row
        Used in multiprocessing based query

        location_row is of type <class 'pandas.core.series.Series'>

        """
        locs_seq = (loc for _, loc in loc_df.iterrows())
        return [self.lookup(loc, peril_id, coverage_type) for
                loc, peril_id, coverage_type in
                itertools.product(locs_seq, self.peril_ids, self.coverage_types)]

    @oasis_log()
    def bulk_lookup(self, locs, **kwargs):
        """
        Bulk vulnerability lookup for a list, tuple, generator, pandas data
        frame or dict of location items, which can be dicts or Pandas series
        objects or any object which has as a dict-like interface.

        Generates results using ``yield``.
        """
        locs_seq = None

        if (isinstance(locs, list) or isinstance(locs, tuple)):
            locs_seq = (loc for loc in locs)
        elif isinstance(locs, types.GeneratorType):
            locs_seq = locs
        elif (isinstance(locs, dict)):
            locs_seq = locs.values()
        elif isinstance(locs, pd.DataFrame):
            locs_seq = (loc for _, loc in locs.iterrows())

        for loc, peril_id, coverage_type in itertools.product(locs_seq, self.peril_ids, self.coverage_types):
            yield self.lookup(loc, peril_id, coverage_type)


class OasisBaseKeysLookup(object):  # pragma: no cover
    """
    Old Oasis base class -deprecated
    """
    @oasis_log()
    def __init__(
        self,
        keys_data_directory=None,
        supplier=None,
        model_name=None,
        model_version=None,
        complex_lookup_config_fp=None,
        output_directory=None
    ):
        """
        Class constructor
        """
        if keys_data_directory is not None:
            self.keys_data_directory = keys_data_directory
        else:
            self.keys_data_directory = os.path.join(os.sep, 'var', 'oasis', 'keys_data')

        self.supplier = supplier
        self.model_name = model_name
        self.model_version = model_version
        self.complex_lookup_config_fp = complex_lookup_config_fp
        self.output_directory = output_directory

    @oasis_log()
    def process_locations(self, loc_df):
        """
        Process location rows - passed in as a pandas dataframe.
        """
        pass

    def _get_area_peril_id(self, record):
        """
        Get the area peril ID for a particular location record.
        """
        return UNKNOWN_ID, "Not implemented"

    def _get_vulnerability_id(self, record):
        """
        Get the vulnerability ID for a particular location record.
        """
        return UNKNOWN_ID, "Not implemented"

    @oasis_log()
    def _get_area_peril_ids(self, loc_data, include_context=True):
        """
        Generates area peril IDs in two modes - if include_context is
        True (default) it will generate location records/rows including
        the area peril IDs, otherwise it will generate pairs of location
        IDs and the corresponding area peril IDs.
        """
        pass

    @oasis_log()
    def _get_vulnerability_ids(self, loc_data, include_context=True):
        """
        Generates vulnerability IDs in two modes - if include_context is
        True (default) it will generate location records/rows including
        the area peril IDs, otherwise it will generate pairs of location
        IDs and the corresponding vulnerability IDs.
        """
        pass

    def _get_custom_lookup_success(self, ap_id, vul_id):
        """
        Determine the status of the keys lookup.
        """
        if ap_id == UNKNOWN_ID or vul_id == UNKNOWN_ID:
            return OASIS_KEYS_STATUS['nomatch']['id']
        return OASIS_KEYS_STATUS['success']['id']


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
            lookup = OasisLookup(
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
        num_cores=None,
        num_partitions=None
    ):
        """
        Used for CPU bound lookup operations, Depends on a method

        `process_locations_multiproc(dataframe)`

        where single_row is a pandas series from a location Pandas DataFrame
        and returns a list of dicts holding the lookup results for that single row
        """
        pool_count = num_cores if num_cores else cpu_count()
        part_count = num_partitions if num_partitions else min(pool_count * 2, len(loc_df))
        locations = pd.np.array_split(loc_df, part_count)

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
    def save_results(
        cls,
        lookup,
        location_df,
        successes_fp=None,
        errors_fp=None,
        multiprocessing=True,
        format='oasis',
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
        sfp = as_path(successes_fp, 'successes_fp', preexists=False)
        efp = as_path(errors_fp, 'errors_fp', preexists=False)

        if (multiprocessing and hasattr(lookup, 'process_locations_multiproc')):
            # Multi-Process
            keys_generator = cls.get_keys_multiproc
        else:
            # Fall back to Single process
            keys_generator = cls.get_keys_builtin if hasattr(lookup, 'config') else (
                             cls.get_keys_base)
        kwargs = {
            "lookup": lookup,
            "loc_df": location_df,
            "success_only": (False if efp else True)
        }

        results = keys_generator(**kwargs)
        successes = []
        nonsuccesses = []

        # ToDO: Move the inside the keys_generators?  and return a tuple of (successes, nonsuccesses)
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


class OasisLookup(OasisBuiltinBaseLookup):
    """
    Combined peril and vulnerability lookup
    """
    @oasis_log()
    def __init__(
        self,
        config=None,
        config_json=None,
        config_fp=None,
        config_dir=None,
        areas=None,
        peril_areas=None,
        peril_areas_index=None,
        peril_areas_index_props=None,
        loc_to_global_areas_boundary_min_distance=0,
        vulnerabilities=None
    ):
        super(self.__class__, self).__init__(
            config=config,
            config_json=config_json,
            config_fp=config_fp,
            config_dir=config_dir,
        )

        self.peril_lookup = OasisPerilLookup(
            config=self.config,
            config_dir=self.config_dir,
            areas=areas,
            peril_areas=peril_areas,
            peril_areas_index=peril_areas_index,
            peril_areas_index_props=peril_areas_index_props,
            loc_to_global_areas_boundary_min_distance=loc_to_global_areas_boundary_min_distance
        )

        self.peril_area_id_key = str(str(self.config['peril'].get('peril_area_id_col') or '') or 'peril_area_id').lower()

        self.vulnerability_id_key = str(str(self.config['vulnerability'].get('vulnerability_id_col')) or 'vulnerability_id').lower()

        self.vulnerability_lookup = OasisVulnerabilityLookup(
            config=self.config,
            config_dir=self.config_dir,
            vulnerabilities=vulnerabilities
        )

    def lookup(self, loc, peril_id, coverage_type, **kwargs):

        loc_id = loc.get('loc_id') or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        plookup = self.peril_lookup.lookup(loc, peril_id, coverage_type)

        past = plookup['status']
        pamsg = plookup['message']
        paid = plookup['peril_area_id']

        vlookup = self.vulnerability_lookup.lookup(loc, peril_id, coverage_type)

        vlnst = vlookup['status']
        vlnmsg = vlookup['message']
        vlnid = vlookup['vulnerability_id']

        vlookup.pop('status')
        vlookup.pop('message')
        vlookup.pop('vulnerability_id')

        # Could optionally call the status lookup method, but faster
        # to avoid or minimise outside function calls in a `for` loop
        status = (
            OASIS_KEYS_STATUS['success']['id'] if past == vlnst == OASIS_KEYS_STATUS['success']['id']
            else (OASIS_KEYS_STATUS['fail']['id'] if (past == OASIS_KEYS_STATUS['fail']['id'] or vlnst == OASIS_KEYS_STATUS['fail']['id']) else OASIS_KEYS_STATUS['nomatch']['id'])
        )

        message = '{}; {}'.format(pamsg, vlnmsg)

        return {
            k: v for k, v in itertools.chain(
                (
                    ('loc_id', loc_id),
                    ('peril_id', peril_id),
                    ('coverage_type', coverage_type),
                    (self.peril_area_id_key, paid),
                    (self.vulnerability_id_key, vlnid),
                    ('status', status),
                    ('message', message),
                ),
                vlookup.items()
            )
        }


class OasisPerilLookup(OasisBuiltinBaseLookup):
    """
    Single peril, single coverage type, lon/lat point-area poly lookup using
    an Rtree index to store peril areas - index entries are
    #
    #    (peril area ID, peril area bounds)
    #
    # pairs. Areas must be represented as polygons with vertices which are
    # lon/lat coordinates, and be passed in to the constructor as a list,
    # tuple, or generator of triples of the form
    #
    #    (peril area ID, polygon lon/lat vertices, dict with optional properties)
    #
    # An optional distance measure ``loc_to_global_areas_boundary_min_distance`` can be passed
    # that defines how far, in an abstract unit, a given lon/lat location can be
    # from the boundary of the polygon containing all the individual peril area
    # polygons in order to be assigned an peril area ID. By default this distance
    # is 0, which means any lon/lat location outside the polygon containing all
    # peril area polygons will not be assigned a peril area ID.
    """

    @oasis_log()
    def __init__(
        self,
        areas=None,
        config=None,
        config_json=None,
        config_fp=None,
        config_dir=None,
        loc_to_global_areas_boundary_min_distance=0,
        peril_areas=None,
        peril_areas_index=None,
        peril_areas_index_fp=None,
        peril_areas_index_props=None
    ):
        super(self.__class__, self).__init__(config=config, config_json=config_json, config_fp=config_fp, config_dir=config_dir)

        peril_config = self.config.get('peril') or {}

        if areas or peril_areas or peril_config:
            if peril_areas_index:
                self.peril_areas_index = peril_areas_index
                self.peril_areas_index_props = self.peril_areas_index_props.properties.as_dict()
            elif (areas or peril_areas):
                self.index_props = (
                    peril_areas_index_props or
                    peril_config.get('rtree_index') or
                    DEFAULT_RTREE_INDEX_PROPS
                )
                self.peril_areas_index = PerilAreasIndex(areas=areas, peril_areas=peril_areas, properties=self.index_props)
            else:
                areas_rtree_index_config = peril_config.get('rtree_index') or {}
                index_fp = peril_areas_index_fp or areas_rtree_index_config.get('filename')

                if not os.path.isabs(index_fp):
                    index_fp = os.path.join(self.config_dir, index_fp)
                    index_fp = as_path(index_fp, 'index_fp', preexists=False)

                if index_fp:
                    idx_ext = areas_rtree_index_config.get('idx_extension') or 'idx'
                    dat_ext = areas_rtree_index_config.get('dat_extension') or 'dat'
                    if not (os.path.exists('{}.{}'.format(index_fp, idx_ext)) or os.path.exists('{}.{}'.format(index_fp, dat_ext))):
                        raise OasisException('No Rtree file index {}.{{{}, {}}} found'.format(index_fp, idx_ext, dat_ext))
                    self.peril_areas_index = PerilAreasIndex(fp=index_fp)
                    self.peril_areas_index_props = self.peril_areas_index.properties.as_dict()

            self.peril_areas_boundary = box(*self.peril_areas_index.bounds, ccw=False)

            _centroid = self.peril_areas_boundary.centroid
            self.peril_areas_centre = _centroid.x, _centroid.y

            self.loc_to_global_areas_boundary_min_distance = (
                loc_to_global_areas_boundary_min_distance or
                self.config['peril'].get('loc_to_global_areas_boundary_min_distance') or 0
            )

        if self.config.get('exposure') or self.config.get('locations'):
            self.loc_coords_x_col = str.lower(str(self.config['exposure'].get('coords_x_col')) or 'lon')
            self.loc_coords_y_col = str.lower(str(self.config['exposure'].get('coords_y_col')) or 'lat')
            self.loc_coords_x_bounds = tuple(self.config['exposure'].get('coords_x_bounds') or ()) or (-180, 180)
            self.loc_coords_y_bounds = tuple(self.config['exposure'].get('coords_y_bounds') or ()) or (-90, 90)

    def lookup(self, loc, peril_id, coverage_type, **kwargs):
        """
        Area peril lookup for an individual lon/lat location item, which can be
        provided as a dict or a Pandas series. The data structure should contain
        the keys `lon` or `longitude` for longitude and `lat` or `latitude` for
        latitude.
        """
        idx = self.peril_areas_index
        boundary = self.peril_areas_boundary
        loc_to_areas_min_dist = self.loc_to_global_areas_boundary_min_distance

        loc_id = loc.get('loc_id') or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        loc_x_col = self.loc_coords_x_col
        loc_y_col = self.loc_coords_y_col
        loc_x_bounds = self.loc_coords_x_bounds
        loc_y_bounds = self.loc_coords_y_bounds

        x = loc.get(loc_x_col)
        y = loc.get(loc_y_col)

        def _lookup(loc_id, x, y, st, perid, covtype, paid, pabnds, pacoords, msg):
            return {
                'loc_id': loc_id,
                loc_x_col: x,
                loc_y_col: y,
                'peril_id': perid,
                'coverage_type': covtype,
                'status': st,
                'peril_area_id': paid,
                'area_peril_id': paid,
                'area_bounds': pabnds,
                'area_coordinates': pacoords,
                'message': msg
            }

        try:
            x = float(x)
            y = float(y)
            if not ((loc_x_bounds[0] <= x <= loc_x_bounds[1]) and (loc_y_bounds[0] <= y <= loc_y_bounds[1])):
                raise ValueError('{}/{} out of bounds'.format(loc_x_col, loc_y_col))
        except (ValueError, TypeError) as e:
            msg = (
                'Peril area lookup: invalid {}/{} ({}, {}) - {}'
                .format(loc_x_col, loc_y_col, x, y, str(e))
            )
            return _lookup(loc_id, x, y, OASIS_KEYS_STATUS['fail']['id'], peril_id, coverage_type, None, None, None, msg)

        st = OASIS_KEYS_STATUS['nomatch']['id']
        msg = 'No peril area match'
        paid = None
        pabnds = None
        pacoords = None
        point = x, y

        try:
            results = list(idx.intersection(point, objects='raw'))

            if not results:
                raise IndexError

            for _perid, _covtype, _paid, _pabnds, _pacoords in results:
                if (peril_id, coverage_type) == (_perid, _covtype):
                    paid, pabnds, pacoords = _paid, _pabnds, _pacoords
                    break

            if paid is None:
                raise IndexError
        except IndexError:
            try:
                results = list(idx.nearest(point, objects='raw'))

                if not results:
                    raise IndexError

                for _perid, _covtype, _paid, _pabnds, _pacoords in results:
                    if (peril_id, coverage_type) == (_perid, _covtype):
                        paid, pabnds, pacoords = _paid, _pabnds, _pacoords
                        break

                if paid is None:
                    msg = 'No intersecting or nearest peril area found for peril ID {} and coverage type {}'.format(peril_id, coverage_type)
                    return _lookup(loc_id, x, y, OASIS_KEYS_STATUS['nomatch']['id'], peril_id, coverage_type, None, None, None, msg)
            except IndexError:
                pass
            else:
                p = Point(x, y)
                min_dist = p.distance(boundary)
                if min_dist > loc_to_areas_min_dist:
                    msg = (
                        'Peril area lookup: location is {} units from the '
                        'peril areas global boundary -  the required minimum '
                        'distance is {} units'
                        .format(min_dist, loc_to_areas_min_dist)
                    )
                    return _lookup(loc_id, x, y, OASIS_KEYS_STATUS['fail']['id'], peril_id, coverage_type, None, None, None, msg)
                st = OASIS_KEYS_STATUS['success']['id']
                msg = (
                    'Successful peril area lookup: {}'.format(paid)
                )
        except RTreeError as e:
            return _lookup(loc_id, x, y, OASIS_KEYS_STATUS['fail']['id'], peril_id, coverage_type, None, None, None, str(e))
        else:
            st = OASIS_KEYS_STATUS['success']['id']
            msg = 'Successful peril area lookup: {}'.format(paid)

        return _lookup(loc_id, x, y, st, peril_id, coverage_type, paid, pabnds, pacoords, msg)


class OasisVulnerabilityLookup(OasisBuiltinBaseLookup):
    """
    Simple key-value based vulnerability lookup
    """

    @oasis_log()
    def __init__(
        self,
        config=None,
        config_json=None,
        config_fp=None,
        config_dir=None,
        vulnerabilities=None
    ):
        super(self.__class__, self).__init__(config=config, config_json=config_json, config_fp=config_fp, config_dir=config_dir)

        if vulnerabilities or self.config.get('vulnerability'):
            self.col_dtypes, self.key_cols, self.vuln_id_col, self.vulnerabilities = self.get_vulnerabilities(vulnerabilities=vulnerabilities)

    @oasis_log()
    def get_vulnerabilities(self, vulnerabilities=None):
        if not self.config:
            raise OasisException(
                'No lookup configuration provided or set - use `get_config` '
                'on this instance to set it and provide either an actual '
                'model config dict (use `model_config` argument), or a model '
                'config JSON string (use `model_config_json` argument, or a '
                'model config JSON file path (use `model_config_fp` argument)'
            )

        vuln_config = self.config.get('vulnerability')

        if not vuln_config:
            raise OasisException('No vulnerability config set in the lookup config')

        col_dtypes = vuln_config.get('col_dtypes')

        if not col_dtypes:
            raise OasisException(
                'Vulnerability file column data types must be defined as a '
                '(col, data type) dict in the vulnerability section of the '
                'lookup config'
            )

        col_dtypes = {
            k.lower(): getattr(builtins, v) for k, v in col_dtypes.items()
        }

        key_cols = vuln_config.get('key_cols')

        if not vuln_config.get('key_cols'):
            raise OasisException(
                'The vulnerability file key column names must be listed in the '
                'vulnerability section of the lookup config'
            )

        key_cols = tuple(col.lower() for col in key_cols)

        vuln_id_col = str(str(self.config['vulnerability'].get('vulnerability_id_col')) or 'vulnerability_id').lower()

        def _vuln_dict(vulns_seq, key_cols, vuln_id_col):
            return (
                {v[key_cols[0]]: (v.get(vuln_id_col) or v.get('vulnerability_id')) for _, v in vulns_seq} if len(key_cols) == 1
                else OrderedDict(
                    {tuple(v[key_cols[i]] for i in range(len(key_cols))): (v.get(vuln_id_col) or v.get('vulnerability_id')) for v in vulns_seq}
                )
            )

        if vulnerabilities:
            return col_dtypes, key_cols, vuln_id_col, _vuln_dict(enumerate(vulnerabilities), key_cols)

        src_fp = vuln_config.get('file_path')

        if not src_fp:
            raise OasisException(
                'No vulnerabilities file path provided in the lookup config'
            )

        if not os.path.isabs(src_fp):
            src_fp = os.path.join(self.config_dir, src_fp)
            src_fp = os.path.abspath(src_fp)

        self.config['vulnerability']['file_path'] = src_fp

        src_type = str(str(vuln_config.get('file_type')) or 'csv').lower()

        float_precision = 'high' if vuln_config.get('float_precision_high') else None

        non_na_cols = vuln_config.get('non_na_cols') or tuple(col.lower() for col in list(key_cols) + [vuln_id_col])

        sort_cols = vuln_config.get('sort_cols') or vuln_id_col
        sort_ascending = vuln_config.get('sort_ascending')

        vuln_df = get_dataframe(
            src_fp=src_fp,
            src_type=src_type,
            float_precision=float_precision,
            lowercase_cols=True,
            non_na_cols=non_na_cols,
            col_dtypes=col_dtypes,
            sort_cols=sort_cols,
            sort_ascending=sort_ascending
        )

        return col_dtypes, key_cols, vuln_id_col, _vuln_dict((v for _, v in vuln_df.iterrows()), key_cols, vuln_id_col)

    def lookup(self, loc, peril_id, coverage_type, **kwargs):
        """
        Vulnerability lookup for an individual location item, which could be a dict or a
        Pandas series.
        """
        loc_id = loc.get('loc_id') or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        key_cols = self.key_cols
        col_dtypes = self.col_dtypes

        loc_key_col_values = OrderedDict({
            key_col: loc.get(key_col) for key_col in key_cols
        })

        if not loc_key_col_values['peril_id']:
            loc_key_col_values['peril_id'] = peril_id

        if not loc_key_col_values['coverage_type']:
            loc_key_col_values['coverage_type'] = loc.get('coverage') or coverage_type

        def _lookup(loc_id, vlnperid, vlncovtype, vlnst, vlnid, vlnmsg):
            return {
                k: v for k, v in itertools.chain(
                    (
                        ('loc_id', loc_id),
                        ('peril_id', vlnperid),
                        ('coverage_type', vlncovtype),
                        ('status', vlnst),
                        ('vulnerability_id', vlnid),
                        ('message', vlnmsg)
                    ),
                    loc_key_col_values.items()
                )
            }

        try:
            for key_col in key_cols:
                key_col_dtype = col_dtypes[key_col]
                key_col_dtype(loc_key_col_values[key_col])
        except (TypeError, ValueError):
            return _lookup(loc_id, peril_id, coverage_type, OASIS_KEYS_STATUS['fail']['id'], None, 'Vulnerability lookup: invalid key column value(s) for location')

        vlnperid = peril_id
        vlncovtype = coverage_type
        vlnst = OASIS_KEYS_STATUS['nomatch']['id']
        vlnmsg = 'No vulnerability match'
        vlnid = None

        try:
            vlnid = (
                self.vulnerabilities[tuple(loc_key_col_values[col] for col in key_cols)] if len(key_cols) > 1
                else self.vulnerabilities[loc[key_cols[0]]]
            )
        except KeyError:
            pass
        else:
            vlnperid = peril_id
            vlncovtype = coverage_type
            vlnst = OASIS_KEYS_STATUS['success']['id']
            vlnmsg = 'Successful vulnerability lookup: {}'.format(vlnid)

        return _lookup(loc_id, vlnperid, vlncovtype, vlnst, vlnid, vlnmsg)
