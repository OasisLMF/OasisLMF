# -*- coding: utf-8 -*-

from __future__ import unicode_literals, absolute_import

__all__ = [
    'OasisBaseLookup',
    'OasisBaseKeysLookup',
    'OasisLookup',
    'OasisPerilLookup',
    'OasisVulnerabilityLookup',
    'OasisLookupFactory'
]

import builtins
import csv
import imp
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

import pandas as pd
import six

from shapely.geometry import (
    box,
    Point,
)

from shapely import speedups as shapely_speedups

if shapely_speedups.available:
    shapely_speedups.enable()

from rtree.core import RTreeError

import six

from ..utils.data import get_dataframe
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.peril import (
    DEFAULT_RTREE_INDEX_PROPS,
    PerilAreasIndex,
)
from ..utils.status import (
    KEYS_STATUS_FAIL,
    KEYS_STATUS_NOMATCH,
    KEYS_STATUS_SUCCESS,
)
from ..utils.values import is_string


UNKNOWN_ID = -1

def as_path(value, name, preexists=True):
    """
    Processes the path and returns the absolute path.

    If the path does not exist and ``preexists`` is true
    an ``OasisException`` is raised.

    :param value: The path to process
    :type value: str

    :param name: The name of the path (used for error reporting)
    :type name: str

    :param preexists: Flag whether to raise an error if the path
        does not exist.
    :type preexists: bool

    :return: The absolute path of the input path
    """
    if value is not None:
        value = os.path.abspath(value) if not os.path.isabs(value) else value

    if preexists and not (value is not None and os.path.exists(value)):
        raise OasisException('{} does not exist: {}'.format(name, value))

    return value


class OasisBaseLookup(object):

    @oasis_log()
    def __init__(self, config=None, config_json=None, config_fp=None):
        if config:
            self._config = config
        elif config_json:
            self._config = json.loads(config_json)
        elif config_fp:
            _config_fp = as_path(config_fp, 'config_fp')
            with io.open(_config_fp, 'r', encoding='utf-8') as f:
                self._config = json.load(f)

        keys_data_path = self._config.get('keys_data_path') or ''

        self._config['keys_data_path'] = as_path(keys_data_path, 'keys_data_path', preexists=(True if keys_data_path else False))

        self.__tweak_config_data__()

    def __tweak_config_data__(self):
        for section in ('locations', 'peril', 'vulnerability',):
            section_config = self._config.get(section) or {}
            for k, v in six.iteritems(section_config):
                if is_string(v) and '%%KEYS_DATA_PATH%%' in v:
                    self._config[section][k] = v.replace('%%KEYS_DATA_PATH%%', self._config['keys_data_path'])
                elif type(v) == list:
                    self._config[section][k] = tuple(v)
                elif isinstance(v, dict):
                    for _k, _v in six.iteritems(v):
                        if is_string(_v) and '%%KEYS_DATA_PATH%%' in _v:
                            self._config[section][k][_k] = _v.replace('%%KEYS_DATA_PATH%%', self._config['keys_data_path'])

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, c):
        self._config = c
        self.__tweak_config_data__()

    def lookup(self, loc, **kwargs):
        """
        Lookup for an individual location item, which could be a dict or a
        Pandas series object.
        """
        pass

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
            locs_seq = six.itervalues(locs)
        elif isinstance(locs, pd.DataFrame):
            locs_seq = (loc for _, loc in locs.iterrows())

        for loc in locs_seq:
            yield self.lookup(loc)


class OasisBaseKeysLookup(object):  # pragma: no cover
    """
    A base class / interface that serves a template for model-specific keys
    lookup classes.
    """
    @oasis_log()
    def __init__(
        self,
        keys_data_directory=None,
        supplier=None,
        model_name=None,
        model_version=None
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

    @oasis_log()
    def process_locations(self, loc_df):
        """
        Process location rows - passed in as a pandas dataframe.
        """
        pass

    def _get_location_record(self, raw_loc_item):
        """
        Returns a dict of standard location keys and values based on
        a raw location item, which is a row in a Pandas dataframe.
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

    def _get_lookup_success(self, ap_id, vul_id):
        """
        Determine the status of the keys lookup.
        """
        if ap_id == UNKNOWN_ID or vul_id == UNKNOWN_ID:
            return KEYS_STATUS_NOMATCH
        return KEYS_STATUS_SUCCESS


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
        with io.open(model_version_file_path, 'r', encoding='utf-8') as f:
            return next(csv.DictReader(
                f, fieldnames=['supplier_id', 'model_id', 'model_version']
            ))

    @classmethod
    def get_lookup_package(cls, lookup_package_path):
        """
        Returns the lookup service parent package (called `keys_server` and
        located in `src` in the model keys server Git repository or in
        `var/www/oasis` in the keys server Docker container) from the given
        path.
        """
        if lookup_package_path and not os.path.isabs(lookup_package_path):
            lookup_package_path = os.path.abspath(lookup_package_path)

        parent_dir = os.path.abspath(os.path.dirname(lookup_package_path))
        package_name = re.sub(r'\.py$', '', os.path.basename(lookup_package_path))

        sys.path.insert(0, parent_dir)
        lookup_package = importlib.import_module(package_name)
        imp.reload(lookup_package)
        sys.path.pop(0)

        return lookup_package

    @classmethod
    def get_lookup_class_instance(cls, lookup_package, keys_data_path, model_info):
        """
        Get the keys lookup class instance.
        """
        klc = getattr(lookup_package, '{}KeysLookup'.format(model_info['model_id']))

        return klc(
            keys_data_directory=keys_data_path,
            supplier=model_info['supplier_id'],
            model_name=model_info['model_id'],
            model_version=model_info['model_version']
        )

    @classmethod
    def get_model_exposures(cls, model_exposures=None, model_exposures_file_path=None):
        """
        Get the model exposures/location file data as a pandas dataframe given
        either the path of the model exposures file or the string contents of
        such a file.
        """
        if model_exposures_file_path:
            loc_df = pd.read_csv(os.path.abspath(model_exposures_file_path), float_precision='high')
        elif model_exposures:
            loc_df = pd.read_csv(six.StringIO(model_exposures), float_precision='high')
        else:
            raise OasisException('Either model_exposures_file_path or model_exposures must be specified')

        loc_df = loc_df.where(loc_df.notnull(), None)
        loc_df.columns = loc_df.columns.str.lower()

        return loc_df

    @classmethod
    def write_oasis_keys_file(cls, records, output_file_path, id_col='id'):
        """
        Writes an Oasis keys file from an iterable of keys records.
        """
        heading_row = OrderedDict([
            (id_col, 'LocID'),
            ('peril_id', 'PerilID'),
            ('coverage', 'CoverageID'),
            ('area_peril_id', 'AreaPerilID'),
            ('vulnerability_id', 'VulnerabilityID'),
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
    def write_oasis_keys_errors_file(cls, records, output_file_path, id_col='id'):
        """
        Writes an Oasis keys errors file from an iterable of keys records.
        """
        heading_row = OrderedDict([
            (id_col, 'LocID'),
            ('peril_id', 'PerilID'),
            ('coverage', 'CoverageID'),
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
        lookup_package_path=None,
        lookup_config=None,
        lookup_config_json=None,
        lookup_config_fp=None,
        lookup_type='combined',
        loc_id_col='id'
    ):
        """
        Creates a keys lookup class instance for the given model and supplier -
        local file paths are required for the model keys data folder, the model
        version file and the Git repository for the model keys server. Returns a
        pair ``(model_info, klc)``, where ``model_info`` is a dictionary holding
        model information from the model version file and `klc` is the lookup
        service class instance for the model.
        """
        if (lookup_config or lookup_config_json or lookup_config_fp):
            lookup = OasisLookup(
                config=lookup_config,
                config_json=lookup_config_json,
                config_fp=lookup_config_fp,
                loc_id_col=loc_id_col
            )
            model_info = lookup.config.get('model')
            if lookup_type == 'base':
                lookup = OasisBaseLookup(
                    config=lookup_config,
                    config_json=lookup_config_json,
                    config_fp=lookup_config_fp
                )
                return lookup.config.get('model'), lookup
            elif lookup_type == 'combined':
                return model_info, lookup
            elif lookup_type == 'peril':
                return model_info, lookup.peril_lookup
            elif lookup_type == 'vulnerability':
                return model_info, lookup.vulnerability_lookup
        else:
            _model_keys_data_path = as_path(model_keys_data_path, 'model_keys_data_path', preexists=True)
            _model_version_file_path = as_path(model_version_file_path, 'model_version_file_path', preexists=True)
            _lookup_package_path = as_path(lookup_package_path, 'lookup_package_path', preexists=True)

            model_info = cls.get_model_info(_model_version_file_path)
            lookup_package = cls.get_lookup_package(_lookup_package_path)
        
            return model_info, cls.get_lookup_class_instance(lookup_package, _model_keys_data_path, model_info)

    @classmethod
    def get_keys(
        cls,
        lookup=None,
        model_exposures=None,
        model_exposures_file_path=None,
        success_only=True
    ):
        """
        Generates keys records (JSON) for the given model and supplier -
        requires an instance of the lookup service (which can be created using
        the `create` method in this factory class), and either the model
        location file path or the string contents of such a file.

        The optional keyword argument ``success_only`` indicates whether only
        records with successful lookups should be returned (default), or all
        records.
        """
        if not (model_exposures or model_exposures_file_path):
            raise OasisException('No model exposures provided')

        model_loc_df = cls.get_model_exposures(
            model_exposures_file_path=model_exposures_file_path,
            model_exposures=model_exposures
        )

        for record in lookup.process_locations(model_loc_df):
            if success_only:
                if record['status'].lower() == KEYS_STATUS_SUCCESS:
                    yield record
            else:
                yield record

    @classmethod
    def get_results(
        cls,
        lookup,
        model_exposures=None,
        model_exposures_fp=None,
        successes_only=False,
        **kwargs
    ):
        """
        Generates lookup results (dicts) for the given model and supplier -
        requires a lookup instance (which can be created using the `create2`
        method in this factory class), and the model exposures/locations
        dataframe.

        The optional keyword argument ``success_only`` indicates whether only
        results with successful lookup status should be returned (default),
        or all results.
        """
        if not (model_exposures or model_exposures_fp):
            raise OasisException('No model exposures data or file path provided')

        peril_config = lookup.config.get('peril')
        if not peril_config:
            raise OasisException('No peril config defined in the lookup config')

        _model_exposures_fp = as_path(model_exposures_fp, 'model_exposures_fp', preexists=False)

        loc_config = lookup.config.get('locations') or {}
        src_type = 'csv'

        kwargs = {
            'src_data': model_exposures,
            'src_fp': _model_exposures_fp,
            'src_type': 'csv',
            'non_na_cols': tuple(loc_config.get('non_na_cols') or ()),
            'col_dtypes': loc_config.get('col_dtypes') or {},
            'sort_col': loc_config.get('sort_col'),
            'sort_ascending': loc_config.get('sort_ascending')
        }

        model_exposures_df =  get_dataframe(**kwargs)

        locations = (loc for _, loc in model_exposures_df.iterrows())

        for result in lookup.bulk_lookup(locations):
            if successes_only:
                if result['status'].lower() == KEYS_STATUS_SUCCESS:
                    yield result
            else:
                yield result

    @classmethod
    def save_keys(
        cls,
        lookup=None,
        keys_id_col='id',
        keys_file_path=None,
        keys_errors_file_path=None,
        keys_format='oasis',
        model_exposures=None,
        model_exposures_file_path=None,
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
        if not (model_exposures or model_exposures_file_path):
            raise OasisException('No model exposures or model exposures file path provided')

        _keys_file_path = as_path(keys_file_path, 'keys_file_path', preexists=False)
        _keys_errors_file_path = as_path(keys_errors_file_path, 'keys_errors_file_path', preexists=False)
        _model_exposures_file_path = as_path(model_exposures_file_path, 'model_exposures_file_path', preexists=False)

        keys = cls.get_keys(
            lookup=lookup,
            model_exposures=model_exposures,
            model_exposures_file_path=_model_exposures_file_path,
            success_only=(True if not keys_errors_file_path else False)
        )

        successes = []
        nonsuccesses = []
        for k in keys:
            successes.append(k) if k['status'] == KEYS_STATUS_SUCCESS else nonsuccesses.append(k)

        if keys_format == 'json':
            if _keys_error_file_path:
                fp1, n1 = cls.write_json_keys_file(successes, _keys_file_path)
                fp2, n2 = cls.write_json_keys_file(nonsuccesses, _keys_errors_file_path)
                return fp1, n1, fp2, n2
            return cls.write_json_keys_file(successes, _keys_file_path)
        elif keys_format == 'oasis':
            if _keys_errors_file_path:
                fp1, n1 = cls.write_oasis_keys_file(successes, _keys_file_path, id_col=keys_id_col)
                fp2, n2 = cls.write_oasis_keys_errors_file(nonsuccesses, _keys_errors_file_path, id_col=keys_id_col)
                return fp1, n1, fp2, n2
            return cls.write_oasis_keys_file(successes, _keys_file_path, id_col=keys_id_col)
        else:
            raise OasisException("Unrecognised keys file output format - valid formats are 'oasis' or 'json'")

    @classmethod
    def save_results(
        cls,
        lookup,
        successes_fp,
        errors_fp=None,
        model_exposures=None,
        model_exposures_fp=None,
        format='oasis'
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
        if not (model_exposures or model_exposures_fp):
            raise OasisException('No model exposures data or file path provided')

        mfp = as_path(model_exposures_fp, 'model_exposures_fp', preexists=False)

        sfp = as_path(successes_fp, 'successes_fp', preexists=False)
        efp = as_path(errors_fp, 'errors_fp', preexists=False)

        results = None

        try:
            config = lookup.config
        except AttributeError:
            results = cls.get_keys(
                lookup=lookup,
                model_exposures=model_exposures,
                model_exposures_file_path=mfp,
                success_only=(False if efp else True)
            )
        else:
            results = cls.get_results(
                lookup,
                model_exposures=model_exposures,
                model_exposures_fp=mfp,
                successes_only=(False if efp else True)
            )

        successes = []
        nonsuccesses = []
        for r in results:
            successes.append(r) if r['status'] == KEYS_STATUS_SUCCESS else nonsuccesses.append(r)

        if format == 'json':
            if efp:
                fp1, n1 = cls.write_json_keys_file(successes, sfp)
                fp2, n2 = cls.write_json_keys_file(nonsuccesses, efp)
                return fp1, n1, fp2, n2
            return cls.write_json_keys_file(successes, sfp)
        elif format == 'oasis':
            loc_id_col = None
            try:
                loc_id_col = lookup.loc_id_col
            except AttributeError:
                loc_id_col = 'id'
            else:
                loc_id_col = loc_id_col.lower()
            if efp:
                fp1, n1 = cls.write_oasis_keys_file(successes, sfp, id_col=loc_id_col)
                fp2, n2 = cls.write_oasis_keys_errors_file(nonsuccesses, efp, id_col=loc_id_col)
                return fp1, n1, fp2, n2
            return cls.write_oasis_keys_file(successes, sfp, id_col=loc_id_col)
        else:
            raise OasisException("Unrecognised lookup file output format - valid formats are 'oasis' or 'json'")


class OasisLookup(OasisBaseLookup):
    """
    Combined peril and vulnerability lookup
    """
    @oasis_log()
    def __init__(
        self,
        config=None,
        config_json=None,
        config_fp=None,
        areas=None,
        peril_id=None,
        peril_areas=None,
        peril_areas_index=None,
        peril_areas_index_props=None,
        loc_to_global_areas_boundary_min_distance=0,
        vulnerabilities=None,
        loc_id_col='id'
    ):
        super(self.__class__, self).__init__(
            config=config,
            config_json=config_json,
            config_fp=config_fp
        )

        loc_config = self.config.get('locations')
        self.loc_id_col = str.lower(str(loc_config.get('id_col') or loc_id_col))

        self.peril_lookup = OasisPerilLookup(
            config=self.config,
            areas=areas,
            peril_id=peril_id,
            peril_areas=peril_areas,
            peril_areas_index=peril_areas_index,
            peril_areas_index_props=peril_areas_index_props,
            loc_to_global_areas_boundary_min_distance=loc_to_global_areas_boundary_min_distance,
            loc_id_col=self.loc_id_col
        )

        self.peril_id = peril_id or self.config['peril'].get('peril_id')

        self.peril_area_id_key = str(str(self.config['peril'].get('peril_area_id_col') or '') or 'peril_area_id').lower()

        self.vulnerability_id_key = str(str(self.config['vulnerability'].get('vulnerability_id_col')) or 'vulnerability_id').lower()

        self.vulnerability_lookup = OasisVulnerabilityLookup(
            config=self.config,
            vulnerabilities=vulnerabilities,
            loc_id_col=self.loc_id_col
        )

    def lookup(self, loc):

        loc_id_col = self.loc_id_col
        loc_id = loc.get(loc_id_col) or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        plookup = self.peril_lookup.lookup(loc)
        past = plookup['status']
        pamsg = plookup['message']
        paid = plookup['peril_area_id']
        
        vlookup = self.vulnerability_lookup.lookup(loc)
        vlnst = vlookup['status']
        vlnmsg = vlookup['message']
        vlnid = vlookup['vulnerability_id']
        vlookup.pop('status')
        vlookup.pop('message')
        vlookup.pop('vulnerability_id')

        # Could optionally call the status lookup method, but it is always
        # better to avoid outside function calls in a `for` loop if possible
        status = (
            KEYS_STATUS_SUCCESS if past == vlnst == KEYS_STATUS_SUCCESS
            else (KEYS_STATUS_FAIL if (past == KEYS_STATUS_FAIL or vlnst == KEYS_STATUS_FAIL) else KEYS_STATUS_NOMATCH)
        )
        
        message = '{}; {}'.format(pamsg, vlnmsg)

        return {
            k:v for k, v in itertools.chain(
                (
                    (loc_id_col, loc_id),
                    ('peril_id', self.peril_id),
                    (self.peril_area_id_key, paid),
                    (self.vulnerability_id_key, vlnid),
                    ('status', status),
                    ('message', message),
                ),
                six.iteritems(vlookup)
            )
        }


class OasisPerilLookup(OasisBaseLookup):
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
        loc_to_global_areas_boundary_min_distance=0,
        peril_areas=None,
        peril_areas_index=None,
        peril_areas_index_fp=None,
        peril_areas_index_props=None,
        peril_id=None,
        loc_id_col='id'
    ):
        super(self.__class__, self).__init__(config=config, config_json=config_json, config_fp=config_fp)

        peril_config = self.config.get('peril') or {}

        self.peril_id = peril_id or peril_config.get('peril_id')

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
                index_fp = as_path(peril_areas_index_fp or areas_rtree_index_config.get('filename'), 'index_fp', preexists=False)
                if index_fp:
                    idx_ext = areas_rtree_index_config.get('idx_extension') or 'idx'
                    dat_ext = areas_rtree_index_config.get('dat_extension') or 'dat'
                    if not (os.path.exists('{}.{}'.format(index_fp, idx_ext)) or os.path.exists('{}.{}'.format(index_fp, dat_ext))):
                        raise OasisException('No Rtree file index {}.{{idx_ext, dat_ext}} found'.format(index_fp))
                    self.peril_areas_index = PerilAreasIndex(fp=index_fp)
                    self.peril_areas_index_props = self.peril_areas_index.properties.as_dict()

            self.peril_areas_boundary = box(*self.peril_areas_index.bounds, ccw=False)

            _centroid = self.peril_areas_boundary.centroid
            self.peril_areas_centre = _centroid.x, _centroid.y

            self.loc_to_global_areas_boundary_min_distance = (
                loc_to_global_areas_boundary_min_distance or 
                self.config['peril'].get('loc_to_global_areas_boundary_min_distance') or 0
            )

        if self.config.get('locations'):
            self.loc_id_col = str.lower(str(self.config['locations'].get('id_col') or loc_id_col))
            self.loc_coords_x_col = str.lower(str(self.config['locations'].get('coords_x_col')) or 'lon')
            self.loc_coords_y_col = str.lower(str(self.config['locations'].get('coords_y_col')) or 'lat')
            self.loc_coords_x_bounds = tuple(self.config['locations'].get('coords_x_bounds') or ()) or (-180, 180)
            self.loc_coords_y_bounds = tuple(self.config['locations'].get('coords_y_bounds') or ()) or (-90, 90)

    def lookup(self, loc):
        """
        Area peril lookup for an individual lon/lat location item, which can be
        provided as a dict or a Pandas series. The data structure should contain
        the keys `lon` or `longitude` for longitude and `lat` or `latitude` for
        latitude.
        """
        peril_id = self.peril_id
        idx = self.peril_areas_index
        boundary = self.peril_areas_boundary
        loc_to_areas_min_dist = self.loc_to_global_areas_boundary_min_distance

        loc_id_col = self.loc_id_col

        loc_id = loc.get(loc_id_col) or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        loc_x_col = self.loc_coords_x_col
        loc_y_col = self.loc_coords_y_col
        loc_x_bounds = self.loc_coords_x_bounds
        loc_y_bounds = self.loc_coords_y_bounds

        x = loc.get(loc_x_col)
        y = loc.get(loc_y_col)

        _lookup = lambda loc_id, x, y, st, paid, pabnds, msg: {
            loc_id_col: loc_id,
            loc_x_col: x,
            loc_y_col: y,
            'peril_id': peril_id,
            'status': st,
            'peril_area_id': paid,
            'area_peril_id': paid,
            'area_bounds': pabnds,
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
            return _lookup(loc_id, x, y, KEYS_STATUS_FAIL, None, None, msg)

        st = KEYS_STATUS_NOMATCH
        msg = 'No peril area match'
        paid = None
        pabnds = None
        point = x, y

        try:
            paid, pabnds = list(idx.intersection(point, objects='raw'))[0]
        except IndexError:
            try:
                paid, pabnds = list(idx.nearest(point, objects='raw'))[0]
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
                    return _lookup(loc_id, x, y, KEYS_STATUS_FAIL, None, None, msg)
                st = KEYS_STATUS_SUCCESS
                msg = (
                    'Successful peril area lookup: {}'.format(paid)
                )
        except RTreeError as e:
            return _lookup(loc_id, x, y, KEYS_STATUS_FAIL, None, None, str(e))
        else:
            st = KEYS_STATUS_SUCCESS
            msg = 'Successful peril area lookup: {}'.format(paid)

        return _lookup(loc_id, x, y, st, paid, pabnds, msg)


class OasisVulnerabilityLookup(OasisBaseLookup):
    """
    Simple key-value based vulnerability lookup
    """

    @oasis_log()
    def __init__(
        self,
        config=None,
        config_json=None,
        config_fp=None,
        vulnerabilities=None,
        loc_id_col='id'
    ):
        super(self.__class__, self).__init__(config=config, config_json=config_json, config_fp=config_fp)

        if vulnerabilities or self.config.get('vulnerability'):
            self.col_dtypes, self.key_cols, self.vuln_id_col, self.vulnerabilities = self.get_vulnerabilities(vulnerabilities=vulnerabilities)

        if self.config.get('locations'):
            self.loc_id_col = str.lower(str(self.config['locations'].get('id_col') or loc_id_col))

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
            k.lower():getattr(builtins, v) for k, v in six.iteritems(col_dtypes)
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
                {v[key_cols[0]]:(v.get(vuln_id_col) or v.get('vulnerability_id')) for _, v in vulns_seq} if len(key_cols) == 1
                else OrderedDict(
                    {tuple(v[key_cols[i]] for i in range(len(key_cols))):(v.get(vuln_id_col) or v.get('vulnerability_id')) for _, v in vulns_seq}
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
            src_fp = os.path.abspath(src_fp)

        self.config['vulnerability']['file_path'] = src_fp

        src_type = str(str(vuln_config.get('file_type')) or 'csv').lower()

        float_precision = 'high' if vuln_config.get('float_precision_high') else None

        non_na_cols = vuln_config.get('non_na_cols') or tuple(col.lower() for col in list(key_cols) + [vuln_id_col])

        sort_col = vuln_config.get('sort_col') or vuln_id_col
        sort_ascending = vuln_config.get('sort_ascending')

        vuln_df = get_dataframe(
            src_fp=src_fp,
            src_type=src_type,
            float_precision=float_precision,
            lowercase_cols=True,
            index_col=True,
            non_na_cols=non_na_cols,
            col_dtypes=col_dtypes,
            sort_col=sort_col,
            sort_ascending=sort_ascending
        )

        return col_dtypes, key_cols, vuln_id_col, _vuln_dict(vuln_df.iterrows(), key_cols, vuln_id_col)

    def lookup(self, loc):
        """
        Vulnerability lookup for an individual location item, which could be a dict or a
        Pandas series.
        """
        loc_id_col = self.loc_id_col
        loc_id = loc.get(loc_id_col) or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        key_cols = self.key_cols
        col_dtypes = self.col_dtypes
        vuln_id_col = self.vuln_id_col

        loc_key_col_values = OrderedDict({
            key_col: loc[key_col] for key_col in key_cols
        })

        _lookup = lambda loc_id, vlnst, vlnid, vlnmsg: {
            k:v for k, v in itertools.chain(
                (
                    (loc_id_col, loc_id),
                    ('status', vlnst),
                    ('vulnerability_id', vlnid),
                    ('message', vlnmsg)
                ),
                six.iteritems(loc_key_col_values)
            )
        }

        try:
            for key_col in key_cols:
                key_col_dtype = col_dtypes[key_col]
                key_col_dtype(loc_key_col_values[key_col])
        except (TypeError, ValueError):
            return _lookup(loc_id, KEYS_STATUS_FAIL, None, 'Vulnerability lookup: invalid key column value(s) for location')

        vlnst = KEYS_STATUS_NOMATCH
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
            vlnst = KEYS_STATUS_SUCCESS
            vlnmsg = 'Successful vulnerability lookup: {}'.format(vlnid)

        return _lookup(loc_id, vlnst, vlnid, vlnmsg)
