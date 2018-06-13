# -*- coding: utf-8 -*-

from __future__ import unicode_literals, absolute_import

__all__ = [
    'OasisBaseLookup',
    'OasisBaseKeysLookup',
    'OasisPerilAndVulnerabilityLookup',
    'OasisPerilLookup',
    'OasisVulnerabilityLookup',
    'OasisKeysLookupFactory'
]

import csv
import imp
import importlib
import io
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

from six import StringIO

from ..utils.data import get_dataframe
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.peril import (
    DEFAULT_RTREE_INDEX_PROPS,
    get_peril_areas_index,
    PerilArea,
)
from ..utils.status import (
    KEYS_STATUS_FAIL,
    KEYS_STATUS_NOMATCH,
    KEYS_STATUS_SUCCESS,
)


UNKNOWN_ID = -1

class OasisBaseLookup(object):

    def __init__(self, model_config=None, model_config_json=None, model_config_fp=None):
        self.model_config = model_config or self.load_model_config(model_config_json=model_config_json, model_config_fp=model_config_fp)
        self.supplier_id = self.model_config['model']['supplier_id']
        self.model_id = self.model_config['model']['model_id']
        self.model_version = self.model_config['model']['model_version']

    def load_model_config(self, model_config_json=None, model_config_fp=None):
        if model_config_json:
            return json.loads(model_config_json)

        if model_config_fp:
            _model_config_fp = os.path.abspath(model_config_fp) if not os.path.isabs(model_config_fp) else model_config_fp
            with io.open(_model_config_fp, 'r', encoding='utf-8') as f:
                return json.load(f)

    def lookup(self, loc, **kwargs):
        """
        Lookup for an individual location item, which could be a dict or a
        Pandas series object.
        """
        pass

    def bulk_lookup(self, locs, **kwargs):
        """
        Bulk vulnerability lookup for a generator, list, tuple or dict of
        location items, which can be dicts or Pandas series objects.

        Generates results using ``yield``.
        """
        _locs = (
            enumerate(locs) if isinstance(locs, tuple) or isinstance(locs, list) or isinstance(locs, types.GeneratorType)
            else six.iteritems(locs)
        )
        for _, loc in _locs:
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


class OasisKeysLookupFactory(object):
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
                f, fieldnames=['supplier_id', 'model_id', 'model_version_id']
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
            model_version=model_info['model_version_id']
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
            loc_df = pd.read_csv(StringIO(model_exposures), float_precision='high')
        else:
            raise OasisException('Either model_exposures_file_path or model_exposures must be specified')

        loc_df = loc_df.where(loc_df.notnull(), None)
        loc_df.columns = loc_df.columns.str.lower()

        return loc_df

    @classmethod
    def write_oasis_keys_file(cls, records, output_file_path):
        """
        Writes an Oasis keys file from an iterable of keys records.
        """
        heading_row = OrderedDict([
            ('id', 'LocID'),
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
    def write_oasis_keys_errors_file(cls, records, output_file_path):
        """
        Writes an Oasis keys errors file from an iterable of keys records.
        """
        heading_row = OrderedDict([
            ('id', 'LocID'),
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
    ):
        """
        Creates a keys lookup class instance for the given model and supplier -
        local file paths are required for the model keys data folder, the model
        version file and the Git repository for the model keys server. Returns a
        pair ``(model_info, klc)``, where ``model_info`` is a dictionary holding
        model information from the model version file and `klc` is the lookup
        service class instance for the model.
        """
        for p in [model_keys_data_path, model_version_file_path, lookup_package_path]:
            p = os.path.abspath(p) if p and not os.path.isabs(p) else p

        model_info = cls.get_model_info(model_version_file_path)
        lookup_package = cls.get_lookup_package(lookup_package_path)
        klc = cls.get_lookup_class_instance(lookup_package, model_keys_data_path, model_info)
        return model_info, klc

    @classmethod
    def get_keys(
        cls,
        lookup=None,
        model_exposures=None,
        model_exposures_file_path=None,
        success_only=True
    ):
        """
        Generates keys keys records (JSON) for the given model and supplier -
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
    def save_keys(
        cls,
        lookup=None,
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

        keys_file_path, keys_errors_file_path, model_exposures_file_path = map(
            lambda p: os.path.abspath(p) if p and not os.path.isabs(p) else p,
            [keys_file_path, keys_errors_file_path, model_exposures_file_path]
        )

        keys = cls.get_keys(
            lookup=lookup,
            model_exposures=model_exposures,
            model_exposures_file_path=model_exposures_file_path,
            success_only=(True if not keys_errors_file_path else False)
        )

        successes = []
        nonsuccesses = []
        for k in keys:
            successes.append(k) if k['status'] == KEYS_STATUS_SUCCESS else nonsuccesses.append(k)

        if keys_format == 'json':
            if keys_error_file_path:
                fp1, n1 = cls.write_json_keys_file(successes, keys_file_path)
                fp2, n2 = cls.write_json_keys_file(nonsuccesses, keys_errors_file_path)
                return fp1, n1, fp2, n2
            return cls.write_json_keys_file(successes, keys_file_path)
        elif keys_format == 'oasis':
            if keys_errors_file_path:
                fp1, n1 = cls.write_oasis_keys_file(successes, keys_file_path)
                fp2, n2 = cls.write_oasis_keys_errors_file(nonsuccesses, keys_errors_file_path)
                return fp1, n1, fp2, n2
            return cls.write_oasis_keys_file(successes, keys_file_path)
        else:
            raise OasisException("Unrecognised keys file output format - valid formats are 'oasis' or 'json'")




    def __init__(self, model_config=None, model_config_json=None, model_config_fp=None):
        self.model_config = model_config or self.load_model_config(model_config_json=model_config_json, model_config_fp=model_config_fp)
        self.supplier_id = self.model_config['model']['supplier_id']
        self.model_id = self.model_config['model']['model_id']
        self.model_version = self.model_config['model']['model_version']

    def load_model_config(self, model_config_json=None, model_config_fp=None):
        if model_config_json:
            return json.loads(model_config_json)

        if model_config_fp:
            _model_config_fp = os.path.abspath(model_config_fp) if not os.path.isabs(model_config_fp) else model_config_fp
            with io.open(_model_config_fp, 'r', encoding='utf-8') as f:
                return json.load(f)

    def lookup(self, loc, **kwargs):
        """
        Lookup for an individual location item, which could be a dict or a
        Pandas series object.
        """
        pass

    def bulk_lookup(self, locs, **kwargs):
        """
        Bulk vulnerability lookup for a generator, list, tuple or dict of
        location items, which can be dicts or Pandas series objects.

        Generates results using ``yield``.
        """
        _locs = (
            enumerate(locs) if isinstance(locs, tuple) or isinstance(locs, list) or isinstance(locs, types.GeneratorType)
            else six.iteritems(locs)
        )
        for _, loc in _locs:
            yield self.lookup(loc)


class OasisPerilAndVulnerabilityLookup(OasisBaseLookup):
    """
    Combined peril and vulnerability lookup
    """
    def __init__(
        self,
        model_config=None,
        model_config_json=None,
        model_config_fp=None,
        areas=None,
        peril_id=None,
        peril_areas=None,
        peril_areas_index=None,
        peril_areas_index_props=None,
        loc_to_global_areas_boundary_min_distance=0,
        vulnerabilities=None
    ):
        super(self.__class__, self).__init__(
            model_config=model_config,
            model_config_json=model_config_json,
            model_config_fp=model_config_fp
        )

        self.peril_lookup = OasisPerilLookup(
            model_config=self.model_config,
            areas=areas,
            peril_id=peril_id,
            peril_areas=peril_areas,
            peril_areas_index=peril_areas_index,
            peril_areas_index_props=None,
            loc_to_global_areas_boundary_min_distance=loc_to_global_areas_boundary_min_distance
        )

        self.peril_id = peril_id or self.model_config['peril']['peril_id'] or self.peril_lookup.peril_id

        self.peril_area_id_key = self.model_config['peril']['peril_area_id_col'].lower()

        self.vulnerability_id_key = self.model_config['vulnerability']['vulnerability_id_col'].lower()

        self.vulnerability_lookup = OasisVulnerabilityLookup(
            model_config=self.model_config,
            vulnerabilities=vulnerabilities
        )

    def lookup(self, loc, loc_id_key='id'):

        loc_id = loc.get(loc_id_key) or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        pa_lookup = self.peril_lookup.lookup(loc)
        past = pa_lookup['status']
        pamsg = pa_lookup['message']
        
        vln_lookup = self.vulnerability_lookup.lookup(loc)
        vlnst = vln_lookup['status']
        vlnmsg = vln_lookup['message']
        
        # Could optionally call the status lookup method, but it is always
        # better to avoid outside function calls in a `for` loop if possible
        status = (
            KEYS_STATUS_SUCCESS if past == vlnst == KEYS_STATUS_SUCCESS
            else (KEYS_STATUS_FAIL if (past == KEYS_STATUS_FAIL or vlnst == KEYS_STATUS_FAIL) else KEYS_STATUS_NOMATCH)
        )
        
        message = '{}; {}'.format(pamsg, vlnmsg)

        return {
            'id': loc_id,
            'peril_id': self.peril_id,
            self.peril_area_id_key: pa_lookup['peril_area_id'],
            self.vulnerability_id_key: vln_lookup['vulnerability_id'],
            'status': status,
            'message': message
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
    def __init__(
        self,
        model_config=None,
        model_config_json=None,
        model_config_fp=None,
        areas=None,
        peril_id=None,
        peril_areas=None,
        peril_areas_index=None,
        peril_areas_index_props=None,
        loc_to_global_areas_boundary_min_distance=0
    ):
        super(self.__class__, self).__init__(model_config=model_config, model_config_json=model_config_json, model_config_fp=model_config_fp)

        self.peril_id = peril_id or self.model_config['peril']['peril_id']

        self.areas = self.load_areas(areas=areas)

        self.peril_areas = self.load_peril_areas(areas=self.areas, peril_areas=peril_areas)

        self.peril_areas_dict = {pa.id: pa for _, pa in enumerate(self.peril_areas)}

        self.peril_areas_index, self.peril_areas_index_props = self.load_peril_areas_index(
            peril_areas=self.peril_areas,
            peril_areas_index=peril_areas_index,
            peril_areas_index_props=peril_areas_index_props
        )

        self.peril_areas_boundary = box(*self.peril_areas_index.bounds, ccw=False)

        _centroid = self.peril_areas_boundary.centroid
        self.peril_areas_centre = _centroid.x, _centroid.y

        self.loc_to_global_areas_boundary_min_distance = loc_to_global_areas_boundary_min_distance or self.model_config['peril']['loc_to_global_areas_boundary_min_distance']

    def load_areas(self, areas=None):

        if areas:
            return tuple(areas) if not isinstance(areas, tuple) else areas

        peril_config = self.model_config['peril']

        src_fp = peril_config['file_path']
        if not os.path.isabs(src_fp):
            src_fp = os.path.abspath(src_fp)

        src_type = peril_config['file_type'].lower() if peril_config.get('file_type') else "csv"

        float_precision = 'high' if peril_config.get('float_precision_high') else None

        non_na_cols = tuple(col.lower() for col in peril_config['non_na_cols']) if peril_config.get('non_na_cols') else ()

        peril_area_id_col = peril_config['peril_area_id_col'].lower()
        col_dtypes = {peril_area_id_col: int} if peril_config.get('col_dtypes') == "infer" else {}

        sort_col = peril_config.get('sort_col')
        sort_ascending = peril_config.get('sort_ascending')

        areas_df = get_dataframe(
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

        coords_cols = peril_config['area_lonlat_coords_cols']
        np = sum(1 if re.match(r'lon[1-9](\d+)?', k) else 0 for k in six.iterkeys(coords_cols))

        area_lonlat_point_inferred_radius = peril_config.get('area_lonlat_point_inferred_radius') or 0.1

        other_props_cols = {'peril_id': self.peril_id, 'area_lonlat_point_inferred_radius': area_lonlat_point_inferred_radius}

        return tuple(
            (
                ar[peril_area_id_col],
                tuple(
                    (ar.get(coords_cols['lon{}'.format(i)].lower()) or 0, ar.get(coords_cols['lat{}'.format(i)].lower()) or 0)
                    for i in range(1, np + 1)
                ),
                other_props_cols
            ) for _, ar in areas_df.iterrows()
        )

    def load_peril_areas(self, areas=None, peril_areas=None):

        if peril_areas:
            return tuple(peril_areas) if not isinstance(peril_areas, tuple) else peril_areas
        
        return tuple(
            PerilArea(coordinates, peril_area_id=peril_area_id, **other_props)
            for peril_area_id, coordinates, other_props in (areas or self.areas)
        )

    def load_peril_areas_index(self, peril_areas=None, peril_areas_index=None, peril_areas_index_props=None):

        if peril_areas_index:
            return peril_areas_index, peril_areas_index.properties.as_dict()

        _peril_areas_index_props = peril_areas_index_props or self.model_config['peril']['rtree_index'] or DEFAULT_RTREE_INDEX_PROPS

        _peril_areas_index_props['index_capacity'] = len(peril_areas)
        _peril_areas_index_props['index_pool_capacity'] = len(peril_areas)

        peril_areas = peril_areas or self.peril_areas

        _peril_areas_index = get_peril_areas_index(
            peril_areas=peril_areas,
            objects=None,
            properties=_peril_areas_index_props
        )

        return _peril_areas_index, _peril_areas_index_props

    def lookup(self, loc, loc_id_key='id'):
        """
        Area peril lookup for an individual lon/lat location item, which can be
        provided as a dict or a Pandas series. The data structure should contain
        the keys `lon` or `longitude` for longitude and `lat` or `latitude` for
        latitude.
        """
        peril_id = self.peril_id
        peril_areas = self.peril_areas_dict
        idx = self.peril_areas_index
        boundary = self.peril_areas_boundary
        loc_to_global_areas_boundary_min_distance = self.loc_to_global_areas_boundary_min_distance

        loc_id = loc.get(loc_id_key) or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        lon = loc.get('lon') or loc.get('longitude')
        lat = loc.get('lat') or loc.get('latitude')

        _pa_lookup = lambda loc_id, lon, lat, st, paid, pabnds, msg: {
            'id': loc_id,
            'lon': lon,
            'lat': lat,
            'peril_id': peril_id,
            'status': st,
            'peril_area_id': paid,
            'area_peril_id': paid,
            'area_bounds': pabnds,
            'message': msg
        }

        try:
            lon = float(lon)
            lat = float(lat)
            if not ((-180 <= lon <= 180) and (-90 <= lat <= 90)):
                raise ValueError('lon/lat out of bounds')
        except (ValueError, TypeError) as e:
            msg = (
                'Peril area lookup: invalid lon/lat ({}, {}) - {}'
                .format(lon, lat, str(e))
            )
            return _pa_lookup(loc_id, lon, lat, KEYS_STATUS_FAIL, None, None, msg)

        st = KEYS_STATUS_NOMATCH
        msg = 'No peril area match'
        paid = None
        pabnds = None
        point = lon, lat

        try:
            paid = list(idx.intersection(point))[0]
        except IndexError:
            try:
                paid = list(idx.nearest(point))[0]
            except IndexError:
                pass
            else:
                p = Point(lon, lat)
                min_dist = p.distance(boundary)
                if min_dist > loc_to_global_areas_boundary_min_distance:
                    msg = (
                        'Peril area lookup: location is {} units from the '
                        'peril areas global boundary -  the required minimum '
                        'distance is {} units'
                        .format(min_dist, loc_to_global_areas_boundary_min_distance)
                    )
                    return _pa_lookup(loc_id, lon, lat, KEYS_STATUS_FAIL, None, None, msg)
                pabnds = peril_areas[paid].bounds
                st = KEYS_STATUS_SUCCESS
                msg = (
                    'Successful peril area lookup: {}'.format(paid)
                )
        except RTreeError as e:
            return _pa_lookup(loc_id, lon, lat, KEYS_STATUS_FAIL, None, None, str(e))
        else:
            pabnds = peril_areas[paid].bounds
            st = KEYS_STATUS_SUCCESS
            msg = 'Successful peril area lookup: {}'.format(paid)

        return _pa_lookup(loc_id, lon, lat, st, paid, pabnds, msg)


class OasisVulnerabilityLookup(OasisBaseLookup):
    """
    Simple key-value based vulnerability lookup
    """
    def __init__(
        self,
        model_config=None,
        model_config_json=None,
        model_config_fp=None,
        vulnerabilities=None
    ):
        super(self.__class__, self).__init__(model_config=model_config, model_config_json=model_config_json, model_config_fp=model_config_fp)
        
        self.vulnerabilities = self.load_vulnerabilities(vulnerabilities=vulnerabilities)

    def load_vulnerabilities(self, vulnerabilities=None):
        vuln_config = self.model_config['vulnerability']

        cols = tuple(col.lower() for col in vuln_config['cols'])
        key_cols = tuple(col.lower() for col in vuln_config['key_cols'])
        vuln_id_col = self.model_config['vulnerability']['vulnerability_id_col']

        def _vuln_dict(vuln_enum, key_cols, vuln_id_col):
            return (
                {v[key_cols[0]]:(v.get(vuln_id_col) or v.get('vulnerability_id')) for _, v in vuln_enum} if len(key_cols) == 1
                else {tuple(v[key_cols[i]] for i in range(len(key_cols))):(v.get(vuln_id_col) or v.get('vulnerability_id')) for _, v in vuln_enum}
            )

        if vulnerabilities:
            return _vuln_dict(enumerate(vulnerabilities), key_cols)

        src_fp = vuln_config['file_path']
        if not os.path.isabs(src_fp):
            src_fp = os.path.abspath(src_fp)

        src_type = vuln_config['file_type'].lower() if vuln_config.get('file_type') else "csv"

        float_precision = 'high' if vuln_config.get('float_precision_high') else None

        non_na_cols = tuple(col.lower() for col in vuln_config['non_na_cols']) if vuln_config.get('non_na_cols') else ()

        col_dtypes = {vuln_id_col.lower(): int} if vuln_config.get('col_dtypes') == "infer" else {}

        sort_col = vuln_config.get('sort_col')
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

        return _vuln_dict(vuln_df.iterrows(), key_cols, vuln_id_col)

    def lookup(self, loc, loc_id_key='id'):
        """
        Vulnerability lookup for an individual location item, which could be a dict or a
        Pandas series.
        """
        loc_id = loc.get(loc_id_key) or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)
        coverage = loc['coverage']
        class_1 = loc['class_1']

        vln_lookup = lambda loc_id, vlnst, vlnid, vlnmsg: {
            'id': loc_id,
            'status': vlnst,
            'vulnerability_id': vlnid,
            'message': vlnmsg
        }

        try:
            int(coverage) and str(class_1)
        except (TypeError, ValueError):
            return vln_lookup(loc_id, KEYS_STATUS_FAIL, None, 'Vulnerability lookup: invalid location coverage or class 1')

        vlnst = KEYS_STATUS_NOMATCH
        vlnmsg = 'No vulnerability match'
        vlnid = None

        try:
            vlnid = self.vulnerabilities[(loc['coverage'], loc['class_1'])]
        except KeyError:
            pass
        else:
            vlnst = KEYS_STATUS_SUCCESS
            vlnmsg = 'Successful vulnerability lookup: {}'.format(vlnid)

        return vln_lookup(loc_id, vlnst, vlnid, vlnmsg)
