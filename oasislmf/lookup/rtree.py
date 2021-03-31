__all__ = [
    'DeterministicLookup',
    'RTreeLookup',
    'RTreePerilLookup',
    'RTreeVulnerabilityLookup',
    'generate_index_entries',
    'get_peril_areas',
    'get_peril_areas_index',
    'get_rtree_index',
    'PerilArea',
    'PerilAreasIndex',
]

# 'OasisLookup' -> 'RTreeLookup'
# 'OasisPerilLookup' -> RTreePerilLookup
# 'OasisVulnerabilityLookup' -> 'RTreeVulnerabilityLookup'

from .base import OasisBaseLookup, AbstractBasicKeyLookup

import copy
import re
import types
import builtins
import itertools
import os
import uuid
import pickle
import pandas as pd

from collections import OrderedDict

from rtree.core import RTreeError
from rtree.index import (
    Index as RTreeIndex,
    Property as RTreeIndexProperty,
)

from shapely import speedups as shapely_speedups
from shapely.geometry import (
    box,
    Point,
    MultiPoint,
    Polygon,
)

from ..utils.data import get_dataframe
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.defaults import DEFAULT_RTREE_INDEX_PROPS
from ..utils.status import OASIS_KEYS_STATUS
from ..utils.path import as_path

if shapely_speedups.available:
    shapely_speedups.enable()


class DeterministicLookup(AbstractBasicKeyLookup):
    multiproc_enabled = False

    def process_locations(self, locations):
        loc_ids = (loc_it['loc_id'] for _, loc_it in locations.loc[:, ['loc_id']].sort_values('loc_id').iterrows())
        success_status= OASIS_KEYS_STATUS['success']['id']
        return pd.DataFrame.from_records((
            {'loc_id': _loc_id, 'peril_id': peril, 'coverage_type': cov_type, 'area_peril_id': i + 1,
             'vulnerability_id': i + 1, 'status': success_status}
            for i, (_loc_id, peril, cov_type) in enumerate(itertools.product(loc_ids, range(1, 1 + self.config['num_subperils']),
                                                                             self.config['supported_oed_coverage_types']))
        ))


# ---- RTree Lookup classes ---------------------------------------------------


class RTreeLookup(OasisBaseLookup):
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
        vulnerabilities=None,
        user_data_dir=None,
        output_dir=None
    ):
        super(self.__class__, self).__init__(
            config=config,
            config_json=config_json,
            config_fp=config_fp,
            config_dir=config_dir,
            user_data_dir=user_data_dir,
            output_dir=output_dir
        )

        self.peril_lookup = RTreePerilLookup(
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

        self.vulnerability_lookup = RTreeVulnerabilityLookup(
            config=self.config,
            config_dir=self.config_dir,
            vulnerabilities=vulnerabilities
        )

    def __new__(cls,
                config=None,
                config_json=None,
                config_fp=None,
                config_dir=None,
                areas=None,
                peril_areas=None,
                peril_areas_index=None,
                peril_areas_index_props=None,
                loc_to_global_areas_boundary_min_distance=0,
                vulnerabilities=None,
                user_data_dir=None,
                output_dir=None):
        if config:
            builtin_lookup_type = config.get('builtin_lookup_type', 'combined')
        else:
            builtin_lookup_type = 'combined'

        if builtin_lookup_type == 'combined':
            return super().__new__(cls)
        elif builtin_lookup_type == 'peril':
            return RTreePerilLookup(config=config,
                                    config_json=config_json,
                                    config_fp=config_fp,
                                    config_dir=config_dir,
                                    areas=areas,
                                    peril_areas=peril_areas,
                                    peril_areas_index=peril_areas_index,
                                    peril_areas_index_props=peril_areas_index_props,
                                    loc_to_global_areas_boundary_min_distance=loc_to_global_areas_boundary_min_distance,
                                    user_data_dir=user_data_dir,
                                    output_dir=output_dir)
        elif builtin_lookup_type == 'vulnerability':
            return RTreeVulnerabilityLookup(config=config,
                                            config_json=config_json,
                                            config_fp=config_fp,
                                            config_dir=config_dir,
                                            vulnerabilities=vulnerabilities,
                                            user_data_dir=user_data_dir,
                                            output_dir=output_dir)
        else:
            raise OasisException(f'Unknown builtin_lookup_type: {builtin_lookup_type}')

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


class RTreePerilLookup(OasisBaseLookup):
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
        peril_areas_index_props=None,
        user_data_dir=None,
        output_dir=None
    ):
        super(self.__class__, self).__init__(
            config=config,
            config_json=config_json,
            config_fp=config_fp,
            config_dir=config_dir,
            user_data_dir=user_data_dir,
            output_dir=output_dir
        )

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


class RTreeVulnerabilityLookup(OasisBaseLookup):
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
        vulnerabilities=None,
        user_data_dir=None,
        output_dir=None
    ):
        super(self.__class__, self).__init__(
            config=config,
            config_json=config_json,
            config_fp=config_fp,
            config_dir=config_dir,
            user_data_dir=user_data_dir,
            output_dir=output_dir
        )

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



# ---- RTree Peril funcs ------------------------------------------------------


def generate_index_entries(items, objects=None):
    if objects:
        for (key, poly_bounds), obj in zip(items, objects):
            yield key, poly_bounds, obj
    else:
        for key, poly_bounds in items:
            yield key, poly_bounds, None


def get_peril_areas(areas):
    for peril_id, coverage_type, peril_area_id, coordinates, other_props in areas:
        yield PerilArea(coordinates, peril_id=peril_id, coverage_type=coverage_type, peril_area_id=peril_area_id, **other_props)


def get_peril_areas_index(
    areas=None,
    peril_areas=None,
    properties=None
):
    if not (areas or peril_areas):
        raise OasisException('Either areas or peril areas must be provided')

    return PerilAreasIndex(areas=areas, peril_areas=peril_areas, properties=properties)


def get_rtree_index(
    items,
    objects=None,
    properties=None
):
    return (
        RTreeIndex(generate_index_entries(items, objects=objects), properties=RTreeIndexProperty(**properties)) if properties
        else RTreeIndex(generate_index_entries(items, objects=objects))
    )




class PerilArea(Polygon):

    def __init__(self, coords, **kwargs):

        _coords = tuple(c for c in coords)

        if not _coords:
            raise OasisException('No peril area coordinates')

        if len(_coords) > 2:
            self._multipoint = MultiPoint(_coords)
        elif len(_coords) == 2:
            minx, miny, maxx, maxy = tuple(_c for c in _coords for _c in c)
            self._multipoint = MultiPoint(box(minx, miny, maxx, maxy).exterior.coords)
        elif len(_coords) == 1:
            x, y = _coords[0][0], _coords[0][1]
            r = kwargs.get('area_reg_poly_radius') or 0.0016
            self._multipoint = MultiPoint(
                tuple((x + r * (-1)**i, y + r * (-1)**j) for i in range(2) for j in range(2))
            )

        super(self.__class__, self).__init__(shell=self._multipoint.convex_hull.exterior.coords)

        self._coordinates = tuple(self.exterior.coords)

        self._centre = self.centroid.x, self.centroid.y

        self._coverage_type = kwargs.get('coverage_type')

        self._peril_id = kwargs.get('peril_id')

        self._id = kwargs.get('area_peril_id') or kwargs.get('peril_area_id') or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

    @property
    def multipoint(self):
        return self._multipoint

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def centre(self):
        return self._centre

    @property
    def coverage_type(self):
        return self._coverage_type

    @property
    def peril_id(self):
        return self._peril_id

    @property
    def id(self):
        return self._id


class PerilAreasIndex(RTreeIndex):

    def __init__(self, *args, **kwargs):

        self._protocol = pickle.HIGHEST_PROTOCOL

        idx_fp = kwargs.get('fp')

        areas = kwargs.get('areas')
        peril_areas = kwargs.get('peril_areas')

        props = kwargs.get('properties') or copy.deepcopy(DEFAULT_RTREE_INDEX_PROPS)

        if not (idx_fp or areas or peril_areas):
            self._peril_areas = self._stream = None
            kwargs['properties'] = RTreeIndexProperty(**props)
            super(self.__class__, self).__init__(*args, **kwargs)
        elif idx_fp:
            self._peril_areas = self._stream = None
            _idx_fp = idx_fp
            if not os.path.isabs(_idx_fp):
                _idx_fp = os.path.abspath(_idx_fp)

            idx_ext = props.get('idx_extension') or 'idx'
            dat_ext = props.get('dat_extension') or 'dat'

            if not (os.path.exists('{}.{}'.format(_idx_fp, idx_ext)) or os.path.exists('{}.{}'.format(_idx_fp, dat_ext))):
                kwargs['properties'] = RTreeIndexProperty(**props)

            super(self.__class__, self).__init__(_idx_fp, *args, **kwargs)
        else:
            self._peril_areas = OrderedDict({
                pa.id: pa for pa in (peril_areas if peril_areas else self._get_peril_areas(areas))
            })
            self._stream = self._generate_index_entries(
                ((paid, pa.bounds) for paid, pa in self._peril_areas.items()),
                objects=((paid, pa.bounds, pa.coordinates) for paid, pa in self._peril_areas.items())
            )
            kwargs['properties'] = RTreeIndexProperty(**props)
            super(self.__class__, self).__init__(self._stream, *args, **kwargs)

    def dumps(self, obj):
        return pickle.dumps(obj, protocol=self.protocol)

    def loads(self, data):
        return pickle.loads(data)

    def _get_peril_areas(self, areas):
        for peril_id, coverage_type, peril_area_id, coordinates, other_props in areas:
            yield PerilArea(coordinates, peril_id=peril_id, coverage_type=coverage_type, peril_area_id=peril_area_id, **other_props)

    def _generate_index_entries(self, items, objects=None):
        if objects:
            for (key, poly_bounds), obj in zip(items, objects):
                yield key, poly_bounds, obj
        else:
            for key, poly_bounds in items:
                yield key, poly_bounds, None

    @property
    def protocol(self):
        return self._protocol

    @property
    def peril_areas(self):
        return self._peril_areas

    @property
    def stream(self):
        if self._peril_areas:
            self._stream = self._generate_index_entries(self._peril_areas)
            return self._stream
        return None

    @classmethod
    def create_from_peril_areas_file(
        cls,
        src_fp=None,
        src_type='csv',
        peril_id_col='peril_id',
        coverage_type_col='coverage_type',
        peril_area_id_col='area_peril_id',
        non_na_cols=('peril_id', 'coverage_type', 'area_peril_id',),
        col_dtypes={'peril_id': int, 'coverage_type': int, 'area_peril_id': int},
        sort_cols=['area_peril_id'],
        area_poly_coords_cols={},
        area_poly_coords_seq_start_idx=1,
        area_reg_poly_radius=0.00166,
        static_props={},
        index_fp=None,
        index_props=copy.deepcopy(DEFAULT_RTREE_INDEX_PROPS)
    ):
        if not src_fp:
            raise OasisException(
                'An areas source CSV or JSON file path must be provided'
            )

        _src_fp = src_fp
        if not os.path.isabs(_src_fp):
            _src_fp = os.path.abspath(_src_fp)

        _non_na_cols = set(non_na_cols)

        _peril_id_col = peril_id_col.lower()
        _coverage_type_col = coverage_type_col.lower()
        _peril_area_id_col = peril_area_id_col.lower()

        if not set(_non_na_cols).intersection([_peril_id_col, _coverage_type_col, _peril_area_id_col]):
            _non_na_cols = _non_na_cols.union({_peril_id_col, _coverage_type_col, _peril_area_id_col})

        for col in area_poly_coords_cols.values():
            if col not in _non_na_cols:
                _non_na_cols = _non_na_cols.union({col.lower()})

        _non_na_cols = tuple(_non_na_cols)

        _sort_cols = [col.lower() for col in sort_cols]

        areas_df = get_dataframe(
            src_fp=_src_fp,
            src_type=src_type,
            non_na_cols=_non_na_cols,
            col_dtypes=col_dtypes,
            sort_cols=(_sort_cols or [_peril_area_id_col])
        )

        coords_cols = area_poly_coords_cols

        seq_start = area_poly_coords_seq_start_idx

        len_seq = sum(1 if re.match(r'x(\d+)?', k) else 0 for k in coords_cols.keys())

        peril_areas = cls()._get_peril_areas(
            (
                ar[_peril_id_col],
                ar[_coverage_type_col],
                ar[_peril_area_id_col],
                tuple(
                    (ar.get(coords_cols['x{}'.format(i)].lower()) or 0, ar.get(coords_cols['y{}'.format(i)].lower()) or 0)
                    for i in range(seq_start, len_seq + 1)
                ),
                static_props
            ) for _, ar in areas_df.iterrows()
        )

        _index_fp = index_fp
        if not _index_fp:
            raise OasisException('No output file index path provided')

        if not os.path.isabs(_index_fp):
            _index_fp = os.path.abspath(_index_fp)

        try:
            return cls().save(
                _index_fp,
                peril_areas=peril_areas,
                index_props=index_props
            )
        except OasisException:
            raise

    def save(
        self,
        index_fp,
        peril_areas=None,
        index_props=DEFAULT_RTREE_INDEX_PROPS
    ):
        _index_fp = index_fp

        if not os.path.isabs(_index_fp):
            _index_fp = os.path.abspath(_index_fp)

        if os.path.exists(_index_fp):
            os.remove(_index_fp)

        class myindex(RTreeIndex):
            def __init__(self, *args, **kwargs):
                self.protocol = pickle.HIGHEST_PROTOCOL
                super(self.__class__, self).__init__(*args, **kwargs)

            def dumps(self, obj):
                return pickle.dumps(obj, protocol=self.protocol)

            def loads(self, obj):
                return pickle.loads(obj)

        try:
            index = myindex(_index_fp, properties=RTreeIndexProperty(**index_props))

            _peril_areas = self._peril_areas or peril_areas

            if not _peril_areas:
                raise OasisException(
                    'No peril areas found in instance or in arguments - '
                    'this is required to write the index to file'
                )

            peril_areas_seq = None

            if (isinstance(peril_areas, list) or isinstance(peril_areas, tuple)):
                peril_areas_seq = (pa for pa in peril_areas)
            elif isinstance(peril_areas, types.GeneratorType):
                peril_areas_seq = peril_areas
            elif (isinstance(peril_areas, dict)):
                peril_areas_seq = peril_areas.values()

            for pa in peril_areas_seq:
                index.insert(pa.id, pa.bounds, obj=(pa.peril_id, pa.coverage_type, pa.id, pa.bounds, pa.coordinates))

            index.close()
        except (IOError, OSError, RTreeError):
            raise

        return _index_fp
