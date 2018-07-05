# -*- coding: utf-8 -*-

__all__ = [
    'DEFAULT_RTREE_INDEX_PROPS',
    'generate_index_entries',
    'get_peril_areas',
    'get_peril_areas_index',
    'get_rtree_index',
    'PerilArea',
    'PerilAreasIndex',
    'PERIL_ID_FLOOD',
    'PERIL_ID_QUAKE',
    'PERIL_ID_SURGE',
    'PERIL_ID_WIND',
    'PerilPoint'
]

import copy
import io
import json
import os
import re
import uuid

from collections import OrderedDict

import rtree

from rtree.index import (
    Index as RTreeIndex,
    Property as RTreeIndexProperty,
)

from rtree.core import RTreeError

from shapely import speedups as shapely_speedups

if shapely_speedups.available:
    shapely_speedups.enable()

from shapely.geometry import (
    box,
    Point,
    MultiPoint,
    Polygon,
)

import six

from six.moves import cPickle as cpickle

from .exceptions import OasisException
from .data import get_dataframe


PERIL_ID_WIND = 1
PERIL_ID_SURGE = 2
PERIL_ID_QUAKE = 3
PERIL_ID_FLOOD = 4

DEFAULT_RTREE_INDEX_PROPS = {
    'buffering_capacity': 10,
    'custom_storage_callbacks': None,
    'custom_storage_callbacks_size': 0,
    'dat_extension': u'dat',
    'dimension': 2,
    'filename': u'',
    'fill_factor': 0.7,
    'idx_extension': u'idx',
    'index_capacity': 100,
    'index_id': None,
    'leaf_capacity': 100,
    'near_minimum_overlap_factor': 32,
    'overwrite': True,
    'pagesize': 4096,
    'point_pool_capacity': 500,
    'region_pool_capacity': 1000,
    'reinsert_factor': 0.3,
    'split_distribution_factor': 0.4,
    'storage': 0,
    'tight_mbr': True,
    'tpr_horizon': 20.0,
    'type': 0,
    'variant': 2,
    'writethrough': False
}


def generate_index_entries(items, objects=None):
    if objects:
        for (key, poly_bounds), obj in zip(items, objects):
            yield key, poly_bounds, obj
    else:
        for key, poly_bounds in items:
            yield key, poly_bounds, None


def get_peril_areas(areas):
    for peril_area_id, coordinates, other_props in areas:
        yield PerilArea(coordinates, peril_area_id=peril_area_id, **other_props)


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

        self._args = coords, kwargs

        _coords = tuple(c for c in coords)

        if not _coords:
            raise OasisException('No peril area coordinates')

        if len(_coords) > 2:
            self.multipoint = MultiPoint(_coords)
        elif len(_coords) == 2:
            minx, miny, maxx, maxy = tuple(_c for c in coords for _c in c)
            self.multipoint = MultiPoint(box(minx, miny, maxx, maxy).exterior.coords)
        elif len(_coords) == 1:
            x, y = _coords[0][0], _coords[0][1]
            r = kwargs.get('area_reg_poly_radius') or 0.0016
            self.multipoint = MultiPoint(
                tuple((x + r*(-1)**i, y + r*(-1)**j) for i in range(2) for j in range(2))
            )

        super(self.__class__, self).__init__(shell=self.multipoint.convex_hull.exterior.coords)
        
        self.coordinates = tuple(self.exterior.coords)

        self.centre = self.centroid.x, self.centroid.y

        self.peril_id = kwargs.get('peril_id')

        self.id = kwargs.get('area_peril_id') or kwargs.get('peril_area_id') or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)

    def __getinitargs__(self):
        return self._args[0]

    def __getnewargs__(self):
        return self._args[0]

    def __getnewargs_ex__(self):
        return self._args


class PerilPoint(Point):

    def __init__(self, *args, **kwargs):

        super(self.__class__, self).__init__(*args)
        
        self.coordinates = tuple(t for t in self.coords[0])

        self.peril_id = kwargs.get('peril_id')

        self.id = kwargs.get('area_peril_id') or kwargs.get('peril_area_id') or int(uuid.UUID(bytes=os.urandom(16)).hex[:16], 16)

        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


class PerilAreasIndex(RTreeIndex):

    def __init__(self, *args, **kwargs):
            idx_fp = kwargs.get('fp')

            areas = kwargs.get('areas')
            peril_areas = kwargs.get('peril_areas')
            
            props = RTreeIndexProperty(**(kwargs.get('properties') or {}))
            kwargs['properties'] = props

            if not (idx_fp or areas or peril_areas):
                self._peril_areas = self._stream = None
                super(self.__class__, self).__init__(*args, **kwargs)
            elif idx_fp:
                self._peril_areas = self._stream = None
                _idx_fp = idx_fp
                if not os.path.isabs(_idx_fp):
                    _idx_fp = os.path.abspath(_idx_fp)
                super(self.__class__, self).__init__(_idx_fp, *args, **kwargs)
            else:
                self._peril_areas = OrderedDict({
                    pa.id:pa for pa in (peril_areas if peril_areas else self._get_peril_areas(areas))
                })
                self._stream = self._generate_index_entries(
                    ((paid, pa.bounds) for paid, pa in six.iteritems(self._peril_areas)),
                    objects=((paid, pa.bounds) for paid, pa in six.iteritems(self._peril_areas))
                )
                super(self.__class__, self).__init__(self._stream, *args, **kwargs)

    def dumps(self, obj):
        return cpickle.dumps(obj, protocol=(2 if six.sys.version_info[0] < 3 else -1))

    def loads(self, data):
        return cpickle.loads(data)

    def _get_peril_areas(self, areas):
        for peril_area_id, coordinates, other_props in areas:
            yield PerilArea(coordinates, peril_area_id=peril_area_id, **other_props)

    def _generate_index_entries(self, items, objects=None):
        if objects:
            for (key, poly_bounds), obj in zip(items, objects):
                yield key, poly_bounds, obj
        else:
            for key, poly_bounds in items:
                yield key, poly_bounds, None

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
        peril_area_id_col='area_peril_id',
        non_na_cols=('area_peril_id',),
        col_dtypes={'area_peril_id': int},
        sort_col='area_peril_id',
        area_poly_coords_cols={},
        area_poly_coords_seq_start_idx=1,
        area_reg_poly_radius=0.00166,
        static_props={},
        index_fp=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rtree-index')
    ):
        _index_fp = index_fp
        if not os.path.isabs(_index_fp):
            _index_fp = os.path.abspath(_index_fp)

        if not src_fp:
            raise OasisException(
                'An areas source CSV or JSON file path must be provided'
            )

        _src_fp = src_fp
        if not os.path.isabs(_src_fp):
            _src_fp = os.path.abspath(_src_fp)

        _non_na_cols = set(non_na_cols)

        _peril_area_id_col = peril_area_id_col.lower()

        if not _peril_area_id_col in _non_na_cols:
            _non_na_cols = _non_na_cols.union({_peril_area_id_col})
        for col in six.itervalues(area_poly_coords_cols):
            if not col in _non_na_cols:
                _non_na_cols = _non_na_cols.union({col})

        _non_na_cols = tuple(_non_na_cols)

        areas_df = get_dataframe(
            src_fp=_src_fp,
            src_type=src_type,
            non_na_cols=_non_na_cols,
            col_dtypes=col_dtypes,
            sort_col=(peril_area_id_col or sort_col)
        )

        coords_cols = area_poly_coords_cols

        seq_start = area_poly_coords_seq_start_idx

        len_seq = sum(1 if re.match(r'x(\d+)?', k) else 0 for k in six.iterkeys(coords_cols))

        peril_areas = get_peril_areas(
            (
                ar[_peril_area_id_col],
                tuple(
                    (ar.get(coords_cols['x{}'.format(i)].lower()) or 0, ar.get(coords_cols['y{}'.format(i)].lower()) or 0)
                    for i in range(seq_start, len_seq + 1)
                ),
                static_props
            ) for _, ar in areas_df.iterrows()
        )

        try:
            return cls().save(
                peril_areas=peril_areas,
                index_fp=index_fp
            )
        except OasisException as e:
            raise

    def save(
        self,
        peril_areas=None,
        index_fp=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rtree-index')
    ):
        _index_fp = index_fp
        if not os.path.isabs(_index_fp):
            _index_fp = os.path.abspath(_index_fp)

        try:
            class _myindex(RTreeIndex):
                def dumps(self, obj):
                    return cpickle.dumps(obj, protocol=(2 if six.sys.version_info[0] < 3 else -1))

            index = _myindex(_index_fp)

            _peril_areas = self._peril_areas or peril_areas

            if not _peril_areas:
                raise OasisException(
                    'No peril areas found in instance or in arguments - '
                    'this is required to write the index to file'
                )

            peril_areas_seq = enumerate(_peril_areas) if not isinstance(_peril_areas, dict) else enumerate(six.itervalues(_peril_areas))
            for _, pa in peril_areas_seq:
                index.insert(pa.id, pa.bounds, obj=(pa.id, pa.bounds))

            index.close()
        except (IOError, OSError, RTreeError) as e:
            raise

        return _index_fp
