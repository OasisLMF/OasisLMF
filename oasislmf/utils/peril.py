__all__ = [
    'DEFAULT_RTREE_INDEX_PROPS',
    'generate_index_entries',
    'get_peril_areas',
    'get_peril_areas_index',
    'get_rtree_index',
    'PerilArea',
    'PerilAreasIndex',
    'PERILS',
    'PERIL_GROUPS'
]

import copy
import os
import re
import types
import uuid

from collections import OrderedDict

from rtree.core import RTreeError
from rtree.index import (
    Index as RTreeIndex,
    Property as RTreeIndexProperty,
)

from shapely.geometry import (
    box,
    MultiPoint,
    Polygon,
)

from shapely import speedups as shapely_speedups

import pickle

from .exceptions import OasisException
from .data import get_dataframe

if shapely_speedups.available:
    shapely_speedups.enable()


PRL_BBF = 'BBF'
PRL_BFR = 'BFR'
PRL_BSK = 'BSK'
PRL_MNT = 'MNT'
PRL_MTR = 'MTR'
PRL_ORF = 'ORF'
PRL_OSF = 'OSF'
PRL_QEQ = 'QEQ'
PRL_QFF = 'QFF'
PRL_QLF = 'QLF'
PRL_QLS = 'QLS'
PRL_QSL = 'QSL'
PRL_QTS = 'QTS'
PRL_WEC = 'WEC'
PRL_WSS = 'WSS'
PRL_WTC = 'WTC'
PRL_XHL = 'XHL'
PRL_XLT = 'XLT'
PRL_XSL = 'XSL'
PRL_XTD = 'XTD'
PRL_ZFZ = 'ZFZ'
PRL_ZIC = 'ZIC'
PRL_ZSN = 'ZSN'
PRL_ZST = 'ZST'

PRL_GRP_AA1 = 'AA1'
PRL_GRP_BB1 = 'BB1'
PRL_GRP_MM1 = 'MM1'
PRL_GRP_OO1 = 'OO1'
PRL_GRP_QQ1 = 'QQ1'
PRL_GRP_WW1 = 'WW1'
PRL_GRP_WW2 = 'WW2'
PRL_GRP_XX1 = 'XX1'
PRL_GRP_XZ1 = 'XZ1'
PRL_GRP_ZZ1 = 'ZZ1'

PERIL_GROUPS = OrderedDict({
    'all': {'id': PRL_GRP_AA1, 'desc': 'All perils', 'peril_ids': [PRL_QEQ, PRL_QFF, PRL_QTS, PRL_QSL, PRL_QLS, PRL_QLF, PRL_WTC, PRL_WEC, PRL_WSS, PRL_ORF, PRL_OSF, PRL_XSL, PRL_XTD, PRL_XHL, PRL_ZSN, PRL_ZIC, PRL_ZFZ, PRL_BFR, PRL_BBF, PRL_MNT, PRL_MTR, PRL_XLT, PRL_ZST, PRL_BSK]},
    'wildfire w/ smoke': {'id': PRL_GRP_BB1, 'desc': 'Wildfire with smoke', 'peril_ids': [PRL_BBF, PRL_BSK]},
    'terrorism': {'id': PRL_GRP_MM1, 'desc': 'Terrorism', 'peril_ids': [PRL_MNT, PRL_MTR]},
    'flood w/o storm surge': {'id': PRL_GRP_OO1, 'desc': 'Flood w/o storm surge', 'peril_ids': [PRL_ORF, PRL_OSF]},
    'earthquake': {'id': PRL_GRP_QQ1, 'desc': 'All EQ perils', 'peril_ids': [PRL_QEQ, PRL_QFF, PRL_QTS, PRL_QSL, PRL_QLS, PRL_QLF]},
    'windstorm w/ storm surge': {'id': PRL_GRP_WW1, 'desc': 'Windstorm with storm surge', 'peril_ids': [PRL_WTC, PRL_WEC, PRL_WSS]},
    'windstorm w/o storm surge': {'id': PRL_GRP_WW2, 'desc': 'Windstorm w/o storm surge', 'peril_ids': [PRL_WTC, PRL_WEC]},
    'convective storm': {'id': PRL_GRP_XX1, 'desc': 'Convective Storm', 'peril_ids': [PRL_XSL, PRL_XTD, PRL_XHL, PRL_XLT]},
    'convective storm rms': {'id': PRL_GRP_XZ1, 'desc': 'Convective storm (incl winter storm) - for RMS users', 'peril_ids': [PRL_XSL, PRL_XTD, PRL_XHL, PRL_ZSN, PRL_ZIC, PRL_ZFZ, PRL_XLT, PRL_ZST]},
    'winter storm': {'id': PRL_GRP_ZZ1, 'desc': 'Winter storm', 'peril_ids': [PRL_ZSN, PRL_ZIC, PRL_ZFZ, PRL_ZST]}
})

PERILS = OrderedDict({
    'wildfire': {'id': PRL_BBF, 'desc': 'Wildfire / Bushfire', 'group_peril': 'wildfire w/ smoke'},
    'noncat': {'id': PRL_BFR, 'desc': 'NonCat', 'group_peril': 'wildfire w/ smoke'},
    'smoke': {'id': PRL_BSK, 'desc': 'Smoke', 'group_peril': 'wildfire w/ smoke'},
    'nbcr terrorism': {'id': PRL_MNT, 'desc': 'NBCR Terrorism', 'group_peril': 'terrorism'},
    'terrorism': {'id': PRL_MTR, 'desc': 'Conventional Terrorism', 'group_peril': 'terrorism'},
    'river flood': {'id': PRL_ORF, 'desc': 'River / Fluvial Flood', 'group_peril': 'flood w/o storm surge'},
    'flash flood': {'id': PRL_OSF, 'desc': 'Flash / Surface / Pluvial Flood', 'group_peril': 'flood w/o storm surge'},
    'earthquake': {'id': PRL_QEQ, 'desc': 'Earthquake - Shake only', 'group_peril': 'earthquake'},
    'fire following': {'id': PRL_QFF, 'desc': 'Fire Following', 'group_peril': 'earthquake'},
    'liquefaction': {'id': PRL_QLF, 'desc': 'Liquefaction', 'group_peril': 'earthquake'},
    'landslide': {'id': PRL_QLS, 'desc': 'Landslide', 'group_peril': 'earthquake'},
    'sprinkler leakage': {'id': PRL_QSL, 'desc': 'Sprinkler Leakage', 'group_peril': 'earthquake'},
    'tsunami': {'id': PRL_QTS, 'desc': 'Tsunami', 'group_peril': 'earthquake'},
    'extra tropical cyclone': {'id': PRL_WEC, 'desc': 'Extra Tropical Cyclone', 'group_peril': 'windstorm w/o storm surge'},
    'storm surge': {'id': PRL_WSS, 'desc': 'Storm Surge', 'group_peril': 'windstorm with storm surge'},
    'tropical cyclone': {'id': PRL_WTC, 'desc': 'Tropical Cyclone', 'group_peril': 'windstorm w/o storm surge'},
    'hail': {'id': PRL_XHL, 'desc': 'Hail', 'group_peril': 'convective storm rms'},
    'lightning': {'id': PRL_XLT, 'desc': 'Lightning', 'group_peril': 'convective storm'},
    'convective wind': {'id': PRL_XSL, 'desc': 'Straight-line / other convective wind', 'group_peril': 'convective storm'},
    'tornado': {'id': PRL_XTD, 'desc': 'Tornado', 'group_peril': 'convective storm'},
    'freeze': {'id': PRL_ZFZ, 'desc': 'Freeze', 'group_peril': 'winter storm'},
    'ice': {'id': PRL_ZIC, 'desc': 'Ice', 'group_peril': 'winter storm'},
    'snow': {'id': PRL_ZSN, 'desc': 'Snow', 'eqv_oasis_peril': None, 'group_peril': 'winter storm'},
    'winterstorm wind': {'id': PRL_ZST, 'desc': 'Winterstorm Wind', 'group_peril': 'winter storm'}
})


DEFAULT_RTREE_INDEX_PROPS = {
    'buffering_capacity': 10,
    'custom_storage_callbacks': None,
    'custom_storage_callbacks_size': 0,
    'dat_extension': 'dat',
    'dimension': 2,
    'filename': '',
    'fill_factor': 0.7,
    'idx_extension': 'idx',
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
