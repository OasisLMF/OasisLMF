"""Module for the built-in Lookup Class

in the future we may want to improve on the management of files used to generate the keys
tutorial for pandas and parquet https://towardsdatascience.com/a-gentle-introduction-to-apache-arrow-with-apache-spark-and-pandas-bb19ffe0ddae
"""
import warnings

import numba as nb
import numpy as np
import pandas as pd
from ods_tools.oed import fill_empty, is_empty

try:  # needed for rtree
    import shapely
    from shapely.geometry import Point

    # Hide numerous warnings similar to:
    # > ...lib64/python3.8/site-packages/geopandas/_compat.py:112: UserWarning: The Shapely GEOS
    # > version (3.8.0-CAPI-1.13.1 ) is incompatible with the GEOS version PyGEOS was compiled with
    # > (3.10.3-CAPI-1.16.1). Conversions between both will be slow.
    # We're not in a position to fix these without compiling shapely and pygeos from source.
    # We're also not aware of any performance issues caused by this.
    # Upgrading to Shapely 2 will likely address this issue.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module="geopandas._compat",
                                message="The Shapely GEOS version")
        import geopandas as gpd
    try:  # needed only for min distance
        from sklearn.neighbors import BallTree
    except ImportError:
        BallTree = None
except ImportError:
    Point = gpd = BallTree = None

try:  # needed for geotiff
    import rasterio
    from rasterio.windows import Window
except ImportError:
    rasterio = Window = None

try:  # needed for transform_coordinates
    import pyproj
except ImportError:
    pyproj = None

try:  # needed for h3 lookup
    import h3
    # the int flavour of the api returns cell indexes as python ints, which saves building
    # (and re-parsing) the hexadecimal string representation for every location
    from h3.api import basic_int as h3_int
except ImportError:
    h3 = h3_int = None

import copy
import gc
import json
import logging
import math
import os
import re
import time
from contextlib import contextmanager

import psutil

from oasislmf.lookup.base import AbstractBasicKeyLookup, MultiprocLookupMixin
from oasislmf.utils.deprecation import warn_deprecated
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.peril import get_peril_groups_df
from oasislmf.utils.status import OASIS_KEYS_STATUS, OASIS_UNKNOWN_ID

logger = logging.getLogger(__name__)

GEOG_LOOKUP_DEFAULT_SLOTS = 30  # GeogScheme{N}/GeogName{N} pairs a geog_lookup step scans by default
OPT_INSTALL_MESSAGE = 'install oasislmf with the spatial packages by running \'pip install "oasislmf[spatial]"\''


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points.

    The distances returned are Euclidean distances, not distances on a sphere or ellipsoid.
    """
    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
    """For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
            Distance returned is the Euclidean distance, not the true distance on a sphere or ellipsoid.
    """
    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.transpose(np.array((left_gdf.geometry.x * np.pi / 180, left_gdf.geometry.y * np.pi / 180)))
    right_radians = np.transpose(np.array((right_gdf.geometry.x * np.pi / 180, right_gdf.geometry.y * np.pi / 180)))

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)
    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.set_index(left_gdf.index)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius

    return closest_points


EARTH_RADIUS_M = 6371000

# Data of the 'rtree' and 'merge' steps loaded by Lookup.multiproc_shared_data before the lookup
# worker processes are forked, so that the workers share one copy instead of each loading the whole
# file. Keyed by _shared_data_key.
_SHARED_STEP_DATA = {}


class _RtreeGeometries:
    """The geometries of one peril (or of all perils when the file has no peril_id column) and
    the spatial indexes used to join locations to them.

    It is built once and only read afterwards, so forked worker processes share it without
    copying it.
    """

    def __init__(self, geometries, attributes, centers=None):
        # attributes: the file's non-geometry columns, positionally aligned with geometries
        self.attributes = attributes.reset_index(drop=True)
        self.tree = shapely.STRtree(geometries)
        if centers is None:
            self.ball_tree = None
        else:
            # (longitude, latitude) order, as nearest_neighbor has always used
            self.ball_tree = BallTree(
                np.radians(np.column_stack([shapely.get_x(centers), shapely.get_y(centers)])),
                leaf_size=15, metric='haversine')


def _memory_usage():
    """This process's resident memory, for the logs (pages shared with forked workers count in each of them)."""
    try:
        return f"{psutil.Process().memory_info().rss / 2**30:.1f} GiB"
    except Exception:  # memory use is only logged, it must never fail the lookup
        return "unknown"


def _shared_data_key(step_type, abs_path, *settings):
    """Identify the data a step loaded from a file with given settings, including the file's
    version so an edited file is never matched to stale shared data.
    """
    try:
        stat = os.stat(abs_path)
        version = (stat.st_size, stat.st_mtime_ns)
    except (OSError, TypeError, ValueError):
        version = None
    return (step_type, abs_path, version) + tuple(repr(setting) for setting in settings)


def _load_rtree_geometries(abs_path, file_path, file_type, file_read_params, nearest_neighbor):
    """Read a geometry file and index it, per peril if it has a peril_id column.

    Returns:
        dict: {peril_id: _RtreeGeometries}, or {None: _RtreeGeometries} without a peril_id column.
    """
    logger.info("rtree: reading geometry file %s", abs_path)
    start = time.perf_counter()
    gdf_geometry = getattr(gpd, f"read_{file_type}")(abs_path, **file_read_params)
    logger.info("rtree: read %d geometries with columns %s in %.1fs, process memory %s",
                len(gdf_geometry), list(gdf_geometry.columns), time.perf_counter() - start, _memory_usage())

    if gdf_geometry.crs is None:
        warnings.warn(
            f"Geometry file '{file_path}' has no CRS metadata — assuming EPSG:4326. "
            "Re-save the file with a CRS to suppress this warning.",
            UserWarning,
        )
        gdf_geometry = gdf_geometry.set_crs("EPSG:4326")
    elif gdf_geometry.crs != "EPSG:4326":
        logger.info("rtree: reprojecting geometries from %s to EPSG:4326", gdf_geometry.crs)
        gdf_geometry = gdf_geometry.to_crs("EPSG:4326")

    start = time.perf_counter()
    centers = None
    if nearest_neighbor:
        with warnings.catch_warnings():
            # In the context of a nearest neighbour search, we accept a potentially significant
            # deviation between the spherical or ellipsoidal distance and the cartesian approximation
            # that's used here.
            warnings.filterwarnings('ignore', category=UserWarning, message="Geometry is in a geographic CRS.")
            centers = np.asarray(gdf_geometry.centroid.array)

    geometries = np.asarray(gdf_geometry.geometry.array)
    attributes = pd.DataFrame(gdf_geometry.drop(columns=gdf_geometry.geometry.name))
    del gdf_geometry

    if 'peril_id' not in attributes.columns:
        per_peril = {None: _RtreeGeometries(geometries, attributes, centers)}
        logger.info("rtree: indexed %d geometries (no peril_id column, used for all perils)%s in %.1fs, process memory %s",
                    len(geometries), " with centroids for nearest neighbour" if nearest_neighbor else "",
                    time.perf_counter() - start, _memory_usage())
        return per_peril

    peril_ids = attributes['peril_id'].to_numpy()
    attributes = attributes.drop(columns=['peril_id'])
    per_peril = {}
    for peril_id in np.unique(peril_ids):
        mask = peril_ids == peril_id
        per_peril[peril_id] = _RtreeGeometries(
            geometries[mask], attributes[mask], None if centers is None else centers[mask])
    logger.info("rtree: indexed geometries per peril %s%s in %.1fs, process memory %s",
                {str(peril_id): len(geometries.attributes) for peril_id, geometries in per_peril.items()},
                " with centroids for nearest neighbour" if nearest_neighbor else "",
                time.perf_counter() - start, _memory_usage())
    return per_peril


def _rtree_join(located, geometries, nearest_neighbor_max_distance):
    """Left spatial join of locations (all with coordinates) to the geometries they intersect.

    Gives the same result as geopandas.sjoin(how='left') followed by the nearest neighbour
    fallback: one row per (location, intersecting geometry) pair, in location order, keeping
    the location index; locations that intersect nothing get the nearest geometry centroid
    within nearest_neighbor_max_distance, or else NaN attributes.
    """
    x = located['longitude'].to_numpy(dtype='float64')
    y = located['latitude'].to_numpy(dtype='float64')
    loc_pos, geom_pos = geometries.tree.query(shapely.points(x, y), predicate='intersects')

    n_pairs = len(loc_pos)
    is_unmatched = np.ones(len(located), dtype=bool)
    is_unmatched[loc_pos] = False
    unmatched = np.flatnonzero(is_unmatched)
    n_in_geometry = len(located) - unmatched.size

    if nearest_neighbor_max_distance > 0 and unmatched.size and geometries.ball_tree is not None:
        distance, nearest = geometries.ball_tree.query(
            np.radians(np.column_stack([x[unmatched], y[unmatched]])), k=1)
        close = distance[:, 0] * EARTH_RADIUS_M <= nearest_neighbor_max_distance
        loc_pos = np.concatenate([loc_pos, unmatched[close]])
        geom_pos = np.concatenate([geom_pos, nearest[close, 0]])
        unmatched = unmatched[~close]

    logger.debug("rtree: %d locations: %d in a geometry (%d location-geometry pairs), %d matched by nearest "
                 "neighbour, %d unmatched", len(located), n_in_geometry, n_pairs,
                 len(located) - n_in_geometry - unmatched.size, unmatched.size)

    loc_pos = np.concatenate([loc_pos, unmatched])
    geom_pos = np.concatenate([geom_pos, np.full(unmatched.size, -1, dtype=geom_pos.dtype)])
    # stable sort by location only: a location's geometries stay in the STRtree's query order,
    # as with geopandas.sjoin, so the keys' row order is unchanged
    order = np.argsort(loc_pos, kind='stable')
    loc_pos, geom_pos = loc_pos[order], geom_pos[order]

    joined = located.iloc[loc_pos]
    # -1 is not a label of the RangeIndex, so unmatched locations get NaN attributes
    attributes = geometries.attributes.reindex(geom_pos)

    overlap = joined.columns.intersection(attributes.columns)
    if len(overlap):  # same suffixes as geopandas.sjoin
        joined = joined.rename(columns={col: f"{col}_left" for col in overlap})
        attributes = attributes.rename(columns={col: f"{col}_right" for col in overlap})

    attributes.index = joined.index
    return pd.concat([joined, attributes], axis=1)


class _MergeTable:
    """The table of a 'merge' step (columns lower-cased), with an index of its rows by join key.

    Merging each chunk of locations with the whole table makes pandas hash the join keys of
    every row of the table again for every chunk. Instead, the index is built once per set of
    join columns, each chunk looks up the rows its keys can match, and only those rows are
    merged, with the same pandas merge as before so matching and dtypes are unchanged.

    When the join keys are unique in the table, the index's hash table is built once and each
    chunk's lookup only costs as much as its own keys. Otherwise pandas has no hash table for
    the lookup and scans every row of the index for each chunk, which still avoids the merge
    with the whole table.
    """

    def __init__(self, df):
        self.df = df
        self._indexes = {}
        self._pid = os.getpid()

    def key_index(self, file_keys):
        """MultiIndex of the table's file_keys columns, ready for rows_matching.

        Whether the keys are unique is worked out here, and for unique keys the hash table
        rows_matching looks them up in is built, so both are computed once (in the parent
        process when the table is shared) rather than for the first chunk of each worker.
        """
        index = self._indexes.get(file_keys)
        if index is None:
            if os.getpid() != self._pid:
                logger.warning(
                    "merge step: indexing the table of %d rows on %s in lookup worker process %d. The parent "
                    "process didn't index it on these columns, so each worker builds and holds its own copy "
                    "of the index. Set join_columns %s on the step so the parent indexes it once for all workers.",
                    len(self.df), list(file_keys), os.getpid(), list(file_keys))
            start = time.perf_counter()
            index = pd.MultiIndex.from_frame(self.df[list(file_keys)])
            if index.is_unique:
                index.get_indexer(index[:1])  # builds the hash table now, not on the first chunk
            else:
                index.get_indexer_non_unique(index[:1])  # builds the index's engine now, not on the first chunk
            self._indexes[file_keys] = index
            logger.info("merge step: indexed %d rows on %s (%s keys) in %.1fs, process memory %s", len(index),
                        list(file_keys), "unique" if index.is_unique else "non-unique",
                        time.perf_counter() - start, _memory_usage())
        return index

    def rows_matching(self, file_keys, location_keys):
        """The table rows whose file_keys values are a row of location_keys, in table order.

        location_keys holds the locations' join columns, in file_keys order. Rows the merge
        would not match may be included, but every row it would match is.
        """
        try:
            index = self.key_index(file_keys)
            target = pd.MultiIndex.from_frame(location_keys.drop_duplicates())
            if index.is_unique:
                # hash table lookup of the chunk's keys only (-1 for a key not in the table)
                positions = index.get_indexer(target)
            else:
                # no hash table for non-unique keys: pandas scans every row of the index
                positions, _ = index.get_indexer_non_unique(target)
        except (TypeError, ValueError) as error:  # keys that can't be looked up: merge with the whole table
            logger.debug("merge step: merging with the whole table, key lookup failed: %s", error)
            return self.df
        positions = np.unique(positions[positions >= 0])
        if positions.size == 0:
            # No row matches, so any row can be merged without matching. Merging with an empty
            # table instead could change the dtype pandas gives the join columns (e.g. for a
            # categorical key), so the result would differ from merging with the whole table.
            positions = np.arange(min(1, len(self.df)))
        return self.df.iloc[positions]


# upper bound on the memory one read of a geotiff lookup may use, counting the values, their masks and any dtype
# conversion copy, so a large raster never has to be held in memory in full
GEOTIFF_READ_BUDGET_BYTES = 256 * 1024 ** 2


def geotiff_value_dtype(band_dtypes, defaults):
    """Dtype holding every requested band's values and every default exactly.

    The bands' common dtype is widened only for a default it cannot hold, such as -1 for a uint8 band (int16) or
    NaN for an integer band (a float dtype), so defaults that fit leave the output dtype as it was.

    Args:
        band_dtypes (list): dtypes of the requested bands
        defaults (dict): default value of each requested band, by output column name

    Returns:
        np.dtype: dtype for the lookup's output columns

    Raises:
        OasisException: if a default is not a number numpy can store (e.g. a string, None or a huge int)
    """
    dtype = np.result_type(*band_dtypes)
    for col_name, default in defaults.items():
        default_dtype = np.min_scalar_type(default)
        if default_dtype.kind not in 'biuf':
            raise OasisException(f"geotiff band {col_name} has default {default!r}, which is not a number numpy can store")
        with np.errstate(invalid='ignore', over='ignore'):
            cast = np.array(default, dtype=default_dtype).astype(dtype)
        if not (cast == default or (np.isnan(cast) and np.isnan(default))):
            dtype = np.result_type(dtype, default_dtype)
    return dtype


def geotiff_read_plan(row, col, block_shape, bytes_per_pixel, height, width, budget):
    """Group located pixels into reads of the raster blocks that hold them.

    Compressed rasters are decompressed a whole block at a time, so reading only the blocks holding locations
    avoids decompressing the space between them. Adjacent blocks in a block row are merged into one read, up to
    the budget, to keep the number of reads low where locations are dense. A block that alone exceeds the budget
    (e.g. a file stored as a single strip) is split into smaller tiles.

    Args:
        row (np.ndarray): pixel row of each location, all inside the raster
        col (np.ndarray): pixel column of each location, all inside the raster
        block_shape (tuple): (height, width) of the raster's blocks
        bytes_per_pixel (int): memory a read uses per pixel, over all requested bands
        height (int): number of rows in the raster
        width (int): number of columns in the raster
        budget (int): upper bound on the memory of a single read, in bytes

    Returns:
        tuple: (order, reads) where order sorts the locations by block and reads is a list of
            (start, end, window): the rasterio Window to read for the sorted locations order[start:end].
    """
    tile_h, tile_w = block_shape
    if tile_h * tile_w * bytes_per_pixel > budget:
        tile_h = max(1, budget // (tile_w * bytes_per_pixel))
        if tile_h * tile_w * bytes_per_pixel > budget:
            tile_w = max(1, budget // bytes_per_pixel)
    max_tiles = max(1, budget // (tile_h * tile_w * bytes_per_pixel))

    tile_row, tile_col = row // tile_h, col // tile_w
    order = np.lexsort((tile_col, tile_row))
    if not order.size:
        return order, []
    tile_row, tile_col = tile_row[order], tile_col[order]

    # first location of each distinct tile, then of each run of adjacent tiles in a tile row, split to fit the budget
    tile_start = np.flatnonzero(np.r_[True, (tile_row[1:] != tile_row[:-1]) | (tile_col[1:] != tile_col[:-1])])
    tiles_row, tiles_col = tile_row[tile_start], tile_col[tile_start]
    new_run = np.r_[True, (tiles_row[1:] != tiles_row[:-1]) | (tiles_col[1:] != tiles_col[:-1] + 1)]
    run_first_col = tiles_col[np.flatnonzero(new_run)][np.cumsum(new_run) - 1]
    run_chunk = (tiles_col - run_first_col) // max_tiles
    read_first_tile = np.flatnonzero(new_run | np.r_[True, run_chunk[1:] != run_chunk[:-1]])
    read_last_tile = np.r_[read_first_tile[1:], tiles_row.size] - 1

    starts = tile_start[read_first_tile]
    ends = np.r_[starts[1:], order.size]
    reads = []
    for start, end, first, last in zip(starts, ends, read_first_tile, read_last_tile):
        row_off, col_off = int(tiles_row[first]) * tile_h, int(tiles_col[first]) * tile_w
        row_end = min(row_off + tile_h, height)
        col_end = min((int(tiles_col[last]) + 1) * tile_w, width)
        reads.append((int(start), int(end), Window(col_off, row_off, col_end - col_off, row_end - row_off)))
    return order, reads


def geotiff_pixel_index(inv_transform, lon, lat, height, width):
    """Find the raster pixel containing each location.

    Args:
        inv_transform (affine.Affine): inverse of the raster's transform, mapping (x, y) coordinates to (col, row)
        lon (np.ndarray): location longitudes (x coordinates)
        lat (np.ndarray): location latitudes (y coordinates)
        height (int): number of rows in the raster
        width (int): number of columns in the raster

    Returns:
        tuple: (row, col, inside) where row and col are int64 pixel indexes, only meaningful where the boolean
            array inside is True; locations outside the raster or with missing coordinates are not inside.
    """
    # pixel (i, j) covers [j, j + 1) x [i, i + 1) in pixel space, so the pixel holding a point is the floor
    # of its pixel coordinates; NaN coordinates produce NaN here, which every comparison below rejects
    with np.errstate(invalid='ignore'):
        col = np.floor(inv_transform.a * lon + inv_transform.b * lat + inv_transform.c)
        row = np.floor(inv_transform.d * lon + inv_transform.e * lat + inv_transform.f)
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    row = np.where(inside, row, 0).astype(np.int64)
    col = np.where(inside, col, 0).astype(np.int64)
    return row, col, inside


@nb.njit(cache=True)
def z_index(x, y):
    """Returns the Z-order index of cell (x,y) in a grid"""
    bits = int(max(np.floor(np.log2(x + 1)) + 1, np.floor(np.log2(y + 1)) + 1))
    index = 0
    for i in range(bits):
        index |= ((x >> i) & 1) << (2 * i)
        index |= ((y >> i) & 1) << (2 * i + 1)
    return index


@nb.njit(cache=True)
def undo_z_index(z):
    """Returns the (x,y) coordinates of a z-order index in a grid"""
    x = y = 0
    for i in range(32):
        x |= ((z >> (2 * i)) & 1) << i
        y |= ((z >> (2 * i + 1)) & 1) << i
    return x, y


@nb.njit(cache=True)
def z_index_to_normal(index, size_across):
    """Converts from z-indexing to linear ordering"""
    if index == OASIS_UNKNOWN_ID:
        return index
    index -= 1
    lat, long = undo_z_index(index)
    return (lat + long * size_across + 1)


@nb.njit(cache=True)
def normal_to_z_index(index, size_across):
    """Converts from linear ordering to z-indexing"""
    if index == OASIS_UNKNOWN_ID:
        return index
    index -= 1
    return z_index(index % size_across, index // size_across) + 1


def create_lat_lon_id_functions(
    lat_min, lat_max, lon_min, lon_max, arc_size, lat_reverse, lon_reverse
):
    """Returns a function to give grid co-ordinates of a location"""
    lat_cell_size = arc_size
    lon_cell_size = arc_size

    if lat_reverse:
        @nb.njit()
        def lat_id(lat):
            return math.floor((lat_max - lat) / lat_cell_size)
    else:
        @nb.njit()
        def lat_id(lat):
            return math.floor((lat - lat_min) / lat_cell_size)

    if lon_reverse:
        @nb.njit()
        def lon_id(lon):
            return math.floor((lon_max - lon) / lon_cell_size)
    else:
        @nb.njit()
        def lon_id(lon):
            return math.floor((lon - lon_min) / lon_cell_size)

    return lat_id, lon_id


@nb.jit
def jit_geo_grid_lookup(
    lat, lon, lat_min, lat_max, lon_min, lon_max, compute_id,
    lat_id, lon_id
):
    """Returns an array of area peril IDs for all lats given"""
    area_peril_id = np.empty_like(lat, dtype=np.int64)
    for i in range(lat.shape[0]):
        if lat_min < lat[i] < lat_max and lon_min < lon[i] < lon_max:
            area_peril_id[i] = compute_id(lat[i], lon[i], lat_id, lon_id)
        else:
            area_peril_id[i] = OASIS_UNKNOWN_ID
    return area_peril_id


def get_step(grid):
    """Returns the grid size using the max and min long and latitude and arc size"""
    length = round((grid["lon_max"] - grid["lon_min"]) / grid["arc_size"])
    width = round((grid["lat_max"] - grid["lat_min"]) / grid["arc_size"])
    return length * width


SORTED_LOOKUP_META = 'sorted_lookup.json'


def write_sorted_lookup(table, out_dir, key_column, value_columns=None, file_type=None):
    """Write a table as the sorted arrays read by the builtin lookup's ``sorted_lookup`` step.

    Run once when building the model's keys data, not at lookup time.

    Args:
        table (pd.DataFrame, str): the table, or the path of a csv or parquet file holding it.
        out_dir (str): directory to write ``<column>.npy`` arrays and ``sorted_lookup.json`` into.
        key_column (str): integer column to look up on; its values must be unique and not null.
        value_columns (list, None): columns to return (default: every other column).
        file_type (str, None): ``'csv'`` or ``'parquet'`` when ``table`` is a path (default: from
            its extension).
    """
    if not isinstance(table, pd.DataFrame):
        file_type = file_type or ('parquet' if str(table).endswith(('.parquet', '.pq')) else 'csv')
        columns = None if value_columns is None else [key_column, *value_columns]
        if file_type == 'parquet':
            table = pd.read_parquet(table, columns=columns)
        else:
            table = pd.read_csv(table, usecols=columns)
    if key_column not in table.columns:
        raise OasisException(f"write_sorted_lookup: key column '{key_column}' not in {sorted(table.columns)}")
    if value_columns is None:
        value_columns = [col for col in table.columns if col != key_column]
    missing = [col for col in value_columns if col not in table.columns]
    if missing or not value_columns:
        raise OasisException(f"write_sorted_lookup: value columns {missing or value_columns} not usable; "
                             f"table columns are {sorted(table.columns)}")
    if key_column in value_columns:
        raise OasisException("write_sorted_lookup: the key column cannot also be a value column")

    key_series = table[key_column]
    if key_series.isna().any() or not pd.api.types.is_integer_dtype(key_series.dtype):
        raise OasisException(f"write_sorted_lookup: key column '{key_column}' must be integers with no nulls, "
                             f"got dtype {key_series.dtype}")
    keys = plain_numpy(key_series)
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    duplicated = keys[1:] == keys[:-1]
    if duplicated.any():
        raise OasisException(f"write_sorted_lookup: key column '{key_column}' has {int(duplicated.sum())} duplicated "
                             f"value(s), e.g. {np.unique(keys[1:][duplicated])[:3].tolist()}")

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{key_column}.npy'), keys)
    meta_values = {}
    for column in value_columns:
        series = table[column]
        if pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_float_dtype(series.dtype) \
                or pd.api.types.is_bool_dtype(series.dtype):
            if series.isna().any():
                raise OasisException(f"write_sorted_lookup: value column '{column}' has nulls")
            array = plain_numpy(series)
        else:
            raise OasisException(f"write_sorted_lookup: value column '{column}' must be numeric, got {series.dtype}")
        np.save(os.path.join(out_dir, f'{column}.npy'), array[order])
        meta_values[column] = str(array.dtype)
    with open(os.path.join(out_dir, SORTED_LOOKUP_META), 'w') as meta_file:
        json.dump({'key_column': key_column, 'rows': int(len(keys)), 'key_dtype': str(keys.dtype),
                   'value_columns': meta_values}, meta_file, indent=2)
    return out_dir


def plain_numpy(series):
    """A null-free numeric or boolean column as a plain numpy array (nullable pandas dtypes included)."""
    numpy_dtype = getattr(series.dtype, 'numpy_dtype', None)
    return series.to_numpy(dtype=numpy_dtype) if numpy_dtype is not None else series.to_numpy()


def load_sorted_lookup(directory, value_columns=None):
    """Open the arrays written by :func:`write_sorted_lookup`, memory-mapped.

    Returns:
        (keys, {value_column: array}, key_column)
    """
    meta_path = os.path.join(directory, SORTED_LOOKUP_META)
    if not os.path.isfile(meta_path):
        raise OasisException(f"sorted_lookup: '{meta_path}' not found. Build the directory with "
                             "oasislmf.lookup.builtin.write_sorted_lookup")
    with open(meta_path) as meta_file:
        meta = json.load(meta_file)
    key_column = meta['key_column']
    if value_columns is None:
        value_columns = list(meta['value_columns'])
    elif isinstance(value_columns, str):
        value_columns = [value_columns]
    unknown = [col for col in value_columns if col not in meta['value_columns']]
    if unknown:
        raise OasisException(f"sorted_lookup: value columns {unknown} not in '{directory}' "
                             f"({sorted(meta['value_columns'])})")

    def open_array(column):
        array = np.load(os.path.join(directory, f'{column}.npy'), mmap_mode='r')
        if array.ndim != 1 or len(array) != meta['rows']:
            raise OasisException(f"sorted_lookup: '{column}.npy' in '{directory}' has shape {array.shape}, "
                                 f"expected ({meta['rows']},) per {SORTED_LOOKUP_META}")
        return array

    keys = open_array(key_column)
    if not np.issubdtype(keys.dtype, np.integer):
        raise OasisException(f"sorted_lookup: keys in '{directory}' must be integers, got {keys.dtype}")
    return keys, {col: open_array(col) for col in value_columns}, key_column


def integer_keys(series, key_dtype):
    """Return (keys, valid) for a location key column: ``valid`` is False where the value is null,
    not an integer, or outside the range of ``key_dtype``. Invalid positions hold 0.
    """
    n = len(series)
    info = np.iinfo(key_dtype)
    if pd.api.types.is_integer_dtype(series.dtype):
        valid = series.notna().to_numpy()
        numpy_dtype = getattr(series.dtype, 'numpy_dtype', None)
        raw = series.to_numpy(dtype=numpy_dtype, na_value=0) if numpy_dtype is not None else series.to_numpy()
        if not np.issubdtype(raw.dtype, np.signedinteger):  # uint64 may exceed int64
            valid = valid & (raw <= info.max)
        else:
            valid = valid & (raw >= info.min) & (raw <= info.max)
        keys = np.zeros(n, dtype=key_dtype)
        keys[valid] = raw[valid]
        return keys, valid
    # float, object or text keys: float64 is exact for integers up to 2**53
    numeric = pd.to_numeric(series, errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
    valid = ~np.isnan(numeric)
    valid[valid] = (numeric[valid] == np.floor(numeric[valid])) & (numeric[valid] >= info.min) & (numeric[valid] <= info.max)
    keys = np.zeros(n, dtype=key_dtype)
    keys[valid] = numeric[valid]
    return keys, valid


def masked_take(array, position, rows, n):
    """Values of ``array`` at ``position`` placed at ``rows`` of a length ``n`` column; other rows are null.
    Integer and boolean values come back as pandas nullable arrays, floats with NaN.
    """
    taken = np.asarray(array[position])
    if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        data = np.zeros(n, dtype=array.dtype)
        data[rows] = taken
        mask = np.ones(n, dtype=bool)
        mask[rows] = False
        if np.issubdtype(array.dtype, np.bool_):
            return pd.arrays.BooleanArray(data, mask)
        return pd.arrays.IntegerArray(data, mask)
    data = np.full(n, np.nan, dtype=array.dtype)
    data[rows] = taken
    return data


key_columns = ['loc_id', 'peril_id', 'coverage_type', 'area_peril_id', 'vulnerability_id', 'status', 'message']


def expand_perils_covered(locations, perils_covered_column, peril_groups_df):
    """The perils each location row covers, with the peril group they came from.

    A peril reached through several listed codes (a group listed with perils it contains, e.g.
    OO1;ORF;OSF, or two groups that contain it) is kept once, with the first listed code that
    contains it. Identical location rows give their perils once. Only the perils covered column
    is expanded, so the memory used does not grow with the number of location columns.
    peril_groups_df maps peril_group_id to peril_id, see get_peril_groups_df.

    Returns:
        pd.DataFrame: row (position, not label, of the location row), peril_group_id, peril_id
    """
    rows = np.flatnonzero(~locations.duplicated().to_numpy())
    codes = pd.Series(locations[perils_covered_column].iloc[rows].astype(str).str.split(';').to_numpy(), index=rows).explode()
    perils = pd.DataFrame({'row': codes.index.to_numpy(),
                           'pos': np.arange(len(codes)),
                           'peril_group_id': codes.astype(peril_groups_df['peril_group_id'].dtype).array}).merge(peril_groups_df)
    # pandas < 2.2 does not keep the left row order in an inner merge (GH 54611), so restore the listed
    # order before keeping the first code per peril; pos increases across rows, so it alone is enough
    perils = perils.take(np.argsort(perils['pos'].to_numpy(), kind='stable')).drop(columns='pos')
    return perils.drop_duplicates(subset=['row', 'peril_id'], ignore_index=True)


def not_modelled_perils(locations, perils_covered_column, peril_groups_df, model_perils_covered):
    """The keys error message of each location with perils the model does not cover, one line per loc_id.

    The message lists those perils, ';'-joined: 'WEC;WSS not modelled'. A row whose codes name no
    known peril gives its codes as listed ('ZZZ not modelled'), and a blank row (None, NaN, '')
    gives 'no perils covered', so every location ends in keys or keys-errors. Rows sharing a loc_id
    give one line, the sorted union of their perils; a blank row adds nothing to the other rows of
    its loc_id. model_perils_covered None means the model covers every peril. Each distinct perils
    covered value is expanded once, so the cost hardly grows with the rows.

    Returns:
        pd.DataFrame: row (position of the loc_id's first location row) and message, in row order
    """
    codes, values = pd.factorize(locations[perils_covered_column], use_na_sentinel=False)
    blank = np.array([pd.isna(value) or str(value).strip() == '' for value in values], dtype=bool)
    values = np.array(['' if is_blank else str(value) for value, is_blank in zip(values, blank)], dtype=object)
    perils = expand_perils_covered(pd.DataFrame({'codes': values}), 'codes', peril_groups_df)

    # per distinct value: the perils not modelled, '' when there are none to list
    known = np.zeros(len(values), dtype=bool)
    known[perils['row'].to_numpy()] = True
    text = np.where(known | blank, '', values).astype(object)
    has_key = known  # the value gives at least one keys row
    if model_perils_covered is not None:
        modelled = perils['peril_id'].isin(model_perils_covered).to_numpy()
        has_key = np.zeros(len(values), dtype=bool)
        has_key[perils['row'].to_numpy()[modelled]] = True
        unmodelled = perils[~modelled].sort_values(['row', 'peril_id'], kind='stable')
        joined = unmodelled.groupby('row', sort=False)['peril_id'].agg(';'.join)
        text[joined.index.to_numpy()] = joined.to_numpy()

    loc_ids = locations['loc_id'].array
    row_text = text[codes]
    rows = np.flatnonzero(row_text != '')
    lines = pd.DataFrame({'row': rows, 'loc_id': loc_ids.take(rows), 'perils': row_text[rows]})
    lines = lines.drop_duplicates(subset=['loc_id', 'perils'])
    shared = lines['loc_id'].duplicated(keep=False).to_numpy()
    if shared.any():
        # rows of one loc_id listing different perils: a single line with all of them
        merged = (lines[shared].assign(perils=lines['perils'][shared].str.split(';')).explode('perils')
                  .drop_duplicates(subset=['loc_id', 'perils']).sort_values(['loc_id', 'perils'], kind='stable')
                  .groupby('loc_id', sort=False).agg(row=('row', 'min'), perils=('perils', ';'.join)).reset_index())
        lines = pd.concat([lines[~shared], merged])
    lines['message'] = lines['perils'] + ' not modelled'

    blank_rows = np.flatnonzero(blank[codes])
    if len(blank_rows):
        # a loc_id whose rows give neither a keys row nor perils to list covers no peril at all
        reported = np.flatnonzero(has_key[codes] | (row_text != ''))
        empty = pd.DataFrame({'row': blank_rows, 'loc_id': loc_ids.take(blank_rows)})
        empty = empty[~empty['loc_id'].isin(loc_ids.take(reported))].drop_duplicates(subset=['loc_id'])
        lines = pd.concat([lines, empty.assign(message='no perils covered')])
    return lines.sort_values('row', kind='stable', ignore_index=True)[['row', 'message']]


class PerilCoveredDeterministicLookup(AbstractBasicKeyLookup):
    multiproc_enabled = False

    def process_locations(self, locations):
        peril_groups_df = get_peril_groups_df()
        model_perils_covered = np.unique(pd.DataFrame({'peril_group_id': self.config['model_perils_covered']})
                                         .merge(peril_groups_df)['peril_id'])
        peril_covered_column = 'LocPerilsCovered' if 'LocPerilsCovered' in locations.columns else 'PolPerilsCovered'

        perils = expand_perils_covered(locations, peril_covered_column, peril_groups_df)
        keys_df = pd.DataFrame({'loc_id': locations['loc_id'].array.take(perils['row'].to_numpy()),
                                'peril_id': perils['peril_id'].array})

        coverage_df = pd.DataFrame({'coverage_type': self.config['supported_oed_coverage_types']}, dtype='Int32')
        keys_df = keys_df.merge(coverage_df, how="cross").sort_values(['loc_id', 'peril_id', 'coverage_type'], kind='stable')
        keys_df['message'] = ''
        success_df = keys_df['peril_id'].isin(model_perils_covered)
        success_df_len = keys_df[success_df].shape[0]
        keys_df.loc[success_df, 'area_peril_id'] = np.arange(1, success_df_len + 1)
        keys_df.loc[success_df, 'vulnerability_id'] = np.arange(1, success_df_len + 1)
        keys_df.loc[success_df, 'status'] = OASIS_KEYS_STATUS['success']['id']
        keys_df.loc[~success_df, ['status', 'message']
                    ] = OASIS_KEYS_STATUS['noreturn']['id'], 'unsuported peril_id'
        keys_df[['area_peril_id', 'vulnerability_id']] = keys_df[['area_peril_id', 'vulnerability_id']].astype('Int32')

        return keys_df


class Lookup(AbstractBasicKeyLookup, MultiprocLookupMixin):
    """Built-in Lookup class that implements the OasisLookupInterface.

    The aim of this class is to provide a data-driven lookup capability that is both flexible
    and efficient.

    It provides several generic function factories that can be defined in the config under the
    ``step_definition`` key, for example::

        "step_definition": {
            "split_loc_perils_covered": {
                "type": "split_loc_perils_covered",
                "columns": ["locperilscovered"],
                "parameters": {
                    "model_perils_covered": ["WTC", "WSS"]
                }
            },
            "vulnerability": {
                "type": "merge",
                "columns": ["peril_id", "coverage_type", "occupancycode"],
                "parameters": {
                    "file_path": "%%KEYS_DATA_PATH%%/vulnerability_dict.csv",
                    "id_columns": ["vulnerability_id"]
                }
            },
            "custom_status": {
                "type": "set_status",
                "columns": ["custom_status"],
                "parameters": {
                    "status_column": "custom_status"
                }
            }
        }

    ``set_status`` lets a model developer explicitly mark locations as, for example,
    ``notatrisk`` or ``notmodelled`` (see ``oasislmf.utils.status.OASIS_KEYS_STATUS`` for the
    full list of valid values) based on their own criteria — a spatial mask or lookup table
    merged into the Locations DataFrame by an earlier step — rather than relying only on the
    implicit success/fail logic derived from area_peril_id/vulnerability_id.

    Where each entry means:

    - **mapper key**: the ``step_name``; it is added to the lookup object as a method once the
      function has been built. It can take any value but must not collide with an existing method.
    - **type**: the function factory to call. For ``type`` ``<fct_type>`` the factory called is
      ``build_<fct_type>`` (e.g. ``"type": "merge"`` calls ``build_merge``).
    - **columns**: the columns required to apply the step. These matter because any column
      (except ``loc_id``) from the original Locations DataFrame that is not used by any step is
      dropped to reduce memory consumption.
    - **optional_columns**: columns the step uses if the location file has them. They are kept
      like ``columns`` but a missing one doesn't fail the step (e.g. ``latitude``/``longitude``,
      which are optional in OED). The step must cope with them being absent.
    - **parameters**: the parameters passed to the function factory.

    Once all the functions have been defined, the order in which they are applied is set in the
    config under the ``strategy`` key, for example::

        "strategy": ["split_loc_perils_covered", "vulnerability"]

    It is possible to subclass ``Lookup`` to create a custom step or function factory.

    For a custom **step**, add its definition to the ``mapper`` with no parameters::

        "my_custom_step": {
            "type": "custom_type",
            "columns": [...]
        }

    add it to your strategy —
    ``["split_loc_perils_covered", "vulnerability", "my_custom_step"]`` — and code the function
    in your subclass::

        class MyLookup(Lookup):
            @staticmethod
            def my_custom_step(locations):
                # do something on locations
                return modified_locations

    For a custom **function factory**, add its definition to ``step_definition`` with the
    required parameters::

        "my_custom_step": {
            "type": "custom_type",
            "columns": [...],
            "parameters": {
                "param1": "value1"
            }
        }

    add the step to your strategy and code the factory in your subclass::

        class MyLookup(Lookup):
            def build_custom_type(self, param1):
                def fct(locations):
                    # do something on locations that depends on param1
                    return modified_locations
                return fct
    """
    interface_version = "1"

    def set_step_function(self, step_name, step_config, function_being_set=None):
        """Set the step as a function of the lookup object if it's not already done and return it.
        if the step is composed of several child steps, it will set the child steps recursively.

        Args:
            step_name (str): name of the strategy for this step
            step_config (dict): config of the strategy for this step
            function_being_set (set, None): set of all the strategy that are parent of this step

        Returns:
            function: function corresponding this step
        """
        if hasattr(self, step_name):
            step_function = getattr(self, step_name)
        else:
            if step_config['type'] == 'combine':  # we need to build the child function
                if function_being_set is None:  # make sure we catch cyclic strategy definition
                    function_being_set = {step_name}
                elif step_name in function_being_set:
                    raise OasisException(f"lookup config has a cyclic strategy definition {function_being_set} then {step_name} again")
                else:
                    function_being_set.add(step_name)

                functions = []
                for child_step_name in step_config["parameters"]['strategy']:
                    child_step_config = self.config['step_definition'][child_step_name]
                    child_fct = self.set_step_function(
                        step_name=child_step_name,
                        step_config=child_step_config,
                        function_being_set=function_being_set)
                    functions.append({'name': child_step_name, 'function': child_fct,
                                      'columns': set(child_step_config.get("columns", []))})
                step_config['parameters']['strategy'] = functions

            step_function = getattr(self, f"build_{step_config['type']}")(**step_config['parameters'])
            setattr(self, step_name, step_function)
        return step_function

    def process_locations(self, locations):
        missing = [key for key in ('step_definition', 'strategy') if not self.config.get(key)]
        if missing:
            raise OasisException(
                f"lookup config is missing or has empty required key(s) {missing}. "
                "A built-in lookup needs 'step_definition' to define its steps and 'strategy' to order them"
            )

        # drop all unused columns and remove duplicate rows, find and rename useful columns
        lower_case_column_map = {column.lower(): column for column in locations.columns}
        useful_cols = self.useful_columns()

        useful_cols_map = {lower_case_column_map[useful_col.lower()]: useful_col
                           for useful_col in useful_cols
                           if useful_col.lower() in lower_case_column_map}
        locations = locations.rename(columns=useful_cols_map)
        locations = locations[list(useful_cols.intersection(locations.columns))].drop_duplicates()

        # set default status and message
        locations['status'] = OASIS_KEYS_STATUS['success']['id']
        locations['message'] = ''

        # process each step of the strategy
        for step_name in self.config["strategy"]:
            step_config = self.config['step_definition'][step_name]
            needed_column = set(step_config.get("columns", []))
            if not needed_column.issubset(locations.columns):
                raise OasisException(
                    f"Key Server Issue: missing columns {needed_column.difference(locations.columns)} for step {step_name}")
            step_function = self.set_step_function(step_name, step_config)
            locations = step_function(locations)

        key_columns = [
            'loc_id', 'peril_id', 'coverage_type', 'area_peril_id',
            'vulnerability_id', 'status', 'message'
        ]
        additional_columns = ['amplification_id', 'model_data', 'section_id', 'intensity_adjustment', 'return_period']
        for col in additional_columns:
            if col in locations.columns:
                key_columns += [col]

        locations = locations[key_columns]

        # check all ids are of the good type
        id_cols = ['coverage_type', 'area_peril_id', 'vulnerability_id']
        if 'amplification_id' in locations.columns:
            id_cols += ['amplification_id']
        # a location with no section fails like one with no areaperil; a step that already joined
        # sections into text ('3001;3002') is left as it is
        if 'section_id' in locations.columns and pd.api.types.is_numeric_dtype(locations['section_id']):
            id_cols += ['section_id']
        self.set_id_columns(locations, id_cols)
        # check all success location have all ids set correctly
        success_locations = locations.loc[locations['status'] == OASIS_KEYS_STATUS['success']['id']]
        for id_col in id_cols:
            unknown_ids = success_locations[id_col] == OASIS_UNKNOWN_ID
            fail_locations = success_locations.loc[unknown_ids].index
            # keep a step's explanation (e.g. why transform_coordinates gave no coordinates)
            detail = success_locations.loc[unknown_ids, 'message']
            locations.loc[fail_locations, 'status'] = OASIS_KEYS_STATUS['fail']['id']
            locations.loc[fail_locations, 'message'] = (f'{id_col} has an unknown id'
                                                        + detail.where(detail == '', ': ' + detail))
            success_locations = success_locations.loc[~unknown_ids]

        return locations

    def useful_columns(self):
        """The location columns kept for the steps, each under the one spelling the steps use.

        A column is kept under a single name, so two steps naming it with different capitalisation
        (in ``columns`` or ``optional_columns``) can't both find it: raise rather than let one of
        them fail with a KeyError. The ``geog_lookup`` slots are matched ignoring case, so they take
        whatever spelling a step lists and never clash.
        """
        spellings = {'loc_id': {'loc_id': ['(always kept)']}}  # lower -> spelling -> steps naming it
        for step_name, step_config in self.config['step_definition'].items():
            for col in list(step_config.get("columns", [])) + self.step_optional_columns(step_config, geog_slots=False):
                spellings.setdefault(col.lower(), {}).setdefault(col, []).append(step_name)
        clashes = {lower: steps for lower, steps in spellings.items() if len(steps) > 1}
        if clashes:
            details = "; ".join(
                ", ".join(f"'{spelling}' in {sorted(set(names))}" for spelling, names in steps.items())
                for steps in clashes.values())
            raise OasisException(
                f"Lookup config names the same location column with different capitalisation: {details}. "
                f"Use one spelling in every step's 'columns' and 'optional_columns'")
        useful_cols = {next(iter(steps)) for steps in spellings.values()}
        for step_config in self.config['step_definition'].values():
            for col in self.step_optional_columns(step_config):
                if col.lower() not in spellings:
                    useful_cols.add(col)
                    spellings[col.lower()] = {col: []}
        return useful_cols

    @staticmethod
    def step_optional_columns(step_config, geog_slots=True):
        """Columns a step uses if the location file has them, without requiring them.

        These are kept through the column filter in ``process_locations`` but, unlike ``columns``,
        a missing one doesn't fail the step. They are the step's ``optional_columns`` plus, for a
        ``geog_lookup`` step, the ``GeogScheme{N}``/``GeogName{N}`` slots it scans.
        """
        optional_columns = step_config.get("optional_columns", [])
        if isinstance(optional_columns, str):
            optional_columns = [optional_columns]
        optional_columns = list(optional_columns)
        if geog_slots and step_config.get("type") == "geog_lookup":
            slots = step_config.get("parameters", {}).get("slots", GEOG_LOOKUP_DEFAULT_SLOTS)
            for n in range(1, slots + 1):
                optional_columns += [f"geogscheme{n}", f"geogname{n}"]
        return optional_columns

    def to_abs_filepath(self, filepath):
        """Replace placeholder r'%%(.+?)%%' (ex: %%KEYS_DATA_PATH%%) with the path set in self.config

        Args:
            filepath (str): filepath with potentially a placeholder

        Returns:
            str: filepath where placeholder are replace their actual value.
        """
        placeholder_keys = set(re.findall(r'%%(.+?)%%', filepath))
        for placeholder_key in placeholder_keys:
            filepath = filepath.replace(f'%%{placeholder_key}%%', self.config[placeholder_key.lower()])
        if "keys_data_path" in [key.lower() for key in placeholder_keys]:
            return filepath
        else:
            return self.storage.get_storage_url(filepath, encode_params=False)[1].replace('file://', '')

    @staticmethod
    def set_id_columns(df, id_columns):
        """In Dataframes, only float column can have nan values. So after a left join for example if you have nan values
        that will change the type of the original column into float.
        this function replace the nan value with the OASIS_UNKNOWN_ID and reset the column type to int
        """
        # Allow caller to pass single string without wrapping in a list.
        if isinstance(id_columns, str):
            id_columns = [id_columns]
        for col in id_columns:
            if col not in df:
                df[col] = OASIS_UNKNOWN_ID
            else:
                df[col] = df[col].astype('Int64')
                df.loc[(df[col].isna()) | (df[col] <= 0), col] = OASIS_UNKNOWN_ID
        return df

    def build_interval_to_index(self, value_column_name, sorted_array, index_column_name=None, side='left'):
        """Allow to map a value column to an index according to it's index in the interval defined by sorted_array.
        nan value are kept as nan

        Args:
            value_column_name: name of the column to map
            sorted_array: sorted value that define the interval to map to
            index_column_name: name of the output column
            side: define what index is returned (left or right) in case of equality with one of the interval boundary

        Returns:
            function: return the mapping function
        """
        if isinstance(sorted_array, list):
            pass
        elif isinstance(sorted_array, str):
            sorted_array = [float(val) for val in open(self.to_abs_filepath(sorted_array)) if val.strip()]
        else:
            raise OasisException("sorted_array must be a list of the interval sorted or a path to a csv file containing those interval")

        if index_column_name is None:
            index_column_name = value_column_name + '_idx'

        def fct(locations):
            locations[index_column_name] = np.searchsorted(sorted_array, locations[value_column_name], side=side)
            empty_values = is_empty(locations, value_column_name)
            locations.loc[empty_values, index_column_name] = locations.loc[empty_values, value_column_name]
            return locations

        return fct

    @staticmethod
    def build_combine(id_columns, strategy, logical_type='or'):
        """Build a function that will combine several strategy trying to achieve the same purpose by different mean into one.
        for example, finding the correct area_peril_id for a location with one method using (latitude, longitude)
        and one using postcode.
        each strategy will be applied sequentially on the location that steal have OASIS_UNKNOWN_ID in their id_columns after the precedent strategy

        'or' example (note: ``id_columns`` is a list)::

            "vulnerability": {
                "type": "combine",
                "parameters": {
                    "id_columns": ["vulnerability_id"],
                    "strategy": ["vuln_cov_Building_Content", "vuln_cov_car"],
                    "logical_type": "or"
                }
            }

        'and' example (note: ``id_columns`` is a list of list)::

            "vuln_cov_car": {
                "type": "combine",
                "columns": ["autocode"],
                "parameters": {
                    "id_columns": [["vuln_id_car"], ["vulnerability_id"]],
                    "strategy": ["vulnerability_car", "coverage_type_car"],
                    "logical_type": "and"
                }
            }

        Args:
            id_columns (list): columns that will be checked to determine if a strategy has succeeded
            strategy (list): list of strategy to apply
            logical_type: if 'or' apply the next strategy only on invalid id_columns;
                if 'and' apply the next strategy only on valid id_columns (id_columns needs to be
                a list of list of columns so that each sublist is checked sequentially).
                With 'or', a strategy whose own ``columns`` are missing is skipped; with 'and',
                it raises an OasisException since every strategy is required.

        Returns:
            function: function combining all strategies
        """
        if logical_type.lower() == 'or':
            def fct(locations):
                initial_columns = locations.columns
                result = []
                for i, child_strategy in enumerate(strategy):
                    if not child_strategy['columns'].issubset(locations.columns):  # needed column not present to run this strategy
                        logger.warning("combine: skipping step %s, missing columns %s",
                                       child_strategy.get('name', i),
                                       sorted(child_strategy['columns'].difference(locations.columns)))
                        continue
                    locations = child_strategy['function'](locations)
                    locations = Lookup.set_id_columns(locations, id_columns)
                    is_valid = (locations[id_columns] != OASIS_UNKNOWN_ID).any(axis=1)
                    result.append(locations[is_valid])
                    locations = locations[~is_valid][initial_columns].copy()
                    if locations.empty:
                        break
                result.append(locations)
                return Lookup.set_id_columns(pd.concat(result, ignore_index=True), id_columns)

        elif logical_type.lower() == 'and':
            def fct(locations):
                initial_columns = locations.columns
                result = []
                for i, child_strategy in enumerate(strategy):
                    # every strategy is required in 'and', so a missing column can't fall back to the next one
                    if not child_strategy['columns'].issubset(locations.columns):
                        raise OasisException(
                            f"Key Server Issue: missing columns {child_strategy['columns'].difference(locations.columns)} "
                            f"for step {child_strategy.get('name', i)} of 'and' combine")
                    locations = child_strategy['function'](locations)
                    locations = Lookup.set_id_columns(locations, id_columns[i])
                    is_valid = (locations[id_columns[i]] != OASIS_UNKNOWN_ID).any(axis=1)
                    result.append(locations[~is_valid][initial_columns])
                    locations = locations[is_valid]

                    if locations.empty:
                        break
                result.append(locations)
                return Lookup.set_id_columns(pd.concat(result, ignore_index=True), id_columns[-1])

        else:
            raise OasisException(f"Unsupported logical_type {logical_type}")
        return fct

    @staticmethod
    def build_split_loc_perils_covered(model_perils_covered=None, not_covered_status='notmodelled'):
        """Split the value of LocPerilsCovered into multiple line, taking peril group into account
        drop all line that are not in the list model_perils_covered

        Each peril gives one row per location. When it is reached through more than one listed
        group (e.g. ``OO1;ORF``), the row kept has as ``peril_group_id`` the first listed group
        that contains it.

        Each location covering a peril the model does not also gets one ``not_covered_status`` row,
        whose message lists those perils (``WEC;WSS not modelled``), even when its other perils are
        modelled. A location listing no known peril gets one with its codes as listed
        (``ZZZ not modelled``), and a blank one (None, NaN or '') gets ``no perils covered``.

        Args:
            model_perils_covered (list, str, None): perils covered by the model. Locations whose
                LocPerilsCovered/PolPerilsCovered has perils outside this list get a row marked
                with ``not_covered_status`` listing them.
            not_covered_status (str): status key (see ``oasislmf.utils.status.OASIS_KEYS_STATUS``)
                to assign to the row listing the perils not covered by the model. Defaults to
                ``'notmodelled'``. Set to ``'notatrisk'`` to restore the pre-4.x behaviour.
        """
        if not_covered_status not in OASIS_KEYS_STATUS:
            raise OasisException(
                f"unknown not_covered_status '{not_covered_status}', must be one of {sorted(OASIS_KEYS_STATUS)}")

        peril_groups_df = get_peril_groups_df()

        def fct(locations):
            for col in locations.columns:
                if col.lower() == 'locperilscovered':
                    perils_covered_column = col
                    break
                elif col.lower() == 'polperilscovered':
                    perils_covered_column = col
                    break
            else:
                raise OasisException('missing PerilsCovered column in location')

            model = pd.Series(model_perils_covered) if model_perils_covered else None
            perils = expand_perils_covered(locations, perils_covered_column, peril_groups_df)
            if model is not None:
                # filter before copying the location rows; the output is ordered by peril_id
                perils = perils[perils['peril_id'].isin(model)]
                perils = perils.take(np.argsort(perils['peril_id'].to_numpy(), kind='stable'))
            peril_locations = locations.iloc[perils['row'].to_numpy()].reset_index(drop=True)
            peril_locations['peril_group_id'] = perils['peril_group_id'].array
            peril_locations['peril_id'] = perils['peril_id'].array

            not_modelled = not_modelled_perils(locations, perils_covered_column, peril_groups_df, model)
            if len(not_modelled):
                not_covered_location = locations.iloc[not_modelled['row'].to_numpy()].reset_index(drop=True)
                not_covered_location['status'] = OASIS_KEYS_STATUS[not_covered_status]['id']
                not_covered_location['message'] = not_modelled['message'].to_numpy()
                peril_locations = pd.concat([peril_locations, not_covered_location], ignore_index=True)
            return peril_locations
        return fct

    @staticmethod
    def build_set_status(status_column, message_column=None):
        """Set the status (and optionally message) of locations from a model-developer-supplied column.

        Allows a model developer to explicitly mark locations (e.g. as ``notatrisk`` or
        ``notmodelled``) based on their own criteria (a spatial mask, a lookup table, ...) merged
        into the Locations DataFrame ahead of this step, rather than relying on the implicit
        success/fail logic derived from area_peril_id/vulnerability_id.

        Args:
            status_column (str): name of the column containing the status to set for each
                location. Rows where this column is empty keep whatever status was already set.
            message_column (str, None): name of the column containing the message to set
                alongside the status. If not provided, the message is left untouched.
        """
        valid_status_ids = {status['id'] for status in OASIS_KEYS_STATUS.values()}

        def fct(locations):
            if status_column not in locations.columns:
                raise OasisException(f"missing column {status_column} in location")

            set_status = ~is_empty(locations, status_column)
            unknown_status = set_status & ~locations[status_column].isin(valid_status_ids)
            if unknown_status.any():
                raise OasisException(
                    f"unknown status value(s) {sorted(locations.loc[unknown_status, status_column].unique())} "
                    f"in column {status_column}, must be one of {sorted(valid_status_ids)}")

            locations.loc[set_status, 'status'] = locations.loc[set_status, status_column]
            if message_column is not None:
                if message_column not in locations.columns:
                    raise OasisException(f"missing column {message_column} in location")
                set_message = set_status & ~is_empty(locations, message_column)
                locations.loc[set_message, 'message'] = locations.loc[set_message, message_column]
            return locations
        return fct

    PREPARE_OPTIONS = ('default', 'strip', 'upper', 'lower', 'max', 'min', 'type', 'rename', 'drop')

    @staticmethod
    def build_prepare(**kwargs):
        """Prepare the dataframe column by column.

        Each key is a column name and each value a dict of preparations, applied in this order:

        - **default**: create the column if missing and replace the nan value with the default value
        - **strip**: (``true``) remove leading and trailing whitespace from text values
        - **upper**: (``true``) upper-case text values
        - **lower**: (``true``) lower-case text values (cannot be combined with ``upper``)
        - **max**: truncate the values in a column to the specified max
        - **min**: truncate the values in a column to the specified min
        - **type**: convert the type of the column to the specified numpy dtype. Note that we use
          the string representation of numpy dtype available at
          https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
        - **rename**: rename the column to this name. An existing column with that name is replaced.
        - **drop**: (``true``) remove the column. A missing column is ignored.

        ``strip``, ``upper`` and ``lower`` only change string values; null and non-text values
        (e.g. a numeric postcode) are left as they are. Use them to normalise a column before a
        ``merge`` on it, as ``merge`` matches values exactly.

        Columns are processed in the order they are listed, so a later entry sees the result of an
        earlier ``rename``.
        """
        for column_name, preparations in kwargs.items():
            if not isinstance(preparations, dict):
                raise OasisException(
                    f"prepare step: the preparations for column '{column_name}' must be a dict, got {preparations!r}")
            unknown = set(preparations) - set(Lookup.PREPARE_OPTIONS)
            if unknown:
                raise OasisException(
                    f"prepare step: unknown option(s) {sorted(unknown)} for column '{column_name}'. "
                    f"Valid options are {list(Lookup.PREPARE_OPTIONS)}")
            if preparations.get('upper') and preparations.get('lower'):
                raise OasisException(
                    f"prepare step: column '{column_name}' cannot have both 'upper' and 'lower'")
            if 'rename' in preparations and not (isinstance(preparations['rename'], str) and preparations['rename']):
                raise OasisException(
                    f"prepare step: 'rename' for column '{column_name}' must be a non-empty column name")

        def text_op(series, op):
            if isinstance(series.dtype, pd.CategoricalDtype):
                # OED string columns are categories: transform each category once, not each row.
                categories = series.cat.categories
                new_categories = text_op(pd.Series(categories, dtype=categories.dtype), op)
                if new_categories.is_unique:
                    return series.cat.rename_categories(pd.Index(new_categories))
                # e.g. 'ab' and 'AB ' both become 'AB': merge them into one category
                return series.map(dict(zip(categories, new_categories))).astype('category')
            if isinstance(series.dtype, pd.StringDtype):
                return getattr(series.str, op)()
            if pd.api.types.is_object_dtype(series.dtype):
                return series.map(lambda value: getattr(value, op)() if isinstance(value, str) else value)
            return series  # numeric, bool, datetime...: nothing to normalise

        def prepare(locations):
            for column_name, preparations in kwargs.items():
                if "default" in preparations:
                    if column_name not in locations.columns:
                        locations[column_name] = preparations["default"]
                    else:
                        fill_empty(locations, column_name, preparations["default"])
                if preparations.get('drop'):
                    locations = locations.drop(columns=column_name, errors='ignore')
                    continue
                if column_name not in locations.columns and set(preparations) - {'default'}:
                    raise OasisException(
                        f"prepare step: column '{column_name}' is missing. List it in the step's 'columns' "
                        f"so it is kept, or give it a 'default' to create it")
                for op in ('strip', 'upper', 'lower'):
                    if preparations.get(op):
                        locations[column_name] = text_op(locations[column_name], op)
                if 'max' in preparations:
                    locations.loc[locations[column_name] > preparations['max'], column_name] = preparations['max']
                if 'min' in preparations:
                    locations.loc[locations[column_name] < preparations['min'], column_name] = preparations['min']
                if 'type' in preparations:
                    locations[column_name] = locations[column_name].astype(preparations['type'])
                new_name = preparations.get('rename')
                if new_name is not None and new_name != column_name:
                    if column_name not in locations.columns:
                        raise OasisException(
                            f"prepare step: cannot rename missing column '{column_name}' to '{new_name}'")
                    locations = (locations.drop(columns=new_name, errors='ignore')
                                 .rename(columns={column_name: new_name}))
            return locations
        return prepare

    def build_rtree(
        self,
        file_path,
        file_type,
        id_columns,
        area_peril_read_params=None,
        file_read_params=None,
        nearest_neighbor_min_distance=-1,
        nearest_neighbor_max_distance=-1
    ):
        """Function Factory to associate location to a geometry (e.g. area peril) based on the rtree method

        !!!
        please note that this method is quite time consuming (especially if you use the nearest neighbor option
        if your geometries are square you should use the function fixed_size_geo_grid
        !!!

        file_path: is the path to the file containing the geometries.
            this file must be a geopandas Dataframe with a valid geometry.
            an example on how to create such dataframe is available in PiWind
            if you are new to geo data (in python) and want to learn more, you may have a look at this excellent course:
            https://autogis-site.readthedocs.io/

        file_type: can be any format readable by geopandas ('file', 'parquet', ...)
            see: https://geopandas.readthedocs.io/en/latest/docs/reference/io.html
            you may have to install additional library such as pyarrow for parquet

        id_columns (str or list(str)): column to transform to an 'id_column' (type int32 with nan replace by -1).

        file_read_params (dict or None): extra kwargs to pass to the file reading function.

        area_peril_read_params: deprecated alias for file_read_params.

        nearest_neighbor_max_distance: option to look for the nearest neighboring geometry up to
            this distance away if no containing geometries are found.
            We use:
            https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
            but alternatives can be found here:
            https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
            This distance should be provided in metres and is the Euclidean distance, not the
            distance on a sphere or ellipsoid. It is the maximum accepted distance between the point
            locations and the containing geometry centroids.

        nearest_neighbor_min_distance: deprecated alias for nearest_neighbor_max_distance. May be
            removed in a future version.

        A location falling in several overlapping geometries gets one row per geometry. For
        section_id that is the expected form: the keys rows of an item that differ only in
        section_id are combined when the GUL inputs are built.

        Memory: the file is read and indexed once per lookup. When keys are generated with several
        processes, it is read once in the parent process before the workers are forked (see
        multiproc_shared_data) and the workers share that copy. Each chunk of locations is joined
        with an STRtree query, so the memory it needs grows with the matches, not the file size.
        To keep only some of the file's columns, pass e.g. {"columns": [...]} in file_read_params.
        """
        abs_path, file_type, file_read_params, nearest_neighbor_max_distance = self._rtree_settings(
            file_path, file_type, area_peril_read_params, file_read_params,
            nearest_neighbor_min_distance, nearest_neighbor_max_distance)
        nearest_neighbor = nearest_neighbor_max_distance > 0

        key = _shared_data_key('rtree', abs_path, file_type, file_read_params, nearest_neighbor)
        per_peril = _SHARED_STEP_DATA.get(key)
        if per_peril is None:
            per_peril = _load_rtree_geometries(abs_path, file_path, file_type, file_read_params, nearest_neighbor)
        else:
            logger.info("rtree: using the geometries of %s already loaded by the parent process", abs_path)

        def get_geometry(locations, geometries):
            has_coordinates = locations["longitude"].notna() & locations["latitude"].notna()
            located = locations[has_coordinates]
            if not has_coordinates.all():
                logger.debug("rtree: %d locations without coordinates are left unmatched", (~has_coordinates).sum())
            if located.empty:
                gdf_loc = locations.copy()
            else:
                gdf_loc = _rtree_join(located, geometries, nearest_neighbor_max_distance)
                if not has_coordinates.all():
                    gdf_loc = pd.concat([gdf_loc, locations[~has_coordinates]])
            self.set_id_columns(gdf_loc, id_columns)

            # columns the previous geopandas.sjoin implementation created, kept dropped for compatibility
            gdf_loc = gdf_loc.drop(columns=['index_right', 'index_left', 'geometry', 'center'], errors='ignore')

            return gdf_loc

        def fct(locations):
            if None in per_peril:
                return get_geometry(locations, per_peril[None])

            res = [locations[~locations['peril_id'].isin(list(per_peril))]]
            if len(res[0]):
                logger.debug("rtree: %d locations have a peril with no geometries, perils %s",
                             len(res[0]), res[0]['peril_id'].unique().tolist())
            for peril_id, geometries in per_peril.items():
                logger.debug("rtree: joining locations for peril %s", peril_id)
                res.append(get_geometry(locations.loc[locations['peril_id'] == peril_id], geometries))

            return pd.concat(res).reset_index()
        return fct

    def _rtree_settings(self, file_path, file_type, area_peril_read_params=None, file_read_params=None,
                        nearest_neighbor_min_distance=-1, nearest_neighbor_max_distance=-1, **kwargs):
        """Validate build_rtree's parameters and resolve the deprecated ones.

        Returns:
            (abs_path, file_type, file_read_params, nearest_neighbor_max_distance)
        """
        if nearest_neighbor_min_distance > 0:
            warn_deprecated("Parameter `nearest_neighbor_min_distance` is deprecated and may be "
                            "removed in a future version. Please use `nearest_neighbor_max_distance` "
                            "instead.")
            if not nearest_neighbor_max_distance > 0:
                nearest_neighbor_max_distance = nearest_neighbor_min_distance

        if area_peril_read_params:
            warn_deprecated("Parameter `area_peril_read_params` is deprecated and may be removed in a "
                            "future version. Please use `file_read_params` instead.")
            if not file_read_params:
                file_read_params = area_peril_read_params

        if Point is None:
            raise OasisException(f"shapely and geopandas modules are needed for rtree, {OPT_INSTALL_MESSAGE}")

        if not hasattr(gpd, f"read_{file_type}"):
            raise OasisException(f"Unrecognised Geopandas read type {file_type}")

        if nearest_neighbor_max_distance > 0 and BallTree is None:
            raise OasisException(f"scikit-learn modules are needed for rtree with nearest_neighbor_max_distance, {OPT_INSTALL_MESSAGE}")

        return self.to_abs_filepath(file_path), file_type, file_read_params or {}, nearest_neighbor_max_distance

    @contextmanager
    def multiproc_shared_data(self, sample_locations=None):
        """Load the data of the lookup steps that worker processes can share, for the duration of the context.

        Called in the parent process before the lookup workers are forked (see
        KeyServer.generate_key_files_multiproc). The files of the 'rtree' and 'merge' steps are
        read and indexed once here, and each worker's build_rtree and build_merge reuse that copy
        (shared copy-on-write) instead of reading the file again.

        A merge step joins on the columns its file has in common with the locations it gets,
        and earlier steps can add those columns (e.g. an rtree step adding area_peril_id), so
        they're only known when the lookup runs. Unless the step sets join_columns, its table
        is indexed by running the lookup on sample_locations (a few rows of the locations)
        here, so the workers share that index instead of each building its own.
        """
        keys = []
        frozen = False
        merge_tables = []
        try:
            used_steps = self._strategy_step_names()
            for step_name, step_config in (self.config.get('step_definition') or {}).items():
                step_type = step_config.get('type')
                if step_type not in ('rtree', 'merge'):
                    continue
                if step_name not in used_steps:
                    logger.debug("%s step %s is not in the strategy, its file is not loaded", step_type, step_name)
                    continue
                # a subclass overriding the step's function factory may not use the data loaded here
                if getattr(type(self), f'build_{step_type}') is not getattr(Lookup, f'build_{step_type}'):
                    continue
                parameters = step_config.get('parameters', {})
                if step_type == 'rtree':
                    abs_path, file_type, file_read_params, nearest_neighbor_max_distance = self._rtree_settings(**parameters)
                    nearest_neighbor = nearest_neighbor_max_distance > 0
                    key = _shared_data_key('rtree', abs_path, file_type, file_read_params, nearest_neighbor)
                    if key not in _SHARED_STEP_DATA:
                        logger.info("rtree: loading %s in the parent process, to share with the lookup workers", abs_path)
                        _SHARED_STEP_DATA[key] = _load_rtree_geometries(
                            abs_path, parameters['file_path'], file_type, file_read_params, nearest_neighbor)
                        keys.append(key)
                else:
                    abs_path, file_type, read_kwargs, join_columns = self._merge_settings(**parameters)
                    key = _shared_data_key('merge', abs_path, file_type, read_kwargs)
                    if key not in _SHARED_STEP_DATA:
                        logger.info("merge step: loading %s in the parent process, to share with the lookup workers", abs_path)
                        _SHARED_STEP_DATA[key] = self._read_merge_table(abs_path, file_type, read_kwargs)
                        keys.append(key)
                    table = _SHARED_STEP_DATA[key]
                    merge_tables.append(table)
                    if join_columns and set(join_columns).issubset(table.df.columns):
                        table.key_index(tuple(sorted(join_columns)))
            if merge_tables and sample_locations is not None:
                self._index_merge_tables_on_sample(sample_locations, merge_tables)
            if keys:
                # Keep the workers' garbage collector from writing to the shared objects' headers,
                # which would copy their memory pages into every worker.
                gc.freeze()
                frozen = True
            yield
        finally:
            if frozen:
                gc.unfreeze()
            for key in keys:
                _SHARED_STEP_DATA.pop(key, None)

    def _index_merge_tables_on_sample(self, sample_locations, merge_tables):
        """Run the lookup on sample_locations, so the merge steps index their shared tables on the
        columns they actually join on. The results are discarded.
        """
        start = time.perf_counter()
        try:
            # building the steps changes the config (combine strategies become functions), so the
            # sample runs on its own copy, leaving the config the workers' lookups are made from as it is
            sample_lookup = copy.copy(self)
            sample_lookup.config = copy.deepcopy(self.config)
            sample_lookup.process_locations(sample_locations)
        except Exception as error:  # only an optimisation, the tables it didn't index are indexed by the workers
            logger.warning("merge step: the lookup failed on the %d sample locations used to index the merge tables "
                           "(%s: %s)", len(sample_locations), type(error).__name__, error)
        logger.info("merge step: ran the lookup on %d sample locations to index the merge tables in %.1fs, "
                    "process memory %s", len(sample_locations), time.perf_counter() - start, _memory_usage())
        for table in merge_tables:
            if not table._indexes:
                logger.warning("merge step: the table with columns %s wasn't indexed in the parent process, so each "
                               "lookup worker that uses it will index it itself. Set join_columns on its step to "
                               "index it once for all workers.", list(table.df.columns))

    def _strategy_step_names(self):
        """Names of the steps the lookup runs: those in the config's 'strategy', and the steps
        of the 'combine' steps among them, recursively. Other steps in 'step_definition' are
        never run.
        """
        step_definition = self.config.get('step_definition') or {}
        used = set()
        to_visit = list(self.config.get('strategy') or [])
        while to_visit:
            step = to_visit.pop()
            # once a combine step is built, its strategy holds {'name': ..., 'function': ...} instead of names
            name = step['name'] if isinstance(step, dict) else step
            if name in used:
                continue
            used.add(name)
            step_config = step_definition.get(name) or {}
            if step_config.get('type') == 'combine':
                to_visit.extend((step_config.get('parameters') or {}).get('strategy') or [])
        return used

    @staticmethod
    def build_fixed_size_geo_grid_multi_peril(perils_dict):
        """Create multiple grids of varying resolution, one per peril, and
        associate an id to each square of the grid using the
        `fixed_size_geo_grid` method.

        Args:
            perils_dict (dict): Dictionary with ``peril_id`` as key and ``fixed_size_geo_grid``
                parameter dict as value, i.e. ``{'peril_id': {fixed_size_geo_grid parameters}}``
        """
        def fct(locs_peril):
            start_index = 0
            step = get_step(next(iter(perils_dict.values())))

            locs_peril["area_peril_id"] = OASIS_UNKNOWN_ID  # if `peril_id` not in `perils_dict`
            for peril_id, fixed_geo_grid_params in perils_dict.items():
                curr_grid_fct = Lookup.build_fixed_size_geo_grid(**fixed_geo_grid_params)

                curr_locs_peril = locs_peril[locs_peril['peril_id'] == peril_id]
                curr_locs_peril = curr_grid_fct(curr_locs_peril)
                curr_locs_peril.loc[
                    curr_locs_peril["area_peril_id"] != OASIS_UNKNOWN_ID,
                    "area_peril_id"
                ] = curr_locs_peril["area_peril_id"] + start_index

                start_index += step

                locs_peril[locs_peril["peril_id"] == peril_id] = curr_locs_peril
            return locs_peril
        return fct

    @staticmethod
    def build_fixed_size_geo_grid(lat_min, lat_max, lon_min, lon_max, arc_size, lat_reverse=False, lon_reverse=False, lon_first=False):
        """Associate an id to each square of the grid define by the limit of lat and lon
        reverse allow to change the ordering of id from (min to max) to (max to min)
        """
        lat_id, lon_id = create_lat_lon_id_functions(
            lat_min, lat_max, lon_min, lon_max, arc_size,
            lat_reverse, lon_reverse
        )

        if lon_first:
            grid_size = round((lon_max - lon_min) / arc_size)
        else:
            grid_size = round((lat_max - lat_min) / arc_size)

        @nb.jit(cache=True)
        def get_id(lat, lon, lat_id, lon_id):
            if lon_first:
                return lon_id(lon) + lat_id(lat) * grid_size + 1
            return lon_id(lon) * grid_size + lat_id(lat) + 1

        def geo_grid_lookup(locations):
            locations['area_peril_id'] = jit_geo_grid_lookup(
                locations['latitude'].to_numpy(),
                locations['longitude'].to_numpy(),
                lat_min, lat_max, lon_min, lon_max,
                get_id, lat_id, lon_id
            )
            return locations

        return geo_grid_lookup

    @staticmethod
    def build_fixed_size_z_index_geo_grid_multi_peril(perils_dict):
        """Create multiple grids of varying resolution, one per peril, and associate an id to each square of the grid using the
        `fixed_size_z_index_geo_grid` method.

        Args:
            perils_dict (dict): Dictionary with ``peril_id`` as key and ``fixed_size_geo_grid``
                parameter dict as value, i.e. ``{'peril_id': {fixed_size_geo_grid parameters}}``
        """
        def fct(locs_peril):
            locs_peril["area_peril_id"] = OASIS_UNKNOWN_ID
            # if `peril_id` not in `perils_dict`
            shift = len(perils_dict.items())
            for index, (peril_id, fixed_geo_grid_params) in enumerate(perils_dict.items()):
                curr_grid_fct = Lookup.build_fixed_size_z_index_geo_grid(**fixed_geo_grid_params)

                curr_locs_peril = locs_peril[locs_peril['peril_id'] == peril_id]
                curr_locs_peril = curr_grid_fct(curr_locs_peril)
                curr_locs_peril.loc[
                    curr_locs_peril["area_peril_id"] != OASIS_UNKNOWN_ID,
                    "area_peril_id"
                ] = curr_locs_peril["area_peril_id"] * shift + index

                locs_peril[locs_peril["peril_id"] == peril_id] = curr_locs_peril

            return locs_peril
        return fct

    @staticmethod
    def build_fixed_size_z_index_geo_grid(
        lat_min, lat_max, lon_min, lon_max, arc_size,
        lat_reverse=False, lon_reverse=False, lon_first=False
    ):
        """Associate an id to each square of the grid defined by z-order indexing.
        reverse allow to change the ordering of id from (min to max) to
        (max to min)
        """
        lat_id, lon_id = create_lat_lon_id_functions(
            lat_min, lat_max, lon_min, lon_max, arc_size,
            lat_reverse, lon_reverse
        )

        @nb.jit(cache=True)
        def get_id(lat, lon, lat_id, lon_id):
            if lon_first:
                return z_index(lon_id(lon), lat_id(lat)) + 1
            return z_index(lat_id(lat), lon_id(lon)) + 1

        def geo_grid_lookup(locations):
            locations['area_peril_id'] = jit_geo_grid_lookup(
                locations['latitude'].to_numpy(),
                locations['longitude'].to_numpy(),
                lat_min, lat_max, lon_min, lon_max,
                get_id, lat_id, lon_id
            )
            return locations

        return geo_grid_lookup

    def build_geotiff(self, file_path, band_info):
        """Build a lookup function that assigns geotiff band values to each lat/lon.

        The raster is not held in memory: each call reads only the requested bands of the raster blocks holding its
        locations, in reads that fit within GEOTIFF_READ_BUDGET_BYTES (see geotiff_read_plan).

        The raster must use a geographic CRS (or none, in which case it is assumed to be in longitude/latitude).

        Pixels the raster marks as invalid, through a band's nodata value or a GDAL mask band, take the band's
        default, as do locations outside the raster or with missing coordinates.

        Args:
            file_path: path to the geotiff file
            band_info: a dict where keys are assigned column name, and values are dicts with
                id is the id of the band in the tiff file (starting at 1)
                default is the value for nodata pixels and for locations outside of the raster or with missing
                coordinates; a default the band's dtype cannot hold (e.g. -1 for uint8, NaN for an integer band)
                widens the column's dtype
        Returns:
            function to assign band value to each corresponding lat lon
        """
        if rasterio is None:
            raise OasisException(
                f"rasterio module is needed for geotiff (it replaces gdal, which geotiff no longer uses), {OPT_INSTALL_MESSAGE}"
            )

        abs_file_path = self.to_abs_filepath(file_path)
        with rasterio.open(abs_file_path) as dataset:
            band_count = dataset.count
            band_dtypes = dataset.dtypes
            inv_transform = ~dataset.transform
            height, width = dataset.height, dataset.width
            crs = dataset.crs
            block_shapes = dataset.block_shapes

        # locations are matched by longitude/latitude, which only line up with the pixels of a geographic raster
        if crs and not crs.is_geographic:
            raise OasisException(
                f"geotiff {file_path} uses the projected CRS {crs.to_string()}, but locations are matched by "
                "longitude/latitude: reproject the raster to a geographic CRS such as EPSG:4326"
            )

        band_ids = []
        for col_name, info in band_info.items():
            if not 1 <= info['id'] <= band_count:
                raise OasisException(f"band {col_name}, {info} has id outside of [1-{band_count}]")
            band_ids.append(info['id'])
        dtype = geotiff_value_dtype([band_dtypes[band_id - 1] for band_id in band_ids],
                                    {col_name: info['default'] for col_name, info in band_info.items()})
        defaults = np.array([info['default'] for info in band_info.values()], dtype=dtype)
        # bands of a GeoTIFF normally share a block shape; reads follow the first requested band's
        block_shape = block_shapes[band_ids[0] - 1]
        # per pixel and band: the value in dtype, the value as read before conversion to dtype, and the mask both
        # as GDAL's uint8 and as numpy's bool
        native_itemsize = max(np.dtype(band_dtypes[band_id - 1]).itemsize for band_id in band_ids)
        bytes_per_pixel = len(band_ids) * (dtype.itemsize + native_itemsize + 2)

        def geotiff_lookup(locations):
            row, col, inside = geotiff_pixel_index(inv_transform,
                                                   locations['longitude'].to_numpy(dtype='float64', na_value=np.nan),
                                                   locations['latitude'].to_numpy(dtype='float64', na_value=np.nan),
                                                   height, width)
            res = np.empty((len(locations), len(band_ids)), dtype=dtype)
            res[:] = defaults

            loc_idx = np.flatnonzero(inside)
            if loc_idx.size:
                order, reads = geotiff_read_plan(row[loc_idx], col[loc_idx], block_shape, bytes_per_pixel,
                                                 height, width, GEOTIFF_READ_BUDGET_BYTES)
                loc_idx = loc_idx[order]
                row, col = row[loc_idx], col[loc_idx]
                with rasterio.open(abs_file_path) as dataset:
                    for start, end, window in reads:
                        read_row = row[start:end] - window.row_off
                        read_col = col[start:end] - window.col_off
                        # (band, row, col), masked where the band's nodata value or mask band marks a pixel invalid
                        data = dataset.read(band_ids, window=window, out_dtype=dtype, masked=True)
                        values = data.data[:, read_row, read_col].T
                        invalid = np.ma.getmaskarray(data)[:, read_row, read_col].T
                        res[loc_idx[start:end]] = np.where(invalid, defaults, values)

            for col_i, col_name in enumerate(band_info.keys()):
                locations[col_name] = res[:, col_i]
            return locations

        return geotiff_lookup

    def build_h3(self, resolution, file_path, file_type='csv', **kwargs):
        """Function factory to look up area_peril_id using H3 hexagonal grid indexing.

        Converts latitude/longitude to an H3 cell at the specified resolution,
        converts the cell to its int64 representation, then maps to an int32
        area_peril_id using a mapping file supplied by the model provider.

        The mapping file must contain at least two columns:
            - ``h3_int64``      : H3 cell index as a 64-bit integer
            - ``area_peril_id`` : Oasis area peril ID (int32)

        ``h3_int64`` must be unique: a point falls in exactly one cell at a given resolution, so a
        repeated cell index is a mapping file error and raises ``OasisException`` at build time.
        A mapping file with duplicate cells previously emitted one key per duplicate row, which
        changed the row count of the keys file rather than reporting the problem.

        ``area_peril_id`` must be an exact integer on every row and also raises ``OasisException``
        at build time otherwise. A single null or fractional value makes pandas read the whole
        column as ``float64``, which rounds any other id in the file above 2^53 -- so the failure
        would not be confined to the offending row.

        A location whose coordinates are null, infinite, or absent from the mapping file resolves
        to ``OASIS_UNKNOWN_ID``, which the lookup reports as a per-location ``fail`` status.

        Config example::

            "h3_area_peril": {
                "type": "h3",
                "columns": ["latitude", "longitude"],
                "parameters": {
                    "resolution": 5,
                    "file_path": "%%KEYS_DATA_PATH%%/h3_to_areaperil.csv"
                }
            }

        Args:
            resolution (int): H3 resolution level (0–15). Higher values produce finer cells.
            file_path (str): Path to the int64→area_peril_id mapping file.
                Supports the ``%%KEYS_DATA_PATH%%`` placeholder.
            file_type (str): Pandas read function suffix (``'csv'``, ``'parquet'``, etc.).
                Defaults to ``'csv'``.
            **kwargs: Additional keyword arguments forwarded to the pandas read function.

        Returns:
            function: function assigning an area_peril_id to each location from its latitude and
                longitude, set to OASIS_UNKNOWN_ID where the H3 cell is missing from the mapping
                file or the coordinates are null or non-finite.
        """
        if h3 is None:
            raise OasisException(
                f"h3 module is required for h3 lookups, {OPT_INSTALL_MESSAGE}"
            )

        read_func = getattr(pd, f"read_{file_type}", None)
        if callable(read_func):
            h3_mapping_df = read_func(self.to_abs_filepath(file_path), **kwargs)
        else:
            h3_mapping_df = pd.read_csv(self.to_abs_filepath(file_path), **kwargs)

        h3_mapping_df.columns = [c.lower() for c in h3_mapping_df.columns]
        missing_columns = {'h3_int64', 'area_peril_id'}.difference(h3_mapping_df.columns)
        if missing_columns:
            raise OasisException(
                f"H3 mapping file must contain {sorted(missing_columns)} column(s), found: {list(h3_mapping_df.columns)}"
            )
        h3_mapping_df['h3_int64'] = h3_mapping_df['h3_int64'].astype('int64')

        # a point falls in exactly one cell at a given resolution, so a repeated cell index is a
        # mapping file error; fail here rather than silently emitting a key per duplicate row
        duplicated_cells = h3_mapping_df['h3_int64'].duplicated()
        if duplicated_cells.any():
            raise OasisException(
                f"H3 mapping file must have a unique 'h3_int64' per row, found {duplicated_cells.sum()} "
                f"duplicate(s), starting with {h3_mapping_df.loc[duplicated_cells, 'h3_int64'].iloc[0]}"
            )
        # a null or fractional area_peril_id means a float64 column, and that rounds every other id
        # above 2^53 in the same file; only the offending row itself is caught downstream by
        # set_id_columns, and only because a nan cast to int64 happens to land on a value it treats
        # as unknown. fail on the whole class here instead
        area_peril_ids = h3_mapping_df['area_peril_id']
        if pd.api.types.is_integer_dtype(area_peril_ids.dtype):
            unusable = area_peril_ids.isna()
        else:
            numeric_ids = pd.to_numeric(area_peril_ids, errors='coerce')
            unusable = numeric_ids.isna() | (numeric_ids % 1 != 0) | (numeric_ids.abs() >= 2 ** 53)
        if unusable.any():
            raise OasisException(
                f"H3 mapping file must have an exact integer 'area_peril_id' per row, found {unusable.sum()} "
                f"value(s) that are null, fractional or beyond the exactly representable range of the "
                f"'{area_peril_ids.dtype}' column, starting with "
                f"'{area_peril_ids[unusable].iloc[0]}' at cell {h3_mapping_df.loc[unusable, 'h3_int64'].iloc[0]}"
            )
        area_peril_by_cell = h3_mapping_df.set_index('h3_int64')['area_peril_id'].astype('int64')

        def h3_lookup(locations):
            latitude = locations['latitude'].to_numpy(dtype='float64', na_value=np.nan)
            longitude = locations['longitude'].to_numpy(dtype='float64', na_value=np.nan)
            # isfinite rather than ~isnan: an infinite coordinate would survive an isnan test, and
            # 1j * inf has a nan real part, so it would reach factorize as an NA and take its -1
            # sentinel, which indexes back from the end of the mapped array and silently returns
            # another location's area_peril_id. infinite coordinates are unknown, like null ones.
            valid = np.isfinite(latitude) & np.isfinite(longitude)

            # int64 throughout: a float64 intermediate would round area_peril_id above 2^53, which
            # an h3 model keyed on the cell index itself (~6e17) would hit on every row
            area_peril_id = np.full(len(locations), OASIS_UNKNOWN_ID, dtype='int64')
            if valid.any():
                # h3 has no array api, so the conversion stays a python level loop; factorising the
                # points first means it is paid once per distinct coordinate rather than once per
                # location. the complex encoding is only a way to hash the (latitude, longitude)
                # pair in one go, and is safe here because valid guarantees both parts are finite.
                codes, points = pd.factorize(latitude[valid] + 1j * longitude[valid], sort=False)
                cells = np.fromiter(
                    (h3_int.latlng_to_cell(point.real, point.imag, resolution) for point in points.tolist()),
                    dtype='int64', count=points.size,
                )
                # resolving the mapping per distinct cell keeps the join off the per location path
                area_peril_id[valid] = area_peril_by_cell.reindex(
                    cells, fill_value=OASIS_UNKNOWN_ID).to_numpy(dtype='int64')[codes]

            locations['area_peril_id'] = area_peril_id
            locations.reset_index(drop=True, inplace=True)  # as the left join this replaces used to
            return self.set_id_columns(locations, ['area_peril_id'])

        return h3_lookup

    def build_merge(self, file_path, id_columns=[], file_type='csv', join_columns=None, **kwargs):
        """Merges the locations Dataframe with the Dataframe present in file_path
        All non match column present in id_columns will be set to -1

        this is an efficient way to map a combination of column that have a finite scope to an idea.

        By default the join is on every column name the file and the locations have in common
        (matched case-insensitively). Set ``join_columns`` to join only on those columns instead,
        e.g. to reuse a file that carries a column (such as ``CountryCode``) the locations also
        have but that must not be matched on. With ``join_columns``, any other column in both
        the file and the locations keeps its locations value, except ``id_columns``, which take
        the file's value.

        If the join value is held in an OED GeogName column by scheme, add a 'geog_lookup' step first.

        Memory: the file is read once per lookup, and indexed by its join columns. When keys are
        generated with several processes, it is read in the parent process before the workers
        are forked (see multiproc_shared_data) and the workers share that copy. Each chunk of
        locations is merged only with the rows its join keys can match, so it doesn't re-process
        the whole file.
        """
        abs_path, file_type, read_kwargs, join_columns = self._merge_settings(
            file_path, id_columns, file_type, join_columns, **kwargs)
        table = _SHARED_STEP_DATA.get(_shared_data_key('merge', abs_path, file_type, read_kwargs))
        if table is None:
            table = self._read_merge_table(abs_path, file_type, read_kwargs)
        else:
            logger.info("merge step: using the table of %s already loaded by the parent process", abs_path)
        df_to_merge = table.df

        if join_columns is not None:
            missing_in_file = [col for col in join_columns if col not in df_to_merge.columns]
            if missing_in_file:
                raise OasisException(
                    f"merge step: join_columns {missing_in_file} not found in '{file_path}'. "
                    f"Table columns: {sorted(df_to_merge.columns)}")
            id_columns_lower = {col.lower() for col in id_columns}

        def merge(locations: pd.DataFrame):
            rename_map = {col.lower(): col for col in locations.columns if col.lower() in df_to_merge.columns}
            if not rename_map:
                raise OasisException(
                    f"merge step: nothing to join on. '{file_path}' and the locations share no column "
                    f"name (matched case-insensitively). Table columns: {sorted(df_to_merge.columns)}. "
                    f"Location columns available to this step: {sorted(locations.columns)}. The join keys "
                    f"are the names the two have in common.")
            if join_columns is None:
                logger.debug("merge step: joining on %s", sorted(rename_map.values()))
                file_keys = tuple(sorted(rename_map))
                rows = table.rows_matching(file_keys, locations[[rename_map[col] for col in file_keys]])
                logger.debug("merge step: %d locations, %d of the table's %d rows can match",
                             len(locations), len(rows), len(df_to_merge))
                locations = locations.merge(rows.rename(columns=rename_map), how='left')
                return self.set_id_columns(locations, id_columns)

            missing_in_locations = [col for col in join_columns if col not in rename_map]
            if missing_in_locations:
                raise OasisException(
                    f"merge step: join_columns {missing_in_locations} not found in the locations. "
                    f"Location columns available to this step: {sorted(locations.columns)}")
            # columns in both that aren't join keys: id_columns come from the file, the rest stay as in locations
            shared = [col for col in rename_map if col not in join_columns]
            from_file = [col for col in shared if col in id_columns_lower]
            from_locations = [col for col in shared if col not in id_columns_lower]
            on = [rename_map[col] for col in join_columns]
            logger.debug("merge step: joining on %s", on)
            file_keys = tuple(sorted(join_columns))
            rows = table.rows_matching(file_keys, locations[[rename_map[col] for col in file_keys]])
            logger.debug("merge step: %d locations, %d of the table's %d rows can match",
                         len(locations), len(rows), len(df_to_merge))
            locations = locations.drop(columns=[rename_map[col] for col in from_file]).merge(
                rows.drop(columns=from_locations).rename(columns=rename_map), on=on, how='left')
            return self.set_id_columns(locations, id_columns)
        return merge

    def _merge_settings(self, file_path, id_columns=[], file_type='csv', join_columns=None, **kwargs):
        """Resolve build_merge's parameters.

        Returns:
            (abs_path, file_type, read_kwargs, join_columns): join_columns lower-cased, or None.
        """
        if isinstance(join_columns, str):
            join_columns = [join_columns]
        if join_columns is not None:
            join_columns = [col.lower() for col in join_columns]
        return self.to_abs_filepath(file_path), file_type, kwargs, join_columns

    @staticmethod
    def _read_merge_table(abs_path, file_type, read_kwargs):
        """Read a merge step's file with pandas.read_<file_type> (read_csv if there is none), lower-casing its columns."""
        start = time.perf_counter()
        read_func = getattr(pd, f"read_{file_type}", None)
        if callable(read_func):
            df_to_merge = read_func(abs_path, **read_kwargs)
        else:
            df_to_merge = pd.read_csv(abs_path, **read_kwargs)
        df_to_merge = df_to_merge.rename(columns={column: column.lower() for column in df_to_merge.columns})
        logger.info("merge step: read %d rows with columns %s from %s in %.1fs, process memory %s",
                    len(df_to_merge), list(df_to_merge.columns), abs_path, time.perf_counter() - start,
                    _memory_usage())
        return _MergeTable(df_to_merge)

    def build_sorted_lookup(self, file_path, value_columns=None, id_columns=()):
        """Look up one integer key column in a large table stored as sorted, memory-mapped arrays.

        For a table too large for ``merge`` (e.g. hundreds of millions of grid cell -> area peril
        rows), which reads the whole file into every lookup process. ``file_path`` is a directory
        written by :func:`write_sorted_lookup`: one ``.npy`` array of unique keys in ascending order,
        one ``.npy`` array per value column, and a ``sorted_lookup.json`` describing them. The arrays
        are opened with ``np.load(..., mmap_mode='r')``, so the OS page cache holds one copy shared
        by every process, and each chunk of locations is matched with ``np.searchsorted``.

        The key column named in ``sorted_lookup.json`` must be in the locations (list it in the
        step's ``columns``). Locations whose key is null, not an integer, or not in the table get
        null values, and -1 in ``id_columns``.

        Args:
            file_path (str): directory written by ``write_sorted_lookup``; supports placeholders.
            value_columns (list, None): value columns to add to the locations (default: all).
            id_columns (list): columns to transform to an id column (Int64 with nulls replaced by -1).
        """
        directory = self.to_abs_filepath(file_path)
        keys, values, key_column = load_sorted_lookup(directory, value_columns)
        if isinstance(id_columns, str):
            id_columns = [id_columns]
        unknown_ids = [col for col in id_columns if col not in values]
        if unknown_ids:
            raise OasisException(
                f"sorted_lookup step: id_columns {unknown_ids} are not value columns of '{file_path}' "
                f"({sorted(values)})")

        def sorted_lookup(locations):
            if key_column not in locations.columns:
                raise OasisException(
                    f"sorted_lookup step: key column '{key_column}' of '{file_path}' is not in the locations. "
                    f"Location columns available to this step: {sorted(locations.columns)}")
            query, valid = integer_keys(locations[key_column], keys.dtype)
            position = np.searchsorted(keys, query[valid])
            in_range = position < len(keys)
            found_in_valid = np.zeros(len(position), dtype=bool)
            found_in_valid[in_range] = keys[position[in_range]] == query[valid][in_range]
            rows = np.flatnonzero(valid)[found_in_valid]
            position = position[found_in_valid]
            logger.debug("sorted_lookup(%s): matched %d/%d locations", file_path, len(rows), len(locations))
            for column, array in values.items():
                locations[column] = masked_take(array, position, rows, len(locations))
            return self.set_id_columns(locations, id_columns)
        return sorted_lookup

    @staticmethod
    def build_transform_coordinates(target_crs, source_crs='EPSG:4326', x_column='x', y_column='y',
                                    latitude_column='latitude', longitude_column='longitude', area_of_use_only=True):
        """Project the locations' latitude/longitude into another coordinate reference system.

        Writes the projected coordinates to ``x_column`` and ``y_column`` (easting and northing for
        most projected CRSs), so a following step can index a grid defined in that CRS, e.g. one
        ``interval_to_index`` step per axis. Uses ``pyproj`` with ``always_xy``, so ``x`` is the
        longitude-like axis whatever the CRS's own axis order.

        Locations with a null latitude or longitude, or that can't be projected, get NaN in both
        columns. pyproj projects points far outside a CRS's area of use without complaint, so by
        default locations outside the target CRS's area of use (as published with the CRS, e.g.
        Great Britain for EPSG:27700) also get NaN.

        Args:
            target_crs (str, int): CRS to project to, in any form ``pyproj.CRS`` accepts
                (``"EPSG:27700"``, ``27700``, a PROJ string or WKT).
            source_crs (str, int): CRS of the latitude/longitude columns (default WGS 84).
            x_column (str): output column for the x coordinate.
            y_column (str): output column for the y coordinate.
            latitude_column (str): input latitude column.
            longitude_column (str): input longitude column.
            area_of_use_only (bool): give NaN to locations outside the target CRS's area of use.
                Only applies when ``source_crs`` is geographic and the target CRS has an area of use.
        """
        if pyproj is None:
            raise OasisException(f"pyproj module is needed for transform_coordinates, {OPT_INSTALL_MESSAGE}")
        try:
            source, target = pyproj.CRS.from_user_input(source_crs), pyproj.CRS.from_user_input(target_crs)
        except pyproj.exceptions.CRSError as error:
            raise OasisException(f"transform_coordinates step: invalid CRS: {error}") from error
        bounds = None
        if area_of_use_only:
            if not source.is_geographic:
                logger.info(f"transform_coordinates step: area_of_use_only ignored, source CRS {source_crs} is not geographic")
            elif target.area_of_use is None:
                logger.info(f"transform_coordinates step: area_of_use_only ignored, target CRS {target_crs} has no area of use")
            else:
                bounds = target.area_of_use.bounds  # (west, south, east, north) in degrees
        transformers = {}  # one per process: a pyproj Transformer is not safe to share across a fork

        def transform_coordinates(locations):
            for column in (latitude_column, longitude_column):
                if column not in locations.columns:
                    raise OasisException(
                        f"transform_coordinates step: column '{column}' is not in the locations. "
                        f"Location columns available to this step: {sorted(locations.columns)}")
            transformer = transformers.get(os.getpid())
            if transformer is None:
                transformer = transformers[os.getpid()] = pyproj.Transformer.from_crs(source, target, always_xy=True)
            lon = pd.to_numeric(locations[longitude_column], errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
            lat = pd.to_numeric(locations[latitude_column], errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
            x, y = transformer.transform(lon, lat, errcheck=False)
            x, y = np.asarray(x, dtype='float64'), np.asarray(y, dtype='float64')
            missing = np.isnan(lon) | np.isnan(lat)
            not_projected = ~missing & ~(np.isfinite(x) & np.isfinite(y))
            out_of_area = np.zeros(len(locations), dtype=bool)
            if bounds is not None:
                west, south, east, north = bounds
                # an area crossing the antimeridian has west > east
                in_lon = (lon >= west) & (lon <= east) if west <= east else (lon >= west) | (lon <= east)
                out_of_area = ~missing & ~not_projected & ~(in_lon & (lat >= south) & (lat <= north))
            failed = missing | not_projected | out_of_area
            x[failed] = np.nan
            y[failed] = np.nan
            locations[x_column] = x
            locations[y_column] = y
            if 'message' in locations.columns and failed.any():
                # explain why x/y are null; the status is set later, when the id they lead to is missing
                for mask, reason in ((missing, f"missing {latitude_column} or {longitude_column}"),
                                     (not_projected, f"could not be projected to {target_crs}"),
                                     (out_of_area, f"outside the area of use of {target_crs}")):
                    mask = mask & (locations['message'] == '').to_numpy()
                    if mask.any():
                        locations.loc[mask, 'message'] = f"transform_coordinates failure: {reason}"
            return locations
        return transform_coordinates

    def build_geog_lookup(self, geog_scheme, output_column, slots=GEOG_LOOKUP_DEFAULT_SLOTS,
                          case_insensitive=True, on_missing='null'):
        """Resolve an OED scheme-tagged value into a single column so it can be joined on.

        OED stores a location code in a pair of columns ``GeogScheme{N}`` (the scheme, e.g.
        ``W3W``) and ``GeogName{N}`` (the value), for ``N`` in ``1..slots``. The scheme of
        interest can appear in any slot, so there is no fixed column a ``merge`` step can join
        on. This step scans the pairs and, where ``GeogScheme{N}`` matches ``geog_scheme``,
        copies the paired ``GeogName{N}`` into ``output_column``. A subsequent ``merge`` step
        can then join on ``output_column``.

        The lowest-numbered matching slot wins. Rows with no matching scheme get a null in
        ``output_column`` (``on_missing='null'``) or raise (``on_missing='error'``).

        The ``GeogScheme{N}``/``GeogName{N}`` columns for ``N`` in ``1..slots`` are kept
        automatically and are optional, so they don't need to be listed in the step's
        ``columns``. Listing one there makes it required.

        Args:
            geog_scheme (str): the OED GeogScheme value to resolve (e.g. ``"W3W"``).
            output_column (str): the column to write the resolved value into.
            slots (int): highest ``GeogScheme{N}``/``GeogName{N}`` index to scan (default 30).
            case_insensitive (bool): match the scheme ignoring case and surrounding whitespace.
            on_missing (str): ``'null'`` to leave unmatched rows null, ``'error'`` to raise.
        """
        if on_missing not in ('null', 'error'):
            raise OasisException(f"build_geog_lookup: on_missing must be 'null' or 'error', got {on_missing!r}")
        target = geog_scheme.upper() if case_insensitive else geog_scheme

        def geog_lookup(locations):
            col_map = {col.lower(): col for col in locations.columns}
            resolved = pd.Series(pd.NA, index=locations.index, dtype='object')
            for n in range(1, slots + 1):
                scheme_col, name_col = col_map.get(f"geogscheme{n}"), col_map.get(f"geogname{n}")
                if scheme_col is None or name_col is None:
                    continue
                scheme_vals = locations[scheme_col].astype('string').str.strip()
                if case_insensitive:
                    scheme_vals = scheme_vals.str.upper()
                # empty slots (NaN scheme) compare to NA; fillna(False) keeps the
                # mask a plain boolean so .loc never sees NA (sparse slots are the
                # common OED shape).
                hit = ((scheme_vals == target) & resolved.isna()).fillna(False)
                resolved.loc[hit] = locations.loc[hit, name_col]
            n_resolved = int(resolved.notna().sum())
            logger.info("geog_lookup(%s): resolved %d/%d locations into '%s'",
                        geog_scheme, n_resolved, len(locations), output_column)
            if on_missing == 'error' and n_resolved < len(locations):
                raise OasisException(
                    f"geog_lookup: {len(locations) - n_resolved} location(s) have no "
                    f"GeogScheme == '{geog_scheme}' in slots 1..{slots}")
            locations[output_column] = resolved
            return locations
        return geog_lookup

    @staticmethod
    def build_simple_pivot(pivots, remove_pivoted_col=True):
        """Allow to pivot columns of the locations dataframe into multiple rows

        each pivot in the pivot list may define:

        - **on**: to rename a column into a new one
        - **new_cols**: to create a new column with a certain values

        For example, with::

            "pivots": [{"on": {"vuln_str": "vulnerability_id"},
                        "new_cols": {"coverage_type": 1}},
                       {"on": {"vuln_con": "vulnerability_id"},
                        "new_cols": {"coverage_type": 3}}]

        the input::

            loc_id  vuln_str    vuln_con
            1       3           2
            2       18          4

        becomes::

            loc_id  vuln_str    vuln_con    vulnerability_id    coverage_type
            1       3           2           3                   1
            2       18          4           18                  1
            1       3           2           2                   3
            2       18          4           4                   3
        """
        def simple_pivot(locations):
            pivoted_dfs = []
            pivoted_cols = set()
            for pivot in pivots:
                pivot_df = locations.copy()
                for old_name, new_name in pivot.get("on", {}).items():
                    pivot_df[new_name] = pivot_df[old_name]
                    pivoted_cols.add(old_name)
                for col_name, value in pivot.get("new_cols", {}).items():
                    pivot_df[col_name] = value
                pivoted_dfs.append(pivot_df)
            locations = pd.concat(pivoted_dfs, ignore_index=True)
            if remove_pivoted_col:
                locations.drop(columns=pivoted_cols, inplace=True)
            return locations

        return simple_pivot

    @staticmethod
    def build_model_data(columns):
        """Serialises specified columns from the OED file into a model_data dict"""
        def model_data(locations):
            locations['model_data'] = locations[columns].to_dict('records')
            return locations

        return model_data

    @staticmethod
    def build_dynamic_model_adjustment(intensity_adjustment_col, return_period_col):
        """Converts specified columns from the OED file into intensity adjustments and
        return period protection.
        """
        def adjustments(locations):
            locations['intensity_adjustment'] = locations[intensity_adjustment_col].values
            locations['return_period'] = locations[return_period_col].values
            return locations

        return adjustments
