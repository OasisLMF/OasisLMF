"""
Module for the built-in Lookup Class

in the future we may want to improve on the management of files used to generate the keys
tutorial for pandas and parquet https://towardsdatascience.com/a-gentle-introduction-to-apache-arrow-with-apache-spark-and-pandas-bb19ffe0ddae

"""
import warnings

import numba as nb
import numpy as np
import pandas as pd
from ods_tools.oed import fill_empty, is_empty

try:  # needed for rtree
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
    Point = gdp = None

try:  # needed for geotiff
    from osgeo import gdal
except ImportError:
    gdal = None

import math
import re

from oasislmf.lookup.base import AbstractBasicKeyLookup, MultiprocLookupMixin
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.peril import get_peril_groups_df
from oasislmf.utils.status import OASIS_KEYS_STATUS, OASIS_UNKNOWN_ID

OPT_INSTALL_MESSAGE = "install oasislmf with extra packages by running 'pip install oasislmf[extra]'"


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

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
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

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


# GeoTransform indexes, https://gdal.org/en/stable/tutorials/geotransforms_tut.html
X_COORDINATE = 0  # x-coordinate of the upper-left corner of the upper-left pixel.
W_E_PIXEL_RESOLUTION = 1  # w-e pixel resolution / pixel width.
ROW_ROTATION = 2  # row rotation (typically zero).
Y_COORDINATE = 3  # y-coordinate of the upper-left corner of the upper-left pixel.
COLUMN_ROTATION = 4  # column rotation (typically zero).
N_S_PIXEL_RESOLUTION = 5  # n-s pixel resolution / pixel height (negative value for a north-up image).


@nb.njit(cache=True)
def jit_gda_loc_to_val(ds_array, inv_gt, x_array, y_array, useful_array_idx, defaults, res):
    for loc_i in range(res.shape[0]):
        px = int(inv_gt[X_COORDINATE] + x_array[loc_i] * inv_gt[W_E_PIXEL_RESOLUTION] + y_array[loc_i] * inv_gt[ROW_ROTATION] + 0.5)
        py = int(inv_gt[Y_COORDINATE] + x_array[loc_i] * inv_gt[COLUMN_ROTATION] + y_array[loc_i] * inv_gt[N_S_PIXEL_RESOLUTION] + 0.5)
        if 0 <= px < ds_array.shape[1] and 0 <= py < ds_array.shape[0]:
            for col_i in range(useful_array_idx.shape[0]):
                res[loc_i, col_i] = ds_array[py, px, useful_array_idx[col_i]]
        else:
            for col_i in range(useful_array_idx.shape[0]):
                res[loc_i, col_i] = defaults[col_i]
    return res


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
    """
    Returns the grid size using the max and min long and latitude and arc size
    """
    length = round((grid["lon_max"] - grid["lon_min"]) / grid["arc_size"])
    width = round((grid["lat_max"] - grid["lat_min"]) / grid["arc_size"])
    return length * width


key_columns = ['loc_id', 'peril_id', 'coverage_type', 'area_peril_id', 'vulnerability_id', 'status', 'message']


class PerilCoveredDeterministicLookup(AbstractBasicKeyLookup):
    multiproc_enabled = False

    def process_locations(self, locations):
        peril_groups_df = get_peril_groups_df()
        model_perils_covered = np.unique(pd.DataFrame({'peril_group_id': self.config['model_perils_covered']})
                                         .merge(peril_groups_df)['peril_id'])
        peril_covered_column = 'LocPerilsCovered' if 'LocPerilsCovered' in locations.columns else 'PolPerilsCovered'

        locations['peril_group_id'] = locations[peril_covered_column].str.split(';')
        keys_df = locations.explode('peril_group_id').drop_duplicates().merge(peril_groups_df)[['loc_id', 'peril_id']]
        locations.drop(columns='peril_group_id')

        coverage_df = pd.DataFrame({'coverage_type': self.config['supported_oed_coverage_types']}, dtype='Int32')
        keys_df = keys_df.sort_values('loc_id', kind='stable').merge(coverage_df, how="cross")
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
    """
    Built-in Lookup class that implement the OasisLookupInterface
    The aim of this class is to provide a data driven lookup capability that will be both flexible and efficient.

    it provide several generic function factory that can be define in the config under the "step_definition" key (ex:)
    "step_definition": {
        "split_loc_perils_covered":{
            "type": "split_loc_perils_covered" ,
            "columns": ["locperilscovered"],
            "parameters": {
                "model_perils_covered": ["WTC", "WSS"]
            }
        },
        "vulnerability": {
            "type": "merge",
            "columns": ["peril_id", "coverage_type", "occupancycode"],
            "parameters": {"file_path": "%%KEYS_DATA_PATH%%/vulnerability_dict.csv",
                           "id_columns": ["vulnerability_id"]
                          }
        }
    }
    mapper key: is called the step_name,
        it will be added the the lookup object method once the function has been built
        it can take any value but make sure it doesn't collide with already existing method
    type: define the function factory to call.
        in the class for type <fct_type> the function factory called will be build_<fct_type>
        ex: "type": "merge" => build_merge
    columns: are the column required to be able to apply the step.
        those are quite important as any column (except 'loc_id')
        from the original Locations Dataframe that is not in any step will be drop to reduce memory consumption
    parameters: the parameter passed the the function factory.

    Once all the functions have been defined, the order in which they must be applied is defined in the config
    under the "strategy" key (ex:)
        "strategy": ["split_loc_perils_covered", "vulnerability"]

    It is totally possible to subclass Lookup in order to create your custom step or function factory
    for custom step:
        add your function definition to the "mapper"with no parameters
    "my_custom_step": {
            "type": "custom_type" ,
            "columns": [...],
    }
    simply add it to your "strategy": ["split_loc_perils_covered", "vulnerability", "my_custom_step"]
    and code the function in your subclass
    class MyLookup(Lookup):
        @staticmethod
        def my_custom_step(locations):
            <do something on locations>
            return modified_locations

    for function factory:
    add your function definition to the "step_definition" with the required parameters
    "my_custom_step": {
            "type": "custom_type" ,
            "columns": [...],
            "parameters": {
                "param1": "value1"
            }
    }
    add your step to "strategy": ["split_loc_perils_covered", "vulnerability", "my_custom_step"]
    and code the function factory in your subclass
    class MyLookup(Lookup):
        def build_custom_type(self, param1):
            def fct(locations):
                <do something on locations that depend on param1>
                return modified_locations

            return fct

    """
    interface_version = "1"

    def set_step_function(self, step_name, step_config, function_being_set=None):
        """
        set the step as a function of the lookup object if it's not already done and return it.
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
                    child_fct = self.set_step_function(
                        step_name=child_step_name,
                        step_config=self.config['step_definition'][child_step_name],
                        function_being_set=function_being_set)
                    functions.append({'function': child_fct, 'columns': set(step_config.get("columns", []))})
                step_config['parameters']['strategy'] = functions

            step_function = getattr(self, f"build_{step_config['type']}")(**step_config['parameters'])
            setattr(self, step_name, step_function)
        return step_function

    def process_locations(self, locations):
        # drop all unused columns and remove duplicate rows, find and rename useful columns
        lower_case_column_map = {column.lower(): column for column in locations.columns}
        useful_cols = set(['loc_id'] + sum((step_config.get("columns", []) for step_config in self.config['step_definition'].values()), []))

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
        self.set_id_columns(locations, id_cols)
        # check all success location have all ids set correctly
        success_locations = locations.loc[locations['status'] == OASIS_KEYS_STATUS['success']['id']]
        for id_col in id_cols:
            unknown_ids = success_locations[id_col] == OASIS_UNKNOWN_ID
            fail_locations = success_locations.loc[unknown_ids].index
            locations.loc[fail_locations, ['status', 'message']] = OASIS_KEYS_STATUS['fail'][
                'id'], f'{id_col} has an unknown id'
            success_locations = success_locations.loc[~unknown_ids]

        return locations

    def to_abs_filepath(self, filepath):
        """
        replace placeholder r'%%(.+?)%%' (ex: %%KEYS_DATA_PATH%%) with the path set in self.config
        Args:
            filepath (str): filepath with potentially a placeholder

        Returns:
            str: filepath where placeholder are replace their actual value.
        """
        placeholder_keys = set(re.findall(r'%%(.+?)%%', filepath))
        for placeholder_key in placeholder_keys:
            filepath = filepath.replace(f'%%{placeholder_key}%%', self.config[placeholder_key.lower()])
        return filepath

    @staticmethod
    def set_id_columns(df, id_columns):
        """
        in Dataframes, only float column can have nan values. So after a left join for example if you have nan values
        that will change the type of the original column into float.
        this function replace the nan value with the OASIS_UNKNOWN_ID and reset the column type to int
        """
        for col in id_columns:
            if col not in df:
                df[col] = OASIS_UNKNOWN_ID
            else:
                df[col] = df[col].astype('Int64')
                df.loc[(df[col].isna()) | (df[col] <= 0), col] = OASIS_UNKNOWN_ID
        return df

    def build_interval_to_index(self, value_column_name, sorted_array, index_column_name=None, side='left'):
        """
        Allow to map a value column to an index according to it's index in the interval defined by sorted_array.
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
        """
        build a function that will combine several strategy trying to achieve the same purpose by different mean into one.
        for example, finding the correct area_peril_id for a location with one method using (latitude, longitude)
        and one using postcode.
        each strategy will be applied sequentially on the location that steal have OASIS_UNKNOWN_ID in their id_columns after the precedent strategy

        'or' example: (note: "id_columns" is a list)
            "vulnerability":{
                "type": "combine",
                "parameters": {
                    "id_columns": ["vulnerability_id"],
                    "strategy": ["vuln_cov_Building_Content", "vuln_cov_car"]
                    "logical_type": "or"
                }
            }

        'and' example: (note: that "id_columns" is a list of list)
            "vuln_cov_car":{
                "type": "combine",
                "columns": ["autocode"],
                "parameters": {
                    "id_columns": [["vuln_id_car"], ["vulnerability_id"]],
                    "strategy": ["vulnerability_car", "coverage_type_car"],
                    "logical_type": "and"
                }
            },

        Args:
            id_columns (list): columns that will be checked to determine if a strategy has succeeded
            strategy (list): list of strategy to apply
            logical_type: if 'or' apply the next strategy only on invalid id_columns
                          if 'and' apply the next strategy only on valid id_columns
                                   id_columns needs to be a list of list of columns that each sublist is checked sequentially


        Returns:
            function: function combining all strategies
        """
        if logical_type.lower() == 'or':
            def fct(locations):
                initial_columns = locations.columns
                result = []
                for child_strategy in strategy:
                    if not child_strategy['columns'].issubset(locations.columns):  # needed column not present to run this strategy
                        continue
                    locations = child_strategy['function'](locations)
                    locations = Lookup.set_id_columns(locations, id_columns)
                    is_valid = (locations[id_columns] != OASIS_UNKNOWN_ID).any(axis=1)
                    result.append(locations[is_valid])
                    locations = locations[~is_valid][initial_columns]
                    if locations.empty:
                        break
                result.append(locations)
                return Lookup.set_id_columns(pd.concat(result, ignore_index=True), id_columns)

        elif logical_type.lower() == 'and':
            def fct(locations):
                initial_columns = locations.columns
                result = []
                for i, child_strategy in enumerate(strategy):
                    if not child_strategy['columns'].issubset(locations.columns):  # needed column not present to run this strategy
                        continue
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
    def build_split_loc_perils_covered(model_perils_covered=None):
        """
        split the value of LocPerilsCovered into multiple line, taking peril group into account
        drop all line that are not in the list model_perils_covered

        usefull inspirational code:
        https://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-rows

        """
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

            locations['peril_group_id'] = locations[perils_covered_column].str.split(';')
            peril_locations = locations.explode('peril_group_id').drop_duplicates().merge(peril_groups_df)
            locations.drop(columns='peril_group_id')

            if model_perils_covered:
                df_model_perils_covered = pd.Series(model_perils_covered)
                df_model_perils_covered.name = 'model_perils_covered'
                peril_locations = peril_locations.merge(df_model_perils_covered,
                                                        left_on='peril_id', right_on='model_perils_covered',
                                                        sort=True)
            not_covered_location = locations[~locations['loc_id'].isin(peril_locations['loc_id'])]
            if not not_covered_location.empty:
                not_covered_location['status'] = OASIS_KEYS_STATUS['notatrisk']
                not_covered_location['message'] = not_covered_location[perils_covered_column].astype(str) + " have no perils modelled"
                peril_locations = pd.concat([peril_locations, not_covered_location], ignore_index=True)
            return peril_locations
        return fct

    @staticmethod
    def build_prepare(**kwargs):
        """
        Prepare the dataframe by setting default, min and max values and type
        support several simple DataFrame preparation:
            default: create the column if missing and replace the nan value with the default value
            max: truncate the values in a column to the specified max
            min: truncate the values in a column to the specified min
            type: convert the type of the column to the specified numpy dtype
                Note that we use the string representation of numpy dtype available at
                https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
        """
        def prepare(locations):
            for column_name, preparations in kwargs.items():
                if "default" in preparations:
                    if column_name not in locations.columns:
                        locations[column_name] = preparations["default"]
                    else:
                        fill_empty(locations, column_name, preparations["default"])
                if 'max' in preparations:
                    locations.loc[locations[column_name] > preparations['max'], column_name] = preparations['max']
                if 'min' in preparations:
                    locations.loc[locations[column_name] > preparations['min'], column_name] = preparations['min']
                if 'type' in preparations:
                    locations[column_name] = locations[column_name].astype(preparations['type'])
            return locations
        return prepare

    def build_rtree(self, file_path, file_type, id_columns, area_peril_read_params=None, nearest_neighbor_min_distance=-1):
        """
        Function Factory to associate location to area_peril based on the rtree method

        !!!
        please note that this method is quite time consuming (specialy if you use the nearest point option
        if your peril_area are square you should use area_peril function fixed_size_geo_grid
        !!!

        file_path: is the path to the file containing the area_peril_dictionary.
            this file must be a geopandas Dataframe with a valid geometry.
            an example on how to create such dataframe is available in PiWind
            if you are new to geo data (in python) and want to learn more, you may have a look at this excellent course:
            https://automating-gis-processes.github.io/site/index.html

        file_type: can be any format readable by geopandas ('file', 'parquet', ...)
            see: https://geopandas.readthedocs.io/en/latest/docs/reference/io.html
            you may have to install additional library such as pyarrow for parquet

        id_columns: column to transform to an 'id_column' (type int32 with nan replace by -1)

        nearest_neighbor_min_distance: option to compute the nearest point if intersection method fails
            we use:
            https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
            but alternatives can be found here:
            https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe

        """
        if Point is None:
            raise OasisException(f"shapely and geopandas modules are needed for rtree, {OPT_INSTALL_MESSAGE}")

        if hasattr(gpd, f"read_{file_type}"):
            if area_peril_read_params is None:
                area_peril_read_params = {}
            gdf_area_peril = getattr(gpd, f"read_{file_type}")(self.to_abs_filepath(file_path), **area_peril_read_params)
        else:
            raise OasisException(f"Unregognised Geopandas read type {file_type}")

        if nearest_neighbor_min_distance > 0:
            if BallTree is None:
                raise OasisException(f"scikit-learn modules are needed for rtree with nearest_neighbor_min_distance, {OPT_INSTALL_MESSAGE}")
            gdf_area_peril['center'] = gdf_area_peril.centroid
            base_geometry_name = gdf_area_peril.geometry.name

        def get_area(locations, gdf_area_peril):
            # this conversion could be done in a separate step allowing more posibilities for the geometry
            null_gdf = locations["longitude"].isna() | locations["latitude"].isna()
            null_gdf_loc = locations[null_gdf]
            if not null_gdf_loc.empty:
                gdf_loc = gpd.GeoDataFrame(locations[~null_gdf], columns=locations.columns)
            else:
                gdf_loc = gpd.GeoDataFrame(locations, columns=locations.columns)

            if not gdf_loc.empty:
                gdf_loc["loc_geometry"] = gdf_loc.apply(lambda row: Point(row["longitude"], row["latitude"]),
                                                        axis=1,
                                                        result_type='reduce')
                gdf_loc = gdf_loc.set_geometry('loc_geometry')

                gdf_loc = gpd.sjoin(gdf_loc, gdf_area_peril, 'left')

                if nearest_neighbor_min_distance > 0:
                    gdf_loc_na = gdf_loc.loc[gdf_loc['index_right'].isna()]

                    if gdf_loc_na.shape[0]:
                        gdf_area_peril.set_geometry('center', inplace=True)
                        nearest_neighbor_df = nearest_neighbor(gdf_loc_na, gdf_area_peril, return_dist=True)

                        gdf_area_peril.set_geometry(base_geometry_name, inplace=True)
                        valid_nearest_neighbor = nearest_neighbor_df['distance'] <= nearest_neighbor_min_distance
                        common_col = list(set(gdf_loc_na.columns) & set(nearest_neighbor_df.columns))
                        gdf_loc.loc[valid_nearest_neighbor.index, common_col] = nearest_neighbor_df.loc[valid_nearest_neighbor, common_col]
            if not null_gdf_loc.empty:
                gdf_loc = pd.concat([gdf_loc, null_gdf_loc])
            self.set_id_columns(gdf_loc, id_columns)

            # index column are created during the sjoin, we can drop them
            gdf_loc = gdf_loc.drop(columns=['index_right', 'index_left'], errors='ignore')

            return gdf_loc

        def fct(locations):
            if 'peril_id' in gdf_area_peril.columns:
                peril_id_covered = np.unique(gdf_area_peril['peril_id'])
                res = [locations[~locations['peril_id'].isin(peril_id_covered)]]
                for peril_id in peril_id_covered:
                    res.append(get_area(locations.loc[locations['peril_id'] == peril_id],
                                        gdf_area_peril.loc[gdf_area_peril['peril_id'] == peril_id].drop(columns=['peril_id'])))

                return pd.concat(res).reset_index()
            else:
                return get_area(locations, gdf_area_peril)
        return fct

    @staticmethod
    def build_fixed_size_geo_grid_multi_peril(perils_dict):
        """
        Create multiple grids of varying resolution, one per peril, and
        associate an id to each square of the grid using the
        `fixed_size_geo_grid` method.

        Parameters
        ----------
        perils_dict: dict
                     Dictionary with `peril_id` as key and `fixed_size_geo_grid` parameter dict as
                     value. i.e `{'peril_id' : {fixed_size_geo_grid parameters}}`
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
        """
        associate an id to each square of the grid define by the limit of lat and lon
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
        """
        Create multiple grids of varying resolution, one per peril, and associate an id to each square of the grid using the
        `fixed_size_z_index_geo_grid` method.

        Parameters
        ----------
        perils_dict: dict
                     Dictionary with `peril_id` as key and `fixed_size_geo_grid` parameter dict as
                     value. i.e `{'peril_id' : {fixed_size_geo_grid parameters}}`
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
        """
        associate an id to each square of the grid defined by z-order indexing.
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
        """

        Args:
            file_path: path to the geotiff file
            band_info: a dict where keys are assigned column name, and values are dicts with
                id is the id of the band in the tiff file
                default is the(value for outside of range location
        Returns:
            function to assign band value to each corresponding lat lon
        """
        if gdal is None:
            raise OasisException(
                "##### gdal need to be installed to use geotiff !!!#####\n"
                "on ubuntu, first install gdal then run pip based on the installed version\n"
                "-> apt-get update && apt-get install -y gdal-bin\n"
                "-> gdalinfo --version\n"
                "-> pip install gdal==<version>"
            )

        tiff_dataset = gdal.Open(self.to_abs_filepath(file_path), gdal.GA_ReadOnly)
        inv_gt = gdal.InvGeoTransform(tiff_dataset.GetGeoTransform())

        defaults = np.empty(len(band_info), dtype=tiff_dataset.GetVirtualMemArray().dtype)
        usefull_array_idx = np.empty(len(band_info), dtype='int')
        for i, (col_name, info) in enumerate(band_info.items()):
            if not 1 <= info['id'] <= tiff_dataset.RasterCount:
                raise OasisException(f"band {col_name}, {info} has id outside of [1-{tiff_dataset.RasterCount}]")
            idx = info['id'] - 1
            usefull_array_idx[i] = idx
            defaults[idx] = info['default']

        def geotiff_lookup(locations):
            tiff_array = tiff_dataset.GetVirtualMemArray()
            if len(tiff_array.shape) == 2:
                tiff_array = tiff_array.reshape((tiff_array.shape[0], tiff_array.shape[1], 1))
            res = np.empty((len(locations), len(band_info)), dtype=tiff_array.dtype)
            jit_gda_loc_to_val(tiff_array,
                               inv_gt,
                               locations['longitude'].to_numpy(),
                               locations['latitude'].to_numpy(),
                               usefull_array_idx,
                               defaults,
                               res)
            for col_i, col_name in enumerate(band_info.keys()):
                locations[col_name] = res[:, col_i]
            return locations

        return geotiff_lookup

    def build_merge(self, file_path, id_columns=[], **kwargs):
        """
        this method will merge the locations Dataframe with the Dataframe present in file_path
        All non match column present in id_columns will be set to -1

        this is an efficient way to map a combination of column that have a finite scope to an idea.
        """

        df_to_merge = pd.read_csv(self.to_abs_filepath(file_path), **kwargs)
        df_to_merge.rename(columns={column: column.lower() for column in df_to_merge.columns}, inplace=True)

        def merge(locations: pd.DataFrame):
            rename_map = {col.lower(): col for col in locations.columns if col.lower() in df_to_merge.columns}
            locations = locations.merge(df_to_merge.rename(columns=rename_map), how='left')
            return self.set_id_columns(locations, id_columns)
        return merge

    @staticmethod
    def build_simple_pivot(pivots, remove_pivoted_col=True):
        """
        allow to pivot columns of the locations dataframe into multiple rows
        each pivot in the pivot list may define:
            "on": to rename a column into a new one
            "new_cols": to create a new column with a certain values
        ex:
        "pivots": [{"on": {"vuln_str": "vulnerability_id"},
                 "new_cols": {"coverage_type": 1}},
                {"on": {"vuln_con": "vulnerability_id"},
                 "new_cols": {"coverage_type": 3}},
               ],
        loc_id  vuln_str    vuln_con
        1       3           2
        2       18          4

        =>
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
        """
        Serialises specified columns from the OED file into a model_data dict
        """
        lst_model_data = []

        def model_data(locations):
            # could improve with apply lambda
            for index, i in locations.iterrows():
                tmp_dict = {}
                for col in columns:
                    tmp_dict[col] = i[col]
                lst_model_data.append(tmp_dict)

            locations['model_data'] = lst_model_data

            return locations

        return model_data

    @staticmethod
    def build_dynamic_model_adjustment(intensity_adjustment_col, return_period_col):
        """
        Converts specified columns from the OED file into intensity adjustments and
        return period protection.
        """
        lst_intensity_adjustment = []
        lst_return_period = []

        def adjustments(locations):
            for index, row in locations.iterrows():
                intensity_adjustment = row[intensity_adjustment_col]
                return_period = row[return_period_col]
                lst_intensity_adjustment.append(intensity_adjustment)
                lst_return_period.append(return_period)

            locations['intensity_adjustment'] = lst_intensity_adjustment
            locations['return_period'] = lst_return_period
            return locations

        return adjustments
