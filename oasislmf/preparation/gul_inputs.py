__all__ = [
    'get_gul_input_items',
    'write_gul_input_files',
]
import os
import warnings
from collections import OrderedDict

import pandas as pd
import numpy as np

from oasislmf.pytools.common.data import (correlations_headers, correlations_dtype, amplifications_dtype, items_dtype,
                                          coverages_dtype, item_adjustment_dtype,
                                          complex_items_meta_dtype,
                                          item_id, coverage_id, group_id, section_id,
                                          DTYPE_IDX)
from oasislmf.utils.data import merge_dataframes
from oasislmf.utils.defaults import (CORRELATION_GROUP_ID,
                                     DAMAGE_GROUP_ID_COLS,
                                     HAZARD_GROUP_ID_COLS,
                                     OASIS_FILES_PREFIXES, SOURCE_IDX,
                                     get_default_exposure_profile)
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.fm import SUPPORTED_FM_LEVELS
from oasislmf.utils.log import oasis_log
from oasislmf.utils.path import as_path
from oasislmf.utils.profiles import get_grouped_fm_profile_by_level_and_term_group, get_oed_hierarchy

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

VALID_OASIS_GROUP_COLS = [
    'item_id',
    'peril_id',
    'coverage_id',
    'coverage_type_id',
    'peril_correlation_group',
    'building_id',
    'risk_id'
]

PERIL_CORRELATION_GROUP_COL = 'peril_correlation_group'


def prepare_sections_df(gul_inputs_df):
    sections = gul_inputs_df.loc[:, ['section_id']].drop_duplicates()
    sections['section_id'] = sections['section_id'].astype(str).str.split(';')
    sections = sections.explode('section_id').drop_duplicates()
    sections['section_id'] = sections['section_id'].astype(section_id[DTYPE_IDX])
    return sections


def coverages_write_gul_bin(data, file_path, dtype):
    df_to_ndarray(data, dtype)["tiv"].tofile(file_path)


def complex_items_write_gul_bin(data, file_path, dtype):
    with open(file_path, 'wb') as f:
        complex_items_write_bin(data, f, dtype)


def amplifications_write_gul_bin(data, file_path, dtype):
    with open(file_path, 'wb') as f:
        amplifications_write_bin(df_to_ndarray(data, dtype), f)


def default_write_gul_bin(data, file_path, dtype):
    df_to_ndarray(data, dtype).tofile(file_path)


files_write_info = {
    'complex_items': {"csv_dtype": {'item_id': item_id[DTYPE_IDX], 'coverage_id': coverage_id[DTYPE_IDX], 'model_data': str, 'group_id': group_id[DTYPE_IDX]},
                      "bin_dtype": complex_items_meta_dtype,
                      "write_bin": complex_items_write_gul_bin,
                      "required_col": {'model_data'}},

    'items': {"csv_dtype": {col: dtype for col, (dtype, _) in items_dtype.fields.items()},
              "bin_dtype": items_dtype},
    'coverages': {"csv_dtype": {col: dtype for col, (dtype, _) in coverages_dtype.fields.items()},
                  "bin_dtype": coverages_dtype,
                  "write_bin": coverages_write_gul_bin},
    'amplifications': {"csv_dtype": {col: dtype for col, (dtype, _) in amplifications_dtype.fields.items()},
                       "bin_dtype": amplifications_dtype,
                       "write_bin": amplifications_write_gul_bin,
                       "required_col": {'amplification_id'}},
    'sections': {"csv_dtype": {'section_id': section_id[DTYPE_IDX]},
                 "prepare_data": prepare_sections_df,
                 "required_col": {'section_id'}},
    'item_adjustments': {"csv_dtype": {col: dtype for col, (dtype, _) in item_adjustment_dtype.fields.items()},
                         "required_col": {'intensity_adjustment'}}
}


def process_group_id_cols(group_id_cols, exposure_df_columns, has_correlation_groups):
    """
    cleans out columns that are not valid oasis group columns.

    Valid group id columns can be either
    1. exist in the location file
    2. be listed as a useful internal col

    Args:
        group_id_cols: (List[str]) the ID columns that are going to be filtered
        exposure_df_columns: (List[str]) the columns in the exposure dataframe
        has_correlation_groups: (bool) if set to True means that we are hashing with correlations in mind therefore the
                             "peril_correlation_group" column is added

    Returns: (List[str]) the filtered columns
    """
    for col in group_id_cols:
        if col not in list(exposure_df_columns) + VALID_OASIS_GROUP_COLS:
            warnings.warn('Column {} not found in loc file, or a valid internal oasis column'.format(col))
            group_id_cols.remove(col)

    if PERIL_CORRELATION_GROUP_COL not in group_id_cols and has_correlation_groups is True:
        group_id_cols.append(PERIL_CORRELATION_GROUP_COL)

    return group_id_cols


@oasis_log
def get_gul_input_items(
    location_df,
    keys_df,
    correlations=False,
    peril_correlation_group_df=None,
    exposure_profile=get_default_exposure_profile(),
    damage_group_id_cols=None,
    hazard_group_id_cols=None,
    do_disaggregation=True
):
    """
    Generates GUL (Ground-Up Loss) input items by combining location and keys data.

    This function creates the foundational data structure for loss calculations by
    merging exposure (location) data with model keys data. Each resulting row
    represents a unique combination of location, peril, coverage type, and building
    that will flow through the loss calculation pipeline.

    Overview of Processing Flow:
    ===========================
    1. SETUP: Load profiles, extract TIV columns, prepare location data
    2. MERGE: Join location_df with keys_df on loc_id to create base GUL items
    3. COVERAGE UNPACKING: Set TIV values based on coverage_type_id
    4. DISAGGREGATION: If enabled, expand rows by NumberOfBuildings
    5. ID ASSIGNMENT: Compute item_id, coverage_id, group_id, hazard_group_id
    6. FINALIZE: Select required columns and return

    Key Data Structures:
    ===================
    - location_df: Source exposure data with TIV columns (BuildingTIV, ContentsTIV, etc.)
    - keys_df: Model lookup results mapping locations to model-specific IDs
      (areaperil_id, vulnerability_id, peril_id, coverage_type_id)
    - gul_inputs_df: Output DataFrame with one row per (location, peril, coverage, building)

    Key Output Columns:
    ==================
    - item_id: Unique identifier for each GUL item (loc_id, peril_id, coverage_type_id, building_id)
    - coverage_id: Groups items by (loc_id, building_id, coverage_type_id)
    - group_id: Damage correlation group (hashed from damage_group_id_cols)
    - hazard_group_id: Hazard correlation group (hashed from hazard_group_id_cols)
    - tiv: Total Insured Value for this item's coverage type
    - areaperil_id, vulnerability_id: Model-specific identifiers from keys

    Disaggregation:
    ==============
    When do_disaggregation=True and NumberOfBuildings > 1:
    - TIV is divided by NumberOfBuildings
    - Rows are repeated NumberOfBuildings times
    - Each repeated row gets a unique building_id (1 to NumberOfBuildings)
    This allows modeling individual buildings within an aggregate location.

    Args:
        location_df (pandas.DataFrame): Exposure data with columns including loc_id,
            PortNumber, AccNumber, LocNumber, TIV columns, NumberOfBuildings, IsAggregate.
        keys_df (pandas.DataFrame): Model keys with columns including locid/loc_id,
            perilid, coveragetypeid, areaperilid, vulnerabilityid.
        correlations (bool, optional): If True, merge with peril_correlation_group_df
            for correlation modeling. Default False.
        peril_correlation_group_df (pandas.DataFrame, optional): Correlation group
            definitions when correlations=True.
        exposure_profile (dict, optional): Maps OED fields to FM term types.
        damage_group_id_cols (list[str], optional): Columns used to compute group_id
            via hashing. Default: ['loc_id', 'peril_correlation_group'].
        hazard_group_id_cols (list[str], optional): Columns used to compute hazard_group_id
            via hashing. Default: ['loc_id'].
        do_disaggregation (bool, optional): If True, split aggregate locations by
            NumberOfBuildings. Default True.

    Returns:
        pandas.DataFrame: GUL inputs with columns including item_id, coverage_id,
            group_id, hazard_group_id, tiv, areaperil_id, vulnerability_id, peril_id,
            coverage_type_id, building_id, and location identifiers.

    Raises:
        OasisException: If exposure profile is missing FM term information.
        OasisException: If merge of location and keys data produces empty result.
        OasisException: If all rows have zero TIV after filtering.
    """
    # =========================================================================
    # SETUP PHASE: Load profiles and extract configuration
    # =========================================================================
    # Get the grouped exposure profile - describes FM terms (TIV, limit, deductible)
    # for site coverage level and OED hierarchy terms (portfolio, account, location)
    profile = get_grouped_fm_profile_by_level_and_term_group(exposure_profile=exposure_profile)

    if not profile:
        raise OasisException(
            'Source exposure profile is possibly missing FM term information: '
            'FM term definitions for TIV, limit, deductible, attachment and/or share.'
        )

    # Get the OED hierarchy terms profile - this defines the column names for loc.
    # ID, acc. ID, policy no. and portfolio no., as used in the source exposure
    # and accounts files. This is to ensure that the method never makes hard
    # coded references to the corresponding columns in the source files, as
    # that would mean that changes to these column names in the source files
    # may break the method
    oed_hierarchy = get_oed_hierarchy(exposure_profile=exposure_profile)
    loc_num = oed_hierarchy['locnum']['ProfileElementName']
    acc_num = oed_hierarchy['accnum']['ProfileElementName']
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName']

    # The (site) coverage FM level ID (# 1 in the OED FM levels hierarchy)
    cov_level_id = SUPPORTED_FM_LEVELS['site coverage']['id']

    # Get the TIV column names and corresponding coverage types
    tiv_terms = OrderedDict({v['tiv']['CoverageTypeID']: v['tiv']['ProfileElementName'] for k, v in profile[cov_level_id].items()})
    tiv_cols = list(set(tiv_col for tiv_col in tiv_terms.values() if tiv_col in location_df.columns))

    # Create the basic GUL inputs dataframe from merging the exposure and
    # keys dataframes on loc. number/loc. ID; filter out any rows with
    # zeros for TIVs for all coverage types, and replace any nulls in the
    # cond.num. and TIV columns with zeros

    # add default values if missing
    if 'IsAggregate' not in location_df.columns:
        location_df['IsAggregate'] = 0
    else:
        location_df['IsAggregate'] = location_df['IsAggregate'].fillna(0)

    # Make sure NumberOfBuildings is there and filled (not mandatory), otherwise assume NumberOfBuildings = 1
    if 'NumberOfBuildings' not in location_df.columns:
        location_df['NumberOfBuildings'] = 1
    else:
        location_df['NumberOfBuildings'] = location_df['NumberOfBuildings'].fillna(1)

    # Select only the columns required. This reduces memory use significantly for portfolios
    # that include many OED columns.
    exposure_df_gul_inputs_cols = ['loc_id', portfolio_num, acc_num, loc_num, 'NumberOfBuildings', 'IsAggregate', 'LocPeril'] + tiv_cols
    if SOURCE_IDX['loc'] in location_df:
        exposure_df_gul_inputs_cols += [SOURCE_IDX['loc']]

    # it is assumed that correlations are False for now, correlations for group ID hashing are assessed later on in
    # the process to re-hash the group ID with the correlation "peril_correlation_group" column name. This is because
    # the correlations is achieved later in the process leading to a chicken and egg problem

    # set damage_group_id_cols
    if not damage_group_id_cols:
        # damage_group_id_cols is None or an empty list
        damage_group_id_cols = DAMAGE_GROUP_ID_COLS
    else:
        # remove any duplicate column names used to assign group_id
        damage_group_id_cols = list(set(damage_group_id_cols))

    # only add damage group col if not an internal oasis col or if not present already in exposure_df_gul_inputs_cols
    for col in damage_group_id_cols:
        if col in VALID_OASIS_GROUP_COLS:
            pass
        elif col not in exposure_df_gul_inputs_cols:
            exposure_df_gul_inputs_cols.append(col)

    # set hazard_group_id_cols
    if not hazard_group_id_cols:
        # hazard_group_id_cols is None or an empty list
        hazard_group_id_cols = HAZARD_GROUP_ID_COLS
    else:
        # remove any duplicate column names used to assign group_id
        hazard_group_id_cols = list(set(hazard_group_id_cols))

    # only add hazard group col if not an internal oasis col or if not present already in exposure_df_gul_inputs_cols
    for col in hazard_group_id_cols:
        if col in VALID_OASIS_GROUP_COLS:
            pass
        elif col not in exposure_df_gul_inputs_cols:
            exposure_df_gul_inputs_cols.append(col)

    # Check if correlation group field is used to drive damage group id
    # and test that it's present and populated with integers

    correlation_group_id = CORRELATION_GROUP_ID
    correlation_field = correlation_group_id[0]
    correlation_check = False
    if damage_group_id_cols == correlation_group_id:
        if correlation_field in location_df.columns:
            if location_df[correlation_field].astype('uint32').isnull().sum() == 0:
                correlation_check = True

    actual_tiv_cols = [tiv_col for tiv_col in tiv_cols if tiv_col in location_df.columns]
    location_df[actual_tiv_cols] = location_df[actual_tiv_cols].fillna(0.0)
    location_df = location_df[(location_df[actual_tiv_cols] != 0).any(axis=1)]

    gul_inputs_df = (location_df[list(set(exposure_df_gul_inputs_cols).intersection(location_df.columns))]
                     .drop_duplicates('loc_id', ignore_index=True))

    # =========================================================================
    # MERGE PHASE: Join location data with model keys
    # =========================================================================
    # Keys file uses camelCase headers; rename to snake_case for consistency
    keys_df.rename(
        columns={
            'locid': 'loc_id' if 'loc_id' not in keys_df else 'locid',
            'perilid': 'peril_id',
            'coveragetypeid': 'coverage_type_id',
            'areaperilid': 'areaperil_id',
            'vulnerabilityid': 'vulnerability_id',
            'amplificationid': 'amplification_id',
            'modeldata': 'model_data',
            'intensityadjustment': 'intensity_adjustment',
            'returnperiod': 'return_period'
        },
        inplace=True,
        copy=False  # Pandas copies column data by default on rename
    )

    # If the keys file relates to a complex/custom model then look for a
    # ``modeldata`` column in the keys file, and ignore the area peril
    # and vulnerability ID columns, unless it's the dynamic model generator which
    # uses them
    if 'model_data' in keys_df and 'areaperil_id' not in keys_df and 'vulnerbaility_id' not in keys_df:
        keys_df['areaperil_id'] = keys_df['vulnerability_id'] = -1
    gul_inputs_df = merge_dataframes(
        keys_df,
        gul_inputs_df,
        join_on='loc_id',
        how='inner',
    )
    if gul_inputs_df.empty:
        raise OasisException(
            'Inner merge of the exposure file dataframe '
            'and the keys file dataframe on loc. number/loc. ID '
            'is empty - '
            'please check that the loc. number and loc. ID columns '
            'in the exposure and keys files respectively have a non-empty '
            'intersection'
        )

    # Free memory after merge, before memory-intensive restructuring of data
    del keys_df

    # =========================================================================
    # COVERAGE UNPACKING: Set TIV based on coverage type
    # =========================================================================
    # Map coverage_type_id to the appropriate TIV column (BuildingTIV, ContentsTIV, etc.)
    cols_by_cov_type = {}
    for cov_type in gul_inputs_df.coverage_type_id.unique():
        tiv_col = tiv_terms[cov_type]
        cols_by_cov_type[cov_type] = {
            'tiv_col': tiv_col
        }

    # If disaggregating, divide TIV by NumberOfBuildings before assigning
    if do_disaggregation:
        # split TIV
        gul_inputs_df[tiv_cols] = gul_inputs_df[tiv_cols].div(np.maximum(1, gul_inputs_df['NumberOfBuildings']), axis=0)

    # Set tiv column based on coverage_type_id (vectorized)
    gul_inputs_df['tiv'] = 0.0
    for cov_type, tiv_col in cols_by_cov_type.items():
        mask = gul_inputs_df['coverage_type_id'] == cov_type
        gul_inputs_df.loc[mask, 'tiv'] = gul_inputs_df.loc[mask, tiv_col['tiv_col']]

    # Filter out rows with zero TIV
    gul_inputs_df = gul_inputs_df[gul_inputs_df['tiv'] > 0]

    if gul_inputs_df.empty:
        raise OasisException('Empty gul_inputs_df dataframe after dropping rows with zero tiv: please check the exposure input files')

    # =========================================================================
    # DISAGGREGATION: Expand rows by NumberOfBuildings
    # =========================================================================
    # For aggregate locations (NumberOfBuildings > 1), create one row per building
    # Each building gets a unique building_id and its share of the TIV
    if do_disaggregation:
        repeat_counts = np.maximum(1, gul_inputs_df['NumberOfBuildings'].values).astype(int)
        # Repeat rows using np.repeat + iloc (faster than iterative expansion)
        gul_inputs_df = gul_inputs_df.iloc[np.repeat(np.arange(len(gul_inputs_df)), repeat_counts)].reset_index(drop=True)
        # Assign building_id: 1, 2, ..., n for each location
        gul_inputs_df['building_id'] = np.concatenate([np.arange(1, n + 1) for n in repeat_counts])
    else:
        gul_inputs_df = gul_inputs_df.copy()
        gul_inputs_df['building_id'] = 1

    # =========================================================================
    # ID ASSIGNMENT: Compute item_id, coverage_id, group_id, hazard_group_id
    # =========================================================================
    # risk_id/NumberOfRisks: Used for aggregate exposure handling in FM
    gul_inputs_df[['risk_id', 'NumberOfRisks']] = gul_inputs_df[['building_id', 'NumberOfBuildings']]
    gul_inputs_df.loc[gul_inputs_df['IsAggregate'] == 0, ['risk_id', 'NumberOfRisks']] = 1, 1
    gul_inputs_df.loc[gul_inputs_df['NumberOfRisks'] == 0, 'NumberOfRisks'] = 1

    # item_id: Unique per (location, peril, coverage_type, building) - the fundamental GUL unit
    gul_inputs_df['item_id'] = gul_inputs_df.groupby(
        ['loc_id', 'peril_id', 'coverage_type_id', 'building_id'], sort=False, observed=True).ngroup().astype('int32') + 1
    # coverage_id: Groups items by (location, building, coverage_type) - ignores peril
    gul_inputs_df['coverage_id'] = gul_inputs_df.groupby(
        ['loc_id', 'building_id', 'coverage_type_id'], sort=False, observed=True).ngroup().astype('int32') + 1

    # group_id and hazard_group_id: Correlation groups for damage/hazard sampling
    # If the group id is set according to the correlation group field then map this field
    # directly, otherwise create an index of the group id fields

    # keep group_id consistance by lower casing column names and sorting
    damage_group_id_cols_map = {c: c.lower() for c in sorted(damage_group_id_cols)}        # mapping from PascalCase -> 'lower_case'
    hazard_group_id_cols_map = {c: c.lower() for c in sorted(hazard_group_id_cols)}        # mapping from PascalCase -> 'lower_case'

    if correlation_check is True:
        gul_inputs_df['group_id'] = gul_inputs_df[correlation_group_id]

    if correlations:
        # do merge with peril correlation df
        gul_inputs_df = gul_inputs_df.merge(peril_correlation_group_df, left_on='peril_id', right_on='id').reset_index()
    else:
        gul_inputs_df[["peril_correlation_group", "damage_correlation_value", "hazard_correlation_value"]] = 0
    gul_inputs_df["group_id"] = (
        pd.util.hash_pandas_object(
            gul_inputs_df.rename(columns=damage_group_id_cols_map)[sorted(list(damage_group_id_cols_map.values()))], index=False).to_numpy() >> 33
    ).astype('uint32')

    gul_inputs_df["hazard_group_id"] = (
        pd.util.hash_pandas_object(
            gul_inputs_df.rename(columns=hazard_group_id_cols_map)[sorted(list(hazard_group_id_cols_map.values()))], index=False).to_numpy() >> 33
    ).astype('uint32')

    # =========================================================================
    # FINALIZE: Select required columns and return
    # =========================================================================
    keyscols = ['peril_id', 'coverage_type_id', 'tiv', 'areaperil_id', 'vulnerability_id']
    additionalcols = ['amplification_id', 'section_id', 'intensity_adjustment', 'return_period']
    for col in additionalcols:
        if col in gul_inputs_df.columns:
            keyscols += [col]
    usecols = (
        ['loc_id', portfolio_num, acc_num, loc_num] +
        ([SOURCE_IDX['loc']] if SOURCE_IDX['loc'] in gul_inputs_df else []) +
        keyscols +
        (['model_data'] if 'model_data' in gul_inputs_df else []) +
        # disagg_id is needed for fm_summary_map
        ['group_id', 'coverage_id', 'item_id', 'status', 'building_id', 'NumberOfBuildings', 'IsAggregate', 'LocPeril'] +
        tiv_cols +
        ["peril_correlation_group", "damage_correlation_value", 'hazard_group_id', "hazard_correlation_value"]
    )

    usecols = [col for col in usecols if col in gul_inputs_df]

    gul_inputs_df = (
        gul_inputs_df
        [usecols]
        .drop_duplicates(subset='item_id')
        .sort_values("item_id", kind='stable')
        .reset_index()
    )

    return gul_inputs_df


@oasis_log
def write_file(gul_inputs_df, file_path, file_dtype, chunksize=100000):
    try:
        gul_inputs_df[file_dtype.keys()].astype(file_dtype).drop_duplicates().to_csv(
            path_or_buf=file_path,
            encoding='utf-8',
            mode=('w' if os.path.exists(file_path) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(f"Exception raised while writing {file_path} {file_dtype}", e)

    return file_path


@oasis_log
def write_gul_input_files(
    gul_inputs_df,
    target_dir,
    correlations_df,
    output_dir,
    oasis_files_prefixes=OASIS_FILES_PREFIXES['gul'],
    chunksize=(2 * 10 ** 5),
    intermediary_csv=False,
):
    """Write standard Oasis GUL input files to a target directory.

    Writes binary files (items.bin, coverages.bin) directly from a pre-generated
    dataframe of GUL input items. Optional files (complex_items.bin,
    amplifications.bin) are written when the corresponding columns are present.
    Files that have no binary consumer (sections.csv, item_adjustments.csv) are
    always written as CSV.

    Args:
        gul_inputs_df (pd.DataFrame): GUL inputs dataframe.
        target_dir (str): Target directory in which to write the files.
        correlations_df (pd.DataFrame): Correlations dataframe. If None, an
            empty dataframe with correlations_headers columns is used.
        output_dir (str): Output directory for correlations files.
        oasis_files_prefixes (dict): Oasis GUL input file name prefixes.
            Defaults to OASIS_FILES_PREFIXES['gul'].
        chunksize (int): Chunk size for writing CSV files.
            Defaults to 200000.
        intermediary_csv (bool): If True, also write CSV files alongside
            binary for debugging. Defaults to False.

    Returns:
        dict: Mapping of file names to their written file paths.
    """
    # Clean the target directory path
    target_dir = as_path(target_dir, 'Target IL input files directory', is_dir=True, preexists=False)

    # write the correlations to a binary file
    if correlations_df is None:
        correlations_df = pd.DataFrame(columns=correlations_headers)
    correlations_df.to_csv(f"{output_dir}/correlations.csv", index=False)
    correlations_df_np_data = np.array([r for r in correlations_df.itertuples(index=False)], dtype=correlations_dtype)
    correlations_df_np_data.tofile(f"{output_dir}/correlations.bin")

    # Set chunk size for writing the CSV files - default is the minimum of 100K
    # or the GUL inputs frame size
    chunksize = chunksize or min(chunksize, len(gul_inputs_df))

    # A dict of GUL input file names and file paths
    gul_input_files = {}

    # Write the gul_inputs_df files serially
    for fm_name, file_name in oasis_files_prefixes.items():
        file_write_info = files_write_info[fm_name]

        if not file_write_info.get("required_col", set()).issubset(gul_inputs_df.columns):
            continue

        data = file_write_info.get('prepare_data',
                                   lambda x: x[list(file_write_info['csv_dtype'].keys())]  # default
                                   )(gul_inputs_df).drop_duplicates()

        if "bin_dtype" in file_write_info:
            file_path = os.path.join(target_dir, f"{file_name}.bin")
            gul_input_files[fm_name] = file_path
            oasis_log(file_write_info.get("write_bin", default_write_gul_bin))(data, file_path, file_write_info['bin_dtype'])
            if intermediary_csv:
                file_path = os.path.join(target_dir, f"{file_name}.csv")
                write_file(data, file_path, file_write_info['csv_dtype'], chunksize=chunksize)
        else:
            file_path = os.path.join(target_dir, f"{file_name}.csv")
            gul_input_files[fm_name] = file_path
            write_file(data, file_path, file_write_info['csv_dtype'], chunksize=chunksize)

    return gul_input_files
