__all__ = [
    'get_gul_input_items',
    'write_amplifications_file',
    'write_coverages_file',
    'write_gul_input_files',
    'write_items_file',
    'write_complex_items_file',
    'write_sections_file'
]
import copy
import os
import sys
import warnings
from collections import OrderedDict

import pandas as pd
import numpy as np

from oasislmf.pytools.data_layer.oasis_files.correlations import \
    CorrelationsData
from oasislmf.utils.data import (factorize_ndarray, merge_dataframes,
                                 set_dataframe_column_dtypes)
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
    Generates and returns a Pandas dataframe of GUL input items.

    :param exposure_df: Exposure dataframe
    :type exposure_df: pandas.DataFrame

    :param keys_df: Keys dataframe
    :type keys_df: pandas.DataFrame

    :param output_dir: the output directory where input files are stored
    :type output_dir: str

    :param exposure_profile: Exposure profile
    :type exposure_profile: dict

    :param damage_group_id_cols: Columns to be used to generate a hashed damage group id.
    :type damage_group_id_cols: list[str]

    :param hazard_group_id_cols: Columns to be used to generate a hashed hazard group id.
    :type hazard_group_id_cols: list[str]

    :param do_disaggregation: If True, disaggregates by the number of buildings
    :type do_disaggregation: bool

    :return: GUL inputs dataframe
    :rtype: pandas.DataFrame

    """
    # Get the grouped exposure profile - this describes the financial terms to
    # to be found in the source exposure file, which are for the following
    # FM levels: site coverage (# 1), site pd (# 2), site all (# 3). It also
    # describes the OED hierarchy terms present in the exposure file, namely
    # portfolio num., acc. num., loc. num., and cond. num.

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
        location_df['IsAggregate'].fillna(0, inplace=True)

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

    gul_inputs_df = location_df[list(set(exposure_df_gul_inputs_cols).intersection(location_df.columns))]
    gul_inputs_df.drop_duplicates('loc_id', inplace=True, ignore_index=True)

    # Rename the main keys dataframe columns - this is due to the fact that the
    # keys file headers use camel case, and don't use underscored names, which
    # is the convention used for the GUL and IL inputs dataframes in the MDK
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

    # prepare column mappings for all coverage types
    cols_by_cov_type = {}
    for cov_type in gul_inputs_df.coverage_type_id.unique():
        tiv_col = tiv_terms[cov_type]
        cols_by_cov_type[cov_type] = {
            'tiv_col': tiv_col
        }
    # coverage unpacking and disaggregation loop:
    #  - one row representing N coverages is being transformed to N rows, one per coverage.
    #  - if NumberOfBuildings > 1, on top of unpacking the coverage, it performs the disaggregation of the items
    #    by repeating the rows `NumberOfBuildings` times and assigning to each row a unique `disagg_id`` number,
    #    useful for generating `item_id` later.
    #  - group the rows in the GUL inputs table by coverage type
    #  - set the IL terms (and BI coverage boolean) in each group and update the corresponding frame section in the GUL inputs table
    gul_inputs_reformatted_chunks = []
    if do_disaggregation:
        # split TIV
        gul_inputs_df[tiv_cols] = gul_inputs_df[tiv_cols].div(np.maximum(1, gul_inputs_df['NumberOfBuildings']), axis=0)

    for (number_of_buildings, cov_type), cov_type_group in gul_inputs_df.groupby(by=['NumberOfBuildings', 'coverage_type_id'], sort=True):
        if tiv_terms[cov_type] not in gul_inputs_df:
            continue
        cov_type_group['tiv'] = cov_type_group[cols_by_cov_type[cov_type]['tiv_col']]
        cov_type_group['coverage_type_id'] = cov_type
        cov_type_group = cov_type_group[cov_type_group['tiv'] > 0]
        if do_disaggregation:
            # if NumberOfBuildings == 0: still add one entry
            disagg_df_chunk = (cov_type_group.reset_index()
                               .join(pd.DataFrame({'building_id': range(1, max(number_of_buildings, 1) + 1)}), how='cross')
                               .set_index('index'))
        else:
            disagg_df_chunk = cov_type_group.copy().assign(building_id=1)

        gul_inputs_reformatted_chunks.append(disagg_df_chunk)

    for disagg_df_chunk in gul_inputs_reformatted_chunks:
        if disagg_df_chunk.shape[0]:
            break
    else:
        raise OasisException('Empty gul_inputs_df dataframe after dropping rows with zero tiv: please check the exposure input files')

    # concatenate all the unpacked chunks. Sort by index to preserve `item_id` order as in the original code
    gul_inputs_df = (
        pd.concat(gul_inputs_reformatted_chunks)
        .sort_index()
        .reset_index(drop=True)
    )

    # add risk_id to gul_inputs_df
    gul_inputs_df[['risk_id', 'NumberOfRisks']] = gul_inputs_df[['building_id', 'NumberOfBuildings']]
    gul_inputs_df.loc[gul_inputs_df['IsAggregate'] == 0, ['risk_id', 'NumberOfRisks']] = 1, 1
    gul_inputs_df.loc[gul_inputs_df['NumberOfRisks'] == 0, 'NumberOfRisks'] = 1

    # set 'disagg_id', `item_id` and `coverage_id`
    gul_inputs_df['item_id'] = factorize_ndarray(
        gul_inputs_df.loc[:, ['loc_id', 'peril_id', 'coverage_type_id', 'building_id']].values, col_idxs=range(4))[0]
    gul_inputs_df['coverage_id'] = factorize_ndarray(gul_inputs_df.loc[:, ['loc_id', 'building_id', 'coverage_type_id']].values, col_idxs=range(3))[0]

    # set default data types
    gul_inputs_df = set_dataframe_column_dtypes(gul_inputs_df, {'item_id': 'int32', 'coverage_id': 'int32'})

    # Set the group ID
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

    # Select only required columns
    # Order here matches test output expectations
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
def write_complex_items_file(gul_inputs_df, complex_items_fp, chunksize=100000):
    """
    Writes a complex model items file.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param complex_items_fp: Complex/custom model items file path
    :type complex_items_fp: str

    :return: Complex/custom model items file path
    :rtype: str
    """
    try:
        gul_inputs_df.loc[:, ['item_id', 'coverage_id', 'model_data', 'group_id']].drop_duplicates().to_csv(
            path_or_buf=complex_items_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(complex_items_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_complex_items_file'", e)


@oasis_log
def write_sections_file(gul_inputs_df, sections_fp, chunksize=100000):
    """
    Writes a section id file based on the input location area perils.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param dynamic_events_fp: events file path to output
    :type sections_fp: str
    """
    try:
        sections = gul_inputs_df.loc[:, ['section_id']].drop_duplicates()
        sections['section_id'] = sections['section_id'].astype(str)
        sections['section_id'] = sections['section_id'].str.split(';')
        sections = sections.explode('section_id').drop_duplicates()
        sections['section_id'] = sections['section_id'].astype(int)
        sections.to_csv(
            path_or_buf=sections_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(sections_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_sections_file'", e)


@oasis_log
def write_item_adjustments_file(gul_inputs_df, item_adjustments_fp, chunksize=100000):
    """
    Writes a item_adjustments id file based on the gul inputs.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param item_adjustments_fp: item_adjustments file path to output
    :type sections_fp: str
    """
    try:
        gul_inputs_df.loc[:, ['item_id', 'intensity_adjustment', 'return_period']].drop_duplicates().to_csv(
            path_or_buf=item_adjustments_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(item_adjustments_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_item_adjustments_file'", e)


@oasis_log
def write_amplifications_file(gul_inputs_df, amplifications_fp, chunksize=100000):
    """
    Writes an amplifications file. This is the mapping between item IDs and
    amplifications IDs.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param amplifications_fp: amplifications file path
    :type amplifications_fp: str

    :return: amplifications file path
    :rtype: str
    """
    try:
        gul_inputs_df.loc[:, ['item_id', 'amplification_id']].drop_duplicates().to_csv(
            path_or_buf=amplifications_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(amplifications_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raise in 'write_amplifications_file'", e)

    return amplifications_fp


@oasis_log
def write_items_file(gul_inputs_df, items_fp, chunksize=100000):
    """
    Writes an items file.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param items_fp: Items file path
    :type items_fp: str

    :return: Items file path
    :rtype: str
    """
    try:
        gul_inputs_df.loc[:, ['item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id']].drop_duplicates().to_csv(
            path_or_buf=items_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(items_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_items_file'", e)

    return items_fp


@oasis_log
def write_coverages_file(gul_inputs_df, coverages_fp, chunksize=100000):
    """
    Writes a coverages file.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param coverages_fp: Coverages file path
    :type coverages_fp: str

    :return: Coverages file path
    :rtype: str
    """
    try:
        gul_inputs_df.loc[:, ['coverage_id', 'tiv']].drop_duplicates().to_csv(
            path_or_buf=coverages_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(coverages_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_coverages_file'", e)

    return coverages_fp


@oasis_log
def write_gul_input_files(
    gul_inputs_df,
    target_dir,
    correlations_df,
    output_dir,
    oasis_files_prefixes=OASIS_FILES_PREFIXES['gul'],
    chunksize=(2 * 10 ** 5),
):
    """
    Writes the standard Oasis GUL input files to a target directory, using a
    pre-generated dataframe of GUL input items. The files written are
    ::

        items.csv
        coverages.csv

    and optionally a complex items file in case of a complex/custom model.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param target_dir: Target directory in which to write the files
    :type target_dir: str

    :param oasis_files_prefixes: Oasis GUL input file name prefixes
    :param oasis_files_prefixes: dict

    :param chunksize: The chunk size to use when writing out the
                      input files
    :type chunksize: int

    :return: GUL input files dict
    :rtype: dict
    """
    # Clean the target directory path
    target_dir = as_path(target_dir, 'Target IL input files directory', is_dir=True, preexists=False)
    oasis_files_prefixes = copy.deepcopy(oasis_files_prefixes)

    if correlations_df is None:
        correlations_df = pd.DataFrame(columns=CorrelationsData.COLUMNS)

    # write the correlations to a binary file
    correlation_data_handle = CorrelationsData(data=correlations_df)
    correlation_data_handle.to_bin(file_path=f"{output_dir}/correlations.bin")
    correlation_data_handle.to_csv(file_path=f"{output_dir}/correlations.csv")

    # Set chunk size for writing the CSV files - default is the minimum of 100K
    # or the GUL inputs frame size
    chunksize = chunksize or min(chunksize, len(gul_inputs_df))
    # If no complex model data present then remove the corresponding file
    # name from the files prefixes dict, which is used for writing the
    # GUl input files
    if 'model_data' not in gul_inputs_df:
        oasis_files_prefixes.pop('complex_items', None)

    # If no amplification IDs then remove corresponding file name from files
    # prefixes dict
    if 'amplification_id' not in gul_inputs_df:
        oasis_files_prefixes.pop('amplifications', None)

    # If no section IDs then remove corresponding file name from files
    # prefixes dict
    if 'section_id' not in gul_inputs_df:
        oasis_files_prefixes.pop('sections', None)

    # If no adjustments data then remove corresponding file name from files
    # prefixes dict
    if 'intensity_adjustment' not in gul_inputs_df:
        oasis_files_prefixes.pop('item_adjustments', None)

    # A dict of GUL input file names and file paths
    gul_input_files = {
        fn: os.path.join(target_dir, '{}.csv'.format(oasis_files_prefixes[fn]))
        for fn in oasis_files_prefixes
    }
    this_module = sys.modules[__name__]
    # Write the files serially
    for fn in gul_input_files:
        getattr(this_module, 'write_{}_file'.format(fn))(gul_inputs_df.copy(deep=True), gul_input_files[fn], chunksize)

    return gul_input_files
