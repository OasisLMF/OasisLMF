__all__ = [
    'get_gul_input_items',
    'write_coverages_file',
    'write_gul_input_files',
    'write_items_file',
    'write_complex_items_file'
]
import copy
import os
import sys
import warnings
from collections import OrderedDict
from typing import List

import pandas as pd

from ..utils.coverages import SUPPORTED_COVERAGE_TYPES
from ..utils.data import (
    factorize_array,
    factorize_ndarray,
    merge_dataframes,
    set_dataframe_column_dtypes,
)
from ..utils.defaults import (
    SOURCE_IDX,
)
from ..utils.defaults import (
    get_default_exposure_profile,
    GROUP_ID_COLS,
    CORRELATION_GROUP_ID,
    OASIS_FILES_PREFIXES,
)
from ..utils.exceptions import OasisException
from ..utils.fm import SUPPORTED_FM_LEVELS
from ..utils.log import oasis_log
from ..utils.path import as_path
from ..utils.profiles import (
    get_fm_terms_oed_columns,
    get_grouped_fm_profile_by_level_and_term_group,
    get_grouped_fm_terms_by_level_and_term_group,
    get_oed_hierarchy,
)
from oasislmf.pytools.data_layer.oasis_files.correlations import CorrelationsData

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

VALID_OASIS_GROUP_COLS = [
    'item_id',
    'peril_id',
    'coverage_id',
    'coverage_type_id',
    'peril_correlation_group'
]


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

    peril_correlation_group = 'peril_correlation_group'
    if peril_correlation_group not in group_id_cols and has_correlation_groups is True:
        group_id_cols.append(peril_correlation_group)
    return group_id_cols


def hash_group_id(gul_inputs_df: pd.DataFrame, hashing_columns: List[str]) -> pd.DataFrame:
    """
    Creates a hash for the group ID field for the input data frame.

    Args:
        gul_inputs_df: (pd.DataFrame) the gul inputs that are doing the have the group_id field rewritten with a hash
        hashing_columns: (List[str]) the list of columns used in the hashing algorithm

    Returns: (pd.DataFrame) the gul_inputs_df with the new hashed group_id
    """
    gul_inputs_df["group_id"] = (pd.util.hash_pandas_object(gul_inputs_df[hashing_columns],
                                                            index=False).to_numpy() >> 33)
    return gul_inputs_df


@oasis_log
def get_gul_input_items(
    exposure_df,
    keys_df,
    correlations=False,
    peril_correlation_group_df=None,
    exposure_profile=get_default_exposure_profile(),
    group_id_cols=["PortNumber", "AccNumber", "LocNumber"],
    hashed_group_id=True
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

    :return: GUL inputs dataframe
    :rtype: pandas.DataFrame

    :return: Exposure dataframe
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
    tiv_cols = list(tiv_terms.values())

    # Get the list of coverage type IDs - financial terms for the coverage
    # level are grouped by coverage type ID in the grouped version of the
    # exposure profile (profile of the financial terms sourced from the
    # source exposure file)
    cov_types = [v['id'] for v in SUPPORTED_COVERAGE_TYPES.values()]

    # Get the FM terms profile (this is a simplfied view of the main grouped
    # profile, containing only information about the financial terms), and
    # the list of OED colum names for the financial terms for the site coverage
    # (# 1 ) FM level
    fm_terms = get_grouped_fm_terms_by_level_and_term_group(grouped_profile_by_level_and_term_group=profile, lowercase=False)
    terms_floats = ['deductible', 'deductible_min', 'deductible_max', 'limit']
    terms_ints = ['ded_code', 'ded_type', 'lim_code', 'lim_type']
    terms = terms_floats + terms_ints
    term_cols_floats = get_fm_terms_oed_columns(
        fm_terms,
        levels=['site coverage'],
        term_group_ids=cov_types,
        terms=terms_floats
    )
    term_cols_ints = get_fm_terms_oed_columns(
        fm_terms,
        levels=['site coverage'],
        term_group_ids=cov_types,
        terms=terms_ints
    )
    term_cols = term_cols_floats + term_cols_ints

    # Create the basic GUL inputs dataframe from merging the exposure and
    # keys dataframes on loc. number/loc. ID; filter out any rows with
    # zeros for TIVs for all coverage types, and replace any nulls in the
    # cond.num. and TIV columns with zeros

    # Select only the columns required. This reduces memory use significantly for portfolios
    # that include many OED columns.
    exposure_df_gul_inputs_cols = ['loc_id', portfolio_num, acc_num, loc_num] + term_cols + tiv_cols
    if SOURCE_IDX['loc'] in exposure_df:
        exposure_df_gul_inputs_cols += [SOURCE_IDX['loc']]
    # Remove any duplicate column names used to assign group_id
    group_id_cols = list(set(group_id_cols))

    # it is assumed that correlations are False for now, correlations for group ID hashing are assessed later on in
    # the process to re-hash the group ID with the correlation "peril_correlation_group" column name. This is because
    # the correlations is achieved later in the process leading to a chicken and egg problem
    # group_id_cols = process_group_id_cols(group_id_cols=group_id_cols,
    #                                       exposure_df_columns=list(exposure_df.columns),
    #                                       has_correlation_groups=False)

    # Should list of column names used to group_id be empty, revert to
    # default
    if len(group_id_cols) == 0:
        group_id_cols = GROUP_ID_COLS

    # Only add group col if not internal oasis col
    missing_group_id_cols = []
    for col in group_id_cols:
        if col in VALID_OASIS_GROUP_COLS:
            pass
        elif col not in exposure_df_gul_inputs_cols:
            missing_group_id_cols.append(col)
    exposure_df_gul_inputs_cols += missing_group_id_cols

    # Check if correlation group field used to drive group id
    # and test that it's present and poulated with integers
    correlation_group_id = CORRELATION_GROUP_ID
    correlation_field = correlation_group_id[0]
    correlation_check = False
    if group_id_cols == correlation_group_id:
        if correlation_field in exposure_df.columns:
            if exposure_df[correlation_field].astype('uint32').isnull().sum() == 0:
                correlation_check = True

    query_nonzero_tiv = " | ".join(f"({tiv_col} != 0)" for tiv_col in tiv_cols)
    exposure_df.loc[:, tiv_cols] = exposure_df.loc[:, tiv_cols].fillna(0.0)
    exposure_df.query(query_nonzero_tiv, inplace=True, engine='numexpr')

    gul_inputs_df = exposure_df[set(exposure_df_gul_inputs_cols).intersection(exposure_df.columns)]
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
            'modeldata': 'model_data'
        },
        inplace=True,
        copy=False  # Pandas copies column data by default on rename
    )

    # If the keys file relates to a complex/custom model then look for a
    # ``modeldata`` column in the keys file, and ignore the area peril
    # and vulnerability ID columns
    if 'model_data' in keys_df:
        keys_df['areaperil_id'] = keys_df['vulnerability_id'] = -1

    keys_df['loc_id']
    gul_inputs_df = merge_dataframes(
        gul_inputs_df,
        keys_df,
        join_on='loc_id',
        how='inner'
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

    gul_inputs_df[tiv_terms.values()].fillna(0, inplace=True)

    # Group the rows in the GUL inputs table by coverage type, and set the
    # IL terms (and BI coverage boolean) in each group and update the
    # corresponding frame section in the GUL inputs table
    gul_inputs_reformatted_chunks = []
    terms_found = set()
    for cov_type, cov_type_group in gul_inputs_df.groupby(by=['coverage_type_id'], sort=True):
        tiv_col = tiv_terms[cov_type]

        # Remove rows with null/0 TIV
        cov_type_group.query(f"{tiv_col} > 0.0", inplace=True)

        # Drop columns corresponding to other cov types
        other_cov_types = [v['id'] for v in SUPPORTED_COVERAGE_TYPES.values() if v['id'] != cov_type]
        other_cov_type_term_cols = (
            [v for k, v in tiv_terms.items() if k != cov_type] +
            get_fm_terms_oed_columns(fm_terms=fm_terms, levels=['site coverage'], term_group_ids=other_cov_types, terms=terms)
        )
        other_tiv_cols = list(set(tiv_terms.values()) - {tiv_col})
        cov_type_group.drop(
            columns=other_cov_type_term_cols + other_tiv_cols,
            errors="ignore",  # Ignore if any of these cols don't exist
            inplace=True
        )

        cov_type_group['is_bi_coverage'] = cov_type == SUPPORTED_COVERAGE_TYPES['bi']['id']

        # Convert null T&C values to 0
        # Move all T&C values from coverage-specific columns to generic columns
        # One row with four coverages is being transformed to four rows, one per coverage,
        # in this loop
        cov_type_terms = [t for t in terms if fm_terms[cov_level_id][cov_type].get(t)]
        cov_type_term_cols = get_fm_terms_oed_columns(fm_terms, levels=['site coverage'], term_group_ids=[cov_type], terms=cov_type_terms)
        column_mapping_dict = {
            generic_col: cov_col
            for generic_col, cov_col in zip(cov_type_term_cols, cov_type_terms) if generic_col in cov_type_group.columns
        }
        column_mapping_dict[tiv_col] = 'tiv'
        cov_type_group.rename(columns=column_mapping_dict, inplace=True, copy=False)
        cov_type_group.loc[:, ['tiv'] + list(column_mapping_dict.values())]
        terms_found.update(column_mapping_dict.values())
        cov_type_group['coverage_type_id'] = cov_type
        gul_inputs_reformatted_chunks.append(cov_type_group)
    # Concatenate chunks. Sort by index to preserve item_id order in generated outputs compared
    # to original code.
    gul_inputs_df = pd.concat(gul_inputs_reformatted_chunks)
    gul_inputs_df[terms_found].fillna(0., inplace=True)
    gul_inputs_df = gul_inputs_df.sort_index().reset_index(drop=True)
    term_int_found = list(set(gul_inputs_df.columns).intersection(set(term_cols_ints + terms_ints)))
    gul_inputs_df[term_int_found] = gul_inputs_df[term_int_found].fillna(0)
    # Set default values and data types for BI coverage boolean, TIV, deductibles and limit
    dtypes = {
        **{t: 'uint8' for t in term_cols_ints + terms_ints},
        **{'is_bi_coverage': 'bool'}
    }
    gul_inputs_df = set_dataframe_column_dtypes(gul_inputs_df, dtypes)

    if gul_inputs_df.empty:
        raise OasisException(
            'Empry gul_inputs_df dataframe after dropping rows with zero for tiv, '
            'please check the exposure input files'
        )

    # Set the item IDs and coverage IDs, and defaults and data types for
    # layer IDs and agg. IDs
    gul_inputs_df['item_id'] = factorize_ndarray(
        gul_inputs_df.loc[:, ['loc_id', 'peril_id', 'coverage_type_id']].values, col_idxs=range(3))[0]
    gul_inputs_df['coverage_id'] = factorize_ndarray(
        gul_inputs_df.loc[:, ['loc_id', 'coverage_type_id']].values, col_idxs=range(2))[0]

    gul_inputs_df.drop_duplicates(subset='item_id', inplace=True)
    dtypes = {
        **{t: 'int32' for t in ['item_id', 'coverage_id']},
        **{t: 'uint8' for t in terms_ints}
    }
    gul_inputs_df = set_dataframe_column_dtypes(gul_inputs_df, dtypes)

    # Set the group ID
    # If the group id is set according to the correlation group field then map this field
    # directly, otherwise create an index of the group id fields

    # Keep group_id consistance by lower casing column names and sorting
    group_id_cols.sort()
    group_id_cols_list = sorted([c.lower() for c in group_id_cols])  # list of case insessative column names
    group_id_cols_map = {c: c.lower() for c in group_id_cols}         # mapping from PascalCase -> 'lower_case'

    if correlation_check is True:
        gul_inputs_df['group_id'] = gul_inputs_df[correlation_group_id]

    elif hashed_group_id is False:

        if len(group_id_cols) > 1:
            gul_inputs_df['group_id'] = factorize_ndarray(
                gul_inputs_df.loc[:, group_id_cols].values,
                col_idxs=range(len(group_id_cols)),
                sort_opt=True
            )[0]

        else:
            gul_inputs_df['group_id'] = factorize_array(
                gul_inputs_df[group_id_cols[0]].values
            )[0]

    # this block gets fired if the hashed_group_id is True
    elif correlations is False:
        gul_inputs_df["group_id"] = (pd.util.hash_pandas_object(gul_inputs_df.rename(columns=group_id_cols_map)[group_id_cols_list],
                                                                index=False).to_numpy() >> 33)
    else:
        # do merge with peril correlation df
        gul_inputs_df = gul_inputs_df.merge(peril_correlation_group_df, left_on='peril_id', right_on='id').reset_index()
        gul_inputs_df["group_id"] = (pd.util.hash_pandas_object(gul_inputs_df.rename(columns=group_id_cols_map)[group_id_cols_list],
                                                                index=False).to_numpy() >> 33)

    gul_inputs_df['group_id'] = gul_inputs_df['group_id'].astype('uint32')

    # Select only required columns
    # Order here matches test output expectations
    usecols = (
        ['loc_id', portfolio_num, acc_num, loc_num] +
        ([SOURCE_IDX['loc']] if SOURCE_IDX['loc'] in gul_inputs_df else []) +
        ['peril_id', 'coverage_type_id', 'tiv', 'areaperil_id', 'vulnerability_id'] +
        terms +
        (['model_data'] if 'model_data' in gul_inputs_df else []) +
        ['is_bi_coverage', 'group_id', 'coverage_id', 'item_id', 'status']
    )
    if correlations is True:
        usecols += ["peril_correlation_group", "correlation_value"]

    usecols = [col for col in usecols if col in gul_inputs_df]
    gul_inputs_df = gul_inputs_df[usecols]

    gul_inputs_df = gul_inputs_df.sort_values("item_id")
    gul_inputs_df = gul_inputs_df.reindex(columns=list(gul_inputs_df))

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
    oasis_files_prefixes=copy.deepcopy(OASIS_FILES_PREFIXES['gul']),
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

    if correlations_df is None:
        correlations_df = pd.DataFrame(columns=['item_id', 'peril_correlation_group', 'correlation_value'])

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
        if oasis_files_prefixes.get('complex_items'):
            oasis_files_prefixes.pop('complex_items')

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
