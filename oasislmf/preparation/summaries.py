__all__ = [
    'get_summary_mapping',
    'generate_summaryxref_files',
    'merge_oed_to_mapping',
    'write_exposure_summary',
    'get_exposure_summary',
    'get_useful_summary_cols',
    'get_xref_df',
    'write_summary_levels',
    'write_mapping_file',
]

import pathlib

import io
import json
import os

import numpy as np
import pandas as pd

from ..utils.coverages import SUPPORTED_COVERAGE_TYPES
from ..utils.data import (
    factorize_dataframe,
    factorize_ndarray,
    get_dataframe,
    get_json,
    merge_dataframes,
    set_dataframe_column_dtypes,
    fill_na_with_categoricals
)
from ..utils.defaults import (
    SOURCE_IDX,
    SUMMARY_MAPPING,
    SUMMARY_OUTPUT,
    SUMMARY_TOP_LEVEL_COLS,
    get_default_exposure_profile,
)
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import as_path
from ..utils.status import OASIS_KEYS_STATUS, OASIS_KEYS_STATUS_MODELLED

MAP_SUMMARY_DTYPES = {
    'loc_id': 'int',
    SOURCE_IDX['loc']: 'int',
    SOURCE_IDX['acc']: 'int',
    'item_id': 'int',
    'layer_id': 'int',
    'coverage_id': 'int',
    'peril_id': 'category',
    'agg_id': 'int',
    'output_id': 'int',
    'coverage_type_id': 'int',
    'tiv': 'float',
    'building_id': 'int',
    'risk_id': 'int',
}


def get_useful_summary_cols(oed_hierarchy):
    return [
        oed_hierarchy['accnum']['ProfileElementName'],
        oed_hierarchy['locnum']['ProfileElementName'],
        'loc_id',
        oed_hierarchy['polnum']['ProfileElementName'],
        oed_hierarchy['portnum']['ProfileElementName'],
        SOURCE_IDX['loc'],
        SOURCE_IDX['acc'],
        'item_id',
        'layer_id',
        'coverage_id',
        'peril_id',
        'agg_id',
        'output_id',
        'coverage_type_id',
        'tiv',
        'building_id',
        'risk_id'
    ]


def get_xref_df(il_inputs_df):
    top_level_layers_df = il_inputs_df.loc[il_inputs_df['level_id'] == il_inputs_df['level_id'].max(), ['top_agg_id'] + SUMMARY_TOP_LEVEL_COLS]
    bottom_level_layers_df = il_inputs_df[il_inputs_df['level_id'] == 1]
    bottom_level_layers_df.drop(columns=SUMMARY_TOP_LEVEL_COLS, inplace=True)
    return (merge_dataframes(bottom_level_layers_df, top_level_layers_df, join_on=['top_agg_id'])
            .drop_duplicates(subset=['gul_input_id', 'layer_id'], keep='first')
            .sort_values(['gul_input_id', 'layer_id'])
            )


@oasis_log
def get_summary_mapping(inputs_df, oed_hierarchy, is_fm_summary=False):
    """
    Create a DataFrame with linking information between Ktools `OasisFiles`
    And the Exposure data

    :param inputs_df: datafame from gul_inputs.get_gul_input_items(..)  / il_inputs.get_il_input_items(..)
    :type inputs_df: pandas.DataFrame

    :param is_fm_summary: Indicates whether an FM summary mapping is required
    :type is_fm_summary: bool

    :return: Subset of columns from gul_inputs_df / il_inputs_df
    :rtype: pandas.DataFrame
    """
    # Case GUL+FM (based on il_inputs_df)
    if is_fm_summary:
        summary_mapping = get_xref_df(inputs_df)
        summary_mapping['agg_id'] = summary_mapping['gul_input_id']
        summary_mapping = summary_mapping.reindex(sorted(summary_mapping.columns, key=str.lower), axis=1)
        summary_mapping['output_id'] = factorize_ndarray(
            summary_mapping.loc[:, ['gul_input_id', 'layer_id']].values,
            col_idxs=range(2)
        )[0]

    # GUL Only
    else:
        summary_mapping = inputs_df.copy(deep=True)
        summary_mapping['layer_id'] = 1
        summary_mapping['agg_id'] = summary_mapping['item_id']

    summary_mapping.drop(
        [c for c in summary_mapping.columns if c not in get_useful_summary_cols(oed_hierarchy)],
        axis=1,
        inplace=True
    )
    acc_num = oed_hierarchy['accnum']['ProfileElementName']
    loc_num = oed_hierarchy['locnum']['ProfileElementName']
    policy_num = oed_hierarchy['polnum']['ProfileElementName']
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName']

    dtypes = {
        **{t: 'str' for t in [portfolio_num, policy_num, acc_num, loc_num, 'peril_id']},
        **{t: 'uint8' for t in ['coverage_type_id']},
        **{t: 'uint32' for t in [SOURCE_IDX['loc'], SOURCE_IDX['acc'], 'loc_id', 'item_id', 'layer_id', 'coverage_id', 'agg_id', 'output_id',
                                 'building_id', 'risk_id']},
        **{t: 'float64' for t in ['tiv']}
    }
    summary_mapping = set_dataframe_column_dtypes(summary_mapping, dtypes)
    return summary_mapping


def merge_oed_to_mapping(summary_map_df, exposure_df, oed_column_join, oed_column_info):
    """
    Create a factorized col (summary ids) based on a list of oed column names

    :param :summary_map_df dataframe return from get_summary_mapping
    :type summary_map_df: pandas.DataFrame

    :param exposure_df: Summary map file path
    :type exposure_df: pandas.DataFrame

    :param oed_column_join: column to join on
    :type oed_column_join: list

    :param oed_column_info: Dictionary of columns to pick from exposure_df and their default value
    :type oed_column_info: dict

    {'Col_A': 0, 'Col_B': 1, 'Col_C': 2}

    :return: New DataFrame of summary_map_df + exposure_df merged on exposure index
    :rtype: pandas.DataFrame
    """

    column_set = set(oed_column_info)
    columns_found = [c for c in column_set if c in exposure_df.columns and c not in summary_map_df.columns]
    columns_missing = list(set(column_set) - set(columns_found))

    new_summary_map_df = merge_dataframes(summary_map_df, exposure_df.loc[:, columns_found + oed_column_join], join_on=oed_column_join, how='inner')
    for col, default in oed_column_info.items():
        if col in columns_missing:
            new_summary_map_df[col] = default
    fill_na_with_categoricals(new_summary_map_df, oed_column_info)

    return new_summary_map_df


def group_by_oed(oed_col_group, summary_map_df, exposure_df, sort_by, accounts_df=None):
    """
    Adds list of OED fields from `column_set` to summary map file

    :param :summary_map_df dataframe return from get_summary_mapping
    :type summary_map_df: pandas.DataFrame

    :param exposure_df: DataFrame loaded from location.csv
    :type exposure_df: pandas.DataFrame

    :param accounts_df: DataFrame loaded from accounts.csv
    :type accounts_df: pandas.DataFrame

    :return: subset of columns from exposure_df to merge
    :rtype: list

        summary_ids[0] is an int list 1..n  array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, ... ])
        summary_ids[1] is an array of values used to factorize  `array(['Layer1', 'Layer2'], dtype=object)`
    """
    oed_cols = oed_col_group  # All requred columns
    unmapped_cols = [c for c in oed_cols if c not in summary_map_df.columns]  # columns which in locations / Accounts file
    mapped_cols = [c for c in oed_cols + [SOURCE_IDX['loc'], SOURCE_IDX['acc'], sort_by]
                   if c in summary_map_df.columns]  # Columns already in summary_map_df
    tiv_cols = ['tiv', 'loc_id', 'building_id', 'coverage_type_id']

    # Extract mapped_cols from summary_map_df
    summary_group_df = summary_map_df.loc[:, list(set(tiv_cols).union(mapped_cols))]

    # Search Loc / Acc files and merge in remaing
    if unmapped_cols is not []:
        # Location file columns
        exposure_cols = [c for c in unmapped_cols if c in exposure_df.columns]
        exposure_col_df = exposure_df.loc[:, exposure_cols + [SOURCE_IDX['loc']]]
        summary_group_df = merge_dataframes(summary_group_df, exposure_col_df, join_on=SOURCE_IDX['loc'], how='left')

        # Account file columns
        if isinstance(accounts_df, pd.DataFrame):
            accounts_cols = [c for c in unmapped_cols if c in set(accounts_df.columns) - set(exposure_df.columns)]
            if accounts_cols:
                accounts_col_df = accounts_df.loc[:, accounts_cols + [SOURCE_IDX['acc']]]
                summary_group_df = merge_dataframes(summary_group_df, accounts_col_df, join_on=SOURCE_IDX['acc'], how='left')

    fill_na_with_categoricals(summary_group_df, 0)
    summary_group_df.sort_values(by=[sort_by], inplace=True)
    summary_ids = factorize_dataframe(summary_group_df, by_col_labels=oed_cols)
    summary_tiv = summary_group_df.drop_duplicates(['loc_id', 'building_id', 'coverage_type_id'] + oed_col_group,
                                                   keep='first').groupby(oed_col_group, observed=True).agg({'tiv': np.sum})

    return summary_ids[0], summary_ids[1], summary_tiv


@oasis_log
def write_summary_levels(exposure_df, accounts_df, exposure_data, target_dir):
    '''
    Json file with list Available / Recommended columns for use in the summary reporting

    Available: Columns which exists in input files and has at least one non-zero / NaN value
    Recommended: Columns which are available + also in the list of `useful` groupings SUMMARY_LEVEL_LOC

    {
        'GUL': {
            'available': ['AccNumber',
                         'LocNumber',
                         'istenant',
                         'buildingid',
                         'countrycode',
                         'latitude',
                         'longitude',
                         'streetaddress',
                         'postalcode',
                         'occupancycode',
                         'constructioncode',
                         'locperilscovered',
                         'BuildingTIV',
                         'ContentsTIV',
                         'BITIV',
                         'PortNumber'],

        'IL': {
                ... etc ...
        }
    }
    '''
    # Manage internal columns, (Non-OED exposure input)
    int_excluded_cols = ['loc_id', SOURCE_IDX['loc']]
    desc_non_oed = 'Not an OED field'
    int_oasis_cols = {
        'coverage_type_id': 'Oasis coverage type',
        'peril_id': 'OED peril code',
        'coverage_id': 'Oasis coverage identifier',
    }

    # GUL perspective (loc columns only)
    l_col_list = exposure_df.replace(0, np.nan).dropna(how='any', axis=1).columns.to_list()
    l_col_info = exposure_data.get_input_fields('Loc')
    gul_avail = {k: l_col_info[k.lower()]["Type & Description"] if k.lower() in l_col_info else desc_non_oed
                 for k in set([c for c in l_col_list]).difference(int_excluded_cols)}

    # IL perspective (join of acc + loc col with no dups)
    il_avail = {}
    if accounts_df is not None:
        a_col_list = accounts_df.loc[:, ~accounts_df.isnull().all()].columns.to_list()
        a_col_info = exposure_data.get_input_fields('Acc')
        a_avail = set([c for c in a_col_list])
        il_avail = {k: a_col_info[k.lower()]["Type & Description"] if k.lower() in a_col_info else desc_non_oed
                    for k in a_avail.difference(gul_avail.keys())}

    # Write JSON
    gul_summary_lvl = {'GUL': {'available': {**gul_avail, **il_avail, **int_oasis_cols}}}
    il_summary_lvl = {'IL': {'available': {**gul_avail, **il_avail, **int_oasis_cols}}} if il_avail else {}
    with io.open(os.path.join(target_dir, 'exposure_summary_levels.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps({**gul_summary_lvl, **il_summary_lvl}, sort_keys=True, ensure_ascii=False, indent=4))


@oasis_log
def write_mapping_file(sum_inputs_df, target_dir, is_fm_summary=False):
    """
    Writes a summary map file, used to build summarycalc xref files.

    :param summary_mapping: dataframe return from get_summary_mapping
    :type summary_mapping: pandas.DataFrame

    :param sum_mapping_fp: Summary map file path
    :type sum_mapping_fp: str

    :param is_fm_summary: Indicates whether an FM summary mapping is required
    :type is_fm_summary: bool

    :return: Summary xref file path
    :rtype: str
    """
    target_dir = as_path(
        target_dir,
        'Target IL input files directory',
        is_dir=True,
        preexists=False
    )

    # Set chunk size for writing the CSV files - default is max 20K, min 1K
    chunksize = min(2 * 10 ** 5, max(len(sum_inputs_df), 1000))

    if is_fm_summary:
        sum_mapping_fp = os.path.join(target_dir, SUMMARY_MAPPING['fm_map_fn'])
    else:
        sum_mapping_fp = os.path.join(target_dir, SUMMARY_MAPPING['gul_map_fn'])
    try:
        sum_inputs_df.to_csv(
            path_or_buf=sum_mapping_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(sum_mapping_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_mapping_file'", e)

    return sum_mapping_fp


def get_column_selection(summary_set):
    """
    Given a analysis_settings summary definition, return either
        1. the set of OED columns requested to group by
        2. If no information key 'oed_fields', then group all outputs into a single summary_set

    :param summary_set: summary group dictionary from the `analysis_settings.json`
    :type summary_set: dict

    :return: List of selected OED columns to create summary groups from
    :rtype: list
    """
    if "oed_fields" not in summary_set:
        return []
    if not summary_set["oed_fields"]:
        return []

    # Use OED column list set in analysis_settings file
    elif isinstance(summary_set['oed_fields'], list) and len(summary_set['oed_fields']) > 0:
        return [c for c in summary_set['oed_fields']]
    elif isinstance(summary_set['oed_fields'], str) and len(summary_set['oed_fields']) > 0:
        return [summary_set['oed_fields']]
    else:
        raise OasisException(
            'Error processing settings file: "oed_fields" '
            'is expected to be a list of strings, not {}'.format(type(summary_set['oed_fields']))
        )


def get_ri_settings(run_dir):
    """
    Return the contents of ri_layers.json

    Example:
    {
        "1": {
            "inuring_priority": 1,
            "risk_level": "LOC",
            "directory": "  ... /runs/ProgOasis-20190501145127/RI_1"
        }
    }

    :param run_dir: The file path of the model run directory
    :type run_dir: str

    :return: metadata for the Reinsurance layers
    :rtype: dict
    """
    return get_json(src_fp=os.path.join(run_dir, 'ri_layers.json'))


def write_df_to_csv_file(df, target_dir, filename):
    """
    Write a generated summary xref dataframe to disk in csv format.

    :param df: The dataframe output of get_df( .. )
    :type df:  pandas.DataFrame

    :param target_dir: Abs directory to write a summary_xref file
    :type target_dir:  str

    :param filename: Name of output file
    :type filename:  str
    """
    target_dir = as_path(target_dir, 'Input files directory', is_dir=True, preexists=False)
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    chunksize = min(2 * 10 ** 5, max(len(df), 1000))
    csv_fp = os.path.join(target_dir, filename)
    try:
        df.to_csv(
            path_or_buf=csv_fp,
            encoding='utf-8',
            mode=('w'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_df_to_csv_file'", e)

    return csv_fp


def write_df_to_parquet_file(df, target_dir, filename):
    """
    Write a generated summary xref dataframe to disk in parquet format.

    :param df: The dataframe output of get_df( .. )
    :type df: pandas.DataFrame

    :param target_dir: Abs directory to write a summary_xref file
    :type target_dir: str

    :param filename: Name of output file
    :type filename: str
    """
    target_dir = as_path(
        target_dir, 'Output files directory', is_dir=True, preexists=False
    )
    parquet_fp = os.path.join(target_dir, filename)
    try:
        df.to_parquet(path=parquet_fp, engine='pyarrow')
    except (IOError, OSError) as e:
        raise OasisException(
            "Exception raised in 'write_df_to_parquet_file'", e
        )

    return parquet_fp


def get_summary_xref_df(
    map_df, exposure_df, accounts_df, summaries_info_dict, summaries_type,
    id_set_index='output_id'
):
    """
    Create a Dataframe for either gul / il / ri  based on a section
    from the analysis settings


    :param map_df: Summary Map dataframe (GUL / IL)
    :type map_df:  pandas.DataFrame

    :param exposure_df: Location OED data
    :type exposure_df:  pandas.DataFrame

    :param accounts_df: Accounts OED data
    :type accounts_df:  pandas.DataFrame

    :param summaries_info_dict: list of dictionary definitionfor a summary group from the analysis_settings file
    :type summaries_info_dict:  list

    [{
        "summarycalc": true,
        "eltcalc": true,
        "aalcalc": true,
        "pltcalc": true,
        "id": 1,
        "oed_fields": [],
        "lec_output": true,
        "leccalc": {
          "return_period_file": true,
          "outputs": {
            "full_uncertainty_aep": true,
            "full_uncertainty_oep": true,
            "wheatsheaf_aep": true,
            "wheatsheaf_oep": true
          }
        }
      },

      ...
     ]

    :param summaries_type: Text label to use as key in summary description either ['gul', 'il', 'ri']
    :type summaries_type: String

    :return summaryxref_df: Dataframe containing abstracted summary data for ktools
    :rtypwrite_xref_filee: pandas.DataFrame

    :return summary_desc: dictionary of dataFrames listing what summary_ids map to
    :rtype: dictionary
    """
    summaryxref_df = pd.DataFrame()
    summary_desc = {}

    all_cols = set(map_df.columns.to_list() + exposure_df.columns.to_list())
    if isinstance(accounts_df, pd.DataFrame):
        all_cols.update(accounts_df.columns.to_list())

    # Extract the summary id index column depending on id_set_index
    map_df.sort_values(id_set_index, inplace=True)
    ids_set_df = map_df.loc[:, [id_set_index]].rename(columns={'output_id': "output"})

    # For each granularity build a set grouping
    for summary_set in summaries_info_dict:
        summary_set_df = ids_set_df
        cols_group_by = get_column_selection(summary_set)
        file_extension = 'csv'
        if summary_set.get('ord_output', {}).get('parquet_format'):
            file_extension = 'parquet'
        desc_key = '{}_S{}_summary-info.{}'.format(
            summaries_type, summary_set['id'], file_extension
        )

        # an empty intersection means no selected columns from the input data
        if not set(cols_group_by).intersection(all_cols):

            # is the intersection empty because the columns don't exist?
            if set(cols_group_by).difference(all_cols):
                err_msg = 'Input error: Summary set columns missing from the input files: {}'.format(
                    set(cols_group_by).difference(all_cols))
                raise OasisException(err_msg)

            # Fall back to setting all in single group
            summary_set_df['summary_id'] = 1
            summary_desc[desc_key] = pd.DataFrame(data=['All-Risks'], columns=['_not_set_'])
            summary_desc[desc_key].insert(loc=0, column='summary_id', value=1)
            summary_desc[desc_key].insert(loc=len(summary_desc[desc_key].columns), column='tiv',
                                          value=map_df.drop_duplicates(['building_id', 'loc_id', 'coverage_type_id'], keep='first').tiv.sum())
        else:
            (
                summary_set_df['summary_id'],
                set_values,
                tiv_values
            ) = group_by_oed(cols_group_by, map_df, exposure_df, id_set_index, accounts_df)

            # Build description file
            summary_desc_df = pd.DataFrame(data=list(set_values), columns=cols_group_by)
            summary_desc_df.insert(loc=0, column='summary_id', value=range(1, len(set_values) + 1))
            summary_desc[desc_key] = pd.merge(summary_desc_df, tiv_values, left_on=cols_group_by, right_on=cols_group_by)

        # Appends summary set to '__summaryxref.csv'
        summary_set_df['summaryset_id'] = summary_set['id']
        summaryxref_df = pd.concat([summaryxref_df, summary_set_df.drop_duplicates()], sort=True, ignore_index=True)

    dtypes = {
        t: 'uint32' for t in ['coverage_id', 'summary_id', 'summaryset_id']
    }
    summaryxref_df = set_dataframe_column_dtypes(summaryxref_df, dtypes)
    return summaryxref_df, summary_desc


@oasis_log
def generate_summaryxref_files(
    location_df, account_df, model_run_fp, analysis_settings, il=False,
    ri=False, rl=False, gul_item_stream=False, fmpy=False
):
    """
    Top level function for creating the summaryxref files from the manager.py

    :param model_run_fp: Model run directory file path
    :type model_run_fp:  str

    :param analysis_settings: Model analysis settings file
    :type analysis_settings:  dict

    :param il: Boolean to indicate the insured loss level mode - false if the
               source accounts file path not provided to Oasis files gen.
    :type il: bool

    :param ri: Boolean to indicate the RI loss level mode - false if the
               source accounts file path not provided to Oasis files gen.
    :type ri: bool

    :param rl: Boolean to indicate the RL loss level mode - false if the
               source accounts file path not provided to Oasis files gen.
    :type rl: bool

    :param gul_items: Boolean to gul to use item_id instead of coverage_id
    :type gul_items: bool

    :param fmpy: Boolean to indicate whether fmpy python version will be used
    :type fmpy: bool
    """

    # Boolean checks for summary generation types (gul / il / ri)
    gul_summaries = all([
        analysis_settings.get('gul_output'),
        analysis_settings.get('gul_summaries'),
    ])
    il_summaries = all([
        analysis_settings.get('il_output'),
        analysis_settings.get('il_summaries'),
        il,
    ])
    ri_summaries = all([
        analysis_settings.get('ri_output'),
        analysis_settings.get('ri_summaries'),
        ri,
    ])
    rl_summaries = all([
        analysis_settings.get('rl_output'),
        analysis_settings.get('rl_summaries'),
        rl,
    ])

    # Load il_map if present
    if il_summaries or ri_summaries or rl_summaries:
        if account_df is None:
            raise OasisException('No account file found.')
        il_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['fm_map_fn'])
        il_map_df = get_dataframe(
            src_fp=il_map_fp,
            lowercase_cols=False,
            col_dtypes=MAP_SUMMARY_DTYPES,
            empty_data_error_msg='No summary map file found.',
        )
        il_map_df = il_map_df[list(set(il_map_df).intersection(MAP_SUMMARY_DTYPES))]

        if gul_summaries:
            gul_map_df = il_map_df
            gul_map_df['item_id'] = gul_map_df['agg_id']

    elif gul_summaries:
        gul_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['gul_map_fn'])
        gul_map_df = get_dataframe(
            src_fp=gul_map_fp,
            lowercase_cols=False,
            col_dtypes=MAP_SUMMARY_DTYPES,
            empty_data_error_msg='No summary map file found.',
        )
        gul_map_df = gul_map_df[list(set(gul_map_df).intersection(MAP_SUMMARY_DTYPES))]

    if gul_summaries:
        # Load GUL summary map
        id_set_index = 'item_id' if gul_item_stream else 'coverage_id'
        gul_summaryxref_df, gul_summary_desc = get_summary_xref_df(
            gul_map_df,
            location_df,
            account_df,
            analysis_settings['gul_summaries'],
            'gul',
            id_set_index
        )
        # Write Xref file
        write_df_to_csv_file(gul_summaryxref_df, os.path.join(model_run_fp, 'input'), SUMMARY_OUTPUT['gul'])

        # Write summary_id description files
        for desc_key in gul_summary_desc:
            if desc_key.split('.')[-1] == 'parquet':
                write_df_to_parquet_file(gul_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)
            else:
                write_df_to_csv_file(gul_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)

    if il_summaries:
        # Load FM summary map
        il_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['fm_map_fn'])
        il_map_df = get_dataframe(
            src_fp=il_map_fp,
            lowercase_cols=False,
            col_dtypes=MAP_SUMMARY_DTYPES,
            empty_data_error_msg='No summary map file found.',
        )
        il_map_df = il_map_df[list(set(il_map_df).intersection(MAP_SUMMARY_DTYPES))]

        il_summaryxref_df, il_summary_desc = get_summary_xref_df(
            il_map_df,
            location_df,
            account_df,
            analysis_settings['il_summaries'],
            'il'
        )
        # Write Xref file
        write_df_to_csv_file(il_summaryxref_df, os.path.join(model_run_fp, 'input'), SUMMARY_OUTPUT['il'])

        # Write summary_id description files
        for desc_key in il_summary_desc:
            if desc_key.split('.')[-1] == 'parquet':
                write_df_to_parquet_file(il_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)
            else:
                write_df_to_csv_file(il_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)

    if ri_summaries or rl_summaries:
        if ('il_summaries' not in analysis_settings) or (not il_summaries):
            il_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['fm_map_fn'])
            il_map_df = get_dataframe(
                src_fp=il_map_fp,
                lowercase_cols=False,
                col_dtypes=MAP_SUMMARY_DTYPES,
                empty_data_error_msg='No summary map file found.',
            )
            il_map_df = il_map_df[list(set(il_map_df).intersection(MAP_SUMMARY_DTYPES))]

        ri_summaryxref_df, ri_summary_desc = get_summary_xref_df(
            il_map_df,
            location_df,
            account_df,
            analysis_settings['ri_summaries'],
            'ri'
        )
        # Write Xref file for each inuring priority where output has been requested
        ri_settings = get_ri_settings(os.path.join(model_run_fp, 'input'))
        ri_layers = {int(x) for x in ri_settings}
        max_layer = max(ri_layers)
        ri_inuring_priorities = set(analysis_settings.get('ri_inuring_priorities', []))
        ri_inuring_priorities.add(max_layer)
        if not fmpy:
            if len(ri_inuring_priorities) > 1:
                raise OasisException('Outputs at intermediate inuring priorities not compatible with fmcalc c++ option.')
        if not ri_inuring_priorities.issubset(ri_layers):
            ri_missing_layers = ri_inuring_priorities.difference(ri_layers)
            ri_missing_layers = [str(layer) for layer in ri_missing_layers]
            missing_layers = ', '.join(ri_missing_layers[:-1])
            missing_layers += ' and ' * (len(ri_missing_layers) > 1) + ri_missing_layers[-1]
            missing_layers = ('priority ' if len(ri_missing_layers) == 1 else 'priorities ') + missing_layers
            raise OasisException(f'Requested outputs for inuring priorities {missing_layers} lie outside of scope.')
        # If requested, gross RL output at every inuring priority
        if rl_summaries:
            ri_inuring_priorities = ri_layers
        for inuring_priority in ri_inuring_priorities:
            summary_ri_fp = os.path.join(
                model_run_fp, 'input', os.path.basename(ri_settings[str(inuring_priority)]['directory']))
            write_df_to_csv_file(ri_summaryxref_df, summary_ri_fp, SUMMARY_OUTPUT['il'])

        # Write summary_id description files
        for desc_key in ri_summary_desc:
            if desc_key.split('.')[-1] == 'parquet':
                write_df_to_parquet_file(ri_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)
            else:
                write_df_to_csv_file(ri_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)


def get_exposure_summary_by_status(df, exposure_summary, peril_id, status):
    """
    Populate dictionary of TIV and number of locations, grouped by peril and
    validity respectively

    :param df: dataframe from gul_inputs.get_gul_input_items(..)
    :type df: pandas.DataFrame

    :param peril_id: Descriptive OED peril key, e.g. "WTC"
    :type peril_id: str

    :param status: status returned by lookup ('success', 'fail' or 'nomatch')
    :type status: str

    :return: populated exposure_summary dictionary
    :rtype: dict
    """
    # Separate TIVs and number of distinct locations by coverage type and acquire sum
    for coverage_type in SUPPORTED_COVERAGE_TYPES:
        tiv_sum = df.loc[
            (df['peril_id'] == peril_id) & (df['coverage_type_id'] == SUPPORTED_COVERAGE_TYPES[coverage_type]['id']),
            'tiv'
        ].sum()
        tiv_sum = float(tiv_sum)
        exposure_summary[peril_id][status]['tiv_by_coverage'][coverage_type] = tiv_sum
        exposure_summary[peril_id][status]['tiv'] += tiv_sum

        loc_count = df.loc[
            (df['peril_id'] == peril_id) & (df['coverage_type_id'] == SUPPORTED_COVERAGE_TYPES[coverage_type]['id']),
            'loc_id'
        ].drop_duplicates().count()
        loc_count = int(loc_count)
        exposure_summary[peril_id][status]['number_of_locations_by_coverage'][coverage_type] = loc_count

    # Find number of locations
    loc_count = df.loc[df['peril_id'] == peril_id, 'loc_id'].drop_duplicates().count()
    loc_count = int(loc_count)
    exposure_summary[peril_id][status]['number_of_locations'] = loc_count

    return exposure_summary


def get_exposure_summary_all(df, exposure_summary, peril_id):
    """
    Populate dictionary of TIV and number of locations, grouped by peril

    :param df: dataframe from gul_inputs.get_gul_input_items(..)
    :type df: pandas.DataFrame

    :param exposure_summary: dictionary to populate created in write_exposure_summary(..)
    :type exposure_summary: dict

    :param peril_id: Descriptive OED peril key, e.g. "WTC"
    :type peril_id: str

    :return: populated exposure_summary dictionary
    :rtype: dict
    """

    # Separate TIVs and number of distinct locations by coverage type and acquire sum
    for coverage_type in SUPPORTED_COVERAGE_TYPES:
        tiv_sum = df.loc[
            (df['peril_id'] == peril_id) &
            (df['coverage_type_id'] == SUPPORTED_COVERAGE_TYPES[coverage_type]['id']),
            'tiv'
        ].sum()
        tiv_sum = float(tiv_sum)
        exposure_summary[peril_id]['all']['tiv_by_coverage'][coverage_type] = tiv_sum
        exposure_summary[peril_id]['all']['tiv'] += tiv_sum

        loc_count = df.loc[
            (df['peril_id'] == peril_id) &
            (df['coverage_type_id'] == SUPPORTED_COVERAGE_TYPES[coverage_type]['id']),
            'loc_id'
        ].drop_duplicates().count()
        loc_count = int(loc_count)
        exposure_summary[peril_id]['all']['number_of_locations_by_coverage'][coverage_type] = loc_count

    # Find number of locations total
    loc_count = df.loc[df['peril_id'] == peril_id, 'loc_id'].drop_duplicates().count()
    loc_count = int(loc_count)
    exposure_summary[peril_id]['all']['number_of_locations'] = loc_count

    # Find number of locations by coverage type

    return exposure_summary


@oasis_log
def get_exposure_totals(df):
    """
    Return dictionary with total TIVs and number of locations

    :param df: dataframe `df_summary_peril` from `get_exposure_summary`
    :type df: pandas.DataFrame

    :return: totals section for exposure_summary dictionary
    :rtype: dict
    """

    dedupe_cols = ['loc_id', 'coverage_type_id']

    within_scope_tiv = df[df.status.isin(OASIS_KEYS_STATUS_MODELLED)].drop_duplicates(subset=dedupe_cols)['tiv'].sum()
    within_scope_num = len(df[df.status.isin(OASIS_KEYS_STATUS_MODELLED)]['loc_id'].unique())

    outside_scope_tiv = df[~df.status.isin(OASIS_KEYS_STATUS_MODELLED)].drop_duplicates(subset=dedupe_cols)['tiv'].sum()
    outside_scope_num = len(df[~df.status.isin(OASIS_KEYS_STATUS_MODELLED)]['loc_id'].unique())

    portfolio_tiv = df.drop_duplicates(subset=dedupe_cols)['tiv'].sum()
    portfolio_num = len(df['loc_id'].unique())

    return {
        "modelled": {
            "tiv": within_scope_tiv,
            "number_of_locations": within_scope_num
        },
        "not-modelled": {
            "tiv": outside_scope_tiv,
            "number_of_locations": outside_scope_num
        },
        "portfolio": {
            "tiv": portfolio_tiv,
            "number_of_locations": portfolio_num
        }
    }


@oasis_log
def get_exposure_summary(
        exposure_df,
        keys_df,
        exposure_profile=get_default_exposure_profile(),
):
    """
    Create exposure summary as dictionary of TIVs and number of locations
    grouped by peril and validity respectively. returns a python dict().

    :param exposure_df: source exposure dataframe
    :type exposure df: pandas.DataFrame

    :param keys_df: dataFrame holding keys data (success and errors)
    :type keys_errors_df: pandas.DataFrame

    :param exposure_profile: profile defining exposure file
    :type exposure_profile: dict

    :return: Exposure summary dictionary
    :rtype: dict
    """

    # get location tivs by coveragetype
    df_summary = []

    for field in exposure_profile:
        if 'FMTermType' in exposure_profile[field].keys():
            if exposure_profile[field]['FMTermType'] == 'TIV':
                cov_name = exposure_profile[field]['ProfileElementName']
                coverage_type_id = exposure_profile[field]['CoverageTypeID']
                tmp_df = exposure_df[['loc_id', cov_name]]
                tmp_df.columns = ['loc_id', 'tiv']
                tmp_df['coverage_type_id'] = coverage_type_id
                df_summary.append(tmp_df)
    df_summary = pd.concat(df_summary)

    # get all perils
    peril_list = keys_df['peril_id'].drop_duplicates().to_list()

    df_summary_peril = []
    for peril_id in peril_list:
        tmp_df = df_summary.copy()
        tmp_df['peril_id'] = peril_id
        df_summary_peril.append(tmp_df)
    df_summary_peril = pd.concat(df_summary_peril)

    df_summary_peril = df_summary_peril.merge(keys_df, how='left', on=['loc_id', 'coverage_type_id', 'peril_id'])
    no_return = OASIS_KEYS_STATUS['noreturn']['id']
    df_summary_peril['status'] = df_summary_peril['status'].fillna(no_return)

    # Compile summary of exposure data
    exposure_summary = {}

    # Create totals section
    exposure_summary['total'] = get_exposure_totals(df_summary_peril)

    for peril_id in peril_list:

        exposure_summary[peril_id] = {}
        # Create dictionary structure for all and each validity status
        for status in ['all'] + list(OASIS_KEYS_STATUS.keys()):
            exposure_summary[peril_id][status] = {}
            exposure_summary[peril_id][status]['tiv'] = 0.0
            exposure_summary[peril_id][status]['tiv_by_coverage'] = {}
            exposure_summary[peril_id][status]['number_of_locations'] = 0
            exposure_summary[peril_id][status]['number_of_locations_by_coverage'] = {}
            # Fill exposure summary dictionary
            if status == 'all':
                exposure_summary = get_exposure_summary_all(
                    df_summary_peril,
                    exposure_summary,
                    peril_id
                )
            else:
                exposure_summary = get_exposure_summary_by_status(
                    df_summary_peril[df_summary_peril['status'] == status],
                    exposure_summary,
                    peril_id,
                    status
                )

    return exposure_summary


@oasis_log
def write_gul_errors_map(
        target_dir,
        exposure_df,
        keys_errors_df
):
    """
    Create csv file to help map keys errors back to original exposures.

    :param target_dir: directory on disk to write csv file
    :type target_dir: str

    :param exposure_df: source exposure dataframe
    :type exposure df: pandas.DataFrame

    :param keys_errors_df: keys errors dataframe
    :type keys_errors_df: pandas.DataFrame
    """

    cols = ['loc_id', 'PortNumber', 'AccNumber', 'LocNumber', 'peril_id', 'coverage_type_id', 'tiv', 'status', 'message']
    gul_error_map_fp = os.path.join(target_dir, 'gul_errors_map.csv')

    exposure_id_cols = ['loc_id', 'PortNumber', 'AccNumber', 'LocNumber']
    keys_error_cols = ['loc_id', 'peril_id', 'coverage_type_id', 'status', 'message']
    tiv_maps = {1: 'BuildingTIV', 2: 'OtherTIV', 3: 'ContentsTIV', 4: 'BITIV'}
    exposure_cols = exposure_id_cols + list(tiv_maps.values())

    keys_errors_df.columns = keys_error_cols

    gul_inputs_errors_df = exposure_df[exposure_cols].merge(keys_errors_df[keys_error_cols], on=['loc_id'])
    gul_inputs_errors_df['tiv'] = 0.0
    for cov_type in tiv_maps:
        tiv_field = tiv_maps[cov_type]
        gul_inputs_errors_df['tiv'] = np.where(
            gul_inputs_errors_df['coverage_type_id'] == cov_type,
            gul_inputs_errors_df[tiv_field],
            gul_inputs_errors_df['tiv']
        )
    gul_inputs_errors_df['tiv'] = gul_inputs_errors_df['tiv'].fillna(0.0)

    gul_inputs_errors_df[cols].to_csv(gul_error_map_fp, index=False)


@oasis_log
def write_exposure_summary(
        target_dir,
        exposure_df,
        keys_fp,
        keys_errors_fp,
        exposure_profile
):
    """
    Create exposure summary as dictionary of TIVs and number of locations
    grouped by peril and validity respectively. Writes dictionary as json file
    to disk.

    :param target_dir: directory on disk to write exposure summary file
    :type target_dir: str

    :param exposure_df: source exposure dataframe
    :type exposure df: pandas.DataFrame

    :param keys_fp: file path to keys file
    :type keys_fp: str

    :param keys_errors_fp: file path to keys errors file
    :type keys_errors_fp: str

    :param exposure_profile: profile defining exposure file
    :type exposure_profile: dict

    :return: Exposure summary file path
    :rtype: str
    """
    keys_success_df = keys_errors_df = None

    # get keys success
    if keys_fp:
        keys_success_df = pd.read_csv(keys_fp)[['LocID', 'PerilID', 'CoverageTypeID']]
        keys_success_df['status'] = OASIS_KEYS_STATUS['success']['id']
        keys_success_df.columns = ['loc_id', 'peril_id', 'coverage_type_id', 'status']

    # get keys errors
    if keys_errors_fp:
        keys_errors_df = pd.read_csv(keys_errors_fp)[['LocID', 'PerilID', 'CoverageTypeID', 'Status', 'Message']]
        keys_errors_df.columns = ['loc_id', 'peril_id', 'coverage_type_id', 'status', 'message']
        if not keys_errors_df.empty:
            write_gul_errors_map(target_dir, exposure_df, keys_errors_df)

    # concatinate keys responses & run
    df_keys = pd.concat([keys_success_df, keys_errors_df])
    exposure_summary = get_exposure_summary(exposure_df, df_keys, exposure_profile)

    # write exposure summary as json fileV
    fp = os.path.join(target_dir, 'exposure_summary_report.json')
    with io.open(fp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(exposure_summary, ensure_ascii=False, indent=4))

    return fp
