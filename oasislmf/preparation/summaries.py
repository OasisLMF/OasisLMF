__all__ = [
    'get_summary_mapping',
    'generate_summaryxref_files',
    'merge_oed_to_mapping',
    'write_exposure_summary',
    'get_exposure_summary',
    'get_xref_df',
    'write_summary_levels',
    'write_mapping_file',
]

import io
import json
import os
import warnings

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
    get_dtypes_and_required_cols,
    reduce_df,
    fill_na_with_categoricals
)
from ..utils.defaults import (
    find_exposure_fp,
    SOURCE_IDX,
    SUMMARY_MAPPING,
    SUMMARY_OUTPUT,
    get_loc_dtypes,
    get_acc_dtypes,
    get_default_exposure_profile,
)
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import as_path
from ..utils.peril import PERILS, PERIL_GROUPS
from ..utils.status import OASIS_KEYS_STATUS, OASIS_KEYS_STATUS_MODELLED


def get_usefull_summary_cols(oed_hierarchy):
    return [
        oed_hierarchy['accnum']['ProfileElementName'].lower(),
        oed_hierarchy['locnum']['ProfileElementName'].lower(),
        'loc_id',
        oed_hierarchy['polnum']['ProfileElementName'].lower(),
        oed_hierarchy['portnum']['ProfileElementName'].lower(),
        SOURCE_IDX['loc'],
        SOURCE_IDX['acc'],
        'item_id',
        'layer_id',
        'coverage_id',
        'peril_id',
        'agg_id',
        'output_id',
        'coverage_type_id',
        'tiv'
    ]


def get_xref_df(il_inputs_df):
    top_level_cols = ['layer_id', SOURCE_IDX['acc'], 'polnumber']
    top_level_layers_df = il_inputs_df.loc[il_inputs_df['level_id'] == il_inputs_df['level_id'].max(), ['top_agg_id'] + top_level_cols]
    bottom_level_layers_df = il_inputs_df[il_inputs_df['level_id'] == 1]
    bottom_level_layers_df.drop(columns = top_level_cols, inplace=True)
    return merge_dataframes(bottom_level_layers_df, top_level_layers_df, join_on=['top_agg_id'])


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
        summary_mapping = get_xref_df(inputs_df).drop_duplicates(subset=['gul_input_id', 'layer_id'], keep='first')
        summary_mapping['agg_id'] = summary_mapping['gul_input_id']
        summary_mapping = summary_mapping.reindex(sorted(summary_mapping.columns), axis=1)
        summary_mapping['output_id'] = factorize_ndarray(
            summary_mapping.loc[:, ['gul_input_id', 'layer_id']].values,
            col_idxs=range(2)
        )[0]

    # GUL Only
    else:
        summary_mapping = inputs_df.copy(deep=True)
        summary_mapping['layer_id']=1
        summary_mapping['agg_id'] = summary_mapping['item_id']

    summary_mapping.drop(
        [c for c in summary_mapping.columns if c not in get_usefull_summary_cols(oed_hierarchy)],
        axis=1,
        inplace=True
    )
    acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
    loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()
    policy_num = oed_hierarchy['polnum']['ProfileElementName'].lower()
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()

    dtypes = {
        **{t: 'str' for t in [portfolio_num, policy_num, acc_num, loc_num, 'peril_id']},
        **{t: 'uint8' for t in ['coverage_type_id']},
        **{t: 'uint32' for t in [SOURCE_IDX['loc'], SOURCE_IDX['acc'], 'loc_id', 'item_id', 'layer_id', 'coverage_id', 'agg_id', 'output_id']},
        **{t: 'float64' for t in ['tiv']}
    }
    summary_mapping = set_dataframe_column_dtypes(summary_mapping, dtypes)

    return summary_mapping


def merge_oed_to_mapping(summary_map_df, exposure_df, oed_column_set, defaults=None):
    """
    Create a factorized col (summary ids) based on a list of oed column names

    :param :summary_map_df dataframe return from get_summary_mapping
    :type summary_map_df: pandas.DataFrame

    :param exposure_df: Summary map file path
    :type exposure_df: pandas.DataFrame

    :param defaults: Dictionary of vaules to fill NaN columns with
    :type defaults: dict

    {'Col_A': 0, 'Col_B': 1, 'Col_C': 2}

    :return: New DataFrame of summary_map_df + exposure_df merged on exposure index
    :rtype: pandas.DataFrame
    """

    column_set = [c.lower() for c in oed_column_set]
    columns_found = [c for c in column_set if c in exposure_df.columns.to_list()]
    columns_missing = list(set(column_set) - set(columns_found))

    # Select DF with matching cols
    exposure_col_df = exposure_df.loc[:, columns_found + [SOURCE_IDX['loc']]]
    # Add default value if optional column is missing
    for col in columns_missing:
        if col in defaults:
            exposure_col_df[col] = defaults[col]
        else:
            raise OasisException('Column to merge "{}" not in locations dataframe or defined with a default value'.format(col))

    new_summary_map_df = merge_dataframes(summary_map_df, exposure_col_df, join_on=SOURCE_IDX['loc'], how='inner')
    if defaults:
        fill_na_with_categoricals(new_summary_map_df, defaults)
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
    oed_cols = [c.lower() for c in oed_col_group]                                               # All requred columns
    unmapped_cols = [c for c in oed_cols if c not in summary_map_df.columns]                    # columns which in locations / Accounts file
    mapped_cols = [c for c in oed_cols + [SOURCE_IDX['loc'], SOURCE_IDX['acc'], sort_by] if c in summary_map_df.columns]    # Columns already in summary_map_df
    tiv_cols = ['tiv', 'loc_id', 'coverage_type_id']

    # Extract mapped_cols from summary_map_df
    summary_group_df = summary_map_df.loc[:, set(mapped_cols).union(tiv_cols)]

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
    summary_tiv = summary_group_df.drop_duplicates(['loc_id', 'coverage_type_id'] + oed_col_group, keep='first').groupby(oed_col_group).agg({'tiv': np.sum})

    return summary_ids[0], summary_ids[1], summary_tiv


@oasis_log
def write_summary_levels(exposure_df, accounts_fp, target_dir):
    '''
    Json file with list Available / Recommended columns for use in the summary reporting

    Available: Columns which exists in input files and has at least one non-zero / NaN value
    Recommended: Columns which are available + also in the list of `useful` groupings SUMMARY_LEVEL_LOC

    {
        'GUL': {
            'available': ['accnumber',
                         'locnumber',
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
                         'buildingtiv',
                         'contentstiv',
                         'bitiv',
                         'portnumber'],

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
    #l_col_list = exposure_df.loc[:, exposure_df.any()].columns.to_list()
    # NOTE: work around for pandas==1.2.0, any() not returning return the 'category' field types
    l_col_list = exposure_df.replace(0, np.nan).dropna(how='any', axis=1).columns.to_list()

    l_col_info = get_loc_dtypes()
    for k in list(l_col_info.keys()):
        l_col_info[k.lower()] = l_col_info[k]
        del l_col_info[k]

    gul_avail = {k: l_col_info[k]['desc'] if k in l_col_info else desc_non_oed
                 for k in set([c.lower() for c in l_col_list]).difference(int_excluded_cols)}
    gul_summary_lvl = {'GUL': {'available': {**gul_avail, **int_oasis_cols}}}

    # IL perspective (join of acc + loc col with no dups)
    il_summary_lvl = {}
    if accounts_fp:
        accounts_df = get_dataframe(accounts_fp, lowercase_cols=False)
        a_col_list = accounts_df.loc[:, ~accounts_df.isnull().all()].columns.to_list()
        a_col_info = get_acc_dtypes()
        a_avail = set([c.lower() for c in a_col_list])

        il_avail = {k: a_col_info[k]['desc'] if k in a_col_info else desc_non_oed
                    for k in a_avail.difference(gul_avail.keys())}
        il_summary_lvl = {'IL': {'available': {**gul_avail, **il_avail, **int_oasis_cols}}}

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
    chunksize = min(2 * 10**5, max(len(sum_inputs_df), 1000))

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
        return [c.lower() for c in summary_set['oed_fields']]
    elif isinstance(summary_set['oed_fields'], str) and len(summary_set['oed_fields']) > 0:
        return [summary_set['oed_fields'].lower()]
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


def write_df_to_file(df, target_dir, filename):
    """
    Write a generated summary xref dataframe to disk

    :param df: The dataframe output of get_df( .. )
    :type df:  pandas.DataFrame

    :param target_dir: Abs directory to write a summary_xref file to
    :type target_dir:  str

    :param filename: Name of file to store as
    :type filename:  str
    """
    target_dir = as_path(target_dir, 'Input files directory', is_dir=True, preexists=False)
    chunksize = min(2 * 10**5, max(len(df), 1000))
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
        raise OasisException("Exception raised in 'write_df_to_file'", e)

    return csv_fp


def get_summary_xref_df(map_df, exposure_df, accounts_df, summaries_info_dict, summaries_type, id_set_index='output_id'):
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
        desc_key = '{}_S{}_summary-info.csv'.format(summaries_type, summary_set['id'])

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
            summary_desc[desc_key].insert(loc=len(summary_desc[desc_key].columns), column='tiv', value=map_df.drop_duplicates(['loc_id', 'coverage_type_id'], keep='first').tiv.sum())
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
def generate_summaryxref_files(model_run_fp, analysis_settings, il=False, ri=False, gul_item_stream=False):
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
    :type il: bool

    :param gul_items: Boolean to gul to use item_id instead of coverage_id
    :type gul_items: bool
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

    # Load locations file for GUL OED fields
    input_dir = os.path.join(model_run_fp, 'input')
    exposure_fp = find_exposure_fp(input_dir, 'loc')
    loc_dtypes, loc_required_cols = get_dtypes_and_required_cols(get_loc_dtypes)
    exposure_df = get_dataframe(
        src_fp=exposure_fp,
        empty_data_error_msg='No source exposure file found.',
        col_dtypes=loc_dtypes,
        required_cols=loc_required_cols)
    exposure_df[SOURCE_IDX['loc']] = exposure_df.index

    # Load accounts file and the more complete il_map if present
    try:
        accounts_fp = find_exposure_fp(input_dir, 'acc')
        acc_dtypes, acc_required_cols = get_dtypes_and_required_cols(get_acc_dtypes)
        accounts_df = get_dataframe(
            src_fp=accounts_fp,
            empty_data_error_msg='No source accounts file found.',
            col_dtypes=acc_dtypes,
            required_cols=acc_required_cols)
        accounts_df[SOURCE_IDX['acc']] = accounts_df.index

        il_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['fm_map_fn'])
        il_map_df = get_dataframe(
            src_fp=il_map_fp,
            empty_data_error_msg='No summary map file found.'
        )
        if gul_summaries:
            gul_map_df = il_map_df
            gul_map_df['item_id'] = gul_map_df['agg_id']
    except Exception:
        if not (il_summaries or ri_summaries): # accounts is not compulsory
            accounts_df = None
            if gul_summaries:
                gul_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['gul_map_fn'])
                gul_map_df = get_dataframe(
                    src_fp=gul_map_fp,
                    empty_data_error_msg='No summary map file found.')
        else:
            raise

    if gul_summaries:
        # Load GUL summary map
        id_set_index = 'item_id' if gul_item_stream else 'coverage_id'
        gul_summaryxref_df, gul_summary_desc = get_summary_xref_df(
            gul_map_df,
            exposure_df,
            accounts_df,
            analysis_settings['gul_summaries'],
            'gul',
            id_set_index
        )
        # Write Xref file
        write_df_to_file(gul_summaryxref_df, os.path.join(model_run_fp, 'input'), SUMMARY_OUTPUT['gul'])

        # Write summary_id description files
        for desc_key in gul_summary_desc:
            write_df_to_file(gul_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)

    if il_summaries:
        # Load FM summary map
        il_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['fm_map_fn'])
        il_map_df = get_dataframe(
            src_fp=il_map_fp,
            empty_data_error_msg='No summary map file found.'
        )

        il_summaryxref_df, il_summary_desc = get_summary_xref_df(
            il_map_df,
            exposure_df,
            accounts_df,
            analysis_settings['il_summaries'],
            'il'
        )
        # Write Xref file
        write_df_to_file(il_summaryxref_df, os.path.join(model_run_fp, 'input'), SUMMARY_OUTPUT['il'])

        # Write summary_id description files
        for desc_key in il_summary_desc:
            write_df_to_file(il_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)

    if ri_summaries:
        ri_layers = get_ri_settings(model_run_fp)
        max_layer = str(max([int(x) for x in ri_layers]))
        summary_ri_fp = os.path.join(
            model_run_fp, os.path.basename(ri_layers[max_layer]['directory']))

        ri_summaryxref_df = pd.DataFrame()
        if ('il_summaries' not in analysis_settings) or (not il_summaries):
            il_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['fm_map_fn'])
            il_map_df = get_dataframe(
                src_fp=il_map_fp,
                empty_data_error_msg='No summary map file found.'
            )

        ri_summaryxref_df, ri_summary_desc = get_summary_xref_df(
            il_map_df,
            exposure_df,
            accounts_df,
            analysis_settings['ri_summaries'],
            'ri'
        )
        # Write Xref file
        write_df_to_file(ri_summaryxref_df, summary_ri_fp, SUMMARY_OUTPUT['il'])

        # Write summary_id description files
        for desc_key in ri_summary_desc:
            write_df_to_file(ri_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)


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
            (df['peril_id'] == peril_id) &
            (df['coverage_type_id'] == SUPPORTED_COVERAGE_TYPES[coverage_type]['id']),
            'tiv'
        ].sum()
        tiv_sum = float(tiv_sum)
        exposure_summary[peril_id][status]['tiv_by_coverage'][coverage_type] = tiv_sum
        exposure_summary[peril_id][status]['tiv'] += tiv_sum

        loc_count = df.loc[
            (df['peril_id'] == peril_id) &
            (df['coverage_type_id'] == SUPPORTED_COVERAGE_TYPES[coverage_type]['id']),
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

    #get location tivs by coveragetype
    df_summary = pd.DataFrame(columns=['loc_id','coverage_type_id','tiv'])

    for field in exposure_profile:
        if 'FMTermType' in exposure_profile[field].keys():
            if exposure_profile[field]['FMTermType'] == 'TIV':
                cov_name = str.lower(exposure_profile[field]['ProfileElementName'])
                coverage_type_id = exposure_profile[field]['CoverageTypeID']
                tmp_df = exposure_df[['loc_id',cov_name]]
                tmp_df.columns=['loc_id','tiv']
                tmp_df['coverage_type_id']=coverage_type_id
                df_summary = pd.concat([df_summary,tmp_df])

    #get all perils
    peril_list = keys_df['peril_id'].drop_duplicates().to_list()

    df_summary_peril = pd.DataFrame(columns=['loc_id','coverage_type_id','tiv','peril_id'])

    for peril_id in peril_list:
        tmp_df = df_summary
        tmp_df['peril_id']=peril_id
        df_summary_peril = pd.concat([df_summary_peril,tmp_df])

    df_summary_peril = df_summary_peril.merge(keys_df,how='left',on=['loc_id','coverage_type_id','peril_id'])
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

    cols = ['loc_id','portnumber','accnumber','locnumber','peril_id','coverage_type_id','tiv','status','message']
    gul_error_map_fp = os.path.join(target_dir, 'gul_errors_map.csv')

    exposure_id_cols = ['loc_id','portnumber','accnumber','locnumber']
    keys_error_cols = ['loc_id','peril_id','coverage_type_id','status','message']
    tiv_maps = {1:'buildingtiv',2:'othertiv',3:'contentstiv',4:'bitiv'}
    exposure_cols = exposure_id_cols + list(tiv_maps.values())

    keys_errors_df.columns = keys_error_cols

    gul_inputs_errors_df = exposure_df[exposure_cols].merge(keys_errors_df[keys_error_cols],on=['loc_id'])
    gul_inputs_errors_df['tiv']=0.0
    for cov_type in tiv_maps:
        tiv_field = tiv_maps[cov_type]
        gul_inputs_errors_df['tiv']=np.where(
            gul_inputs_errors_df['coverage_type_id']==cov_type,
            gul_inputs_errors_df[tiv_field],
            gul_inputs_errors_df['tiv']
        )
    gul_inputs_errors_df['tiv'] = gul_inputs_errors_df['tiv'].fillna(0.0)

    gul_inputs_errors_df[cols].to_csv(gul_error_map_fp,index=False)

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

    #get keys success
    if keys_fp:
        keys_success_df = pd.read_csv(keys_fp)[['LocID', 'PerilID', 'CoverageTypeID']]
        keys_success_df['status'] = OASIS_KEYS_STATUS['success']['id']
        keys_success_df.columns = ['loc_id','peril_id','coverage_type_id','status']

    #get keys errors
    if keys_errors_fp:
        keys_errors_df = pd.read_csv(keys_errors_fp)[['LocID', 'PerilID', 'CoverageTypeID', 'Status', 'Message']]
        keys_errors_df.columns = ['loc_id','peril_id','coverage_type_id','status','message']
        if not keys_errors_df.empty:
            write_gul_errors_map(target_dir,exposure_df,keys_errors_df)

    #concatinate keys responses & run
    df_keys = pd.concat([keys_success_df,keys_errors_df])
    exposure_summary = get_exposure_summary(exposure_df, df_keys, exposure_profile)

    #write exposure summary as json fileV
    fp = os.path.join(target_dir, 'exposure_summary_report.json')
    with io.open(fp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(exposure_summary, ensure_ascii=False, indent=4))

    return fp
