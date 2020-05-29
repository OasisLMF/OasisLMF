__all__ = [
    'get_summary_mapping',
    'generate_summaryxref_files',
    'merge_oed_to_mapping',
    'write_exposure_summary',
    'write_summary_levels',
    'write_mapping_file',
]

import io
import json
import os
import warnings

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
    reduce_df
)
from ..utils.defaults import (
    find_exposure_fp,
    SOURCE_IDX,
    SUMMARY_MAPPING,
    SUMMARY_OUTPUT,
    get_loc_dtypes,
    get_acc_dtypes,
)

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import as_path
from ..utils.peril import PERILS, PERIL_GROUPS
from ..utils.status import OASIS_KEYS_STATUS
from .gul_inputs import get_gul_input_items


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
    acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
    loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()
    policy_num = oed_hierarchy['polnum']['ProfileElementName'].lower()
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()

    # Case GUL+FM (based on il_inputs_df)
    if is_fm_summary:
        summary_mapping = inputs_df[inputs_df['level_id'] == inputs_df['level_id'].max()].drop_duplicates(subset=['gul_input_id', 'layer_id'], keep='first')
        summary_mapping['agg_id'] = summary_mapping['gul_input_id']
        summary_mapping['output_id'] = factorize_ndarray(
            summary_mapping.loc[:, ['gul_input_id', 'layer_id']].values,
            col_idxs=range(2)
        )[0]
        summary_mapping.drop('item_id', axis=1, inplace=True)
    # GUL Only
    else:
        summary_mapping = inputs_df.copy(deep=True)

    usecols = [
        acc_num,
        loc_num,
        'loc_id',
        policy_num,
        portfolio_num,
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

    summary_mapping.drop(
        [c for c in summary_mapping.columns if c not in usecols],
        axis=1,
        inplace=True
    )
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
    exposure_col_df = exposure_df.loc[:, columns_found]
    # Add default value if optional column is missing
    for col in columns_missing:
        if col in defaults:
            exposure_col_df[col] = defaults[col]
        else:
            raise OasisException('Column to merge "{}" not in locations dataframe or defined with a default value'.format(col))

    exposure_col_df[SOURCE_IDX['loc']] = exposure_df.index
    new_summary_map_df = merge_dataframes(summary_map_df, exposure_col_df, join_on=SOURCE_IDX['loc'], how='inner')
    if defaults:
        new_summary_map_df.fillna(value=defaults, inplace=True)
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

    # Extract mapped_cols from summary_map_df
    summary_group_df = summary_map_df.loc[:, mapped_cols]

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

    summary_group_df.fillna(0, inplace=True)
    summary_group_df.sort_values(by=[sort_by], inplace=True)
    summary_ids = factorize_dataframe(summary_group_df, by_col_labels=oed_cols)

    return summary_ids[0], summary_ids[1]


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
    l_col_list = exposure_df.loc[:, exposure_df.any()].columns.to_list()
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
        accounts_df = pd.read_csv(accounts_fp)
        a_col_list = accounts_df.loc[:, accounts_df.any()].columns.to_list()
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


@oasis_log
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


@oasis_log
def get_summary_xref_df(map_df, exposure_df, accounts_df, summaries_info_dict, summaries_type, gul_items=False):
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

    # Extract the summary id index column depending on summary grouping type
    if 'output_id' in map_df:
        id_set_index = 'output_id'
        ids_set_df = map_df.loc[:, [id_set_index]].rename(columns={id_set_index: "output"})
    elif gul_items:
        id_set_index = 'item_id'
        ids_set_df = map_df.loc[:, [id_set_index]]
    else:
        id_set_index = 'coverage_id'
        ids_set_df = map_df.loc[:, [id_set_index]]


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

        else:
            (
                summary_set_df['summary_id'],
                set_values
            ) = group_by_oed(cols_group_by, map_df, exposure_df, id_set_index, accounts_df)

            # Build description file
            summary_desc[desc_key] = pd.DataFrame(data=list(set_values), columns=cols_group_by)
            summary_desc[desc_key].insert(loc=0, column='summary_id', value=range(1, len(set_values) + 1))

        # Appends summary set to '__summaryxref.csv'
        summary_set_df['summaryset_id'] = summary_set['id']
        summaryxref_df = pd.concat([summaryxref_df, summary_set_df], sort=True, ignore_index=True)

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
        analysis_settings['gul_output'] if 'gul_output' in analysis_settings else False,
        analysis_settings['gul_summaries'] if 'gul_summaries' in analysis_settings else False,
    ])
    il_summaries = all([
        analysis_settings['il_output'] if 'il_output' in analysis_settings else False,
        analysis_settings['il_summaries'] if 'il_summaries' in analysis_settings else False,
        il,
    ])
    ri_summaries = all([
        analysis_settings['ri_output'] if 'ri_output' in analysis_settings else False,
        analysis_settings['ri_summaries'] if 'ri_summaries' in analysis_settings else False,
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

    # Load accounts file for IL OED fields
    if (il_summaries or ri_summaries):
        accounts_fp = find_exposure_fp(input_dir, 'acc')
        acc_dtypes, acc_required_cols = get_dtypes_and_required_cols(get_acc_dtypes)
        accounts_df = get_dataframe(
            src_fp=accounts_fp,
            empty_data_error_msg='No source accounts file found.',
            col_dtypes=acc_dtypes,
            required_cols=acc_required_cols)
        accounts_df[SOURCE_IDX['acc']] = accounts_df.index

    if gul_summaries:
        # Load GUL summary map
        gul_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['gul_map_fn'])
        gul_map_df = get_dataframe(
            src_fp=gul_map_fp,
            empty_data_error_msg='No summary map file found.')

        gul_summaryxref_df, gul_summary_desc = get_summary_xref_df(
            gul_map_df,
            exposure_df,
            None,
            analysis_settings['gul_summaries'],
            'gul',
            gul_item_stream
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
        max_layer = max(ri_layers)
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


@oasis_log
def get_exposure_summary(df, exposure_summary, peril_key, peril_id, status):
    """
    Populate dictionary with TIVs and number of locations, grouped by peril and
    validity respectively

    :param df: dataframe from gul_inputs.get_gul_input_items(..)
    :type df: pandas.DataFrame

    :param exposure_summary: dictionary to populate created in write_exposure_summary(..)
    :type exposure_summary: dict

    :param peril: descriptive name of peril
    :type peril: str

    :param peril_key: Descriptive OED peril key, e.g. "river flood", "tropical cyclone"
    :type peril_key: str

    :param status: status returned by lookup ('success', 'fail' or 'nomatch')
    :type status: str

    :return: populated exposure_summary dictionary
    :rtype: dict
    """

    # Separate TIVs by coverage type and acquire sum
    for coverage_type in SUPPORTED_COVERAGE_TYPES:
        tiv_sum = df.loc[
            (df['peril_id'] == peril_id) &
            (df['coverage_type_id'] == SUPPORTED_COVERAGE_TYPES[coverage_type]['id']),
            'tiv'
        ].sum()
        tiv_sum = float(tiv_sum)
        exposure_summary[peril_key][status]['tiv_by_coverage'][coverage_type] = tiv_sum
        if coverage_type in exposure_summary[peril_key]['all']['tiv_by_coverage']:
            exposure_summary[peril_key]['all']['tiv_by_coverage'][coverage_type] += tiv_sum
        else:
            exposure_summary[peril_key]['all']['tiv_by_coverage'][coverage_type] = tiv_sum
        exposure_summary[peril_key][status]['tiv'] += tiv_sum
        exposure_summary[peril_key]['all']['tiv'] += tiv_sum

    # Find number of locations
    loc_count = df.loc[df['peril_id'] == peril_id, 'loc_id'].drop_duplicates().count()
    loc_count = int(loc_count)
    exposure_summary[peril_key][status]['number_of_locations'] = loc_count
    exposure_summary[peril_key]['all']['number_of_locations'] += loc_count

    return exposure_summary


@oasis_log
def get_exposure_totals(df, df_errors):
    """
    Return dictionary with total TIVs and number of locations

    :param df: dataframe from `gul_inputs_df`
    :type df: pandas.DataFrame

    :param df_errors: dataframe from `gul_input_errors_df`
    :type df_errors: pandas.DataFrame

    :param exposure_summary: dictionary to populate created in write_exposure_summary(..)
    :type exposure_summary: dict

    :return: totals section for exposure_summary dictionary
    :rtype: dict
    """
    within_scope     = df.drop_duplicates(subset=['loc_id', 'coverage_type_id'])['tiv']
    within_scope_tiv = within_scope.sum()
    within_scope_num = int(within_scope.count())

    outside_scope     = df_errors.drop_duplicates(subset=['loc_id', 'coverage_type_id'])['tiv']
    outside_scope_tiv = outside_scope.sum()
    outside_scope_num = int(outside_scope.count())

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
            "tiv": within_scope_tiv + outside_scope_tiv,
            "number_of_locations": within_scope_num + outside_scope_num
        }
    }


@oasis_log
def write_exposure_summary(
    target_dir,
    gul_inputs_df,
    exposure_df,
    keys_errors_fp,
    exposure_profile,
    oed_hierarchy
):
    """
    Create exposure summary as dictionary of TIVs and number of locations
    grouped by peril and validity respectively. Writes dictionary as json file
    to disk.

    :param target_dir: directory on disk to write exposure summary file
    :type target_dir: str

    :param gul_inputs_df: dataframe from gul_inputs.get_gul_input_items(..)
    :type gul_inputs_df: pandas.DataFrame

    :param exposure_df: source exposure dataframe
    :type exposure df: pandas.DataFrame

    :param keys_errors_fp: file path to keys erors file
    :type keys_errors_fp: str

    :param exposure_profile: profile defining exposure file
    :type exposure_profile: dict

    :param oed_hierarchy: exposure dataframe column names
    :type oed_hierarchy: dict

    :return: Exposure summary file path
    :rtype: str
    """
    loc_per_cov = oed_hierarchy['locperilid']['ProfileElementName'].lower()
    gul_inputs_df = reduce_df(
        gul_inputs_df,
        cols=['peril_id', 'coverage_type_id', 'loc_id', 'tiv']
    )

    # Get GUL input items dataframe to process keys errors
    try:
        gul_inputs_errors_df = get_gul_input_items(
            exposure_df, keys_errors_fp, exposure_profile=exposure_profile
        )

        # Store the gul_input_errors for debugging then reduce
        store_cols = ['loc_id', 'portnumber', 'accnumber', 'locnumber', 'condnumber']
        reduce_cols = ['peril_id', 'coverage_type_id', 'loc_id', 'tiv', 'status']

        gul_inputs_errors_df[store_cols + reduce_cols].to_csv(
            os.path.join(target_dir, 'gul_errors_map.csv'),
            index=False
        )
        gul_inputs_errors_df = reduce_df(
            gul_inputs_errors_df,
            cols=reduce_cols
        )
    except OasisException:   # Empty dataframe (due to empty keys errors file)
        gul_inputs_errors_df = pd.DataFrame(
            columns=gul_inputs_df.columns.append(pd.Index(['status']))
        )

    # Dictionary to map perils with peril groups
    peril_groups = {v['id']: v['peril_ids'] for k, v in PERIL_GROUPS.items()}

    # Remove group peril codes in exposure if all constituent peril codes are present
    exposure_df = reduce_df(exposure_df, cols=[loc_per_cov, 'loc_id'])
    exposure_df = remove_duplicate_perils(exposure_df, loc_per_cov, peril_groups)

    peril_ids = {v['id']: [v['id']] for k, v in PERILS.items()}
    peril_groups.update(peril_ids)

    # Merge GUL input items and source exposure dataframes to leave covered perils
    model_peril_ids = gul_inputs_df['peril_id'].unique()

    # Split rows with multiple peril codes
    exposure_df[loc_per_cov] = exposure_df[loc_per_cov].str.replace(' ','')
    exposure_df[loc_per_cov] = exposure_df[loc_per_cov].str.replace(';$','', regex=True)

    exp_perils_df = pd.DataFrame(
        exposure_df[loc_per_cov].str.split(';').to_list(),
        index=exposure_df['loc_id']
    ).stack()

    # Display warnings for invalid peril codes
    for peril_id in exp_perils_df[~exp_perils_df.isin(peril_groups.keys())]:
        warnings.warn('"{}" is not a valid OED peril ID/code. Please check the source exposure file.'.format(peril_id))

    # Split rows with peril codes corresponding to peril groups
    exp_perils_df = pd.DataFrame(
        exp_perils_df.map(peril_groups).to_list(),
        index=exp_perils_df.index
    ).stack()

    exp_perils_df = exp_perils_df.reset_index([0, 'loc_id'])

    exp_perils_df.columns = ['loc_id', 'peril_id']
    exp_perils_df = exp_perils_df.astype(
        {'loc_id': 'int64', 'peril_id': 'object'}
    )

    exposure_df = merge_dataframes(
        exposure_df,
        exp_perils_df,
        on='loc_id',
        how='right'
    )
    gul_inputs_df = merge_dataframes(
        gul_inputs_df,
        exposure_df,
        on=['loc_id', 'peril_id'],
        how='inner'
    )
    gul_inputs_errors_df = merge_dataframes(
        gul_inputs_errors_df,
        exposure_df,
        on=['loc_id', 'peril_id'],
        how='inner'
    )

    # Convert loc_id to uint32 in case column has been converted to type float
    gul_inputs_df['loc_id'] = gul_inputs_df['loc_id'].astype('uint32')
    gul_inputs_errors_df['loc_id'] = gul_inputs_errors_df['loc_id'].astype('uint32')

    # Compile summary of exposure data
    exposure_summary = {}

    # Create totals section
    exposure_summary['total'] = get_exposure_totals(
        gul_inputs_df,
        gul_inputs_errors_df
    )

    for peril_id in model_peril_ids:
        # Use descriptive names of perils as keys
        try:
            peril_key = [k for k, v in PERILS.items() if v['id'] == peril_id][0]
        except IndexError:
            warnings.warn('"{}" is not a valid OED peril ID/code. Please check the source exposure file.'.format(peril_id))
            return None
        exposure_summary[peril_key] = {}
        # Create dictionary structure for all and each validity status
        for status in ['all'] + list(OASIS_KEYS_STATUS.keys()):
            exposure_summary[peril_key][status] = {}
            exposure_summary[peril_key][status]['tiv'] = 0.0
            exposure_summary[peril_key][status]['tiv_by_coverage'] = {}
            exposure_summary[peril_key][status]['number_of_locations'] = 0
            # Fill exposure summary dictionary
            if status == 'success':
                exposure_summary = get_exposure_summary(
                    gul_inputs_df,
                    exposure_summary,
                    peril_key,
                    peril_id,
                    status
                )
            elif status != 'all':
                exposure_summary = get_exposure_summary(
                    gul_inputs_errors_df[gul_inputs_errors_df['status'] == status],
                    exposure_summary,
                    peril_key,
                    peril_id,
                    status
                )

    # Write exposure summary as json file
    fp = os.path.join(target_dir, 'exposure_summary_report.json')
    with io.open(fp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(exposure_summary, ensure_ascii=False, indent=4))

    return fp


def remove_duplicate_perils(exposure_df, loc_per_cov, peril_groups):
    """
    Remove grouped peril codes in the case that all component peril codes are also provided.

    :param exposure_df: source exposure dataframe
    :type exposure df: pandas.DataFrame

    :param loc_per_cov: dataframe column name covering the peril codes for an exposure
    :type loc_per_cov: str

    :param peril_groups: peril group codes and associated perils
    :type peril_groups: dict

    :return: dataframe with peril group code removed if all component perils are provided
    :rtype: pandas.DataFrame
    """
    unique_loc_per_cov = exposure_df[loc_per_cov].unique()
    to_replace = {}
    for exposure_perils_set in unique_loc_per_cov:
        exposure_perils = exposure_perils_set.split(';')
        for peril_group in peril_groups:
            if peril_group in exposure_perils and all(
                peril in exposure_perils for peril in peril_groups[peril_group]
            ):
                exposure_perils.remove(peril_group)
                to_replace[exposure_perils_set] = ';'.join(exposure_perils)
    exposure_df = exposure_df.replace(to_replace=to_replace)
    return exposure_df
