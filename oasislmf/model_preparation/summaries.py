__all__ = [
    'get_summary_mapping',
    'generate_summaryxref_files',
    'merge_oed_to_mapping',
    'write_mapping_file',
]

import os

import pandas as pd

from ..utils.data import (
    factorize_dataframe,
    factorize_ndarray,
    get_dataframe,
    get_json,
    merge_dataframes,
)
from ..utils.defaults import (
    SOURCE_FILENAMES,
    SOURCE_IDX,
    SUMMARY_MAPPING,
    SUMMARY_GROUPING,
    SUMMARY_OUTPUT,
)
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import as_path
from ..utils.profiles import get_oed_hierarchy_terms


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
    usecols = ([
        oed_hierarchy['accid'],
        oed_hierarchy['locid'],
        oed_hierarchy['polid'],
        oed_hierarchy['portid'],
        SOURCE_IDX['loc'],
        'item_id',
        'layer_id',
        'coverage_id',
        'peril_id',
        'agg_id',
        'output_id',
        'coverage_type_id',
        'tiv',
    ])

    # Case GUL+FM (based on il_inputs_df)
    if is_fm_summary:
        #import ipdb; ipdb.set_trace()
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

    summary_mapping.drop(
        [c for c in summary_mapping.columns if c not in usecols],
        axis=1,
        inplace=True
    )

    return summary_mapping


def merge_oed_to_mapping(summary_map_df, exposure_df, oed_column_set, defaults=None):
    """
    Adds list of OED fields from `column_set` to summary map file

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
    exposure_col_df = exposure_df.loc[:, column_set]
    exposure_col_df[SOURCE_IDX['loc']] = exposure_df.index
    new_summary_map_df = merge_dataframes(summary_map_df, exposure_col_df, join_on=SOURCE_IDX['loc'], how='inner')
    if defaults:
        new_summary_map_df.fillna(value=defaults, inplace=True)
    return new_summary_map_df


def group_by_oed(summary_map_df, exposure_df, oed_col_group):
    """
    Adds list of OED fields from `column_set` to summary map file

    :param :summary_map_df dataframe return from get_summary_mapping
    :type summary_map_df: pandas.DataFrame

    :param exposure_df: Summary map file path
    :type exposure_df: pandas.DataFrame

    :return: subset of columns from exposure_df to merge
    :rtype: list
    """

    exposure_cols = [c.lower() for c in oed_col_group if c not in summary_map_df.columns]
    mapping_cols = [SOURCE_IDX['loc']] + [c.lower() for c in oed_col_group if c in summary_map_df.columns]

    summary_group_df = summary_map_df.loc[:, mapping_cols]
    if exposure_cols is not []:
        exposure_col_df = exposure_df.loc[:, exposure_cols]
        exposure_col_df[SOURCE_IDX['loc']] = exposure_df.index
        summary_group_df = merge_dataframes(summary_group_df, exposure_df, join_on=SOURCE_IDX['loc'], how='inner')

    summary_group_df.fillna(0, inplace=True)  # factorize with all values NaN, leads to summary_id == 0
    summary_ids = factorize_dataframe(summary_group_df, by_col_labels=oed_col_group)[0]

    return summary_ids


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

    # Set chunk size for writing the CSV files - default is 100K
    chunksize = min(2 * 10**5, len(sum_inputs_df))

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
        raise OasisException from e

    return sum_mapping_fp


def get_column_selection(summary_set):
    """
    Given a analysis_settings summary definition, return either
        1. the set of OED columns requested to group by
        2. A set columns for one of the following default groupings
            ['lob', 'county', 'policy', 'state', 'location', 'prog']
        3. If no information key 'oed_fields', then group all outputs into a single summary_set

    :param summary_set: summary group dictionary from the `analysis_settings.json`
    :type summary_set: dict

    :return: List of selected OED columns to create summary groups from
    :rtype: list
    """
    if "oed_fields" not in summary_set:
        return None

    # Select Default Grouping for either
    elif isinstance(summary_set['oed_fields'], str):
        if summary_set['oed_fields'] in SUMMARY_GROUPING:
            return SUMMARY_GROUPING[summary_set['oed_fields']]
        else:
            raise OasisException('oed_fields value Invalid: {}'.format(summary_set['oed_fields']))

    # Use OED column list set in analysis_settings file
    elif isinstance(summary_set['oed_fields'], list) and len(summary_set['oed_fields']) > 0:
        return summary_set['oed_fields']
    else:
        raise OasisException('Unable to process settings file')


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
def write_xref_file(summary_xref_df, target_dir):
    """
    Write a generated summary xref dataframe to disk

    :param summary_xref_df: The dataframe output of get_summary_xref_df( .. )
    :type summary_xref_df:  pandas.DataFrame

    :param target_dir: Abs directory to write a summary_xref file to
    :type target_dir:  str

    """
    target_dir = as_path(
        target_dir,
        'Target IL input files directory',
        is_dir=True,
        preexists=False
    )

    # Set chunk size for writing the CSV files - default is 100K
    chunksize = min(2 * 10**5, len(summary_xref_df))

    if 'output' in summary_xref_df.columns:
        summary_xref_fp = os.path.join(target_dir, SUMMARY_OUTPUT['il'])
    else:
        summary_xref_fp = os.path.join(target_dir, SUMMARY_OUTPUT['gul'])
    try:
        summary_xref_df.to_csv(
            path_or_buf=summary_xref_fp,
            encoding='utf-8',
            mode=('w'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException from e

    return summary_xref_fp


@oasis_log
def get_summary_xref_df(map_df, exposure_df, summaries_info_dict):
    """
    Create a Dataframe for either gul / il / ri  based on a section
    from the analysis settings


    :param map_df: Summary Map dataframe (GUL / IL)
    :type map_df:  pandas.DataFrame

    :param exposure_df: Location exposure data
    :type exposure_df:  pandas.DataFrame

    :param summaries_info_dict: list of dictionary definitionfor a summary group from the analysis_settings file
    :type summaries_info_dict:  list

    [{
        "summarycalc": true,
        "eltcalc": true,
        "aalcalc": true,
        "pltcalc": true,
        "id": 1,
        "oed_fields": "prog",
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


    :return: Dataframe holding summaryxref information
    :rtype: pandas.DataFrame
    """
    summaryxref_df = pd.DataFrame()

    # Infer il / gul xref type based on 'map_df'
    if 'output_id' in map_df:
        ids_set_df = map_df.loc[:, ['output_id']].rename(columns={"output_id": "output"})
    else:
        ids_set_df = map_df.loc[:, ['coverage_id']]

    # For each granularity build a set grouping
    for summary_set in summaries_info_dict:
        summary_set_df = ids_set_df
        cols_group_by = get_column_selection(summary_set)

        if isinstance(cols_group_by, list):
            summary_set_df['summary_id'] = group_by_oed(map_df, exposure_df, cols_group_by)
        else:
            # Fall back to setting all in single group
            summary_set_df['summary_id'] = 1

        # Appends summary set to '__summaryxref.csv'
        summary_set_df['summaryset_id'] = summary_set['id']
        summaryxref_df = pd.concat([summaryxref_df, summary_set_df], sort=True, ignore_index=True)

    return summaryxref_df


@oasis_log
def generate_summaryxref_files(model_run_fp, analysis_settings):
    """
    Top level function for creating the summaryxref files from the manager.py

    :param model_run_fp: Model run directory file path
    :type model_run_fp:  str

    :param analysis_settings: model run settings file
    :type analysis_settings:  dict
    """

    # Load Exposure file for extra OED fields
    exposure_fp = os.path.join(model_run_fp, 'input', SOURCE_FILENAMES['loc'])
    exposure_df = get_dataframe(
        src_fp=exposure_fp,
        empty_data_error_msg='No source exposure file found.')

    if 'gul_summaries' in analysis_settings:
        # Load GUL summary map
        gul_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['gul_map_fn'])
        gul_map_df = get_dataframe(
            src_fp=gul_map_fp,
            empty_data_error_msg='No summary map file found.')

        gul_summaryxref_df = get_summary_xref_df(
            gul_map_df,
            exposure_df,
            analysis_settings['gul_summaries']
        )
        write_xref_file(gul_summaryxref_df, os.path.join(model_run_fp, 'input'))

    if 'il_summaries' in analysis_settings:
        # Load FM summary map
        il_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['fm_map_fn'])
        il_map_df = get_dataframe(
            src_fp=il_map_fp,
            empty_data_error_msg='No summary map file found.'
        )

        il_summaryxref_df = get_summary_xref_df(
            il_map_df,
            exposure_df,
            analysis_settings['il_summaries']
        )
        write_xref_file(il_summaryxref_df, os.path.join(model_run_fp, 'input'))

    if 'ri_summaries' in analysis_settings:
        ri_layers = get_ri_settings(model_run_fp)
        max_layer = max(ri_layers)
        summary_ri_fp = os.path.join(
            model_run_fp, os.path.basename(ri_layers[max_layer]['directory']))

        ri_summaryxref_df = pd.DataFrame()
        if 'il_summaries' not in analysis_settings:
            il_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['fm_map_fn'])
            il_map_df = get_dataframe(
                src_fp=il_map_fp,
                empty_data_error_msg='No summary map file found.'
            )

        ri_summaryxref_df = get_summary_xref_df(
            il_map_df,
            exposure_df,
            analysis_settings['ri_summaries']
        )
        write_xref_file(ri_summaryxref_df, summary_ri_fp)
