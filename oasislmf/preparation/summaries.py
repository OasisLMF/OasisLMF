__all__ = [
    'get_summary_mapping',
    'generate_summaryxref_files',
    'get_ri_inuring_priority_output_levels',
    'get_ri_summaryxref_dirs',
    'merge_oed_to_mapping',
    'write_exposure_summary',
    'get_exposure_summary',
    'get_useful_summary_cols',
    'get_xref_df',
    'write_summary_levels',
    'write_mapping_file',
    'get_summary_map_fp',
    'calculated_summary_cols'
]

import logging
import pathlib
import shutil

import io
import json
import os

import numpy as np
import pandas as pd

from ..utils.coverages import SUPPORTED_COVERAGE_TYPES
from ..utils.data import (
    factorize_dataframe,
    get_dataframe,
    get_json,
    merge_dataframes,
    set_dataframe_column_dtypes,
    fill_na_with_categoricals
)
from ..utils.defaults import (
    SOURCE_IDX,
    SAR_ID,
    SUMMARY_MAPPING,
    SUMMARY_OUTPUT,
    SUMMARY_TOP_LEVEL_COLS,
    get_default_exposure_profile,
)
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import as_path
from ..utils.status import OASIS_KEYS_STATUS, OASIS_KEYS_STATUS_MODELLED
from oasislmf.pytools.common.data import gul_summary_xref_dtype, fm_summary_xref_dtype
from oasislmf.pytools.converters.csvtobin.utils.common import df_to_ndarray
from oasislmf.utils.fm import SUPPORTED_FM_LEVELS


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

logger = logging.getLogger(__name__)


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
        'risk_id',
        'intensity_adjustment',
        'return_period'
    ]


def is_property_damage(summary_df):
    property_damage_cov_id = [SUPPORTED_COVERAGE_TYPES['buildings']['id'], SUPPORTED_COVERAGE_TYPES['contents']['id']]
    summary_df['is_property_damage'] = summary_df['coverage_type_id'].isin(property_damage_cov_id)
    return summary_df


calculated_summary_cols = {'is_property_damage': is_property_damage}


def get_xref_df(il_inputs_df):
    top_level_layers_df = il_inputs_df.loc[il_inputs_df['level_id'] == il_inputs_df['level_id'].max(),
                                           ['top_agg_id'] + SUMMARY_TOP_LEVEL_COLS].drop_duplicates()
    bottom_level_layers_df = il_inputs_df[il_inputs_df['level_id'] == 0].copy()
    bottom_level_layers_df.drop(columns=SUMMARY_TOP_LEVEL_COLS, inplace=True)
    return (merge_dataframes(bottom_level_layers_df, top_level_layers_df, join_on=['top_agg_id'])
            .drop_duplicates(subset=['gul_input_id', 'layer_id'], keep='first')
            .sort_values(['gul_input_id', 'layer_id'], kind='stable')
            )


@oasis_log
def get_summary_mapping(inputs_df, oed_hierarchy, is_fm_summary=False):
    """Create a DataFrame with linking information between Ktools `OasisFiles`
    And the Exposure data

    Args:
        inputs_df (pandas.DataFrame): datafame from gul_inputs.get_gul_input_items(..)  / il_inputs.get_il_input_items(..)
        oed_hierarchy (dict): OED profile hierarchy, used to resolve the acc/loc/pol/port column names to keep
        is_fm_summary (bool): Indicates whether an FM summary mapping is required

    Returns:
        pandas.DataFrame: Subset of columns from gul_inputs_df / il_inputs_df
    """
    # Case GUL+FM (based on il_inputs_df)
    if is_fm_summary:
        summary_mapping = inputs_df.rename(columns={'gul_input_id': 'agg_id'})

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
    """Create a factorized col (summary ids) based on a list of oed column names

    {'Col_A': 0, 'Col_B': 1, 'Col_C': 2}

    Args:
        summary_map_df (pandas.DataFrame): dataframe return from get_summary_mapping
        exposure_df (pandas.DataFrame): Summary map file path
        oed_column_join (list): column to join on
        oed_column_info (dict): Dictionary of columns to pick from exposure_df and their default value

    Returns:
        pandas.DataFrame: New DataFrame of summary_map_df + exposure_df merged on exposure index
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
    """Adds list of OED fields from `column_set` to summary map file

    Args:
        oed_col_group (list): OED column names the rows are grouped by to form the summaries
        summary_map_df (pandas.DataFrame): dataframe return from get_summary_mapping
        exposure_df (pandas.DataFrame): DataFrame loaded from location.csv
        sort_by (str): column the grouped rows are ordered by
        accounts_df (pandas.DataFrame): DataFrame loaded from accounts.csv

    Returns:
        tuple: a 3-tuple ``(summary_ids, summary_id_values, summary_tiv)``:

        - summary_ids (numpy.ndarray): integer summary id (1..n) for each row,
          e.g. ``array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, ...])``
        - summary_id_values (numpy.ndarray): distinct values used to factorize the
          rows into summaries, e.g. ``array(['Layer1', 'Layer2'], dtype=object)``
        - summary_tiv (pandas.DataFrame): total TIV aggregated per summary group
    """
    oed_cols = oed_col_group  # All required columns
    exposure_cols = [c for c in oed_cols if c not in summary_map_df.columns
                     and c not in calculated_summary_cols]  # columns which are in locations / Accounts file
    mapped_cols = [c for c in oed_cols + [SOURCE_IDX['loc'], SOURCE_IDX['acc'], sort_by]
                   if c in summary_map_df.columns]  # Columns already in summary_map_df
    to_calculate_column = [c for c in oed_cols if c in calculated_summary_cols]

    tiv_cols = ['tiv', 'loc_id', 'building_id', 'coverage_type_id']

    # Extract mapped_cols from summary_map_df
    summary_group_df = summary_map_df.loc[:, list(set(tiv_cols).union(mapped_cols))]

    # Search Loc / Acc files and merge in remaining
    if exposure_cols:
        # Subject-at-risk (location / account) file columns
        exposure_cols_loc = [c for c in exposure_cols if c in exposure_df.columns]
        exposure_col_df = exposure_df.loc[:, exposure_cols_loc + [SAR_ID]].drop_duplicates(subset=SAR_ID, keep='first')
        summary_group_df = merge_dataframes(summary_group_df, exposure_col_df, join_on=SAR_ID, how='left')

        # Account file columns
        if isinstance(accounts_df, pd.DataFrame):
            accounts_cols = [c for c in exposure_cols if c in set(accounts_df.columns) - set(exposure_df.columns)]
            if accounts_cols:
                accounts_col_df = accounts_df.loc[:, accounts_cols + [SOURCE_IDX['acc']]]
                summary_group_df = merge_dataframes(summary_group_df, accounts_col_df, join_on=SOURCE_IDX['acc'], how='left')

    for col in to_calculate_column:
        summary_group_df = calculated_summary_cols[col](summary_group_df)

    fill_na_with_categoricals(summary_group_df, 0)
    summary_group_df.sort_values(by=[sort_by], inplace=True, kind='stable')
    summary_ids = factorize_dataframe(summary_group_df, by_col_labels=oed_cols)
    summary_tiv = summary_group_df.drop_duplicates(['loc_id', 'building_id', 'coverage_type_id'] + oed_col_group,
                                                   keep='first').groupby(oed_col_group, observed=True).agg({'tiv': "sum"})

    return summary_ids[0], summary_ids[1], summary_tiv


@oasis_log
def write_summary_levels(exposure_df, accounts_df, exposure_data, target_dir):
    """Json file with list Available / Recommended columns for use in the summary reporting

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
    """
    # Manage internal columns, (Non-OED exposure input)
    int_excluded_cols = ['loc_id', SOURCE_IDX['loc']]
    desc_non_oed = 'Not an OED field'
    int_oasis_cols = {
        'coverage_type_id': 'Oasis coverage type',
        'peril_id': 'OED peril code',
        'coverage_id': 'Oasis coverage identifier',
    }

    # GUL perspective (loc columns only)
    l_col_list = (exposure_df.drop(columns=['peril_group_id'], errors='ignore')
                             .replace(0, np.nan)
                             .dropna(how='any', axis=1)
                             .columns.to_list())
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


def _default_csv_chunksize(n_rows):
    """Row count to pass as `to_csv`'s `chunksize`.

    pandas' to_csv converts pyarrow-backed columns to NumPy once per chunk before
    formatting each value, so a small default chunk size (as when chunksize is left
    unset) pays that conversion many times over for a large frame. Capped at 200k
    rows so a very large frame isn't held as a single chunk in memory.
    """
    return min(2 * 10 ** 5, max(n_rows, 1000))


@oasis_log
def write_mapping_file(sum_inputs_df, target_dir, is_fm_summary=False, intermediary_csv=False):
    """Writes a summary map file, used to build summarycalc xref files.

    Args:
        sum_inputs_df (pandas.DataFrame): dataframe return from get_summary_mapping
        target_dir (str): directory the summary map file is written to
        is_fm_summary (bool): Indicates whether an FM summary mapping is required
        intermediary_csv (bool): If True, also write a csv copy of the summary map

    Returns:
        str: Summary map file path
    """
    target_dir = as_path(
        target_dir,
        'Target IL input files directory',
        is_dir=True,
        preexists=False
    )

    map_fn = SUMMARY_MAPPING['fm_map_fn'] if is_fm_summary else SUMMARY_MAPPING['gul_map_fn']
    if map_fn.endswith('.parquet'):
        sum_mapping_fp = write_df_to_parquet_file(sum_inputs_df, target_dir, map_fn)
        if intermediary_csv:
            write_df_to_csv_file(sum_inputs_df, target_dir, pathlib.Path(map_fn).with_suffix('.csv').name)
    else:
        sum_mapping_fp = write_df_to_csv_file(sum_inputs_df, target_dir, map_fn)

    return sum_mapping_fp


def get_summary_map_fp(input_dir, map_fn):
    """Return the path of a summary map file, falling back to the same file
    in the other supported format (parquet or csv) if it is not present,
    e.g. inputs generated by older versions of oasislmf.

    Args:
        input_dir (str): directory containing the summary map file
        map_fn (str): summary map file name, from SUMMARY_MAPPING

    Returns:
        str: Summary map file path
    """
    map_fp = pathlib.Path(input_dir, map_fn)
    if not map_fp.exists():
        alt_fp = map_fp.with_suffix('.csv' if map_fp.suffix == '.parquet' else '.parquet')
        if alt_fp.exists():
            return str(alt_fp)
    return str(map_fp)


def get_column_selection(summary_set):
    """Given a analysis_settings summary definition, return either
    1. the set of OED columns requested to group by
    2. If no information key 'oed_fields', then group all outputs into a single summary_set

    Args:
        summary_set (dict): summary group dictionary from the `analysis_settings.json`

    Returns:
        list: List of selected OED columns to create summary groups from
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
    """Return the contents of ri_layers.json

    Example:
    {
        "1": {
            "inuring_priority": 1,
            "risk_level": "LOC",
            "directory": "  ... /runs/ProgOasis-20190501145127/RI_1"
        }
    }

    Args:
        run_dir (str): The file path of the model run directory

    Returns:
        dict: metadata for the Reinsurance layers
    """
    return get_json(src_fp=os.path.join(run_dir, 'ri_layers.json'))


def get_ri_inuring_priority_output_levels(run_dir):
    """Load the mapping from OED InuringPriority to RI output level from the input directory.

    The mapping is created during input generation and records, for each OED
    InuringPriority value, the index of the last RI layer (output level) that
    belongs to that priority.  When an InuringPriority spans multiple risk
    levels (e.g. LOC and ACC), its output level is the highest RI layer index
    among those risk levels.

    Args:
        run_dir (str): Directory containing ``ri_inuring_priority_output_levels.json``

    Returns:
        dict: mapping ``{inuring_priority: output_level}`` with integer keys/values
    """
    raw = get_json(src_fp=os.path.join(run_dir, 'ri_inuring_priority_output_levels.json'))
    return {int(k): int(v) for k, v in raw.items()}


def get_ri_summaryxref_dirs(input_dir, analysis_settings, all_layers=False):
    """Resolve the RI layer directories that receive an ``fmsummaryxref`` file.

    ``analysis_settings['ri_inuring_priorities']`` holds OED InuringPriority
    values; the mapping written during input generation converts each to its RI
    output level (the last RI layer index for that priority). The final priority
    always receives output. Gross RL output is written for every RI layer, so
    pass ``all_layers=True`` for that case.

    This is the single definition of which RI directories are summarised - both
    :func:`generate_summaryxref_files` and loss generation resolve it here so
    the two cannot drift apart.

    Args:
        input_dir (str): The run directory's ``input`` sub directory, holding
            ``ri_layers.json`` and ``ri_inuring_priority_output_levels.json``
        analysis_settings (dict): Model analysis settings file
        all_layers (bool): If True return every RI layer directory rather than
            only the net output levels

    Returns:
        list[str]: Absolute paths of the RI layer directories, output level order

    Raises:
        OasisException: If a requested inuring priority is not in the structure
    """
    ri_settings = get_ri_settings(input_dir)
    inuring_priority_to_output_level = get_ri_inuring_priority_output_levels(input_dir)
    valid_oed_priorities = set(inuring_priority_to_output_level.keys())

    ri_inuring_priorities_oed = set(int(p) for p in analysis_settings.get('ri_inuring_priorities', []))
    ri_inuring_priorities_oed.add(max(valid_oed_priorities))

    if not ri_inuring_priorities_oed.issubset(valid_oed_priorities):
        ri_missing = ri_inuring_priorities_oed.difference(valid_oed_priorities)
        ri_missing = [str(p) for p in sorted(ri_missing)]
        missing_str = ', '.join(ri_missing[:-1])
        missing_str += ' and ' * (len(ri_missing) > 1) + ri_missing[-1]
        missing_str = ('priority ' if len(ri_missing) == 1 else 'priorities ') + missing_str
        raise OasisException(f'Requested outputs for inuring {missing_str} lie outside of scope.')

    if all_layers:
        ri_output_levels = {int(x) for x in ri_settings}
    else:
        ri_output_levels = {inuring_priority_to_output_level[p] for p in ri_inuring_priorities_oed}

    return [
        os.path.join(input_dir, os.path.basename(ri_settings[str(output_level)]['directory']))
        for output_level in sorted(ri_output_levels)
    ]


def write_df_to_csv_file(df, target_dir, filename):
    """Write a generated summary xref dataframe to disk in csv format.

    Args:
        df (pandas.DataFrame): The dataframe output of get_df( .. )
        target_dir (str): Abs directory to write a summary_xref file
        filename (str): Name of output file
    """
    target_dir = as_path(target_dir, 'Input files directory', is_dir=True, preexists=False)
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    chunksize = _default_csv_chunksize(len(df))
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
    """Write a generated summary xref dataframe to disk in parquet format.

    Args:
        df (pandas.DataFrame): The dataframe output of get_df( .. )
        target_dir (str): Abs directory to write a summary_xref file
        filename (str): Name of output file
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
    """Create a Dataframe for either gul / il / ri  based on a section
    from the analysis settings

    Args:
        map_df (pandas.DataFrame): Summary Map dataframe (GUL / IL)
        exposure_df (pandas.DataFrame): Location OED data
        accounts_df (pandas.DataFrame): Accounts OED data
        id_set_index (str): column of map_df the summary xref ids are taken from
        summaries_info_dict (list): list of dictionary definition for a summary group from the
            analysis_settings file, e.g.::

                [{
                    "id": 1,
                    "oed_fields": [],
                      ...
                  },

                  ...
                 ]
        summaries_type (str): Text label to use as key in summary description either ['gul', 'il', 'ri']

    Returns:
        summaryxref_df (pandas.DataFrame): Dataframe containing abstracted summary data for ktools
        summary_desc (dictionary): dictionary of dataFrames listing what summary_ids map to
    """
    summaryxref_df = pd.DataFrame()
    summary_desc = {}

    all_cols = set(map_df.columns.to_list() + exposure_df.columns.to_list() + list(calculated_summary_cols.keys()))
    if isinstance(accounts_df, pd.DataFrame):
        all_cols.update(accounts_df.columns.to_list())

    # Extract the summary id index column depending on id_set_index
    map_df.sort_values(id_set_index, inplace=True, kind='stable')
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
    ri=False, rl=False, intermediary_csv=False
):
    """Top level function for creating the summaryxref files from the manager.py

    Args:
        location_df (pandas.DataFrame): Source locations, joined to the summary map to build the
            summary groupings and their description files
        account_df (pandas.DataFrame): Source accounts, joined the same way. Required for the il,
            ri and rl summary levels
        model_run_fp (str): Model run directory file path
        analysis_settings (dict): Model analysis settings file
        il (bool): Boolean to indicate the insured loss level mode - false if the
            source accounts file path not provided to Oasis files gen.
        ri (bool): Boolean to indicate the RI loss level mode - false if the
            source accounts file path not provided to Oasis files gen.
        rl (bool): Boolean to indicate the RL loss level mode - false if the
            source accounts file path not provided to Oasis files gen.
        intermediary_csv (bool): If True, also write a csv copy of each summaryxref file
            alongside the binary
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

    # Verify account file + il_map file
    il_map_fp = get_summary_map_fp(os.path.join(model_run_fp, 'input'), SUMMARY_MAPPING['fm_map_fn'])
    if il_summaries or ri_summaries or rl_summaries:
        if account_df is None:
            raise OasisException('No account file found.')

        if not os.path.exists(il_map_fp):
            raise OasisException('No summary map file found.')

    # Load il_map if present
    if os.path.exists(il_map_fp):
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
        gul_map_fp = get_summary_map_fp(os.path.join(model_run_fp, 'input'), SUMMARY_MAPPING['gul_map_fn'])
        gul_map_df = get_dataframe(
            src_fp=gul_map_fp,
            lowercase_cols=False,
            col_dtypes=MAP_SUMMARY_DTYPES,
            empty_data_error_msg='No summary map file found.',
        )
        gul_map_df = gul_map_df[list(set(gul_map_df).intersection(MAP_SUMMARY_DTYPES))]

    if gul_summaries:
        # Load GUL summary map
        id_set_index = 'item_id'
        gul_summaryxref_df, gul_summary_desc = get_summary_xref_df(
            gul_map_df,
            location_df,
            account_df,
            analysis_settings['gul_summaries'],
            'gul',
            id_set_index
        )
        # Write Xref file
        df_to_ndarray(gul_summaryxref_df, gul_summary_xref_dtype).tofile(os.path.join(model_run_fp, 'input', f"{SUMMARY_OUTPUT['gul']}.bin"))
        if intermediary_csv:
            write_df_to_csv_file(gul_summaryxref_df, os.path.join(model_run_fp, 'input'), f"{SUMMARY_OUTPUT['gul']}.csv")

        # Write summary_id description files
        for desc_key in gul_summary_desc:
            if desc_key.split('.')[-1] == 'parquet':
                write_df_to_parquet_file(gul_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)
            else:
                write_df_to_csv_file(gul_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)

    if il_summaries:
        # Load FM summary map
        il_map_fp = get_summary_map_fp(os.path.join(model_run_fp, 'input'), SUMMARY_MAPPING['fm_map_fn'])
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
        df_to_ndarray(il_summaryxref_df, fm_summary_xref_dtype).tofile(os.path.join(model_run_fp, 'input', f"{SUMMARY_OUTPUT['il']}.bin"))
        if intermediary_csv:
            write_df_to_csv_file(il_summaryxref_df, os.path.join(model_run_fp, 'input'), f"{SUMMARY_OUTPUT['il']}.csv")

        # Write summary_id description files
        for desc_key in il_summary_desc:
            if desc_key.split('.')[-1] == 'parquet':
                write_df_to_parquet_file(il_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)
            else:
                write_df_to_csv_file(il_summary_desc[desc_key], os.path.join(model_run_fp, 'output'), desc_key)

    if ri_summaries or rl_summaries:
        if ('il_summaries' not in analysis_settings) or (not il_summaries):
            il_map_fp = get_summary_map_fp(os.path.join(model_run_fp, 'input'), SUMMARY_MAPPING['fm_map_fn'])
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
        # Write Xref file for each inuring priority where output has been requested.
        summary_ri_fps = get_ri_summaryxref_dirs(
            os.path.join(model_run_fp, 'input'), analysis_settings, all_layers=bool(rl_summaries)
        )

        for summary_ri_fp in summary_ri_fps:
            df_to_ndarray(ri_summaryxref_df, fm_summary_xref_dtype).tofile(os.path.join(summary_ri_fp, f"{SUMMARY_OUTPUT['il']}.bin"))
            if intermediary_csv:
                write_df_to_csv_file(ri_summaryxref_df, summary_ri_fp, f"{SUMMARY_OUTPUT['il']}.csv")

        # Write summary_id description files
        output_dir = os.path.join(model_run_fp, 'output')
        for desc_key in ri_summary_desc:
            if desc_key.split('.')[-1] == 'parquet':
                write_df_to_parquet_file(ri_summary_desc[desc_key], output_dir, desc_key)
            else:
                write_df_to_csv_file(ri_summary_desc[desc_key], output_dir, desc_key)

        # Copy the inuring-priority-to-output-level mapping into the output directory so it is
        # available alongside the results for downstream consumers.
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            os.path.join(model_run_fp, 'input', 'ri_inuring_priority_output_levels.json'),
            os.path.join(output_dir, 'ri_inuring_priority_output_levels.json'),
        )


def convert_col_name(col_name):
    """Convert a column from OED format to exposure summary report format. For
    example `CountryCode` will be converted to `country_code`.

    Args:
        col_name (str): original OED column name

    Returns:
        str: exposure summary report field name
    """
    col_list = [col_name[0].lower()]
    for i, c in enumerate(col_name[1:]):
        if c.isupper() and not col_name[i].isupper():
            col_list += ['_']
        col_list += c.lower()
    return ''.join(col_list)


EXPOSURE_SUMMARY_KEYS_COLS = ['loc_id', 'peril_id', 'coverage_type_id', 'status']

# columns the exposure summary builds itself, which a summary field of the same name would shadow
EXPOSURE_SUMMARY_RESERVED_FIELDS = {'loc_id', 'coverage_type_id', 'tiv', 'peril_id', 'status'}

SUMMARY_COUNT_NAMES = ['number_of_locations', 'number_of_buildings', 'number_of_risks']


def select_keys_columns(df, columns):
    """Select `columns` from a keys or keys errors frame, whichever header convention it uses.

    The keys files have OED style headers (`LocID`, `PerilID`, ...), which `get_dataframe` lowercases
    by default (`locid`, `perilid`, ...) and `get_gul_input_items` renames to snake case (`loc_id`,
    `peril_id`, ...). All three are accepted.

    Args:
        df (pandas.DataFrame): keys or keys errors frame
        columns (list[str]): snake case names of the columns to select

    Returns:
        pandas.DataFrame: a new frame holding the columns under their snake case names. An empty `df`
        gives an empty frame with every column, as an empty keys errors file is read without them.
    """
    by_normalised_name = {}
    for col in df.columns:
        by_normalised_name.setdefault(str(col).lower().replace('_', ''), col)

    selected = {}
    for col in columns:
        source_col = by_normalised_name.get(col.replace('_', ''))
        if source_col is not None:
            selected[col] = df[source_col].reset_index(drop=True)
        elif len(df):
            raise OasisException(f'Keys data has no "{col}" column, found columns: {list(df.columns)}')
        else:
            selected[col] = pd.Series([], dtype=object)

    return pd.DataFrame(selected, copy=False)


def _factorize_parts(parts):
    """Factorize a column split across several frames, as if they had been concatenated.

    The keys and keys errors are coded separately, which saves concatenating them: the keys hold
    the peril id as a categorical, which concatenation with the strings of the keys errors would
    turn back into one Python string per row.

    Args:
        parts (list[pandas.Series]): the pieces of the column, in concatenation order

    Returns:
        tuple: ``(codes, uniques)``, the integer code of every row of the concatenated column and the
        list of distinct values in order of first appearance, as `pandas.factorize` would give them
        with ``use_na_sentinel=False``.
    """
    uniques, code_of, all_codes = [], {}, []
    missing = object()  # NaN != NaN, so missing values share one key of their own

    for part in parts:
        codes, part_uniques = pd.factorize(part, sort=False, use_na_sentinel=False)
        remap = np.empty(len(part_uniques), dtype=np.int64)
        for i, value in enumerate(part_uniques.tolist()):
            key = missing if pd.api.types.is_scalar(value) and pd.isna(value) else value
            if key not in code_of:
                code_of[key] = len(uniques)
                uniques.append(value)
            remap[i] = code_of[key]
        all_codes.append(remap[codes])

    codes = np.concatenate(all_codes) if all_codes else np.empty(0, dtype=np.int64)
    return codes, uniques


def _value_codes(uniques):
    """The ``{value: code}`` lookup the summary fields are filled from, leaving out missing values.

    A missing value is listed among the field values but has no code, so its bucket is reported
    empty, as it always has been.
    """
    return {value: code for code, value in enumerate(uniques)
            if not (pd.api.types.is_scalar(value) and pd.isna(value))}


def get_exposure_summary_rows(exposure_df, exposure_profile, additional_fields):
    """Collect the per location values the exposure summary is built from.

    Args:
        exposure_df (pandas.DataFrame): source exposure dataframe
        exposure_profile (dict): profile defining exposure file
        additional_fields (list): extra exposure columns to group the summary by

    Returns:
        tuple: ``(rows, field_names, coverage_type_ids, tiv)``. ``rows`` has one row per exposure
        row, holding `loc_id`, the summary fields under their report names, `number_of_buildings` and
        `number_of_risks`. ``tiv`` is the matching ``(rows, coverages)`` array of TIVs, missing TIVs
        as 0, for the coverage types in ``coverage_type_ids``.
    """
    fields = {}
    for field_name in additional_fields:
        if field_name not in exposure_df.columns:
            logger.warning(f'exposure summary field not found: {field_name}')
        elif convert_col_name(field_name) in EXPOSURE_SUMMARY_RESERVED_FIELDS:
            logger.warning(f'exposure summary field ignored, its name is used by the summary itself: {field_name}')
        else:
            fields.setdefault(convert_col_name(field_name), field_name)

    # the first TIV column of each coverage type, in profile order
    tiv_cols = {}
    for term_info in exposure_profile.values():
        if term_info.get('FMTermType') == 'TIV' and term_info['ProfileElementName'] in exposure_df.columns:
            tiv_cols.setdefault(term_info['CoverageTypeID'], term_info['ProfileElementName'])
    coverage_type_ids = list(tiv_cols)

    rows = pd.DataFrame({'loc_id': exposure_df['loc_id'].to_numpy()})
    for name, field_name in fields.items():
        rows[name] = exposure_df[field_name].reset_index(drop=True)

    # assigned after the fields, which they overwrite if a field has the same name
    if 'NumberOfBuildings' in exposure_df:
        rows['number_of_buildings'] = exposure_df['NumberOfBuildings'].to_numpy()
    else:
        rows['number_of_buildings'] = 1
    rows['number_of_risks'] = rows['number_of_buildings']
    if 'IsAggregate' in exposure_df:
        rows.loc[(exposure_df['IsAggregate'] == 0).to_numpy(), 'number_of_risks'] = 1
    rows = rows.replace({'number_of_buildings': 0, 'number_of_risks': 0}, 1)

    tiv = np.zeros((len(rows), len(coverage_type_ids)))
    for j, col in enumerate(tiv_cols.values()):
        tiv[:, j] = pd.to_numeric(exposure_df[col], errors='coerce').fillna(0).to_numpy(dtype='float64')

    return rows, list(fields), coverage_type_ids, tiv


def _counts_array(col):
    """A building or risk count column as floats, missing counts as 0 as `sum` treats them."""
    return pd.to_numeric(col, errors='coerce').fillna(0).to_numpy(dtype='float64')


class _SummaryStats:
    """The ``(tiv, by_coverage, overall)`` dicts `fill_exposure_summary_field` reads, built up per group.

    Only the groups holding a location are stored, as `fill_exposure_summary_field` reports the rest
    as empty.
    """

    def __init__(self, coverage_type_ids):
        self.coverage_type_ids = coverage_type_ids
        self.tiv, self.by_coverage, self.overall = {}, {}, {}

    def as_tuple(self):
        return self.tiv, self.by_coverage, self.overall

    def add_coverage(self, key, cov, tiv, counts):
        """Add the TIV and counts of one group for one coverage.

        Args:
            key (tuple): the group, ``(field value code[, status code])``
            cov (int): coverage index
            tiv (float): TIV of the group
            counts (dict): location, building and risk counts of the group
        """
        if counts['number_of_locations']:
            key = key + (self.coverage_type_ids[cov],)
            self.tiv[key] = tiv
            self.by_coverage[key] = counts

    def add_overall(self, key, counts):
        """Add the counts of one group over all coverages, as `add_coverage`."""
        if counts['number_of_locations']:
            self.overall[key if len(key) > 1 else key[0]] = counts

    def add_coverage_by_value(self, key_suffix, cov, tiv, counts):
        """`add_coverage` for every field value code, from arrays indexed by the code."""
        for code in np.flatnonzero(counts['number_of_locations']).tolist():
            self.add_coverage((code,) + key_suffix, cov, tiv[code],
                              {name: counts[name][code] for name in SUMMARY_COUNT_NAMES})

    def add_overall_by_value(self, key_suffix, counts):
        """`add_overall` for every field value code, from arrays indexed by the code."""
        for code in np.flatnonzero(counts['number_of_locations']).tolist():
            self.add_overall((code,) + key_suffix, {name: counts[name][code] for name in SUMMARY_COUNT_NAMES})


class _FieldRows:
    """The exposure rows summarised for one additional summary field.

    One row per (location, field value), the first exposure row holding it, as the summary keeps
    the first row of each group. A location appears more than once only when exposure rows sharing
    its `loc_id` hold different values of the field.
    """

    def __init__(self, field_column, loc_codes, tiv, n_buildings, n_risks):
        value_codes, values = pd.factorize(field_column, sort=False)
        self.field_list = field_column.drop_duplicates().to_list()
        self.value_codes = _value_codes(values.tolist())
        self.n_values = len(values)

        rows = np.flatnonzero(value_codes >= 0)  # a missing value has no code, and is reported empty
        first = ~pd.Series(loc_codes[rows].astype(np.int64) * self.n_values + value_codes[rows]).duplicated().to_numpy()
        rows = rows[first]
        self.value, self.loc = value_codes[rows], loc_codes[rows]
        self.tiv = np.ascontiguousarray(tiv[rows].T)  # (coverage, row)
        self.n_buildings, self.n_risks = n_buildings[rows], n_risks[rows]

    def counts(self, mask):
        """Location, building and risk counts per field value code, over the rows in `mask`."""
        value = self.value[mask]
        return {
            'number_of_locations': np.bincount(value, minlength=self.n_values),
            'number_of_buildings': np.bincount(value, weights=self.n_buildings[mask], minlength=self.n_values),
            'number_of_risks': np.bincount(value, weights=self.n_risks[mask], minlength=self.n_values),
        }


@oasis_log
def get_exposure_summary(
        exposure_df,
        keys_df,
        exposure_profile=get_default_exposure_profile(),
        additional_fields=[]
):
    """Create exposure summary as dictionary of TIVs and number of locations
    grouped by peril and validity respectively. returns a python dict().

    Every location has a lookup status for every combination of coverage type (with a TIV column in
    the exposure) and peril (returned by the lookup): the statuses of its keys rows, or `noreturn` if
    it has none. The summary is aggregated one status at a time from a boolean (peril, coverage,
    location) grid of which combinations hold that status, rather than from rows for every combination.

    Args:
        exposure_df (pandas.DataFrame): source exposure dataframe
        keys_df (pandas.DataFrame or list): dataFrame holding keys data (success and errors), with
            the location, peril, coverage type and status columns under any naming
            `select_keys_columns` accepts, or a list of such frames
        exposure_profile (dict): profile defining exposure file
        additional_fields (list): extra exposure columns to group the summary by, on top of loc_id

    Returns:
        dict: Exposure summary dictionary
    """
    keys_parts = keys_df if isinstance(keys_df, (list, tuple)) else [keys_df]
    keys_parts = [select_keys_columns(part, EXPOSURE_SUMMARY_KEYS_COLS) for part in keys_parts]

    rows, field_names, coverage_type_ids, tiv = get_exposure_summary_rows(
        exposure_df, exposure_profile, additional_fields)
    n_buildings = _counts_array(rows['number_of_buildings'])
    n_risks = _counts_array(rows['number_of_risks'])

    # locations, taking the first exposure row of a loc_id shared by several
    loc_codes, loc_ids = pd.factorize(rows['loc_id'], sort=False)
    first_rows = np.flatnonzero(~pd.Series(loc_codes).duplicated().to_numpy())
    loc_tiv, loc_buildings, loc_risks = tiv[first_rows], n_buildings[first_rows], n_risks[first_rows]

    # code the keys; the perils of keys outside the exposure are still summarised
    peril_codes, perils = _factorize_parts([part['peril_id'] for part in keys_parts])
    status_codes, statuses = _factorize_parts([part['status'] for part in keys_parts])
    no_return = OASIS_KEYS_STATUS['noreturn']['id']
    if no_return not in statuses:
        statuses.append(no_return)
    no_return_code = statuses.index(no_return)
    missing_status = [i for i, status in enumerate(statuses) if pd.api.types.is_scalar(status) and pd.isna(status)]
    status_codes[np.isin(status_codes, missing_status)] = no_return_code

    keys_loc = pd.Index(loc_ids).get_indexer(pd.concat([part['loc_id'] for part in keys_parts], ignore_index=True))
    keys_cov = pd.Index(coverage_type_ids).get_indexer(
        pd.concat([part['coverage_type_id'] for part in keys_parts], ignore_index=True))
    del keys_parts
    in_exposure = (keys_loc >= 0) & (keys_cov >= 0)

    n_locs, n_covs, n_perils = len(loc_ids), len(coverage_type_ids), len(perils)
    # (peril, coverage, location), so that every location vector of the grid is contiguous
    grid_shape = (n_perils, n_covs, n_locs)
    cell = np.ravel_multi_index(
        (peril_codes[in_exposure], keys_cov[in_exposure], keys_loc[in_exposure]), grid_shape)
    cell_status = status_codes[in_exposure]
    del keys_loc, keys_cov, peril_codes, status_codes, in_exposure

    returned = np.zeros(grid_shape, dtype=bool)
    returned.reshape(-1)[cell] = True

    loc_tiv = np.ascontiguousarray(loc_tiv.T)  # (coverage, location)
    field_rows = {name: _FieldRows(rows[name], loc_codes, tiv, n_buildings, n_risks) for name in field_names}
    peril_stats = _SummaryStats(coverage_type_ids)
    field_stats = {name: _SummaryStats(coverage_type_ids) for name in field_names}
    in_scope = {'modelled': np.zeros((n_covs, n_locs), dtype=bool),
                'not-modelled': np.zeros((n_covs, n_locs), dtype=bool)}

    for status_code in sorted(set(np.unique(cell_status).tolist()) | {no_return_code}):
        # the (peril, coverage, location) combinations with this status
        has_status = np.zeros(grid_shape, dtype=bool)
        has_status.reshape(-1)[cell[cell_status == status_code]] = True
        if status_code == no_return_code:
            has_status |= ~returned

        perils_with_status = has_status.sum(axis=0, dtype=np.int64)  # (coverage, location)
        has_any_peril = perils_with_status > 0
        scope = 'modelled' if statuses[status_code] in OASIS_KEYS_STATUS_MODELLED else 'not-modelled'
        in_scope[scope] |= has_any_peril

        # by peril, from the first exposure row of each location
        for peril in range(n_perils):
            for cov in range(n_covs):
                mask = has_status[peril, cov]
                peril_stats.add_coverage((peril, status_code), cov, _masked_sum(loc_tiv[cov], mask),
                                         _location_counts(mask, loc_buildings, loc_risks))
            peril_stats.add_overall((peril, status_code),
                                    _location_counts(has_status[peril].any(axis=0), loc_buildings, loc_risks))

        # by each additional field, a location's TIV counting once per peril with the status
        for name, field in field_rows.items():
            n_perils_of_row = perils_with_status[:, field.loc]
            for cov in range(n_covs):
                field_stats[name].add_coverage_by_value((status_code,), cov, np.bincount(
                    field.value, weights=field.tiv[cov] * n_perils_of_row[cov], minlength=field.n_values),
                    field.counts(n_perils_of_row[cov] > 0))
            field_stats[name].add_overall_by_value((status_code,), field.counts(has_any_peril[:, field.loc].any(axis=0)))
        del has_status, perils_with_status, has_any_peril

    # over all statuses, every location has a status for every coverage type and peril
    has_rows = n_perils > 0 and n_covs > 0
    peril_stats_for_all = _SummaryStats(coverage_type_ids)
    all_locs = np.full(n_locs, has_rows)
    all_locs_counts = _location_counts(all_locs, loc_buildings, loc_risks)
    for peril in range(n_perils):
        for cov in range(n_covs):
            peril_stats_for_all.add_coverage((peril,), cov, _masked_sum(loc_tiv[cov], all_locs), all_locs_counts)
        peril_stats_for_all.add_overall((peril,), all_locs_counts)

    # Compile summary of exposure data
    exposure_summary = {}
    exposure_summary['total'] = {
        scope: _exposure_total(mask, loc_tiv, loc_buildings, loc_risks)
        for scope, mask in [('modelled', in_scope['modelled']),
                            ('not-modelled', in_scope['not-modelled']),
                            ('portfolio', np.full((n_covs, n_locs), has_rows))]
    }

    status_lookup = _value_codes(statuses)
    exposure_summary['peril_id'] = fill_exposure_summary_field(
        perils, _value_codes(perils), status_lookup,
        peril_stats.as_tuple(), peril_stats_for_all.as_tuple())

    for name, field in field_rows.items():
        stats_for_all = _SummaryStats(coverage_type_ids)
        every_row = np.full(len(field.value), has_rows)
        for cov in range(n_covs):
            stats_for_all.add_coverage_by_value((), cov, n_perils * np.bincount(
                field.value, weights=field.tiv[cov], minlength=field.n_values), field.counts(every_row))
        stats_for_all.add_overall_by_value((), field.counts(every_row))

        exposure_summary[name] = fill_exposure_summary_field(
            field.field_list, field.value_codes, status_lookup, field_stats[name].as_tuple(), stats_for_all.as_tuple())

    return exposure_summary


def _masked_sum(values, mask):
    """Sum of `values` where `mask` is set.

    Multiplies rather than calling `numpy.dot`, which hands a float/bool product to a multithreaded
    BLAS that is no faster here and takes every core.
    """
    return (values * mask).sum()


def _location_counts(mask, n_buildings, n_risks):
    """Location, building and risk counts over the locations in `mask`."""
    return {
        'number_of_locations': np.count_nonzero(mask),
        'number_of_buildings': _masked_sum(n_buildings, mask),
        'number_of_risks': _masked_sum(n_risks, mask),
    }


def _exposure_total(mask, loc_tiv, loc_buildings, loc_risks):
    """Total TIV and counts over the (coverage, location) pairs in `mask`."""
    locs = mask.any(axis=0)
    return {
        "tiv": _masked_sum(loc_tiv, mask),
        "number_of_locations": int(np.count_nonzero(locs)),
        "number_of_buildings": int(_masked_sum(loc_buildings, locs)),
        "number_of_risks": int(_masked_sum(loc_risks, locs)),
    }


def fill_exposure_summary_field(field_list, value_codes, status_codes, stats_by_status, stats_for_all):
    """Lay out the summary of one field, for every value and lookup status.

    Args:
        field_list (list): the field values to report, in report order
        value_codes (dict): code of each field value in the stats, a value without one being empty
        status_codes (dict): code of each lookup status in the stats, a status without one being empty
        stats_by_status (tuple): ``(tiv, by_coverage, overall)`` dicts, the TIV and the counts keyed
            by ``(value code, status code, coverage type id)`` and the counts over all coverage types
            keyed by ``(value code, status code)``
        stats_for_all (tuple): the same over all statuses, keyed by ``(value code, coverage type id)``
            and by the value code

    Returns:
        dict: the summary section of the field, holding the TIV and the location, building and risk
        counts per field value and lookup status
    """
    coverage_ids = [(coverage_type, info['id']) for coverage_type, info in SUPPORTED_COVERAGE_TYPES.items()]

    field_summary_by_value = {}
    for field_value in field_list:
        field_summary_by_value[field_value] = {}
        # Create dictionary structure for all and each validity status
        for status in ['all'] + list(OASIS_KEYS_STATUS.keys()):
            field_summary_by_value[field_value][status] = {
                'tiv': 0.0,
                'tiv_by_coverage': {},
                'number_of_locations': 0,
                'number_of_locations_by_coverage': {},
                'number_of_buildings': 0,
                'number_of_buildings_by_coverage': {},
                'number_of_risks': 0,
                'number_of_risks_by_coverage': {},
            }

    for status in ['all'] + list(OASIS_KEYS_STATUS.keys()):
        if status == 'all':
            (tiv, by_coverage, overall), group_key = stats_for_all, ()
        else:
            # a status absent from the frame has no code, and so matches no group
            (tiv, by_coverage, overall), group_key = stats_by_status, (status_codes.get(status),)

        for field_value in field_list:
            field_summary = field_summary_by_value[field_value][status]
            field_code = value_codes.get(field_value)

            for coverage_type, coverage_type_id in coverage_ids:
                coverage_key = (field_code,) + group_key + (coverage_type_id,)
                tiv_sum = float(tiv.get(coverage_key, 0.0))
                field_summary['tiv_by_coverage'][coverage_type] = tiv_sum
                field_summary['tiv'] += tiv_sum

                counts = by_coverage.get(coverage_key)
                for count_name in SUMMARY_COUNT_NAMES:
                    field_summary[f'{count_name}_by_coverage'][coverage_type] = int(counts[count_name]) if counts else 0

            counts = overall.get((field_code,) + group_key if group_key else field_code)
            for count_name in SUMMARY_COUNT_NAMES:
                field_summary[count_name] = int(counts[count_name]) if counts else 0

    return field_summary_by_value


@oasis_log
def write_gul_errors_map(
        target_dir,
        exposure_df,
        keys_errors_df,
        exposure_profile,
        intermediary_csv=False,
):
    """Create parquet file to help map keys errors back to original exposures.

    Args:
        target_dir (str): directory on disk to write parquet file
        exposure_df (pandas.DataFrame): source exposure dataframe
        keys_errors_df (pandas.DataFrame): keys errors dataframe
        exposure_profile (dict): profile defining exposure file
        intermediary_csv (bool): If True, also write a csv copy of the errors map

    Returns:
        str: gul errors map file path
    """
    cols = ['loc_id', 'PortNumber', 'AccNumber', 'LocNumber', 'peril_id', 'coverage_type_id', 'tiv', 'status', 'message']

    exposure_id_cols = ['loc_id', 'PortNumber', 'AccNumber', 'LocNumber']
    keys_error_cols = ['loc_id', 'peril_id', 'coverage_type_id', 'status', 'message']
    cov_level_id = SUPPORTED_FM_LEVELS['site coverage']['id']
    tiv_maps = {term_info['CoverageTypeID']: term_info['ProfileElementName'] for term_info in exposure_profile.values()
                if (term_info.get('FMTermType') == 'TIV'
                    and term_info.get('FMLevel') == cov_level_id
                    and term_info['ProfileElementName'] in exposure_df.columns)}

    exposure_cols = list(set(exposure_id_cols + list(tiv_maps.values())).intersection(exposure_df.columns))

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

    out_df = gul_inputs_errors_df[[c for c in cols if c in gul_inputs_errors_df.columns]].copy()
    # OED id columns can hold mixed types, which parquet cannot store in a single column.
    # Only cast non-null values: astype(str) turns NaN/None into 'nan'/'None' on pandas < 3
    # A column already of the string dtype astype(str) gives (pandas >= 3) holds nothing else, so is
    # left as it is: the keys errors are read that way, and casting their message is most of the time
    str_dtype = pd.Series(['']).astype(str).dtype
    str_cols = [c for c in ['PortNumber', 'AccNumber', 'LocNumber', 'peril_id', 'status', 'message'] if c in out_df.columns]
    for c in str_cols:
        if str_dtype == object or out_df[c].dtype != str_dtype:
            out_df[c] = out_df[c].where(out_df[c].isna(), out_df[c].astype(str))
    gul_error_map_fp = write_df_to_parquet_file(out_df, target_dir, 'gul_errors_map.parquet')
    if intermediary_csv:
        write_df_to_csv_file(out_df, target_dir, 'gul_errors_map.csv')
    return gul_error_map_fp


def _read_keys_file(fp, columns):
    """Read a keys or keys errors file, an unreadable (empty) file giving no rows."""
    try:
        df = get_dataframe(src_fp=fp, lowercase_cols=False)
    except OasisException:
        # Assume empty file on read error.
        df = pd.DataFrame()
    return select_keys_columns(df, columns)


@oasis_log
def write_exposure_summary(
        target_dir,
        exposure_df,
        keys_fp=None,
        keys_errors_fp=None,
        exposure_profile=get_default_exposure_profile(),
        additional_fields=[],
        intermediary_csv=False,
        keys_df=None,
        keys_errors_df=None,
):
    """Create exposure summary as dictionary of TIVs and number of locations
    grouped by peril and validity respectively. Writes dictionary as json file
    to disk.

    The keys can be given as frames already in memory, as `GenerateFiles` has them, or else are read
    from the keys files.

    Args:
        target_dir (str): directory on disk to write exposure summary file
        exposure_df (pandas.DataFrame): source exposure dataframe
        keys_fp (str): file path to keys file, read if `keys_df` is not given
        keys_errors_fp (str): file path to keys errors file, read if `keys_errors_df` is not given
        exposure_profile (dict): profile defining exposure file
        additional_fields (list[str]): list of additional OED fields to add to exposure summary file
        intermediary_csv (bool): If True, also write a csv copy of the gul errors map
        keys_df (pandas.DataFrame): keys, with the location, peril and coverage type columns under
            any naming `select_keys_columns` accepts. It is not modified.
        keys_errors_df (pandas.DataFrame): keys errors, likewise with status and message columns

    Returns:
        str: Exposure summary file path
    """
    keys_cols = ['loc_id', 'peril_id', 'coverage_type_id']
    keys_errors_cols = keys_cols + ['status', 'message']
    keys_parts = []

    if keys_df is not None:
        keys_success_df = select_keys_columns(keys_df, keys_cols)
    elif keys_fp:
        keys_success_df = _read_keys_file(keys_fp, keys_cols)
    else:
        keys_success_df = None
    if keys_success_df is not None:
        keys_success_df['status'] = pd.Categorical.from_codes(
            np.zeros(len(keys_success_df), dtype=np.int8), categories=[OASIS_KEYS_STATUS['success']['id']])
        keys_parts.append(keys_success_df)

    if keys_errors_df is not None:
        keys_errors_df = select_keys_columns(keys_errors_df, keys_errors_cols)
    elif keys_errors_fp:
        keys_errors_df = _read_keys_file(keys_errors_fp, keys_errors_cols)
    if keys_errors_df is not None:
        if not keys_errors_df.empty:
            write_gul_errors_map(target_dir, exposure_df, keys_errors_df, exposure_profile, intermediary_csv=intermediary_csv)
        keys_parts.append(keys_errors_df[EXPOSURE_SUMMARY_KEYS_COLS])

    exposure_summary = get_exposure_summary(exposure_df, keys_parts, exposure_profile, additional_fields)

    # write exposure summary as json file
    fp = os.path.join(target_dir, 'exposure_summary_report.json')
    with io.open(fp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(exposure_summary, ensure_ascii=False, indent=4))

    return fp
