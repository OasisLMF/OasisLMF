__all__ = [
    'get_summary_mapping',
    'create_summary_xref',
    'merge_oed_to_mapping',
    'write_mapping_file',
    'write_gulsummaryxref_file',
    'write_fmsummaryxref_file',
]

import os

import pandas as pd
import numpy as np

from ..utils.data import (
    #fast_zip_arrays,
    factorize_dataframe,
    factorize_ndarray,
    get_dataframe,
    merge_dataframes,
    set_dataframe_column_dtypes,
)
from ..utils.defaults import (
    SOURCE_FILENAMES,
    SUMMARY_MAPPING,
    SUMMARY_GROUPING,
)
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import as_path

"""
Module to generate the summarycalc input files

gulsummaryxref.csv
fmsummaryxref.csv

based on the analysis_settings.json (with added OED selection fields) + mapping file

Stage 1:  At input generatation (manager)
          Create mapping reference file
            a. Mode GUL
            b. Mode GUL + IL


Stage 2:  During model execution, generate summarycalc xref files
         a. Read mapping file
         b. Read read analysis_settings.json
         c. Create & write gulsummaryxref.csv / fmsummaryxref.csv based on mapping "run mode"

    import ipdb; ipbd.set_trace()
    pd.set_option('display.max_rows', 500)
"""


@oasis_log
def is_fm_summary(inputs_df):
    """
    Return type of summary_xref based on columns of data columns

    """
    col_headers_df = inputs_df.columns.to_list()
    #if oed_col['polid'] or 'output_id' in col_headers_df:
    if ('polnumber' in col_headers_df) or  ('output_id' in col_headers_df):
        return True
    else:
        return False



@oasis_log
def get_summary_mapping(
    inputs_df,
    oed_col):
    """

    inputs_df == (il_inputs_df || gul_inputs_df)

    oed_col == OrderedDict([
        ('accid', 'accnumber'),
        ('condid', 'condnumber'),
        ('locid', 'locnumber'),
        ('polid', 'polnumber'),
        ('portid', 'portnumber')
    ])
    """

    ## 'exposure_idx' maps -> idx in exposure_df (Join on this column when creating `__summaryxref.csv`)
    ## Select filter based on GUL / IF dataframe
    usecols = ([
        oed_col['accid'],
        oed_col['locid'],
        oed_col['polid'],
        oed_col['portid'],
        'exposure_idx',
        'item_id',
        'layer_id',
        'coverage_id',
        'peril_id',
        'agg_id',
        'output_id',
        'coverage_type_id',
        'tiv',
    ])


    #import ipdb; ipdb.set_trace()
    ## Case GUL+FM (based on il_inputs_df)
    if is_fm_summary(inputs_df):
        summary_mapping = inputs_df[inputs_df['level_id'] == inputs_df['level_id'].max()]
        summary_mapping['agg_id'] = summary_mapping.gul_input_id
        summary_mapping['output_id'] = factorize_ndarray(summary_mapping[['gul_input_id', 'layer_id']].values, col_idxs=range(2))[0]
        summary_mapping.drop('item_id' ,axis=1, inplace=True)

    ## GUL Only
    else:
        summary_mapping = inputs_df.copy(deep=True)


    #import ipdb; ipdb.set_trace()
    summary_mapping.drop(
        [c for c in summary_mapping.columns if c not in usecols],
        axis=1,
        inplace=True
    )
    return summary_mapping



@oasis_log
def merge_oed_to_mapping(summary_map_df, exposure_df, column_set):
    """
    Adds list of OED fields from `column_set` to summary map file

    :param :summary_map_df dataframe return from get_summary_mapping
    :type summary_map_df: pandas.DataFrame

    :param exposure_df: Summary map file path
    :type exposure_df: pandas.DataFrame

    :return: subset of columns from exposure_df to merge
    :rtype: list
    """

    # Guard check that set(column_set) < set(exposure_col_df.columns.to_list())
    # --> OasisExecption if not
    exposure_col_df = exposure_df[column_set]
    exposure_col_df['index'] = exposure_df.index.values
    return summary_map_df.merge(exposure_col_df, left_on='exposure_idx', right_on='index').drop('index', axis=1)


@oasis_log
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

    # Guard check that set(column_set) < set(exposure_col_df.columns.to_list())
    # --> OasisExecption if not


    # -> Check is column_set subset of summary_map_df.columns.to_list()


    # N -> import missing from OED
    import ipdb; ipdb.set_trace()
    # Note tidy this up later
    exposure_cols = [c for c in oed_col_group if c not in summary_map_df.columns.to_list()]
    mapping_cols = ['exposure_idx'] + [c for c in oed_col_group if c in summary_map_df.columns.to_list()]

    summary_group_df = summary_map_df[mapping_cols]
    if not exposure_cols == []:
        exposure_col_df = exposure_df[exposure_cols]
        exposure_col_df['index'] = exposure_df.index.values
        summary_group_df = summary_group_df.merge(exposure_col_df, left_on='exposure_idx', right_on='index').drop('index', axis=1)

    summary_ids = factorize_dataframe(summary_group_df, by_col_labels=oed_col_group)[0]
    return summary_ids


@oasis_log
def write_mapping_file(sum_inputs_df, target_dir):
    """
    Writes a summary map file, used to build summarycalc xref files.

    :param summary_mapping: dataframe return from get_summary_mapping
    :type summary_mapping: pandas.DataFrame

    :param sum_mapping_fp: Summary map file path
    :type sum_mapping_fp: str

    :return: Summary xref file path
    :rtype: str
    """
    # Clean the target directory path
    target_dir = as_path(
        target_dir,
        'Target IL input files directory',
        is_dir=True,
        preexists=False
    )

    # Set chunk size for writing the CSV files - default is 100K
    chunksize = min(2 * 10**5, len(sum_inputs_df))

    if is_fm_summary(sum_inputs_df):
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
    # Summary Set Not defined in settings file
    if "oed_fields" not in summary_set.keys():
        return None


    # Select Default Grouping for either 
    # ['lob', 'county', 'policy', 'state', 'location', 'prog']
    elif isinstance(summary_set['oed_fields'], str):
        if summary_set['oed_fields'] in SUMMARY_GROUPING.keys():
            # Load default based on group label
            return SUMMARY_GROUPING[summary_set['oed_fields']]
        else:
            pass
            # Raise OasisExpection (summary_set['oed_fields']) not set in default grouping set


    # Use OED column list set in analysis_settings file 
    elif isinstance(summary_set['oed_fields'], list) and len(summary_set['oed_fields']) > 0:
        return summary_set['oed_fields']
    else:
        pass
        ## Raise OasisExecption (Unable to process Settings file)



@oasis_log
def create_summary_xref(model_run_fp, analysis_settings):
    """
    Creates the summary_xref dataframes based on selection in analysis_settings file

    :param summary_map_df:
    :type :  ... pandas.DataFrame

    :param exposure_df:
    :type : ... pandas.DataFrame

    :param analysis_settings:
    :type : ...


    -> summary_map_df copy and drop all but needed cols


    for each set in analysis file
        -> Merge in OED rows for that set
        -> factorize_dataframe(sum, by_col_labels=['polnumber'])[0]
        -> appened to summary_xref dataframe


    ->


    Settings file:

        Sections (Summary files) "gul_summaries"
                                 "il_summaries"
                                 "ri_summaries"


        Subsections (Summary sets)
            "id" -> set id
            "oed_fields" -> Merge into mapping and group_by (or factorize)

    """


    # Load Exposure file for extra OED fields
    #
    # FIX Performace (Later) 
    #   - Only load rows which are reference in (GUL/FM)_map_df `exposure_idx`
    exposure_fp = os.path.join(model_run_fp, 'input', SOURCE_FILENAMES['loc'])
    exposure_df = get_dataframe(
        src_fp=exposure_fp,
        empty_data_error_msg='No source exposure file found.'
    )

    # Load GUL summary map
    gul_map_fp = os.path.join(model_run_fp, 'input', SUMMARY_MAPPING['gul_map_fn'])
    gul_map_df = get_dataframe(
        src_fp=gul_map_fp,
        empty_data_error_msg='No summary map file found.'
    )
    gul_map_size = len(gul_map_df)
    gul_summaryxref_df = pd.DataFrame()

    # For each granularity in GUL  
    import ipdb; ipdb.set_trace()
    for summary_set in analysis_settings['gul_summaries']:
        cols_group_by = get_column_selection(summary_set)
        gul_summaryset_df = gul_map_df[['coverage_id']]
        gul_summaryset_df['summaryset_id'] =  pd.Series(summary_set['id'], index=range(gul_map_size))
        #gul_summaryset_df[''] = pd.DataFrame(summary_set['id'], index=gul_map_size, columns=['summaryset_id'])

        if isinstance(cols_group_by, list):
            gul_summaryset_df['summary_id'] = group_by_oed(gul_map_df, exposure_df, cols_group_by)
        else:
            # Fall back to setting all in single group 
            gul_summaryset_df['summary_id'] =  pd.Series(1, index=range(gul_map_size))

        # Appends summary set to 'gulsummaryxref.csv'    
        gul_summaryxref_df.append(gul_summaryset_df)



    # If FM append 'output_id' as 'output'
    #
    # if GUL append 'coverage_id'

    # for



    # For first iteration RI -> Create single fm_summaryxref.csv file and copy into all `RI_n` dirs
    # for later UI / analysis_setting iterations aim to create each based on each `RI_n`



    #factorize_dataframe(sum, by_col_labels=['polnumber'])[0]
    #return summary_xref_df







'''
@oasis_log
def write_gulsummaryxref_file(gul_inputs_df, gulsummaryxref_fp, chunksize=100000):
    """
    Writes a summary xref file.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param gulsummaryxref_fp: Summary xref file path
    :type gulsummaryxref_fp: str

    :return: Summary xref file path
    :rtype: str
    """
    try:
        gul_inputs_df[['coverage_id', 'summary_id', 'summaryset_id']].drop_duplicates().to_csv(
            path_or_buf=gulsummaryxref_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(gulsummaryxref_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException from e

    return gulsummaryxref_fp



@oasis_log
def write_fmsummaryxref_file(il_inputs_df, fmsummaryxref_fp, chunksize=100000):
    """
    Writes a summary xref file.

    :param il_inputs_df: IL inputs dataframe
    :type il_inputs_df: pandas.DataFrame

    :param fmsummaryxref_fp: Summary xref file path
    :type fmsummaryxref_fp: str

    :return: Summary xref file path
    :rtype: str
    """
    try:
        cov_level_layers_df = il_inputs_df[il_inputs_df['level_id'] == il_inputs_df['level_id'].max()]
        pd.DataFrame(
            {
                'output': factorize_ndarray(cov_level_layers_df[['gul_input_id', 'layer_id']].values, col_idxs=range(2))[0],
                'summary_id': 1,
                'summaryset_id': 1
            }
        ).drop_duplicates().to_csv(
            path_or_buf=fmsummaryxref_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fmsummaryxref_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException from e

    return fmsummaryxref_fp



'''
