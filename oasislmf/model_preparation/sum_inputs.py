__all__ = [
    'get_summary_mapping',
    'get_summary_xref',
    'merge_oed_to_mapping',
    'write_mapping_file',
    'write_gulsummaryxref_file',
    'write_fmsummaryxref_file',
]

import pandas as pd
import numpy as np

from ..utils.data import (
    factorize_ndarray,
    fast_zip_arrays,
    get_dataframe,
    merge_dataframes,
    set_dataframe_column_dtypes,
)

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log


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
    if oed_col['polid'] in inputs_df.columns:
        summary_mapping = inputs_df[inputs_df['level_id'] == inputs_df['level_id'].max()]
        summary_mapping['agg_id'] = summary_mapping.gul_input_id
        summary_mapping['output_id'] = factorize_ndarray(summary_mapping[['gul_input_id', 'layer_id']].values, col_idxs=range(2))[0]
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
    exposure_col_df = exposure_df[['index'] + column_set]
    return summary_map_df.merge(exposure_col_df, left_on='exposure_idx', right_on='index').drop('index', axis=1)


@oasis_log
def write_mapping_file(summary_map_df, sum_mapping_fp, chunksize=100000):
    """
    Writes a summary map file, used to build summarycalc xref files.

    :param summary_mapping: dataframe return from get_summary_mapping
    :type summary_mapping: pandas.DataFrame

    :param sum_mapping_fp: Summary map file path
    :type sum_mapping_fp: str

    :return: Summary xref file path
    :rtype: str
    """
    try:
        summary_map_df.to_csv(
            path_or_buf=sum_mapping_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(sum_mapping_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException from e

    return sum_mapping_fp


@oasis_log
def get_summary_xref(summary_map_df, exposure_df, analysis_settings):
    """
    Creates the summary_xref dataframes based on selection in analysis_settings file

    :param summary_map_df: 
    :type :  ... pandas.DataFrame

    :param exposure_df: 
    :type : ... pandas.DataFrame

    :param analysis_settings: 
    :type : ... 

    """

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
