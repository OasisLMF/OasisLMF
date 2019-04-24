__all__ = [
    'get_summary_mapping',
    'write_mapping_file',
    'write_gulsummaryxref_file',
    'write_fmsummaryxref_file',
    'write_fm_xref_file'
#    'write_gul_input_files',
#    'write_items_file',
#    'write_complex_items_file'
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

from ..utils.profiles import (
    get_grouped_fm_profile_by_level_and_term_group,
    get_grouped_fm_terms_by_level_and_term_group,
    get_oed_hierarchy_terms,
)

## Col names should be updated based on vars from ..utils.profiles --> get_oed_hierarchy_terms

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

    summary_mapping.drop(
        [c for c in summary_mapping.columns if c not in usecols],
        axis=1,
        inplace=True
    )
    return summary_mapping

