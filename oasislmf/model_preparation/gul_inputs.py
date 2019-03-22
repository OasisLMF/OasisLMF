# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

__all__ = [
    'get_gul_input_items',
    'write_coverages_file',
    'write_gulsummaryxref_file',
    'write_gul_input_files',
    'write_items_file',
    'write_complex_items_file'
]

import copy
import os
import multiprocessing
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import OrderedDict
from itertools import (
    chain,
    product,
)
from future.utils import (
    viewitems,
    viewkeys,
    viewvalues,
)

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from ..utils.concurrency import (
    multithread,
    Task,
)
from ..utils.data import (
    factorize_dataframe,
    get_dataframe,
    merge_dataframes,
)
from ..utils.defaults import (
    get_default_exposure_profile,
    OASIS_FILES_PREFIXES,
)
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.metadata import COVERAGE_TYPES
from ..utils.path import as_path
from .il_inputs import (
    unified_fm_profile_by_level_and_term_group,
    unified_fm_terms_by_level_and_term_group,
    unified_id_terms,
)


@oasis_log
def get_gul_input_items(
    exposure_fp,
    keys_fp,
    exposure_profile=get_default_exposure_profile()
):
    """
    Generates and returns a Pandas dataframe of GUL input items.

    :param exposure_fp: Exposure file
    :type exposure_fp: str

    :param keys_fp: Keys file path
    :type keys_fp: str

    :param exposure_profile: Exposure profile
    :type exposure_profile: dict

    :return: GUL inputs dataframe
    :rtype: pandas.DataFrame

    :return: Exposure dataframe
    :rtype: pandas.DataFrame
    """
    # Get the exposure profile and a higher-level profile from that with terms
    # grouped by level and term group (and also term type)
    exppf = exposure_profile
    ufp = unified_fm_profile_by_level_and_term_group(profiles=(exppf,))

    if not ufp:
        raise OasisException(
            'Source exposure profile is possibly missing FM term information: '
            'FM term definitions for TIV, limit, deductible, attachment and/or share.'
        )

    # Get another profile describing certain key ID columns in the exposure
    # file, namely loc. number, acc. number and portfolio number.
    id_terms = unified_id_terms(unified_profile_by_level_and_term_group=ufp)
    loc_id = id_terms['locid']
    acc_id = id_terms['accid']
    portfolio_num = id_terms['portid']

    # Get the TIV column names and corresponding coverage types
    tiv_terms = OrderedDict({v['tiv']['CoverageTypeID']:v['tiv']['ProfileElementName'].lower() for k, v in viewitems(ufp[1])})

    # Define the cov. level and get the cov. level IL/FM terms
    cov_level = COVERAGE_TYPES['buildings']['id']
    cov_fm_terms = unified_fm_terms_by_level_and_term_group(unified_profile_by_level_and_term_group=ufp)[cov_level]

    # Load the exposure and keys dataframes
    exposure_df = get_dataframe(
        src_fp=exposure_fp,
        col_dtypes={loc_id: 'str', acc_id: 'str', portfolio_num: 'str'},
        required_cols=(loc_id, acc_id, portfolio_num,),
        empty_data_error_msg='No data found in the source exposure (loc.) file',
        memory_map=True
    )

    for col in [v for v in viewvalues(tiv_terms)] + [_v for v in viewvalues(cov_fm_terms) for _v in viewvalues(v) if _v]:
        exposure_df[col] = exposure_df.get(col,  0.0)

    keys_df = get_dataframe(
        src_fp=keys_fp,
        col_dtypes={'locid': 'str'},
        empty_data_error_msg='No keys found in the keys file',
        memory_map=True
    )

    # If the keys file relates to a complex/custom model then look for a
    # ``modeldata`` column in the keys file, and ignore the area peril
    # and vulnerability ID columns
    if keys_df.get('modeldata'):
        keys_df.rename({'modeldata': 'model_data'}, inplace=True)
        keys_df['areaperilid'] = keys_df['vulnerabilityid'] = -1

    try:
        # Create the basic GUL inputs dataframe from merging the exposure and
        # keys dataframes on loc. number/loc. ID; filter out any rows with
        # zeros for TIVs for all coverage types, and replace any nulls in the
        # TIV columns with zeros
        gul_inputs_df = merge_dataframes(exposure_df, keys_df, left_on=loc_id, right_on='locid', how='inner')

        if gul_inputs_df.empty:
            raise OasisException(
                'Inner merge of the exposure file dataframe ({}) '
                'and the keys file dataframe ({}) on loc. number/loc. ID '
                'is empty - '
                'please check that the loc. number and loc. ID columns '
                'in the exposure and keys files respectively have a non-empty '
                'intersection'.format(exposure_fp, keys_fp)
            )

        del keys_df

        tiv_cols = [v for v in viewvalues(tiv_terms)]
        gul_inputs_df = gul_inputs_df[(gul_inputs_df[tiv_cols] != 0).any(axis=1)]
        gul_inputs_df.loc[:, tiv_cols] = gul_inputs_df[tiv_cols].where(gul_inputs_df.notnull(), 0.0)

        # Rename the peril ID, coverage type ID, area peril ID & vulnerability
        # ID columns (sourced from the keys frame)
        gul_inputs_df.rename(
            columns={
                'perilid': 'peril_id',
                'coveragetypeid': 'coverage_type_id',
                'areaperilid': 'areaperil_id',
                'vulnerabilityid': 'vulnerability_id'
            },
            inplace=True
        )

        # Set defaults for BI coverage boolean, TIV, deductibles and limit
        gul_inputs_df = gul_inputs_df.assign(
            is_bi_coverage=False, tiv=0.0, deductible=0.0, deductible_min=0.0, deductible_max=0.0, limit=0.0
        )

        # Group the rows in the GUL inputs table by coverage type, and set the
        # IL terms (and BI coverage boolean) in each group and update the
        # corresponding frame section in the GUL inputs table
        terms = ['tiv', 'deductible', 'deductible_min', 'deductible_max', 'limit']

        for cov_type, cov_type_group in gul_inputs_df.groupby(by=['coverage_type_id'], sort=True):
            cov_type_group['is_bi_coverage'] = np.where(cov_type == COVERAGE_TYPES['bi']['id'], True, False)
            term_cols = [tiv_terms[cov_type]] + [(term_col or term) for term, term_col in viewitems(cov_fm_terms[cov_type]) if term != 'share']
            cov_type_group.loc[:, term_cols] = cov_type_group.loc[:, term_cols].where(cov_type_group.notnull(), 0.0)
            cov_type_group.loc[:, terms] = cov_type_group.loc[:, term_cols].values
            cov_type_group = cov_type_group[(cov_type_group[['tiv']] != 0).any(axis=1)]
            if not cov_type_group.any().any():
                cov_type_group[terms] = 0.0
            else:
                cov_type_group['deductible'] = np.where(
                    (cov_type_group['deductible'] == 0) | (cov_type_group['deductible'] >= 1),
                    cov_type_group['deductible'],
                    cov_type_group['tiv'] * cov_type_group['deductible'],
                )
                cov_type_group['limit'] = np.where(
                    (cov_type_group['limit'] == 0) | (cov_type_group['limit'] >= 1),
                    cov_type_group['limit'],
                    cov_type_group['tiv'] * cov_type_group['limit'],
                )
            other_cov_type_term_cols = [v for k, v in viewitems(tiv_terms) if k != cov_type] + [
                _v for k, v in viewitems(cov_fm_terms) for _v in viewvalues(v) if _v if k != 1
            ]
            cov_type_group.loc[:, other_cov_type_term_cols] = 0
            gul_inputs_df.loc[cov_type_group.index, ['is_bi_coverage'] + terms] = cov_type_group[['is_bi_coverage'] + terms]

        # Remove any rows with zeros in the ``tiv`` column and reset the index
        gul_inputs_df = gul_inputs_df[(gul_inputs_df[['tiv']] != 0).any(axis=1)].reset_index()

        # Set the group ID - group by loc. number
        gul_inputs_df['group_id'] = factorize_dataframe(gul_inputs_df, [loc_id], enumerate_only=True)

        # Set the item IDs and coverage IDs, and defaults for summary and
        # summary set IDs
        item_ids = range(1, len(gul_inputs_df) + 1)
        gul_inputs_df = gul_inputs_df.assign(
            item_id=item_ids,
            coverage_id=item_ids,
            summary_id=1,
            summaryset_id=1
        )

        # Drop all unnecessary columns
        usecols = (
            [loc_id, acc_id, portfolio_num] +
            ['tiv'] + terms +
            ['peril_id', 'coverage_type_id', 'areaperil_id', 'vulnerability_id'] +
            (['model_data'] if 'model_data' in gul_inputs_df else []) +
            ['is_bi_coverage', 'group_id', 'item_id', 'coverage_id', 'summary_id', 'summaryset_id']
        )
        gul_inputs_df.drop(
            [c for c in gul_inputs_df.columns if c not in usecols],
            axis=1,
            inplace=True
        )
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        raise OasisException(e)

    return gul_inputs_df, exposure_df


def write_complex_items_file(gul_inputs_df, complex_items_fp, chunksize=100000):
    """
    Writes an items file.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param compex_items_fp: Complex/custom model items file path
    :type complex_items_fp: str

    :return: Complex/custom model items file path
    :rtype: str
    """
    try:
        gul_inputs_df.to_csv(
            columns=['item_id', 'coverage_id', 'model_data', 'group_id'],
            path_or_buf=complex_items_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(complex_items_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)


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
        gul_inputs_df.to_csv(
            columns=['item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id'],
            path_or_buf=items_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(items_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return items_fp


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
        gul_inputs_df.to_csv(
            columns=['coverage_id', 'tiv'],
            path_or_buf=coverages_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(coverages_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return coverages_fp


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
        gul_inputs_df.to_csv(
            columns=['coverage_id', 'summary_id', 'summaryset_id'],
            path_or_buf=gulsummaryxref_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(gulsummaryxref_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return gulsummaryxref_fp


@oasis_log
def write_gul_input_files(
    gul_inputs_df,
    target_dir,
    oasis_files_prefixes=copy.deepcopy(OASIS_FILES_PREFIXES['gul']),
    write_inputs_table_to_file=False
):
    """
    Writes the standard Oasis GUL input files to a target directory, using a
    pre-generated dataframe of GUL input items. The files written are
    ::

        items.csv
        coverages.csv
        gulsummaryxref.csv

    and optionally a complex items file in case of a complex/custom model.

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param target_dir: Target directory in which to write the files
    :type target_dir: str

    :param oasis_files_prefixes: Oasis GUL input file name prefixes
    :param oasis_files_prefixes: dict

    :param write_inputs_table_to_file: Whether to write the GUL inputs table to file
    :param write_inputs_table_to_file: bool

    :return: GUL input files dict
    :rtype: dict
    """
    # Clean the target directory path
    target_dir = as_path(target_dir, 'Target IL input files directory', is_dir=True, preexists=False)

    chunksize = min(10**5, len(gul_inputs_df))

    if write_inputs_table_to_file:
        gul_inputs_df.to_csv(path_or_buf=os.path.join(target_dir, 'gul_inputs.csv'), index=False, encoding='utf-8', chunksize=chunksize)

    if not gul_inputs_df.get('model_data'):
        if oasis_files_prefixes.get('complex_items'):
            oasis_files_prefixes.pop('complex_items')

    gul_input_files = {
        fn: os.path.join(target_dir, '{}.csv'.format(oasis_files_prefixes[fn])) 
        for fn in oasis_files_prefixes
    }

    this_module = sys.modules[__name__]

    if len(gul_inputs_df) <= chunksize:
        concurrent_tasks = (
            Task(getattr(this_module, 'write_{}_file'.format(fn)), args=(gul_inputs_df.copy(deep=True), gul_input_files[fn], chunksize,), key=fn)
            for fn in gul_input_files
        )
        num_ps = min(len(gul_input_files), multiprocessing.cpu_count())
        for _, _ in multithread(concurrent_tasks, pool_size=num_ps):
            pass
    else:
        for fn, fp in viewitems(gul_input_files):
            getattr(this_module, 'write_{}_file'.format(fn))(gul_inputs_df, fp, chunksize)

    return gul_input_files
