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
import swifter

from ..utils.concurrency import (
    multithread,
    Task,
)
from ..utils.data import (
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
    get_sub_layer_calcrule_ids,
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

    # Load the exposure and keys dataframes
    exposure_df = get_dataframe(
        src_fp=exposure_fp,
        col_dtypes={loc_id: 'str', acc_id: 'str', portfolio_num: 'str'},
        required_cols=(loc_id, acc_id, portfolio_num,),
        empty_data_error_msg='No data found in the source exposure (loc.) file'
    )

    keys_df = get_dataframe(
        src_fp=keys_fp,
        col_dtypes={'locid': 'str'},
        empty_data_error_msg='No keys found in the keys file'
    )

    # If the keys file relates to a complex/custom model then look for a
    # ``modeldata`` column in the keys file, and ignore the area peril
    # and vulnerability ID columns
    if keys_df.get('modeldata'):
        keys_df.rename({'modeldata': 'model_data'}, inplace=True)
        keys_df['areaperilid'] = keys_df['vulnerabilityid'] = -1

    # Get the TIV column names and corresponding coverage types
    tiv_terms = OrderedDict({v['tiv']['CoverageTypeID']:v['tiv']['ProfileElementName'].lower() for k, v in viewitems(ufp[1])})

    # Define the cov. level and get the cov. level IL/FM terms
    cov_level = COVERAGE_TYPES['buildings']['id']
    cov_fm_terms = unified_fm_terms_by_level_and_term_group(unified_profile_by_level_and_term_group=ufp)[cov_level]

    try:
        # Create the basic GUL input dataframe from merging the exposure and
        # keys dataframes on loc. number/loc. ID, and filter out any row with
        # zero values for TIVs for all coverage types
        gul_inputs_df = merge_dataframes(exposure_df, keys_df, left_on=loc_id, right_on='locid', how='outer')
        gul_inputs_df = gul_inputs_df[(gul_inputs_df[[v for v in viewvalues(tiv_terms)]] != 0).any(axis=1)]

        # Set the group ID - group by loc. number
        gul_inputs_df['group_id'] = [
            gidx + 1 for gidx, (_, group) in enumerate(
                gul_inputs_df.groupby(by=[loc_id])) for _, (gidx, _) in enumerate(product([gidx], group[loc_id].tolist())
            )
        ]

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

        # Set the BI coverage boolean column - this is used during IL inputs
        # generation to exclude deductibles and limits relating to BI coverages
        # from being included in higher FM levels
        bi_cov_type = COVERAGE_TYPES['bi']['id']
        gul_inputs_df['is_bi_coverage'] = np.where(gul_inputs_df['coverage_type_id'] == bi_cov_type, True, False)
        #gul_inputs_df['is_bi_coverage'] = gul_inputs_df['coverage_type_id'].where(gul_inputs_df['coverage_type_id'] == COVERAGE_TYPES['bi']['id'], False)

        # A list of column names to use for processing the coverage level
        # IL terms
        reduced_cols = ['coverage_type_id'] + [v for v in viewvalues(tiv_terms)] + [v[t] for v in viewvalues(cov_fm_terms) for t in v if v[t]]

        # The coverage level FM/IL terms generator - various options were tried
        # for processing these terms into the GUL inputs table, including
        # ``pandas.DataFrame.apply``, but a ``for`` loop as used in this
        # generator was the quickest
        def _generate_il_terms():
            for _, row in gul_inputs_df[reduced_cols].iterrows():
                yield [
                    row[tiv_terms[row['coverage_type_id']]],
                    row.get(cov_fm_terms[row['coverage_type_id']].get('deductible')) or 0.0,
                    row.get(cov_fm_terms[row['coverage_type_id']].get('deductible_min')) or 0.0,
                    row.get(cov_fm_terms[row['coverage_type_id']].get('deductible_max')) or 0.0,
                    row.get(cov_fm_terms[row['coverage_type_id']].get('limit')) or 0.0
                ]

        # Process the coverage level IL terms
        tiv_and_il_term_cols = ['tiv', 'deductible', 'deductible_min', 'deductible_max', 'limit']
        gul_inputs_df = gul_inputs_df.join(pd.DataFrame(data=[it for it in _generate_il_terms()], columns=tiv_and_il_term_cols, index=gul_inputs_df.index))

        # For deductibles and limits convert any fractional values > 0 and < 1
        # to TIV shares
        gul_inputs_df['deductible'] = gul_inputs_df['deductible'].where(
            (gul_inputs_df['deductible'] == 0) | (gul_inputs_df['deductible'] >= 1),
            gul_inputs_df['tiv'] * gul_inputs_df['deductible'],
            axis=0
        )
        gul_inputs_df['limit'] = gul_inputs_df['limit'].where(
            (gul_inputs_df['limit'] == 0) | (gul_inputs_df['limit'] >= 1),
            gul_inputs_df['tiv'] * gul_inputs_df['limit'],
            axis=0
        )

        # Set the item IDs and coverage IDs, and defaults for summary and
        # summary set IDs
        item_ids = range(1, len(gul_inputs_df) + 1)
        gul_inputs_df = gul_inputs_df.assign(
            item_id=item_ids,
            coverage_id=item_ids,
            summary_id=1,
            summaryset_id=1
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
            mode='a',
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
            mode='a',
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
            mode='a',
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
            mode='a',
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return gulsummaryxref_fp


@oasis_log
def write_gul_input_files(
    exposure_fp,
    keys_fp,
    target_dir,
    exposure_profile=get_default_exposure_profile(),
    oasis_files_prefixes=copy.deepcopy(OASIS_FILES_PREFIXES['gul']),
    write_inputs_table_to_file=False
):
    """
    Writes the standard Oasis GUL input files, namely::

        items.csv
        coverages.csv
        gulsummaryxref.csv

    and optionally a complex items file in case of a complex/custom model.

    :param exposure_fp: Exposure file path
    :type exposure_fp: str

    :param keys_fp: Keys file path
    :type keys_fp: str

    :param target_dir: Target directory in which to write the files
    :type target_dir: str

    :param exposure_profile: Exposure profile (optional)
    :type exposure_profile: dict

    :param oasis_files_prefixes: Oasis GUL input file name prefixes
    :param oasis_files_prefixes: dict

    :param write_inputs_table_to_file: Whether to write the GUL inputs table to file
    :param write_inputs_table_to_file: bool

    :return: GUL input files dict
    :rtype: dict

    :return: GUL inputs dataframe
    :rtype: pandas.DataFrame

    :return: Exposure dataframe
    :rtype: pandas.DataFrame
    """
    # Clean the target directory path
    target_dir = as_path(target_dir, 'Target IL input files directory', is_dir=True, preexists=False)

    gul_inputs_df, exposure_df = get_gul_input_items(exposure_fp, keys_fp, exposure_profile=exposure_profile)

    if write_inputs_table_to_file:
        gul_inputs_df.to_csv(path_or_buf=os.path.join(target_dir, 'gul_inputs.csv'), index=False, encoding='utf-8', chunksize=100000)

    if not gul_inputs_df.get('model_data'):
        if oasis_files_prefixes.get('complex_items'):
            oasis_files_prefixes.pop('complex_items')

    gul_input_files = {
        fn: os.path.join(target_dir, '{}.csv'.format(oasis_files_prefixes[fn])) 
        for fn in oasis_files_prefixes
    }

    this_module = sys.modules[__name__]
    concurrent_tasks = (
        Task(getattr(this_module, 'write_{}_file'.format(fn)), args=(gul_inputs_df.copy(deep=True), gul_input_files[fn], 100000,), key=fn)
        for fn in gul_input_files
    )
    num_ps = min(len(gul_input_files), multiprocessing.cpu_count())
    for _, _ in multithread(concurrent_tasks, pool_size=num_ps):
        pass

    return gul_input_files, gul_inputs_df, exposure_df
