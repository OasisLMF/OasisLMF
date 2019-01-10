# -*- coding: utf-8 -*-

"""
Deterministic loss validation
"""
# Python standard library imports
import argparse
import io
import itertools
import json
import multiprocessing
import os
import six
import subprocess
import time

from shutil import copyfile

# Custom library imports
import pandas as pd

from tabulate import tabulate

# MDK imports
import oasislmf.model_execution.bin as ktools_bin
from oasislmf.exposures.manager import OasisExposuresManager as oem
from oasislmf.exposures import oed
from oasislmf.keys.lookup import OasisLookupFactory as olf
from oasislmf.utils.concurrency import (
    multithread,
    Task,
)

def generate_oasis_files(
    input_dir,
    srcexp_fp,
    srcexptocan_trans_fp,
    srcacc_fp,
    srcacctocan_trans_fp,
    canexp_profile,
    canacc_profile,
    fm_agg_profile
):
    """
    Generates Oasis input files (GUL + FM/IL) using OED source exposure +
    accounts data, and canonical OED exposure and accounts profiles and
    an FM OED aggregation profile, in the specified ``input_dir``, using
    simulated keys data. This is a model independent way of generating
    Oasis files.
    """

    # Create exposure manager instance
    manager = oem()

    # Generate the canonical loc./exposure and accounts files from the source files (in ``input_dir``)
    canexp_fp = os.path.join(input_dir, 'canexp.csv')
    manager.transform_source_to_canonical(
        source_exposures_file_path=srcexp_fp,
        source_to_canonical_exposures_transformation_file_path=srcexptocan_trans_fp,
        canonical_exposures_file_path=canexp_fp
    )
    canacc_fp = os.path.join(input_dir, 'canacc.csv')
    manager.transform_source_to_canonical(
        source_type='accounts',
        source_accounts_file_path=srcacc_fp,
        source_to_canonical_accounts_transformation_file_path=srcacctocan_trans_fp,
        canonical_accounts_file_path=canacc_fp
    )

    # Mock up the keys file (in ``input_dir``) - keys are generated for assumed
    # coverage type set of {1,2,3,4} present in the source exposure file. These
    # are the columns
    #
    # BuildingTIV,OtherTIV,ContentsTIV,BITIV
    #
    # This means that if there are n locations in the source file then 4 x n 
    # keys items are written out in the keys file. This means that 4 x n GUL
    # items will be present in the items and coverages and GUL summary xref files.
    n = len(pd.read_csv(srcexp_fp))
    keys = [
        {'id': i + 1 , 'peril_id': 1, 'coverage_type': j, 'area_peril_id': i + 1, 'vulnerability_id': i + 1}
        for i, j in itertools.product(range(n), [1,2,3,4])
    ]
    keys_fp, _ = olf.write_oasis_keys_file(keys, os.path.join(input_dir, 'keys.csv'))

    # Generate the GUL files (in ``input_dir``)
    gul_items_df, canexp_df = manager.load_gul_items(
        canexp_profile,
        canexp_fp,
        keys_fp
    )
    gul_files = {
        'items': os.path.join(input_dir, 'items.csv'),
        'coverages': os.path.join(input_dir, 'coverages.csv'),
        'gulsummaryxref': os.path.join(input_dir, 'gulsummaryxref.csv')
    }
    concurrent_tasks = (
        Task(getattr(manager, 'write_{}_file'.format(f)), args=(gul_items_df.copy(deep=True), gul_files[f],), key=f)
        for f in gul_files
    )
    num_ps = min(len(gul_files), multiprocessing.cpu_count())
    for _, _ in multithread(concurrent_tasks, pool_size=num_ps):
        pass

    # Generate the FM files (in ``input_dir``)
    fm_items_df, canacc_df = manager.load_fm_items(
        canexp_df,
        gul_items_df,
        canexp_profile,
        canacc_profile,
        canacc_fp,
        fmap
    )
    fm_files = {
        'fm_policytc': os.path.join(input_dir, 'fm_policytc.csv'),
        'fm_programme': os.path.join(input_dir, 'fm_programme.csv'),
        'fm_profile': os.path.join(input_dir, 'fm_profile.csv'),
        'fm_xref': os.path.join(input_dir, 'fm_xref.csv'),
        'fmsummaryxref': os.path.join(input_dir, 'fmsummaryxref.csv')
    }

    concurrent_tasks = (
        Task(getattr(manager, 'write_{}_file'.format(f)), args=(fm_items_df.copy(deep=True), fm_files[f],), key=f)
        for f in fm_files
    )
    num_ps = min(len(fm_files), multiprocessing.cpu_count())
    for _, _ in multithread(concurrent_tasks, pool_size=num_ps):
        pass

    # By this stage all the input files, including source and intermediate files
    # should be in ``input_dir``.

def apply_fm(input_dir, loss_percentage_of_tiv=1.0, net=False):
    # Generate an items and coverages dataframe and set column types (important!!)
    items_df = pd.merge(
        pd.read_csv(os.path.join(input_dir, 'items.csv')),
        pd.read_csv(os.path.join(input_dir, 'coverages.csv'))
    )
    for col in items_df:
        if col != 'tiv':
            items_df[col] = items_df[col].astype(int)
        else:
            items_df[col] = items_df[col].astype(float)

    guls_list = []
    for item_id, tiv in zip(items_df['item_id'], items_df['tiv']):
        event_loss = loss_percentage_of_tiv * tiv
        guls_list += [
            oed.GulRecord(event_id=1, item_id=item_id, sidx=-1, loss=event_loss),
            oed.GulRecord(event_id=1, item_id=item_id, sidx=-2, loss=0),
            oed.GulRecord(event_id=1, item_id=item_id, sidx=1, loss=event_loss)
        ]

    guls_df = pd.DataFrame(guls_list)
    guls_file = os.path.join(input_dir, "guls.csv")
    guls_df.to_csv(guls_file, index=False)

    net_flag = ""
    if net:
        net_flag = "-n"
    command = "gultobin -S 1 < {} | fmcalc -p {} {} -a {} | tee ils.bin | fmtocsv > ils.csv".format(
        guls_file, input_dir, net_flag, oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID)
    print(command)
    proc = subprocess.Popen(command, shell=True)
    proc.wait()
    if proc.returncode != 0:
        raise Exception("Failed to run fm")

    losses_df = pd.read_csv("ils.csv")
    losses_df.drop(losses_df[losses_df.sidx != 1].index, inplace=True)
    del losses_df['sidx']

    # Set ``event_id`` and ``output_id`` column data types to ``object``
    # to prevent ``tabulate`` from int -> float conversion during console printing
    losses_df['event_id'] = losses_df['event_id'].astype(object)
    losses_df['output_id'] = losses_df['output_id'].astype(object)

    guls_df.drop(guls_df[guls_df.sidx != 1].index, inplace=True)
    del guls_df['event_id']
    del guls_df['sidx']

    return losses_df
