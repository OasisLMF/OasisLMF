# -*- coding: utf-8 -*-

__all__ = [
    'generate_oasis_files',
    'generate_losses'
]

"""
Deterministic loss generation
"""

# Standard library imports
import copy
import itertools
import multiprocessing
import os
import shutil
import subprocess
import json

from collections import OrderedDict

from future.utils import (
    string_types,
    viewitems,
)

# 3rd party imports
import pandas as pd

# Oasis imports (including loading profile defaults)
from ..model_preparation import oed
from ..model_preparation.manager import OasisManager as om
from ..model_preparation.lookup import OasisLookupFactory as olf
from .concurrency import (
    multithread,
    Task,
)
from .exceptions import OasisException
from .oed_profiles import (
    get_default_canonical_oed_loc_profile,
    get_default_canonical_oed_acc_profile,
    get_default_fm_oed_aggregation_profile,
)
from ..model_execution.bin import (
    create_binary_files
)
from oasislmf.model_preparation.reinsurance_layer import (
    create_xref_description,
    generate_files_for_reinsurance,
)


def generate_oasis_files(input_dir, output_dir):

    srcexp_fp = os.path.join(input_dir, 'location.csv')
    srcexptocan_trans_fp = os.path.join(input_dir, 'MappingMapToOED_CanLocA.xslt')

    srcacc_fp = os.path.join(input_dir, 'account.csv')
    srcacctocan_trans_fp = os.path.join(input_dir, 'MappingMapToOED_CanAccA.xslt')

    _generate_il_files(
        output_dir,
        srcexp_fp,
        srcexptocan_trans_fp,
        srcacc_fp,
        srcacctocan_trans_fp
    )

    xref_descriptions = create_xref_description(pd.read_csv(srcacc_fp), pd.read_csv(srcexp_fp))
    (ri_info_df, ri_scope_df, do_reinsurance) = oed.load_oed_dfs(input_dir)
    ri_layers = None
    if do_reinsurance:
        oed_validator = oed.OedValidator()
        (is_valid, error_msgs) = oed_validator.validate(ri_info_df, ri_scope_df)
        if not is_valid:
            print("Validation Failed:")
            for m in error_msgs:
                print(json.dumps(m, indent=4, sort_keys=True))
            exit(1)
        items = pd.read_csv(os.path.join(output_dir, "items.csv"))
        coverages = pd.read_csv(os.path.join(output_dir, "coverages.csv"))
        fm_xrefs = pd.read_csv(os.path.join(output_dir, "fm_xref.csv"))
        ri_layers = generate_files_for_reinsurance(
            items=items,
            coverages=coverages,
            fm_xrefs=fm_xrefs,
            xref_descriptions=xref_descriptions,
            ri_info_df=ri_info_df,
            ri_scope_df=ri_scope_df,
            direct_oasis_files_dir=output_dir
        )

    return (ri_layers, xref_descriptions)


def _generate_il_files(
    target_dir,
    srcexp_fp,
    srcexptocan_trans_fp,
    srcacc_fp,
    srcacctocan_trans_fp,
    canexp_prof=get_default_canonical_oed_loc_profile(),
    canacc_prof=get_default_canonical_oed_acc_profile(),
    fm_agg_prof=get_default_fm_oed_aggregation_profile()
):
    """
    Generates model-independent Oasis input files (GUL + FM/IL) using OED
    source exposure + accounts data, and canonical OED exposure and accounts
    profiles and an FM OED aggregation profile, in the specified ``target_dir``,
    using simulated keys data.
    """

    # Create exposure manager instance
    manager = om()

    # Prepare input directory and asset target file paths
    _target_dir = ''.join(target_dir) if os.path.isabs(target_dir) else os.path.abspath(''.join(target_dir))
    if not os.path.exists(_target_dir):
        os.mkdir(_target_dir)

    _srcexp_fp = ''.join(srcexp_fp) if os.path.isabs(srcexp_fp) else os.path.abspath(''.join(srcexp_fp))
    fname = os.path.basename(_srcexp_fp)
    if not os.path.exists(os.path.join(_target_dir, fname)):
        _srcexp_fp = shutil.copy2(_srcexp_fp, _target_dir)

    _srcexptocan_trans_fp = ''.join(srcexptocan_trans_fp) if os.path.isabs(srcexptocan_trans_fp) else os.path.abspath(''.join(srcexptocan_trans_fp))
    fname = os.path.basename(_srcexptocan_trans_fp)
    if not os.path.exists(os.path.join(_target_dir, fname)):
        _srcexptocan_trans_fp = shutil.copy2(_srcexptocan_trans_fp, _target_dir)

    _srcacc_fp = ''.join(srcacc_fp) if os.path.isabs(srcacc_fp) else os.path.abspath(''.join(srcacc_fp))
    fname = os.path.basename(_srcacc_fp)
    if not os.path.exists(os.path.join(_target_dir, fname)):
        _srcacc_fp = shutil.copy2(_srcacc_fp, _target_dir)

    _srcacctocan_trans_fp = ''.join(srcacctocan_trans_fp) if os.path.isabs(srcacctocan_trans_fp) else os.path.abspath(''.join(srcacctocan_trans_fp))
    fname = os.path.basename(_srcacctocan_trans_fp)
    if not os.path.exists(os.path.join(_target_dir, fname)):
        _srcacctocan_trans_fp = shutil.copy2(_srcacctocan_trans_fp, _target_dir)

    _canexp_prof = canexp_prof
    if isinstance(_canexp_prof, string_types):
        canexp_prof_fp = ''.join(_canexp_prof) if os.path.isabs(_canexp_prof) else os.path.abspath(''.join(_canexp_prof))
        fname = os.path.basename(canexp_prof_fp)
        if not os.path.exists(os.path.join(_target_dir, fname)):
            canexp_prof_fp = shutil.copy2(canexp_prof_fp, _target_dir)
        _canexp_prof = manager.load_canonical_exposure_profile(canonical_exposure_profile_path=canexp_prof_fp)
    else:
        _canexp_prof = copy.deepcopy(canexp_prof)

    _canacc_prof = canacc_prof
    if isinstance(_canacc_prof, string_types):
        canacc_prof_fp = ''.join(_canacc_prof) if os.path.isabs(_canacc_prof) else os.path.abspath(''.join(_canacc_prof))
        fname = os.path.basename(canacc_prof_fp)
        if not os.path.exists(os.path.join(_target_dir, fname)):
            canacc_prof_fp = shutil.copy2(canacc_prof_fp, _target_dir)
        _canacc_prof = manager.load_canonical_accounts_profile(canonical_accounts_profile_path=canacc_prof_fp)
    else:
        _canacc_prof = copy.deepcopy(canacc_prof)

    _fm_agg_prof = fm_agg_prof
    if isinstance(_fm_agg_prof, string_types):
        fm_agg_prof_fp = ''.join(_fm_agg_prof) if os.path.isabs(_fm_agg_prof) else os.path.abspath(''.join(_fm_agg_prof))
        fname = os.path.basename(fm_agg_prof_fp)
        if not os.path.exists(os.path.join(_target_dir, fname)):
            fm_agg_prof_fp = shutil.copy2(fm_agg_prof_fp, _target_dir)
        _fm_agg_prof = manager.load_fm_aggregation_profile(fm_agg_prof_path=fm_agg_prof_fp)
    else:
        _fm_agg_prof = copy.deepcopy(fm_agg_prof)

    # Generate the canonical loc./exposure and accounts files from the source files (in ``target_dir``)
    canexp_fp = os.path.join(target_dir, 'canexp.csv')
    manager.transform_source_to_canonical(
        source_exposure_file_path=_srcexp_fp,
        source_to_canonical_exposure_transformation_file_path=_srcexptocan_trans_fp,
        canonical_exposure_file_path=canexp_fp
    )
    canacc_fp = os.path.join(target_dir, 'canacc.csv')
    manager.transform_source_to_canonical(
        source_type='accounts',
        source_accounts_file_path=_srcacc_fp,
        source_to_canonical_accounts_transformation_file_path=_srcacctocan_trans_fp,
        canonical_accounts_file_path=canacc_fp
    )

    # Mock up the keys file (in ``target_dir``) - keys are generated for assumed
    # coverage type set of {1,2,3,4} present in the source exposure file. These
    # are the columns
    #
    # BuildingTIV,OtherTIV,ContentsTIV,BITIV
    #
    # This means that if there are n locations in the source file then 4 x n
    # keys items are written out in the keys file. This means that 4 x n GUL
    # items will be present in the items and coverages and GUL summary xref files.
    n = len(pd.read_csv(_srcexp_fp))
    keys = [
        {'id': i + 1, 'peril_id': 1, 'coverage_type': j, 'area_peril_id': i + 1, 'vulnerability_id': i + 1}
        for i, j in itertools.product(range(n), [1, 2, 3, 4])
    ]
    keys_fp, _ = olf.write_oasis_keys_file(keys, os.path.join(target_dir, 'keys.csv'))

    # Load the canonical profile from the file path argument

    # Generate the GUL files (in ``target_dir``)
    gul_items_df, canexp_df = manager.get_gul_input_items(
        _canexp_prof,
        canexp_fp,
        keys_fp
    )
    gul_inputs = {
        'items': os.path.join(target_dir, 'items.csv'),
        'coverages': os.path.join(target_dir, 'coverages.csv'),
        'gulsummaryxref': os.path.join(target_dir, 'gulsummaryxref.csv')
    }
    tasks = (
        Task(getattr(manager, 'write_{}_file'.format(f)), args=(gul_items_df.copy(deep=True), gul_inputs[f],), key=f)
        for f in gul_inputs
    )
    num_ps = min(len(gul_inputs), multiprocessing.cpu_count())
    for _, _ in multithread(tasks, pool_size=num_ps):
        pass

    # Generate the FM files (in ``target_dir``)
    fm_items_df, canacc_df = manager.get_fm_input_items(
        canexp_df,
        gul_items_df,
        _canexp_prof,
        _canacc_prof,
        canacc_fp,
        _fm_agg_prof
    )
    fm_inputs = {
        'fm_policytc': os.path.join(target_dir, 'fm_policytc.csv'),
        'fm_programme': os.path.join(target_dir, 'fm_programme.csv'),
        'fm_profile': os.path.join(target_dir, 'fm_profile.csv'),
        'fm_xref': os.path.join(target_dir, 'fm_xref.csv'),
        'fmsummaryxref': os.path.join(target_dir, 'fmsummaryxref.csv')
    }

    tasks = (
        Task(getattr(manager, 'write_{}_file'.format(f)), args=(fm_items_df.copy(deep=True), fm_inputs[f],), key=f)
        for f in fm_inputs
    )
    num_ps = min(len(fm_inputs), multiprocessing.cpu_count())
    for _, _ in multithread(tasks, pool_size=num_ps):
        pass

    # By this stage all the input files, including source and intermediate files
    # should have been generated in ``target_dir``.

    return {k: v for k, v in itertools.chain(viewitems(gul_inputs), viewitems(fm_inputs))}


def generate_losses(output_dir, xref_descriptions, loss_percentage_of_tiv=1.0, ri_layers=None):

    net_losses = OrderedDict()
    net_losses['Direct'] = _generate_losses_il(output_dir, xref_descriptions, loss_percentage_of_tiv=1.0)

    if ri_layers is not None:
        for idx in ri_layers:
            if idx < 2:
                input_name = "ils"
            else:
                input_name = ri_layers[idx - 1]['directory']

            create_binary_files(ri_layers[idx]['directory'],
                                ri_layers[idx]['directory'],
                                do_il=True)
            reinsurance_layer_losses_df = _run_fm_ri(
                output_dir, input_name, ri_layers[idx]['directory'], xref_descriptions)
            output_name = "Inuring_priority:{} - Risk_level:{}".format(
                ri_layers[idx]['inuring_priority'],
                ri_layers[idx]['risk_level'])
            net_losses[output_name] = reinsurance_layer_losses_df
    return net_losses


def _generate_losses_il(input_dir, xref_descriptions, output_dir=None, loss_percentage_of_tiv=1.0, net=False):
    """
    Generates insured losses from preexisting Oasis files with a specified
    damage ratio (loss % of TIV).
    """
    _input_dir = ''.join(input_dir) if os.path.isabs(input_dir) else os.path.abspath(''.join(input_dir))

    if not os.path.exists(_input_dir):
        raise OasisException('Input directory containing Oasis files does not exist!')

    _output_dir = output_dir
    if not _output_dir:
        _output_dir = _input_dir
    else:
        _output_dir = ''.join(_output_dir) if os.path.isabs(_output_dir) else os.path.abspath(''.join(_output_dir))

    if not os.path.exists(_output_dir):
        os.mkdir(_output_dir)

    create_binary_files(_input_dir, _output_dir, do_il=True)

    # Generate an items and coverages dataframe and set column types (important!!)
    items_df = pd.merge(
        pd.read_csv(os.path.join(_input_dir, 'items.csv')),
        pd.read_csv(os.path.join(_input_dir, 'coverages.csv'))
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
    guls_fp = os.path.join(_output_dir, "guls.csv")
    guls_df.to_csv(guls_fp, index=False)

    return _run_fm_il(
        guls_df,
        _output_dir,
        xref_descriptions,
        allocation=oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID)


def _run_fm_il(
        guls_df,
        analysis_dir,
        xref_descriptions,
        allocation=oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID):

    guls_fp = os.path.join(analysis_dir, 'guls.csv')

    ils_fp = os.path.join(analysis_dir, 'ils.csv')
    ils_bin_fp = os.path.join(analysis_dir, 'ils.bin')

    command = "gultobin -S 1 < {0} | fmcalc -p {1} -a {2} | tee {3} | fmtocsv > {4}".format(
        guls_fp, analysis_dir, oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID, ils_bin_fp, ils_fp)
    print("\nRunning command: {}\n".format(command))
    proc = subprocess.Popen(command, shell=True)
    proc.wait()
    if proc.returncode != 0:
        raise OasisException("Failed to run fm")

    losses_df = pd.read_csv(ils_fp)

    losses_df.drop(losses_df[losses_df.sidx != 1].index, inplace=True)
    del losses_df['sidx']
    guls_df.drop(guls_df[guls_df.sidx != 1].index, inplace=True)
    del guls_df['event_id']
    del guls_df['sidx']
    guls_df = pd.merge(
        xref_descriptions,
        guls_df, left_on=['xref_id'], right_on=['item_id'])
    losses_df = pd.merge(
        guls_df,
        losses_df, left_on='xref_id', right_on='output_id',
        suffixes=["_gul", "_il"])
    del losses_df['event_id']
    del losses_df['output_id']
    del losses_df['xref_id']
    del losses_df['item_id']

    return losses_df


def _run_fm_ri(
        analysis_dir,
        input_name,
        output_name,
        xref_descriptions,
        allocation=oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID,):

    ceded_fp = os.path.join(analysis_dir, '{}.csv'.format(input_name))
    ceded_bin_fp = os.path.join(analysis_dir, '{}.bin'.format(input_name))

    net_fp = os.path.join(analysis_dir, '{}.csv'.format(output_name))
    net_bin_fp = os.path.join(analysis_dir, '{}.bin'.format(output_name))

    command = "fmcalc -p {0} -n -a {2} < {1} | tee {3} | fmtocsv > {4}".format(
        os.path.join(analysis_dir, output_name),
        ceded_bin_fp,
        allocation,
        net_bin_fp, net_fp)
    proc = subprocess.Popen(command, shell=True)
    proc.wait()
    if proc.returncode != 0:
        raise Exception("Failed to run fm")

    losses_df = pd.read_csv(net_fp)
    inputs_df = pd.read_csv(ceded_fp)

    losses_df.drop(losses_df[losses_df.sidx != 1].index, inplace=True)
    inputs_df.drop(inputs_df[inputs_df.sidx != 1].index, inplace=True)
    losses_df = pd.merge(
        inputs_df,
        losses_df, left_on='output_id', right_on='output_id',
        suffixes=('_ceded', '_net'))

    losses_df = pd.merge(
        xref_descriptions,
        losses_df, left_on='xref_id', right_on='output_id')

    del losses_df['event_id_ceded']
    del losses_df['sidx_ceded']
    del losses_df['event_id_net']
    del losses_df['sidx_net']
    del losses_df['output_id']
    del losses_df['xref_id']

    return losses_df
