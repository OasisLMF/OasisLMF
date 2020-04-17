
__all__ = [
    'generate_deterministic_losses'
]

import io
import json
import logging
import os
import re

from collections import OrderedDict
from itertools import product

from .defaults import (
    KTOOLS_MEAN_SAMPLE_IDX,
    KTOOLS_STD_DEV_SAMPLE_IDX,
    KTOOLS_TIV_SAMPLE_IDX
)

try:
    from json import JSONDecodeError
except ImportError:
    from builtins import ValueError as JSONDecodeError

from subprocess import (
    CalledProcessError,
    check_call,
)

import pandas as pd

from ..model_execution.bin import csv_to_bin

from .data import (
    fast_zip_dataframe_columns,
    get_dataframe,
    merge_dataframes,
    set_dataframe_column_dtypes,
)
from .defaults import (
    KTOOLS_ALLOC_IL_DEFAULT, KTOOLS_ALLOC_RI_DEFAULT)
from .exceptions import OasisException
from .log import oasis_log


@oasis_log
def generate_deterministic_losses(
    input_dir,
    output_dir=None,
    include_loss_factor=True,
    loss_factors=[1.0],
    net_ri=False,
    il_alloc_rule=KTOOLS_ALLOC_IL_DEFAULT,
    ri_alloc_rule=KTOOLS_ALLOC_RI_DEFAULT
):
    logger = logging.getLogger()
    losses = OrderedDict({
        'gul': None, 'il': None, 'ri': None
    })

    output_dir = output_dir or input_dir

    il = all(p in os.listdir(input_dir) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])

    ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(input_dir))

    step_flag = ''
    try:
        pd.read_csv(os.path.join(input_dir, 'fm_profile.csv'))['step_id']
    except (OSError, FileNotFoundError, KeyError):
        pass
    else:
        step_flag = '-S'

    csv_to_bin(input_dir, output_dir, il=il, ri=ri)

    # Generate an items and coverages dataframe and set column types (important!!)
    items = merge_dataframes(
        pd.read_csv(os.path.join(input_dir, 'items.csv')),
        pd.read_csv(os.path.join(input_dir, 'coverages.csv')),
        left_index=True, right_index=True
    )

    dtypes = {t: ('uint32' if t != 'tiv' else 'float32') for t in items.columns}
    items = set_dataframe_column_dtypes(items, dtypes)

    gulcalc_sidxs = \
        [KTOOLS_MEAN_SAMPLE_IDX, KTOOLS_STD_DEV_SAMPLE_IDX, KTOOLS_TIV_SAMPLE_IDX] + \
        list(range(1, len(loss_factors) + 1))

    # Set damage percentages corresponing to the special indexes.
    # We don't care about mean and std_dev, but
    # TIV needs to be set correctly.
    special_loss_factors = {
        KTOOLS_MEAN_SAMPLE_IDX: 0.,
        KTOOLS_STD_DEV_SAMPLE_IDX: 0.,
        KTOOLS_TIV_SAMPLE_IDX: 1.
    }

    guls_items = [
        OrderedDict({
            'event_id': 1,
            'item_id': item_id,
            'sidx': sidx,
            'loss':
            tiv * special_loss_factors[sidx] if sidx < 0
            else (tiv * loss_factors[sidx - 1])
        })
        for (item_id, tiv), sidx in product(
            fast_zip_dataframe_columns(items, ['item_id', 'tiv']), gulcalc_sidxs
        )
    ]

    guls = get_dataframe(
        src_data=guls_items,
        col_dtypes={
            'event_id': int,
            'item_id': int,
            'sidx': int,
            'loss': float})
    guls_fp = os.path.join(output_dir, "raw_guls.csv")
    guls.to_csv(guls_fp, index=False)

    ils_fp = os.path.join(output_dir, 'raw_ils.csv')
    cmd = 'gultobin -S {} < {} | fmcalc -p {} -a {} {} | tee ils.bin | fmtocsv > {}'.format(
        len(loss_factors), guls_fp, output_dir, il_alloc_rule, step_flag, ils_fp
    )
    try:
        logger.debug("RUN: " + cmd)
        check_call(cmd, shell=True)
    except CalledProcessError as e:
        raise OasisException("Exception raised in 'generate_deterministic_losses'", e)

    guls.drop(guls[guls['sidx'] < 1].index, inplace=True)
    guls.reset_index(drop=True, inplace=True)
    if include_loss_factor:
        guls['loss_factor_idx'] = guls.apply(
            lambda r: int(r['sidx'] - 1), axis='columns')
    guls.drop('sidx', axis=1, inplace=True)
    guls = guls[(guls[['loss']] != 0).any(axis=1)]

    losses['gul'] = guls

    ils = get_dataframe(src_fp=ils_fp)
    ils.drop(ils[ils['sidx'] < 0].index, inplace=True)
    ils.reset_index(drop=True, inplace=True)
    if include_loss_factor:
        ils['loss_factor_idx'] = ils.apply(
            lambda r: int(r['sidx'] - 1), axis='columns')
    ils.drop('sidx', axis=1, inplace=True)
    ils = ils[(ils[['loss']] != 0).any(axis=1)]
    losses['il'] = ils

    if ri:
        try:
            [fn for fn in os.listdir(input_dir) if fn == 'ri_layers.json'][0]
        except IndexError:
            raise OasisException(
                'No RI layers JSON file "ri_layers.json " found in the '
                'input directory despite presence of RI input files'
            )
        else:
            try:
                with io.open(os.path.join(input_dir, 'ri_layers.json'), 'r', encoding='utf-8') as f:
                    ri_layers = len(json.load(f))
            except (IOError, JSONDecodeError, OSError, TypeError) as e:
                raise OasisException('Error trying to read the RI layers file: {}'.format(e))
            else:
                def run_ri_layer(layer):
                    layer_inputs_fp = os.path.join(output_dir, 'RI_{}'.format(layer))
                    _input = 'gultobin -S 1 < {} | fmcalc -p {} -a {} {} | tee ils.bin |'.format(
                        guls_fp, output_dir, il_alloc_rule, step_flag
                    ) if layer == 1 else ''
                    pipe_in_previous_layer = '< ri{}.bin'.format(layer - 1) if layer > 1 else ''
                    ri_layer_fp = os.path.join(output_dir, 'ri{}.csv'.format(layer))
                    net_flag = "-n" if net_ri else ""
                    cmd = '{} fmcalc -p {} {} -a {} {} {} | tee ri{}.bin | fmtocsv > {}'.format(
                        _input,
                        layer_inputs_fp,
                        net_flag,
                        ri_alloc_rule,
                        pipe_in_previous_layer,
                        step_flag,
                        layer,
                        ri_layer_fp
                    )
                    try:
                        logger.debug("RUN: " + cmd)
                        check_call(cmd, shell=True)
                    except CalledProcessError as e:
                        raise OasisException("Exception raised in 'generate_deterministic_losses'", e)
                    rils = get_dataframe(src_fp=ri_layer_fp)
                    rils.drop(rils[rils['sidx'] < 0].index, inplace=True)
                    if include_loss_factor:
                        rils['loss_factor_idx'] = rils.apply(
                            lambda r: int(r['sidx'] - 1), axis='columns')

                    rils.drop('sidx', axis=1, inplace=True)
                    rils.reset_index(drop=True, inplace=True)
                    rils = rils[(rils[['loss']] != 0).any(axis=1)]

                    return rils

                for i in range(1, ri_layers + 1):
                    rils = run_ri_layer(i)
                    if i in [1, ri_layers]:
                        losses['ri'] = rils

    return losses
