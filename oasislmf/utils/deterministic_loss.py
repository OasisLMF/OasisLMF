
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
from .defaults import KTOOLS_ALLOC_IL_DEFAULT, KTOOLS_ALLOC_RI_DEFAULT
from .exceptions import OasisException
from .log import oasis_log

@oasis_log
def generate_deterministic_losses(
    input_dir,
    output_dir=None,
    loss_percentage_of_tiv=1.0,
    net_ri=False,
    il_alloc_rule=KTOOLS_ALLOC_IL_DEFAULT,
    ri_alloc_rule=KTOOLS_ALLOC_RI_DEFAULT
):
    logger = logging.getLogger()
    lf = loss_percentage_of_tiv
    losses = OrderedDict({
        'gul': None, 'il': None, 'ri': None
    })

    output_dir = output_dir or input_dir

    il = all(p in os.listdir(input_dir) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])

    ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(input_dir))

    csv_to_bin(input_dir, output_dir, il=il, ri=ri)

    # Generate an items and coverages dataframe and set column types (important!!)
    items = merge_dataframes(
        pd.read_csv(os.path.join(input_dir, 'items.csv')),
        pd.read_csv(os.path.join(input_dir, 'coverages.csv')),
        left_index=True, right_index=True
    )

    dtypes = {t: ('uint32' if t != 'tiv' else 'float32') for t in items.columns}

    items = set_dataframe_column_dtypes(items, dtypes)

    # Gulcalc sidx (sample index) list - -1 represents the numerical integration mean,
    # -2 the numerical integration standard deviation, and 1 the unsampled/raw loss
    gulcalc_sidxs = [-1, -2, 1]
    guls_items = [
        OrderedDict({'event_id': 1, 'item_id': item_id, 'sidx': sidx, 'loss': (tiv * lf if sidx != -2 else 0)})
        for (item_id, tiv), sidx in product(
            fast_zip_dataframe_columns(items, ['item_id', 'tiv']), gulcalc_sidxs
        )
    ]
    guls = get_dataframe(src_data=guls_items)
    guls_fp = os.path.join(output_dir, "raw_guls.csv")
    guls.to_csv(guls_fp, index=False)

    ils_fp = os.path.join(output_dir, 'raw_ils.csv')
    cmd = 'gultobin -S 1 < {} | fmcalc -p {} -a {} | tee ils.bin | fmtocsv > {}'.format(
        guls_fp, output_dir, il_alloc_rule, ils_fp
    )
    try:
        logger.debug("RUN: " + cmd)
        check_call(cmd, shell=True)
    except CalledProcessError as e:
        raise OasisException from e

    guls.drop(guls[guls['sidx'] != 1].index, inplace=True)
    guls.reset_index(drop=True, inplace=True)
    guls.drop('sidx', axis=1, inplace=True)
    guls = guls[(guls[['loss']] != 0).any(axis=1)]
    guls['item_id'] = guls.index + 1
    losses['gul'] = guls

    ils = get_dataframe(src_fp=ils_fp)
    ils.drop(ils[ils['sidx'] != (-1 if lf < 1.0 else -3)].index, inplace=True)
    ils.reset_index(drop=True, inplace=True)
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
                    _input = 'gultobin -S 1 < {} | fmcalc -p {} -a {} | tee ils.bin |'.format(
                        guls_fp, output_dir, il_alloc_rule
                    ) if layer == 1 else ''
                    pipe_in_previous_layer = '< ri{}.bin'.format(layer - 1) if layer > 1 else ''
                    ri_layer_fp = os.path.join(output_dir, 'ri{}.csv'.format(layer))
                    net_flag = "-n" if net_ri else ""
                    cmd = '{} fmcalc -p {} {} -a {} {}| tee ri{}.bin | fmtocsv > {}'.format(
                        _input,
                        layer_inputs_fp,
                        net_flag,
                        ri_alloc_rule,
                        pipe_in_previous_layer,
                        layer,
                        ri_layer_fp
                    )
                    try:
                        logger.debug("RUN: " + cmd)
                        check_call(cmd, shell=True)
                    except CalledProcessError as e:
                        raise OasisException from e
                    rils = get_dataframe(src_fp=ri_layer_fp)
                    rils.drop(rils[rils['sidx'] != (-1 if lf < 1 else -3)].index, inplace=True)
                    rils.drop('sidx', axis=1, inplace=True)
                    rils.reset_index(drop=True, inplace=True)
                    rils = rils[(rils[['loss']] != 0).any(axis=1)]

                    return rils

                for i in range(1, ri_layers + 1):
                    rils = run_ri_layer(i)
                    if i in [1, ri_layers]:
                        losses['ri'] = rils

    return losses
