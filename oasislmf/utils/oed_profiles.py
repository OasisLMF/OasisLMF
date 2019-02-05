# -*- coding: utf-8 -*-

__all__ = [
    'get_default_canonical_oed_acc_profile',
    'get_default_canonical_oed_loc_profile',
    'get_default_fm_oed_aggregation_profile'
]

import io
import json
import os

from future.utils import iteritems

static_data_fp = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), '_data')


def get_default_canonical_oed_loc_profile(data_fp=static_data_fp):
    with io.open(os.path.join(static_data_fp, 'canonical-oed-loc-profile.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def get_default_canonical_oed_acc_profile(data_fp=static_data_fp):
    with io.open(os.path.join(static_data_fp, 'canonical-oed-acc-profile.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def get_default_fm_oed_aggregation_profile(data_fp=static_data_fp):
    with io.open(os.path.join(static_data_fp, 'fm-oed-agg-profile.json'), 'r', encoding='utf-8') as f:
        return {int(k): v for k, v in iteritems(json.load(f))}
