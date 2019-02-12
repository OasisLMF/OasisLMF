# -*- coding: utf-8 -*-

__all__ = [
    'get_default_accounts_profile',
    'get_default_deterministic_analysis_settings',
    'get_default_exposure_profile',
    'get_default_fm_aggregation_profile',
    KTOOLS_NUM_PROCESSES,
    KTOOLS_MEM_LIMIT,
    KTOOLS_FIFO_RELATIVE,
    KTOOLS_ALLOC_RULE
]

import os

from .data import get_json


static_data_fp = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), '_data')


def get_default_accounts_profile(data_fp=static_data_fp):
    return get_json(src_fp=os.path.join(data_fp, 'oed-acc-profile.json'))


def get_default_deterministic_analysis_settings(data_fp=static_data_fp):
    return get_json(src_fp=os.path.join(data_fp, 'analysis_settings.json'))


def get_default_exposure_profile(data_fp=static_data_fp):
    return get_json(src_fp=os.path.join(data_fp, 'oed-loc-profile.json'))


def get_default_fm_aggregation_profile(data_fp=static_data_fp):
    return get_json(src_fp=os.path.join(data_fp, 'fm-oed-agg-profile.json'), key_transform=int)


KTOOLS_NUM_PROCESSES = 2
KTOOLS_MEM_LIMIT = False
KTOOLS_FIFO_RELATIVE = False
KTOOLS_ALLOC_RULE = 2
