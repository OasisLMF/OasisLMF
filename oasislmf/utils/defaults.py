# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

from future.utils import viewitems

__all__ = [
    'get_calc_rules',
    'get_default_accounts_profile',
    'get_default_deterministic_analysis_settings',
    'get_default_exposure_profile',
    'get_default_fm_aggregation_profile',
    'KTOOLS_NUM_PROCESSES',
    'KTOOLS_MEM_LIMIT',
    'KTOOLS_FIFO_RELATIVE',
    'KTOOLS_ALLOC_RULE',
    'OASIS_FILES_PREFIXES',
    'STATIC_DATA_FP'
]

import os

from collections import OrderedDict

from .data import (
    get_dataframe,
    get_json,
)


# Path for storing static data/metadata files used in the package
STATIC_DATA_FP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), '_data')

# Default profiles that describe the financial terms in the OED acc. and loc.
# (exposure) files, as well as how aggregation of FM input items is performed
# in the different OED FM levels
def get_default_accounts_profile(path=False):
    src_fp = os.path.join(STATIC_DATA_FP, 'oed-acc-profile.json')
    return get_json(src_fp=src_fp) if not path else src_fp


def get_default_exposure_profile(path=False):
    src_fp = os.path.join(STATIC_DATA_FP, 'oed-loc-profile.json')
    return get_json(src_fp=src_fp) if not path else src_fp


def get_default_fm_aggregation_profile(path=False):
    src_fp = os.path.join(STATIC_DATA_FP, 'fm-oed-agg-profile.json')
    return {int(k): v for k, v in viewitems(get_json(src_fp=src_fp))} if not path else src_fp


# Ktools calc. rules
def get_calc_rules(path=False):
    src_fp=os.path.join(STATIC_DATA_FP, 'calc-rules.csv')
    return get_dataframe(src_fp=src_fp) if not path else src_fp


# Default name prefixes of the Oasis input files (GUL + IL)
OASIS_FILES_PREFIXES = OrderedDict({
    'gul': {
        'complex_items': 'complex_items',
        'items': 'items',
        'coverages': 'coverages',
        'gulsummaryxref': 'gulsummaryxref'
    },
    'il': {
        'fm_policytc': 'fm_policytc',
        'fm_profile': 'fm_profile',
        'fm_programme': 'fm_programme',
        'fm_xref': 'fm_xref',
        'fmsummaryxref': 'fmsummaryxref'
    }
})


# Default analysis settings for deterministic loss generation
def get_default_deterministic_analysis_settings(path=False):
    src_fp = os.path.join(STATIC_DATA_FP, 'analysis_settings.json')
    return get_json(src_fp=src_fp) if not path else src_fp


# Ktools runtime parameters - defaults used during model execution
KTOOLS_NUM_PROCESSES = 2
KTOOLS_MEM_LIMIT = False
KTOOLS_FIFO_RELATIVE = False
KTOOLS_ALLOC_RULE = 2
