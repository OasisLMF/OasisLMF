__all__ = [
    'get_default_accounts_profile',
    'get_default_deterministic_analysis_settings',
    'get_default_exposure_profile',
    'get_default_fm_aggregation_profile',
    'get_default_unified_profile',
    'KTOOLS_ALLOC_RULE',
    'KTOOLS_FIFO_RELATIVE',
    'KTOOLS_DEBUG',
    'KTOOLS_MEM_LIMIT',
    'KTOOLS_NUM_PROCESSES',
    'OASIS_FILES_PREFIXES',
    'SUMMARY_MAPPING',
    'SUMMARY_GROUPING',
    'SUMMARY_OUTPUT',
    'SOURCE_IDX',
    'SOURCE_FILENAMES',
    'STATIC_DATA_FP'
]

import os

from collections import OrderedDict

from .data import (
    get_json,
)


SOURCE_FILENAMES = OrderedDict({
    'loc': 'location.csv',
    'acc': 'account.csv',
    'info': 'reinsinfo.csv',
    'scope': 'reinsscope.csv'
})

# Store index from merged source files (for later slice & dice)
SOURCE_IDX = OrderedDict({
    'loc': 'loc_idx',
    'acc': 'acc_idx',
    'info': 'info_idx',
    'scope': 'scope_idx'
})

SUMMARY_MAPPING = OrderedDict({
    'gul_map_fn': 'gul_summary_map.csv',
    'fm_map_fn': 'fm_summary_map.csv'
})

SUMMARY_OUTPUT = OrderedDict({
    'gul': 'gulsummaryxref.csv',
    'il': 'fmsummaryxref.csv'
})

# Update with load OED column names
# NOTE:  this should be removed once UI column picker feature has been added
SUMMARY_GROUPING = OrderedDict({
    'prog': None,
    'state': ['countrycode'],
    'county': ['geogname1'],
    'location': ['locnumber'],
    'lob': ['occupancycode'],    # <-- "Work around, this value should come from 'LOB' in the accounts file"
    'policy': ['polnumber'],
})

# Path for storing static data/metadata files used in the package
STATIC_DATA_FP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), '_data')

# Default profiles that describe the financial terms in the OED acc. and loc.
# (exposure) files, as well as how aggregation of FM input items is performed
# in the different OED FM levels


def get_default_accounts_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_acc_profile.json')
    return get_json(src_fp=fp) if not path else fp


def get_default_exposure_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_loc_profile.json')
    return get_json(src_fp=fp) if not path else fp


def get_default_unified_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_unified_profile.json')
    return get_json(src_fp=fp) if not path else fp


def get_default_fm_aggregation_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_fm_agg_profile.json')
    return {int(k): v for k, v in get_json(src_fp=fp).items()} if not path else fp


# Default name prefixes of the Oasis input files (GUL + IL)
OASIS_FILES_PREFIXES = OrderedDict({
    'gul': {
        'complex_items': 'complex_items',
        'items': 'items',
        'coverages': 'coverages',
    },
    'il': {
        'fm_policytc': 'fm_policytc',
        'fm_profile': 'fm_profile',
        'fm_programme': 'fm_programme',
        'fm_xref': 'fm_xref',
    }
})


# Default analysis settings for deterministic loss generation
def get_default_deterministic_analysis_settings(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'analysis_settings.json')
    return get_json(src_fp=fp) if not path else fp


# Defaults for Ktools runtime parameters
KTOOLS_NUM_PROCESSES = 2
KTOOLS_MEM_LIMIT = False
KTOOLS_FIFO_RELATIVE = False
KTOOLS_ALLOC_RULE = 2
KTOOLS_DEBUG = False
