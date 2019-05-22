__all__ = [
    'DEDUCTIBLE_AND_LIMIT_TYPES',
    'DEDUCTIBLE_CODES',
    'FM_LEVELS',
    'FM_TERMS',
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
    'LIMIT_CODES',
    'OASIS_FILES_PREFIXES',
    'SUMMARY_MAPPING',
    'SUMMARY_GROUPING',
    'SUMMARY_OUTPUT',
    'SOURCE_IDX',
    'SOURCE_FILENAMES',
    'STATIC_DATA_FP',
    'SUPPORTED_FM_LEVELS'
]

import os

from collections import OrderedDict

from .data import (
    fast_zip_arrays,
    get_dataframe,
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

DED_CODE_REG = 0
DED_CODE_ANAGG = 1
DED_CODE_FRDED = 2
DED_CODE_NRDED = 3
DED_CODE_RSDED = 4
DED_CODE_CEAHO = 5
DED_CODE_CEAHOC = 6

DEDUCTIBLE_CODES = OrderedDict({
    'reg': {'id': DED_CODE_REG, 'desc': 'Regular'},
    'anagg': {'id': DED_CODE_ANAGG, 'desc': 'Annual aggregate'},
    'frded': {'id': DED_CODE_FRDED, 'desc': 'Franchise deductible'},
    'nrded': {'id': DED_CODE_NRDED, 'desc': 'Non-residential deductible'},
    'rsded': {'id': DED_CODE_RSDED, 'desc': 'Residential deductible'},
    'ceaho': {'id': DED_CODE_CEAHO, 'desc': 'CEA Homeowners'},
    'ceahoc': {'id': DED_CODE_CEAHOC, 'desc': 'CEA Homeowners Choice'}
})

DED_LIMIT_TYPE_FLT = 0
DED_LIMIT_TYPE_PCLOSS = 1
DED_LIMIT_TYPE_PCTIV = 2

DEDUCTIBLE_AND_LIMIT_TYPES = OrderedDict({
    'flat': {'id': DED_LIMIT_TYPE_FLT, 'desc': 'Flat monetary amount'},
    'pcloss': {'id': DED_LIMIT_TYPE_PCLOSS, 'desc': 'Percentage of loss'},
    'pctiv': {'id': DED_LIMIT_TYPE_PCTIV, 'desc': 'Percentage of TIV'}
})

FML_SITCOV = 1
FML_SITPDM = 2
FML_SITALL = 3
FML_CNDCOV = 4
FML_CNDPDM = 5
FML_CNDALL = 6
FML_POLCOV = 7
FML_POLPDM = 8
FML_POLALL = 9
FML_POLLAY = 10
FML_ACCCOV = 11
FML_ACCPDM = 12
FML_ACCALL = 13

FM_LEVELS = OrderedDict({
    'site coverage': {'id': FML_SITCOV, 'desc': 'site coverage'},
    'site pd': {'id': FML_SITPDM, 'desc': 'site property damage'},
    'site all': {'id': FML_SITALL, 'desc': 'site all (coverage + property damage)'},
    'cond coverage': {'id': FML_CNDCOV, 'desc': 'conditional coverage'},
    'cond pd': {'id': FML_CNDPDM, 'desc': 'conditional property damage'},
    'cond all': {'id': FML_CNDALL, 'desc': 'conditional all (coverage + property damage)'},
    'policy coverage': {'id': FML_POLCOV, 'desc': 'policy coverage'},
    'policy pd': {'id': FML_POLPDM, 'desc': 'policy property damage'},
    'policy all': {'id': FML_POLALL, 'desc': 'policy all (coverage + property damage)'},
    'policy layer': {'id': FML_POLLAY, 'desc': 'policy layer'},
    'account coverage': {'id': FML_ACCCOV, 'desc': 'account coverage'},
    'account pd': {'id': FML_ACCPDM, 'desc': 'account property damage'},
    'account all': {'id': FML_ACCALL, 'desc': 'account all (coverage + property damage)'}
})

SUPPORTED_FM_LEVELS = OrderedDict({
    level: level_dict for level, level_dict in FM_LEVELS.items()
    if level in ['site coverage', 'site pd', 'site all', 'cond all', 'policy all', 'policy layer']
})

FMT_DED = 'deductible'
FMT_DED_CODE = 'ded_code'
FMT_DED_TYPE = 'ded_type'
FMT_DED_MIN = 'deductible_min'
FMT_DED_MAX = 'deductible_max'
FMT_LIM = 'limit'
FMT_LIM_CODE = 'lim_code'
FMT_LIM_TYPE = 'lim_type'
FMT_ATT = 'attachment'
FMT_SHR = 'share'

FM_TERMS = OrderedDict({
    'deductible': {'id': FMT_DED, 'desc': 'Blanket deductible'},
    'deductible code': {'id': FMT_DED_CODE, 'desc': 'Blanket deductible code'},
    'deductible type': {'id': FMT_DED_TYPE, 'desc': 'Blanket deductible type'},
    'min deductible': {'id': FMT_DED_MIN, 'desc': 'Minimum deductible'},
    'max deductible': {'id': FMT_DED_MAX, 'desc': 'Maximum deductible'},
    'limit': {'id': FMT_LIM, 'desc': 'Limit'},
    'limit code': {'id': FMT_LIM_CODE, 'desc': 'Limit code'},
    'limit type': {'id': FMT_LIM_TYPE, 'desc': 'Limit type'},
    'attachment': {'id': FMT_ATT, 'desc': 'Attachment'},
    'share': {'id': FMT_SHR, 'desc': 'Share'}
})

LIM_CODE_REG = 0
LIM_CODE_ANAGG = 1

LIMIT_CODES = OrderedDict({
    'reg': {'id': LIM_CODE_REG, 'desc': 'Regular'},
    'anagg': {'id': LIM_CODE_ANAGG, 'desc': 'Annual aggregate'}
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


# Ktools runtime parameters - defaults used during model execution
KTOOLS_NUM_PROCESSES = 2
KTOOLS_MEM_LIMIT = False
KTOOLS_FIFO_RELATIVE = False
KTOOLS_ALLOC_RULE = 2
KTOOLS_DEBUG = False
