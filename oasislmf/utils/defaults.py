__all__ = [
    'get_calc_rules',
    'get_default_accounts_profile',
    'get_default_deterministic_analysis_settings',
    'get_default_exposure_profile',
    'get_default_fm_aggregation_profile',
    'get_default_unified_profile',
    'KTOOLS_NUM_PROCESSES',
    'KTOOLS_MEM_LIMIT',
    'KTOOLS_FIFO_RELATIVE',
    'KTOOLS_ALLOC_RULE',
    'KTOOLS_DEBUG',
    'OASIS_FILES_PREFIXES',
    'STATIC_DATA_FP',
    'update_calc_rules'
]

import os

from collections import OrderedDict

from .data import (
    fast_zip_arrays,
    get_dataframe,
    get_json,
)


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


def update_calc_rules():
    fp = os.path.join(STATIC_DATA_FP, 'calc_rules.csv')

    terms = ['deductible', 'deductible_min', 'deductible_max', 'limit', 'share', 'attachment']
    terms_indicators = ['{}_gt_0'.format(t) for t in terms]
    types_and_codes = ['deductible_type', 'deductible_code', 'limit_type', 'limit_code']

    dtypes = {
        'id_key': 'str',
        **{t: 'int32' for t in ['calcrule_id'] + terms_indicators + types_and_codes}
    }
    calc_rules = get_dataframe(
        src_fp=fp,
        col_dtypes=dtypes
    )

    calc_rules['id_key'] = [t for t in fast_zip_arrays(*calc_rules[terms_indicators + types_and_codes].transpose().values)]

    calc_rules.to_csv(path_or_buf=fp, index=False, encoding='utf-8')


# Ktools calc. rules
def get_calc_rules(path=False, update=False):
    fp = os.path.join(STATIC_DATA_FP, 'calc_rules.csv')
    if path:
        return fp

    if update:
        update_calc_rules()

    return get_dataframe(src_fp=fp)


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
    fp = os.path.join(STATIC_DATA_FP, 'analysis_settings.json')
    return get_json(src_fp=fp) if not path else fp


# Ktools runtime parameters - defaults used during model execution
KTOOLS_NUM_PROCESSES = 2
KTOOLS_MEM_LIMIT = False
KTOOLS_FIFO_RELATIVE = False
KTOOLS_ALLOC_RULE = 2
KTOOLS_DEBUG = False
