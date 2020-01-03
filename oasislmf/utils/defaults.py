__all__ = [
    'get_config_profile',
    'get_default_accounts_profile',
    'get_default_deterministic_analysis_settings',
    'get_default_exposure_profile',
    'get_default_fm_aggregation_profile',
    'get_default_unified_profile',
    'get_loc_dtypes',
    'get_acc_dtypes',
    'get_scope_dtypes',
    'get_info_dtypes',
    'store_exposure_fp',
    'find_exposure_fp',
    'GROUP_ID_COLS',
    'KTOOLS_ALLOC_GUL_MAX',
    'KTOOLS_ALLOC_FM_MAX',
    'KTOOLS_FIFO_RELATIVE',
    'KTOOLS_DEBUG',
    'KTOOLS_ERR_GUARD',
    'KTOOLS_NUM_PROCESSES',
    'OASIS_FILES_PREFIXES',
    'SUMMARY_MAPPING',
    'SUMMARY_OUTPUT',
    'SOURCE_IDX',
    'STATIC_DATA_FP',
    'WRITE_CHUNKSIZE'
]

import os
import glob

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

# Path for storing static data/metadata files used in the package
STATIC_DATA_FP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), '_data')

# Default profiles that describe the financial terms in the OED acc. and loc.
# (exposure) files, as well as how aggregation of FM input items is performed
# in the different OED FM levels



def store_exposure_fp(fp, exposure_type):
    """
    Preserve original exposure file extention if its in a pandas supported
    compressed format

    compression : {‘infer’, ‘gzip’, ‘bz2’, ‘zip’, ‘xz’, None}, default ‘infer’
                  For on-the-fly decompression of on-disk data. If ‘infer’ and 
                  filepath_or_buffer is path-like, then detect compression from 
                  the following extensions: ‘.gz’, ‘.bz2’, ‘.zip’, or ‘.xz’ 
                  (otherwise no decompression). 
                  
                  If using ‘zip’, the ZIP file must contain only one data file 
                  to be read in. Set to None for no decompression.

                New in version 0.18.1: support for ‘zip’ and ‘xz’ compression.
    """
    compressed_ext = ('.gz', '.bz2', '.zip','.xz')
    filename = SOURCE_FILENAMES[exposure_type]
    if fp.endswith(compressed_ext):
        return '.'.join([filename, fp.rsplit('.')[-1]])
    else:    
        return filename

def find_exposure_fp(input_dir, exposure_type):
    """ 
    Find an OED exposure file stored in the oasis inputs dir
    while preserving the compressed ext  
    """ 
    fp = glob.glob(os.path.join(input_dir, SOURCE_FILENAMES[exposure_type] + '*'))
    return fp.pop()

def get_default_accounts_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_acc_profile.json')
    return get_json(src_fp=fp) if not path else fp


def get_default_exposure_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_loc_profile.json')
    return get_json(src_fp=fp) if not path else fp


def get_config_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'config_compatibility_profile.json')
    return get_json(src_fp=fp) if not path else fp


def get_default_unified_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_unified_profile.json')
    return get_json(src_fp=fp) if not path else fp


def get_default_fm_aggregation_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_fm_agg_profile.json')
    return {int(k): v for k, v in get_json(src_fp=fp).items()} if not path else fp


def get_loc_dtypes():
    fp = os.path.join(STATIC_DATA_FP, 'loc_dtypes.json')
    return get_json(src_fp=fp)


def get_acc_dtypes():
    fp = os.path.join(STATIC_DATA_FP, 'acc_dtypes.json')
    return get_json(src_fp=fp)


def get_scope_dtypes():
    fp = os.path.join(STATIC_DATA_FP, 'scope_dtypes.json')
    return get_json(src_fp=fp)


def get_info_dtypes():
    fp = os.path.join(STATIC_DATA_FP, 'info_dtypes.json')
    return get_json(src_fp=fp)


WRITE_CHUNKSIZE = 2 * (10 ** 5)

GROUP_ID_COLS = ['loc_id']

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
KTOOLS_NUM_PROCESSES = -1
KTOOLS_FIFO_RELATIVE = False
KTOOLS_ERR_GUARD = True
KTOOLS_ALLOC_FM_MAX = 3
KTOOLS_ALLOC_GUL_MAX = 1     # 1 = new item stream, 0 = use prev Coverage stream
KTOOLS_ALLOC_GUL_DEFAULT = 1
KTOOLS_ALLOC_IL_DEFAULT = 2
KTOOLS_ALLOC_RI_DEFAULT = 3

KTOOLS_DEBUG = False
