__all__ = [
    'get_config_profile',
    'get_default_accounts_profile',
    'get_default_deterministic_analysis_settings',
    'get_default_exposure_profile',
    'get_default_fm_profile_field_values',
    'get_default_step_policies_profile',
    'get_default_fm_aggregation_profile',
    'get_default_unified_profile',
    'get_loc_dtypes',
    'get_acc_dtypes',
    'get_scope_dtypes',
    'get_info_dtypes',
    'get_oed_default_values',
    'assign_defaults_to_il_inputs',
    'store_exposure_fp',
    'find_exposure_fp',
    'GROUP_ID_COLS',
    'CORRELATION_GROUP_ID',
    'API_EXAMPLE_AUTH',
    'KEY_NAME_TO_FILE_NAME',
    'DEFAULT_RTREE_INDEX_PROPS',
    'KTOOLS_ALLOC_GUL_MAX',
    'KTOOLS_ALLOC_FM_MAX',
    'KTOOLS_FIFO_RELATIVE',
    'KTOOLS_DEBUG',
    'KTOOLS_DISABLE_ERR_GUARD',
    'KTOOLS_NUM_PROCESSES',
    'KTOOLS_GUL_LEGACY_STREAM',
    'OASIS_FILES_PREFIXES',
    'SUMMARY_MAPPING',
    'SUMMARY_OUTPUT',
    'SOURCE_IDX',
    'STATIC_DATA_FP',
    'WRITE_CHUNKSIZE'
]

import os
import io
import glob
import json

from collections import OrderedDict
from itertools import chain

from .fm import SUPPORTED_FM_LEVELS
from .exceptions import OasisException

try:
    from json import JSONDecodeError
except ImportError:
    from builtins import ValueError as JSONDecodeError

SOURCE_FILENAMES = OrderedDict({
    'loc': 'location.csv',
    'acc': 'account.csv',
    'info': 'reinsinfo.csv',
    'scope': 'reinsscope.csv',
    'complex_lookup': 'analysis_settings.json',
    'oed_location_csv': 'location.csv',
    'oed_accounts_csv': 'account.csv',
    'oed_info_csv': 'reinsinfo.csv',
    'oed_scope_csv': 'reinsscope.csv',
    'lookup_config_json': 'lookup.json',
    'profile_loc_json': 'profile_location.json',
    'keys_data_csv': 'keys.csv',
    'model_version_csv': 'model_version.csv',
    'lookup_complex_config_json': 'lookup_complex.json',
    'profile_acc_json': 'profile_account.json',
    'profile_fm_agg_json': 'profile_fm_agg.json',
})

API_EXAMPLE_AUTH = OrderedDict({
    'user': 'admin',
    'pass': 'password',
})

DEFAULT_RTREE_INDEX_PROPS = {
    'buffering_capacity': 10,
    'custom_storage_callbacks': None,
    'custom_storage_callbacks_size': 0,
    'dat_extension': 'dat',
    'dimension': 2,
    'filename': '',
    'fill_factor': 0.7,
    'idx_extension': 'idx',
    'index_capacity': 100,
    'index_id': None,
    'leaf_capacity': 100,
    'near_minimum_overlap_factor': 32,
    'overwrite': True,
    'pagesize': 4096,
    'point_pool_capacity': 500,
    'region_pool_capacity': 1000,
    'reinsert_factor': 0.3,
    'split_distribution_factor': 0.4,
    'storage': 0,
    'tight_mbr': True,
    'tpr_horizon': 20.0,
    'type': 0,
    'variant': 2,
    'writethrough': False
}


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
    compressed_ext = ('.gz', '.bz2', '.zip', '.xz')
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

def get_default_json(src_fp):
    """
    Loads JSON from file.

    :param src_fp: Source JSON file path
    :type src_fp: str

    :return: dict
    :rtype: dict
    """
    try:
        with io.open(src_fp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, JSONDecodeError, OSError, TypeError):
        raise OasisException('Error trying to load JSON from {}'.format(src_fp))


def get_default_accounts_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_acc_profile.json')
    return get_default_json(src_fp=fp) if not path else fp


def get_default_exposure_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_loc_profile.json')
    return get_default_json(src_fp=fp) if not path else fp


def get_default_fm_profile_field_values(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_fm_profile_field_values.json')
    return get_default_json(src_fp=fp) if not path else fp


def get_default_step_policies_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_step_policies_profile.json')
    return get_default_json(src_fp=fp) if not path else fp


def get_config_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'config_compatibility_profile.json')
    return get_default_json(src_fp=fp) if not path else fp


def get_default_unified_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_unified_profile.json')
    return get_default_json(src_fp=fp) if not path else fp


def get_default_fm_aggregation_profile(path=False):
    fp = os.path.join(STATIC_DATA_FP, 'default_fm_agg_profile.json')
    return {int(k): v for k, v in get_default_json(src_fp=fp).items()} if not path else fp


def get_loc_dtypes():
    fp = os.path.join(STATIC_DATA_FP, 'loc_dtypes.json')
    return get_default_json(src_fp=fp)


def get_acc_dtypes():
    fp = os.path.join(STATIC_DATA_FP, 'acc_dtypes.json')
    return get_default_json(src_fp=fp)


def get_scope_dtypes():
    fp = os.path.join(STATIC_DATA_FP, 'scope_dtypes.json')
    return get_default_json(src_fp=fp)


def get_info_dtypes():
    fp = os.path.join(STATIC_DATA_FP, 'info_dtypes.json')
    return get_default_json(src_fp=fp)


def get_oed_default_values(terms):
    """
    Get defaults value of OED terms.

    :param terms: list of OED terms
    :type terms: list

    :return: default values for OED terms
    :rtype: dict
    """
    loc_defaults = {k.lower(): v['default_value'] for k, v in get_loc_dtypes().items() if k.lower() in terms}
    acc_defaults = {k.lower(): v['default_value'] for k, v in get_acc_dtypes().items() if k.lower() in terms}
    defaults = dict(
        chain.from_iterable(d.items() for d in (loc_defaults, acc_defaults))
    )

    return defaults


def assign_defaults_to_il_inputs(df):
    """
    Assign default values to IL inputs.

    :param df: IL input items dataframe
    :type df: pandas.DataFrame

    :return: IL input items dataframe
    :rtype: pandas.DataFrame
    """
    # Get default values for IL inputs
    default_fm_profile_field_values = get_default_fm_profile_field_values()

    for level in default_fm_profile_field_values.keys():
        level_id = SUPPORTED_FM_LEVELS[level]['id']
        for k, v in default_fm_profile_field_values[level].items():
            # Evaluate condition for assigning default values if present
            if v.get('condition'):
                df.loc[df.level_id == level_id, k] = df.loc[
                    df.level_id == level_id, k
                ].where(eval(
                    'df.loc[df.level_id == level_id, k]' + v['condition']),
                    v['default_value']
                )
            else:
                df.loc[df.level_id == level_id, k] = v['default_value']

    return df


WRITE_CHUNKSIZE = 2 * (10 ** 5)

GROUP_ID_COLS = ['loc_id']

CORRELATION_GROUP_ID = ['CorrelationGroup']

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
    return get_default_json(src_fp=fp) if not path else fp


# Defaults for Ktools runtime parameters
KTOOLS_NUM_PROCESSES = -1
KTOOLS_FIFO_RELATIVE = False
KTOOLS_DISABLE_ERR_GUARD = False
KTOOLS_GUL_LEGACY_STREAM = False
# ktools gul alloc rules:
# 2 = total loss is maximum subperil loss
# 1 = default with back allocation
# 0 = default without back allocation
KTOOLS_ALLOC_GUL_MAX = 2
KTOOLS_ALLOC_FM_MAX = 3
KTOOLS_ALLOC_GUL_DEFAULT = 0
KTOOLS_ALLOC_IL_DEFAULT = 2
KTOOLS_ALLOC_RI_DEFAULT = 3
KTOOLS_TIV_SAMPLE = -2
KTOOLS_MEAN_SAMPLE_IDX = -1
KTOOLS_STD_DEV_SAMPLE_IDX = -2
KTOOLS_TIV_SAMPLE_IDX = -3
KTOOL_N_GUL_PER_LB = 0
KTOOL_N_FM_PER_LB = 0

# Values for event shuffle rules
EVE_NO_SHUFFLE = 0
EVE_ROUND_ROBIN = 1
EVE_FISHER_YATES = 2
EVE_STD_SHUFFLE = 3
EVE_DEFAULT_SHUFFLE = EVE_ROUND_ROBIN

KTOOLS_DEBUG = False
