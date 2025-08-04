__all__ = [
    'GENERAL_SETTINGS_FILE',
    'GUL_INPUT_FILES',
    'GUL_SUMMARIES_FILE',
    'INPUT_FILES',
    'IL_INPUT_FILES',
    'IL_SUMMARIES_FILE',
    'MODEL_SETTINGS_FILE',
    'OPTIONAL_INPUT_FILES',
    'TAR_FILE'
]

INPUT_FILES = {
    'complex_items': {
        'name': 'complex_items',
        'type': 'gul',
        'conversion_tool': 'complex_itemtobin',
        'csvtobin_type': 'complex_items',
    },
    'items': {
        'name': 'items',
        'type': 'gul',
        'conversion_tool': 'itemtobin',
        'csvtobin_type': 'items',
    },
    'coverages': {
        'name': 'coverages',
        'type': 'gul',
        'conversion_tool': 'coveragetobin',
        'csvtobin_type': 'coverages',
    },
    'gulsummaryxref': {
        'name': 'gulsummaryxref',
        'type': 'gul',
        'conversion_tool': 'gulsummaryxreftobin',
        'csvtobin_type': 'gul_summary_xref',
    },
    'events': {
        'name': 'events',
        'type': 'optional',
        'conversion_tool': 'evetobin',
        'csvtobin_type': 'eve',
    },
    'amplifications': {
        'name': 'amplifications',
        'type': 'optional',
        'conversion_tool': 'amplificationstobin',
        'csvtobin_type': 'amplifications',
    },
    'fm_policytc': {
        'name': 'fm_policytc',
        'type': 'il',
        'conversion_tool': 'fmpolicytctobin',
        'csvtobin_type': 'fm_policytc',
    },
    'fm_profile': {
        'name': 'fm_profile',
        'type': 'il',
        'conversion_tool': 'fmprofiletobin',
        'step_flag': '-S',
        'csvtobin_type': 'fm_profile',
    },
    'fm_programme': {
        'name': 'fm_programme',
        'type': 'il',
        'conversion_tool': 'fmprogrammetobin',
        'csvtobin_type': 'fm_programme',
    },
    'fm_xref': {
        'name': 'fm_xref',
        'type': 'il',
        'conversion_tool': 'fmxreftobin',
        'csvtobin_type': 'fm_xref',
    },
    'fmsummaryxref': {
        'name': 'fmsummaryxref',
        'type': 'il',
        'conversion_tool': 'fmsummaryxreftobin',
        'csvtobin_type': 'fm_summary_xref',
    }
}
GUL_INPUT_FILES = {k: v for k, v in INPUT_FILES.items() if v['type'] == 'gul'}
IL_INPUT_FILES = {k: v for k, v in INPUT_FILES.items() if v['type'] == 'il'}
OPTIONAL_INPUT_FILES = {k: v for k, v in INPUT_FILES.items() if v['type'] == 'optional'}

TAR_FILE = 'inputs.tar.gz'

GENERAL_SETTINGS_FILE = "general_settings.csv"
MODEL_SETTINGS_FILE = "model_settings.csv"
GUL_SUMMARIES_FILE = "gul_summaries.csv"
IL_SUMMARIES_FILE = "il_summaries.csv"
