from __future__ import unicode_literals

import six

INPUT_FILES = {
    'items': {'name': 'items', 'type': 'gul', 'conversion_tool': 'itemtobin'},
    'coverages': {'name': 'coverages', 'type': 'gul', 'conversion_tool': 'coveragetobin'},
    'gulsummaryxref': {'name': 'gulsummaryxref', 'type': 'gul', 'conversion_tool': 'gulsummaryxreftobin'},
    'events': {'name': 'events', 'type': 'optional', 'conversion_tool': 'evetobin'},
    'fm_policytc': {'name': 'fm_policytc', 'type': 'il', 'conversion_tool': 'fmpolicytctobin'},
    'fm_profile': {'name': 'fm_profile', 'type': 'il', 'conversion_tool': 'fmprofiletobin'},
    'fm_programme': {'name': 'fm_programme', 'type': 'il', 'conversion_tool': 'fmprogrammetobin'},
    'fm_xref': {'name': 'fm_xref', 'type': 'il', 'conversion_tool': 'fmxreftobin'},
    'fmsummaryxref': {'name': 'fmsummaryxref', 'type': 'il', 'conversion_tool': 'fmsummaryxreftobin'}
}
GUL_INPUT_FILES = {k: v for k, v in six.iteritems(INPUT_FILES) if v['type'] == 'gul'}
IL_INPUT_FILES = {k: v for k, v in six.iteritems(INPUT_FILES) if v['type'] == 'il'}
OPTIONAL_INPUT_FILES = {k: v for k, v in six.iteritems(INPUT_FILES) if v['type'] == 'optional'}

TAR_FILE = 'inputs.tar.gz'

GENERAL_SETTINGS_FILE = "general_settings.csv"
MODEL_SETTINGS_FILE = "model_settings.csv"
GUL_SUMMARIES_FILE = "gul_summaries.csv"
IL_SUMMARIES_FILE = "il_summaries.csv"
