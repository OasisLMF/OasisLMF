__all__ = [
    'COVERAGE_TYPES',
    'SUPPORTED_COVERAGE_TYPES'
]

from collections import OrderedDict


COVT_BLD = 1
COVT_OTH = 2
COVT_CON = 3
COVT_BIT = 4
COVT_PDM = 5
COVT_ALL = 6


COVERAGE_TYPES = OrderedDict({
    'buildings': {'id': COVT_BLD, 'desc': 'buildings'},
    'other': {'id': COVT_OTH, 'desc': 'other (typically appurtenant structures)'},
    'contents': {'id': COVT_CON, 'desc': 'contents'},
    'bi': {'id': COVT_BIT, 'desc': 'business interruption or other time-based coverage'},
    'pd': {'id': COVT_PDM, 'desc': 'property damage (buildings + other + contents)'},
    'all': {'id': COVT_ALL, 'desc': 'all (property damage + business interruption)'}
})


SUPPORTED_COVERAGE_TYPES = OrderedDict({
    cov_type: cov_type_dict for cov_type, cov_type_dict in COVERAGE_TYPES.items()
    if cov_type in ['buildings', 'other', 'contents', 'bi']
})
