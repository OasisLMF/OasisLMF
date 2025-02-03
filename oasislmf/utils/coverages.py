__all__ = [
    'COVERAGE_TYPES',
    'SUPPORTED_COVERAGE_TYPES'
]

from collections import OrderedDict


# Property
COVT_BLD = 1
COVT_OTH = 2
COVT_CON = 3
COVT_BIT = 4
COVT_PDM = 5
COVT_ALL = 6
# Cyber
COVT_BINP = 7
COVT_CBI = 8
COVT_DIAS = 9
COVT_EXT = 10
COVT_FIN = 11
COVT_INRE = 12
COVT_LIAB = 13
COVT_REG = 14
COVT_ENO = 15
COVT_CYB = 16


COVERAGE_TYPES = OrderedDict({
    'buildings': {'id': COVT_BLD, 'desc': 'buildings'},
    'other': {'id': COVT_OTH, 'desc': 'other (typically appurtenant structures)'},
    'contents': {'id': COVT_CON, 'desc': 'contents'},
    'bi': {'id': COVT_BIT, 'desc': 'business interruption or other time-based coverage'},
    'pd': {'id': COVT_PDM, 'desc': 'property damage (buildings + other + contents)'},
    'all': {'id': COVT_ALL, 'desc': 'all (property damage + business interruption)'},
    'binp': {'id': COVT_BINP, 'desc': 'Business Interruption (non-property)'},
    'cbi': {'id': COVT_CBI, 'desc': 'Contingent Business Interruption'},
    'dias': {'id': COVT_DIAS, 'desc': 'Digital Assets (data, software and/or hardware recovery/replacement)'},
    'ext': {'id': COVT_EXT, 'desc': 'Extortion  (ransom amounts, negotiation fees, reward payments)'},
    'fin': {'id': COVT_FIN, 'desc': 'Financial Fraud - Theft of electronic funds, goods, services'},
    'inre': {'id': COVT_INRE, 'desc': 'Incident Response (IT Forensics, response obligations assessment, public relations, notifications and customer relations, monitoring services)'},
    'liab': {'id': COVT_LIAB, 'desc': 'Liability - 3rd party cyber liability (e.g. Network security, privacy, media) and directly associated costs (e.g legal defence costs)'},
    'reg': {'id': COVT_REG, 'desc': 'Regulatory and payment card industry (PCI) fines and proceedings'},
    'eno': {'id': COVT_ENO, 'desc': 'Errors & Omissions/Professional indemnity liability'},
    'cyb': {'id': COVT_CYB, 'desc': 'All Cyber coverages'}
})


SUPPORTED_COVERAGE_TYPES = OrderedDict({
    cov_type: cov_type_dict for cov_type, cov_type_dict in COVERAGE_TYPES.items()
    if cov_type in ['buildings', 'other', 'contents', 'bi', 'binp', 'cbi', 'dias', 'ext', 'fin', 'inre', 'liab', 'reg', 'eno']
})
