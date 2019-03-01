# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

__all__ = [
    'COVERAGE_TYPES',
    'DEDUCTIBLE_CODES',
    'DEDUCTIBLE_AND_LIMIT_TYPES',
    'FM_LEVELS',
    'FM_TERMS',
    'LIMIT_CODES',
    'OASIS_KEYS_STATUS',
    'OASIS_TASK_STATUS',
    'PERILS',
    'PERIL_GROUPS',
]

"""
Metadata definitions for Oasis and OED exposure data + insurance data + peril
& cat. modelling data + Oasis keys status and Celery task status flags

For Oasis exposure data and modelling concepts please refer to

https://github.com/OasisLMF/OasisLMF/blob/master/ModelsInOasis_V1_0.pdf

For OED exposure data and modelling concepts please refer to

https://github.com/Simplitium/OED/blob/master/Open%20Exposure%20Data.pdf
"""
import sys

from collections import OrderedDict

class _metadata(object):

    COVT_BLD = 1
    COVT_OTH = 2
    COVT_CON = 3
    COVT_BIT = 4
    COVT_PDM = 5
    COVT_ALL = 6

    COVERAGE_TYPES = OrderedDict({
        'buildings': {'id': 1, 'desc': 'buildings'},
        'other': {'id': 2, 'desc': 'other (typically appurtenant structures)'},
        'contents': {'id': 3, 'desc': 'contents'},
        'bi': {'id': 4, 'desc': 'business interruption'},
        'pd': {'id': 5, 'desc': 'property damage (buildings + other + contents)'},
        'all': {'id': 6, 'desc': 'all (property damage + business interruption)'}
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

    FMT_TIV = 'tiv'
    FMT_DED = 'ded'
    FMT_DED_MIN = 'ded_min',
    FMT_DED_MAX = 'ded_max',
    FMT_LIM = 'lim'
    FMT_SHR = 'shr'

    FM_TERMS = OrderedDict({
        'tiv': {'id': FMT_TIV, 'desc': 'TIV'},
        'deductible': {'id': FMT_DED, 'desc': 'Deductible'},
        'deductible_min': {'id': FMT_DED_MIN, 'desc': 'DeductibleMin'},
        'deductible_max': {'id': FMT_DED_MAX, 'desc': 'DeductibleMax'},
        'limit': {'id': FMT_LIM, 'desc': 'Limit'},
        'share': {'id': FMT_SHR, 'desc': 'Share'}
    })

    LIM_CODE_REG = 0
    LIM_CODE_ANAGG = 1

    LIMIT_CODES = OrderedDict({
        'reg': {'id': LIM_CODE_REG, 'desc': 'Regular'},
        'anagg': {'id': LIM_CODE_ANAGG, 'desc': 'Annual aggregate'}
    })

    OASIS_KEYS_SC = 'success'
    OASIS_KEYS_FL = 'fail'
    OASIS_KEYS_NM = 'nomatch'

    OASIS_KEYS_STATUS = {
        'success': {'id': OASIS_KEYS_SC,'desc': 'Success'},
        'fail': {'id': OASIS_KEYS_FL, 'desc': 'Failure'},
        'nomatch': {'id': OASIS_KEYS_NM, 'desc': 'No match'}
    }

    OASIS_TASK_PN = 'PENDING'
    OASIS_TASK_RN = 'RUNNING'
    OASIS_TASK_SC = 'SUCCESS'
    OASIS_TASK_FL = 'FAILURE'

    OASIS_TASK_STATUS = {
        'pending': {'id': OASIS_TASK_PN, 'desc': 'Pending'},
        'running': {'id': OASIS_TASK_RN, 'desc': 'Running'},
        'success': {'id': OASIS_TASK_SC, 'desc': 'Success'},
        'failure': {'id': OASIS_TASK_FL, 'desc': 'Failure'}
    }

    PRL_BBF = 'BBF'
    PRL_BFR = 'BFR'
    PRL_BSK = 'BSK'
    PRL_MNT = 'MNT'
    PRL_MTR = 'MTR'
    PRL_ORF = 'ORF'
    PRL_OSF = 'OSF'
    PRL_QEQ = 'QEQ'
    PRL_QFF = 'QFF'
    PRL_QLF = 'QLF'
    PRL_QLS = 'QLS'
    PRL_QSL = 'QSL'
    PRL_QTS = 'QTS'
    PRL_WEC = 'WEC'
    PRL_WSS = 'WSS'
    PRL_WTC = 'WTC'
    PRL_XHL = 'XHL'
    PRL_XLT = 'XLT'
    PRL_XSL = 'XSL'
    PRL_XTD = 'XTD'
    PRL_ZFZ = 'ZFZ'
    PRL_ZIC = 'ZIC'
    PRL_ZSN = 'ZSN'
    PRL_ZST = 'ZST'

    PRL_GRP_AA1 = 'AA1'
    PRL_GRP_BB1 = 'BB1'
    PRL_GRP_MM1 = 'MM1'
    PRL_GRP_OO1 = 'OO1'
    PRL_GRP_QQ1 = 'QQ1'
    PRL_GRP_WW1 = 'WW1'
    PRL_GRP_WW2 = 'WW2'
    PRL_GRP_XX1 = 'XX1'
    PRL_GRP_XZ1 = 'XZ1'
    PRL_GRP_ZZ1 = 'ZZ1'

    PERIL_GROUPS = OrderedDict({
        'AA1': {'id': PRL_GRP_AA1, 'desc': 'All perils'},
        'BB1': {'id': PRL_GRP_BB1, 'desc': 'Wildfire with smoke'},
        'MM1': {'id': PRL_GRP_MM1, 'desc': 'Terrorism'},
        'OO1': {'id': PRL_GRP_OO1, 'desc': 'Flood w/o storm surge'},
        'QQ1': {'id': PRL_GRP_QQ1, 'desc': 'All EQ perils'},
        'WW1': {'id': PRL_GRP_WW1, 'desc': 'Windstorm with storm surge'},
        'WW2': {'id': PRL_GRP_WW2, 'desc': 'Windstorm w/o storm surge'},
        'XX1': {'id': PRL_GRP_XX1, 'desc': 'Convective Storm'},
        'XZ1': {'id': PRL_GRP_XZ1, 'desc': 'Convective storm (incl winter storm) - for RMS users'},
        'ZZ1': {'id': PRL_GRP_ZZ1, 'desc': 'Winter storm'}
    })

    PERILS = OrderedDict({
        'BBF': {'id': PRL_BBF, 'desc': 'Wildfire / Bushfire', 'group_peril': PRL_GRP_BB1},
        'BFR': {'id': PRL_BFR, 'desc': 'NonCat', 'group_peril': PRL_GRP_BB1},
        'BSK': {'id': PRL_BSK, 'desc': 'Smoke', 'group_peril': PRL_GRP_BB1},
        'MNT': {'id': PRL_MNT, 'desc': 'NBCR Terrorism', 'group_peril': PRL_GRP_MM1},
        'MTR': {'id': PRL_MTR, 'desc': 'Conventional Terrorism', 'group_peril': PRL_GRP_MM1},
        'ORF': {'id': PRL_ORF, 'desc': 'River / Fluvial Flood', 'group_peril': PRL_GRP_OO1},
        'OSF': {'id': PRL_OSF, 'desc': 'Flash / Surface / Pluvial Flood', 'group_peril': PRL_GRP_OO1},
        'QEQ': {'id': PRL_QEQ, 'desc': 'Earthquake - Shake only', 'group_peril': PRL_GRP_QQ1},
        'QFF': {'id': PRL_QFF, 'desc': 'Fire Following', 'group_peril': PRL_GRP_QQ1},
        'QLF': {'id': PRL_QLF, 'desc': 'Liquefaction', 'group_peril': PRL_GRP_QQ1},
        'QLS': {'id': PRL_QLS, 'desc': 'Landslide', 'group_peril': PRL_GRP_QQ1},
        'QSL': {'id': PRL_QSL, 'desc': 'Sprinkler Leakage', 'group_peril': PRL_GRP_QQ1},
        'QTS': {'id': PRL_QTS, 'desc': 'Tsunami', 'group_peril': PRL_GRP_QQ1},
        'WEC': {'id': PRL_WEC, 'desc': 'Extra Tropical Cyclone', 'group_peril': PRL_GRP_WW2},
        'WSS': {'id': PRL_WSS, 'desc': 'Storm Surge', 'group_peril': PRL_GRP_WW1},
        'WTC': {'id': PRL_WTC, 'desc': 'Tropical Cyclone', 'group_peril': PRL_GRP_WW2},
        'XHL': {'id': PRL_XHL, 'desc': 'Hail', 'group_peril': PRL_GRP_XZ1},
        'XLT': {'id': PRL_XLT, 'desc': 'Lightning', 'group_peril': PRL_GRP_XX1},
        'XSL': {'id': PRL_XSL, 'desc': 'Straight-line / other convective wind', 'group_peril': PRL_GRP_XX1},
        'XTD': {'id': PRL_XTD, 'desc': 'Tornado', 'group_peril': PRL_GRP_XX1},
        'ZFZ': {'id': PRL_ZFZ, 'desc': 'Freeze', 'group_peril': PRL_GRP_ZZ1},
        'ZIC': {'id': PRL_ZIC, 'desc': 'Ice', 'group_peril': PRL_GRP_ZZ1},
        'ZSN': {'id': PRL_ZSN, 'desc': 'Snow', 'eqv_oasis_peril': None, 'group_peril': PRL_GRP_ZZ1},
        'ZST': {'id': PRL_ZST, 'desc': 'Winterstorm Wind', 'group_peril': PRL_GRP_ZZ1}
    })


    class AttributeModificationError(BaseException):
        pass

    def __setattr__(self, attrib_name, val):
        try:
            self.__class__.__dict__[attrib_name]
        except KeyError:
            raise AttributeError('Module {} has no attribute "{}"'.format(__name__, attrib_name))

        raise self.AttributeModificationError('Cannot rebind module attribute "{}.{}" - it is meant to be a constant'.format(__name__, attrib_name))

    def __delattr__(self, attrib_name):
        try:
            self.__class__.__dict__[attrib_name]
        except KeyError:
            raise AttributeError('Module {} has no attribute "{}"'.format(__name__, attrib_name))

        raise self.AttributeModificationError('Cannot delete module attribute "{}.{}" - it is meant to be a constant'.format(__name_, attrib_name))

sys.modules[__name__] = _metadata()
