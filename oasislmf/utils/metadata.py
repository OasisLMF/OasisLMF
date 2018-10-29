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

    DED_TYPE_BLN = 'B'
    DED_TYPE_MIN = 'MI'
    DED_TYPE_MAX = 'MA'

    DEDUCTIBLE_TYPES = {
        'blanket': {'id': DED_TYPE_BLN, 'desc': 'blanket'},
        'minimum': {'id': DED_TYPE_MIN, 'desc': 'minimum'},
        'maximum': {'id': DED_TYPE_MAX, 'desc': 'maximum'}
    }

    FMT_TIV = 'tiv'
    FMT_DED = 'ded'
    FMT_LIM = 'lim'
    FMT_SHR = 'shr'

    FM_TERMS = {
        'tiv': {'id': FMT_TIV, 'desc': 'TIV'},
        'deductible': {'id': FMT_DED, 'desc': 'Deductible'},
        'limit': {'id': FMT_LIM, 'desc': 'Limit'},
        'share': {'id': FMT_SHR, 'desc': 'Share'}
    }

    OASIS_COVT_BLD = 1
    OASIS_COVT_OTH = 2
    OASIS_COVT_CON = 3
    OASIS_COVT_TIM = 4

    OED_COVT_BLD = 1
    OED_COVT_OTH = 2
    OED_COVT_CON = 3
    OED_COVT_BIT = 4
    OED_COVT_PDM = 5
    OED_COVT_ALL = 6

    OED_COVERAGE_TYPES = OrderedDict({
        'buildings': {'id': 1, 'desc': 'buildings', 'eqv_oasis_covt': 'buildings'},
        'other': {'id': 2, 'desc': 'other (typically appurtenant structures)', 'eqv_oasis_covt': 'other'},
        'contents': {'id': 3, 'desc': 'contents', 'eqv_oasis_covt': 'contents'},
        'bi': {'id': 4, 'desc': 'business interruption', 'eqv_oasis_covt': 'time'},
        'pd': {'id': 5, 'desc': 'property damage (buildings + other + contents)', 'eqv_oasis_covt': None},
        'all': {'id': 6, 'desc': 'all (property damage + business interruption)', 'eqv_oasis_covt': None}
    })

    OASIS_COVERAGE_TYPES = OrderedDict({
        'buildings': {'id': 1, 'desc': 'buildings', 'eqv_oed_covt': 'buildings'},
        'other': {'id': 2, 'desc': 'other structures', 'eqv_oed_covt': 'other'},
        'contents': {'id': 3, 'desc': 'contents', 'eqv_oed_covt': 'contents'},
        'time': {'id': 4, 'desc': 'time', 'eqv_oed_covt': 'time'}
    })

    OASIS_FML_COV = 1
    OASIS_FML_COM = 2
    OASIS_FML_SIT = 3
    OASIS_FML_SUB = 4
    OASIS_FML_ACC = 5
    OASIS_FML_LAY = 6

    OED_FML_SITCOV = 1
    OED_FML_SITPDM = 2
    OED_FML_SITALL = 3
    OED_FML_CNDCOV = 4
    OED_FML_CNDPDM = 5
    OED_FML_CNDALL = 6
    OED_FML_POLCOV = 7
    OED_FML_POLPDM = 8
    OED_FML_POLALL = 9
    OED_FML_POLLAY = 10
    OED_FML_ACCCOV = 11
    OED_FML_ACCPDM = 12
    OED_FML_ACCALL = 13

    OASIS_FM_LEVELS = OrderedDict({
        'coverage': {'id': OASIS_FML_COV, 'desc': 'coverage', 'eqv_oed_fml': 'site coverage'},
        'combined': {'id': OASIS_FML_COM, 'desc': 'combined', 'eqv_oed_fml': 'site pd'},
        'site': {'id': OASIS_FML_SIT, 'desc': 'site', 'eqv_oed_fml': 'site all'},
        'sublimit': {'id': OASIS_FML_SUB, 'desc': 'sublimit', 'eqv_oed_fml': 'policy coverage'},
        'account': {'id': OASIS_FML_ACC, 'desc': 'account', 'eqv_oed_fml': 'policy all'},
        'layer': {'id': OASIS_FML_LAY, 'desc': 'layer', 'eqv_oed_fml': 'policy layer'},
    })

    OED_FM_LEVELS = OrderedDict({
        'site coverage': {'id': OED_FML_SITCOV, 'desc': 'site coverage', 'eqv_oasis_fml': 'coverage'},
        'site pd': {'id': OED_FML_SITPDM, 'desc': 'site property damage', 'eqv_oasis_fml': 'combined'},
        'site all': {'id': OED_FML_SITALL, 'desc': 'site all (coverage + property damage)', 'eqv_oasis_fml': 'site'},
        'cond coverage': {'id': OED_FML_CNDCOV, 'desc': 'conditional coverage'},
        'cond pd': {'id': OED_FML_CNDPDM, 'desc': 'conditional property damage'},
        'cond all': {'id': OED_FML_CNDALL, 'desc': 'conditional all (coverage + property damage)'},
        'policy coverage': {'id': OED_FML_POLCOV, 'desc': 'policy coverage', 'eqv_oasis_fml': 'sublimit'},
        'policy pd': {'id': OED_FML_POLPDM, 'desc': 'policy property damage'},
        'policy all': {'id': OED_FML_POLALL, 'desc': 'policy all (coverage + property damage)', 'eqv_oasis_fml': 'account'},
        'policy layer': {'id': OED_FML_POLLAY, 'desc': 'policy layer', 'eqv_oasis_fml': 'layer'},
        'account coverage': {'id': OED_FML_ACCCOV, 'desc': 'account coverage'},
        'account pd': {'id': OED_FML_ACCPDM, 'desc': 'account property damage'},
        'account all': {'id': OED_FML_ACCALL, 'desc': 'account all (coverage + property damage)'}
    })

    OASIS_PRL_WND = 1
    OASIS_PRL_SRG = 2
    OASIS_PRL_QKE = 3
    OASIS_PRL_FLD = 4

    OED_PRL_BBF = 'BBF'
    OED_PRL_BFR = 'BFR'
    OED_PRL_BSK = 'BSK'
    OED_PRL_MNT = 'MNT'
    OED_PRL_MTR = 'MTR'
    OED_PRL_ORF = 'ORF'
    OED_PRL_OSF = 'OSF'
    OED_PRL_QEQ = 'QEQ'
    OED_PRL_QFF = 'QFF'
    OED_PRL_QLF = 'QLF'
    OED_PRL_QLS = 'QLS'
    OED_PRL_QSL = 'QSL'
    OED_PRL_QTS = 'QTS'
    OED_PRL_WEC = 'WEC'
    OED_PRL_WSS = 'WSS'
    OED_PRL_WTC = 'WTC'
    OED_PRL_XHL = 'XHL'
    OED_PRL_XLT = 'XLT'
    OED_PRL_XSL = 'XSL'
    OED_PRL_XTD = 'XTD'
    OED_PRL_ZFZ = 'ZFZ'
    OED_PRL_ZIC = 'ZIC'
    OED_PRL_ZSN = 'ZSN'
    OED_PRL_ZST = 'ZST'

    OED_GRP_PRL_AA1 = 'AA1'
    OED_GRP_PRL_BB1 = 'BB1'
    OED_GRP_PRL_MM1 = 'MM1'
    OED_GRP_PRL_OO1 = 'OO1'
    OED_GRP_PRL_QQ1 = 'QQ1'
    OED_GRP_PRL_WW1 = 'WW1'
    OED_GRP_PRL_WW2 = 'WW2'
    OED_GRP_PRL_XX1 = 'XX1'
    OED_GRP_PRL_XZ1 = 'XZ1'
    OED_GRP_PRL_ZZ1 = 'ZZ1'

    OASIS_PERILS = OrderedDict({
        'wind': {'id': OASIS_PRL_WND, 'desc': 'wind', 'eqv_oed_peril': None},
        'surge': {'id': OASIS_PRL_SRG, 'desc': 'surge', 'eqv_oed_peril': None},
        'quake': {'id': OASIS_PRL_QKE, 'desc': 'quake', 'eqv_oed_peril': None},
        'flood': {'id': OASIS_PRL_FLD, 'desc': 'flood', 'eqv_oed_peril': None}
    })

    OED_GROUP_PERILS = OrderedDict({
        'AA1': {'id': OED_GRP_PRL_AA1, 'desc': 'All perils'},
        'BB1': {'id': OED_GRP_PRL_BB1, 'desc': 'Wildfire with smoke'},
        'MM1': {'id': OED_GRP_PRL_MM1, 'desc': 'Terrorism'},
        'OO1': {'id': OED_GRP_PRL_OO1, 'desc': 'Flood w/o storm surge'},
        'QQ1': {'id': OED_GRP_PRL_QQ1, 'desc': 'All EQ perils'},
        'WW1': {'id': OED_GRP_PRL_WW1, 'desc': 'Windstorm with storm surge'},
        'WW2': {'id': OED_GRP_PRL_WW2, 'desc': 'Windstorm w/o storm surge'},
        'XX1': {'id': OED_GRP_PRL_XX1, 'desc': 'Convective Storm'},
        'XZ1': {'id': OED_GRP_PRL_XZ1, 'desc': 'Convective storm (incl winter storm) - for RMS users'},
        'ZZ1': {'id': OED_GRP_PRL_ZZ1, 'desc': 'Winter storm'}
    })

    OED_PERILS = OrderedDict({
        'BBF': {'id': OED_PRL_BBF, 'desc': 'Wildfire / Bushfire', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_BB1},
        'BFR': {'id': OED_PRL_BFR, 'desc': 'NonCat', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_BB1},
        'BSK': {'id': OED_PRL_BSK, 'desc': 'Smoke', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_BB1},
        'MNT': {'id': OED_PRL_MNT, 'desc': 'NBCR Terrorism', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_MM1},
        'MTR': {'id': OED_PRL_MTR, 'desc': 'Conventional Terrorism', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_MM1},
        'ORF': {'id': OED_PRL_ORF, 'desc': 'River / Fluvial Flood', 'eqv_oasis_peril': 'flood', 'group_peril': OED_GRP_PRL_OO1},
        'OSF': {'id': OED_PRL_OSF, 'desc': 'Flash / Surface / Pluvial Flood', 'eqv_oasis_peril': 'flood', 'group_peril': OED_GRP_PRL_OO1},
        'QEQ': {'id': OED_PRL_QEQ, 'desc': 'Earthquake - Shake only', 'eqv_oasis_peril': 'quake', 'group_peril': OED_GRP_PRL_QQ1},
        'QFF': {'id': OED_PRL_QFF, 'desc': 'Fire Following', 'eqv_oasis_peril': 'quake', 'group_peril': OED_GRP_PRL_QQ1},
        'QLF': {'id': OED_PRL_QLF, 'desc': 'Liquefaction', 'eqv_oasis_peril': 'quake', 'group_peril': OED_GRP_PRL_QQ1},
        'QLS': {'id': OED_PRL_QLS, 'desc': 'Landslide', 'eqv_oasis_peril': 'quake', 'group_peril': OED_GRP_PRL_QQ1},
        'QSL': {'id': OED_PRL_QSL, 'desc': 'Sprinkler Leakage', 'eqv_oasis_peril': 'quake', 'group_peril': OED_GRP_PRL_QQ1},
        'QTS': {'id': OED_PRL_QTS, 'desc': 'Tsunami', 'eqv_oasis_peril': 'quake', 'group_peril': OED_GRP_PRL_QQ1},
        'WEC': {'id': OED_PRL_WEC, 'desc': 'Extra Tropical Cyclone', 'eqv_oasis_peril': 'flood', 'group_peril': OED_GRP_PRL_WW2},
        'WSS': {'id': OED_PRL_WSS, 'desc': 'Storm Surge', 'eqv_oasis_peril': 'surge', 'group_peril': OED_GRP_PRL_WW1},
        'WTC': {'id': OED_PRL_WTC, 'desc': 'Tropical Cyclone', 'eqv_oasis_peril': 'wind', 'group_peril': OED_GRP_PRL_WW2},
        'XHL': {'id': OED_PRL_XHL, 'desc': 'Hail', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_XZ1},
        'XLT': {'id': OED_PRL_XLT, 'desc': 'Lightning', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_XX1},
        'XSL': {'id': OED_PRL_XSL, 'desc': 'Straight-line / other convective wind', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_XX1},
        'XTD': {'id': OED_PRL_XTD, 'desc': 'Tornado', 'eqv_oasis_peril': 'wind', 'group_peril': OED_GRP_PRL_XX1},
        'ZFZ': {'id': OED_PRL_ZFZ, 'desc': 'Freeze', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_ZZ1},
        'ZIC': {'id': OED_PRL_ZIC, 'desc': 'Ice', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_ZZ1},
        'ZSN': {'id': OED_PRL_ZSN, 'desc': 'Snow', 'eqv_oasis_peril': None, 'group_peril': OED_GRP_PRL_ZZ1},
        'ZST': {'id': OED_PRL_ZST, 'desc': 'Winterstorm Wind', 'eqv_oasis_peril': 'wind', 'group_peril': OED_GRP_PRL_ZZ1}
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
