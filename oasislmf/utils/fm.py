__all__ = [
    'DEDUCTIBLE_CODES',
    'DEDUCTIBLE_AND_LIMIT_TYPES',
    'FM_LEVELS',
    'FM_TERMS',
    'LIMIT_CODES',
    'SUPPORTED_FM_LEVELS',
    'STEP_TRIGGER_TYPES',
    'COVERAGE_AGGREGATION_METHODS',
    'CALCRULE_ASSIGNMENT_METHODS'
]

from collections import OrderedDict

from .coverages import SUPPORTED_COVERAGE_TYPES


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
    if level in ['site coverage', 'site pd', 'site all', 'cond all', 'policy coverage', 'policy all', 'policy layer']
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

STEP_TRIGGER_TYPES = OrderedDict({
    1: {'coverage_aggregation_method': 1, 'calcrule_assignment_method': 1},
    2: {'coverage_aggregation_method': 1, 'calcrule_assignment_method': 2},
    3: {'coverage_aggregation_method': 2, 'calcrule_assignment_method': 3},
    5: {
        'coverage_aggregation_method': 1, 'calcrule_assignment_method': 4,
        'sub_step_trigger_types': {
            SUPPORTED_COVERAGE_TYPES['buildings']['id']: 1,
            SUPPORTED_COVERAGE_TYPES['contents']['id']: 2
        }
    }
})

COVERAGE_AGGREGATION_METHODS = OrderedDict({
    1: {
        SUPPORTED_COVERAGE_TYPES['buildings']['id']: 1,
        SUPPORTED_COVERAGE_TYPES['other']['id']: 2,
        SUPPORTED_COVERAGE_TYPES['contents']['id']: 3,
        SUPPORTED_COVERAGE_TYPES['bi']['id']: 4
    },
    2: {
        SUPPORTED_COVERAGE_TYPES['buildings']['id']: 1,
        SUPPORTED_COVERAGE_TYPES['other']['id']: 2,
        SUPPORTED_COVERAGE_TYPES['contents']['id']: 1,
        SUPPORTED_COVERAGE_TYPES['bi']['id']: 3
    }
})

CALCRULE_ASSIGNMENT_METHODS = OrderedDict({
    1: {1: True, 2: False, 3: False, 4: False},
    2: {1: False, 2: False, 3: True, 4: False},
    3: {1: True, 2: False, 3: False},
    4: {1: True, 2: False, 3: True, 4: False}
})
