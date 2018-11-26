from __future__ import unicode_literals

__all__ = [
    'calcrule_ids',
    'canonical_accounts_data',
    'canonical_accounts_profile',
    'canonical_exposures_data',
    'canonical_exposures_profile',
    'canonical_oed_accounts_data',
    'canonical_oed_accounts_profile',
    'canonical_oed_exposures_data',
    'canonical_oed_exposures_profile',
    'coverage_type_ids',
    'deductible_types',
    'fm_items_data',
    'fm_level_names',
    'fm_level_names_simple',
    'fm_levels',
    'fm_levels_simple',
    'gul_items_data',
    'keys_data',
    'keys_status_flags',
    'oasis_fm_agg_profile',
    'oasis_tiv_elements',
    'oed_fm_agg_profile',
    'oed_tiv_elements',
    'peril_ids',
    'write_canonical_files',
    'write_canonical_oed_files',
    'write_keys_files'
]

import copy
import itertools
import six
import string
import six

from itertools import chain
from chainmap import ChainMap
from collections import OrderedDict

import pandas as pd

from hypothesis import given
from hypothesis.strategies import (
    booleans,
    fixed_dictionaries,
    integers,
    just,
    lists,
    nothing,
    sampled_from,
    text,
    tuples,
    floats,
)

from oasislmf.utils.metadata import (
    DEDUCTIBLE_TYPES,
    FM_TERMS,
    OASIS_COVERAGE_TYPES,
    OASIS_FM_LEVELS,
    OASIS_KEYS_STATUS,
    OASIS_PERILS,
    OED_COVERAGE_TYPES,
    OED_FM_LEVELS,
    OED_PERILS,
)

from oasislmf.model_execution.files import (
    GUL_INPUT_FILES,
    IL_INPUT_FILES,
    OPTIONAL_INPUT_FILES,
    TAR_FILE, INPUT_FILES,
)

calcrule_ids = (1, 4, 5, 6, 7, 8, 10, 11, 12, 12, 13, 14, 15, 16, 19, 21,)

canonical_accounts_profile = {
    "BLANDEDAMT": {
        "ProfileElementName": "BLANDEDAMT",
        "DeductibleType": "B",
        "FieldName": "BlanketDeductible",
        "FMLevelName": "Account",
        "FMLevel": 5,
        "FMTermType": "Deductible",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "MINDEDAMT": {
        "ProfileElementName": "MINDEDAMT",
        "DeductibleType": "MI",
        "FieldName": "BlanketMinDeductible",
        "FMLevelName": "Account",
        "FMLevel": 5,
        "FMTermType": "DeductibleMin",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "MAXDEDAMT": {
        "ProfileElementName": "MAXDEDAMT",
        "DeductibleType": "MA",
        "FieldName": "BlanketMaxDeductible",
        "FMLevelName": "Account",
        "FMLevel": 5,
        "FMTermType": "DeductibleMax",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "UNDCOVAMT": {
        "ProfileElementName": "UNDCOVAMT",
        "DeductibleType": "B",
        "FieldName": "AttachmentPoint",
        "FMLevelName": "Layer",
        "FMLevel": 6,
        "FMTermType": "Deductible",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "PARTOF": {
        "ProfileElementName": "PARTOF",
        "FieldName": "LayerLimit",
        "FMLevelName": "Layer",
        "FMLevel": 6,
        "FMTermType": "Limit",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "BLANLIMAMT": {
        "ProfileElementName": "BLANLIMAMT",
        "FieldName": "BlanketLimit",
        "FMLevelName": "Layer",
        "FMLevel": 6,
        "FMTermType": "Share",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    }
}


canonical_exposures_profile = {
    'COND1DEDUCTIBLE': {
        'DeductibleType': 'B',
        'FMLevel': 4,
        'FMLevelName': 'Sublimit',
        'FMTermGroupID': 1,
        'FMTermType': 'Deductible',
        'FieldName': 'SubLimitDeductible',
        'ProfileElementName': 'COND1DEDUCTIBLE',
        'ProfileType': 'Loc'
    },
    'COND1LIMIT': {
        'FMLevel': 4,
        'FMLevelName': 'Sublimit',
        'FMTermGroupID': 1,
        'FMTermType': 'Limit',
        'FieldName': 'SubLimitLimit',
        'ProfileElementName': 'COND1LIMIT',
        'ProfileType': 'Loc'
    },
    'WSCOMBINEDDED': {
        'DeductibleType': 'B',
        'FMLevel': 2,
        'FMLevelName': 'Combined',
        'FMTermGroupID': 1,
        'FMTermType': 'Deductible',
        'FieldName': 'CombinedDeductible',
        'ProfileElementName': 'WSCOMBINEDDED',
        'ProfileType': 'Loc'
    },
    'WSCOMBINEDLIM': {
        'FMLevel': 2,
        'FMLevelName': 'Combined',
        'FMTermGroupID': 1,
        'FMTermType': 'Limit',
        'FieldName': 'CombinedLimit',
        'ProfileElementName': 'WSCOMBINEDLIM',
        'ProfileType': 'Loc'
    },
    'WSCV1DED': {
        'CoverageTypeID': 1,
        'DeductibleType': 'B',
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 1,
        'FMTermType': 'Deductible',
        'FieldName': 'BuildingsDeductible',
        'ProfileElementName': 'WSCV1DED',
        'ProfileType': 'Loc'
    },
    'WSCV1LIMIT': {
        'CoverageTypeID': 1,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 1,
        'FMTermType': 'Limit',
        'FieldName': 'BuildingsLimit',
        'ProfileElementName': 'WSCV1LIMIT',
        'ProfileType': 'Loc'
    },
    'WSCV1VAL': {
        'CoverageTypeID': 1,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 1,
        'FMTermType': 'TIV',
        'FieldName': 'BuildingsTIV',
        'ProfileElementName': 'WSCV1VAL',
        'ProfileType': 'Loc'
    },
    'WSCV2DED': {
        'CoverageTypeID': 2,
        'DeductibleType': 'B',
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 2,
        'FMTermType': 'Deductible',
        'FieldName': 'OtherDeductible',
        'ProfileElementName': 'WSCV2DED',
        'ProfileType': 'Loc'
    },
    'WSCV2LIMIT': {
        'CoverageTypeID': 2,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 2,
        'FMTermType': 'Limit',
        'FieldName': 'OtherLimit',
        'ProfileElementName': 'WSCV2LIMIT',
        'ProfileType': 'Loc'
    },
    'WSCV2VAL': {
        'CoverageTypeID': 2,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 2,
        'FMTermType': 'TIV',
        'FieldName': 'OtherTIV',
        'ProfileElementName': 'WSCV2VAL',
        'ProfileType': 'Loc'
    },
    'WSCV3DED': {
        'CoverageTypeID': 3,
        'DeductibleType': 'B',
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 3,
        'FMTermType': 'Deductible',
        'FieldName': 'ContentsDeductible',
        'ProfileElementName': 'WSCV3DED',
        'ProfileType': 'Loc'
    },
    'WSCV3LIMIT': {
        'CoverageTypeID': 3,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 3,
        'FMTermType': 'Limit',
        'FieldName': 'ContentsLimit',
        'ProfileElementName': 'WSCV3LIMIT',
        'ProfileType': 'Loc'
    },
    'WSCV3VAL': {
        'CoverageTypeID': 3,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 3,
        'FMTermType': 'TIV',
        'FieldName': 'ContentsTIV',
        'ProfileElementName': 'WSCV3VAL',
        'ProfileType': 'Loc'
    },
    'WSCV4DED': {
        'CoverageTypeID': 4,
        'DeductibleType': 'B',
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 4,
        'FMTermType': 'Deductible',
        'FieldName': 'TimeDeductible',
        'ProfileElementName': 'WSCV4DED',
        'ProfileType': 'Loc'
    },
    'WSCV4LIMIT': {
        'CoverageTypeID': 4,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 4,
        'FMTermType': 'Limit',
        'FieldName': 'TimeLimit',
        'ProfileElementName': 'WSCV4LIMIT',
        'ProfileType': 'Loc'
    },
    'WSCV4VAL': {
        'CoverageTypeID': 4,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 4,
        'FMTermType': 'TIV',
        'FieldName': 'TimeTIV',
        'ProfileElementName': 'WSCV4VAL',
        'ProfileType': 'Loc'
    },
    'WSSITEDED': {
        'DeductibleType': 'B',
        'FMLevel': 3,
        'FMLevelName': 'Site',
        'FMTermGroupID': 1,
        'FMTermType': 'Deductible',
        'FieldName': 'SiteDeductible',
        'ProfileElementName': 'WSSITEDED',
        'ProfileType': 'Loc'
    },
    'WSSITELIM': {
        'FMLevel': 3,
        'FMLevelName': 'Site',
        'FMTermGroupID': 1,
        'FMTermType': 'Limit',
        'FieldName': 'SiteLimit',
        'ProfileElementName': 'WSSITELIM',
        'ProfileType': 'Loc'
    }
}

canonical_exposures_profile_simple = {
    'WSCV1DED': {
        'CoverageTypeID': 1,
        'DeductibleType': 'B',
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 1,
        'FMTermType': 'Deductible',
        'FieldName': 'CoverageDeductible',
        'PerilID': 1,
        'ProfileElementName': 'WSCV1DED',
        'ProfileType': 'Loc'
    },
    'WSCV1LIMIT': {
        'CoverageTypeID': 1,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 1,
        'FMTermType': 'Limit',
        'FieldName': 'CoverageLimit',
        'PerilID': 1,
        'ProfileElementName': 'WSCV1LIMIT',
        'ProfileType': 'Loc'
    },
    'WSCV1VAL': {
        'CoverageTypeID': 1,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 1,
        'FMTermType': 'TIV',
        'FieldName': 'TIV',
        'PerilID': 1,
        'ProfileElementName': 'WSCV1VAL',
        'ProfileType': 'Loc'
    }
}

canonical_oed_exposures_profile = {
    "BuildingTIV": {
        "ProfileElementName": "BuildingTIV",
        "FieldName": "TIV",
        "PerilID": 1,
        "CoverageTypeID": 1,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "TIV",
        "FMTermGroupID": 1,
        "ProfileType": "Loc"
    },
    "LocDed1Building": {
        "ProfileElementName": "LocDed1Building",
        "FieldName": "CoverageDeductible",
        "DeductibleType": "B",
        "PerilID": 1,
        "CoverageTypeID": 1,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "Deductible",
        "FMTermGroupID": 1,
        "ProfileType": "Loc"
    },
    "LocLimit1Building": {
        "ProfileElementName": "LocLimit1Building",
        "FieldName": "CoverageLimit",
        "PerilID": 1,
        "CoverageTypeID": 1,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "Limit",
        "FMTermGroupID": 1,
        "ProfileType": "Loc"
    },
    "OtherTIV": {
        "ProfileElementName": "OtherTIV",
        "FieldName": "TIV",
        "PerilID": 1,
        "CoverageTypeID": 2,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "TIV",
        "FMTermGroupID": 2,
        "ProfileType": "Loc"
    },
    "LocDed2Other": {
        "ProfileElementName": "LocDed2Other",
        "FieldName": "CoverageDeductible",
        "DeductibleType": "B",
        "PerilID": 1,
        "CoverageTypeID": 2,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "Deductible",
        "FMTermGroupID": 2,
        "ProfileType": "Loc"
    },
    "LocLimit2Other": {
        "ProfileElementName": "LocLimit2Other",
        "FieldName": "CoverageLimit",
        "PerilID": 1,
        "CoverageTypeID": 2,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "Limit",
        "FMTermGroupID": 2,
        "ProfileType": "Loc"
    },
    "ContentsTIV": {
        "ProfileElementName": "ContentsTIV",
        "FieldName": "TIV",
        "PerilID": 1,
        "CoverageTypeID": 3,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "TIV",
        "FMTermGroupID": 3,
        "ProfileType": "Loc"
    },
    "LocDed3Contents": {
        "ProfileElementName": "LocDed3Contents",
        "FieldName": "CoverageDeductible",
        "DeductibleType": "B",
        "PerilID": 1,
        "CoverageTypeID": 3,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "Deductible",
        "FMTermGroupID": 3,
        "ProfileType": "Loc"
    },
    "LocLimit3Contents": {
        "ProfileElementName": "LocLimit3Contents",
        "FieldName": "CoverageLimit",
        "PerilID": 1,
        "CoverageTypeID": 3,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "Limit",
        "FMTermGroupID": 3,
        "ProfileType": "Loc"
    },
    "BITIV": {
        "ProfileElementName": "BITIV",
        "FieldName": "TIV",
        "PerilID": 1,
        "CoverageTypeID": 4,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "TIV",
        "FMTermGroupID": 4,
        "ProfileType": "Loc"
    },
    "LocDed4BI": {
        "ProfileElementName": "LocDed4BI",
        "FieldName": "CoverageDeductible",
        "DeductibleType": "B",
        "PerilID": 1,
        "CoverageTypeID": 4,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "Deductible",
        "FMTermGroupID": 4,
        "ProfileType": "Loc"
    },
    "LocLimit4BI": {
        "ProfileElementName": "LocLimit4BI",
        "FieldName": "CoverageLimit",
        "PerilID": 1,
        "CoverageTypeID": 4,
        "FMLevelName": "SiteCoverage",
        "FMLevel": 1,
        "FMTermType": "Limit",
        "FMTermGroupID": 4,
        "ProfileType": "Loc"
    },
    "LocDed5PD": {
        "ProfileElementName": "LocDed5PD",
        "FieldName": "SitePDDeductible",
        "DeductibleType": "B",
        "FMLevelName": "SitePD",
        "FMLevel": 2,
        "FMTermType": "Deductible",
        "FMTermGroupID": 1,
        "CoverageTypeID": [1,2,3],
        "ProfileType": "Loc"
    },
    "LocLimit5PD": {
        "ProfileElementName": "LocLimit5PD",
        "FieldName": "SitePDLimit",
        "FMLevelName": "SitePD",
        "FMLevel": 2,
        "FMTermType": "Limit",
        "FMTermGroupID": 1,
        "CoverageTypeID": [1,2,3],
        "ProfileType": "Loc"
    },
    "LocDed6All": {
        "ProfileElementName": "LocDed6All",
        "FieldName": "SiteAllDeductible",
        "DeductibleType": "B",
        "FMLevelName": "SiteAll",
        "FMLevel": 3,
        "FMTermType": "Deductible",
        "FMTermGroupID": 1,
        "ProfileType": "Loc"
    },
    "LocLimit6All": {
        "ProfileElementName": "LocLimit6All",
        "FieldName": "SiteAllLimit",
        "FMLevelName": "SiteAll",
        "FMLevel": 3,
        "FMTermType": "Limit",
        "FMTermGroupID": 1,
        "ProfileType": "Loc"
    }
}

canonical_oed_accounts_profile = {
    "CondDed6All": {
        "ProfileElementName": "CondDed6All",
        "FieldName": "CondAllDeductible",
        "DeductibleType": "B",
        "FMLevelName": "CondAll",
        "FMLevel": 6,
        "FMTermType": "Deductible",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "CondLimit6All": {
        "ProfileElementName": "CondLimit6All",
        "FieldName": "CondAllLimit",
        "FMLevelName": "CondAll",
        "FMLevel": 6,
        "FMTermType": "Limit",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "PolDed6All": {
        "ProfileElementName": "PolDed6All",
        "DeductibleType": "B",
        "FieldName": "BlanketDeductible",
        "FMLevelName": "PolAll",
        "FMLevel": 9,
        "FMTermType": "Deductible",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "PolMinDed6All": {
        "ProfileElementName": "PolMinDed6All",
        "DeductibleType": "MI",
        "FieldName": "BlanketMinDeductible",
        "FMLevelName": "PolAll",
        "FMLevel": 9,
        "FMTermType": "DeductibleMin",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "PolMaxDed6All": {
        "ProfileElementName": "PolMaxDed6All",
        "DeductibleType": "MA",
        "FieldName": "BlanketMaxDeductible",
        "FMLevelName": "PolAll",
        "FMLevel": 9,
        "FMTermType": "DeductibleMax",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "LayerAttachment": {
        "ProfileElementName": "LayerAttachment",
        "DeductibleType": "B",
        "FieldName": "AttachmentPoint",
        "FMLevelName": "PolLayer",
        "FMLevel": 10,
        "FMTermType": "Deductible",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "LayerLimit": {
        "ProfileElementName": "LayerLimit",
        "FieldName": "LayerLimit",
        "FMLevelName": "PolLayer",
        "FMLevel": 10,
        "FMTermType": "Limit",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    },
    "LayerParticipation": {
        "ProfileElementName": "LayerParticipation",
        "FieldName": "LayerParticipation",
        "FMLevelName": "PolLayer",
        "FMLevel": 10,
        "FMTermType": "Share",
        "FMTermGroupID": 1,
        "ProfileType": "Acc"
    }
}

coverage_type_ids = tuple(OASIS_COVERAGE_TYPES[k]['id'] for k in OASIS_COVERAGE_TYPES)

deductible_types = tuple(DEDUCTIBLE_TYPES[k]['id'] for k in DEDUCTIBLE_TYPES)

oasis_fm_agg_profile = {
    1: {
        "FMLevel": 1,
        "FMLevelName": "Coverage",
        "FMAggKey": {
            "ItemID": {
                "src": "FM",
                "field": "item_id",
                "name": "Item ID"
            }
        }
    },
    2: {
        "FMLevel": 2,
        "FMLevelName": "Combined",
        "FMAggKey": {
            "LocID": {
                "src": "FM",
                "field": "canexp_id",
                "name": "Location ID"
            },
            "IsBICoverage": {
                "src": "FM",
                "field": "is_bi_coverage",
                "name": "Is BI coverage?"
            }
        }
    },
    3: {
        "FMLevel": 3,
        "FMLevel": "Site",
        "FMAggKey": {
            "LocID":  {
                "src": "FM",
                "field": "canexp_id",
                "name": "Location ID"
            }
        }
    },
    4: {
        "FMLevel": 4,
        "FMLevelName": "Sublimit",
        "FMAggKey": {
            "AccntNum":  {
                "src": "FM",
                "field": "canacc_id",
                "name": "Account no."
            },
            "SublimitRef": {
                "src": "CanExp",
                "field": "cond1name",
                "name": "Sublimit ref."
            }
        }
    },
    5: {
        "FMLevel": 5,
        "FMLevelName": "Account",
        "FMAggKey": {
            "AccntNum": {
                "src": "FM",
                "field": "canacc_id",
                "name": "Account no."
            }
        }
    },
    6: {
        "FMLevel": 6,
        "FMLevelName": "Layer",
        "FMAggKey": {
            "AccntNum": {
                "src": "FM",
                "field": "canacc_id",
                "name": "Account no."
            },
            "PolicyNum": {
                "src": "FM",
                "field": "policy_num",
                "name": "Account policy no."
            }
        }
    }
}

oed_fm_agg_profile = {
    1: {
        "FMLevel": 1,
        "FMLevelName": "SiteCoverage",
        "FMAggKey": {
            "ItemID": {
                "src": "FM",
                "field": "item_id",
                "name": "Item ID"
            }
        }
    },
    2: {
        "FMLevel": 2,
        "FMLevelName": "SitePD",
        "FMAggKey": {
            "LocID": {
                "src": "FM",
                "field": "canexp_id",
                "name": "Location ID"
            },
            "IsBICoverage": {
                "src": "FM",
                "field": "is_bi_coverage",
                "name": "Is BI coverage?"
            }
        }
    },
    3: {
        "FMLevel": 3,
        "FMLevel": "SiteAll",
        "FMAggKey": {
            "LocID":  {
                "src": "FM",
                "field": "canexp_id",
                "name": "Location ID"
            }
        }
    },
    6: {
        "FMLevel": 6,
        "FMLevelName": "CondAll",
        "FMAggKey": {
            "AccntNum":  {
                "src": "FM",
                "field": "canacc_id",
                "name": "Account no."
            }
        }
    },
    9: {
        "FMLevel": 9,
        "FMLevelName": "PolAll",
        "FMAggKey": {
            "AccntNum": {
                "src": "FM",
                "field": "canacc_id",
                "name": "Account no."
            }
        }
    },
    10: {
        "FMLevel": 10,
        "FMLevelName": "PolLayer",
        "FMAggKey": {
            "AccntNum": {
                "src": "FM",
                "field": "canacc_id",
                "name": "Account no."
            },
            "PolicyNum": {
                "src": "FM",
                "field": "policy_num",
                "name": "Account policy no."
            }
        }
    }
}

fm_levels = tuple(OASIS_FM_LEVELS[k]['id'] for k in OASIS_FM_LEVELS)

fm_level_names = tuple(k.capitalize() for k in OASIS_FM_LEVELS)

fm_levels_simple = tuple(
    t for t in set(
        t.get('FMLevel')
        for t in itertools.chain(six.itervalues(canonical_exposures_profile_simple), six.itervalues(canonical_accounts_profile))
    ) if t
)

fm_level_names_simple = tuple(
    t[0] for t in sorted([t for t in set(
        (t.get('FMLevelName'), t.get('FMLevel'))
        for t in itertools.chain(six.itervalues(canonical_exposures_profile_simple), six.itervalues(canonical_accounts_profile))
    ) if t != (None,None)], key=lambda t: t[1])
)

fm_term_types = tuple(FM_TERMS[k]['desc'] for k in FM_TERMS)

fm_profile_types = ('acc', 'loc',)

keys_status_flags = tuple(OASIS_KEYS_STATUS[k]['id'] for k in OASIS_KEYS_STATUS)

peril_ids = tuple(OASIS_PERILS[k]['id'] for k in OASIS_PERILS)
oed_peril_ids = tuple(OED_PERILS[k]['id'] for k in OED_PERILS)

# Used simple echo command rather than ktools conversion utility for testing purposes
ECHO_CONVERSION_INPUT_FILES = {k: ChainMap({'conversion_tool': 'echo'}, v) for k, v in INPUT_FILES.items()}

def standard_input_files(min_size=0):
    return lists(
        sampled_from([target['name'] for target in chain(six.itervalues(GUL_INPUT_FILES), six.itervalues(OPTIONAL_INPUT_FILES))]),
        min_size=min_size,
        unique=True,
    )


def il_input_files(min_size=0):
    return lists(
        sampled_from([target['name'] for target in six.itervalues(IL_INPUT_FILES)]),
        min_size=min_size,
        unique=True,
    )


def tar_file_targets(min_size=0):
    return lists(
        sampled_from([target['name'] + '.bin' for target in six.itervalues(INPUT_FILES)]),
        min_size=min_size,
        unique=True,
    )

oasis_tiv_elements = tuple(v['ProfileElementName'].lower() for v in canonical_exposures_profile.values() if v.get('FMTermType') and v.get('FMTermType').lower() == 'tiv')

oed_tiv_elements = tuple(v['ProfileElementName'].lower() for v in canonical_oed_exposures_profile.values() if v.get('FMTermType') and v.get('FMTermType').lower() == 'tiv')

def canonical_accounts_data(
    from_account_nums=integers(min_value=1, max_value=10**5),
    from_policy_nums=text(alphabet=string.ascii_letters, min_size=2, max_size=10),
    from_policy_types=integers(min_value=1, max_value=10),
    from_account_deductibles=floats(min_value=0.0, max_value=10**6),
    from_account_min_deductibles=floats(min_value=0.0, max_value=10**6),
    from_account_max_deductibles=floats(min_value=0.0, max_value=10**6),
    from_account_limits=floats(min_value=0.0, max_value=10**6),
    from_layer_deductibles=floats(min_value=0.0, max_value=10**6),
    from_layer_limits=floats(min_value=0.0, max_value=10**6),
    from_layer_shares=floats(min_value=0.0, max_value=10**6),
    size=None,
    min_size=0,
    max_size=10
):
    def _sequence(li):
        for i, r in enumerate(li):
            r['row_id'] = i + 1

        return li

    return lists(
        fixed_dictionaries(
            {
                'accntnum': from_account_nums,
                'policynum': from_policy_nums,
                'policytype': from_policy_types,
                'undcovamt': from_layer_deductibles,
                'partof': from_layer_limits,
                'blandedamt': from_account_deductibles,
                'mindedamt': from_account_min_deductibles,
                'maxdedamt': from_account_max_deductibles,
                'blanlimamt': from_account_limits
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())

def canonical_oed_accounts_data(
    from_account_nums=integers(min_value=1, max_value=10**6),
    from_portfolio_nums=integers(min_value=1, max_value=10**6),
    from_policy_nums=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_policy_perils=sampled_from(oed_peril_ids),
    from_sublimit_deductibles=floats(min_value=0.0, max_value=10**6),
    from_sublimit_limits=floats(min_value=0.0, max_value=10**6),
    from_account_deductibles=floats(min_value=0.0, max_value=10**6),
    from_account_min_deductibles=floats(min_value=0.0, max_value=10**6),
    from_account_max_deductibles=floats(min_value=0.0, max_value=10**6),
    from_layer_deductibles=floats(min_value=0.0, max_value=10**6),
    from_layer_limits=floats(min_value=0.0, max_value=10**6),
    from_layer_shares=floats(min_value=0.0, max_value=10**6),
    size=None,
    min_size=0,
    max_size=10
):
    def _sequence(li):
        for i, r in enumerate(li):
            r['row_id'] = i + 1

        return li

    return lists(
        fixed_dictionaries(
            {
                'accnumber': from_account_nums,
                'portnumber': from_portfolio_nums,
                'polnumber': from_policy_nums,
                'polperil': from_policy_perils,
                'condded6all': from_sublimit_deductibles,
                'condlimit6all': from_sublimit_limits,
                'polded6all': from_account_deductibles,
                'polminded6all': from_account_min_deductibles,
                'polmaxded6all': from_account_max_deductibles,
                'layerattachment': from_layer_deductibles,
                'layerlimit': from_layer_limits,
                'layerparticipation': from_layer_shares
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())

def canonical_exposures_data(
    from_account_nums=integers(min_value=1, max_value=10**6),
    from_building_classes=integers(min_value=1, max_value=3),
    from_building_schemes=text(alphabet=string.ascii_letters, min_size=1, max_size=3),
    from_cities=text(alphabet=string.ascii_letters, min_size=2, max_size=20),
    from_countries=text(alphabet=string.ascii_letters, min_size=2, max_size=20),
    from_cresta_ids=text(alphabet=string.ascii_letters, min_size=1, max_size=10),
    from_deductibles1=floats(min_value=0.0, allow_infinity=False),
    from_deductibles2=floats(min_value=0.0, allow_infinity=False),
    from_deductibles3=floats(min_value=0.0, allow_infinity=False),
    from_deductibles4=floats(min_value=0.0, allow_infinity=False),
    from_latitudes=floats(min_value=0.0, max_value=90.0),
    from_limits1=floats(min_value=0.0, allow_infinity=False),
    from_limits2=floats(min_value=0.0, allow_infinity=False),
    from_limits3=floats(min_value=0.0, allow_infinity=False),
    from_limits4=floats(min_value=0.0, allow_infinity=False),
    from_location_nums=integers(min_value=1, max_value=10**12),
    from_longitudes=floats(min_value=-180.0, max_value=180.0),
    from_num_builings=integers(min_value=1, max_value=10),
    from_num_stories=integers(min_value=1, max_value=10),
    from_occ_schemes=text(alphabet=string.ascii_letters, min_size=1, max_size=3),
    from_occ_types=text(alphabet=string.ascii_letters, min_size=1, max_size=3),
    from_postal_codes=text(alphabet=string.ascii_letters, min_size=6, max_size=8),
    from_states=text(alphabet=string.ascii_letters, min_size=2, max_size=20),
    from_tivs1=floats(min_value=0.0, allow_infinity=False),
    from_tivs2=floats(min_value=0.0, allow_infinity=False),
    from_tivs3=floats(min_value=0.0, allow_infinity=False),
    from_tivs4=floats(min_value=0.0, allow_infinity=False),
    from_years_built=integers(min_value=1900, max_value=2018),
    from_years_upgraded=integers(min_value=1900, max_value=2018),
    size=None,
    min_size=0,
    max_size=10
):
    def _sequence(li):
        for i, r in enumerate(li):
            r['row_id'] = r['locnum'] = i + 1

        return li

    return lists(
        fixed_dictionaries(
            {
                'accntnum': from_account_nums,
                'bldgclass': from_building_classes,
                'bldgscheme': from_building_schemes,
                'city': from_cities,
                'country': from_countries,
                'cresta': from_cresta_ids,
                'latitude': from_latitudes,
                'longitude': from_longitudes,
                'numbldgs': from_num_builings,
                'numstories': from_num_stories,
                'occscheme': from_occ_schemes,
                'occtype': from_occ_types,
                'postalcode': from_postal_codes,
                'state': from_states,
                'wscv1ded': from_deductibles1,
                'wscv1limit': from_limits1,
                'wscv1val': from_tivs1,
                'wscv2ded': from_deductibles2,
                'wscv2limit': from_limits2,
                'wscv2val': from_tivs2,
                'wscv3ded': from_deductibles3,
                'wscv3limit': from_limits3,
                'wscv3val': from_tivs3,
                'wscv4ded': from_deductibles4,
                'wscv4limit': from_limits4,
                'wscv4val': from_tivs4,
                'yearbuilt': from_years_built,
                'yearupgrad': from_years_upgraded
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())

def canonical_oed_exposures_data(
    from_account_nums=integers(min_value=1, max_value=10**6),
    from_location_nums=integers(min_value=1, max_value=10**6),
    from_location_names=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_country_codes=text(alphabet=string.ascii_uppercase, min_size=2, max_size=2),
    from_area_codes=text(min_size=1, max_size=20),
    from_location_perils=sampled_from(oed_peril_ids),
    from_buildings_tivs=floats(min_value=0.0, allow_infinity=False),
    from_buildings_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_buildings_limits=floats(min_value=0.0, allow_infinity=False),
    from_other_tivs=floats(min_value=0.0, allow_infinity=False),
    from_other_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_other_limits=floats(min_value=0.0, allow_infinity=False),
    from_contents_tivs=floats(min_value=0.0, allow_infinity=False),
    from_contents_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_contents_limits=floats(min_value=0.0, allow_infinity=False),
    from_bi_tivs=floats(min_value=0.0, allow_infinity=False),
    from_bi_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_bi_limits=floats(min_value=0.0, allow_infinity=False),
    from_combined_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_combined_limits=floats(min_value=0.0, allow_infinity=False),
    from_site_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_site_limits=floats(min_value=0.0, allow_infinity=False),
    size=None,
    min_size=0,
    max_size=10
):
    def _sequence(li):
        for i, r in enumerate(li):
            r['row_id'] = r['locnumber'] = i + 1
            r['locname'] = 'Location {}'.format(i + 1)

        return li

    return lists(
        fixed_dictionaries(
            {
                'accnumber': from_account_nums,
                'locnumber': from_location_nums,
                'locname': from_location_names,
                'countrycode': from_country_codes,
                'areacode': from_area_codes,
                'locperil': from_location_perils,
                'buildingtiv': from_buildings_tivs,
                'locded1building': from_buildings_deductibles,
                'loclimit1building': from_buildings_limits,
                'othertiv': from_other_tivs,
                'locded2other': from_other_deductibles,
                'loclimit2other': from_other_limits,
                'contentstiv': from_contents_tivs,
                'locded3contents': from_contents_deductibles,
                'loclimit3contents': from_contents_limits,
                'bitiv': from_bi_tivs,
                'locded4bi': from_bi_deductibles,
                'loclimit4bi': from_bi_limits,
                'locded5pd': from_combined_deductibles,
                'loclimit5pd': from_combined_limits,
                'locded6all': from_site_deductibles,
                'loclimit6all': from_site_limits
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())


def fm_items_data(
    from_canexp_ids=integers(min_value=0, max_value=9),
    from_canacc_ids=integers(min_value=0, max_value=9),
    from_policy_nums=text(alphabet=string.ascii_letters, min_size=2, max_size=10),
    from_peril_ids=just(OASIS_PERILS['wind']['id']),
    from_coverage_type_ids=sampled_from(coverage_type_ids),
    from_coverage_ids=integers(min_value=0, max_value=9),
    from_is_bi_coverage=sampled_from([True, False]),
    from_level_ids=integers(min_value=1, max_value=10),
    from_layer_ids=integers(min_value=1, max_value=10),
    from_agg_ids=integers(min_value=1, max_value=10),
    from_policytc_ids=integers(min_value=1, max_value=10),
    from_deductible_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_min_deductible_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_max_deductible_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_attachments=floats(min_value=0.0, allow_infinity=False),
    from_limit_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_limits=floats(min_value=0.0, allow_infinity=False),
    from_share_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_shares=floats(min_value=0.0, allow_infinity=False),
    from_calcrule_ids=sampled_from(calcrule_ids),
    from_tiv_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_tiv_tgids=integers(min_value=1, max_value=10),
    from_tivs=floats(min_value=1.0, allow_infinity=False),
    size=None,
    min_size=0,
    max_size=10
):
    def _sequence(li):
        for i, r in enumerate(li):
            r['canexp_id'] = i
            r['item_id'] = r['gul_item_id'] = i + 1

            loc_ids = {r['canexp_id'] for r in li}
            cov_type_ids = {r['coverage_type_id'] for r in li}
            coverage_ids = {
                loc_id:i + 1 for i, (loc_id, cov_type_id) in enumerate(itertools.product(loc_ids, cov_type_ids))
            }

            for r in li:
                r['coverage_id'] = coverage_ids[r['canexp_id']]

        return li

    return lists(
        fixed_dictionaries(
            {
                'canexp_id': from_canexp_ids,
                'canacc_id': from_canacc_ids,
                'policy_num': from_policy_nums,
                'peril_id': from_peril_ids,
                'coverage_type_id': from_coverage_type_ids,
                'coverage_id': from_coverage_ids,
                'is_bi_coverage': from_is_bi_coverage,
                'level_id': from_level_ids,
                'layer_id': from_layer_ids,
                'agg_id': from_agg_ids,
                'policytc_id': from_policytc_ids,
                'deductible': from_deductibles,
                'deductible_min': from_min_deductibles,
                'deductible_max': from_max_deductibles,
                'attachment': from_attachments,
                'limit': from_limits,
                'share': from_shares,
                'calcrule_id': from_calcrule_ids,
                'tiv_elm': from_tiv_elements,
                'tiv_tgid': from_tiv_tgids,
                'tiv': from_tivs,
                'lim_elm': from_limit_elements,
                'ded_elm': from_deductible_elements,
                'ded_min_elm': from_min_deductible_elements,
                'ded_max_elm': from_max_deductible_elements,
                'shr_elm': from_share_elements
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())

def gul_items_data(
    from_canexp_ids=integers(min_value=0, max_value=9),
    from_peril_ids=just(OASIS_PERILS['wind']['id']),
    from_coverage_type_ids=sampled_from(coverage_type_ids),
    from_coverage_ids=integers(min_value=0, max_value=9),
    from_is_bi_coverage=sampled_from([True, False]),
    from_tiv_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_tiv_tgids=integers(min_value=1, max_value=10),
    from_tivs=floats(min_value=1.0, max_value=10**6),
    from_limit_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_deductible_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_min_deductible_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_max_deductible_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_share_elements=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_area_peril_ids=integers(min_value=1, max_value=10),
    from_vulnerability_ids=integers(min_value=1, max_value=10),
    from_summary_ids=integers(min_value=1, max_value=10),
    from_summaryset_ids=integers(min_value=1, max_value=10),
    size=None,
    min_size=0,
    max_size=10
):

    def _sequence(li):
        for i, r in enumerate(li):
            r['canexp_id'] = i
            r['item_id'] = r['group_id'] = i + 1

        loc_ids = {r['canexp_id'] for r in li}
        cov_type_ids = {r['coverage_type_id'] for r in li}
        coverage_ids = {
            loc_id:i + 1 for i, (loc_id, cov_type_id) in enumerate(itertools.product(loc_ids, cov_type_ids))
        }

        for r in li:
            r['coverage_id'] = coverage_ids[r['canexp_id']]

        return li

    return (lists(
        fixed_dictionaries(
            {
                'canexp_id': from_canexp_ids,
                'peril_id': from_peril_ids,
                'coverage_type_id': from_coverage_type_ids,
                'coverage_id': from_coverage_ids,
                'is_bi_coverage': from_is_bi_coverage,
                'tiv_elm': from_tiv_elements,
                'tiv_tgid': from_tiv_tgids,
                'tiv': from_tivs,
                'lim_elm': from_limit_elements,
                'ded_elm': from_deductible_elements,
                'ded_min_elm': from_min_deductible_elements,
                'ded_max_elm': from_max_deductible_elements,
                'shr_elm': from_share_elements,
                'area_peril_id': from_area_peril_ids,
                'vulnerability_id': from_vulnerability_ids,
                'summary_id': from_summary_ids,
                'summaryset_id': from_summaryset_ids
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else min_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing()))


def keys_data(
    from_peril_ids=just(OASIS_PERILS['wind']['id']),
    from_coverage_type_ids=just(OASIS_COVERAGE_TYPES['buildings']['id']),
    from_area_peril_ids=integers(min_value=1, max_value=10),
    from_vulnerability_ids=integers(min_value=1, max_value=10),
    from_statuses=sampled_from(keys_status_flags),
    from_messages=text(min_size=1, max_size=100, alphabet=string.ascii_letters),
    size=None,
    min_size=0,
    max_size=10
):
    def _sequence(li):
        for i, data in enumerate(li):
            data['id'] = i + 1

        return li

    return lists(
        fixed_dictionaries(
            {
                'peril_id': from_peril_ids,
                'coverage_type': from_coverage_type_ids,
                'area_peril_id': from_area_peril_ids,
                'vulnerability_id': from_vulnerability_ids,
                'status': from_statuses,
                'message': from_messages
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())


def write_canonical_files(
    canonical_exposures=None,
    canonical_exposures_file_path=None,
    canonical_accounts=None,
    canonical_accounts_file_path=None
):
    if canonical_exposures_file_path:
        heading_row = OrderedDict([
            ('row_id', 'ROW_ID'),
            ('accntnum', 'ACCNTNUM'),
            ('locnum', 'LOCNUM'),
            ('postalcode', 'POSTALCODE'),
            ('cresta', 'CRESTA'),
            ('city', 'CITY'),
            ('state', 'STATE'),
            ('country', 'COUNTRY'),
            ('latitude', 'LATITUDE'),
            ('longitude', 'LONGITUDE'),
            ('bldgscheme', 'BLDGSCHEME'),
            ('bldgclass', 'BLDGCLASS'),
            ('occscheme', 'OCCSCHEME'),
            ('occtype', 'OCCTYPE'),
            ('yearbuilt', 'YEARBUILT'),
            ('yearupgrad', 'YEARUPGRAD'),
            ('numstories', 'NUMSTORIES'),
            ('numbldgs', 'NUMBLDGS'),
            ('wscv1val', 'WSCV1VAL'),
            ('wscv2val', 'WSCV2VAL'),
            ('wscv3val', 'WSCV3VAL'),
            ('wscv4val', 'WSCV4VAL'),
            ('wscv1limit', 'WSCV1LIMIT'),
            ('wscv2limit', 'WSCV2LIMIT'),
            ('wscv3limit', 'WSCV3LIMIT'),
            ('wscv4limit', 'WSCV4LIMIT'),
            ('wscv1ded', 'WSCV1DED'),
            ('wscv2ded', 'WSCV2DED'),
            ('wscv3ded', 'WSCV3DED'),
            ('wscv4ded', 'WSCV4DED')
        ])

        canexp_df = pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row]+canonical_exposures
        )
        
        canexp_df.to_csv(
            path_or_buf=canonical_exposures_file_path,
            index=False,
            encoding='utf-8',
            header=False
        )

    if canonical_accounts_file_path:
        heading_row = OrderedDict([
            ('row_id', 'ROW_ID'),
            ('accntnum', 'ACCNTNUM'),
            ('policynum', 'POLICYNUM'),
            ('policytype', 'POLICYTYPE'),
            ('undcovamt', 'UNDCOVAMT'),
            ('partof', 'PARTOF'),
            ('blandedamt', 'BLANDEDAMT'),
            ('mindedamt', 'MINDEDAMT'),
            ('maxdedamt', 'MAXDEDAMT'),
            ('blanlimamt', 'BLANLIMAMT')
        ])

        canacc_df = pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row]+canonical_accounts
        )

        canacc_df.to_csv(
            path_or_buf=canonical_accounts_file_path,
            index=False,
            encoding='utf-8',
            header=False
        )


def write_canonical_oed_files(
    exposures=None,
    exposures_fp=None,
    accounts=None,
    accounts_fp=None
):
    if exposures_fp:
        heading_row = OrderedDict([
            ('row_id', 'ROW_ID'),
            ('accnumber', 'AccNumber'),
            ('locnumber', 'LocNumber'),
            ('locname', 'LocName'),
            ('areacode', 'AreaCode'),
            ('countrycode', 'CountryCode'),
            ('locperil', 'LocPeril'),
            ('buildingtiv', 'BuildingTIV'),
            ('locded1building', 'LocDed1Building'),
            ('loclimit1building', 'LocLimit1Building'),
            ('othertiv', 'OtherTIV'),
            ('locded2other', 'LocDed2Other'),
            ('loclimit2other', 'LocLimit2Other'),
            ('contentstiv', 'ContentsTIV'),
            ('locded3contents', 'LocDed3Contents'),
            ('loclimit3contents', 'LocLimit3Contents'),
            ('bitiv', 'BITIV'),
            ('locded4bi', 'LocDed4BI'),
            ('loclimit4bi', 'LocLimit4BI'),
            ('locded5pd', 'LocDed5PD'),
            ('loclimit5pd', 'LocLimit5PD'),
            ('locded6all', 'LocDed6All'),
            ('loclimit6all', 'LocLimit6All')
        ])

        canexp_df = pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row]+exposures
        )
        
        canexp_df.to_csv(
            path_or_buf=exposures_fp,
            index=False,
            encoding='utf-8',
            header=False
        )

    if accounts_fp:
        heading_row = OrderedDict([
            ('row_id', 'ROW_ID'),
            ('accnumber', 'AccNumber'),
            ('portnumber', 'PortNumber'),
            ('polnumber', 'PolNumber'),
            ('polperil', 'PolPeril'),
            ('condded6all', 'CondDed6All'),
            ('condlimit6all', 'CondLimit6All'),
            ('polded6all', 'PolDed6All'),
            ('polminded6all', 'PolMinDed6All'),
            ('polmaxded6all', 'PolMaxDed6All'),
            ('layerattachment', 'LayerAttachment'),
            ('layerlimit', 'LayerLimit'),
            ('layerparticipation', 'LayerParticipation')
        ])

        canacc_df = pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row]+accounts
        )

        canacc_df.to_csv(
            path_or_buf=accounts_fp,
            index=False,
            encoding='utf-8',
            header=False
        )


def write_keys_files(
    keys,
    keys_file_path,
    keys_errors=None,
    keys_errors_file_path=None
):

    heading_row = OrderedDict([
        ('id', 'LocID'),
        ('peril_id', 'PerilID'),
        ('coverage_type', 'CoverageTypeID'),
        ('area_peril_id', 'AreaPerilID'),
        ('vulnerability_id', 'VulnerabilityID')
    ])

    pd.DataFrame(
        columns=heading_row.keys(),
        data=[heading_row]+keys,
    ).to_csv(
        path_or_buf=keys_file_path,
        index=False,
        encoding='utf-8',
        header=False
    )

    if keys_errors and keys_errors_file_path:
        heading_row = OrderedDict([
            ('id', 'LocID'),
            ('peril_id', 'PerilID'),
            ('coverage_type', 'CoverageTypeID'),
            ('message', 'Message'),
        ])

        pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row]+keys_errors,
        ).to_csv(
            path_or_buf=keys_errors_file_path,
            index=False,
            encoding='utf-8',
            header=False
        )
