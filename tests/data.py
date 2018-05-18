from __future__ import unicode_literals

__all__ = [
    'canonical_accounts_data',
    'canonical_accounts_profile_piwind',
    'canonical_exposures_data',
    'canonical_exposures_profile_piwind',
    'canonical_exposures_profile_piwind_simple',
    'calcrule_ids',
    'coverage_type_ids',
    'deductible_types',
    'deductible_types_piwind',
    'fm_items_data',
    'fm_level_names_piwind',
    'fm_level_names_piwind_simple',
    'fm_levels_piwind',
    'fm_levels_piwind_simple',
    'gul_items_data',
    'keys_data',
    'keys_status_flags',
    'peril_ids',
    'write_canonical_files',
    'write_keys_files'
]

import string

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

from oasislmf.utils.coverage import (
    BUILDING_COVERAGE_CODE,
    CONTENTS_COVERAGE_CODE,
    OTHER_STRUCTURES_COVERAGE_CODE,
    TIME_COVERAGE_CODE,
)
from oasislmf.utils.peril import (
    PERIL_ID_FLOOD,
    PERIL_ID_QUAKE,
    PERIL_ID_SURGE,
    PERIL_ID_WIND,
)
from oasislmf.utils.status import (
    KEYS_STATUS_FAIL,
    KEYS_STATUS_NOMATCH,
    KEYS_STATUS_SUCCESS,
)

calcrule_ids = (1, 2, 10, 11, 12, 15,)

canonical_accounts_profile_piwind = {
    'ACCNTNUM': {
        'FieldName': 'AccountNumber',
        'ProfileElementName': 'ACCNTNUM',
        'ProfileType': 'Acc'
    },
    'BLANDEDAMT': {
        'DeductibleType': 'B',
        'FMLevel': 5,
        'FMLevelName': 'Account',
        'FMTermGroupID': 1,
        'FMTermType': 'Deductible',
        'FieldName': 'BlanketDeductible',
        'ProfileElementName': 'BLANDEDAMT',
        'ProfileType': 'Acc'
    },
    'BLANLIMAMT': {
        'FMLevel': 6,
        'FMLevelName': 'Layer',
        'FMTermGroupID': 1,
        'FMTermType': 'Share',
        'FieldName': 'BlanketLimit',
        'ProfileElementName': 'BLANLIMAMT',
        'ProfileType': 'Acc'
    },
    'PARTOF': {
        'FMLevel': 6,
        'FMLevelName': 'Layer',
        'FMTermGroupID': 1,
        'FMTermType': 'Limit',
        'FieldName': 'LayerLimit',
        'ProfileElementName': 'PARTOF',
        'ProfileType': 'Acc'
    },
    'POLICYNUM': {
        'FieldName': 'PolicyNumber',
        'ProfileElementName': 'POLICYNUM',
        'ProfileType': 'Acc'
    },
    'POLICYTYPE': {
        'FieldName': 'PolicyType',
        'ProfileElementName': 'POLICYTYPE',
        'ProfileType': 'Acc'
    },
    'ROW_ID': {
        'FieldName': 'LocationID',
        'ProfileElementName': 'ROW_ID',
        'ProfileType': 'Acc'
    },
    'UNDCOVAMT': {
        'DeductibleType': 'B',
        'FMLevel': 6,
        'FMLevelName': 'Layer',
        'FMTermGroupID': 1,
        'FMTermType': 'Deductible',
        'FieldName': 'AttachmentPoint',
        'ProfileElementName': 'UNDCOVAMT',
        'ProfileType': 'Acc'
    }
}

canonical_exposures_profile_piwind = {
    'ACCNTNUM': {
        'FieldName': 'AccountNumber',
        'ProfileElementName': 'ACCNTNUM',
        'ProfileType': 'Loc'
    },
    'ADDRMATCH': {},
    'BLDGCLASS': {},
    'BLDGSCHEME': {},
    'CITY': {},
    'CITYCODE': {},
    'CNTRYCODE': {},
    'CNTRYSCHEME': {},
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
    'COND1NAME': {
        'FieldName': 'SubLimitReference',
        'ProfileElementName': 'COND1NAME'
    },
    'COND1TYPE': {},
    'COUNTRY': {},
    'COUNTRYGEOID': {},
    'COUNTY': {},
    'COUNTYCODE': {},
    'CRESTA': {},
    'LATITUDE': {},
    'LOCNUM': {},
    'LONGITUDE': {},
    'NUMBLDGS': {},
    'NUMSTORIES': {},
    'OCCSCHEME': {},
    'OCCTYPE': {},
    'POSTALCODE': {},
    'ROOFGEOM': {},
    'ROW_ID': {
        'FieldName': 'LocationID',
        'ProfileElementName': 'ROW_ID',
        'ProfileType': 'Loc'
    },
    'STATE': {},
    'STATECODE': {},
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
    'WSCV10DED': {},
    'WSCV10LMT': {},
    'WSCV10VAL': {},
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
    },
    'WSCV2DED': {},
    'WSCV2LIMIT': {},
    'WSCV2VAL': {},
    'WSCV3DED': {},
    'WSCV3LIMIT': {},
    'WSCV3VAL': {},
    'WSCV4DED': {
        'CoverageTypeID': 1,
        'DeductibleType': 'B',
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 2,
        'FMTermType': 'Deductible',
        'FieldName': 'CoverageDeductible',
        'PerilID': 1,
        'ProfileElementName': 'WSCV4DED',
        'ProfileType': 'Loc'
    },
    'WSCV4LIMIT': {
        'CoverageTypeID': 1,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 2,
        'FMTermType': 'Limit',
        'FieldName': 'CoverageLimit',
        'PerilID': 1,
        'ProfileElementName': 'WSCV4LIMIT',
        'ProfileType': 'Loc'
    },
    'WSCV4VAL': {
        'CoverageTypeID': 1,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 2,
        'FMTermType': 'TIV',
        'FieldName': 'TIV',
        'PerilID': 1,
        'ProfileElementName': 'WSCV4VAL',
        'ProfileType': 'Loc'
    },
    'WSCV5DED': {},
    'WSCV5LIMIT': {},
    'WSCV5VAL': {},
    'WSCV6DED': {},
    'WSCV6LIMIT': {},
    'WSCV6VAL': {},
    'WSCV7DED': {},
    'WSCV7LIMIT': {},
    'WSCV7VAL': {},
    'WSCV8DED': {},
    'WSCV8LIMIT': {},
    'WSCV8VAL': {},
    'WSCV9DED': {},
    'WSCV9LIMIT': {},
    'WSCV9VAL': {},
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
    },
    'YEARBUILT': {},
    'YEARUPGRAD': {}
}

canonical_exposures_profile_piwind_simple = {
    'ACCNTNUM': {
        'FieldName': 'AccountNumber',
        'ProfileElementName': 'ACCNTNUM',
        'ProfileType': 'Loc'
    },
    'BLDGCLASS': {},
    'BLDGSCHEME': {},
    'CITY': {},
    'COUNTRY': {},
    'CRESTA': {},
    'LATITUDE': {},
    'LOCNUM': {},
    'LONGITUDE': {},
    'NUMBLDGS': {},
    'NUMSTORIES': {},
    'OCCSCHEME': {},
    'OCCTYPE': {},
    'POSTALCODE': {},
    'ROW_ID': {
        'FieldName': 'LocationID',
        'ProfileElementName': 'ROW_ID',
        'ProfileType': 'Loc'
    },
    'STATE': {},
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
    },
    'WSCV2DED': {
        'CoverageTypeID': 1,
        'DeductibleType': 'B',
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 2,
        'FMTermType': 'Deductible',
        'FieldName': 'CoverageDeductible',
        'PerilID': 1,
        'ProfileElementName': 'WSCV2DED',
        'ProfileType': 'Loc'
    },
    'WSCV2LIMIT': {
        'CoverageTypeID': 1,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 2,
        'FMTermType': 'Limit',
        'FieldName': 'CoverageLimit',
        'PerilID': 1,
        'ProfileElementName': 'WSCV2LIMIT',
        'ProfileType': 'Loc'
    },
    'WSCV2VAL': {
        'CoverageTypeID': 1,
        'FMLevel': 1,
        'FMLevelName': 'Coverage',
        'FMTermGroupID': 2,
        'FMTermType': 'TIV',
        'FieldName': 'TIV',
        'PerilID': 1,
        'ProfileElementName': 'WSCV2VAL',
        'ProfileType': 'Loc'
    },
    'YEARBUILT': {},
    'YEARUPGRAD': {}
}

coverage_type_ids = (BUILDING_COVERAGE_CODE, CONTENTS_COVERAGE_CODE, OTHER_STRUCTURES_COVERAGE_CODE, TIME_COVERAGE_CODE,)

deductible_types = ('B', 'MA', 'MI',)

deductible_types_piwind = tuple(
    t for t in set(t['DeductibleType'] if t.get('FMTermType') and t.get('FMTermType').lower() == 'deductible' else None for t in canonical_exposures_profile_piwind_simple.values() + canonical_accounts_profile_piwind.values()) if t
)

fm_levels_piwind = tuple(
    t for t in set(t.get('FMLevel') for t in canonical_exposures_profile_piwind.values() + canonical_accounts_profile_piwind.values()) if t
)

fm_levels_piwind_simple = tuple(
    t for t in set(t.get('FMLevel') for t in canonical_exposures_profile_piwind_simple.values() + canonical_accounts_profile_piwind.values()) if t
)

fm_level_names_piwind = tuple(
    t[0] for t in sorted(set((t.get('FMLevelName'), t.get('FMLevel')) for t in canonical_exposures_profile_piwind.values() + canonical_accounts_profile_piwind.values()), key=lambda t: t[1]) if t[0]
)

fm_level_names_piwind_simple = tuple(
    t[0] for t in sorted(set((t.get('FMLevelName'), t.get('FMLevel')) for t in canonical_exposures_profile_piwind_simple.values() + canonical_accounts_profile_piwind.values()), key=lambda t: t[1]) if t[0]
)

fm_term_types = ('Deductible', 'Limit', 'Share', 'TIV',)

fm_profile_types = ('acc', 'loc',)

keys_status_flags = (KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH, KEYS_STATUS_SUCCESS,)

peril_ids = (PERIL_ID_FLOOD, PERIL_ID_QUAKE, PERIL_ID_QUAKE, PERIL_ID_WIND,)


def canonical_accounts_data(
    from_accounts_nums=integers(min_value=1, max_value=10**5),
    from_policy_nums=text(alphabet=string.ascii_letters, min_size=2, max_size=10),
    from_policy_types=integers(min_value=1, max_value=10),
    from_attachment_points=floats(min_value=0.0, max_value=10**6),
    from_layer_limits=floats(min_value=0.0, max_value=10**6),
    from_blanket_limits=floats(min_value=0.0, max_value=10**6),
    from_blanket_deductibles=floats(min_value=0.0, max_value=10**6),
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
                'accntnum': from_accounts_nums,
                'policynum': from_policy_nums,
                'policytype': from_policy_types,
                'undcovamt': from_attachment_points,
                'partof': from_layer_limits,
                'blandedamt': from_blanket_deductibles,
                'blanlimamt': from_blanket_limits
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())


def canonical_exposures_data(
    from_accounts_nums=integers(min_value=1, max_value=10**6),
    from_building_classes=integers(min_value=1, max_value=3),
    from_building_schemes=text(alphabet=string.ascii_letters, min_size=1, max_size=3),
    from_cities=text(alphabet=string.ascii_letters, min_size=2, max_size=20),
    from_countries=text(alphabet=string.ascii_letters, min_size=2, max_size=20),
    from_cresta_ids=text(alphabet=string.ascii_letters, min_size=1, max_size=10),
    from_deductibles1=floats(min_value=0.0, allow_infinity=False),
    from_deductibles2=floats(min_value=0.0, allow_infinity=False),
    from_latitudes=floats(min_value=0.0, max_value=90.0),
    from_limits1=floats(min_value=0.0, allow_infinity=False),
    from_limits2=floats(min_value=0.0, allow_infinity=False),
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
                'accntnum': from_accounts_nums,
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
                'yearbuilt': from_years_built,
                'yearupgrad': from_years_upgraded
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())


def fm_items_data(
    from_canexp_ids=integers(min_value=0, max_value=9),
    from_canacc_ids=integers(min_value=0, max_value=9),
    from_level_ids=sampled_from(fm_levels_piwind),
    from_layer_ids=integers(min_value=1, max_value=10),
    from_agg_ids=integers(min_value=1, max_value=10),
    from_policytc_ids=integers(min_value=1, max_value=10),
    from_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_limits=floats(min_value=0.0, allow_infinity=False),
    from_shares=floats(min_value=0.0, allow_infinity=False),
    from_deductible_types=sampled_from(deductible_types),
    from_calcrule_ids=sampled_from(calcrule_ids),
    from_tivs=floats(min_value=1.0, allow_infinity=False),
    size=None,
    min_size=0,
    max_size=10
):
    def _sequence(li):
        for i, r in enumerate(li):
            r['canexp_id'] = i
            r['item_id'] = i + 1

        return li

    return lists(
        fixed_dictionaries(
            {
                'canexp_id': from_canexp_ids,
                'canacc_id': from_canacc_ids,
                'level_id': from_level_ids,
                'layer_id': from_layer_ids,
                'agg_id': from_agg_ids,
                'policytc_id': from_policytc_ids,
                'deductible': from_deductibles,
                'limit': from_limits,
                'share': from_shares,
                'deductible_type': from_deductible_types,
                'calcrule_id': from_calcrule_ids,
                'tiv': from_tivs
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())

def gul_items_data(
    from_canexp_ids=integers(min_value=0, max_value=9),
    from_tivs=floats(min_value=1.0, max_value=10**6),
    from_area_peril_ids=integers(min_value=1, max_value=10),
    from_vulnerability_ids=integers(min_value=1, max_value=10),
    from_summary_ids=integers(min_value=1, max_value=10),
    from_summaryset_ids=integers(min_value=1, max_value=10),
    with_fm=False,
    size=None,
    min_size=0,
    max_size=10
):

    def _sequence(li):
        for i, r in enumerate(li):
            r['canexp_id'] = i
            r['item_id'] = r['coverage_id'] = r['group_id'] = i + 1
            if with_fm:
                r['canacc_id'] = i

        return li

    return (lists(
        fixed_dictionaries(
            {
                'canexp_id': from_canexp_ids,
                'tiv': from_tivs,
                'area_peril_id': from_area_peril_ids,
                'vulnerability_id': from_vulnerability_ids,
                'summary_id': from_summary_ids,
                'summaryset_id': from_summaryset_ids
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else min_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())) if not with_fm else lists(
        fixed_dictionaries(
            {
                'canexp_id': from_canexp_ids,
                'canacc_id': from_canacc_ids,
                'tiv': from_tivs,
                'area_peril_id': from_area_peril_ids,
                'vulnerability_id': from_vulnerability_ids,
                'summary_id': from_summary_ids,
                'summaryset_id': from_summaryset_ids
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())


def keys_data(
    from_peril_ids=just(PERIL_ID_WIND),
    from_coverage_type_ids=just(BUILDING_COVERAGE_CODE),
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
                'coverage': from_coverage_type_ids,
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

    if canonical_exposures and canonical_exposures_file_path:
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
            ('wscv1limit', 'WSCV1LIMIT'),
            ('wscv2limit', 'WSCV2LIMIT'),
            ('wscv1ded', 'WSCV1DED'),
            ('wscv2ded', 'WSCV2DED')
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

    if canonical_accounts and canonical_accounts_file_path:
        heading_row = OrderedDict([
            ('row_id', 'ROW_ID'),
            ('accntnum', 'ACCNTNUM'),
            ('policynum', 'POLICYNUM'),
            ('policytype', 'POLICYTYPE'),
            ('undcovamt', 'UNDCOVAMT'),
            ('partof', 'PARTOF'),
            ('blandedamt', 'BLANDEDAMT'),
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


def write_keys_files(
    keys,
    keys_file_path,
    keys_errors=None,
    keys_errors_file_path=None
):

    heading_row = OrderedDict([
        ('id', 'LocID'),
        ('peril_id', 'PerilID'),
        ('coverage', 'CoverageID'),
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
            ('coverage', 'CoverageID'),
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
