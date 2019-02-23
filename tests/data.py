from __future__ import unicode_literals

__all__ = [
    'calcrule_ids',
    'canonical_accounts',
    'canonical_accounts_profile',
    'canonical_exposure',
    'canonical_exposure_profile',
    'canonical_oed_accounts',
    'canonical_oed_accounts_profile',
    'canonical_oed_exposure',
    'canonical_oed_exposure_profile',
    'coverage_type_ids',
    'deductible_types',
    'fm_input_items',
    'fm_level_names',
    'fm_level_names_simple',
    'fm_levels',
    'fm_levels_simple',
    'gul_input_items',
    'keys',
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
import string

from itertools import chain
from chainmap import ChainMap
from collections import OrderedDict
from future.utils import itervalues

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
    OASIS_KEYS_STATUS,
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

coverage_type_ids = tuple(OED_COVERAGE_TYPES[k]['id'] for k in OED_COVERAGE_TYPES)

deductible_types = tuple(DEDUCTIBLE_TYPES[k]['id'] for k in DEDUCTIBLE_TYPES)

fm_levels = tuple(OED_FM_LEVELS[k]['id'] for k in OED_FM_LEVELS)

fm_level_names = tuple(k.capitalize() for k in OED_FM_LEVELS)

fm_levels_simple = tuple(
    t for t in set(
        t.get('FMLevel')
        for t in itertools.chain(itervalues(canonical_exposure_profile_simple), itervalues(canonical_accounts_profile))
    ) if t
)

fm_level_names_simple = tuple(
    t[0] for t in sorted([t for t in set(
        (t.get('FMLevelName'), t.get('FMLevel'))
        for t in itertools.chain(itervalues(canonical_exposure_profile_simple), itervalues(canonical_accounts_profile))
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
        sampled_from([target['name'] for target in chain(itervalues(GUL_INPUT_FILES), itervalues(OPTIONAL_INPUT_FILES))]),
        min_size=min_size,
        unique=True,
    )


def il_input_files(min_size=0):
    return lists(
        sampled_from([target['name'] for target in itervalues(IL_INPUT_FILES)]),
        min_size=min_size,
        unique=True,
    )


def tar_file_targets(min_size=0):
    return lists(
        sampled_from([target['name'] + '.bin' for target in itervalues(INPUT_FILES)]),
        min_size=min_size,
        unique=True,
    )

oed_tiv_elements = tuple(v['ProfileElementName'].lower() for v in canonical_oed_exposure_profile.values() if v.get('FMTermType') and v.get('FMTermType').lower() == 'tiv')

def source_oed_accounts(
    from_account_nums=integers(min_value=1, max_value=10**6),
    from_portfolio_nums=integers(min_value=1, max_value=10**6),
    from_policy_nums=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_policy_perils=sampled_from(oed_peril_ids),
    from_condall_deductibles=floats(min_value=0.0, max_value=10**6),
    from_condall_limits=floats(min_value=0.0, max_value=10**6),
    from_cond_numbers=integers(min_value=1, max_value=10**6),
    from_policyall_deductibles=floats(min_value=0.0, max_value=10**6),
    from_policyall_min_deductibles=floats(min_value=0.0, max_value=10**6),
    from_policyall_max_deductibles=floats(min_value=0.0, max_value=10**6),
    from_policy_layer_deductibles=floats(min_value=0.0, max_value=10**6),
    from_policy_layer_limits=floats(min_value=0.0, max_value=10**6),
    from_policy_layer_shares=floats(min_value=0.0, max_value=10**6),
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
                'condded6all': from_condall_deductibles,
                'condlimit6all': from_condall_limits,
                'condnumber': from_cond_numbers,
                'polded6all': from_policyall_deductibles,
                'polminded6all': from_policyall_min_deductibles,
                'polmaxded6all': from_policyall_max_deductibles,
                'layerattachment': from_policy_layer_deductibles,
                'layerlimit': from_policy_layer_limits,
                'layerparticipation': from_policy_layer_shares
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())


def source_oed_exposure(
    from_account_nums=integers(min_value=1, max_value=10**6),
    from_location_nums=integers(min_value=1, max_value=10**6),
    from_location_names=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_location_groups=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
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
    from_cond_numbers=integers(min_value=1, max_value=10**6),
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
                'locgroup': from_location_groups,
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
                'loclimit6all': from_site_limits,
                'condnumber': from_cond_numbers,
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())


def fm_input_items(
    from_srcexp_ids=integers(min_value=0, max_value=9),
    from_srcacc_ids=integers(min_value=0, max_value=9),
    from_policy_nums=text(alphabet=string.ascii_letters, min_size=2, max_size=10),
    from_peril_ids=just(OED_PERILS['WTC']['id']),
    from_coverage_type_ids=sampled_from(coverage_type_ids),
    from_coverage_ids=integers(min_value=1, max_value=10),
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

def gul_input_items(
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


def keys(
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
            ('loclimit6all', 'LocLimit6All'),
            ('condtag', 'CondTag')
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
            ('condnumber', 'CondNumber'),
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
