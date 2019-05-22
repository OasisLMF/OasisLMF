__all__ = [
    'all_cov_types',
    'all_fm_levels',
    'all_fm_level_names',
    'calcrules',
    'deductible_and_limit_types',
    'deductible_codes',
    'fm_profile_types',
    'fm_terms',
    'keys',
    'keys_status_flags',
    'limit_codes',
    'perils',
    'peril_groups',
    'source_accounts',
    'source_exposure',
    'supp_cov_types',
    'supp_fm_levels',
    'supp_fm_level_names',
    'write_source_files',
    'write_keys_files'
]

import string

from itertools import chain
from chainmap import ChainMap
from collections import OrderedDict

import pandas as pd

from hypothesis.strategies import (
    fixed_dictionaries,
    integers,
    just,
    lists,
    nothing,
    sampled_from,
    text,
    floats,
)

from oasislmf.utils.coverages import COVERAGE_TYPES
from oasislmf.utils.fm import (
    DEDUCTIBLE_CODES,
    DEDUCTIBLE_AND_LIMIT_TYPES,
    FM_LEVELS,
    FM_TERMS,
    LIMIT_CODES,
)
from oasislmf.utils.peril import (
    PERILS,
    PERIL_GROUPS,
)
from oasislmf.utils.status import OASIS_KEYS_STATUS

from oasislmf.model_execution.files import (
    GUL_INPUT_FILES,
    IL_INPUT_FILES,
    OPTIONAL_INPUT_FILES,
    INPUT_FILES,
)

calcrules = (1, 4, 5, 6, 7, 8, 10, 11, 12, 12, 13, 14, 15, 16, 19, 21,)

all_cov_types = tuple(v['id'] for v in COVERAGE_TYPES.values())

supp_cov_types = tuple(v['id'] for k, v in COVERAGE_TYPES.items() if k not in ['pd', 'all'])

deductible_and_limit_types = tuple(v['id'] for v in DEDUCTIBLE_AND_LIMIT_TYPES.values())

deductible_codes = tuple(v['id'] for v in DEDUCTIBLE_CODES.values())

limit_codes = tuple(v['id'] for v in LIMIT_CODES.values())

all_fm_levels = tuple(v['id'] for v in FM_LEVELS.values())

supp_fm_levels = tuple(v['id'] for k, v in FM_LEVELS.items() if k not in [
    'cond coverage', 'cond pd', 'policy coverage', 'policy pd', 'account coverage', 'account pd', 'account all'
])

all_fm_level_names = tuple(k for k in FM_LEVELS)

supp_fm_level_names = tuple(k for k, v in FM_LEVELS.items() if v['id'] in supp_fm_levels)

fm_terms = tuple(k for k in FM_TERMS)

fm_profile_types = ('acc', 'loc',)

keys_status_flags = tuple(v['id'] for v in OASIS_KEYS_STATUS.values())

perils = tuple(v['id'] for v in PERILS.values())

peril_groups = tuple(v['id'] for v in PERIL_GROUPS.values())

# Used simple echo command rather than ktools conversion utility for testing purposes
ECHO_CONVERSION_INPUT_FILES = {k: ChainMap({'conversion_tool': 'echo'}, v) for k, v in INPUT_FILES.items()}


def standard_input_files(min_size=0):
    return lists(
        sampled_from([target['name'] for target in chain(GUL_INPUT_FILES.values(), OPTIONAL_INPUT_FILES.values())]),
        min_size=min_size,
        unique=True,
    )


def il_input_files(min_size=0):
    return lists(
        sampled_from([target['name'] for target in IL_INPUT_FILES.values()]),
        min_size=min_size,
        unique=True,
    )


def tar_file_targets(min_size=0):
    return lists(
        sampled_from([target['name'] + '.bin' for target in INPUT_FILES.values()]),
        min_size=min_size,
        unique=True,
    )


def source_accounts(
    from_account_ids=text(alphabet=(string.ascii_letters + string.digits), min_size=1, max_size=40),
    from_policy_ids=text(alphabet=(string.ascii_letters + string.digits), min_size=1, max_size=20),
    from_portfolio_ids=text(alphabet=(string.ascii_letters + string.digits), min_size=1, max_size=20),
    from_policy_perils=text(alphabet=(string.ascii_letters + string.digits), min_size=1, max_size=250),
    from_cond_ids=integers(min_value=1, max_value=10**6),
    from_cond_priorities=integers(min_value=1, max_value=10**6),
    from_condall_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_condall_deductible_codes=just(0),
    from_condall_deductible_types=just(0),
    from_condall_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_condall_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_condall_limits=floats(min_value=0.0, allow_infinity=False),
    from_condall_limit_codes=just(0),
    from_condall_limit_types=just(0),
    from_policyall_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_policyall_deductible_codes=just(0),
    from_policyall_deductible_types=just(0),
    from_policyall_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_policyall_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_policyall_limits=floats(min_value=0.0, allow_infinity=False),
    from_policyall_limit_codes=just(0),
    from_policyall_limit_types=just(0),
    from_policylayer_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_policylayer_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_policylayer_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_policylayer_limits=floats(min_value=0.0, allow_infinity=False),
    from_policylayer_shares=floats(min_value=0.0, max_value=10**6),
    size=None,
    min_size=0,
    max_size=10
):
    def _sequence(li):
        for i, r in enumerate(li):
            r['accnumber'] = '{}'.format(i + 1)

        return li

    return lists(
        fixed_dictionaries(
            {
                'accnumber': from_account_ids,
                'polnumber': from_policy_ids,
                'portnumber': from_portfolio_ids,
                'polperil': from_policy_perils,
                'condnumber': from_cond_ids,
                'condpriority': from_cond_priorities,
                'condded6all': from_condall_deductibles,
                'conddedcode6all': from_condall_deductible_codes,
                'conddedtype6all': from_condall_deductible_types,
                'condminded6all': from_condall_min_deductibles,
                'condmaxded6all': from_condall_max_deductibles,
                'condlimit6all': from_condall_limits,
                'condlimitcode6all': from_condall_limit_codes,
                'condlimittype6all': from_condall_limit_types,
                'polded6all': from_policyall_deductibles,
                'poldedcode6all': from_policyall_deductible_codes,
                'poldedtype6all': from_policyall_deductible_types,
                'polminded6all': from_policyall_min_deductibles,
                'polmaxded6all': from_policyall_max_deductibles,
                'pollimit6all': from_policyall_limits,
                'pollimitcode6all': from_policyall_limit_codes,
                'pollimittype6all': from_policyall_limit_types,
                'layerattachment': from_policylayer_deductibles,
                'layerlimit': from_policylayer_limits,
                'layerparticipation': from_policylayer_shares
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())


def source_exposure(
    from_account_ids=text(alphabet=(string.ascii_letters + string.digits), min_size=1, max_size=40),
    from_portfolio_ids=text(alphabet=(string.ascii_letters + string.digits), min_size=1, max_size=20),
    from_location_ids=text(alphabet=(string.ascii_letters + string.digits), min_size=1, max_size=20),
    from_location_names=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_location_groups=text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    from_country_codes=text(alphabet=string.ascii_uppercase, min_size=2, max_size=2),
    from_area_codes=text(min_size=1, max_size=20),
    from_location_perils_covered=sampled_from(perils),
    from_location_perils=sampled_from(perils),
    from_location_currencies=text(alphabet=string.ascii_letters, min_size=3, max_size=3),
    from_building_tivs=floats(min_value=0.0, allow_infinity=False),
    from_building_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_building_deductible_codes=just(0),
    from_building_deductible_types=just(0),
    from_building_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_building_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_building_limits=floats(min_value=0.0, allow_infinity=False),
    from_building_limit_codes=just(0),
    from_building_limit_types=just(0),
    from_other_tivs=floats(min_value=0.0, allow_infinity=False),
    from_other_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_other_deductible_codes=just(0),
    from_other_deductible_types=just(0),
    from_other_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_other_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_other_limits=floats(min_value=0.0, allow_infinity=False),
    from_other_limit_codes=just(0),
    from_other_limit_types=just(0),
    from_contents_tivs=floats(min_value=0.0, allow_infinity=False),
    from_contents_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_contents_deductible_codes=just(0),
    from_contents_deductible_types=just(0),
    from_contents_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_contents_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_contents_limits=floats(min_value=0.0, allow_infinity=False),
    from_contents_limit_codes=just(0),
    from_contents_limit_types=just(0),
    from_bi_tivs=floats(min_value=0.0, allow_infinity=False),
    from_bi_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_bi_deductible_codes=just(0),
    from_bi_deductible_types=just(0),
    from_bi_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_bi_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_bi_limits=floats(min_value=0.0, allow_infinity=False),
    from_bi_limit_codes=just(0),
    from_bi_limit_types=just(0),
    from_sitepd_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_sitepd_deductible_codes=just(0),
    from_sitepd_deductible_types=just(0),
    from_sitepd_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_sitepd_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_sitepd_limits=floats(min_value=0.0, allow_infinity=False),
    from_sitepd_limit_codes=just(0),
    from_sitepd_limit_types=just(0),
    from_siteall_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_siteall_deductible_codes=just(0),
    from_siteall_deductible_types=just(0),
    from_siteall_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_siteall_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_siteall_limits=floats(min_value=0.0, allow_infinity=False),
    from_siteall_limit_codes=just(0),
    from_siteall_limit_types=just(0),
    from_condall_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_condall_deductible_codes=just(0),
    from_condall_deductible_types=just(0),
    from_condall_min_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_condall_max_deductibles=floats(min_value=0.0, allow_infinity=False),
    from_condall_limits=floats(min_value=0.0, allow_infinity=False),
    from_condall_limit_codes=just(0),
    from_condall_limit_types=just(0),
    from_cond_ids=integers(min_value=1, max_value=10**6),
    from_cond_priorities=integers(min_value=1, max_value=10**6),
    size=None,
    min_size=0,
    max_size=10
):
    def _sequence(li):
        for i, r in enumerate(li):
            r['locnumber'] = r['locname'] = '{}'.format(i + 1)

        return li

    return lists(
        fixed_dictionaries(
            {
                'accnumber': from_account_ids,
                'portnumber': from_portfolio_ids,
                'locnumber': from_location_ids,
                'locname': from_location_names,
                'locgroup': from_location_groups,
                'locperilscovered': from_location_perils_covered,
                'locperil': from_location_perils,
                'loccurrency': from_location_currencies,
                'countrycode': from_country_codes,
                'areacode': from_area_codes,
                'buildingtiv': from_building_tivs,
                'locded1building': from_building_deductibles,
                'locdedcode1building': from_building_deductible_codes,
                'locdedtype1building': from_building_deductible_types,
                'locminded1building': from_building_min_deductibles,
                'locmaxded1building': from_building_max_deductibles,
                'loclimit1building': from_building_limits,
                'loclimitcode1building': from_building_limit_codes,
                'loclimittype1building': from_building_limit_types,
                'othertiv': from_other_tivs,
                'locded2other': from_other_deductibles,
                'locdedcode2other': from_other_deductible_codes,
                'locdedtype2other': from_other_deductible_types,
                'locminded2other': from_other_min_deductibles,
                'locmaxded2other': from_other_max_deductibles,
                'loclimit2other': from_other_limits,
                'loclimitcode2other': from_other_limit_codes,
                'loclimittype2other': from_other_limit_types,
                'contentstiv': from_contents_tivs,
                'locded3contents': from_contents_deductibles,
                'locdedcode3contents': from_contents_deductible_codes,
                'locdedtype3contents': from_contents_deductible_types,
                'locminded3contents': from_contents_min_deductibles,
                'locmaxded3contents': from_contents_max_deductibles,
                'loclimit3contents': from_contents_limits,
                'loclimitcode3contents': from_contents_limit_codes,
                'loclimittype3contents': from_contents_limit_types,
                'bitiv': from_bi_tivs,
                'locded4bi': from_bi_deductibles,
                'locdedcode4bi': from_bi_deductible_codes,
                'locdedtype4bi': from_bi_deductible_types,
                'locminded4bi': from_bi_min_deductibles,
                'locmaxded4bi': from_bi_max_deductibles,
                'loclimit4bi': from_bi_limits,
                'loclimitcode4bi': from_bi_limit_codes,
                'loclimittype4bi': from_bi_limit_types,
                'locded5pd': from_sitepd_deductibles,
                'locdedcode5pd': from_sitepd_deductible_codes,
                'locdedtype5pd': from_sitepd_deductible_types,
                'locminded5pd': from_sitepd_min_deductibles,
                'locmaxded5pd': from_sitepd_max_deductibles,
                'loclimit5pd': from_sitepd_limits,
                'loclimitcode5pd': from_sitepd_limit_codes,
                'loclimittype15pd': from_sitepd_limit_types,
                'locded6all': from_siteall_deductibles,
                'locdedcode6all': from_siteall_deductible_codes,
                'locdedtype6all': from_siteall_deductible_types,
                'locminded6all': from_siteall_min_deductibles,
                'locmaxded6all': from_siteall_max_deductibles,
                'loclimit6all': from_siteall_limits,
                'loclimitcode6all': from_siteall_limit_codes,
                'loclimittype6all': from_siteall_limit_types,
                'condnumber': from_cond_ids,
                'condpriority': from_cond_priorities
            }
        ),
        min_size=(size if size is not None else min_size),
        max_size=(size if size is not None else max_size)
    ).map(_sequence) if (size is not None and size > 0) or (max_size is not None and max_size > 0) else lists(nothing())


def keys(
    from_peril_ids=sampled_from(perils),
    from_coverage_type_ids=sampled_from(supp_cov_types),
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
            data['locnumber'] = '{}'.format(i + 1)

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


def write_source_files(
    exposure=None,
    exposure_fp=None,
    accounts=None,
    accounts_fp=None
):
    if exposure_fp:
        heading_row = OrderedDict([
            ('accnumber', 'AccNumber'),
            ('portnumber', 'PortNumber'),
            ('locnumber', 'LocNumber'),
            ('locname', 'LocName'),
            ('locgroup', 'LocGroup'),
            ('locperilscovered', 'LocPerilsCovered'),
            ('locperil', 'LocPeril'),
            ('loccurrency', 'LocCurrency'),
            ('areacode', 'AreaCode'),
            ('countrycode', 'CountryCode'),
            ('buildingtiv', 'BuildingTIV'),
            ('locded1building', 'LocDed1Building'),
            ('locdedcode1building', 'LocDedCode1Building'),
            ('locdedtype1building', 'LocDedType1Building'),
            ('locminded1building', 'LocMinDed1Building'),
            ('locmaxded1building', 'LocMaxDed1Building'),
            ('loclimit1building', 'LocLimit1Building'),
            ('loclimitcode1building', 'LocLimitCode1Building'),
            ('loclimittype1building', 'LocLimitType1Building'),
            ('othertiv', 'OtherTIV'),
            ('locded2other', 'LocDed2Other'),
            ('locdedcode2other', 'LocDedCode2Other'),
            ('locdedtype2other', 'LocDedType2Other'),
            ('locminded2other', 'LocMinDed2Other'),
            ('locmaxded2other', 'LocMaxDed2Other'),
            ('loclimit2other', 'LocLimit2Other'),
            ('loclimitcode2other', 'LocLimitCode2Other'),
            ('loclimittype2other', 'LocLimitType2Other'),
            ('contentstiv', 'ContentsTIV'),
            ('locded3contents', 'LocDed3Contents'),
            ('locdedcode3contents', 'LocDedCode3Contents'),
            ('locdedtype3contents', 'LocDedType3Contents'),
            ('locminded3contents', 'LocMinDed3Contents'),
            ('locmaxded3contents', 'LocMaxDed3Contents'),
            ('loclimit3contents', 'LocLimit3Contents'),
            ('loclimitcode3contents', 'LocLimitCode3Contents'),
            ('loclimittype3contents', 'LocLimitType3Contents'),
            ('bitiv', 'BITIV'),
            ('locded4bi', 'LocDed4BI'),
            ('locdedcode4bi', 'LocDedCode4BI'),
            ('locdedtype4bi', 'LocDedType4BI'),
            ('locminded4bi', 'LocMinDed4BI'),
            ('locmaxded4bi', 'LocMaxDed4BI'),
            ('loclimit4bi', 'LocLimit4BI'),
            ('loclimitcode4bi', 'LocLimitCode4BI'),
            ('loclimittype4bi', 'LocLimitType4BI'),
            ('locded5pd', 'LocDed5PD'),
            ('locdedcode5pd', 'LocDedCode5PD'),
            ('locdedtype5pd', 'LocDedType5PD'),
            ('locminded5pd', 'LocMinDed5PD'),
            ('locmaxded5pd', 'LocMaxDed5PD'),
            ('loclimit5pd', 'LocLimit5PD'),
            ('loclimitcode5pd', 'LocLimitCode5PD'),
            ('loclimittype5pd', 'LocLimitType5PD'),
            ('locded6all', 'LocDed6All'),
            ('locdedcode6all', 'LocDedCode6All'),
            ('locdedtype6all', 'LocDedType6All'),
            ('locminded6all', 'LocMinDed6All'),
            ('locmaxded6all', 'LocMaxDed6All'),
            ('loclimit6all', 'LocLimit6All'),
            ('loclimitcode6all', 'LocLimitCode6All'),
            ('loclimittype6all', 'LocLimitType6All'),
            ('condnumber', 'CondNumber'),
            ('condpriority', 'CondPriority')
        ])

        pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row] + exposure
        ).to_csv(
            path_or_buf=exposure_fp,
            index=False,
            encoding='utf-8',
            header=False
        )

    if accounts_fp:
        heading_row = OrderedDict([
            ('accnumber', 'AccNumber'),
            ('polnumber', 'PolNumber'),
            ('portnumber', 'PortNumber'),
            ('polperil', 'PolPeril'),
            ('condnumber', 'CondNumber'),
            ('condpriority', 'CondPriority'),
            ('condded6all', 'CondDed6All'),
            ('conddedcode6all', 'CondDedCode6All'),
            ('conddedtype6all', 'CondDedType6All'),
            ('condminded6all', 'CondMinDed6All'),
            ('condmaxded6all', 'CondMaxDed6All'),
            ('condlimit6all', 'CondLimit6All'),
            ('condlimitcode6all', 'CondLimitCode6All'),
            ('condlimittype6all', 'CondLimitType6All'),
            ('polded6all', 'PolDed6All'),
            ('poldedcode6all', 'PolDedCode6All'),
            ('poldedtype6all', 'PolDedType6All'),
            ('polminded6all', 'PolMinDed6All'),
            ('polmaxded6all', 'PolMaxDed6All'),
            ('pollimit6all', 'PolLimit6All'),
            ('pollimitcode6all', 'PolLimitCode6All'),
            ('pollimittype6all', 'PolLimitType6All'),
            ('layerattachment', 'LayerAttachment'),
            ('layerlimit', 'LayerLimit'),
            ('layerparticipation', 'LayerParticipation')
        ])

        pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row] + accounts
        ).to_csv(
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
        ('locnumber', 'LocID'),
        ('peril_id', 'PerilID'),
        ('coverage_type', 'CoverageTypeID'),
        ('area_peril_id', 'AreaPerilID'),
        ('vulnerability_id', 'VulnerabilityID')
    ])

    pd.DataFrame(
        columns=heading_row.keys(),
        data=[heading_row] + keys,
    ).to_csv(
        path_or_buf=keys_file_path,
        index=False,
        encoding='utf-8',
        header=False
    )

    if keys_errors and keys_errors_file_path:
        heading_row = OrderedDict([
            ('locnumber', 'LocID'),
            ('peril_id', 'PerilID'),
            ('coverage_type', 'CoverageTypeID'),
            ('message', 'Message'),
        ])

        pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row] + keys_errors,
        ).to_csv(
            path_or_buf=keys_errors_file_path,
            index=False,
            encoding='utf-8',
            header=False
        )
