# -*- coding: utf-8 -*-

__all__ = [
    'load_oed_dfs'
]


import math
import pandas as pd
import os
from oasislmf.utils.data import get_dataframe
from collections import namedtuple


# TODO - add validator
class OedValidator(object):

    def __init__(self, ri_info_rules=None, ri_scope_rules=None):
        self.rules_ode_scope = ri_info_rules
        self.rules_ode_info = ri_scope_rules

        int_or_float = [pd.np.dtype('int64'), pd.np.dtype('float64')]
        float_only = [pd.np.dtype('float64')]
        str_only = [str]
        int_only = [pd.np.dtype('int64')]
        int_or_str = [pd.np.dtype('int64'), str]
        object_or_str = [pd.np.dtype('O'), str]

        self.ri_info_required_cols = [
            'ReinsNumber', 'ReinsPeril', 'PlacedPercent',
            'InuringPriority', 'ReinsType'
            ]

        self.ri_info_defaults = {
#            'ReinsLayerNumber': '',
            'CededPercent': 1.0,
            'RiskLimit': 0.0,
            'RiskAttachment': 0.0,
            'OccLimit': 0.0,
            'OccAttachment': 0.0,
            'TreatyShare': 0.0}

        self.ri_info_expected_dtypes = {
            'ReinsNumber': int_or_float,
#            'ReinsLayerNumber': int_or_float,
            'CededPercent': float_only,
            'RiskLimit': int_or_float,
            'RiskAttachment': int_or_float,
            'OccLimit': int_or_float,
            'OccAttachment': int_or_float,
            'InuringPriority': int_or_float,
            'ReinsType': object_or_str,
            'PlacedPercent': float_only,
            'TreatyShare': float_only}

        self.ri_scope_required_cols = {
            'ReinsNumber', 'RiskLevel'
            }

        self.ri_scope_expected_dtypes = {
            'ReinsNumber': int_or_float,
            'PortNumber': int_or_float + object_or_str,
            'AccNumber': int_or_float + object_or_str,
            'LocGroup': int_or_float + object_or_str,
            'PolNumber': int_or_float + object_or_str,
            'LocNumber': int_or_float + object_or_str,
            'CedantName': int_or_float + object_or_str,
            'ProducerName': int_or_float + object_or_str,
            'LOB': int_or_float + object_or_str,
            'CountryCode': int_or_float + object_or_str,
            'ReinsTag': int_or_float + object_or_str,
            'RiskLevel': object_or_str,
            'CededPercent': float_only}

        self.error_structure = {
        }

    def _unique_reins(self, reins_info_df):
        '''
        check if only one reins type exisits
        '''
        return (len(reins_info_df.ReinsType.unique()) == 1)

    def _find_missing(self, df_src, column_name, df_dest):
        '''
        return list of values if df_dest[`column_name`] values do not exist in df_src
        '''
        src_values = df_dest[column_name].unique().tolist()
        missing_df = df_src[~df_src.isin({column_name: src_values})].dropna()
        return missing_df[column_name].tolist()

    def _check_df_dtypes(self, dtypes_given, dtypes_expected):
        '''
        - All column headers must match
        - All datatypes much match
        '''
        msg_type_check = []
        msg_string = "Type error in column '{}': expected '{}'  found '{}'"

        dtypes_diff = set(dtypes_given.keys()) - set(dtypes_expected.keys()) 
        if len(dtypes_diff) > 0:
            return (False, "Column header mismatch: {}".format(
                ', '.join(dtypes_diff)
            ))

        for col in dtypes_expected.keys():
            if dtypes_given[col] not in dtypes_expected[col]:
                msg_type_check.append(msg_string.format(
                    col, dtypes_expected[col], dtypes_given[col]))
        return msg_type_check

    def _all_scope_non_specific(self, scope_df):
        return scope_df[['AccountNumber',
                         'PolicyNumber',
                         'LocationNumber'
                         ]].isnull().all().all()

    def _all_scope_specific(self, scope_df):
        return scope_df[['AccountNumber',
                         'PolicyNumber',
                         'LocationNumber'
                         ]].notnull().all().all()

    def _error_struture(self, check_type, chcek_scope, msg, info=''):
        return {
            "check": check_type,
            "scope": chcek_scope,
            "messsages": msg,
            "meta_data": info,
        }

    # def validate(self, account_df, location_df, ri_info_df, ri_scope_df):
    def validate(self, ri_info_df, ri_scope_df):
        '''
        Validate OED resinurance structure before running calculations.
        '''
        error_list = []

        # CHECK - Datatypes ri_info
        given_dtypes = ri_info_df.dtypes.to_dict()
        error_msg = self._check_df_dtypes(given_dtypes,
                                          self.ri_info_expected_dtypes)
        if error_msg:
            error_list.append(self._error_struture(
                "datatypes",
                "ri_info file",
                error_msg))

        # CHECK - Datatypes ri_scope
        given_dtypes = ri_scope_df.dtypes.to_dict()
        error_msg = self._check_df_dtypes(given_dtypes,
                                          self.ri_scope_expected_dtypes)
        if error_msg:
            error_list.append(self._error_struture(
                "datatypes",
                "ri_info file",
                error_msg))

        for inuring_priority in range(1, ri_info_df['InuringPriority'].max() + 1):
            inuring_priority_ri_info_df = ri_info_df[ri_info_df.InuringPriority == inuring_priority]
            if inuring_priority_ri_info_df.empty:
                continue

            inuring_scope_ids = inuring_priority_ri_info_df.ReinsNumber.tolist()
            inuring_scopes = [ri_scope_df[ri_scope_df.ReinsNumber == ID] for ID in inuring_scope_ids]
            reins_types_found = inuring_priority_ri_info_df.ReinsType.unique().tolist()

            meta_data = {
                "InuringPriority": inuring_priority,
                "ReinsTypes": reins_types_found,
                "ri_info_ReinsNumbers": inuring_priority_ri_info_df.ReinsNumber.tolist(),
                "ri_info_line_nums": [idx + 2 for idx in inuring_priority_ri_info_df.index.tolist()],
            }

            # CHECK - only single ri_type is set per inuring priority
            if len(reins_types_found) > 1:
                error_list.append(self._error_struture(
                    "inuring_reins_type",
                    "RI_{}".format(inuring_priority),
                    "Inuring layer must have a unique ReinsType.",
                    meta_data
                ))
                continue
            elif len(reins_types_found) < 1:
                error_list.append(self._error_struture(
                    "inuring_reins_type",
                    "RI_{}".format(inuring_priority),
                    "Inuring layer missing a ReinsType.",
                    meta_data
                ))
                continue

            #CHECK - ri_type is supported
            ri_type = reins_types_found[0]
            if ri_type not in REINS_TYPES:
                error_list.append(self._error_struture(
                    "inuring_reins_type",
                    "RI_{}".format(inuring_priority),
                    "Unsupported ReinsType",
                    meta_data
                ))

            # CHECK scope of inuring layer
            for scope_df in inuring_scopes:
                scope_risk_levels = scope_df.RiskLevel.unique()
                meta_data.update({
                    "RiskLevels": scope_risk_levels.tolist(),
                    "ri_scope_ReinsNumber": scope_df.ReinsNumber.tolist(),
                    "ri_scope_line_nums": [idx + 2 for idx in scope_df.index.tolist()],
                })
                # CHECK - each scope only has one risk level type
                if len(scope_risk_levels) > 1:
                    error_list.append(self._error_struture(
                        "inuring_risk_level",
                        "RI_{}".format(inuring_priority),
                        "Mix of risk levels in a single reinsurance scope",
                        meta_data,
                    ))
                    continue
                elif len(scope_risk_levels) < 1:
                    error_list.append(self._error_struture(
                        "inuring_risk_level",
                        "RI_{}".format(inuring_priority),
                        "inuring layer has no reinsurance scope",
                        meta_data,
                    ))
                    continue

                # CHECK - Risk level is supported
                risk_level_id = scope_risk_levels[0]
                if risk_level_id not in SUPPORTED_RISK_LEVELS[ri_type]:
                    error_list.append(self._error_struture(
                        "inuring_risk_level",
                        "RI_{}".format(inuring_priority),
                        "Unsupported risk level",
                        meta_data,
                    ))

        return (not error_list, error_list)


def load_oed_dfs(oed_dir, show_all=False):
    """
    Load OED data files.
    """

    do_reinsurance = True
    if oed_dir is not None:
        if not os.path.exists(oed_dir):
            print("Path does not exist: {}".format(oed_dir))
            exit(1)

        # RI files
        oed_ri_info_file = os.path.join(oed_dir, "ri_info.csv")
        oed_ri_scope_file = os.path.join(oed_dir, "ri_scope.csv")
        oed_ri_info_file_exists = os.path.exists(oed_ri_info_file)
        oed_ri_scope_file_exists = os.path.exists(oed_ri_scope_file)

        if not oed_ri_info_file_exists and not oed_ri_scope_file_exists:
            ri_info_df = None
            ri_scope_df = None
            do_reinsurance = False
        elif oed_ri_info_file_exists and oed_ri_scope_file_exists:
            ri_info_df = get_dataframe(
                oed_ri_info_file, lowercase_cols=False, 
                required_cols=RI_INFO_REQUIRED_COLS,
                default_values=RI_INFO_DEFAULTS)
            ri_scope_df = get_dataframe(
                oed_ri_scope_file, lowercase_cols=False,
                required_cols=RI_SCOPE_REQUIRED_COLS,
                default_values=RI_SCOPE_DEFAULTS)
        else:
            print("Both reinsurance files must exist: {} {}".format(
                oed_ri_info_file, oed_ri_scope_file))

        if do_reinsurance:
            ri_info_df = ri_info_df[OED_REINS_INFO_FIELDS].copy()
            ri_scope_df = ri_scope_df[OED_REINS_SCOPE_FIELDS].copy()

        # Ensure Percent feilds are float
        info_float_cols = ['CededPercent', 'PlacedPercent', 'TreatyShare']
        scope_float_cols = ['CededPercent']
        ri_info_df[info_float_cols] = ri_info_df[info_float_cols].astype(float)
        ri_scope_df[scope_float_cols] = ri_scope_df[scope_float_cols].astype(float)

    return (ri_info_df, ri_scope_df, do_reinsurance)

#
# Ktools constants
# 
DEDUCTIBLE_AND_LIMIT_CALCRULE_ID = 1
FRANCHISE_DEDUCTIBLE_AND_LIMIT_CALCRULE_ID = 3
DEDUCTIBLE_ONLY_CALCRULE_ID = 12
DEDUCTIBLE_AS_A_CAP_ON_THE_RETENTION_OF_INPUT_LOSSES_CALCRULE_ID = 10
DEDUCTIBLE_AS_A_FLOOR_ON_THE_RETENTION_OF_INPUT_LOSSES_CALCRULE_ID = 11
DEDUCTIBLE_LIMIT_AND_SHARE_CALCRULE_ID = 2
DEDUCTIBLE_AND_LIMIT_AS_A_PROPORTION_OF_LOSS_CALCRUKE_ID = 5
DEDUCTIBLE_WITH_LIMIT_AS_A_PROPORTION_OF_LOSS_CALCRUKE_ID = 9
LIMIT_ONLY_CALCRULE_ID = 14
LIMIT_AS_A_PROPORTION_OF_LOSS_CALCRULE_ID = 15
DEDUCTIBLE_AS_A_PROPORTION_OF_LOSS_CALCRULE_ID = 16

CALCRULE_ID_DEDUCTIBLE_AND_LIMIT = 1
CALCRULE_ID_DEDUCTIBLE_ATTACHMENT_LIMIT_AND_SHARE = 2
CALCRULE_ID_FRANCHISE_DEDUCTIBLE_AND_LIMIT = 3
CALCRULE_ID_DEDUCTIBLE_AND_LIMIT_PERCENT_TIV = 4
CALCRULE_ID_DEDUCTIBLE_AND_LIMIT_PERCENT_LOSS = 5
CALCRULE_ID_DEDUCTIBLE_PERCENT_TIV = 6
CALCRULE_ID_LIMIT_AND_MAX_DEDUCTIBLE = 7
CALCRULE_ID_LIMIT_AND_MIN_DEDUCTIBLE = 8
CALCRULE_ID_LIMIT_WITH_DEDUCTIBLE_PERCENT_LIMIT = 9
CALCRULE_ID_MAX_DEDUCTIBLE = 10
CALCRULE_ID_MIN_DEDUCTIBLE = 11
CALCRULE_ID_DEDUCTIBLE_ONLY = 12
CALCRULE_ID_MAIN_AND_MAX_DEDUCTIBLE = 13
CALCRULE_ID_LIMIT_ONLY = 14
CALCRULE_ID_LIMIT_PERCENT_LOSS = 15
CALCRULE_ID_DEDUCTIBLE_PERCENT_LOSS = 16
CALCRULE_ID_DEDUCTIBLE_PERCENT_LOSS_ATTACHMENT_LIMIT_AND_SHARE = 17
CALCRULE_ID_DEDUCTIBLE_PERCENT_TIV_ATTACHMENT_LIMIT_AND_SHARE = 18
CALCRULE_ID_DEDUCTIBLE_PERCENT_LOSS_WITH_MIN_AND_MAX = 19
CALCRULE_ID_REVERSE_FRANCHISE_DEDUCTIBLE = 20
CALCRULE_ID_SHARE_AND_LIMIT = 21
CALCRULE_ID_QUOTA_SHARE = 22
CALCRULE_ID_OCCURRENCE_LIMIT_AND_SHARE = 23
CALCRULE_ID_OCCURRENCE_CATASTROPHE_EXCESS_OF_LOSS = 24
CALCRULE_ID_FACULTATIVE_WITH_POLICY_SHARE = 25

NO_ALLOCATION_ALLOC_ID = 0
ALLOCATE_TO_ITEMS_BY_GUL_ALLOC_ID = 1
ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID = 2

BUILDING_COVERAGE_TYPE_ID = 1
OTHER_BUILDING_COVERAGE_TYPE_ID = 2
CONTENTS_COVERAGE_TYPE_ID = 3
TIME_COVERAGE_TYPE_ID = 4
COVERAGE_TYPES = [
    BUILDING_COVERAGE_TYPE_ID,
    OTHER_BUILDING_COVERAGE_TYPE_ID,
    CONTENTS_COVERAGE_TYPE_ID,
    TIME_COVERAGE_TYPE_ID]

PERIL_WIND = 1
PERILS = [PERIL_WIND]

GUL_INPUTS_FILES = [
    'coverages',
    'gulsummaryxref',
    'items']

IL_INPUTS_FILES = [
    'fm_policytc',
    'fm_profile',
    'fm_programme',
    'fm_xref',
    'fmsummaryxref']

OPTIONAL_INPUTS_FILES = [
    'events']

CONVERSION_TOOLS = {
    'coverages': 'coveragetobin',
    'events': 'evetobin',
    'fm_policytc': 'fmpolicytctobin',
    'fm_profile': 'fmprofiletobin',
    'fm_programme': 'fmprogrammetobin',
    'fm_xref': 'fmxreftobin',
    'fmsummaryxref': 'fmsummaryxreftobin',
    'gulsummaryxref': 'gulsummaryxreftobin',
    'items': "itemtobin"}


NOT_SET_ID = -1
LARGE_VALUE = 9999999999999


# --- OED constants --------------------------------------------------------- #
RI_INFO_REQUIRED_COLS = [
    'ReinsNumber', 
    #'ReinsPeril', 
    'PlacedPercent',
    'InuringPriority', 
    'ReinsType'
    ]

RI_INFO_DEFAULTS = {
#    'ReinsLayerNumber': '',
    'CededPercent': 1.0,
    'RiskLimit': 0.0,
    'RiskAttachment': 0.0,
    'OccLimit': 0.0,
    'OccAttachment': 0.0,
    'TreatyShare': 1.0}

RI_SCOPE_REQUIRED_COLS = {
    'ReinsNumber', 
    'RiskLevel'
    }

RI_SCOPE_DEFAULTS = {
    'PortNumber': '',
    'AccNumber': '',
    'PolNumber': '',
    'LocGroup': '',
    'LocNumber': '',
    'CedantName': '',
    'ProducerName': '',
    'LOB': '',
    'CountryCode': '',
    'ReinsTag': '',
    'CededPercent': 1.0
}

POLICYITEM_LEVEL = 0
LOCATION_LEVEL = 1
POLICY_LEVEL = 2
ACCOUNT_LEVEL = 3
OCCURRENCE_LEVEL = 4

PASSTHROUGH_NODE_TYPE = 1
NOLOSS_NODE_TYPE = 1

REINS_TYPE_FAC = "FAC"
REINS_TYPE_QUOTA_SHARE = "QS"
REINS_TYPE_SURPLUS_SHARE = "SS"
REINS_TYPE_PER_RISK = "PR"
REINS_TYPE_CAT_XL = "CXL"
REINS_TYPE_AGG_XL = "AXL"
REINS_TYPES = [
    REINS_TYPE_FAC,
    REINS_TYPE_QUOTA_SHARE,
    REINS_TYPE_SURPLUS_SHARE,
    REINS_TYPE_PER_RISK,
    REINS_TYPE_CAT_XL,
    #REINS_TYPE_AGG_XL <-- not implemented yet
]

REINS_RISK_LEVEL_PORTFOLIO = "SEL"
REINS_RISK_LEVEL_LOCATION = "LOC"
REINS_RISK_LEVEL_LOCATION_GROUP = "LGR"
REINS_RISK_LEVEL_POLICY = "POL"
REINS_RISK_LEVEL_ACCOUNT = "ACC"
REINS_RISK_LEVELS = [
    REINS_RISK_LEVEL_LOCATION,
    REINS_RISK_LEVEL_LOCATION_GROUP,
    REINS_RISK_LEVEL_POLICY,
    REINS_RISK_LEVEL_ACCOUNT,
    REINS_RISK_LEVEL_PORTFOLIO,
]


SUPPORTED_RISK_LEVELS = {
    REINS_TYPE_FAC: [REINS_RISK_LEVEL_LOCATION, REINS_RISK_LEVEL_LOCATION_GROUP, REINS_RISK_LEVEL_POLICY, REINS_RISK_LEVEL_ACCOUNT],
    REINS_TYPE_SURPLUS_SHARE: [REINS_RISK_LEVEL_LOCATION, REINS_RISK_LEVEL_LOCATION_GROUP, REINS_RISK_LEVEL_POLICY, REINS_RISK_LEVEL_ACCOUNT],
    REINS_TYPE_PER_RISK: [REINS_RISK_LEVEL_LOCATION, REINS_RISK_LEVEL_LOCATION_GROUP, REINS_RISK_LEVEL_POLICY, REINS_RISK_LEVEL_ACCOUNT],
    REINS_TYPE_CAT_XL: [REINS_RISK_LEVEL_PORTFOLIO],
    REINS_TYPE_QUOTA_SHARE: REINS_RISK_LEVELS
}

# Subset of the fields that are currently used
OED_ACCOUNT_FIELDS = [
    'PortNumber',
    'AccNumber',
    'PolNumber',
    'AccPeril',
    'AccDed6All',
    'AccLimit6All'
]

OED_LOCATION_FIELDS = [
    'AccNumber',
    'LocGroup',
    'LocNumber',
    'LocDed6All',
    'LocLimit6All',
    'BuildingTIV',
    'OtherTIV',
    'ContentsTIV',
    'BITIV'
]

OED_REINS_INFO_FIELDS = [
    'ReinsNumber',
#    'ReinsLayerNumber',
    'CededPercent',
    'RiskLimit',
    'RiskAttachment',
    'OccLimit',
    'OccAttachment',
    'InuringPriority',
    'ReinsType',
    'PlacedPercent',
    'TreatyShare'
]

OED_REINS_SCOPE_FIELDS = [
    'ReinsNumber',
    'PortNumber',
    'AccNumber',
    'PolNumber',
    'LocGroup',
    'LocNumber',
    'CedantName',
    'ProducerName',
    'LOB',
    'CountryCode',
    'ReinsTag',    
    'RiskLevel',
    'CededPercent'
]

InuringLayer = namedtuple(
    "InuringLayer",
    "inuring_priority reins_numbers is_valid validation_messages")
Item = namedtuple(
    "Item", "item_id coverage_id areaperil_id vulnerability_id group_id")
Coverage = namedtuple(
    "Coverage", "coverage_id tiv")
FmProgramme = namedtuple(
    "FmProgramme", "from_agg_id level_id to_agg_id")
FmProfile = namedtuple(
    "FmProfile", "profile_id calcrule_id deductible1 deductible2 deductible3 attachment limit share1 share2 share3")
FmPolicyTc = namedtuple(
    "FmPolicyTc", "layer_id level_id agg_id profile_id")
GulSummaryXref = namedtuple(
    "GulSummaryXref", "coverage_id summary_id summaryset_id")
FmSummaryXref = namedtuple(
    "FmSummaryXref", "output_id summary_id summaryset_id")
FmXref = namedtuple(
    "FmXref", "output_id agg_id layer_id")

XrefDescription = namedtuple(
    "Description", ("xref_id portfolio_number policy_number account_number location_number location_group " +\
    "cedant_name producer_name lob country_code reins_tag coverage_type_id peril_id tiv"))
GulRecord = namedtuple(
    "GulRecord", "event_id item_id sidx loss")

def get_no_loss_profile(profile_id):
    return FmProfile(
        profile_id=profile_id,
        calcrule_id=CALCRULE_ID_LIMIT_ONLY,
        deductible1=0,  # Not used
        deductible2=0,  # Not used
        deductible3=0,  # Not used
        attachment=0,   # Not used
        limit=0,
        share1=0,       # Not used
        share2=0,       # Not used
        share3=0        # Not used
    )


def get_pass_through_profile(profile_id):
    return FmProfile(
        profile_id=profile_id,
        calcrule_id=CALCRULE_ID_DEDUCTIBLE_ONLY,
        deductible1=0,
        deductible2=0,  # Not used
        deductible3=0,  # Not used
        attachment=0,   # Not used
        limit=0,        # Not used
        share1=0,       # Not used
        share2=0,       # Not used
        share3=0        # Not used
    )


def get_profile(
    profile_id,
    deductible=0,
    attachment=0,
    limit=0,
    share=1.0
):

    if limit == 0:
        limit = LARGE_VALUE

    return FmProfile(
        profile_id=profile_id,
        calcrule_id=CALCRULE_ID_DEDUCTIBLE_ATTACHMENT_LIMIT_AND_SHARE,
        deductible1=deductible,
        deductible2=0,  # Not used
        deductible3=0,  # Not used
        attachment=attachment,
        limit=limit,
        share1=share,
        share2=0,       # Not used
        share3=0        # Not used
    )


def _value_is_empty(value):
    return (
        value == "" or
        value == None or
        math.isnan(value))

def get_reinsurance_profile(
    profile_id,
    attachment=0,
    limit=0,
    ceded=1.0,
    placement=1.0
):

    if _value_is_empty(attachment):
        attachment = 0
    if _value_is_empty(limit) or limit == 0:
        limit = LARGE_VALUE
    if _value_is_empty(ceded):
        ceded = 1.0
    if _value_is_empty(placement):
        placement = 1.0

    return FmProfile(
        profile_id=profile_id,
        calcrule_id=CALCRULE_ID_OCCURRENCE_CATASTROPHE_EXCESS_OF_LOSS,
        deductible1=0,          # Not used
        deductible2=0,          # Not used
        deductible3=0,          # Not used
        attachment=attachment,
        limit=limit,
        share1=ceded,
        share2=placement,       # PlacementPercent
        share3=1.0              # Not used
    )


def get_occlim_profile(
    profile_id,
    attachment=0,
    limit=0,
    ceded=1.0,
    placement=1.0
):

    if limit == 0:
        limit = LARGE_VALUE

    return FmProfile(
        profile_id=profile_id,
        calcrule_id=CALCRULE_ID_OCCURRENCE_LIMIT_AND_SHARE,
        deductible1=0,      # Not used
        deductible2=0,      # Not used
        deductible3=0,      # Not used
        attachment=attachment,
        limit=limit,
        share1=0,           # Not used
        share2=placement,   # Not used
        share3=1.0          # Not used
    )
