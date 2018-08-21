# -*- coding: utf-8 -*-

__all__ = [
    'load_oed_dfs',
#   'OedValidator'    
]


import pandas as pd
import os
from collections import namedtuple


# TODO - add validator 
class OedValidator(object):

    def __init__(self, ri_info_rules, ri_scope_rules):
        self.rules_ode_scope = ri_info_rules
        self.rules_ode_info = ri_scope_rules
        pass

    def _has_reins_type(self, reins_info_df, reins_type):
        '''
        Is there any <reins_type>?
        '''
        return not reins_info_df[reins_info_df.ReinsType == reins_type].empty

    def _unique_reins(self, reins_info_df):
        '''
        check if only one reins type exisits  
        '''
        return (len(reins_info_df.ReinsType.unique()) == 1)

    def _links_valid(df_src, column_name, df_dest):
        '''
        Check that all unique values in df_src[column_name] map to df_dest[column_name]
        '''
        src_values = df_src[column_name].unique().tolist()
        return df_dest.isin({column_name: src_values}).all()

    def _all_links_valid(self, scope_df, account_df, location_df):
        return (
            self._links_valid(scope_df, "AccountNumber",   account_df),
            self._links_valid(scope_df, "PolicyNumber",    account_df),
            self._links_valid(scope_df, "AccountNumber",   location_df),
            self._links_valid(scope_df, "LocationNumber",  location_df),
        )

    def _all_scope_non_specific(self,scope_df):
        return scope_df[['accountnumber', 
                         'policynumber', 
                         'locationnumber'
                         ]].isnull().all().all()

    def _all_scope_specific(self,scope_df):
        return scope_df[['accountnumber', 
                         'policynumber', 
                         'locationnumber'
                         ]].notnull().all().all()


    def validate(self, account_df, location_df, ri_info_df, ri_scope_df):
        '''
        Validate OED resinurance structure before running calculations.
        '''

        main_is_valid = True
        inuring_layers = {}
        for inuring_priority in range(1, ri_info_df['InuringPriority'].max() + 1):
            inuring_priority_ri_info_df = ri_info_df[ri_info_df.InuringPriority == inuring_priority]
            if inuring_priority_ri_info_df.empty:
                continue

            is_valid = True
            validation_messages = []

            inuring_scope_ids = inuring_priority_ri_info_df.ReinsNumber.tolist()
            inuring_scopes = [ri_scope_df[ri_scope_df.ReinsNumber == ID] for ID in inuring_scope_ids] 

            for ri_type in REINS_TYPES:
                #CHECK - ri_type is supported 
                if self._has_reins_type(inuring_priority_ri_info_df,ri_type):
                    #CHECK - only single ri_type is set per inuring priority 
                    if not self._unique_reins(inuring_priority_ri_info_df):
                        is_valid = False
                        validation_messages.append(
                            "{} cannot be combined with other reinsurance types".format(ri_type))     

                    for scope_df in inuring_scopes:
                        scope_risk_levels = scope_df.RiskLevel.unique()
                        risk_level_id = scope_risk_levels[0]

                        # CHECK - each scope only has one risk level type 
                        if len(scope_risk_levels) is not 1:
                            is_valid = False
                            validation_messages.append(
                                "Mix of risk levels in a single reinsurance scope")
                            continue
                   
                        # CHECK - Risk level is supported 
                        if risk_level_id not in REINS_RISK_LEVELS:
                            is_valid = False
                            validation_messages.append(
                                "Unsupported risk level, {}".format(' '.join(risk_level_id)))

                        # CHECK - that scope is not specific for SS
                        if ri_type in [REINS_TYPE_SURPLUS_SHARE] and not self._scope_specific(scope_df):
                            is_valid = False
                            validation_messages.append(
                                "SS cannot have non-specific scopes")

                        # CHECK - that scope is all specific for QS
                        if ri_type in [REINS_TYPE_QUOTA_SHARE] and not self._scope_non_specific(scope_df):  
                            is_valid = False
                            validation_messages.append(
                                "QS cannot have specific scopes set")

                        # CHECK - all links in scope connect to rows in account/location    
                        if not self._all_links_valid(scope_df,  account_df, location_df):    
                            is_valid = False
                            validation_messages.append(
                                "Non-linking scopes between ri_scope and (ACC,LOC) files")

                    else:
                        is_valid = False
                        validation_messages.append("{} not implemented".format(ri_type))
                        continue

            if not is_valid:
                main_is_valid = False

            inuring_layers[inuring_priority] = InuringLayer(
                inuring_priority=inuring_priority,
                reins_numbers=inuring_priority_ri_info_df.ReinsNumber,
                is_valid=is_valid,
                validation_messages=validation_messages
            )
        return (main_is_valid, inuring_layers)

# -----------------------------------------------------------------------------#
#
def load_oed_dfs(oed_dir, show_all=False):
    """
    Load OED data files.
    """

    do_reinsurance = True
    if oed_dir is not None:
        if not os.path.exists(oed_dir):
            print("Path does not exist: {}".format(oed_dir))
            exit(1)
        # Account file
        oed_account_file = os.path.join(oed_dir, "account.csv")
        if not os.path.exists(oed_account_file):
            print("Path does not exist: {}".format(oed_account_file))
            exit(1)
        account_df = pd.read_csv(oed_account_file)

        # Location file
        oed_location_file = os.path.join(oed_dir, "location.csv")
        if not os.path.exists(oed_location_file):
            print("Path does not exist: {}".format(oed_location_file))
            exit(1)
        location_df = pd.read_csv(oed_location_file)

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
            ri_info_df = pd.read_csv(oed_ri_info_file)
            ri_scope_df = pd.read_csv(oed_ri_scope_file)
        else:
            print("Both reinsurance files must exist: {} {}".format(
                oed_ri_info_file, oed_ri_scope_file))
        if not show_all:
            account_df = account_df[OED_ACCOUNT_FIELDS].copy()
            location_df = location_df[OED_LOCATION_FIELDS].copy()
            if do_reinsurance:
                ri_info_df = ri_info_df[OED_REINS_INFO_FIELDS].copy()
                ri_scope_df = ri_scope_df[OED_REINS_SCOPE_FIELDS].copy()
    return (account_df, location_df, ri_info_df, ri_scope_df, do_reinsurance)



# --- Ktools constant ------------------------------------------------------- #

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
REINS_TYPE_CAT_XL = "CAT XL"
REINS_TYPE_AGG_XL = "XL"
REINS_TYPES = [
    REINS_TYPE_FAC,
    REINS_TYPE_QUOTA_SHARE,
    REINS_TYPE_SURPLUS_SHARE,
    REINS_TYPE_PER_RISK,
    REINS_TYPE_CAT_XL,
#    REINS_TYPE_AGG_XL, <-- not implemented yet
]

REINS_RISK_LEVEL_PORTFOLIO = "SEL"
REINS_RISK_LEVEL_LOCATION = "LOC"
#REINS_RISK_LEVEL_LOCATION_GROUP = "Location Group"
REINS_RISK_LEVEL_POLICY = "POL"
REINS_RISK_LEVEL_ACCOUNT = "ACC"
REINS_RISK_LEVELS = [
    REINS_RISK_LEVEL_LOCATION,
    REINS_RISK_LEVEL_POLICY,
    REINS_RISK_LEVEL_ACCOUNT,
    REINS_RISK_LEVEL_PORTFOLIO,
]


# Subset of the fields that are currently used
OED_ACCOUNT_FIELDS = [
    'PortfolioNumber',
    'AccountNumber',
    'PolicyNumber',
    'PerilCode',
    'Ded6',
    'Limit6'
]

OED_LOCATION_FIELDS = [
    'AccountNumber',
    'LocationNumber',
    'Ded6',
    'Limit6',
    'BuildingTIV',
    'OtherTIV',
    'ContentsTIV',
    'BITIV'
]

OED_REINS_INFO_FIELDS = [
    'ReinsNumber',
    'ReinsLayerNumber',
    'CededPercent',
    'RiskLimit',
    'RiskAttachmentPoint',
    'OccLimit',
    'OccurenceAttachmentPoint',
    'InuringPriority',
    'ReinsType',
    'PlacementPercent',
    'TreatyPercent'
]

OED_REINS_SCOPE_FIELDS = [
    'ReinsNumber',
    'PortfolioNumber',
    'AccountNumber',
    'PolicyNumber',
    'LocationNumber',
    'RiskLevel',
    'CededPercent'
]

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
    "Description", ("xref_id policy_number account_number location_number coverage_type_id peril_id tiv"))
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

def get_reinsurance_profile(
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
        calcrule_id=CALCRULE_ID_OCCURRENCE_CATASTROPHE_EXCESS_OF_LOSS,
        deductible1=0,  # Not used
        deductible2=0,  # Not used
        deductible3=0,  # Not used
        attachment=attachment,
        limit=limit,
        share1=ceded,
        share2=placement, # PlacementPercent
        share3=1.0        # Not used
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
        deductible1=0,  # Not used
        deductible2=0,  # Not used
        deductible3=0,  # Not used
        attachment=attachment,
        limit=limit,
        share1=0,         # Not used
        share2=placement, # Not used
        share3=1.0        # Not used
        )
