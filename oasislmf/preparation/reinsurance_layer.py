__all__ = [
    'ReinsuranceLayer',
    'write_files_for_reinsurance'
]

import logging
import os
import shutil

from collections import namedtuple

import numbers
import pandas as pd

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from . import oed

from ..utils.data import get_dataframe

# Metadata about an inuring layer
InuringLayer = namedtuple(
    "InuringLayer",
    "inuring_priority reins_numbers is_valid validation_messages"
)

RiInputs = namedtuple(
    'RiInputs',
    'inuring_priority risk_level ri_inputs'
)

RiLayerInputs = namedtuple(
    'RiLayerInputs',
    'fm_programme fm_profile fm_policytc'
)


def _get_location_tiv(location, coverage_type_id):
    switcher = {
        oed.BUILDING_COVERAGE_TYPE_ID: location.get('BuildingTIV', 0),
        oed.OTHER_BUILDING_COVERAGE_TYPE_ID: location.get('OtherTIV', 0),
        oed.CONTENTS_COVERAGE_TYPE_ID: location.get('ContentsTIV', 0),
        oed.TIME_COVERAGE_TYPE_ID: location.get('BITIV', 0)
    }
    return switcher.get(coverage_type_id, 0)


def _get_ri_inputs(
        items_df,
        coverages_df,
        xref_descriptions_df,
        ri_info_df,
        ri_scope_df):

    ri_inputs = []
    for inuring_priority in range(1, ri_info_df['InuringPriority'].max() + 1):
        # Filter the reinsNumbers by inuring_priority
        reins_numbers = ri_info_df[ri_info_df['InuringPriority'] == inuring_priority].ReinsNumber.tolist()
        risk_level_set = set(ri_info_df[ri_info_df['ReinsNumber'].isin(reins_numbers)].RiskLevel)

        for risk_level in oed.REINS_RISK_LEVELS:
            if risk_level not in risk_level_set:
                continue

            ri_inputs.append(
                RiInputs(
                    inuring_priority=inuring_priority,
                    risk_level=risk_level,
                    ri_inputs=_generate_inputs_for_reinsurance_risk_level(
                        inuring_priority,
                        items_df,
                        coverages_df,
                        xref_descriptions_df,
                        ri_info_df,
                        ri_scope_df,
                        risk_level)))
    return ri_inputs


@oasis_log
def write_files_for_reinsurance(
        gul_inputs_df,
        xref_descriptions_df,
        ri_info_df,
        ri_scope_df,
        fm_xref_fp,
        output_dir,
        store_tree=False):
    """
    Generate files for reinsurance.
    """
    inuring_metadata = {}

    items_df = gul_inputs_df.loc[:, ['item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id']].drop_duplicates()
    coverages_df = gul_inputs_df.loc[:, ['coverage_id', 'tiv']].drop_duplicates()
    ri_inputs = _get_ri_inputs(
        items_df,
        coverages_df,
        xref_descriptions_df,
        ri_info_df,
        ri_scope_df)

    reinsurance_index = 1

    for ri_input in ri_inputs:

        ri_output_dir = os.path.join(output_dir, "RI_{}".format(reinsurance_index))
        if os.path.exists(ri_output_dir):
            shutil.rmtree(ri_output_dir)
        os.makedirs(ri_output_dir)

        ri_input.ri_inputs.fm_programme.to_csv(
            os.path.join(ri_output_dir, "fm_programme.csv"), index=False)
        ri_input.ri_inputs.fm_profile.to_csv(
            os.path.join(ri_output_dir, "fm_profile.csv"), index=False)
        ri_input.ri_inputs.fm_policytc.to_csv(
            os.path.join(ri_output_dir, "fm_policytc.csv"), index=False)

        fm_xref_df = get_dataframe(fm_xref_fp)
        fm_xref_df['agg_id'] = range(1, 1 + len(fm_xref_df))
        # Net losses across all layers is associated to the max layer ID.
        fm_xref_df['layer_id'] = ri_input.ri_inputs.fm_policytc['layer_id'].max()
        fm_xref_df.to_csv(
            os.path.join(ri_output_dir, "fm_xref.csv"), index=False)

        inuring_metadata[reinsurance_index] = {
            'inuring_priority': ri_input.inuring_priority,
            'risk_level': ri_input.risk_level,
            'directory': ri_output_dir
        }

        reinsurance_index = reinsurance_index + 1

    return inuring_metadata


def _generate_inputs_for_reinsurance_risk_level(
        inuring_priority,
        items_df,
        coverages_df,
        xref_descriptions_df,
        ri_info_df,
        ri_scope_df,
        risk_level):
    """
    Generate files for a reinsurance risk level.
    """
    reins_numbers_1 = ri_info_df[
        ri_info_df['InuringPriority'] == inuring_priority].ReinsNumber
    if reins_numbers_1.empty:
        return None
    reins_numbers_2 = ri_info_df[
        ri_info_df.isin({"ReinsNumber": reins_numbers_1.tolist()}).ReinsNumber
        & (ri_info_df.RiskLevel == risk_level)
    ].ReinsNumber
    if reins_numbers_2.empty:
        return None

    ri_info_inuring_priority_df = ri_info_df[ri_info_df.isin(
        {"ReinsNumber": reins_numbers_2.tolist()}).ReinsNumber]
    output_name = "ri_{}_{}".format(inuring_priority, risk_level)
    reinsurance_layer = ReinsuranceLayer(
        name=output_name,
        ri_info_df=ri_info_inuring_priority_df,
        ri_scope_df=ri_scope_df,
        items_df=items_df,
        coverages_df=coverages_df,
        xref_descriptions_df=xref_descriptions_df,
        risk_level=risk_level
    )

    reinsurance_layer.generate_oasis_structures()
    return RiLayerInputs(
        fm_programme=reinsurance_layer.fmprogrammes_df,
        fm_profile=reinsurance_layer.fmprofiles_df,
        fm_policytc=reinsurance_layer.fm_policytcs_df
    )


class ReinsuranceLayer(object):
    """
    Generates ktools inputs and runs financial module for a reinsurance structure.
    """

    def __init__(
        self,
        name, ri_info_df, ri_scope_df, items_df, coverages_df,
        xref_descriptions_df, risk_level, fmsummaryxref_df=pd.DataFrame(),
        gulsummaryxref_df=pd.DataFrame(), logger=None
    ):

        self.logger = logger or logging.getLogger()
        self.name = name

        self.coverages_df = coverages_df
        self.items_df = items_df
        self.xref_descriptions_df = xref_descriptions_df
        self.fmsummaryxref_df = fmsummaryxref_df
        self.gulsummaryxref_df = gulsummaryxref_df

        self.item_ids = list()
        self.item_tivs = list()
        self.fmprogrammes_df = pd.DataFrame()
        self.fmprofiles_df = pd.DataFrame()
        self.fm_policytcs_df = pd.DataFrame()

        self.risk_level = risk_level

        self.ri_info_df = ri_info_df
        self.ri_scope_df = ri_scope_df

    def _is_valid_id(self, id_to_check):
        is_valid = (
            self._is_defined(id_to_check) and
            (
                (isinstance(id_to_check, str) and id_to_check != "") or
                (isinstance(id_to_check, numbers.Number) and id_to_check > 0)
            )
        )
        return is_valid

    def _get_reins_type_fields(self, risk_level=None, all_fields=False):

        if all_fields:
            return ['portnumber', 'accnumber', 'polnumber', 'locgroup', 'locnumber']

        if risk_level is None:
            risk_level = self.risk_level

        fields = []
        if risk_level == oed.REINS_RISK_LEVEL_PORTFOLIO:
            fields = ['portnumber']
        elif risk_level == oed.REINS_RISK_LEVEL_ACCOUNT:
            fields = ['portnumber', 'accnumber']
        elif risk_level == oed.REINS_RISK_LEVEL_POLICY:
            fields = ['portnumber', 'accnumber', 'polnumber']
        elif risk_level == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
            fields = ['locgroup']
        elif risk_level == oed.REINS_RISK_LEVEL_LOCATION:
            fields = ['portnumber', 'accnumber', 'locnumber']
        else:
            raise OasisException(f"Unknown risk level: {risk_level}")
        return fields

    def _get_valid_fields(self, fields):

        valid_fields = []
        for field in fields:
            valid_field = field + '_valid'
            valid_fields.append(valid_field)
        return valid_fields

    def _match_exact(self, df_, ri_df_, fields=None):

        if fields == None:
            fields = []

        df_ = df_.merge(
            ri_df_[fields + ['layer_id', 'level_id', 'profile_id']].drop_duplicates(),
            how='left', on=fields+['layer_id', 'level_id'], suffixes=['', '_y']
        )
        df_['profile_id'] = df_['profile_id'].where(
            df_['profile_id_y'].isna(), df_['profile_id_y']
        )

        return df_['profile_id'].to_list()

    def _match_portfolio(self, df_, ri_df_):

        fields = self._get_reins_type_fields(oed.REINS_RISK_LEVEL_PORTFOLIO)
        valid_fields = self._get_valid_fields(fields)

        df_['profile_id'] = self._match_exact(
            df_=df_, ri_df_=ri_df_[ri_df_[valid_fields[0]] == True],
            fields=fields
        )

        df_['profile_id'] = self._match_exact(
            df_=df_, ri_df_=ri_df_[ri_df_[valid_fields[0]] == False]
        )

        return df_['profile_id'].to_list()

    def _match_account(self, df_, ri_df_):

        fields = self._get_reins_type_fields(oed.REINS_RISK_LEVEL_ACCOUNT)
        valid_fields = self._get_valid_fields(fields)

        df_['profile_id'] = self._match_exact(
            df_=df_,
            ri_df_=ri_df_[(ri_df_[valid_fields[0]] == True) & (ri_df_[valid_fields[1]] == True)],
            fields=fields
        )

        df_['profile_id'] = self._match_portfolio(
            df_=df_, ri_df_=ri_df_[ri_df_[valid_fields[1]] == False]
        )

        return df_['profile_id'].to_list()

    def _match_policy_or_location(self, df_, ri_df_):

        fields = self._get_reins_type_fields()
        valid_fields = self._get_valid_fields(fields)

        df_['profile_id'] = self._match_exact(
            df_=df_,
            ri_df_=ri_df_[(ri_df_[valid_fields[0]] == True) & (ri_df_[valid_fields[1]] == True) & (ri_df_[valid_fields[2]] == True)],
            fields=fields
        )

        df_['profile_id'] = self._match_account(
            df_=df_, ri_df_=ri_df_[ri_df_[valid_fields[2]] == False]
        )

        return df_['profile_id'].to_list()

    def _match_location_group(self, df_, ri_df_):

        fields = self._get_reins_type_fields()
        valid_fields = self._get_valid_fields(fields)

        df_['profile_id'] = self._match_exact(
            df_=df_, ri_df_=ri_df_[ri_df_[valid_fields[0]] == True],
            fields=fields
        )

        return df_['profile_id'].to_list()

    def _match_row(self, row):
        match = True
        if match and row['portnumber_valid']:
            match = row['portnumber_x'] == row['portnumber_y']
        if match and row['accnumber_valid']:
            match = row['accnumber_x'] == row['accnumber_y']
        if match and row['polnumber_valid']:
            match = row['polnumber_x'] == row['polnumber_y']
        if match and row['locgroup_valid']:
            match = row['locgroup_x'] == row['locgroup_y']
        if match and row['locnumber_valid']:
            match = row['locnumber_x'] == row['locnumber_y']

        return match

    def _get_risk_level_profile_ids(self, df_, ri_df_, exact=False):
        profile_ids = []
        if df_.empty:
            return profile_ids

        if exact:
            profile_ids = self._match_exact(
                df_=df_, ri_df_=ri_df_,
                fields=self._get_reins_type_fields()
            )
        elif self.risk_level == oed.REINS_RISK_LEVEL_PORTFOLIO:
            profile_ids = self._match_portfolio(df_, ri_df_)
        elif self.risk_level == oed.REINS_RISK_LEVEL_ACCOUNT:
            profile_ids = self._match_account(df_, ri_df_)
        elif self.risk_level == oed.REINS_RISK_LEVEL_POLICY:
            profile_ids = self._match_policy_or_location(df_, ri_df_)
        elif self.risk_level == oed.REINS_RISK_LEVEL_LOCATION:
            profile_ids = self._match_policy_or_location(df_, ri_df_)
        elif self.risk_level == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
            profile_ids = self._match_location_group(df_, ri_df_)
        else:
            raise OasisException(f"Unknown risk level: {self.risk_level}")

        return profile_ids

    def _get_filter_level_profile_ids(self, df_, ri_df_):

        if len(df_) == 0 or len(ri_df_) == 0:
            return []

        df_['idx'] = df_.index
        df_ = df_.merge(ri_df_, how='inner', on='layer_id')
        df_['match'] = df_.apply(lambda row: self._match_row(row), axis=1)
        filter_idx = df_[df_['match'] == True]['idx'].drop_duplicates().to_list()

        return filter_idx

    def _is_defined(self, num_to_check):
        # If the value = NaN it will return False
        return num_to_check == num_to_check

    def _check_ri_df_row(self, row):
        # For some treaty types the scope filter must match exactly
        okay = True
        if row.reinstype == oed.REINS_TYPE_FAC or row.reinstype == oed.REINS_TYPE_SURPLUS_SHARE:
            if row.risklevel == oed.REINS_RISK_LEVEL_ACCOUNT:
                okay = self._is_valid_id(row.accnumber) and not self._is_valid_id(row.polnumber) and not self._is_valid_id(row.locnumber)
            elif row.risklevel == oed.REINS_RISK_LEVEL_POLICY:
                okay = self._is_valid_id(row.accnumber) and self._is_valid_id(row.polnumber) and not self._is_valid_id(row.locnumber)
            elif row.risklevel == oed.REINS_RISK_LEVEL_LOCATION:
                okay = self._is_valid_id(row.accnumber) and self._is_valid_id(row.locnumber)
            elif row.risklevel == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
                okay = self._is_valid_id(row.locgroup)
        return okay

    def _log_dataframe(self, df_dict):
        container_name = 'df_dict'
        if isinstance(df_dict, dict):
            container_name += '.items()'

        for df_name, df_ in eval(container_name):
            self.logger.debug(f'{df_name}: {self.name}:')
            self.logger.debug(df_)

    LOCATION_RISK_LEVEL = 2

    def _get_xref_df(self):
        """
        Build the cross-reference dataframe, which serves as a representation
        of the insurance programme depending on the reinsurance risk level.
        Dataframes for programme, risk, filter and items levels are created.
        The fields agg_id, level_id and to_agg_id (agg_id_to), which are used
        to construct the FM Programmes structure, are assigned. The
        aforementioned dataframes are concatenated to form a single dataframe
        called xref_df, which is returned. The returned dataframe features the
        fields necessary for the assignment of profile IDs.
        """
        risk_level_id = self.LOCATION_RISK_LEVEL + 1
        program_node_level_id = risk_level_id + 1

        if self.risk_level == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
            xref_descriptions = self.xref_descriptions_df.sort_values(
                by=["locgroup", "portnumber", "accnumber", "polnumber", "locnumber"])
        elif self.risk_level == oed.REINS_RISK_LEVEL_LOCATION:
            xref_descriptions = self.xref_descriptions_df.sort_values(
                by=["portnumber", "accnumber", "locnumber", "polnumber"])
        else:
            xref_descriptions = self.xref_descriptions_df.sort_values(
                by=["portnumber", "accnumber", "polnumber", "locnumber"])

        df_levels = []
        # Programme level
        programme_level_df = pd.DataFrame(
            {'agg_id': 1, 'level_id': program_node_level_id, 'agg_id_to': 0},
            index=[0]
        )
        df_levels.append('programme_level')

        # Risk level
        fields = self._get_reins_type_fields()
        risk_level_df = pd.DataFrame()
        risk_level_df = xref_descriptions.drop_duplicates(
            subset=fields, keep='first'
        ).reset_index(drop=True)
        risk_level_df['agg_id'] = risk_level_df.index + 1
        risk_level_df['level_id'] = risk_level_id
        risk_level_df['agg_id_to'] = 1
        df_levels.append('risk_level')

        # Filter level
        filter_level_fields = [
            'portnumber', 'accnumber', 'polnumber', 'locnumber', 'locgroup'
        ]
        filter_level_df = xref_descriptions.drop_duplicates(
            subset=filter_level_fields
        ).reset_index(drop=True)
        filter_level_df['agg_id'] = filter_level_df.index + 1
        filter_level_df['level_id'] = 2
        filter_level_df = filter_level_df.merge(
            risk_level_df[fields + ['agg_id']], how='left', on=fields,
            suffixes=['', '_to']
        )
        df_levels.append('filter_level')

        # Item level
        item_level_df = xref_descriptions.reset_index(drop=True)
        item_level_df['agg_id'] = item_level_df['output_id']
        item_level_df['level_id'] = 1
        item_level_df = item_level_df.merge(
            filter_level_df[filter_level_fields + ['agg_id']], how='left',
            on=filter_level_fields, suffixes=['', '_to']
        )
        df_levels.append('items_level')

        df_list = [
            programme_level_df, risk_level_df, filter_level_df, item_level_df
        ]

        if self.logger:
            self._log_dataframe(zip(df_levels, df_list))

        xref_df = pd.concat(df_list, ignore_index=True)

        return xref_df

    def _get_risk_level_id(self):
        risk_level_id = 3
        return risk_level_id

    def _get_filter_level_id(self):
        risk_level_id = 2
        return risk_level_id

    def _get_new_profiles_dataframe(self, profiles, fmprofiles_df):
        profiles_df = pd.DataFrame(
            [
                [
                    profile_id, calcrule_id, deductible1, deductible2,
                    deductible3, attachment, limit, share1, share2, share3
                ] for profile_id, calcrule_id, deductible1, deductible2,
                deductible3, attachment, limit, share1, share2,
                share3 in profiles.values
            ],
            columns=profiles.values[0]._fields
        )
        fmprofiles_df = pd.concat(
            [fmprofiles_df, profiles_df], ignore_index=True
        )

        return fmprofiles_df

    def _get_profiles(
        self, ri_df_, fmprofiles_df, attachment='None', limit='None',
        ceded='None', placement='None'
    ):
        profiles = ri_df_.apply(
            lambda row: oed.get_reinsurance_profile(
                row['profile_id'], attachment=eval(attachment),
                limit=eval(limit), ceded=eval(ceded), placement=eval(placement)
            ), axis=1
        ).drop_duplicates()
        if len(profiles) > 0:
            fmprofiles_df = self._get_new_profiles_dataframe(
                profiles, fmprofiles_df
            )

        return fmprofiles_df

    def _get_occlimit_profiles(
        self, occlimit_df_, fmprofiles_df, limit='None', placement='None'
    ):
        profiles = occlimit_df_.apply(
            lambda row: oed.get_occlim_profile(
                row['profile_id'], limit=eval(limit), placement=eval(placement)
            ), axis=1
        ).drop_duplicates()
        if len(profiles) > 0:
            fmprofiles_df = self._get_new_profiles_dataframe(
                profiles, fmprofiles_df
            )

        return fmprofiles_df

    def generate_oasis_structures(self):
        '''
        Create the Oasis structures - FM Programmes, FM Profiles and FM Policy
        TCs - that represent the reinsurance structure.

        The cross-reference dataframe, which serves as a representation of the
        insurance programme depending on the reinsurance risk level, is built.
        With the exception of facultative contracts, each contract is a
        separate layer. Profile IDs for the risk and filter levels are created
        using the merged reinsurance scope and info dataframes. These profile
        IDs are assigned according to some combination of the fields
        portnumber, accnumber, polnumber, locgroup and locnumber, dependent on
        reinsurance risk level. Individual programme level profile IDs are
        assigned for each row of the reinsurance info dataframe. Finally, the
        Oasis structure is written out.
        '''

        fmprofiles_list = list()

        profile_id = 1
        nolossprofile_id = profile_id
        fmprofiles_list.append(
            oed.get_no_loss_profile(nolossprofile_id))
        profile_id = profile_id + 1
        passthroughprofile_id = profile_id
        fmprofiles_list.append(
            oed.get_pass_through_profile(passthroughprofile_id))
        fmprofiles_df = pd.DataFrame(fmprofiles_list)

        node_layer_profile_map = {}

        self.logger.debug('No loss and passthrough profiles:')
        self.logger.debug(fmprofiles_list)

        # Get cross-reference dataframe which shall be used to build profile map
        # FM Programmes fields agg_id, level_id and to_agg_id are assigned here
        xref_df = self._get_xref_df()

        # Assign default profile IDs
        xref_df['profile_id'] = nolossprofile_id
        xref_df['profile_id'] = xref_df['profile_id'].where(
            (xref_df['level_id'] == self._get_filter_level_id()) | (xref_df['level_id'] == self._get_risk_level_id()),
            passthroughprofile_id
        )

        # Merge RI info and scope dataframes, and assign layers
        # Use as few layers as possible for FAC
        # Otherwise separate layers for each contract
        self.logger.debug(
            'Merging RI info and scope dataframes and assigning layers'
        )
        fields = self._get_reins_type_fields()
        ri_df = self.ri_info_df[self.ri_info_df['RiskLevel'] == self.risk_level].merge(
            self.ri_scope_df,
            on='ReinsNumber', suffixes=['', '_scope']
        )
        ri_df['layer_id'] = 0
        ri_df.columns = ri_df.columns.str.lower()
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_FAC, 'layer_id'] = ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_FAC].groupby(fields).cumcount() + 1
        ri_info_no_fac = self.ri_info_df[self.ri_info_df['ReinsType'] != oed.REINS_TYPE_FAC].reset_index(drop=True)
        ri_info_no_fac.columns = ri_info_no_fac.columns.str.lower()
        ri_info_no_fac['layer_id'] = ri_info_no_fac.index + 1 + ri_df['layer_id'].max()
        ri_df = ri_df.merge(ri_info_no_fac, how='left', on=ri_info_no_fac.columns.to_list()[:-1], suffixes=['', '_y'])
        ri_df['layer_id'] = ri_df['layer_id'].where(ri_df['layer_id_y'].isna(), ri_df['layer_id_y'])
        ri_df = ri_df.drop('layer_id_y', axis=1)
        del(ri_info_no_fac)
        ri_df['valid_row'] = ri_df.apply(lambda row: self._check_ri_df_row(row), axis=1)
        if ri_df['valid_row'].all() == False:
            raise OasisException(
                f'Invalid combination of Risk Level and Reinsurance Type. Please check scope file:\n{ri_df[ri_df["valid_row"] == False]}'
            )
        ri_df = ri_df.drop('valid_row', axis=1)

        profile_maps = [xref_df.copy() for i in range(ri_df['layer_id'].max())]
        for i, df in enumerate(profile_maps):
            df['layer_id'] = i + 1
        profile_map_df = pd.concat(profile_maps, ignore_index=True)

        # Create risk level and filter level profile IDs
        ri_df['level_id'] = 0
        ri_df['profile_id'] = 0
        # Facultative profile IDs
        self.logger.debug('Creating risk level and filter level profile IDs:')
        self.logger.debug(f'{oed.REINS_TYPE_FAC} profiles...')
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_FAC, 'level_id'] = self._get_risk_level_id()
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_FAC, 'profile_id'] = pd.factorize(pd._lib.fast_zip([
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_FAC]['riskattachment'].values,
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_FAC]['risklimit'].values,
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_FAC]['cededpercent'].values,
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_FAC]['placedpercent'].values
        ]))[0] + 1 + fmprofiles_df['profile_id'].max()
        fmprofiles_df = self._get_profiles(
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_FAC],
            fmprofiles_df=fmprofiles_df, attachment="row['riskattachment']",
            limit="row['risklimit']", ceded="row['cededpercent']",
            placement="row['placedpercent']"
        )
        # Per Risk profile IDs
        self.logger.debug(f'{oed.REINS_TYPE_PER_RISK} profiles...')
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_PER_RISK, 'level_id'] = self._get_risk_level_id()
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_PER_RISK, 'profile_id'] = pd.factorize(pd._lib.fast_zip([
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_PER_RISK]['riskattachment'].values,
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_PER_RISK]['risklimit'].values,
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_PER_RISK]['cededpercent'].values
        ]))[0] + 1 + fmprofiles_df['profile_id'].max()
        fmprofiles_df = self._get_profiles(
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_PER_RISK],
            fmprofiles_df=fmprofiles_df, attachment="row['riskattachment']",
            limit="row['risklimit']", ceded="row['cededpercent']"
        )
        # Quota Share profile IDs
        self.logger.debug(f'{oed.REINS_TYPE_QUOTA_SHARE} profiles...')
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_QUOTA_SHARE, 'level_id'] = self._get_risk_level_id()
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_QUOTA_SHARE, 'profile_id'] = pd.factorize(pd._lib.fast_zip([
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_QUOTA_SHARE]['risklimit'].values,
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_QUOTA_SHARE]['cededpercent'].values
        ]))[0] + 1 + fmprofiles_df['profile_id'].max()
        fmprofiles_df = self._get_profiles(
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_QUOTA_SHARE],
            fmprofiles_df=fmprofiles_df, limit="row['risklimit']",
            ceded="row['cededpercent']"
        )
        # Surplus Share profile IDs
        self.logger.debug(f'{oed.REINS_TYPE_SURPLUS_SHARE} profiles...')
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_SURPLUS_SHARE, 'level_id'] = self._get_risk_level_id()
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_SURPLUS_SHARE, 'profile_id'] = pd.factorize(pd._lib.fast_zip([
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_SURPLUS_SHARE]['riskattachment'].values,
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_SURPLUS_SHARE]['risklimit'].values,
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_SURPLUS_SHARE]['cededpercent_scope'].values
        ]))[0] + 1 + fmprofiles_df['profile_id'].max()
        fmprofiles_df = self._get_profiles(
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_SURPLUS_SHARE],
            fmprofiles_df=fmprofiles_df, attachment="row['riskattachment']",
            limit="row['risklimit']", ceded="row['cededpercent_scope']"
        )
        # Cat XL profile IDs = pass through profile ID
        self.logger.debug(f'{oed.REINS_TYPE_CAT_XL} profiles...')
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_CAT_XL, 'level_id'] = self._get_risk_level_id()
        ri_df.loc[ri_df['reinstype'] == oed.REINS_TYPE_CAT_XL, 'profile_id'] = passthroughprofile_id

        ri_df['profile_id'] = ri_df['profile_id'].astype('int64')

        fields = self._get_reins_type_fields(all_fields=True)
        valid_fields = self._get_valid_fields(fields)
        for field, valid_field in zip(fields, valid_fields):
            ri_df[valid_field] = ri_df.apply(lambda row: self._is_valid_id(row[field]), axis=1)
        if self.logger:
            self._log_dataframe({'ri_df': ri_df})

        # Assign risk level and filter level profile IDs
        self.logger.debug('Assigning risk level and filter level profile IDs:')
        # Facultative profile IDs
        self.logger.debug(f'{oed.REINS_TYPE_FAC} profiles...')
        layer_id_set = set(
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_FAC]['layer_id']
        )
        # Risk level
        # Note that Facultative profiles scope much match the filter exactly
        profile_map_df.loc[
            profile_map_df['layer_id'].isin(layer_id_set), 'profile_id'
        ] = self._get_risk_level_profile_ids(
            df_=profile_map_df[profile_map_df['layer_id'].isin(layer_id_set)],
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_FAC], exact=True
        )
        # Filter level
        profile_map_df.loc[
            (profile_map_df['layer_id'].isin(layer_id_set)) & (profile_map_df['level_id'] == self._get_filter_level_id()),
            'profile_id'
        ] = passthroughprofile_id

        # Per Risk profile IDs
        self.logger.debug(f'{oed.REINS_TYPE_PER_RISK} profiles...')
        layer_id_set = set(
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_PER_RISK]['layer_id']
        )
        # Risk level
        profile_map_df.loc[
            profile_map_df['layer_id'].isin(layer_id_set), 'profile_id'
        ] = self._get_risk_level_profile_ids(
            df_=profile_map_df[profile_map_df['layer_id'].isin(layer_id_set)],
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_PER_RISK]
        )
        # Filter level
        filter_idx = self._get_filter_level_profile_ids(
            df_=profile_map_df[(profile_map_df['layer_id'].isin(layer_id_set)) & (profile_map_df['level_id'] == self._get_filter_level_id())],
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_PER_RISK][fields + valid_fields + ['layer_id']]
        )
        profile_map_df.loc[filter_idx, 'profile_id'] = passthroughprofile_id

        # Quota Share profile IDs
        self.logger.debug(f'{oed.REINS_TYPE_QUOTA_SHARE} profiles...')
        layer_id_set = set(
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_QUOTA_SHARE]['layer_id']
        )
        # Risk level
        profile_map_df.loc[
            profile_map_df['layer_id'].isin(layer_id_set), 'profile_id'
        ] = self._get_risk_level_profile_ids(
            df_=profile_map_df[profile_map_df['layer_id'].isin(layer_id_set)],
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_QUOTA_SHARE]
        )
        # Filter level
        filter_idx = self._get_filter_level_profile_ids(
            df_=profile_map_df[(profile_map_df['layer_id'].isin(layer_id_set)) & (profile_map_df['level_id'] == self._get_filter_level_id())],
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_QUOTA_SHARE][fields + valid_fields + ['layer_id']]
        )
        profile_map_df.loc[filter_idx, 'profile_id'] = passthroughprofile_id

        # Surplus Share profile IDs
        self.logger.debug(f'{oed.REINS_TYPE_SURPLUS_SHARE} profiles...')
        layer_id_set = set(
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_SURPLUS_SHARE]['layer_id']
        )
        # Risk level
        # Note that Surplus Share profiles scope much match the filter exactly
        profile_map_df.loc[
            profile_map_df['layer_id'].isin(layer_id_set), 'profile_id'
        ] = self._get_risk_level_profile_ids(
            df_=profile_map_df[profile_map_df['layer_id'].isin(layer_id_set)],
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_SURPLUS_SHARE],
            exact=True
        )
        # Filter level
        profile_map_df.loc[
            (profile_map_df['layer_id'].isin(layer_id_set)) & (profile_map_df['level_id'] == self._get_filter_level_id()),
            'profile_id'
        ] = passthroughprofile_id

        # Cat XL profile IDs
        self.logger.debug(f'{oed.REINS_TYPE_CAT_XL} profiles...')
        layer_id_set = set(
            ri_df[ri_df['reinstype'] == oed.REINS_TYPE_CAT_XL]['layer_id']
        )
        # Risk level
        profile_map_df.loc[
            profile_map_df['layer_id'].isin(layer_id_set), 'profile_id'
        ] = self._get_risk_level_profile_ids(
            df_=profile_map_df[profile_map_df['layer_id'].isin(layer_id_set)],
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_CAT_XL],
        )
        # Filter level
        filter_idx = self._get_filter_level_profile_ids(
            df_=profile_map_df[(profile_map_df['layer_id'].isin(layer_id_set)) & (profile_map_df['level_id'] == self._get_filter_level_id())],
            ri_df_=ri_df[ri_df['reinstype'] == oed.REINS_TYPE_CAT_XL][fields + valid_fields + ['layer_id']]
        )
        profile_map_df.loc[filter_idx, 'profile_id'] = passthroughprofile_id

        # OccLimit / Placed Percent
        occlimit_df = self.ri_info_df.copy()
        occlimit_df.columns = occlimit_df.columns.str.lower()
        occlimit_df = occlimit_df.merge(ri_df[occlimit_df.columns.to_list() + ['layer_id']], how='left').drop_duplicates()
        occlimit_df = occlimit_df[occlimit_df['reinstype'] != oed.REINS_TYPE_FAC]
        occlimit_df['level_id'] = self._get_risk_level_id() + 1
        occlimit_df = occlimit_df.reset_index(drop=True)
        occlimit_df['profile_id'] = occlimit_df.index + 1 + ri_df['profile_id'].max()
        if self.logger:
            self._log_dataframe({'occlimit_df': occlimit_df})

        # Assign programme node level profile IDs
        self.logger.debug('Assigning programme node level profile IDs')
        # Per Risk, Quota Share & Surplus Share profile IDs
        fmprofiles_df = self._get_occlimit_profiles(
            occlimit_df_=occlimit_df[occlimit_df['reinstype'] != oed.REINS_TYPE_CAT_XL],
            fmprofiles_df=fmprofiles_df, limit="row['occlimit']",
            placement="row['placedpercent']"
        )
        # Cat XL profile IDs
        fmprofiles_df = self._get_profiles(
            ri_df_=occlimit_df[occlimit_df['reinstype'] == oed.REINS_TYPE_CAT_XL],
            fmprofiles_df=fmprofiles_df, attachment="row['occattachment']",
            limit="row['occlimit']", ceded="row['cededpercent']",
            placement="row['placedpercent']"
        )

        # Programme level profile IDs
        profile_map_df.loc[
            profile_map_df['level_id'] == 4, 'profile_id'
        ] = self._match_exact(
            profile_map_df[profile_map_df['level_id'] == 4],
            occlimit_df[['layer_id', 'level_id', 'profile_id']]
        )

        profile_map_df['profile_id'] = profile_map_df['profile_id'].astype('int64')
        if self.logger:
            self._log_dataframe({'profile_map_df': profile_map_df})

        # Write out Oasis structure
        self.fmprogrammes_df = xref_df[xref_df['agg_id_to'] != 0][['agg_id', 'level_id', 'agg_id_to']].reset_index(drop=True)
        self.fmprogrammes_df.columns = ['from_agg_id', 'level_id', 'to_agg_id']
        self.fmprofiles_df = fmprofiles_df.sort_values(by='profile_id').reset_index(drop=True)
        self.fm_policytcs_df = profile_map_df[profile_map_df['level_id'] > 1][['layer_id', 'level_id', 'agg_id', 'profile_id']].reset_index(drop=True)
        self.fm_policytcs_df['level_id'] = self.fm_policytcs_df['level_id'] - 1

        if self.logger:
            self._log_dataframe({
                'fm_programmes_df': self.fmprogrammes_df,
                'fmprofiles_df': self.fmprofiles_df,
                'fm_policytcs_df': self.fm_policytcs_df
            })

    def write_oasis_files(self, directory=None):
        '''
        Write out the generated data to Oasis input file format.
        '''

        if directory is None:
            directory = "direct"
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        self.coverages_df.to_csv(
            os.path.join(directory, "coverages.csv"), index=False)
        self.items_df.to_csv(
            os.path.join(directory, "items.csv"), index=False)
        self.fmprogrammes_df.to_csv(
            os.path.join(directory, "fm_programme.csv"), index=False)
        self.fmprofiles_df.to_csv(
            os.path.join(directory, "fm_profile.csv"), index=False)
        self.fm_policytcs_df.to_csv(
            os.path.join(directory, "fm_policytc.csv"), index=False)
        self.fmsummaryxref_df.to_csv(
            os.path.join(directory, "fmsummaryxref.csv"), index=False)
        self.gulsummaryxref_df.to_csv(
            os.path.join(directory, "gulsummaryxref.csv"), index=False)
