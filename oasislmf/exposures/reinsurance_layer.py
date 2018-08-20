import pandas as pd
import os
import logging
import subprocess
import anytree
import shutil
import common
import json
from collections import namedtuple


# Meta-data about an inuring layer
InuringLayer = namedtuple(
    "InuringLayer",
    "inuring_priority reins_numbers is_valid validation_messages")

def any_fac(reins_info_df):
    '''
    Is there any fac?
    '''
    return not reins_info_df[reins_info_df.ReinsType == common.REINS_TYPE_FAC].empty

def any_quota_share(reins_info_df):
    '''
    Is there any quota share?
    '''
    return not reins_info_df[reins_info_df.ReinsType == common.REINS_TYPE_QUOTA_SHARE].empty


def any_surplus_share(reins_info_df):
    '''
    Is there any surplus share?
    '''
    return not reins_info_df[reins_info_df.ReinsType == common.REINS_TYPE_SURPLUS_SHARE].empty


def any_per_risk(reins_info_df):
    '''
    Is there any per-risk?
    '''
    return not reins_info_df[reins_info_df.ReinsType == common.REINS_TYPE_PER_RISK].empty


def any_cat_xl(reins_info_df):
    '''
    Are there any Cat XL?
    '''
    return not reins_info_df[reins_info_df.ReinsType == common.REINS_TYPE_CAT_XL].empty


def any_agg_xl(reins_info_df):
    '''
    Are there any AGG XL?
    '''
    return not reins_info_df[reins_info_df.ReinsType == common.REINS_TYPE_AGG_XL].empty

def valid_links(df_src, column_name, df_dest):
    '''
    Check that all unique values in df_src[column_name] map to df_dest[column_name]
    '''
    src_values = df_src[column_name].unique().tolist()
    return df_dest.isin({column_name: src_values}).all()


def validate_reinsurance_structures(account_df, location_df, ri_info_df, ri_scope_df):
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

        has_fac = any_fac(inuring_priority_ri_info_df)
        has_quota_share = any_quota_share(inuring_priority_ri_info_df)
        has_surplus_share = any_surplus_share(inuring_priority_ri_info_df)
        has_per_risk = any_per_risk(inuring_priority_ri_info_df)
        has_cat_xl = any_cat_xl(inuring_priority_ri_info_df)
        has_agg_xl = any_agg_xl(inuring_priority_ri_info_df)

        inuring_scope_ids = inuring_priority_ri_info_df.ReinsNumber.tolist()
        inuring_scope_df = ri_scope_df[ri_scope_df.isin({"ReinsNumber": inuring_scope_ids}).ReinsNumber]
        scope_risks = inuring_scope_df.RiskLevel.unique().tolist()

        set_all_account   = inuring_scope_df.AccountNumber.notnull().all()
        set_all_policy    = inuring_scope_df.PolicyNumber.notnull().all()
        set_all_location  = inuring_scope_df.LocationNumber.notnull().all()
        null_all_account  = inuring_scope_df.AccountNumber.isnull().all()
        null_all_policy   = inuring_scope_df.PolicyNumber.isnull().all()
        null_all_location = inuring_scope_df.LocationNumber.isnull().all()

        all_scope_non_specific = (null_all_account and null_all_policy and null_all_location)
        all_scope_specific = (set_all_account and set_all_policy and set_all_location)

        all_links_valid = (
            valid_links(ri_scope_df, "AccountNumber",   account_df),
            valid_links(ri_scope_df, "PolicyNumber",    account_df),
            #valid_links(ri_scope_df, "PortfolioNumber", account_df),
            valid_links(ri_scope_df, "AccountNumber",   location_df),
            valid_links(ri_scope_df, "LocationNumber",  location_df),
        )

        if has_agg_xl:
            is_valid = False
            validation_messages.append("Aggreation XL not implemented")
            continue
        if has_fac and (has_quota_share or has_surplus_share or has_per_risk or has_cat_xl or has_agg_xl):
            is_valid = False
            validation_messages.append(
                "Fac cannot be combined with other reinsurance types")
            continue
        if has_per_risk and (has_fac or has_quota_share or has_surplus_share or has_cat_xl or has_agg_xl):
            is_valid = False
            validation_messages.append(
                " per risk cannot be combined with other reinsurance types")
            continue
        if has_cat_xl and (has_fac or has_quota_share or has_surplus_share or has_per_risk or has_agg_xl):
            is_valid = False
            validation_messages.append(
                "Cat XL cannot be combined with other reinsurance types")
            continue
        if has_agg_xl and (has_fac or has_quota_share or has_surplus_share or has_per_risk or has_cat_xl):
            is_valid = False
            validation_messages.append(
                "AGG XL cannot be combined with other reinsurance types")
            continue

        if len(scope_risks) is not 1:
            is_valid = False
            validation_messages.append(
                "Mix of risk levels in a single reinsurance scope")
            continue

        if not (set(scope_risks) < set(common.REINS_RISK_LEVELS)):
            is_valid = False
            validation_messages.append(
                "Invalid risk level, {}".format(' '.join(scope_risks)))
            continue

        if has_surplus_share and not all_scope_specific:
            is_valid = False
            validation_messages.append(
                "SS cannot have non-specific scopes")
            continue

        if has_quota_share and not all_scope_non_specific:
            is_valid = False
            validation_messages.append(
                "QS cannot have specific scopes set")
            continue

        if not all_links_valid:
            is_valid = False
            validation_messages.append(
                "Non-linking scopes between ri_scope and (ACC,LOC) files")
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


class ReinsuranceLayer(object):
    """
    Generates ktools inputs and runs financial module for a reinsurance structure.
    """

    def __init__(self, name, ri_info, ri_scope, accounts, locations,
                 items, coverages, fm_xrefs, xref_descriptions, risk_level, logger=None):

        self.logger = logger or logging.getLogger()
        self.name = name
        self.accounts = accounts
        self.locations = locations

        self.coverages = coverages
        self.items = items
        self.fm_xrefs = fm_xrefs
        self.xref_descriptions = xref_descriptions

        self.item_ids = list()
        self.item_tivs = list()
        self.fmprogrammes = pd.DataFrame()
        self.fmprofiles = pd.DataFrame()
        self.fm_policytcs = pd.DataFrame()

        self.risk_level = risk_level

        self.ri_info = ri_info
        self.ri_scope = ri_scope

        self.add_profiles_args = namedtuple(
            "AddProfilesArgs",
            "program_node, ri_info_row, scope_rows, overlay_loop, layer_id, "
            "node_layer_profile_map, fmprofiles_list, nolossprofile_id, passthroughprofile_id")

    def _add_program_node(self, level_id):
        return anytree.Node(
            "Occurrence",
            parent=None,
            level_id=level_id,
            agg_id=1,
            account_number=common.NOT_SET_ID,
            policy_number=common.NOT_SET_ID,
            location_number=common.NOT_SET_ID)

    def _add_item_node(self, xref_id, parent):
        return anytree.Node(
            "Item_id:{}".format(xref_id),
            parent=parent,
            level_id=1,
            agg_id=xref_id,
            account_number=common.NOT_SET_ID,
            policy_number=common.NOT_SET_ID,
            location_number=common.NOT_SET_ID)

    def _add_location_node(
            self, level_id, agg_id, xref_description, parent):
        return anytree.Node(
            "Account_number:{} Policy_number:{} Location_number:{}".format(
                xref_description.account_number,
                xref_description.policy_number,
                xref_description.location_number),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            account_number=xref_description.account_number,
            policy_number=xref_description.policy_number,
            location_number=xref_description.location_number)

    def _add_policy_node(
            self, level_id, agg_id, xref_description, parent):
        return anytree.Node(
            "Account_number:{} Policy_number:{}".format(
                xref_description.account_number, xref_description.policy_number),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            account_number=xref_description.account_number,
            policy_number=xref_description.policy_number,
            location_number=common.NOT_SET_ID)

    def _add_account_node(
            self, agg_id, level_id, xref_description, parent):
        return anytree.Node(
            "Account_number:{}".format(xref_description.account_number),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            account_number=xref_description.account_number,
            policy_number=common.NOT_SET_ID,
            location_number=common.NOT_SET_ID)



    def _does_location_node_match_scope_row(self, node, ri_scope_row):
        node_summary = (node.account_number,
                        node.policy_number, node.location_number)
        scope_row_summary = (ri_scope_row.AccountNumber,
                             ri_scope_row.PolicyNumber, ri_scope_row.LocationNumber)
        if (node_summary == scope_row_summary):
            self.logger.debug('Matching node: location to scope\n node: {}, ri_scope: {}'.format(
                str(node_summary),
                str(scope_row_summary),
            ))
        return (node_summary == scope_row_summary)

    def _does_policy_node_match_scope_row(self, node, ri_scope_row):
        node_summary = (node.account_number,
                        node.policy_number, common.NOT_SET_ID)
        scope_row_summary = (ri_scope_row.AccountNumber, ri_scope_row.PolicyNumber, common.NOT_SET_ID)
        if (node_summary == scope_row_summary):
            self.logger.debug('Matching node: policy to scope\n node: {}, ri_scope: {}'.format(
                str(node_summary),
                str(scope_row_summary),
            ))
        return (node_summary == scope_row_summary)

    def _does_account_node_match_scope_row(self, node, ri_scope_row):
        node_summary = (node.account_number,
                        common.NOT_SET_ID, common.NOT_SET_ID)
        scope_row_summary = (ri_scope_row.AccountNumber,
                             common.NOT_SET_ID, common.NOT_SET_ID)
        if (node_summary == scope_row_summary):
            self.logger.debug('Matching node: account to scope\n node: {}, ri_scope: {}'.format(
                str(node_summary),
                str(scope_row_summary),
            ))
        return (node_summary == scope_row_summary)

    # More generic but slower (testing only)
    def _match_node(self, node, search_dict):
        node_dict = {
            'AccountNumber':  node.account_number,
            'PolicyNumber':   node.policy_number,
            'LocationNumber': node.location_number,
        }
        self.logger.debug('Matching node: \n\t node: {}, \n\t search: {}'.format(
            str(node_dict),
            str(search_dict),
        ))
        return search_dict.items() <= node_dict.items()

    def _get_tree(self):
        current_location_number = 0
        current_policy_number = 0
        current_account_number = 0
        current_location_node = None
        current_policy_node = None
        current_account_node = None

        program_node_level_id = 3
        if self.risk_level == common.REINS_RISK_LEVEL_PORTFOLIO:
            program_node_level_id = 2
        program_node = self._add_program_node(program_node_level_id)
        xref_descriptions = self.xref_descriptions.sort_values(
            by=["location_number", "policy_number", "account_number"])
        agg_id = 0
        if self.risk_level == common.REINS_RISK_LEVEL_PORTFOLIO:
            for _, row in xref_descriptions.iterrows():
                self._add_item_node(row.xref_id, program_node)
        elif self.risk_level == common.REINS_RISK_LEVEL_ACCOUNT:
            for _, row in xref_descriptions.iterrows():
                if current_account_number != row.account_number:
                    agg_id = agg_id + 1
                    level_id = 2
                    current_account_number = row.account_number
                    current_account_node = self._add_account_node(
                        agg_id, level_id, row, program_node)
                self._add_item_node(row.xref_id, current_account_node)
        elif self.risk_level == common.REINS_RISK_LEVEL_POLICY:
            for _, row in xref_descriptions.iterrows():
                if current_policy_number != row.policy_number:
                    agg_id = agg_id + 1
                    level_id = 2
                    current_policy_number = row.policy_number
                    current_policy_node = self._add_location_node(
                        level_id, agg_id, row, program_node)
                self._add_item_node(row.xref_id, current_policy_node)
        elif self.risk_level == common.REINS_RISK_LEVEL_LOCATION:
            for _, row in xref_descriptions.iterrows():
                if current_location_number != row.location_number:
                    agg_id = agg_id + 1
                    level_id = 2
                    current_location_number = row.location_number
                    current_location_node = self._add_location_node(
                        level_id, agg_id, row, program_node)
                self._add_item_node(row.xref_id, current_location_node)
        return program_node

    def _add_fac_profiles(self, add_profiles_args):
        self.logger.debug("Adding FAC profiles:")
        profile_id = max(
            x.profile_id for x in add_profiles_args.fmprofiles_list)

        # Add pass through nodes at all levels so that the risks
        # not explicitly covered are unaffected

        for node in anytree.iterators.LevelOrderIter(add_profiles_args.program_node):
            add_profiles_args.node_layer_profile_map[(
                node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.nolossprofile_id
        add_profiles_args.node_layer_profile_map[(
            add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id

        profile_id = profile_id + 1
        add_profiles_args.fmprofiles_list.append(common.get_reinsurance_profile(
            profile_id,
            attachment=add_profiles_args.ri_info_row.RiskAttachmentPoint,
            limit=add_profiles_args.ri_info_row.RiskLimit,
            ceded=add_profiles_args.ri_info_row.CededPercent,
            placement=add_profiles_args.ri_info_row.PlacementPercent
        ))

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            if ri_scope_row.RiskLevel == common.REINS_RISK_LEVEL_LOCATION:
                nodes = anytree.search.findall(
                    add_profiles_args.program_node,
                    filter_=lambda node: self._does_location_node_match_scope_row(node, ri_scope_row))
                for node in nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            elif ri_scope_row.RiskLevel == common.REINS_RISK_LEVEL_POLICY:
                nodes = anytree.search.findall(
                    add_profiles_args.program_node,
                    filter_=lambda node: self._does_policy_node_match_scope_row(node, ri_scope_row))
                for node in nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            elif ri_scope_row.RiskLevel == common.REINS_RISK_LEVEL_ACCOUNT:
                nodes = anytree.search.findall(
                    add_profiles_args.program_node,
                    filter_=lambda node: self._does_account_node_match_scope_row(node, ri_scope_row))
                for node in nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            else:
                raise Exception(
                    "Unsupported risk level: {}".format(ri_scope_row.RiskLevel))

    # Need to check Matching rules for Per Risk with Joh
    def _add_per_risk_profiles(self, add_profiles_args):
        self.logger.debug("Adding PR profiles:")
        profile_id = max(
            x.profile_id for x in add_profiles_args.fmprofiles_list)

        # not explcitly covered are unaffected
        for node in anytree.iterators.LevelOrderIter(add_profiles_args.program_node):
            add_profiles_args.node_layer_profile_map[(
                node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.nolossprofile_id
        add_profiles_args.node_layer_profile_map[(
            add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(common.get_reinsurance_profile(
                profile_id,
                attachment=add_profiles_args.ri_info_row.RiskAttachmentPoint,
                limit=add_profiles_args.ri_info_row.RiskLimit,
                placement=add_profiles_args.ri_info_row.PlacementPercent
            ))

            nodes = anytree.search.findall(
                add_profiles_args.program_node, filter_=lambda node: node.level_id == 2)
            for node in nodes:
                add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

        # Check ri_info row for overall OccLimit
        if add_profiles_args.ri_info_row.OccLimit:
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(
                common.get_occlim_profile(
                    profile_id,
                    limit=add_profiles_args.ri_info_row.OccLimit,
            ))
            add_profiles_args.node_layer_profile_map[
                (add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id


    def _add_surplus_share_profiles(self, add_profiles_args):
        self.logger.debug("Adding SS profiles:")
        profile_id = max(
            x.profile_id for x in add_profiles_args.fmprofiles_list)

        # Add pass through nodes at all levels so that the risks
        # not explicitly covered are unaffected
        for node in anytree.iterators.LevelOrderIter(add_profiles_args.program_node):
            add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.nolossprofile_id

        add_profiles_args.node_layer_profile_map[(
            add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(common.get_reinsurance_profile(
                profile_id,
                attachment=add_profiles_args.ri_info_row.RiskAttachmentPoint,
                limit=add_profiles_args.ri_info_row.RiskLimit,
                ceded=ri_scope_row.CededPercent,
                placement=add_profiles_args.ri_info_row.PlacementPercent
            ))
            if ri_scope_row.RiskLevel == common.REINS_RISK_LEVEL_LOCATION:
                nodes = anytree.search.findall(
                    add_profiles_args.program_node,
                    filter_=lambda node: self._does_location_node_match_scope_row(node, ri_scope_row))
                for node in nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            elif ri_scope_row.RiskLevel == common.REINS_RISK_LEVEL_POLICY:
                nodes = anytree.search.findall(
                    add_profiles_args.program_node,
                    filter_=lambda node: self._does_policy_node_match_scope_row(node, ri_scope_row))
                for node in nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            elif ri_scope_row.RiskLevel == common.REINS_RISK_LEVEL_ACCOUNT:
                nodes = anytree.search.findall(
                    add_profiles_args.program_node,
                    filter_=lambda node: self._does_account_node_match_scope_row(node, ri_scope_row))
                for node in nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            else:
                raise Exception(
                    "Unsupported risk level: {}".format(ri_scope_row.RiskLevel))

        # Check ri_info row for overall OccLimit
        if add_profiles_args.ri_info_row.OccLimit:
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(
                common.get_occlim_profile(
                    profile_id,
                    limit=add_profiles_args.ri_info_row.OccLimit,
            ))
            add_profiles_args.node_layer_profile_map[
                (add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id




    def _add_quota_share_profiles(self, add_profiles_args):
        self.logger.debug("Adding QS profiles:")
        profile_id = max(
            x.profile_id for x in add_profiles_args.fmprofiles_list)

        # Add any risk limits
        if self.risk_level == common.REINS_RISK_LEVEL_PORTFOLIO:
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(
                common.get_reinsurance_profile(
                    profile_id,
                    limit=add_profiles_args.ri_info_row.OccLimit,
                    ceded=add_profiles_args.ri_info_row.CededPercent,
                    placement=add_profiles_args.ri_info_row.PlacementPercent
            ))
        else:
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(
                common.get_reinsurance_profile(
                    profile_id,
                    limit=add_profiles_args.ri_info_row.RiskLimit,
                    ceded=add_profiles_args.ri_info_row.CededPercent,
                    placement=add_profiles_args.ri_info_row.PlacementPercent
                ))
            nodes = anytree.search.findall(
                add_profiles_args.program_node, filter_=lambda node: node.level_id == 2)
            for node in nodes:
                add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

            # Add occurrence limit and share
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(
                common.get_reinsurance_profile(
                    profile_id,
                    limit=add_profiles_args.ri_info_row.OccLimit,
                    placement=add_profiles_args.ri_info_row.PlacementPercent
            ))
        add_profiles_args.node_layer_profile_map[
            (add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

    def _add_cat_xl_profiles(self, add_profiles_args):
        self.logger.debug("Adding CAT XL profiles:")
        profile_id = max(
            x.profile_id for x in add_profiles_args.fmprofiles_list)
        profile_id = profile_id + 1
        add_profiles_args.fmprofiles_list.append(
            common.get_reinsurance_profile(
                profile_id,
                attachment=add_profiles_args.ri_info_row.OccurenceAttachmentPoint,
                limit=add_profiles_args.ri_info_row.OccLimit,
                placement=add_profiles_args.ri_info_row.PlacementPercent
            ))
        add_profiles_args.node_layer_profile_map[
            (add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

    def generate_oasis_structures(self):
        '''
        Create the Oasis structures - FM Programmes, FM Profiles and FM Policy TCs -
        that represent the resinsurance structure.

        The algorithm to create the stucture has three steps:
        Step 1 - Build a tree representation of the insurance program, depening on the reinsuarnce risk level.
        Step 2 - Overlay the reinsurance structure. Each resinsuarnce contact is a seperate layer.
        Step 3 - Iterate over the tree and write out the Oasis structure.
        '''

        fmprogrammes_list = list()
        fmprofiles_list = list()
        fm_policytcs_list = list()

        profile_id = 1
        nolossprofile_id = profile_id
        fmprofiles_list.append(
            common.get_no_loss_profile(nolossprofile_id))
        profile_id = profile_id + 1
        passthroughprofile_id = profile_id
        fmprofiles_list.append(
            common.get_pass_through_profile(passthroughprofile_id))

        node_layer_profile_map = {}

        self.logger.debug(fmprofiles_list)


        #
        # Step 1 - Build a tree representation of the insurance program, depening on the reinsuarnce risk level.
        #
        program_node  = self._get_tree()

        if self.logger:
            self.logger.debug('program_node tree: "{}"'.format(self.name))
            self.logger.debug(anytree.RenderTree(program_node))
            #Plot tree to image (graphviz)
            #from anytree.dotexport import RenderTreeGraph
            #RenderTreeGraph(program_node).to_picture(
            #    "{}.png".format(self.name))


        #
        # Step 2 - Overlay the reinsurance structure. Each resinsuarnce contact is a seperate layer.
        #
        layer_id = 0
        overlay_loop = 0
        for _, ri_info_row in self.ri_info.iterrows():
            overlay_loop += 1
            layer_id = ri_info_row.ReinsLayerNumber

            scope_rows = self.ri_scope[
                (self.ri_scope.ReinsNumber == ri_info_row.ReinsNumber) &
                (self.ri_scope.RiskLevel == self.risk_level)]

            if self.logger:
                pd.set_option('display.width', 1000)
                self.logger.debug('ri_scope: "{}"'.format(self.name))
                self.logger.debug(scope_rows)

            if scope_rows.shape[0] == 0:
                continue

            add_profiles_args = self.add_profiles_args(
                program_node, ri_info_row, scope_rows, overlay_loop, layer_id, node_layer_profile_map,
                fmprofiles_list, nolossprofile_id, passthroughprofile_id)

            if ri_info_row.ReinsType == common.REINS_TYPE_FAC:
                self._add_fac_profiles(add_profiles_args)
            elif ri_info_row.ReinsType == common.REINS_TYPE_PER_RISK:
                self._add_per_risk_profiles(add_profiles_args)
            elif ri_info_row.ReinsType == common.REINS_TYPE_QUOTA_SHARE:
                self._add_quota_share_profiles(add_profiles_args)
            elif ri_info_row.ReinsType == common.REINS_TYPE_SURPLUS_SHARE:
                self._add_surplus_share_profiles(add_profiles_args)
            elif ri_info_row.ReinsType == common.REINS_TYPE_CAT_XL:
                self._add_cat_xl_profiles(add_profiles_args)
            else:
                raise Exception("ReinsType not supported yet: {}".format(
                    ri_info_row.ReinsType))


        #
        # Step 3 - Iterate over the tree and write out the Oasis structure.
        #
        for node in anytree.iterators.LevelOrderIter(program_node):
            if node.parent is not None:
                fmprogrammes_list.append(
                    common.FmProgramme(
                        from_agg_id=node.agg_id,
                        level_id=node.level_id,
                        to_agg_id=node.parent.agg_id
                    )
                )

        # Note: Pending confirmation from Joh that ReinsLayerNumber is being used correctly
        for layer in range(1,max(self.ri_info.ReinsLayerNumber.tolist())+1):
            for node in anytree.iterators.LevelOrderIter(program_node):
                if node.level_id > 1:
                    profiles_ids = []

                    # The `overlay_rule` replaces using each resinsuarnce contact in a seperate layer
                    # Collect overlaping unique combinations of (layer_id, level_id, agg_id) and combine into
                    # a single layer
                    for overlay_rule in range(1,overlay_loop+1):
                        try:
                            profiles_ids.append(
                                node_layer_profile_map[(node.name, layer, overlay_rule)])
                        except:
                            pass
                            #print("Invalid keys")
                    fm_policytcs_list.append(common.FmPolicyTc(
                        layer_id=layer,
                        level_id=node.level_id - 1,
                        agg_id=node.agg_id,
                        profile_id=max(profiles_ids)
                    ))

        self.fmprogrammes = pd.DataFrame(fmprogrammes_list)
        self.fmprofiles = pd.DataFrame(fmprofiles_list)
        self.fm_policytcs = pd.DataFrame(fm_policytcs_list)


        # Log Reinsurance structures
        if self.logger:
            self.logger.debug('policytc_map: "{}"'.format(self.name))
            policytc_map = dict()
            for k in add_profiles_args.node_layer_profile_map.keys():
                profile_id = add_profiles_args.node_layer_profile_map[k]
                policytc_map["(Name=%s, layer_id=%s, overlay_loop=%s)" % k] = profile_id
            self.logger.debug(json.dumps(policytc_map, indent=4))
            self.logger.debug('fm_policytcs: "{}"'.format(self.name))
            self.logger.debug(self.fm_policytcs)
            self.logger.debug('fm_profile: "{}"'.format(self.name))
            self.logger.debug(self.fmprofiles)

    def write_oasis_files(self, directory=None):

        if directory is None:
            directory = "direct"
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)

        self.coverages.to_csv(
            os.path.join(directory, "coverages.csv"), index=False)
        self.items.to_csv(
            os.path.join(directory, "items.csv"), index=False)
        self.fmprogrammes.to_csv(
            os.path.join(directory, "fm_programme.csv"), index=False)
        self.fmprofiles.to_csv(
            os.path.join(directory, "fm_profile.csv"), index=False)
        self.fm_policytcs.to_csv(
            os.path.join(directory, "fm_policytc.csv"), index=False)
        self.fm_xrefs.to_csv(
            os.path.join(directory, "fm_xref.csv"), index=False)
        open(os.path.join(directory, "fmsummaryxref.csv"), 'a').close()
        open(os.path.join(directory, "gulsummaryxref.csv"), 'a').close()
