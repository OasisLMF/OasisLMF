# -*- coding: utf-8 -*-

__all__ = [
    'generate_files_for_reinsurance',
    'ReinsuranceLayer',
]

import pandas as pd
import os
import logging
import subprocess
import anytree
import shutil
import json
from collections import namedtuple

from oasislmf.exposures import oed


# Meta-data about an inuring layer
InuringLayer = namedtuple(
    "InuringLayer",
    "inuring_priority reins_numbers is_valid validation_messages")


def generate_files_for_reinsurance(
       # account_df,
       # location_df,
        items,
        coverages,
        fm_xrefs,
        xref_descriptions,
        ri_info_df,
        ri_scope_df,
        direct_oasis_files_dir):
    """
    Generate files for reinsurance.
    """

    inuring_metadata = {}
    previous_inuring_priority = None
    previous_risk_level = None
    reinsurance_index = 1
    for inuring_priority in range(1, ri_info_df['InuringPriority'].max() + 1):
        # Filter the reinsNumbers by inuring_priority
        reins_numbers = ri_info_df[ri_info_df['InuringPriority'] == inuring_priority].ReinsNumber.tolist()
        risk_level_set = set(ri_scope_df[ri_scope_df['ReinsNumber'].isin(reins_numbers)].RiskLevel)

        for risk_level in oed.REINS_RISK_LEVELS:
            if risk_level not in risk_level_set:
                continue

            written_to_dir = _generate_files_for_reinsurance_risk_level(
                inuring_priority,
                #account_df,
                #location_df,
                items,
                coverages,
                fm_xrefs,
                xref_descriptions,
                ri_info_df,
                ri_scope_df,
                previous_inuring_priority,
                previous_risk_level,
                risk_level,
                reinsurance_index,
                direct_oasis_files_dir)

            inuring_metadata[reinsurance_index] = {
                'inuring_priority': inuring_priority,
                'risk_level': risk_level,
                'directory': written_to_dir,
            }
            previous_inuring_priority = inuring_priority
            previous_risk_level = risk_level
            reinsurance_index = reinsurance_index + 1

    return inuring_metadata


def _generate_files_for_reinsurance_risk_level(
        inuring_priority,
        #account_df,
        #location_df,
        items,
        coverages,
        fm_xrefs,
        xref_descriptions,
        ri_info_df,
        ri_scope_df,
        previous_inuring_priority,
        previous_risk_level,
        risk_level,
        reinsurance_index,
        direct_oasis_files_dir):
    """
    Generate files for a reinsurance risk level.
    """
    reins_numbers_1 = ri_info_df[
        ri_info_df['InuringPriority'] == inuring_priority].ReinsNumber
    if reins_numbers_1.empty:
        return None
    reins_numbers_2 = ri_scope_df[
        ri_scope_df.isin({"ReinsNumber": reins_numbers_1.tolist()}).ReinsNumber &
        (ri_scope_df.RiskLevel == risk_level)].ReinsNumber
    if reins_numbers_2.empty:
        return None

    ri_info_inuring_priority_df = ri_info_df[ri_info_df.isin(
        {"ReinsNumber": reins_numbers_2.tolist()}).ReinsNumber]
    output_name = "ri_{}_{}".format(inuring_priority, risk_level)
    reinsurance_layer = ReinsuranceLayer(
        name=output_name,
        ri_info=ri_info_inuring_priority_df,
        ri_scope=ri_scope_df,
        #accounts=account_df,
        #locations=location_df,
        items=items,
        coverages=coverages,
        fm_xrefs=fm_xrefs,
        xref_descriptions=xref_descriptions,
        risk_level=risk_level
    )

    reinsurance_layer.generate_oasis_structures()
    #output_dir = os.path.join(direct_oasis_files_dir, output_name)
    output_dir = os.path.join(direct_oasis_files_dir, "RI_{}".format(reinsurance_index))
    reinsurance_layer.write_oasis_files(output_dir)
    return output_dir




class ReinsuranceLayer(object):
    """
    Generates ktools inputs and runs financial module for a reinsurance structure.
    """

    def __init__(self, name, ri_info, ri_scope, #accounts, locations,
                 items, coverages, fm_xrefs, xref_descriptions, risk_level, logger=None):

        self.logger = logger or logging.getLogger()
        self.name = name
        #self.accounts = accounts
        #self.locations = locations

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
            "Treaty",
            parent=None,
            level_id=level_id,
            agg_id=1,
            account_number=oed.NOT_SET_ID,
            policy_number=oed.NOT_SET_ID,
            location_number=oed.NOT_SET_ID)

    def _add_item_node(self, xref_id, parent):
        return anytree.Node(
            "Item_id:{}".format(xref_id),
            parent=parent,
            level_id=1,
            agg_id=xref_id,
            account_number=oed.NOT_SET_ID,
            policy_number=oed.NOT_SET_ID,
            location_number=oed.NOT_SET_ID)

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
            location_number=oed.NOT_SET_ID)

    def _add_account_node(
            self, agg_id, level_id, xref_description, parent):
        return anytree.Node(
            "Account_number:{}".format(xref_description.account_number),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            account_number=xref_description.account_number,
            policy_number=oed.NOT_SET_ID,
            location_number=oed.NOT_SET_ID)



    def _does_location_node_match_scope_row(self, node, ri_scope_row):
        node_summary = (node.account_number,
                        node.policy_number, node.location_number)
        scope_row_summary = (ri_scope_row.AccNumber,
                             ri_scope_row.PolNumber, ri_scope_row.LocNumber)
        if (node_summary == scope_row_summary):
            self.logger.debug('Matching node: location to scope\n node: {}, ri_scope: {}'.format(
                str(node_summary),
                str(scope_row_summary),
            ))
        return (node_summary == scope_row_summary)

    def _does_policy_node_match_scope_row(self, node, ri_scope_row):
        node_summary = (node.account_number,
                        node.policy_number, oed.NOT_SET_ID)
        scope_row_summary = (ri_scope_row.AccNumber, ri_scope_row.PolNumber, oed.NOT_SET_ID)
        if (node_summary == scope_row_summary):
            self.logger.debug('Matching node: policy to scope\n node: {}, ri_scope: {}'.format(
                str(node_summary),
                str(scope_row_summary),
            ))
        return (node_summary == scope_row_summary)

    def _does_account_node_match_scope_row(self, node, ri_scope_row):
        node_summary = (node.account_number,
                        oed.NOT_SET_ID, oed.NOT_SET_ID)
        scope_row_summary = (ri_scope_row.AccNumber,
                             oed.NOT_SET_ID, oed.NOT_SET_ID)
        if (node_summary == scope_row_summary):
            self.logger.debug('Matching node: account to scope\n node: {}, ri_scope: {}'.format(
                str(node_summary),
                str(scope_row_summary),
            ))
        return (node_summary == scope_row_summary)

    ## More generic but slower (testing only)
    #def _match_node(self, node, search_dict):
    #    node_dict = {
    #        'AccNumber':  node.account_number,
    #        'PolNumber':   node.policy_number,
    #        'LocNumber': node.location_number,
    #    }
    #    self.logger.debug('Matching node: \n\t node: {}, \n\t search: {}'.format(
    #        str(node_dict),
    #        str(search_dict),
    #    ))
    #    return search_dict.items() <= node_dict.items()

    def _filter_nodes(self, nodes_list, scope_row):
        """
        Return subset of `nodes_list` based on values of a row in `ri_scope.csv`

        TODO: Combined filters?
        """
        if (scope_row.RiskLevel == oed.REINS_RISK_LEVEL_ACCOUNT) and self._is_defined(scope_row.AccNumber):
            return list(filter(lambda n: n.account_number == scope_row.AccNumber, nodes_list))
        elif scope_row.RiskLevel == oed.REINS_RISK_LEVEL_LOCATION and self._is_defined(scope_row.LocNumber):
            return list(filter(lambda n: n.location_number == scope_row.LocNumber, nodes_list))
        elif scope_row.RiskLevel == oed.REINS_RISK_LEVEL_POLICY and self._is_defined(scope_row.PolNumber):
            return list(filter(lambda n: n.policy_number == scope_row.PolNumber, nodes_list))
        elif scope_row.RiskLevel == oed.REINS_RISK_LEVEL_PORTFOLIO and self._is_defined(scope_row.PortNumber):
            return list(filter(lambda n: n.policy_number == scope_row.PortNumber, nodes_list))
        else:
            return nodes_list



    def _is_defined(self, num_to_check):
        # If the value = NaN it will return False
        return num_to_check == num_to_check


    def _get_tree(self):
        current_location_number = 0
        current_policy_number = 0
        current_account_number = 0
        current_location_node = None
        current_policy_node = None
        current_account_node = None

        program_node_level_id = 3
        if self.risk_level == oed.REINS_RISK_LEVEL_PORTFOLIO:
            program_node_level_id = 2

        program_node = self._add_program_node(program_node_level_id)

        xref_descriptions = self.xref_descriptions.sort_values(
            by=["location_number", "policy_number", "account_number"])
        agg_id = 0
        if self.risk_level == oed.REINS_RISK_LEVEL_PORTFOLIO:
            for _, row in xref_descriptions.iterrows():
                self._add_item_node(row.xref_id, program_node)
        elif self.risk_level == oed.REINS_RISK_LEVEL_ACCOUNT:
            for _, row in xref_descriptions.iterrows():
                if current_account_number != row.account_number:
                    agg_id = agg_id + 1
                    level_id = 2
                    current_account_number = row.account_number
                    current_account_node = self._add_account_node(
                        agg_id, level_id, row, program_node)
                self._add_item_node(row.xref_id, current_account_node)
        elif self.risk_level == oed.REINS_RISK_LEVEL_POLICY:
            for _, row in xref_descriptions.iterrows():
                if current_policy_number != row.policy_number:
                    agg_id = agg_id + 1
                    level_id = 2
                    current_policy_number = row.policy_number
                    current_policy_node = self._add_location_node(
                        level_id, agg_id, row, program_node)
                self._add_item_node(row.xref_id, current_policy_node)
        elif self.risk_level == oed.REINS_RISK_LEVEL_LOCATION:
            for _, row in xref_descriptions.iterrows():
                if current_location_number != row.location_number:
                    agg_id = agg_id + 1
                    level_id = 2
                    current_location_number = row.location_number
                    current_location_node = self._add_location_node(
                        level_id, agg_id, row, program_node)
                self._add_item_node(row.xref_id, current_location_node)
        return program_node

#    def _add_occ_limit(self, add_profiles_args):



    def _add_fac_profiles(self, add_profiles_args):
        self.logger.debug("Adding FAC profiles:")
        profile_id = max(
            x.profile_id for x in add_profiles_args.fmprofiles_list)

        profile_id = profile_id + 1
        add_profiles_args.fmprofiles_list.append(oed.get_reinsurance_profile(
            profile_id,
            attachment=add_profiles_args.ri_info_row.RiskAttachment,
            limit=add_profiles_args.ri_info_row.RiskLimit,
            ceded=add_profiles_args.ri_info_row.CededPercent,
            placement=add_profiles_args.ri_info_row.PlacedPercent
        ))

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            if ri_scope_row.RiskLevel == oed.REINS_RISK_LEVEL_LOCATION:
                nodes = anytree.search.findall(
                    add_profiles_args.program_node,
                    filter_=lambda node: self._does_location_node_match_scope_row(node, ri_scope_row))
                for node in nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            elif ri_scope_row.RiskLevel == oed.REINS_RISK_LEVEL_POLICY:
                nodes = anytree.search.findall(
                    add_profiles_args.program_node,
                    filter_=lambda node: self._does_policy_node_match_scope_row(node, ri_scope_row))
                for node in nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            elif ri_scope_row.RiskLevel == oed.REINS_RISK_LEVEL_ACCOUNT:
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
        profile_id = max(x.profile_id for x in add_profiles_args.fmprofiles_list)
        nodes_all = anytree.search.findall(
               add_profiles_args.program_node, filter_=lambda node: node.level_id == 2)

        profile_id = profile_id + 1
        add_profiles_args.fmprofiles_list.append(oed.get_reinsurance_profile(
            profile_id,
            attachment=add_profiles_args.ri_info_row.RiskAttachment,
            limit=add_profiles_args.ri_info_row.RiskLimit,
            ceded=add_profiles_args.ri_info_row.CededPercent,
        ))

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            selected_nodes = self._filter_nodes(nodes_all, ri_scope_row)
            for node in selected_nodes:
                add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

        # add OccLimit / Placed Percent
        profile_id = profile_id + 1
        add_profiles_args.fmprofiles_list.append(
            oed.get_occlim_profile(
                profile_id,
                limit=add_profiles_args.ri_info_row.OccLimit,
                placement=add_profiles_args.ri_info_row.PlacedPercent,
        ))
        add_profiles_args.node_layer_profile_map[
            (add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id





    def _add_surplus_share_profiles(self, add_profiles_args):
        self.logger.debug("Adding SS profiles:")
        profile_id = max( x.profile_id for x in add_profiles_args.fmprofiles_list)
        nodes_all = anytree.search.findall(
               add_profiles_args.program_node, filter_=lambda node: node.level_id == 2)

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(oed.get_reinsurance_profile(
                profile_id,
                attachment=add_profiles_args.ri_info_row.RiskAttachment,
                limit=add_profiles_args.ri_info_row.RiskLimit,
                ceded=ri_scope_row.CededPercent,
            ))

            if ri_scope_row.RiskLevel == oed.REINS_RISK_LEVEL_LOCATION:
                selected_nodes = list(filter(lambda n: self._does_location_node_match_scope_row(n,ri_scope_row), nodes_all))
                for node in selected_nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            elif ri_scope_row.RiskLevel == oed.REINS_RISK_LEVEL_POLICY:
                selected_nodes = list(filter(lambda n: self._does_policy_node_match_scope_row(n,ri_scope_row), nodes_all))
                for node in selected_nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            elif ri_scope_row.RiskLevel == oed.REINS_RISK_LEVEL_ACCOUNT:
                selected_nodes = list(filter(lambda n: self._does_account_node_match_scope_row(n,ri_scope_row), nodes_all))
                for node in selected_nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            else:
                raise Exception(
                    "Unsupported risk level: {}".format(ri_scope_row.RiskLevel))


        # add OccLimit / Placed Percent
        profile_id = profile_id + 1
        add_profiles_args.fmprofiles_list.append(
            oed.get_occlim_profile(
                profile_id,
                limit=add_profiles_args.ri_info_row.OccLimit,
                placement=add_profiles_args.ri_info_row.PlacedPercent,
        ))
        add_profiles_args.node_layer_profile_map[
            (add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id




    def _add_quota_share_profiles(self, add_profiles_args):
        self.logger.debug("Adding QS profiles:")

        profile_id = max( x.profile_id for x in add_profiles_args.fmprofiles_list)
        nodes_all = anytree.search.findall(
               add_profiles_args.program_node, filter_=lambda node: node.level_id == 2)

        # Add any risk limits
        # RISK LEVEL SEL
        if self.risk_level == oed.REINS_RISK_LEVEL_PORTFOLIO:
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(
                oed.get_reinsurance_profile(
                    profile_id,
                    limit=add_profiles_args.ri_info_row.OccLimit,
                    ceded=add_profiles_args.ri_info_row.CededPercent,
                    placement=add_profiles_args.ri_info_row.PlacedPercent
            ))
        else:
            for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
                profile_id = profile_id + 1

                add_profiles_args.fmprofiles_list.append(
                    oed.get_reinsurance_profile(
                        profile_id,
                        limit=add_profiles_args.ri_info_row.RiskLimit,
                        ceded=add_profiles_args.ri_info_row.CededPercent,
                    ))

                # Filter
                selected_nodes = self._filter_nodes(nodes_all, ri_scope_row)
                for node in selected_nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

        # add OccLimit / Placed Percent
        profile_id = profile_id + 1
        add_profiles_args.fmprofiles_list.append(
            oed.get_occlim_profile(
                profile_id,
                limit=add_profiles_args.ri_info_row.OccLimit,
                placement=add_profiles_args.ri_info_row.PlacedPercent,
        ))
        add_profiles_args.node_layer_profile_map[
            (add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

    def _add_cat_xl_profiles(self, add_profiles_args):
        self.logger.debug("Adding CAT XL profiles:")
        profile_id = max(
            x.profile_id for x in add_profiles_args.fmprofiles_list)

        if self.risk_level == oed.REINS_RISK_LEVEL_PORTFOLIO:
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(
                oed.get_reinsurance_profile(
                    profile_id,
                    attachment=add_profiles_args.ri_info_row.OccAttachment,
                    limit=add_profiles_args.ri_info_row.OccLimit,
                    placement=add_profiles_args.ri_info_row.PlacedPercent,
                    ceded=add_profiles_args.ri_info_row.CededPercent,
                ))
            add_profiles_args.node_layer_profile_map[
                (add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

        else:
            nodes_all = anytree.search.findall(
                   add_profiles_args.program_node, filter_=lambda node: node.level_id == 2)

            for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
                profile_id = profile_id + 1

                add_profiles_args.fmprofiles_list.append(
                    oed.get_pass_through_profile(
                        profile_id,
                    )
                )

                # Filter
                selected_nodes = self._filter_nodes(nodes_all, ri_scope_row)
                for node in selected_nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

            # add OccLimit / Placed Percent
            profile_id = profile_id + 1
            add_profiles_args.fmprofiles_list.append(
                oed.get_reinsurance_profile(
                    profile_id,
                    attachment=add_profiles_args.ri_info_row.OccAttachment,
                    ceded=add_profiles_args.ri_info_row.CededPercent,
                    limit=add_profiles_args.ri_info_row.OccLimit,
                    placement=add_profiles_args.ri_info_row.PlacedPercent,
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
            oed.get_no_loss_profile(nolossprofile_id))
        profile_id = profile_id + 1
        passthroughprofile_id = profile_id
        fmprofiles_list.append(
            oed.get_pass_through_profile(passthroughprofile_id))

        node_layer_profile_map = {}

        self.logger.debug(fmprofiles_list)


        #
        # Step 1 - Build a tree representation of the insurance program, depening on the reinsuarnce risk level.
        #
        program_node = self._get_tree()



        if self.logger:
            self.logger.debug('program_node tree: "{}"'.format(self.name))
            self.logger.debug(anytree.RenderTree(program_node))
            #Plot tree to image (graphviz)
            #from anytree.dotexport import RenderTreeGraph
            #RenderTreeGraph(program_node).to_picture(
            #    "Init_{}.png".format(self.name))


        #
        # Step 2 - Overlay the reinsurance structure. Each resinsuarnce contact is a seperate layer.
        #
        layer_id = 1        # Current layer ID
        overlay_loop = 0    # Overlays multiple rules in same layer
        prev_reins_number = 1
        for _, ri_info_row in self.ri_info.iterrows():
            overlay_loop += 1
            scope_rows = self.ri_scope[
                (self.ri_scope.ReinsNumber == ri_info_row.ReinsNumber) &
                (self.ri_scope.RiskLevel == self.risk_level)]

            # Three rules for layers
            # 1. if FAC don't inncrement the layer number
            # 2. Otherwise, only increment inline with the reins_number
            # 3. If the reins_number number is the same as prev, increment based on ReinsLayerNumber
            if ri_info_row.ReinsType in ['FAC']:
                pass
            elif prev_reins_number < ri_info_row.ReinsNumber:
                layer_id += 1
                prev_reins_number = ri_info_row.ReinsNumber
            elif layer_id < ri_info_row.ReinsLayerNumber:
                layer_id = ri_info_row.ReinsLayerNumber


            if self.logger:
                pd.set_option('display.width', 1000)
                self.logger.debug('ri_scope: "{}"'.format(self.name))
                self.logger.debug(scope_rows)

            if scope_rows.shape[0] == 0:
                continue

            add_profiles_args = self.add_profiles_args(
                program_node, ri_info_row, scope_rows, overlay_loop, layer_id,
                node_layer_profile_map, fmprofiles_list,
                nolossprofile_id, passthroughprofile_id)

            # Add pass through nodes at all levels so that the risks
            # not explicitly covered are unaffected
            for node in anytree.iterators.LevelOrderIter(add_profiles_args.program_node):
                add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.nolossprofile_id
            add_profiles_args.node_layer_profile_map[(
                add_profiles_args.program_node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id

            if ri_info_row.ReinsType == oed.REINS_TYPE_FAC:
                self._add_fac_profiles(add_profiles_args)
            elif ri_info_row.ReinsType == oed.REINS_TYPE_PER_RISK:
                self._add_per_risk_profiles(add_profiles_args)
            elif ri_info_row.ReinsType == oed.REINS_TYPE_QUOTA_SHARE:
                self._add_quota_share_profiles(add_profiles_args)
            elif ri_info_row.ReinsType == oed.REINS_TYPE_SURPLUS_SHARE:
                self._add_surplus_share_profiles(add_profiles_args)
            elif ri_info_row.ReinsType == oed.REINS_TYPE_CAT_XL:
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
                    oed.FmProgramme(
                        from_agg_id=node.agg_id,
                        level_id=node.level_id,
                        to_agg_id=node.parent.agg_id
                    )
                )

        # Note: Pending confirmation from Joh that ReinsLayerNumber is being used correctly
        for layer in range(1,layer_id+1):
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
                            profiles_ids.append(1)
                            pass
                    fm_policytcs_list.append(oed.FmPolicyTc(
                        layer_id=layer,
                        level_id=node.level_id - 1,
                        agg_id=node.agg_id,
                        profile_id=max(profiles_ids)
                    ))
        self.fmprogrammes = pd.DataFrame(fmprogrammes_list)
        self.fmprofiles = pd.DataFrame(fmprofiles_list)
        self.fm_policytcs = pd.DataFrame(fm_policytcs_list)
        self.fm_xrefs['layer_id'] = pd.Series(layer_id, range(len(self.fm_xrefs.index)))


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
        os.makedirs(directory)

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
