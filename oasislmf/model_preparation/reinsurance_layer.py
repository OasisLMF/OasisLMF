# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

__all__ = [
    'generate_xref_descriptions',
    'generate_files_for_reinsurance',
    'ReinsuranceLayer',
    'write_ri_input_files'
]

import json
import logging
import math
import os
import shutil
import subprocess32 as subprocess

from collections import namedtuple
from itertools import product

import anytree
import numbers
import pandas as pd

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from . import oed

from six import string_types

# Metadata about an inuring layer
InuringLayer = namedtuple(
    "InuringLayer",
    "inuring_priority reins_numbers is_valid validation_messages")



def _get_location_tiv(location, coverage_type_id):
    switcher = {
        oed.BUILDING_COVERAGE_TYPE_ID: location.get('BuildingTIV', 0),
        oed.OTHER_BUILDING_COVERAGE_TYPE_ID: location.get('OtherTIV', 0),
        oed.CONTENTS_COVERAGE_TYPE_ID: location.get('ContentsTIV', 0),
        oed.TIME_COVERAGE_TYPE_ID: location.get('BITIV', 0)
    }
    return switcher.get(coverage_type_id, 0)


def generate_xref_descriptions(accounts_fp, locations_fp):

    accounts = pd.read_csv(accounts_fp)
    locations = pd.read_csv(locations_fp)
    coverage_id = 0
    item_id = 0
    group_id = 0
    policy_agg_id = 0
    profile_id = 0

    site_agg_id = 0

    accounts_and_locations = pd.merge(accounts, locations, left_on='AccNumber', right_on='AccNumber')

    for acc_and_loc, coverage_type, peril in product((acc for _, acc in accounts_and_locations.iterrows()), oed.COVERAGE_TYPES, oed.PERILS):

        tiv = _get_location_tiv(acc_and_loc, coverage_type)

        if tiv > 0:
            policy_agg_id += 1
            profile_id += 1
            group_id += 1
            site_agg_id += 1
            profile_id += 1

            coverage_id += 1
            item_id += 1

            yield oed.XrefDescription(
                xref_id = item_id,
                account_number = acc_and_loc.get('AccNumber'),
                location_number = acc_and_loc.get('LocNumber'),
                location_group = acc_and_loc.get('LocGroup'),
                cedant_name = acc_and_loc.get('CedantName'),
                producer_name = acc_and_loc.get('ProducerName'),
                lob = acc_and_loc.get('LOB'),
                country_code = acc_and_loc.get('CountryCode'),
                reins_tag = acc_and_loc.get('ReinsTag'),
                coverage_type_id = coverage_type,
                peril_id = peril,
                policy_number = acc_and_loc.get('PolNumber'),
                portfolio_number = acc_and_loc.get('PortNumber'),
                tiv = tiv
            )


@oasis_log
def generate_files_for_reinsurance(
        items,
        coverages,
        fm_xrefs,
        xref_descriptions,
        ri_info_df,
        ri_scope_df,
        direct_oasis_files_dir,
        gulsummaryxref=pd.DataFrame(),
        fmsummaryxref=pd.DataFrame()):

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
                items,
                coverages,
                fm_xrefs,
                xref_descriptions,
                gulsummaryxref,
                fmsummaryxref,
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
        items,
        coverages,
        fm_xrefs,
        xref_descriptions,
        gulsummaryxref,
        fmsummaryxref,
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
        ri_scope_df.isin({"ReinsNumber": reins_numbers_1.tolist()}).ReinsNumber
        & (ri_scope_df.RiskLevel == risk_level)].ReinsNumber
    if reins_numbers_2.empty:
        return None

    ri_info_inuring_priority_df = ri_info_df[ri_info_df.isin(
        {"ReinsNumber": reins_numbers_2.tolist()}).ReinsNumber]
    output_name = "ri_{}_{}".format(inuring_priority, risk_level)
    reinsurance_layer = ReinsuranceLayer(
        name=output_name,
        ri_info=ri_info_inuring_priority_df,
        ri_scope=ri_scope_df,
        items=items,
        coverages=coverages,
        fm_xrefs=fm_xrefs,
        xref_descriptions=xref_descriptions,
        gulsummaryxref=gulsummaryxref,
        fmsummaryxref=fmsummaryxref,
        risk_level=risk_level
    )

    reinsurance_layer.generate_oasis_structures()
    output_dir = os.path.join(direct_oasis_files_dir, "RI_{}".format(reinsurance_index))
    reinsurance_layer.write_oasis_files(output_dir)
    return output_dir


@oasis_log
def write_ri_input_files(
        exposure_fp,
        accounts_fp,
        items_fp,
        coverages_fp,
        gulsummaryxref_fp,
        fm_xref_fp,
        fmsummaryxref_fp,
        ri_info_fp,
        ri_scope_fp,
        target_dir
    ):
    xref_descriptions = pd.DataFrame(generate_xref_descriptions(accounts_fp, exposure_fp))
    return generate_files_for_reinsurance(
        pd.read_csv(items_fp),
        pd.read_csv(coverages_fp),
        pd.read_csv(fm_xref_fp),
        xref_descriptions,
        pd.read_csv(ri_info_fp),
        pd.read_csv(ri_scope_fp),
        target_dir,
        gulsummaryxref=pd.read_csv(gulsummaryxref_fp),
        fmsummaryxref=pd.read_csv(fmsummaryxref_fp)
    )


class ReinsuranceLayer(object):
    """
    Generates ktools inputs and runs financial module for a reinsurance structure.
    """

    def __init__(self, 
        name, ri_info, ri_scope, items, coverages, fm_xrefs, 
        xref_descriptions, risk_level, fmsummaryxref=pd.DataFrame(), gulsummaryxref=pd.DataFrame(), logger=None):

        self.logger = logger or logging.getLogger()
        self.name = name

        self.coverages = coverages
        self.items = items
        self.fm_xrefs = fm_xrefs
        self.xref_descriptions = xref_descriptions
        self.fmsummaryxref = fmsummaryxref
        self.gulsummaryxref = gulsummaryxref

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

    def _add_node(self, description, parent, level_id, agg_id,
            portfolio_number=oed.NOT_SET_ID, account_number=oed.NOT_SET_ID,
            policy_number=oed.NOT_SET_ID, location_number=oed.NOT_SET_ID,
            location_group=oed.NOT_SET_ID):

        node = anytree.Node(
            description,
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=portfolio_number,
            account_number=account_number,
            policy_number=policy_number,
            location_group=location_group,
            location_number=location_number)
        
        return node

    def _add_program_node(self, level_id):
        return self._add_node(
            "Treaty",
            parent=None,
            level_id=level_id,
            agg_id=1)

    def _add_item_node(self, xref_id, parent):
        return self._add_node(
            "Item_id:{}".format(xref_id),
            parent=parent,
            level_id=1,
            agg_id=xref_id)

    def _add_location_node(
            self, level_id, agg_id, xref_description, parent):
        return self._add_node(
            "Portfolio_number:{} Account_number:{} Policy_number:{} Location_number:{}".format(
                xref_description.portfolio_number,
                xref_description.account_number,
                xref_description.policy_number,
                xref_description.location_number),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=xref_description.portfolio_number,
            account_number=xref_description.account_number,
            policy_number=xref_description.policy_number,
            location_group=xref_description.location_group,
            location_number=xref_description.location_number)

    def _add_location_group_node(
            self, level_id, agg_id, xref_description, parent):
        return self._add_node(
            "Location_group:{}".format(xref_description.location_group),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            location_group=xref_description.location_group)

    def _add_policy_node(
            self, level_id, agg_id, xref_description, parent):
        return self._add_node(
            "Portfolio number:{} Account_number:{} Policy_number:{}".format(
                xref_description.portfolio_number, xref_description.account_number, xref_description.policy_number),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=xref_description.portfolio_number,
            account_number=xref_description.account_number,
            policy_number=xref_description.policy_number)

    def _add_account_node(
            self, agg_id, level_id, xref_description, parent):
        return self._add_node(
            "Portfolio number:{} Account_number:{}".format(
                xref_description.portfolio_number, xref_description.account_number),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=xref_description.portfolio_number,    
            account_number=xref_description.account_number)

    def _add_portfolio_node(
            self, agg_id, level_id, xref_description, parent):
        return self._add_node(
            "Portfolio number:{}".format(xref_description.portfolio_number),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=xref_description.portfolio_number)

    def _is_valid_id(self, id_to_check):
        is_valid = self._is_defined(id_to_check) and \
            ((isinstance(id_to_check, string_types) and id_to_check != "")
            or 
            (isinstance(id_to_check, numbers.Number) and id_to_check > 0))
        return is_valid

    def _match_portfolio(self, node, scope_row, exact=False):
        if self._is_valid_id(scope_row.PortNumber):
            return node.portfolio_number == scope_row.PortNumber
        else:
            return True

    def _match_account(self, node, scope_row, exact=False):
        match = False
        if exact:
            match = self._match_portfolio(node, scope_row) and node.account_number == scope_row.AccNumber
        else:
            if (self._is_valid_id(scope_row.PortNumber) and self._is_valid_id(scope_row.AccNumber)):
                match = self._match_portfolio(node, scope_row) and node.account_number == scope_row.AccNumber
            else:
                match = self._match_portfolio(node, scope_row)
        return match

    def _match_policy(self, node, scope_row, exact=False):
        match = False
        if exact:
            match = self._match_account(node, scope_row) and node.policy_number == scope_row.PolNumber
        else:
            if (self._is_valid_id(scope_row.PolNumber) and self._is_valid_id(scope_row.AccNumber) and self._is_valid_id(scope_row.PortNumber)):
                match = self._match_account(node, scope_row) and node.policy_number == scope_row.PolNumber
            else:
                match = self._match_account(node, scope_row)
        return match

    def _match_location(self, node, scope_row, exact=False):
        match = False
        if self._is_valid_id(scope_row.PolNumber):
            if exact:
                match = self._match_policy(node, scope_row) and node.location_number == scope_row.LocNumber
            else:
                if self._is_valid_id(scope_row.LocNumber) and self._is_valid_id(scope_row.AccNumber) and self._is_valid_id(scope_row.PortNumber):
                    match = self._match_policy(node, scope_row) and node.location_number == scope_row.LocNumber
                else:
                    match = self._match_policy(node, scope_row)
        else:
            if exact:
                match = self._match_account(node, scope_row) and node.location_number == scope_row.LocNumber
            else:
                if self._is_valid_id(scope_row.LocNumber) and self._is_valid_id(scope_row.AccNumber) and self._is_valid_id(scope_row.PortNumber):
                    match = self._match_account(node, scope_row) and node.location_number == scope_row.LocNumber
                else:
                    match = self._match_account(node, scope_row)
        return match

    def _match_location_group(self, node, scope_row, exact=False):
        match = False
        if self._is_valid_id(scope_row.LocGroup):
            match = node.location_group == scope_row.LocGroup
        return match

    def _is_valid_filter(self, value):
        return (value is not None and value != "" and value == value)

    def _match_row(self, node, scope_row):
        match = True
        if match and self._is_valid_filter(scope_row.PortNumber):
            match = node.portfolio_number == scope_row.PortNumber
        if match and self._is_valid_filter(scope_row.AccNumber):
            match = node.account_number == scope_row.AccNumber
        if match and self._is_valid_filter(scope_row.PolNumber):
            match = node.policy_number == scope_row.PolNumber
        if match and self._is_valid_filter(scope_row.LocGroup):
            match = node.location_group == scope_row.LocGroup
        if match and self._is_valid_filter(scope_row.LocNumber):
            match = node.location_number == scope_row.LocNumber
        # if match and self._is_valid_filter(scope_row.CedantName):

        # if match and self._is_valid_filter(scope_row.ProducerName):

        # if match and self._is_valid_filter(scope_row.LOB):

        # if match and self._is_valid_filter(scope_row.CountryCode):

        # if match and self._is_valid_filter(scope_row.ReinsTag):
        return match


    def _scope_filter(self, nodes_list, scope_row, exact=False):
        """
        Return subset of `nodes_list` based on values of a row in `ri_scope.csv`
        """

        filtered_nodes_list = list(filter(
            lambda n: self._match_row(n, scope_row), 
            nodes_list))
        return filtered_nodes_list

    def _risk_level_filter(self, nodes_list, scope_row, exact=False):
        """
        Return subset of `nodes_list` based on values of a row in `ri_scope.csv`
        """

        if (scope_row.RiskLevel == oed.REINS_RISK_LEVEL_PORTFOLIO):
            return list(filter(
                lambda n: self._match_portfolio(n, scope_row, exact), 
                nodes_list))
        elif (scope_row.RiskLevel == oed.REINS_RISK_LEVEL_ACCOUNT):
            return list(filter(
                lambda n: self._match_account(n, scope_row, exact), 
                nodes_list))
        elif scope_row.RiskLevel == oed.REINS_RISK_LEVEL_POLICY:
            nodes_list = list(filter(
                lambda n: self._match_policy(n, scope_row, exact), 
                nodes_list))
        elif scope_row.RiskLevel == oed.REINS_RISK_LEVEL_LOCATION:
            nodes_list = list(filter(
                lambda n: self._match_location(n, scope_row, exact), 
                nodes_list))
        elif scope_row.RiskLevel == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
            nodes_list = list(filter(
                lambda n: self._match_location_group(n, scope_row, exact), 
                nodes_list))
        else:
            raise OasisException("Unknown risk level: {}".format(scope_row.RiskLevel))
        
        return nodes_list


    def _is_defined(self, num_to_check):
        # If the value = NaN it will return False
        return num_to_check == num_to_check

    def _check_scope_row(self, scope_row):
        # For some treaty types the scope filter much match exactly
        okay = True
        if (scope_row.RiskLevel == oed.REINS_RISK_LEVEL_ACCOUNT):
            okay = \
                self._is_valid_id(scope_row.AccNumber) and \
                not self._is_valid_id(scope_row.PolNumber) and \
                not self._is_valid_id(scope_row.LocNumber)
        elif scope_row.RiskLevel == oed.REINS_RISK_LEVEL_POLICY:
            okay = \
                self._is_valid_id(scope_row.AccNumber) and \
                self._is_valid_id(scope_row.PolNumber) and \
                not self._is_valid_id(scope_row.LocNumber)
        elif scope_row.RiskLevel == oed.REINS_RISK_LEVEL_LOCATION:
            okay = \
                self._is_valid_id(scope_row.AccNumber) and \
                self._is_valid_id(scope_row.LocNumber)
        elif scope_row.RiskLevel == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
            okay = \
                self._is_valid_id(scope_row.LocGroup)
        return okay

    LOCATION_RISK_LEVEL = 2

    def _get_tree(self):
        current_location_number = 0
        current_policy_number = 0
        current_account_number = 0
        current_portfolio_number = 0
        current_location_group = 0

        current_location_node = None
        current_node = None

        if self.risk_level == oed.REINS_RISK_LEVEL_LOCATION:
            risk_level_id = self.LOCATION_RISK_LEVEL
        else:
            risk_level_id = self.LOCATION_RISK_LEVEL + 1
        program_node_level_id = risk_level_id + 1

        program_node = self._add_program_node(program_node_level_id)

        if self.risk_level == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
            xref_descriptions = self.xref_descriptions.sort_values(
                by=["location_group", "portfolio_number", "account_number", "policy_number", "location_number"])
        else:
            xref_descriptions = self.xref_descriptions.sort_values(
                by=["portfolio_number", "account_number", "policy_number", "location_number"])

        agg_id = 0
        loc_agg_id = 0

        for row in xref_descriptions.itertuples():

            if self.risk_level == oed.REINS_RISK_LEVEL_PORTFOLIO:
                if current_portfolio_number != row.portfolio_number:
                    agg_id = agg_id + 1
                    current_node = self._add_portfolio_node(
                        agg_id, risk_level_id, row, program_node)
            elif self.risk_level == oed.REINS_RISK_LEVEL_ACCOUNT:
                if \
                current_portfolio_number != row.portfolio_number or \
                current_account_number != row.account_number:
                    agg_id = agg_id + 1
                    current_node = self._add_account_node(
                        agg_id, risk_level_id, row, program_node)
            elif self.risk_level == oed.REINS_RISK_LEVEL_POLICY:
                if \
                current_portfolio_number != row.portfolio_number or \
                current_account_number != row.account_number or \
                current_policy_number != row.policy_number:
                    agg_id = agg_id + 1
                    current_node = self._add_policy_node(
                        risk_level_id, agg_id, row, program_node)
            elif self.risk_level == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
                if current_location_group != row.location_group:
                    agg_id = agg_id + 1
                    current_node = self._add_location_group_node(
                        risk_level_id, agg_id, row, program_node)

            if \
            current_portfolio_number != row.portfolio_number or \
            current_account_number != row.account_number or \
            current_policy_number != row.policy_number or \
            current_location_number != row.location_number:
                loc_agg_id = loc_agg_id + 1
                level_id = 2
                if self.risk_level == oed.REINS_RISK_LEVEL_LOCATION:
                    current_location_node = self._add_location_node(
                        level_id, loc_agg_id, row, program_node)
                else:
                    current_location_node = self._add_location_node(
                        level_id, loc_agg_id, row, current_node)
                current_portfolio_number = row.portfolio_number
                current_account_number = row.account_number
                current_policy_number = row.policy_number
                current_location_number = row.location_number
                current_location_group = row.location_group

            self._add_item_node(row.xref_id, current_location_node)
            
        return program_node

    def _get_risk_level_id(self):
        if self.risk_level == oed.REINS_RISK_LEVEL_LOCATION:
            risk_level_id = 2
        else:
            risk_level_id = 3
        return risk_level_id

    def _get_filter_level_id(self):
        risk_level_id = 2
        return risk_level_id

    def _get_next_profile_id(self, add_profiles_args):
        profile_id = max(
            x.profile_id for x in add_profiles_args.fmprofiles_list)
        return profile_id + 1

    def _add_fac_profiles(self, add_profiles_args):
        self.logger.debug("Adding FAC profiles:")

        profile_id = self._get_next_profile_id(add_profiles_args)
        add_profiles_args.fmprofiles_list.append(oed.get_reinsurance_profile(
            profile_id,
            attachment=add_profiles_args.ri_info_row.RiskAttachment,
            limit=add_profiles_args.ri_info_row.RiskLimit,
            ceded=add_profiles_args.ri_info_row.CededPercent,
            placement=add_profiles_args.ri_info_row.PlacedPercent
        ))

        nodes_risk_level_all = anytree.search.findall(
            add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_risk_level_id())

        if self.risk_level != oed.REINS_RISK_LEVEL_LOCATION:
            nodes_filter_level_all = anytree.search.findall(
                add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_filter_level_id())
            for node in nodes_filter_level_all:
                add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            # Note that FAC profiles scope much match the filter exactly.
            if not self._check_scope_row(ri_scope_row):
                raise OasisException("Invalid scope row: {}".format(ri_scope_row))
            nodes = self._risk_level_filter(nodes_risk_level_all, ri_scope_row, exact=True)
            for node in nodes:
                add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id

    def _add_per_risk_profiles(self, add_profiles_args):
        self.logger.debug("Adding PR profiles:")
        profile_id = self._get_next_profile_id(add_profiles_args)
        nodes_risk_level_all = anytree.search.findall(
            add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_risk_level_id())
        if self.risk_level != oed.REINS_RISK_LEVEL_LOCATION:
            nodes_filter_level_all = anytree.search.findall(
                add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_filter_level_id())

        add_profiles_args.fmprofiles_list.append(oed.get_reinsurance_profile(
            profile_id,
            attachment=add_profiles_args.ri_info_row.RiskAttachment,
            limit=add_profiles_args.ri_info_row.RiskLimit,
            ceded=add_profiles_args.ri_info_row.CededPercent,
        ))

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            if self.risk_level != oed.REINS_RISK_LEVEL_LOCATION:
                selected_nodes = self._scope_filter(nodes_filter_level_all, ri_scope_row, exact=False)
                for node in selected_nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id
            selected_nodes = self._risk_level_filter(nodes_risk_level_all, ri_scope_row, exact=False)
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
        profile_id = self._get_next_profile_id(add_profiles_args)
        nodes_risk_level_all = anytree.search.findall(
            add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_risk_level_id())
        if self.risk_level != oed.REINS_RISK_LEVEL_LOCATION:
            nodes_filter_level_all = anytree.search.findall(
                add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_filter_level_id())
            for node in nodes_filter_level_all:
                add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            # Note that surplus share profiles scope much match the filter exactly.
            if not self._check_scope_row(ri_scope_row):
                raise OasisException("Invalid scope row: {}".format(ri_scope_row))
            
            add_profiles_args.fmprofiles_list.append(oed.get_reinsurance_profile(
                profile_id,
                attachment=add_profiles_args.ri_info_row.RiskAttachment,
                limit=add_profiles_args.ri_info_row.RiskLimit,
                ceded=ri_scope_row.CededPercent,
            ))
            selected_nodes = self._risk_level_filter(nodes_risk_level_all, ri_scope_row, exact=True)
            for node in selected_nodes:
                add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = profile_id
            profile_id = profile_id + 1

        # add OccLimit / Placed Percent
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

        profile_id = self._get_next_profile_id(add_profiles_args)
        nodes_risk_level_all = anytree.search.findall(
            add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_risk_level_id())
        if self.risk_level != oed.REINS_RISK_LEVEL_LOCATION:
            nodes_filter_level_all = anytree.search.findall(
                add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_filter_level_id())

        add_profiles_args.fmprofiles_list.append(
            oed.get_reinsurance_profile(
                profile_id,
                limit=add_profiles_args.ri_info_row.RiskLimit,
                ceded=add_profiles_args.ri_info_row.CededPercent,
            ))

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            # Filter
            if self.risk_level != oed.REINS_RISK_LEVEL_LOCATION:
                selected_nodes = self._scope_filter(nodes_filter_level_all, ri_scope_row, exact=False)
                for node in selected_nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id
            selected_nodes = self._risk_level_filter(nodes_risk_level_all, ri_scope_row, exact=False)
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
        self.logger.debug("Adding CAT XL profiles")

        profile_id = self._get_next_profile_id(add_profiles_args)
        nodes_risk_level_all = anytree.search.findall(
            add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_risk_level_id())
        if self.risk_level != oed.REINS_RISK_LEVEL_LOCATION:
            nodes_filter_level_all = anytree.search.findall(
                add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_filter_level_id())
        
        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            # Filter
            if self.risk_level != oed.REINS_RISK_LEVEL_LOCATION:
                selected_nodes = self._scope_filter(nodes_filter_level_all, ri_scope_row, exact=False)
                for node in selected_nodes:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id
            selected_nodes = self._risk_level_filter(nodes_risk_level_all, ri_scope_row, exact=False)
            for node in selected_nodes:
                add_profiles_args.node_layer_profile_map[(
                    node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id

        # Add OccLimit / Placed Percent
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


    def _log_reinsurance_structure(self, add_profiles_args):
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
            self.logger.debug('fm_programme: "{}"'.format(self.name))
            self.logger.debug(self.fmprogrammes)

    def _log_tree(self, program_node):
        if self.logger:
            self.logger.debug('program_node tree: "{}"'.format(self.name))
            self.logger.debug(anytree.RenderTree(program_node))

    def _log_reinsurance_structure(self, add_profiles_args):
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
            self.logger.debug('fm_programme: "{}"'.format(self.name))
            self.logger.debug(self.fmprogrammes)



    def generate_oasis_structures(self):
        '''
        Create the Oasis structures - FM Programmes, FM Profiles and FM Policy TCs -
        that represent the reinsurance structure.

        The algorithm to create the stucture has three steps:
        Step 1 - Build a tree representation of the insurance program, depending on the reinsurance risk level.
        Step 2 - Overlay the reinsurance structure. Each reinsurance contact is a seperate layer.
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
        # Step 1 - Build a tree representation of the insurance program, depening on the reinsurance risk level.
        #
        program_node = self._get_tree()
        self._log_tree(program_node)

        #
        # Step 2 - Overlay the reinsurance structure. Each reinsurance contact is a seperate layer.
        #
        layer_id = 1        # Current layer ID
        overlay_loop = 0    # Overlays multiple rules in same layer
        prev_reins_number = 1
        for _, ri_info_row in self.ri_info.iterrows():
            overlay_loop += 1
            scope_rows = self.ri_scope[
                (self.ri_scope.ReinsNumber == ri_info_row.ReinsNumber)
                & (self.ri_scope.RiskLevel == self.risk_level)]

            # If FAC, don't increment the layer number
            # Else, only increment inline with the reins_number
            if ri_info_row.ReinsType in ['FAC']:
                pass
            elif prev_reins_number < ri_info_row.ReinsNumber:
                layer_id += 1
                prev_reins_number = ri_info_row.ReinsNumber

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

            # Add pass through nodes at all levels so that the risks not explicitly covered are unaffected
            for node in anytree.iterators.LevelOrderIter(add_profiles_args.program_node):
                if self.risk_level == oed.REINS_RISK_LEVEL_LOCATION:
                    add_profiles_args.node_layer_profile_map[(
                        node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.nolossprofile_id
                else:
                    if node.level_id == self._get_risk_level_id():
                        add_profiles_args.node_layer_profile_map[(
                            node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.nolossprofile_id
                    elif node.level_id == self._get_filter_level_id():
                        add_profiles_args.node_layer_profile_map[(
                            node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.nolossprofile_id
                    else:
                        add_profiles_args.node_layer_profile_map[(
                            node.name, add_profiles_args.layer_id, add_profiles_args.overlay_loop)] = add_profiles_args.passthroughprofile_id
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
        for layer in range(1, layer_id + 1):
            for node in anytree.iterators.LevelOrderIter(program_node):
                if node.level_id > 1:
                    profiles_ids = []

                    # Collect over-lapping unique combinations of (layer_id, level_id, agg_id) 
                    # and combine into a single layer
                    for overlay_rule in range(1, overlay_loop + 1):
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

        self._log_reinsurance_structure(add_profiles_args)

    def write_oasis_files(self, directory=None):
        '''
        Write out the generated data to Oasis input file format. 
        '''

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
        self.fmsummaryxref.to_csv(
            os.path.join(directory, "fmsummaryxref.csv"), index=False)
        self.gulsummaryxref.to_csv(
            os.path.join(directory, "gulsummaryxref.csv"), index=False)
