__all__ = [
    'ReinsuranceLayer',
    'write_files_for_reinsurance'
]

import json
import logging
import os
import shutil

from collections import namedtuple

import anytree
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
        risk_level_set = set(ri_scope_df[ri_scope_df['ReinsNumber'].isin(reins_numbers)].RiskLevel)

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
        output_dir):
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
    reins_numbers_2 = ri_scope_df[
        ri_scope_df.isin({"ReinsNumber": reins_numbers_1.tolist()}).ReinsNumber
        & (ri_scope_df.RiskLevel == risk_level)
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

        self.add_profiles_args = namedtuple(
            "AddProfilesArgs",
            "program_node, ri_info_row, scope_rows, overlay_loop, layer_id, "
            "node_layer_profile_map, fmprofiles_list, nolossprofile_id, passthroughprofile_id")

    def _add_node(
        self,
        description, parent, level_id, agg_id,
        portfolio_number=oed.NOT_SET_ID, account_number=oed.NOT_SET_ID,
        policy_number=oed.NOT_SET_ID, location_number=oed.NOT_SET_ID,
        location_group=oed.NOT_SET_ID
    ):
        node = anytree.Node(
            description,
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=str(portfolio_number),
            account_number=str(account_number),
            policy_number=str(policy_number),
            location_group=str(location_group),
            location_number=str(location_number)
        )

        return node

    def _add_program_node(self, level_id):
        return self._add_node(
            "Treaty",
            parent=None,
            level_id=level_id,
            agg_id=1)

    def _add_item_node(self, item_id, parent):
        return self._add_node(
            "Item_id:{}".format(item_id),
            parent=parent,
            level_id=1,
            agg_id=item_id)

    def _add_filter_level_node(
            self, level_id, agg_id, xref_description, parent):
        return self._add_node(
            "Portfolio_number:{} Account_number:{} Policy_number:{} Location_number:{}".format(
                xref_description.portnumber,
                xref_description.accnumber,
                xref_description.polnumber,
                xref_description.locnumber),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=xref_description.portnumber,
            account_number=xref_description.accnumber,
            policy_number=xref_description.polnumber,
            location_group=xref_description.locgroup,
            location_number=xref_description.locnumber)

    def _add_location_node(
            self, level_id, agg_id, xref_description, parent):
        return self._add_node(
            "Portfolio_number:{} Account_number:{} Location_number:{}".format(
                xref_description.portnumber,
                xref_description.accnumber,
                xref_description.locnumber),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=xref_description.portnumber,
            account_number=xref_description.accnumber,
            location_group=xref_description.locgroup,
            location_number=xref_description.locnumber)

    def _add_location_group_node(
            self, level_id, agg_id, xref_description, parent):
        return self._add_node(
            "Location_group:{}".format(xref_description.locgroup),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            location_group=xref_description.locgroup)

    def _add_policy_node(
            self, level_id, agg_id, xref_description, parent):
        return self._add_node(
            "Portfolio number:{} Account_number:{} Policy_number:{}".format(
                xref_description.portnumber, xref_description.accnumber, xref_description.polnumber),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=xref_description.portnumber,
            account_number=xref_description.accnumber,
            policy_number=xref_description.polnumber)

    def _add_account_node(
            self, agg_id, level_id, xref_description, parent):
        return self._add_node(
            "Portfolio number:{} Account_number:{}".format(
                xref_description.portnumber, xref_description.accnumber),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=xref_description.portnumber,
            account_number=xref_description.accnumber)

    def _add_portfolio_node(
            self, agg_id, level_id, xref_description, parent):
        return self._add_node(
            "Portfolio number:{}".format(xref_description.portnumber),
            parent=parent,
            level_id=level_id,
            agg_id=agg_id,
            portfolio_number=xref_description.portnumber)

    def _is_valid_id(self, id_to_check):
        is_valid = (
            self._is_defined(id_to_check) and
            (
                (isinstance(id_to_check, str) and id_to_check != "") or
                (isinstance(id_to_check, numbers.Number) and id_to_check > 0)
            )
        )
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

        current_filter_level_node = None
        current_node = None

        risk_level_id = self.LOCATION_RISK_LEVEL + 1
        program_node_level_id = risk_level_id + 1

        program_node = self._add_program_node(program_node_level_id)

        if self.risk_level == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
            xref_descriptions = self.xref_descriptions_df.sort_values(
                by=["locgroup", "portnumber", "accnumber", "polnumber", "locnumber"])
        elif self.risk_level == oed.REINS_RISK_LEVEL_LOCATION:
            xref_descriptions = self.xref_descriptions_df.sort_values(
                by=["portnumber", "accnumber", "locnumber", "polnumber"])
        else:
            xref_descriptions = self.xref_descriptions_df.sort_values(
                by=["portnumber", "accnumber", "polnumber", "locnumber"])

        agg_id = 0
        loc_agg_id = 0

        for row in xref_descriptions.itertuples():

            if self.risk_level == oed.REINS_RISK_LEVEL_PORTFOLIO:
                if current_portfolio_number != row.portnumber:
                    agg_id = agg_id + 1
                    current_node = self._add_portfolio_node(
                        agg_id, risk_level_id, row, program_node)
            elif self.risk_level == oed.REINS_RISK_LEVEL_ACCOUNT:
                if (
                    current_portfolio_number != row.portnumber or
                    current_account_number != row.accnumber
                ):
                    agg_id = agg_id + 1
                    current_node = self._add_account_node(
                        agg_id, risk_level_id, row, program_node)
            elif self.risk_level == oed.REINS_RISK_LEVEL_POLICY:
                if (
                    current_portfolio_number != row.portnumber or
                    current_account_number != row.accnumber or
                    current_policy_number != row.polnumber
                ):
                    agg_id = agg_id + 1
                    current_node = self._add_policy_node(
                        risk_level_id, agg_id, row, program_node)
            elif self.risk_level == oed.REINS_RISK_LEVEL_LOCATION_GROUP:
                if current_location_group != row.locgroup:
                    agg_id = agg_id + 1
                    current_node = self._add_location_group_node(
                        risk_level_id, agg_id, row, program_node)
            elif self.risk_level == oed.REINS_RISK_LEVEL_LOCATION:
                if (
                    current_portfolio_number != row.portnumber or
                    current_account_number != row.accnumber or
                    current_location_number != row.locnumber
                ):
                    agg_id = agg_id + 1
                    current_node = self._add_location_node(
                        risk_level_id, agg_id, row, program_node)
            if (
                current_portfolio_number != row.portnumber or
                current_account_number != row.accnumber or
                current_policy_number != row.polnumber or
                current_location_number != row.locnumber
            ):
                loc_agg_id = loc_agg_id + 1
                level_id = 2
                current_filter_level_node = self._add_filter_level_node(
                    level_id, loc_agg_id, row, current_node)
                current_portfolio_number = row.portnumber
                current_account_number = row.accnumber
                current_policy_number = row.polnumber
                current_location_number = row.locnumber
                current_location_group = row.locgroup

            self._add_item_node(row.output_id, current_filter_level_node)

        return program_node

    def _get_risk_level_id(self):
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
        nodes_filter_level_all = anytree.search.findall(
            add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_filter_level_id())

        add_profiles_args.fmprofiles_list.append(oed.get_reinsurance_profile(
            profile_id,
            attachment=add_profiles_args.ri_info_row.RiskAttachment,
            limit=add_profiles_args.ri_info_row.RiskLimit,
            ceded=add_profiles_args.ri_info_row.CededPercent,
        ))

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
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
            add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_risk_level_id()
        )
        nodes_filter_level_all = anytree.search.findall(
            add_profiles_args.program_node, filter_=lambda node: node.level_id == self._get_filter_level_id()
        )

        for _, ri_scope_row in add_profiles_args.scope_rows.iterrows():
            # Filter
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
            self.logger.debug(self.fm_policytcs_df)
            self.logger.debug('fm_profile: "{}"'.format(self.name))
            self.logger.debug(self.fmprofiles_df)
            self.logger.debug('fm_programme: "{}"'.format(self.name))
            self.logger.debug(self.fmprogrammes_df)

    def _log_tree(self, program_node):
        if self.logger:
            self.logger.debug('program_node tree: "{}"'.format(self.name))
            self.logger.debug(anytree.RenderTree(program_node))

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
        prev_reins_number = -1
        for _, ri_info_row in self.ri_info_df.iterrows():
            overlay_loop += 1
            scope_rows = self.ri_scope_df[
                (self.ri_scope_df.ReinsNumber == ri_info_row.ReinsNumber) &
                (self.ri_scope_df.RiskLevel == self.risk_level)
            ]

            # If FAC, don't increment the layer number
            # Else, only increment inline with the reins_number
            if ri_info_row.ReinsType in ['FAC']:
                pass
            if prev_reins_number == -1:
                prev_reins_number = ri_info_row.ReinsNumber
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
                            profiles_ids.append(node_layer_profile_map[(node.name, layer, overlay_rule)])
                        except Exception:
                            profiles_ids.append(1)
                            pass

                    fm_policytcs_list.append(oed.FmPolicyTc(
                        layer_id=layer,
                        level_id=node.level_id - 1,
                        agg_id=node.agg_id,
                        profile_id=max(profiles_ids)
                    ))
        self.fmprogrammes_df = pd.DataFrame(fmprogrammes_list)
        self.fmprofiles_df = pd.DataFrame(fmprofiles_list)
        self.fm_policytcs_df = pd.DataFrame(fm_policytcs_list)

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
