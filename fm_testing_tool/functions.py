# Note that only the currently used fields are shown unless show_all is set to True.
import os
import pandas as pd
import anytree
from anytree.search import find
from anytree.exporter import DotExporter
import collections

PolicyTuple = collections.namedtuple('PolicyTuple','layer_id agg_id calc_rules')
CalcRuleTuple = collections.namedtuple('CalcRuleTuple', 'policytc_id calcrule_id is_step trig_start trig_end')

def load_df(path, required_file=None):
    if path:
        return  pd.read_csv(path)
    else:
        if required_file:
            raise FileNotFoundError(f"Required File does not exist: {required_file}")
        else:
            return None

def create_fm_tree(fm_programme_df, fm_policytc_df, fm_profile_df, fm_summary_df):
    missing_node_link = False 

    def get_policy_tc(agg_id, level_id):
        policytc = fm_policytc_df.loc[
            (fm_policytc_df['agg_id'] == agg_id) & (fm_policytc_df['level_id'] == level_id)
        ]

        policy_list = []
        for _, policy in policytc.iterrows():

            # Find calc_rule
            profile = fm_profile_df.loc[fm_profile_df.policytc_id == policy.policytc_id]
            calc_rules = []
            for _, step in profile.iterrows():
                trig_start = step.trigger_start if hasattr(step, 'trigger_start') else 0
                trig_end = step.trigger_end if hasattr(step, 'trigger_end') else 0
                is_step_rule = (trig_end > 0 or trig_start > 0)

                calc_rules.append(CalcRuleTuple(
                    policytc_id=int(policy.policytc_id),
                    calcrule_id=int(step.calcrule_id),
                    is_step=is_step_rule,
                    trig_start=trig_start,
                    trig_end=trig_end,
                ))

            policy_list.append(
                PolicyTuple(
                    layer_id=int(policy.layer_id),
                    agg_id=int(policy.agg_id),
                    calc_rules=calc_rules,
                )
            )
        return len(policytc), policy_list

    level_ids = sorted(list(fm_programme_df.level_id.unique()), reverse=True)
    root = anytree.Node('Insured Loss', agg_id=1, level_id=max(level_ids)+1, policy_tc=None)

    for level in level_ids:
        agg_id_idxs = list(fm_programme_df[fm_programme_df.level_id == level].drop_duplicates(subset=['level_id','to_agg_id'], keep="first").index)

        for node_idx in agg_id_idxs:
            node_info = fm_programme_df.iloc[node_idx]
            layer_max, policy_list = get_policy_tc(node_info.to_agg_id, node_info.level_id)

            # Set parent node as root or find based on level/agg ids
            if level == max(level_ids):
                parent_node = root
            else:
                try:
                    matched_id = fm_programme_df.loc[(fm_programme_df.level_id == level+1) & (fm_programme_df.from_agg_id == node_info.to_agg_id)].to_agg_id.item()
                    parent_node = find(root, filter_=lambda node: node.level_id == level+1 and node.agg_id == matched_id)
                except ValueError:
                    missing_node_link = True
                    print('Missing node link: agg_id={}, level_id={}'.format(node_info.to_agg_id,level+1)) 

            # Set node names based on attrs in FM files
            if level >= 3:
                node_name = "policy term {} \nlevel: {}".format(
                    node_info.to_agg_id,
                    node_info.level_id
                )
            elif level == 2:
                node_name = "loc term {} ".format(node_info.to_agg_id)
            else:
                node_name = "cov term {}".format(node_info.to_agg_id)

            for policy in policy_list:
                node_name += "\n\nlayer_id: {}".format(policy.layer_id)
                for rule in policy.calc_rules:
                    if rule.is_step:
                        node_name += "\n   policytc_id {}: step_rule:{}, start:{} end:{}".format(
                            rule.policytc_id,
                            rule.calcrule_id,
                            rule.trig_start,
                            rule.trig_end
                        )
                    else:
                        node_name += "\npolicytc_id: {} \ncalc_rule: {}".format(
                            rule.policytc_id,
                            rule.calcrule_id,
                        )

            # Create Node in FM tree
            node = anytree.Node(
                    node_name,
                    agg_id=node_info.to_agg_id,
                    level_id=level,
                    parent=parent_node,
                    layer_max=layer_max,
                    policy_tc=policy_list,
            )

    # Add item level data
    item_agg_idx = list(fm_summary_df[['agg_id']].drop_duplicates().index)
    for item in item_agg_idx:
        item_info = fm_summary_df.iloc[item]
        matched_id = fm_programme_df.loc[(fm_programme_df.level_id == 1) & (fm_programme_df.from_agg_id == item_info.agg_id)].to_agg_id.item()
        parent_node = find(root, filter_=lambda node: node.level_id == 1 and node.agg_id == matched_id)

        node_name = "\n".join([
           "item {}\n".format(int(item_info.agg_id)),
           "locnumber: {}".format(item_info.locnumber),
            "accnumber: {}".format(item_info.accnumber),
            "polnumber: {}".format(item_info.polnumber),
            "portnumber: {}".format(item_info.portnumber),
            "cov_type: {}".format(item_info.coverage_type_id),
            "peril_id: {}".format(item_info.peril_id),
            "tiv: {}".format(item_info.tiv),
        ])

        node = anytree.Node(
            node_name,
            agg_id=item_info.agg_id,
            level_id=0,
            parent=parent_node,
            locnumber=item_info.locnumber,
            accnumber=item_info.accnumber,
            polnumber=item_info.polnumber,
            portnumber=item_info.polnumber,
            tiv=item_info.tiv,
            coverage_id=item_info.coverage_id,
            coverage_type=item_info.coverage_type_id,
            peril_id=item_info.peril_id,
        )
    return root, missing_node_link


def render_fm_tree(root_node, filename='tree.png'):

    # Function to format nodes in FM tree
    def format_box(node):
        # https://graphviz.org/doc/info/shapes.html
        if node.level_id == 0:
            # Item Level Node
            return "fixedsize=false, shape=rect, fillcolor=lightgrey, style=filled"
        else:
            if not node.policy_tc:
                # Error? missing policy_tc entry for this Node
                return "fixedsize=false, shape=ellipse, fillcolor=pink, style=filled"

            elif len(node.policy_tc) > 1:
                # Node with multiple layers
                return "fixedsize=false, shape=rect, fillcolor=orange, style=filled"
            else:
                # Cov or loc nodes
                return "fixedsize=false, shape=ellipse, fillcolor=lightblue, style=filled"

    # Function to add weighted 'by layer number' edges
    def layered_edge(node, child):
        # https://anytree.readthedocs.io/en/latest/tricks/weightededges.html
        if hasattr(child, 'layer_max'):
            if child.layer_max > 1:
                return 'dir=back, style=bold, label=" {} Layers"'.format(child.layer_max)
        return "dir=back"

    # Render tree to png
    dot_data = DotExporter(
        root_node,
        edgeattrfunc=layered_edge,
        nodeattrfunc=format_box)
    dot_data.to_picture(filename)
