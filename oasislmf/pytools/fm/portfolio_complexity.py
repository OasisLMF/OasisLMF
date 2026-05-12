__all__ = [
    'compute_portfolio_complexity',
    'format_complexity_report',
]

import logging
from collections import Counter

import numpy as np

from .financial_structure import load_static

logger = logging.getLogger(__name__)


def compute_portfolio_complexity(static_path):
    """Compute portfolio complexity metrics from FM static files.

    Reads the raw FM files (fm_programme, fm_policytc, fm_profile, fm_xref,
    items, coverages) from *static_path* and returns a dict of metrics that
    can be used to estimate the compute requirements of a full model run.

    The key cost drivers captured are:

    * **GUL work units** – proportional to the number of items; this is the
      per-event cost of ground-up loss calculation.
    * **FM work units** – proportional to ``total_fm_nodes × num_fm_layers``
      (the per-event cost of financial-module calculation).

    Multiplying either figure by the number of model events gives a rough
    total-run cost for that dimension.  Other factors (model footprint size,
    output file requests, RI structure) will add further cost on top.

    Parameters
    ----------
    static_path : str
        Directory containing the FM static files (.bin or .csv).

    Returns
    -------
    dict
        Flat-ish dict suitable for JSON serialisation.  Keys:

        portfolio_dimensions
            num_items, num_coverages

        fm_structure
            num_fm_levels, num_fm_layers, nodes_per_level,
            total_fm_nodes, avg_children_per_node,
            avg_children_per_level

        profile_complexity
            num_unique_profiles, has_stepped_profiles,
            calcrule_distribution

        outputs
            num_output_ids

        complexity_scores
            gul_work_units, fm_work_units
    """
    programme, policytc, profile, stepped, xref, items, coverages = load_static(static_path)

    metrics = {}

    # ------------------------------------------------------------------
    # Portfolio dimensions
    # ------------------------------------------------------------------
    if len(items) > 0:
        num_items = int(np.unique(items['item_id']).size)
    else:
        # Fall back: items at the bottom of the programme (level 1 from_agg_ids)
        if len(programme) > 0:
            min_level = int(programme['level_id'].min())
            num_items = int(np.unique(programme[programme['level_id'] == min_level]['from_agg_id']).size)
        else:
            num_items = 0

    num_coverages = int(len(coverages)) if len(coverages) > 0 else None

    metrics['portfolio_dimensions'] = {
        'num_items': num_items,
        'num_coverages': num_coverages,
    }

    # ------------------------------------------------------------------
    # FM structure
    # ------------------------------------------------------------------
    if len(programme) > 0:
        levels = np.unique(programme['level_id'])
        num_fm_levels = int(levels.max())

        nodes_per_level = {}
        avg_children_per_level = {}
        for lvl in levels:
            lvl = int(lvl)
            mask = programme['level_id'] == lvl
            rows = programme[mask]
            n_parent_nodes = int(np.unique(rows['to_agg_id']).size)
            n_edges = int(mask.sum())
            nodes_per_level[lvl] = n_parent_nodes
            avg_children_per_level[lvl] = round(n_edges / n_parent_nodes, 2) if n_parent_nodes > 0 else 1.0

        total_fm_nodes = sum(nodes_per_level.values())
        all_children_counts = [avg_children_per_level[k] for k in avg_children_per_level]
        overall_avg_children = round(float(np.mean(all_children_counts)), 2) if all_children_counts else 1.0
    else:
        num_fm_levels = 0
        nodes_per_level = {}
        avg_children_per_level = {}
        total_fm_nodes = 0
        overall_avg_children = 1.0

    num_fm_layers = int(policytc['layer_id'].max()) if len(policytc) > 0 else 1

    metrics['fm_structure'] = {
        'num_fm_levels': num_fm_levels,
        'num_fm_layers': num_fm_layers,
        'nodes_per_level': nodes_per_level,
        'total_fm_nodes': total_fm_nodes,
        'avg_children_per_node': overall_avg_children,
        'avg_children_per_level': avg_children_per_level,
    }

    # ------------------------------------------------------------------
    # Profile complexity
    # ------------------------------------------------------------------
    if len(profile) > 0:
        num_unique_profiles = int(np.unique(profile['profile_id']).size)
        calcrule_counts = Counter(int(x) for x in profile['calcrule_id'])
        calcrule_distribution = {str(k): v for k, v in sorted(calcrule_counts.items())}
    else:
        num_unique_profiles = 0
        calcrule_distribution = {}

    metrics['profile_complexity'] = {
        'num_unique_profiles': num_unique_profiles,
        'has_stepped_profiles': bool(stepped),
        'calcrule_distribution': calcrule_distribution,
    }

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    metrics['outputs'] = {
        'num_output_ids': int(len(xref)),
    }

    # ------------------------------------------------------------------
    # Complexity heuristics
    # ------------------------------------------------------------------
    # GUL work is proportional to number of items per event.
    # FM work is proportional to (total nodes) × (layers) per event.
    # Multiply by number of model events to get a full-run cost proxy.
    gul_work_units = num_items
    fm_work_units = total_fm_nodes * num_fm_layers

    metrics['complexity_scores'] = {
        'gul_work_units': gul_work_units,
        'fm_work_units': fm_work_units,
    }

    return metrics


def format_complexity_report(metrics):
    """Return a human-readable text summary of portfolio complexity metrics.

    Parameters
    ----------
    metrics : dict
        As returned by :func:`compute_portfolio_complexity`.

    Returns
    -------
    str
    """
    pd_ = metrics.get('portfolio_dimensions', {})
    fm_ = metrics.get('fm_structure', {})
    pr_ = metrics.get('profile_complexity', {})
    op_ = metrics.get('outputs', {})
    sc_ = metrics.get('complexity_scores', {})

    num_items = pd_.get('num_items', 0)
    num_covs = pd_.get('num_coverages')
    num_levels = fm_.get('num_fm_levels', 0)
    num_layers = fm_.get('num_fm_layers', 1)
    nodes_per_level = fm_.get('nodes_per_level', {})
    total_nodes = fm_.get('total_fm_nodes', 0)
    avg_children = fm_.get('avg_children_per_node', 1.0)
    num_profiles = pr_.get('num_unique_profiles', 0)
    stepped = pr_.get('has_stepped_profiles', False)
    calcrules = pr_.get('calcrule_distribution', {})
    num_outputs = op_.get('num_output_ids', 0)
    gul_wu = sc_.get('gul_work_units', 0)
    fm_wu = sc_.get('fm_work_units', 0)

    cov_str = f'{num_covs:,}' if num_covs is not None else 'n/a'
    nodes_str = ', '.join(f'L{k}: {v:,}' for k, v in sorted(nodes_per_level.items()))
    calcrule_str = ', '.join(f'rule {k}: {v}' for k, v in calcrules.items())

    lines = [
        '',
        '=' * 54,
        'Portfolio Complexity Report',
        '=' * 54,
        '',
        'Portfolio dimensions:',
        f'  Items (GUL):          {num_items:>10,}',
        f'  Coverages:            {cov_str:>10}',
        '',
        'Financial module (IL) structure:',
        f'  FM levels:            {num_levels:>10,}',
        f'  FM layers:            {num_layers:>10,}',
        f'  Nodes per level:      {nodes_str}',
        f'  Total FM nodes:       {total_nodes:>10,}',
        f'  Avg children/node:    {avg_children:>10.2f}',
        '',
        'Profile complexity:',
        f'  Unique profiles:      {num_profiles:>10,}',
        f'  Stepped profiles:     {"Yes" if stepped else "No":>10}',
        f'  Calc rules:           {calcrule_str}',
        '',
        'Outputs:',
        f'  Output IDs:           {num_outputs:>10,}',
        '',
        'Complexity heuristics (per model event):',
        f'  GUL work units:       {gul_wu:>10,}',
        f'  FM  work units:       {fm_wu:>10,}  (nodes × layers)',
        '  (multiply by event count for full-run cost proxy)',
        '',
        '=' * 54,
        '',
    ]
    return '\n'.join(lines)
