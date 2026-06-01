__all__ = [
    'compute_portfolio_complexity',
    'format_complexity_report',
]

import json
import logging
import os
from collections import Counter

import numpy as np

from oasislmf.pytools.common.data import (
    fm_policytc_dtype, fm_profile_dtype, fm_profile_step_dtype,
    fm_programme_dtype, fm_xref_dtype, items_dtype, load_as_array,
    load_as_ndarray, oasis_float,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_fm_files(path):
    """Return True if *path* contains fm_programme .bin or .csv."""
    return (
        os.path.isfile(os.path.join(path, 'fm_programme.bin')) or
        os.path.isfile(os.path.join(path, 'fm_programme.csv'))
    )


def _load_fm_arrays(path):
    """Load raw FM arrays from *path*.  items/coverages are not required."""
    programme = load_as_ndarray(path, 'fm_programme', fm_programme_dtype)
    policytc = load_as_ndarray(path, 'fm_policytc', fm_policytc_dtype)
    profile = load_as_ndarray(path, 'fm_profile_step', fm_profile_step_dtype, must_exist=False)
    if len(profile) == 0:
        profile = load_as_ndarray(path, 'fm_profile', fm_profile_dtype)
        stepped = False
    else:
        stepped = True
    xref = load_as_ndarray(path, 'fm_xref', fm_xref_dtype)
    return programme, policytc, profile, stepped, xref


def _fm_section(programme, policytc, profile, stepped, xref):
    """Compute the fm_structure and profile_complexity dicts from raw arrays."""
    # FM structure
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
            avg_children_per_level[lvl] = (
                round(n_edges / n_parent_nodes, 2) if n_parent_nodes > 0 else 1.0
            )

        total_fm_nodes = sum(nodes_per_level.values())
        overall_avg = round(float(np.mean(list(avg_children_per_level.values()))), 2)
    else:
        num_fm_levels = 0
        nodes_per_level = {}
        avg_children_per_level = {}
        total_fm_nodes = 0
        overall_avg = 1.0

    num_fm_layers = int(policytc['layer_id'].max()) if len(policytc) > 0 else 1

    fm_structure = {
        'num_fm_levels': num_fm_levels,
        'num_fm_layers': num_fm_layers,
        'nodes_per_level': nodes_per_level,
        'total_fm_nodes': total_fm_nodes,
        'avg_children_per_node': overall_avg,
        'avg_children_per_level': avg_children_per_level,
    }

    # Profile complexity
    if len(profile) > 0:
        num_unique_profiles = int(np.unique(profile['profile_id']).size)
        calcrule_counts = Counter(int(x) for x in profile['calcrule_id'])
        calcrule_distribution = {str(k): v for k, v in sorted(calcrule_counts.items())}
    else:
        num_unique_profiles = 0
        calcrule_distribution = {}

    profile_complexity = {
        'num_unique_profiles': num_unique_profiles,
        'has_stepped_profiles': bool(stepped),
        'calcrule_distribution': calcrule_distribution,
    }

    outputs = {'num_output_ids': int(len(xref))}

    fm_work_units = total_fm_nodes * num_fm_layers

    return fm_structure, profile_complexity, outputs, fm_work_units


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_portfolio_complexity(static_path):
    """Compute portfolio complexity metrics from Oasis static files.

    Works for GUL-only, IL, and IL+RI portfolios.

    **GUL-only** (no account file):  only ``portfolio_dimensions`` and the
    ``gul_work_units`` complexity score are populated.

    **IL** (account file present):  additionally populates ``il`` with the FM
    structure, profile complexity, and output count for the insured-loss layer.

    **RI** (reinsurance files present):  additionally populates ``ri`` with
    per-layer FM metrics and a total RI FM work-unit count.

    The two key per-event cost proxies in ``complexity_scores`` are:

    * ``gul_work_units = num_items``
    * ``fm_work_units = il_fm_work_units + ri_fm_work_units``
      where ``il_fm_work_units = total_il_nodes × il_layers`` and
      ``ri_fm_work_units = Σ(total_ri_nodes_layer_k × ri_layers_k)``

    Multiplying by the number of model events gives a per-dimension
    full-run cost proxy.

    Parameters
    ----------
    static_path : str
        Directory containing the Oasis static files.  For IL this is the
        directory with ``fm_programme.bin`` etc.  RI layer subdirectories
        are discovered automatically via ``ri_layers.json``.

    Returns
    -------
    dict
    """
    metrics = {}

    # ------------------------------------------------------------------
    # Portfolio dimensions (GUL — always available)
    # ------------------------------------------------------------------
    items = load_as_ndarray(static_path, 'items', items_dtype, must_exist=False)
    coverages = load_as_array(static_path, 'coverages', oasis_float, must_exist=False)

    if len(items) > 0:
        num_items = int(np.unique(items['item_id']).size)
    else:
        num_items = 0

    num_coverages = int(len(coverages)) if len(coverages) > 0 else None

    metrics['portfolio_dimensions'] = {
        'num_items': num_items,
        'num_coverages': num_coverages,
    }

    il_fm_work_units = 0
    ri_fm_work_units = 0

    # ------------------------------------------------------------------
    # IL FM structure (present only when an account file was supplied)
    # ------------------------------------------------------------------
    if _has_fm_files(static_path):
        programme, policytc, profile, stepped, xref = _load_fm_arrays(static_path)
        fm_structure, profile_complexity, outputs, il_fm_work_units = _fm_section(
            programme, policytc, profile, stepped, xref
        )
        metrics['il'] = {
            'fm_structure': fm_structure,
            'profile_complexity': profile_complexity,
            'outputs': outputs,
        }

    # ------------------------------------------------------------------
    # RI FM structure (one entry per reinsurance layer)
    # ------------------------------------------------------------------
    ri_layers_json = os.path.join(static_path, 'ri_layers.json')
    if os.path.exists(ri_layers_json):
        with open(ri_layers_json) as fh:
            ri_layers = json.load(fh)

        layer_metrics = []
        for layer_key in sorted(ri_layers.keys(), key=lambda k: int(k)):
            layer_dir = ri_layers[layer_key]['directory']
            if not _has_fm_files(layer_dir):
                continue
            prog, ptc, prof, stepped, xref = _load_fm_arrays(layer_dir)
            fm_structure, profile_complexity, outputs, layer_wu = _fm_section(
                prog, ptc, prof, stepped, xref
            )
            ri_fm_work_units += layer_wu
            layer_metrics.append({
                'layer': int(layer_key),
                'fm_structure': fm_structure,
                'profile_complexity': profile_complexity,
                'outputs': outputs,
                'fm_work_units': layer_wu,
            })

        if layer_metrics:
            metrics['ri'] = {
                'num_ri_layers': len(layer_metrics),
                'layers': layer_metrics,
                'total_ri_fm_work_units': ri_fm_work_units,
            }

    # ------------------------------------------------------------------
    # Complexity scores
    # ------------------------------------------------------------------
    metrics['complexity_scores'] = {
        'gul_work_units': num_items,
        'il_fm_work_units': il_fm_work_units,
        'ri_fm_work_units': ri_fm_work_units,
        'total_fm_work_units': il_fm_work_units + ri_fm_work_units,
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
    il_ = metrics.get('il')
    ri_ = metrics.get('ri')
    sc_ = metrics.get('complexity_scores', {})

    num_items = pd_.get('num_items', 0)
    num_covs = pd_.get('num_coverages')
    cov_str = f'{num_covs:,}' if num_covs is not None else 'n/a'

    lines = [
        '',
        '=' * 54,
        'Portfolio Complexity Report',
        '=' * 54,
        '',
        'Portfolio dimensions:',
        f'  Items (GUL):          {num_items:>10,}',
        f'  Coverages:            {cov_str:>10}',
    ]

    def _fm_lines(fm_, prefix=''):
        out = []
        fm_structure = fm_.get('fm_structure', {})
        pr_ = fm_.get('profile_complexity', {})
        op_ = fm_.get('outputs', {})

        num_levels = fm_structure.get('num_fm_levels', 0)
        num_layers = fm_structure.get('num_fm_layers', 1)
        nodes_per_level = fm_structure.get('nodes_per_level', {})
        total_nodes = fm_structure.get('total_fm_nodes', 0)
        avg_children = fm_structure.get('avg_children_per_node', 1.0)
        num_profiles = pr_.get('num_unique_profiles', 0)
        stepped = pr_.get('has_stepped_profiles', False)
        calcrules = pr_.get('calcrule_distribution', {})
        num_outputs = op_.get('num_output_ids', 0)

        nodes_str = ', '.join(f'L{k}: {v:,}' for k, v in sorted(nodes_per_level.items()))
        calcrule_str = ', '.join(f'rule {k}: {v}' for k, v in calcrules.items())

        out += [
            f'{prefix}  FM levels:            {num_levels:>10,}',
            f'{prefix}  FM layers:            {num_layers:>10,}',
            f'{prefix}  Nodes per level:      {nodes_str}',
            f'{prefix}  Total FM nodes:       {total_nodes:>10,}',
            f'{prefix}  Avg children/node:    {avg_children:>10.2f}',
            f'{prefix}  Unique profiles:      {num_profiles:>10,}',
            f'{prefix}  Stepped profiles:     {"Yes" if stepped else "No":>10}',
            f'{prefix}  Calc rules:           {calcrule_str}',
            f'{prefix}  Output IDs:           {num_outputs:>10,}',
        ]
        return out

    if il_:
        lines += ['', 'IL financial module structure:']
        lines += _fm_lines(il_)

    if ri_:
        num_ri = ri_.get('num_ri_layers', 0)
        lines += ['', f'RI financial module structure ({num_ri} layer{"s" if num_ri != 1 else ""}):']
        for layer_info in ri_.get('layers', []):
            lines.append(f'  Layer {layer_info["layer"]}:')
            lines += _fm_lines(layer_info, prefix='  ')

    lines += [
        '',
        'Complexity heuristics (per model event):',
        f'  GUL work units:       {sc_.get("gul_work_units", 0):>10,}',
    ]
    if il_:
        lines.append(f'  IL  work units:       {sc_.get("il_fm_work_units", 0):>10,}  (IL nodes × layers)')
    if ri_:
        lines.append(f'  RI  work units:       {sc_.get("ri_fm_work_units", 0):>10,}  (Σ RI nodes × layers)')
    if il_ or ri_:
        lines.append(f'  Total FM work units:  {sc_.get("total_fm_work_units", 0):>10,}')
    lines += [
        '  (multiply by event count for full-run cost proxy)',
        '',
        '=' * 54,
        '',
    ]

    return '\n'.join(lines)
