"""Pre-compute and persist gulmc read-only data structures.

Follows the same pattern as ``oasislmf.pytools.fm.financial_structure``:
  - ``create_gulmc_structure`` builds all read-only numpy arrays once and
    saves them as ``.npy`` files.
  - ``load_gulmc_structure`` memory-maps them via ``np.load(mmap_mode='r')``,
    allowing multiple gulmc processes to share physical memory pages through
    the OS page cache.
"""
import logging
import os

import numpy as np
import numpy.lib.recfunctions as rfn
import numba as nb
from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.data import areaperil_int, load_as_ndarray, oasis_int
from oasislmf.utils.exceptions import OasisException
from oasislmf.pytools.common.id_index import build as id_index_build
from oasislmf.pytools.common.input_files import read_coverages, read_correlations
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel.manager import (
    get_damage_bins, get_vulns, get_intensity_bin_dict,
)
from oasislmf.pytools.gul.random import (
    compute_norm_cdf_lookup, compute_norm_inv_cdf_lookup,
    x_min, x_max, norm_inv_N, cdf_min, cdf_max, inv_factor, norm_factor,
)
from oasislmf.pytools.gulmc.aggregate import (
    get_vuln_rngadj,
    process_aggregate_vulnerability, process_vulnerability_weights,
    read_aggregate_vulnerability, read_vulnerability_weights,
)
from oasislmf.pytools.gulmc.common import (
    NormInversionParameters, coverage_type,
)
from oasislmf.pytools.gulmc.items import (
    read_items, generate_item_map,
    build_cdf_group_indices, get_dynamic_footprint_adjustments, get_peril_id,
)
from oasislmf.utils.path import setcwd

logger = logging.getLogger(__name__)

STRUCTURE_DIR = 'gulmc_structure'

# (variable_name, filename) pairs for all arrays that are saved/loaded.
ARRAY_FILES = [
    'items',
    'coverages',
    'item_map_ja_areaperil_ids',
    'item_map_ja_offsets',
    'item_map_ja_vuln_ja_offsets',
    'item_map_ja_vuln_ja_item_idxs',
    'item_map_ja_id_ind',
    'item_cdf_group_idx',
    'areaperil_agg_vuln_idx_ja_offsets',
    'areaperil_agg_vuln_idx_ja_data',
    'damage_bins',
    'vuln_adj',
    'vuln_array',
    'unique_peril_correlation_groups',
    'norm_inv_cdf',
    'norm_cdf',
    'norm_inv_parameters',
    'intensity_bin_peril_ids',
    'intensity_bins',
    'coverage_source_id',
    'coverage_dependents_ja_offsets',
    'coverage_dependents_ja_data',
]


def _validate_acyclic_coverage_dependency(coverage_source_id):
    """Ensure the coverage dependency graph is acyclic.

    Each coverage has a single parent (``coverage_source_id``), so a cycle is a coverage
    reachable from itself by following parents. Only dependent coverages (source > 0) can
    take part in a cycle, so we walk up from each of those.

    Args:
        coverage_source_id (np.ndarray): parent coverage_id per coverage_id (0 = root).

    Raises:
        OasisException: if a cyclic dependency is configured.
    """
    # 0 = unvisited, 1 = on the current path, 2 = known acyclic
    state = np.zeros(len(coverage_source_id), dtype=np.int8)
    for start in np.nonzero(coverage_source_id > 0)[0]:
        if state[start] == 2:
            continue
        path = []
        node = int(start)
        while node != 0 and state[node] == 0:
            state[node] = 1
            path.append(node)
            node = int(coverage_source_id[node])
        if node != 0 and state[node] == 1:
            raise OasisException(
                f"Cyclic coverage dependency detected involving coverage_id {node}; "
                "coverage_dependency_settings must form a directed acyclic graph."
            )
        for nd in path:
            state[nd] = 2


def build_coverage_dependency_forest(items, n_coverages):
    """Build the coverage dependency forest from per-item ``source_coverage_id``.

    Produces ``coverage_source_id`` (indexed by coverage_id, 0 = independent/root) and the
    parent -> dependents jagged array (``coverage_dependents_ja_offsets`` /
    ``coverage_dependents_ja_data``) used by the gulmc DFS push. All items of a coverage
    carry the same source, so a scatter suffices.

    Args:
        items (np.ndarray): items table containing 'coverage_id' and 'source_coverage_id'.
        n_coverages (int): number of coverage slots (coverages.shape[0] == max coverage_id + 1).

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray):
        coverage_source_id (len n_coverages),
        coverage_dependents_ja_offsets (len n_coverages + 1),
        coverage_dependents_ja_data (len = number of dependent coverages).
    """
    # coverage ids are unsigned; reuse the input dtype so the forest matches the items table
    id_dtype = items['source_coverage_id'].dtype
    coverage_source_id = np.zeros(n_coverages, dtype=id_dtype)
    coverage_source_id[items['coverage_id']] = items['source_coverage_id']

    # a source pointing outside the coverage range, or a self-reference, is treated as
    # independent (defensive; should not occur for validly generated inputs)
    coverage_source_id[coverage_source_id >= n_coverages] = 0
    coverage_source_id[coverage_source_id == np.arange(n_coverages)] = 0

    _validate_acyclic_coverage_dependency(coverage_source_id)

    # invert to a parent -> dependents jagged array: dependents grouped by ascending parent
    coverage_dependents_ja_data = np.nonzero(coverage_source_id > 0)[0].astype(id_dtype)
    parents = coverage_source_id[coverage_dependents_ja_data]
    order = np.argsort(parents, kind='stable')
    coverage_dependents_ja_data = coverage_dependents_ja_data[order]
    coverage_dependents_ja_offsets = np.zeros(n_coverages + 1, dtype=oasis_int)
    coverage_dependents_ja_offsets[1:] = np.cumsum(np.bincount(parents, minlength=n_coverages))

    return coverage_source_id, coverage_dependents_ja_offsets, coverage_dependents_ja_data


def _structure_path(run_dir):
    return os.path.join(run_dir, 'input', STRUCTURE_DIR)


def gulmc_structure_exists(run_dir):
    """Check whether pre-computed gulmc structures exist."""
    return os.path.isfile(os.path.join(_structure_path(run_dir), 'metadata.npy'))


def build_structures(run_dir, ignore_file_type, peril_filter, dynamic_footprint, model_df_engine):
    """Build all read-only gulmc data structures from input files.

    This extracts the preparation logic from ``manager.run()`` into a
    standalone callable so that it can be invoked once (by
    ``create_gulmc_structure``) rather than repeated in every parallel
    gulmc process.

    Args:
        run_dir (str): path to the run directory.
        ignore_file_type (set[str]): file extensions to ignore when loading.
        peril_filter (list): list of perils to include (empty = all).
        dynamic_footprint (bool): whether to apply dynamic footprint logic.
        model_df_engine (str): engine for loading model dataframes.

    Returns:
        dict: mapping variable names to numpy arrays / scalars.
    """
    model_storage = get_storage_from_config_path(
        os.path.join(run_dir, 'model_storage.json'),
        os.path.join(run_dir, 'static'),
    )
    input_path = os.path.join(run_dir, 'input')
    ignore_file_type = set(ignore_file_type)

    # --- keys / peril filter ---------------------------------------------------
    if os.path.exists(os.path.join(input_path, 'keys.csv')) or os.path.exists(os.path.join(input_path, 'keys.bin')):
        keys_dtype = np.dtype([('LocID', np.int32), ('PerilID', 'U3'), ('CoverageTypeID', np.int32),
                               ('AreaPerilID', areaperil_int), ('VulnerabilityID', np.int32)])
        keys_tb = load_as_ndarray(input_path, 'keys', keys_dtype)
        if peril_filter:
            peril_set = set(peril_filter)
            mask = np.array([p in peril_set for p in keys_tb['PerilID']])
            valid_areaperil_id = np.unique(keys_tb['AreaPerilID'][mask])
            logger.debug(
                f'Peril specific run: ({peril_filter}), {len(valid_areaperil_id)} AreaPerilID included out of {len(keys_tb)}')
        else:
            valid_areaperil_id = np.unique(keys_tb['AreaPerilID'])
    else:
        valid_areaperil_id = None

    # --- damage bins -----------------------------------------------------------
    logger.debug('import damage bins')
    damage_bins = get_damage_bins(model_storage, ignore_file_type)

    # --- coverages -------------------------------------------------------------
    logger.debug('import coverages')
    coverages_tb = read_coverages(input_path, ignore_file_type)
    coverages = np.zeros(coverages_tb.shape[0] + 1, coverage_type)
    coverages[1:]['tiv'] = coverages_tb

    # --- aggregate vulnerability -----------------------------------------------
    logger.debug('import aggregate vulnerability definitions and vulnerability weights')
    aggregate_vulnerability = read_aggregate_vulnerability(model_storage, ignore_file_type)
    aggregate_weights = read_vulnerability_weights(model_storage, ignore_file_type)
    agg_vuln_ids, agg_vuln_id_ja_id_ind, agg_vuln_id_ja_offsets, agg_vuln_id_ja_vuln_ids = \
        process_aggregate_vulnerability(aggregate_vulnerability)

    if aggregate_vulnerability is not None and aggregate_weights is None:
        raise FileNotFoundError(
            f"Vulnerability weights file not found at {model_storage.get_storage_url('', print_safe=True)[1]}"
        )

    # --- items + correlations --------------------------------------------------
    logger.debug('import items and correlations tables')
    correlations_tb = read_correlations(input_path, ignore_file_type)
    items_tb = read_items(input_path, ignore_file_type)
    if len(correlations_tb) != len(items_tb):
        logger.info(
            f"The items table has length {len(items_tb)} while the correlations table has length {len(correlations_tb)}.\n"
            "It is possible that the correlations are not set up properly in the model settings file."
        )

    items = rfn.join_by(
        'item_id', items_tb, correlations_tb,
        jointype='leftouter', usemask=False,
        defaults={'peril_correlation_group': 0,
                  'damage_correlation_value': 0.,
                  'hazard_group_id': 0,
                  'hazard_correlation_value': 0.,
                  'source_coverage_id': 0}
    )
    if valid_areaperil_id is not None:
        items = items[np.isin(items['areaperil_id'], valid_areaperil_id)]
    items = rfn.merge_arrays((items,
                              np.empty(items.shape,
                                       dtype=nb.from_dtype(np.dtype([("vulnerability_idx", oasis_int),
                                                                     ("areaperil_agg_vuln_idx", oasis_int)])))),
                             flatten=True)
    items['areaperil_agg_vuln_idx'] = -1

    if dynamic_footprint:
        logger.debug('get dynamic footprint adjustments')
        adjustments_tb = get_dynamic_footprint_adjustments(input_path)
        items = rfn.join_by(
            'item_id', items, adjustments_tb,
            jointype='leftouter', usemask=False,
            defaults={'intensity_adjustment': 0, 'return_period': 0}
        )

    if dynamic_footprint:
        logger.debug('get peril_id')
        item_peril = get_peril_id(input_path)
        items = rfn.join_by(
            'item_id', items, item_peril,
            jointype='leftouter', usemask=False,
            defaults={'peril_id': 0}
        )

    # sequential indices for group_id / hazard_group_id
    unique_group_ids_arr, group_seq_ids = np.unique(items['group_id'], return_inverse=True)
    unique_haz_group_ids_arr, haz_group_seq_ids = np.unique(items['hazard_group_id'], return_inverse=True)
    n_unique_groups = len(unique_group_ids_arr)
    n_unique_haz_groups = len(unique_haz_group_ids_arr)
    items = rfn.merge_arrays((items,
                              np.empty(items.shape,
                                       dtype=nb.from_dtype(np.dtype([("group_seq_id", np.int32),
                                                                     ("hazard_group_seq_id", np.int32)])))),
                             flatten=True)
    items['group_seq_id'] = group_seq_ids
    items['hazard_group_seq_id'] = haz_group_seq_ids

    items.sort(order=['areaperil_id', 'vulnerability_id'])

    # --- item map (two-level jagged array) -------------------------------------
    (item_map_ja_areaperil_ids, item_map_ja_offsets,
     item_map_ja_vuln_ja_offsets,
     item_map_ja_vuln_ja_item_idxs,
     vuln_map, vuln_map_keys,
     areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data,
     areaperil_agg_vuln_idx_ja_areaperil_ids) = generate_item_map(
        items,
        coverages,
        agg_vuln_id_ja_id_ind, agg_vuln_id_ja_offsets, agg_vuln_id_ja_vuln_ids)
    item_map_ja_id_ind = id_index_build(item_map_ja_areaperil_ids)

    # CDF group indices
    item_cdf_group_idx, n_cdf_groups = build_cdf_group_indices(
        item_map_ja_vuln_ja_offsets, item_map_ja_vuln_ja_item_idxs,
        items, dynamic_footprint if dynamic_footprint else None)

    if aggregate_weights is not None:
        logger.debug('reconstruct aggregate vulnerability definitions and weights')
        process_vulnerability_weights(areaperil_agg_vuln_idx_ja_areaperil_ids, areaperil_agg_vuln_idx_ja_data,
                                      vuln_map, vuln_map_keys, aggregate_weights)
    del areaperil_agg_vuln_idx_ja_areaperil_ids  # only needed during setup

    # --- peril correlation groups ----------------------------------------------
    unique_peril_correlation_groups = np.unique(items['peril_correlation_group'])

    # --- coverage dependency forest --------------------------------------------
    # NB the dependent-vulnerability guard (each dependent vuln must have one intensity bin per
    # damage bin) is applied per vulnerability in the gulmc manager, where the vuln array is loaded.
    coverage_source_id, coverage_dependents_ja_offsets, coverage_dependents_ja_data = \
        build_coverage_dependency_forest(items, coverages.shape[0])

    # --- footprint (temporary open to get num_intensity_bins) ------------------
    # FootprintParquetDynamic.__enter__ reads input/sections.csv and input/keys.csv
    # via relative paths, so cwd must be the run directory.
    logger.debug('import footprint')
    with setcwd(run_dir), Footprint.load(model_storage, ignore_file_type,
                                         df_engine=model_df_engine,
                                         areaperil_ids=item_map_ja_areaperil_ids) as footprint_obj:
        num_intensity_bins = footprint_obj.num_intensity_bins

    # --- vulnerabilities -------------------------------------------------------
    logger.debug('import vulnerabilities')
    vuln_adj = get_vuln_rngadj(run_dir, vuln_map, vuln_map_keys)
    vuln_array, _, _ = get_vulns(model_storage, run_dir, vuln_map, vuln_map_keys,
                                 num_intensity_bins, ignore_file_type, df_engine=model_df_engine)

    # --- Gaussian lookup tables (deterministic constants) ----------------------
    norm_inv_parameters = np.array(
        (x_min, x_max, norm_inv_N, cdf_min, cdf_max, inv_factor, norm_factor),
        dtype=NormInversionParameters)
    norm_inv_cdf = compute_norm_inv_cdf_lookup(
        norm_inv_parameters['x_min'], norm_inv_parameters['x_max'], norm_inv_parameters['N'])
    norm_cdf = compute_norm_cdf_lookup(
        norm_inv_parameters['cdf_min'], norm_inv_parameters['cdf_max'], norm_inv_parameters['N'])

    # --- dynamic footprint intensity bins --------------------------------------
    if dynamic_footprint:
        intensity_bin_peril_ids, intensity_bins = get_intensity_bin_dict(os.path.join(run_dir, 'static'))
    else:
        intensity_bin_peril_ids = np.empty(0, dtype=np.int32)
        intensity_bins = np.empty((0, 0), dtype=np.int32)

    # --- pack everything into a dict -------------------------------------------
    # Only include arrays used at runtime (event loop). Build-time intermediaries
    # (vuln_map, vuln_map_keys, agg_vuln_id_ja_*, num_intensity_bins) are excluded.
    return {
        'items': items,
        'coverages': coverages,
        'item_map_ja_areaperil_ids': item_map_ja_areaperil_ids,
        'item_map_ja_offsets': item_map_ja_offsets,
        'item_map_ja_vuln_ja_offsets': item_map_ja_vuln_ja_offsets,
        'item_map_ja_vuln_ja_item_idxs': item_map_ja_vuln_ja_item_idxs,
        'item_map_ja_id_ind': item_map_ja_id_ind,
        'item_cdf_group_idx': item_cdf_group_idx,
        'areaperil_agg_vuln_idx_ja_offsets': areaperil_agg_vuln_idx_ja_offsets,
        'areaperil_agg_vuln_idx_ja_data': areaperil_agg_vuln_idx_ja_data,
        'damage_bins': damage_bins,
        'vuln_adj': vuln_adj,
        'vuln_array': vuln_array,
        'unique_peril_correlation_groups': unique_peril_correlation_groups,
        'norm_inv_cdf': norm_inv_cdf,
        'norm_cdf': norm_cdf,
        'norm_inv_parameters': norm_inv_parameters,
        'intensity_bin_peril_ids': intensity_bin_peril_ids,
        'intensity_bins': intensity_bins,
        'coverage_source_id': coverage_source_id,
        'coverage_dependents_ja_offsets': coverage_dependents_ja_offsets,
        'coverage_dependents_ja_data': coverage_dependents_ja_data,
        # scalars
        'n_cdf_groups': n_cdf_groups,
        'n_unique_groups': n_unique_groups,
        'n_unique_haz_groups': n_unique_haz_groups,
    }


def create_gulmc_structure(run_dir, ignore_file_type, peril_filter,
                           dynamic_footprint, model_df_engine):
    """Build and save all read-only gulmc data structures as ``.npy`` files.

    Args:
        run_dir (str): path to the run directory.
        ignore_file_type (set[str]): file extensions to ignore when loading.
        peril_filter (list): list of perils to include (empty = all).
        dynamic_footprint (bool): whether to apply dynamic footprint logic.
        model_df_engine (str): engine for loading model dataframes.
    """
    structures = build_structures(run_dir, ignore_file_type, peril_filter,
                                  dynamic_footprint, model_df_engine)

    structure_path = _structure_path(run_dir)
    os.makedirs(structure_path, exist_ok=True)

    # save all numpy arrays
    for name in ARRAY_FILES:
        np.save(os.path.join(structure_path, name), structures[name])

    # save scalar metadata
    metadata = np.array([
        structures['n_cdf_groups'],
        structures['n_unique_groups'],
        structures['n_unique_haz_groups'],
    ], dtype=np.int64)
    np.save(os.path.join(structure_path, 'metadata'), metadata)

    total_bytes = sum(
        os.path.getsize(os.path.join(structure_path, f'{name}.npy'))
        for name in ARRAY_FILES
    )
    logger.info(f"gulmc structures saved to {structure_path} ({total_bytes / 1024 / 1024:.1f} MB)")


def load_gulmc_structure(run_dir):
    """Load pre-computed gulmc structures via memory-mapped numpy files.

    Each array is loaded with ``mmap_mode='r'`` so that multiple gulmc
    processes share physical memory pages through the OS page cache.

    Args:
        run_dir (str): path to the run directory.

    Returns:
        dict: mapping variable names to numpy arrays / scalars.
    """
    structure_path = _structure_path(run_dir)
    result = {}

    for name in ARRAY_FILES:
        result[name] = np.load(os.path.join(structure_path, f'{name}.npy'), mmap_mode='r')

    metadata = np.load(os.path.join(structure_path, 'metadata.npy'))
    result['n_cdf_groups'] = int(metadata[0])
    result['n_unique_groups'] = int(metadata[1])
    result['n_unique_haz_groups'] = int(metadata[2])

    return result
