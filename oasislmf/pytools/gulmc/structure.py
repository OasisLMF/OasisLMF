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
from oasislmf.pytools.common.data import (conditionalvulnerability_dtype, conditionalvulnerability_headers,
                                          load_as_ndarray, oasis_int, oasis_float)
from oasislmf.utils.exceptions import OasisException
from oasislmf.pytools.common.id_index import build as id_index_build
from oasislmf.pytools.common.input_files import KEYS_DTYPE, filter_area_peril_id, read_coverages, read_correlations
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
    'conditional_vuln_array',
    'vuln_idx_to_cond_idx',
    'unique_peril_correlation_groups',
    'norm_inv_cdf',
    'norm_cdf',
    'norm_inv_parameters',
    'intensity_bin_peril_ids',
    'intensity_bins',
    'coverage_source_id',
    'coverage_dependents_ja_offsets',
    'coverage_dependents_ja_data',
    'source_item_idx',
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
    """Build the coverage dependency forest from the per-item ``source_item_id`` links.

    Produces ``coverage_source_id`` (indexed by coverage_id, 0 = independent/root) and the
    parent -> dependents jagged array (``coverage_dependents_ja_offsets`` /
    ``coverage_dependents_ja_data``) used by the gulmc DFS push, plus ``source_item_idx``: the
    index into ``items`` of each item's source item, or -1 when it has none. The input
    preparation guarantees every item of a dependent coverage links to an item of the same
    source coverage, so the coverage-level forest is a scatter of the resolved coverages.

    Args:
        items (np.ndarray): items table containing 'coverage_id' and 'source_coverage_id'.
        n_coverages (int): number of coverage slots (coverages.shape[0] == max coverage_id + 1).

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        coverage_source_id (len n_coverages),
        coverage_dependents_ja_offsets (len n_coverages + 1),
        coverage_dependents_ja_data (len = number of dependent coverages),
        source_item_idx (len items, -1 where the item has no source).

    Raises:
        OasisException: if a source_item_id does not exist, or resolves to a coverage that is out
          of range, self-referencing or cyclic. These are raised rather than asserted so the
          checks survive ``python -O``, where a bad id would otherwise reach the njit
          ``compute_max_dependency_depth`` and index out of bounds with no boundscheck.
    """
    # resolve each source_item_id to its position in the items table
    source_item_idx = np.full(items.shape[0], -1, dtype=np.int64)
    linked = np.nonzero(items['source_item_id'] > 0)[0]
    if linked.size > 0:
        order = np.argsort(items['item_id'], kind='stable')
        sorted_ids = items['item_id'][order]
        pos = np.searchsorted(sorted_ids, items['source_item_id'][linked])
        missing = (pos >= sorted_ids.shape[0]) | (sorted_ids[np.minimum(pos, sorted_ids.shape[0] - 1)]
                                                  != items['source_item_id'][linked])
        if missing.any():
            raise OasisException(
                f"coverage dependency: source_item_id(s) "
                f"{np.unique(items['source_item_id'][linked][missing]).tolist()} do not exist in the "
                "items table; malformed correlations input."
            )
        source_item_idx[linked] = order[pos]

        # A dependent item and its source item must sit at the same areaperil. Everything
        # downstream leans on it: the per-event item position map and the depth-indexed source
        # stacks in the gulmc kernel are reused across events without clearing, which is only
        # sound because the two items are present or absent together; and a coverage whose source
        # coverage is absent from an event is computed as a root on the same grounds. File
        # generation only ever links items that share an areaperil, so this catches malformed or
        # stale correlations input rather than an expected shape — but silently it would give
        # event-order-dependent losses, so it is checked like every other link malformation here.
        cross_cell = np.nonzero(items['areaperil_id'][linked]
                                != items['areaperil_id'][source_item_idx[linked]])[0]
        if cross_cell.size > 0:
            bad_items = items['item_id'][linked][cross_cell]
            raise OasisException(
                f"coverage dependency: item(s) {bad_items[:10].tolist()} are linked to a source item "
                "at a different areaperil; a dependent item must share its source item's areaperil, "
                "because its damage is driven by damage in that cell. Malformed correlations input."
            )

    # coverage ids are unsigned; reuse the input dtype so the forest matches the items table
    id_dtype = items['coverage_id'].dtype
    coverage_source_id = np.zeros(n_coverages, dtype=id_dtype)
    dependent_coverages = items['coverage_id'][linked]
    source_coverages = items['coverage_id'][source_item_idx[linked]]
    if linked.size > 0:
        # The forest is coverage-level, so all the linked items of one coverage must resolve to the
        # same source coverage. File generation guarantees it (a coverage's items share loc_id,
        # building_id and coverage_type_id, so their sources share a coverage_id), so this catches
        # malformed input; the scatter below would otherwise silently keep whichever write landed
        # last and drive some items from the wrong coverage's depth row.
        pairs = np.unique(np.stack([dependent_coverages, source_coverages], axis=1), axis=0)
        dependent_ids, n_sources = np.unique(pairs[:, 0], return_counts=True)
        ambiguous = dependent_ids[n_sources > 1]
        if ambiguous.size > 0:
            raise OasisException(
                f"coverage dependency: coverage_id(s) {ambiguous[:10].tolist()} have items linked to "
                "items of more than one source coverage; a dependent coverage must have a single "
                "source. Malformed correlations input."
            )
    coverage_source_id[dependent_coverages] = source_coverages

    # A source pointing outside the coverage range, or a coverage referencing itself, can only
    # come from malformed/stale input (a valid source is always an in-range coverage_id of a
    # different coverage at the same location). Fail loudly rather than silently demoting the
    # dependent to independent, which would change losses with no signal. (source == 0 means
    # independent and is excluded from both checks.)
    out_of_range = np.nonzero(coverage_source_id >= n_coverages)[0]
    if out_of_range.size > 0:
        raise OasisException(
            f"coverage dependency: source_coverage_id out of range for coverage_id(s) "
            f"{out_of_range.tolist()} (n_coverages={n_coverages}); malformed correlations input."
        )
    self_ref = np.nonzero((coverage_source_id == np.arange(n_coverages)) & (coverage_source_id != 0))[0]
    if self_ref.size > 0:
        raise OasisException(
            f"coverage dependency: coverage_id(s) {self_ref.tolist()} reference themselves as their "
            "own source; a coverage cannot depend on itself."
        )

    _validate_acyclic_coverage_dependency(coverage_source_id)

    # invert to a parent -> dependents jagged array: dependents grouped by ascending parent
    coverage_dependents_ja_data = np.nonzero(coverage_source_id > 0)[0].astype(id_dtype)
    parents = coverage_source_id[coverage_dependents_ja_data]
    order = np.argsort(parents, kind='stable')
    coverage_dependents_ja_data = coverage_dependents_ja_data[order]
    coverage_dependents_ja_offsets = np.zeros(n_coverages + 1, dtype=oasis_int)
    coverage_dependents_ja_offsets[1:] = np.cumsum(np.bincount(parents, minlength=n_coverages))

    return coverage_source_id, coverage_dependents_ja_offsets, coverage_dependents_ja_data, source_item_idx


def get_conditional_vulns(storage, damage_bins, ignore_file_type=set()):
    """Load the conditional (dependent) vulnerability transition matrices.

    A conditional vulnerability is a damage-transition matrix ``P(dependent damage bin | source
    damage bin)``: it drives a dependent coverage from its source coverage's sampled damage bin
    instead of from the footprint hazard. It is a distinct model input from ``vulnerability`` and
    is correctly sized ``num_damage_bins x num_damage_bins`` (independent of the footprint's
    intensity resolution). Its columns are ``vulnerability_id, source_damage_bin, damage_bin,
    probability``, and its binary layout is interchangeable with a flat vulnerability file (a
    4-byte int32 header, then one record per row), so the ``csvtobin`` / ``bintocsv``
    tools handle it as the ``conditionalvulnerability`` file type. The file is optional; when absent, no coverage may be
    a (conditional) dependent.

    Completeness is not required: a source damage bin the source can never reach may be left with
    no rows. Such a column is filled here with a point mass on the first damage bin — the
    damage_bin_dict's no-damage bin — so it samples as "this source damage => no dependent damage".
    That requires the first damage bin to be a zero-damage point bin, which is checked; leaving it
    as an all-zero column instead would make the sampled loss undefined (an all-zero column
    collapses to a single-element, zero-height cdf, and interpolating within a bin of zero
    probability height divides by zero).

    Args:
        storage (BaseStorage): storage connector for the model static data.
        damage_bins (np.ndarray): the damage_bin_dict (bin_from / bin_to / interpolation).
        ignore_file_type (set[str]): file extensions to ignore.

    Returns:
        tuple(np.ndarray, np.ndarray):
        conditional_vuln_array of shape ``(n_cond, num_damage_bins, num_damage_bins)`` indexed
        ``[cond_idx, dependent_damage_bin - 1, source_damage_bin - 1]``, and cond_vuln_ids
        (the vulnerability ids, ascending, aligned with cond_idx). Both empty when no file exists.

    Raises:
        OasisException: if a bin id is out of range (1..num_damage_bins), or if a source damage bin
          is left undefined while the first damage bin is not a zero-damage point bin.
    """
    num_damage_bins = damage_bins.shape[0]
    input_files = set(storage.listdir())
    recs = None
    if "conditional_vulnerability.bin" in input_files and 'bin' not in ignore_file_type:
        # flat (non-indexed, uncompressed) vulnerability layout: a fixed 4-byte int32 header
        # (num_damage_bins, matching vulnerability.bin's max_damage_bin header — NOT oasis_int-
        # sized) followed by vulnerability_dtype records. We size from the damage_bin_dict, so the
        # header value is skipped. An .idx/compressed conditional file is not supported.
        with storage.open("conditional_vulnerability.bin", 'rb') as f:
            f.read(4)
            recs = np.frombuffer(f.read(), dtype=conditionalvulnerability_dtype)
    elif "conditional_vulnerability.csv" in input_files and 'csv' not in ignore_file_type:
        with storage.open("conditional_vulnerability.csv") as f:
            lines = [line.decode() if isinstance(line, bytes) else line for line in f.readlines()]
        # detect the header rather than assuming one, as read_correlations / read_coverages do:
        # `bintocsv conditionalvulnerability --noheader` writes a headerless file, and skipping a
        # data row there would silently drop a whole (source bin -> damage bin) transition
        has_header = [h.strip() for h in lines[0].strip().split(',')] == conditionalvulnerability_headers
        recs = np.loadtxt(lines[1:] if has_header else lines,
                          dtype=conditionalvulnerability_dtype, delimiter=',', ndmin=1)

    if recs is None or recs.shape[0] == 0:
        return (np.zeros((0, num_damage_bins, num_damage_bins), dtype=oasis_float),
                np.zeros(0, dtype=np.int32))

    source_bin = recs['source_damage_bin']
    damage_bin = recs['damage_bin']
    if source_bin.min() < 1 or source_bin.max() > num_damage_bins \
            or damage_bin.min() < 1 or damage_bin.max() > num_damage_bins:
        raise OasisException(
            f"conditional_vulnerability bins must be in 1..{num_damage_bins}: source damage bin "
            f"range [{int(source_bin.min())}, {int(source_bin.max())}], damage bin "
            f"range [{int(damage_bin.min())}, {int(damage_bin.max())}]."
        )

    cond_vuln_ids = np.unique(recs['vulnerability_id'])
    id_to_idx = {int(v): i for i, v in enumerate(cond_vuln_ids)}
    conditional_vuln_array = np.zeros((cond_vuln_ids.shape[0], num_damage_bins, num_damage_bins), dtype=oasis_float)
    for r in recs:
        conditional_vuln_array[id_to_idx[int(r['vulnerability_id'])],
                               int(r['damage_bin']) - 1, int(r['source_damage_bin']) - 1] = r['probability']

    # An undefined source damage bin (all-zero column) means "no dependent damage": make that
    # explicit as a point mass on the first damage bin, which must therefore be a zero-damage
    # point bin. Left as zeros the column has no sampleable distribution at all.
    column_total = conditional_vuln_array.sum(axis=1)
    undefined = column_total == 0
    if undefined.any():
        if not (damage_bins[0]['bin_from'] == 0. and damage_bins[0]['bin_to'] == 0.):
            cond_i, source_bin_i = np.nonzero(undefined)
            raise OasisException(
                f"conditional_vulnerability leaves source damage bin(s) {(source_bin_i + 1).tolist()} "
                f"(conditional vulnerability index/indices {cond_i.tolist()}) undefined, which means "
                "'no dependent damage'. That requires damage bin 1 of the damage_bin_dict to be a "
                f"zero-damage bin, but it is [{damage_bins[0]['bin_from']}, {damage_bins[0]['bin_to']}]. "
                "Define the missing source damage bin columns explicitly."
            )
        conditional_vuln_array[:, 0, :][undefined] = 1.

    # A column that is defined but does not sum to 1 is a partially specified distribution, which
    # samples past the top of its last defined damage bin. That is checked where the other
    # vulnerability integrity checks live — the csv -> bin converter
    # (oasislmf convert csvtobin conditionalvulnerability) — not here, matching how
    # vulnerability.bin is treated.

    return conditional_vuln_array, cond_vuln_ids.astype(np.int32)


def align_conditional_damage_axis(conditional_vuln_array, Ndamage_bins_max):
    """Align a conditional vulnerability matrix's dependent-damage axis to the vuln array's.

    ``get_conditional_vulns`` sizes both axes from the damage_bin_dict, but the kernel copies a
    conditional column straight into a ``vuln_pdf`` row (``resolve_item_cdfs``), whose width is
    ``Ndamage_bins_max == vuln_array.shape[1]``. That comes from the vulnerability file (parquet
    metadata, the ``vulnerability.bin`` header, or ``max(damage_bin_id)`` for csv) and is not
    guaranteed to equal the damage_bin_dict size — a vulnerability file whose top damage bin is
    unused makes it smaller. Drop the unreachable tail of the dependent-damage axis so the copy is
    always a matching-shape assignment, and refuse the run outright in the opposite case, where a
    source's sampled damage bin could fall outside the conditional matrix's source axis.

    Args:
        conditional_vuln_array (np.ndarray): shape ``(n_cond, num_damage_bins, num_source_bins)``
          as returned by ``get_conditional_vulns``.
        Ndamage_bins_max (int): the vulnerability array's damage bin count (``vuln_array.shape[1]``).

    Returns:
        np.ndarray: shape ``(n_cond, Ndamage_bins_max, num_source_bins)``.

    Raises:
        OasisException: if a dependent damage bin beyond ``Ndamage_bins_max`` carries probability
          (the conditional file reaches a damage bin the vulnerability array cannot represent), or
          if ``Ndamage_bins_max`` exceeds the damage_bin_dict size.
    """
    num_damage_bins = conditional_vuln_array.shape[1]
    if conditional_vuln_array.shape[0] == 0 or num_damage_bins == Ndamage_bins_max:
        return conditional_vuln_array

    if num_damage_bins > Ndamage_bins_max:
        dropped = np.nonzero(conditional_vuln_array[:, Ndamage_bins_max:, :].any(axis=(1, 2)))[0]
        if dropped.size > 0:
            raise OasisException(
                f"conditional_vulnerability assigns probability to damage bin(s) above "
                f"{Ndamage_bins_max}, the number of damage bins in the vulnerability data "
                f"(conditional vulnerability index/indices {dropped.tolist()}); the dependent "
                "damage bins must be representable in the same damage bin space as vulnerability."
            )
        return conditional_vuln_array[:, :Ndamage_bins_max, :].copy()

    # Ndamage_bins_max > num_damage_bins: the vulnerability data declares more damage bins than the
    # damage_bin_dict, so a source coverage can sample a damage bin with no column in the
    # conditional matrix (and no entry in the damage_bin_dict). Padding would silently drive the
    # dependent from an out-of-range bin, so refuse the run.
    raise OasisException(
        f"coverage dependency: the vulnerability data declares {Ndamage_bins_max} damage bins but "
        f"the damage_bin_dict has {num_damage_bins}; a source coverage's sampled damage bin would "
        "fall outside the conditional_vulnerability source axis. The two must agree."
    )


def build_vuln_idx_to_cond_idx(items, cond_vuln_ids, n_vulns):
    """Map each vulnerability's dense index to its conditional (damage-transition) matrix row.

    ``conditional_vuln_array`` is compact (one row per conditional vulnerability), while the kernel
    reaches it through an item's ``vulnerability_idx`` — the dense index assigned by
    ``generate_item_map``. This inverts the two: dense vuln index -> conditional row, or -1 for a
    normal hazard-indexed vulnerability.

    Only non-aggregate items carry a ``vulnerability_idx`` (``generate_item_map`` assigns it in the
    non-aggregate branch only, and sets ``areaperil_agg_vuln_idx`` instead for aggregates), so
    aggregate items are excluded: an aggregate vulnerability id that happens to collide with a
    conditional one must not scatter through an index that was never assigned. A dependent coverage
    may not use an aggregate vulnerability anyway — ``validate_coverage_dependency`` rejects that.

    Args:
        items (np.ndarray): items table with 'vulnerability_id', 'vulnerability_idx' and
            'areaperil_agg_vuln_idx'.
        cond_vuln_ids (np.ndarray): conditional vulnerability ids, ascending (from ``np.unique``).
        n_vulns (int): number of dense vulnerability indices (``vuln_array.shape[0]``).

    Returns:
        np.ndarray[int64]: length ``n_vulns``, conditional row per dense vuln index, else -1. The
          signed dtype keeps the -1 sentinel usable under an unsigned ``oasis_int`` override.
    """
    vuln_idx_to_cond_idx = np.full(n_vulns, -1, dtype=np.int64)
    if cond_vuln_ids.shape[0] == 0:
        return vuln_idx_to_cond_idx

    present = np.isin(items['vulnerability_id'], cond_vuln_ids) & (items['areaperil_agg_vuln_idx'] < 0)
    # cond_vuln_ids is ascending, so searchsorted gives the conditional row
    vuln_idx_to_cond_idx[items['vulnerability_idx'][present]] = \
        np.searchsorted(cond_vuln_ids, items['vulnerability_id'][present]).astype(np.int64)
    return vuln_idx_to_cond_idx


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
        keys_tb = load_as_ndarray(input_path, 'keys', KEYS_DTYPE)
        if peril_filter:
            valid_areaperil_id = filter_area_peril_id(keys_tb, peril_filter)
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
                  'source_item_id': 0}
    )
    if valid_areaperil_id is not None:
        items = items[np.isin(items['areaperil_id'], valid_areaperil_id)]
    items = rfn.merge_arrays((items,
                              np.empty(items.shape,
                                       dtype=nb.from_dtype(np.dtype([("vulnerability_idx", oasis_int),
                                                                     ("areaperil_agg_vuln_idx", oasis_int)])))),
                             flatten=True)
    items['areaperil_agg_vuln_idx'] = -1
    # generate_item_map only assigns vulnerability_idx for non-aggregate items; initialise it so an
    # aggregate item never carries uninitialised memory into an array index
    items['vulnerability_idx'] = 0

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
    coverage_source_id, coverage_dependents_ja_offsets, coverage_dependents_ja_data, source_item_idx = \
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
    # --- conditional (dependent) vulnerabilities -------------------------------
    # A dependent coverage is driven by its source's damage bin via a separate damage-transition
    # matrix P(dependent damage bin | source damage bin), correctly sized num_damage_bins^2 (not
    # the footprint intensity resolution). Loaded before the hazard-indexed vulnerabilities so its
    # ids can be excluded from get_vulns' presence check (they are absent from vulnerability.bin).
    conditional_vuln_array, cond_vuln_ids = get_conditional_vulns(
        model_storage, damage_bins, ignore_file_type)

    vuln_array, _, _ = get_vulns(model_storage, run_dir, vuln_map, vuln_map_keys,
                                 num_intensity_bins, ignore_file_type, df_engine=model_df_engine,
                                 allow_missing_vuln_ids=cond_vuln_ids)

    # the kernel copies a conditional column into a vuln_pdf row of width vuln_array.shape[1]
    # (Ndamage_bins_max), which the vulnerability file — not the damage_bin_dict — determines.
    conditional_vuln_array = align_conditional_damage_axis(conditional_vuln_array, vuln_array.shape[1])

    vuln_idx_to_cond_idx = build_vuln_idx_to_cond_idx(items, cond_vuln_ids, vuln_array.shape[0])

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
        'conditional_vuln_array': conditional_vuln_array,
        'vuln_idx_to_cond_idx': vuln_idx_to_cond_idx,
        'unique_peril_correlation_groups': unique_peril_correlation_groups,
        'norm_inv_cdf': norm_inv_cdf,
        'norm_cdf': norm_cdf,
        'norm_inv_parameters': norm_inv_parameters,
        'intensity_bin_peril_ids': intensity_bin_peril_ids,
        'intensity_bins': intensity_bins,
        'coverage_source_id': coverage_source_id,
        'coverage_dependents_ja_offsets': coverage_dependents_ja_offsets,
        'coverage_dependents_ja_data': coverage_dependents_ja_data,
        'source_item_idx': source_item_idx,
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
