"""Pre-compute and persist gulpy read-only data structures.

Follows the same pattern as ``oasislmf.pytools.gulmc.structure``:
  - ``create_gulpy_structure`` builds all read-only numpy arrays once and
    saves them as ``.npy`` files.
  - ``load_gulpy_structure`` memory-maps them via ``np.load(mmap_mode='r')``,
    allowing multiple gulpy processes to share physical memory pages through
    the OS page cache.
"""
import logging
import os

import numpy as np
from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.data import correlations_dtype, load_as_ndarray, oasis_float
from oasislmf.pytools.common.input_files import KEYS_DTYPE, filter_area_peril_id, read_coverage_buildings, read_coverages, read_correlations
from oasislmf.pytools.getmodel.manager import get_damage_bins
from oasislmf.pytools.gul.common import coverage_type
from oasislmf.utils.exceptions import OasisException
from oasislmf.pytools.gul.manager import gul_get_items, generate_item_map
from oasislmf.pytools.gul.random import (
    compute_norm_cdf_lookup, compute_norm_inv_cdf_lookup,
    x_min, x_max, norm_inv_N, cdf_min, cdf_max,
)

logger = logging.getLogger(__name__)

STRUCTURE_DIR = 'gulpy_structure'

ARRAY_FILES = [
    'damage_bins',
    'coverages',
    'items',
    'item_map_hm',
    'item_map_hm_keys',
    'item_map_ja_offsets',
    'corr_data_by_item_id',
    'unique_peril_correlation_groups',
    'norm_inv_cdf',
    'norm_cdf',
    'n_buildings_by_item_id',
    'damage_correlation_by_item_id',
]


def _structure_path(run_dir):
    return os.path.join(run_dir, 'input', STRUCTURE_DIR)


# every scalar load_gulpy_structure reads out of metadata.npy, in order
METADATA_FIELDS = ['do_correlation', 'building_packing']


def gulpy_structure_exists(run_dir):
    """Check whether a usable pre-computed gulpy structure cache is present.

    The cache is built once per run by ``create_gulpy_structure`` and memory-mapped by every
    parallel gulpy process, so it is always written and read by the same version -- there is no
    version skew to defend against. What can happen is a partially written cache, if the build was
    interrupted. The caller falls back to building the structures itself, so anything unreadable
    counts as absent and is rebuilt, which is always safe.

    The metadata width is checked because it is read positionally (see ``METADATA_FIELDS``): a
    short one would be an IndexError at load rather than a fallback.

    Args:
        run_dir (str): path to the run directory.

    Returns:
        bool: True when a usable cache is present.
    """
    metadata_path = os.path.join(_structure_path(run_dir), 'metadata.npy')
    if not os.path.isfile(metadata_path):
        return False
    try:
        if np.load(metadata_path).shape[0] < len(METADATA_FIELDS):
            logger.info('pre-computed gulpy structures are incomplete: rebuilding')
            return False
    except Exception:
        # a truncated or half-written metadata.npy raises EOFError, a 0-d one IndexError
        logger.info('pre-computed gulpy structures are unreadable: rebuilding')
        return False
    return True


def coverage_building_counts(input_path, ignore_file_type, n_coverages):
    """The signed building count per coverage, as a dense array indexed by ``coverage_id - 1``.

    From coverage_buildings, a file of its own: coverages.bin is a published format that
    third-party models parse directly, so it stays a bare array of tiv. An absent file is the
    ordinary unpacked case and reads as all ones.

    Args:
        input_path (str | os.PathLike): the input directory.
        ignore_file_type (Set[str]): file extensions to ignore.
        n_coverages (int): how many coverages coverages.bin holds, which the file must match.

    Returns:
        numpy.array[int32]: the signed count per coverage, all ones when the file is absent.

    Raises:
        OasisException: if the file covers a different number of coverages than coverages.bin.
    """
    counts = np.ones(n_coverages, dtype='i4')
    buildings = read_coverage_buildings(input_path, ignore_file_type)
    if buildings.shape[0] == 0:
        return counts
    if buildings.shape[0] != n_coverages:
        raise OasisException(
            f"coverage_buildings holds {buildings.shape[0]} coverages where coverages.bin holds "
            f"{n_coverages}; the two are written together and must agree. Regenerate the oasis files."
        )
    counts[buildings['coverage_id'] - 1] = buildings['n_building']
    return counts


def build_structures(run_dir, ignore_file_type, peril_filter):
    """Build all read-only gulpy data structures from input files.

    This extracts the preparation logic from ``manager.run()`` into a
    standalone callable so that it can be invoked once (by
    ``create_gulpy_structure``) rather than repeated in every parallel
    gulpy process.

    Args:
        run_dir (str): path to the run directory.
        ignore_file_type (set[str]): file extensions to ignore when loading.
        peril_filter (list): list of perils to include (empty = all).

    Returns:
        dict: mapping variable names to numpy arrays / scalars.
    """
    model_storage = get_storage_from_config_path(
        os.path.join(run_dir, 'model_storage.json'),
        os.path.join(run_dir, 'static'),
    )
    input_path = os.path.join(run_dir, 'input')
    ignore_file_type = set(ignore_file_type)

    # --- damage bins -----------------------------------------------------------
    logger.debug('import damage bins')
    damage_bins = get_damage_bins(model_storage, ignore_file_type)

    # --- coverages -------------------------------------------------------------
    logger.debug('import coverages')
    coverages_tiv = read_coverages(input_path, ignore_file_type)
    coverages = np.zeros(coverages_tiv.shape[0] + 1, coverage_type)
    coverages[1:]['tiv'] = coverages_tiv
    n_building_by_coverage = coverage_building_counts(input_path, ignore_file_type, coverages_tiv.shape[0])

    # --- items + peril filter --------------------------------------------------
    logger.debug('import items')
    if peril_filter:
        keys_tb = load_as_ndarray(input_path, 'keys', KEYS_DTYPE)
        valid_area_peril_id = filter_area_peril_id(keys_tb, peril_filter)
        logger.debug(
            f'Peril specific run: ({peril_filter}), '
            f'{len(valid_area_peril_id)} AreaPerilID included out of {len(keys_tb)}')
    else:
        valid_area_peril_id = None

    items = gul_get_items(input_path, ignore_file_type)
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
    if valid_area_peril_id is not None:
        items = items[np.isin(items['areaperil_id'], valid_area_peril_id)]

    # --- item map (hashmap + jagged array) -------------------------------------
    logger.debug('generate item map')
    (item_map_hm, item_map_hm_keys,
     item_map_ja_offsets) = generate_item_map(items, coverages)

    # --- correlations ----------------------------------------------------------
    logger.debug('import correlations')
    data = read_correlations(input_path, ignore_file_type, filename='correlations.bin')
    Nperil_correlation_groups = len(data)

    do_correlation = False
    if Nperil_correlation_groups > 0 and any(data['damage_correlation_value'] > 0):
        do_correlation = True

    # Shared by all three by-item_id lookups below. An item absent from correlations keeps the
    # zeroed/default row, which is the "no correlation, one building" case.
    max_item_id = 0
    if len(items):
        max_item_id = int(items['item_id'].max())
    if len(data):
        max_item_id = max(max_item_id, int(data['item_id'].max()))

    if do_correlation:
        # Indexed BY item_id, so it is scattered by item_id rather than filled positionally.
        # Filling it in row order and then reading it by id silently requires item_id to be a
        # dense 1..N, which nothing guarantees: a table whose ids are sparse read the wrong row,
        # and a row past the end read out of bounds -- unchecked, inside njit. Dense ids cost
        # nothing here because max_item_id then equals len(data).
        # NOTE: field first, THEN the fancy index. `a[idx]['field'] = v` assigns into a copy and
        # is silently a no-op.
        corr_data_by_item_id = np.zeros(max_item_id + 1, dtype=correlations_dtype)
        corr_data_by_item_id['peril_correlation_group'][data['item_id']] = data['peril_correlation_group']
        corr_data_by_item_id['damage_correlation_value'][data['item_id']] = data['damage_correlation_value']
        unique_peril_correlation_groups = np.unique(data['peril_correlation_group'])

        # pre-compute Gaussian lookup tables
        norm_inv_cdf = compute_norm_inv_cdf_lookup(x_min, x_max, norm_inv_N)
        norm_cdf = compute_norm_cdf_lookup(cdf_min, cdf_max, norm_inv_N)
    else:
        corr_data_by_item_id = np.zeros(1, dtype=correlations_dtype)
        unique_peril_correlation_groups = np.empty(0, dtype='int64')
        norm_inv_cdf = np.zeros(1, dtype='float64')
        norm_cdf = np.zeros(1, dtype='float64')

    # --- building packing ------------------------------------------------------
    # The building count and the keep-separate flag are ONE signed field on the COVERAGE: buildings
    # belong to the location, so every coverage of one -- and every item of those coverages --
    # carries the same value. Kept signed into the compute and unpacked into (count, flag) at the
    # top of each consuming loop. NOTHING may use the raw value as a bound: range() over a negative
    # silently does nothing. Packing is derived, not configured: more than one building is the
    # signal.
    #
    # Indexed by item_id here because that is what the compute walks; an unpacked run is the
    # all-ones case, which is what lets it treat packing as N == 1 rather than as a second path.
    n_buildings_by_item_id = np.ones(max_item_id + 1, dtype='i4')
    if len(items):
        cov_i = items['coverage_id'] - 1
        if cov_i.min() < 0 or cov_i.max() >= n_building_by_coverage.shape[0]:
            raise OasisException(
                f"items.bin references coverage_id outside 1..{n_building_by_coverage.shape[0]} "
                f"covered by coverages.bin; the two files must be regenerated together."
            )
        n_buildings_by_item_id[items['item_id']] = n_building_by_coverage[cov_i]

    building_counts = np.abs(n_buildings_by_item_id)
    building_packing = bool(building_counts.max() > 1)
    if building_packing:
        logger.info(f'building-packing ENABLED: up to {building_counts.max()} buildings packed per item.')

    # The writer needs a summed item's damage correlation to combine its buildings' variances,
    # and only there -- indexed by item_id like the count above. Zero everywhere when correlation
    # is off, which is what was actually drawn and reduces the combination to sqrt(N).
    damage_correlation_by_item_id = np.zeros(max_item_id + 1, dtype=oasis_float)
    if do_correlation and len(data):
        damage_correlation_by_item_id[data['item_id']] = data['damage_correlation_value']

    # --- pack everything into a dict -------------------------------------------
    return {
        'damage_correlation_by_item_id': damage_correlation_by_item_id,
        'damage_bins': damage_bins,
        'coverages': coverages,
        'items': items,
        'item_map_hm': item_map_hm,
        'item_map_hm_keys': item_map_hm_keys,
        'item_map_ja_offsets': item_map_ja_offsets,
        'corr_data_by_item_id': corr_data_by_item_id,
        'unique_peril_correlation_groups': unique_peril_correlation_groups,
        'norm_inv_cdf': norm_inv_cdf,
        'norm_cdf': norm_cdf,
        'n_buildings_by_item_id': n_buildings_by_item_id,
        # scalars
        'do_correlation': int(do_correlation),
        'building_packing': int(building_packing),
    }


def create_gulpy_structure(run_dir, ignore_file_type, peril_filter):
    """Build and save all read-only gulpy data structures as ``.npy`` files.

    Args:
        run_dir (str): path to the run directory.
        ignore_file_type (set[str]): file extensions to ignore when loading.
        peril_filter (list): list of perils to include (empty = all).
    """
    structures = build_structures(run_dir, ignore_file_type, peril_filter)

    structure_path = _structure_path(run_dir)
    os.makedirs(structure_path, exist_ok=True)

    for name in ARRAY_FILES:
        np.save(os.path.join(structure_path, name), structures[name])

    # save scalar metadata
    metadata = np.array([structures[name] for name in METADATA_FIELDS], dtype=np.int64)
    np.save(os.path.join(structure_path, 'metadata'), metadata)

    total_bytes = sum(
        os.path.getsize(os.path.join(structure_path, f'{name}.npy'))
        for name in ARRAY_FILES
    )
    logger.info(f"gulpy structures saved to {structure_path} ({total_bytes / 1024 / 1024:.1f} MB)")


def load_gulpy_structure(run_dir):
    """Load pre-computed gulpy structures via memory-mapped numpy files.

    Each array is loaded with ``mmap_mode='r'`` so that multiple gulpy
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
    for i, name in enumerate(METADATA_FIELDS):
        result[name] = int(metadata[i])

    return result
