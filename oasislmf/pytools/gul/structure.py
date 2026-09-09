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
from oasislmf.pytools.common.data import correlations_dtype, load_as_ndarray
from oasislmf.pytools.common.input_files import KEYS_DTYPE, filter_area_peril_id, read_coverages, read_correlations
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
]


def _structure_path(run_dir):
    return os.path.join(run_dir, 'input', STRUCTURE_DIR)


# every scalar load_gulpy_structure reads out of metadata.npy, in order
METADATA_FIELDS = ['do_correlation', 'building_packing']


def gulpy_structure_exists(run_dir):
    """Check whether pre-computed gulpy structures exist AND match what this version reads.

    A cache written by an earlier version is missing whatever arrays and metadata have been added
    since, and reporting it as present makes the load fail rather than fall back. Treating an
    incomplete cache as absent rebuilds it instead, which is always safe.

    Args:
        run_dir (str): path to the run directory.

    Returns:
        bool: True when a complete cache is present.
    """
    structure_path = _structure_path(run_dir)
    metadata_path = os.path.join(structure_path, 'metadata.npy')
    if not os.path.isfile(metadata_path):
        return False
    if not all(os.path.isfile(os.path.join(structure_path, f'{name}.npy')) for name in ARRAY_FILES):
        logger.info('pre-computed gulpy structures are incomplete: rebuilding')
        return False
    try:
        if np.load(metadata_path).shape[0] < len(METADATA_FIELDS):
            logger.info('pre-computed gulpy structures predate the current metadata: rebuilding')
            return False
    except Exception:
        # a truncated or half-written metadata.npy raises EOFError, a 0-d one IndexError. The
        # point of this function is to fall back to a rebuild rather than fail the run, so
        # anything unreadable counts as absent.
        logger.info('pre-computed gulpy structures are unreadable: rebuilding')
        return False
    return True


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

    if do_correlation:
        corr_data_by_item_id = np.ndarray(Nperil_correlation_groups + 1, dtype=correlations_dtype)
        # sentinel row 0 (item_id, peril_correlation_group, damage_correlation_value,
        # hazard_group_id, hazard_correlation_value, number_of_buildings)
        corr_data_by_item_id[0] = (0, 0, 0., 0, 0., 1)
        corr_data_by_item_id[1:]['peril_correlation_group'] = data['peril_correlation_group']
        corr_data_by_item_id[1:]['damage_correlation_value'] = data['damage_correlation_value']
        unique_peril_correlation_groups = np.unique(
            corr_data_by_item_id[1:]['peril_correlation_group'])

        # pre-compute Gaussian lookup tables
        norm_inv_cdf = compute_norm_inv_cdf_lookup(x_min, x_max, norm_inv_N)
        norm_cdf = compute_norm_cdf_lookup(cdf_min, cdf_max, norm_inv_N)
    else:
        corr_data_by_item_id = np.ndarray(1, dtype=correlations_dtype)
        unique_peril_correlation_groups = np.empty(0, dtype='int64')
        norm_inv_cdf = np.zeros(1, dtype='float64')
        norm_cdf = np.zeros(1, dtype='float64')

    # --- building packing ------------------------------------------------------
    # The per-item building count, and whether those buildings must reach the financial module as
    # separate blocks, ride on the correlations table (1:1 with items by item_id) as ONE signed
    # field: magnitude is the count, a negative sign marks "keep separate". It is kept signed all
    # the way into the compute, and unpacked into (count, flag) locals at the top of each loop
    # that consumes it -- see the readers in gul/io.py, gul/manager.py and gulmc/manager.py.
    # NOTHING may use the raw value as a loop bound: range() over a negative silently does nothing.
    # Packing is derived, not configured: an item carrying more than one building is the signal.
    packed_buildings = np.abs(data['number_of_buildings']) if len(data) else data['number_of_buildings']
    if len(data) and packed_buildings.max() > 1:
        building_packing = True
        max_item_id = int(data['item_id'].max())
        # The lookups below are indexed by item_id straight from items.bin inside njit code, which
        # does not bounds-check, so an items table reaching past the correlations table would be a
        # silent out-of-range read rather than an error. The two are written together and 1:1, so
        # this only fires on a mismatched input set -- check it once here instead of per item.
        if len(items) and int(items['item_id'].max()) > max_item_id:
            raise OasisException(
                f"items.bin holds item_id up to {int(items['item_id'].max())} but correlations "
                f"only covers up to {max_item_id}; the two files are 1:1 and must be regenerated "
                f"together."
            )
        # stored signed, exactly as it arrived on the wire
        n_buildings_by_item_id = np.ones(max_item_id + 1, dtype='i4')
        n_buildings_by_item_id[data['item_id']] = data['number_of_buildings']
        logger.info(f'building-packing ENABLED: up to {packed_buildings.max()} buildings packed per item.')
    else:
        building_packing = False
        n_buildings_by_item_id = np.ones(1, dtype='i4')

    # --- pack everything into a dict -------------------------------------------
    return {
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
