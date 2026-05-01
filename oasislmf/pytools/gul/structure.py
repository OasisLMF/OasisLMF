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
from oasislmf.pytools.common.data import areaperil_int, correlations_dtype, load_as_ndarray
from oasislmf.pytools.common.input_files import read_coverages, read_correlations
from oasislmf.pytools.getmodel.manager import get_damage_bins
from oasislmf.pytools.gul.common import coverage_type
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
]


def _structure_path(run_dir):
    return os.path.join(run_dir, 'input', STRUCTURE_DIR)


def gulpy_structure_exists(run_dir):
    """Check whether pre-computed gulpy structures exist."""
    return os.path.isfile(os.path.join(_structure_path(run_dir), 'metadata.npy'))


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
    damage_bins = get_damage_bins(model_storage)

    # --- coverages -------------------------------------------------------------
    logger.debug('import coverages')
    coverages_tiv = read_coverages(input_path)
    coverages = np.zeros(coverages_tiv.shape[0] + 1, coverage_type)
    coverages[1:]['tiv'] = coverages_tiv

    # --- items + peril filter --------------------------------------------------
    logger.debug('import items')
    if peril_filter:
        keys_dtype = np.dtype([('LocID', np.int32), ('PerilID', 'U3'), ('CoverageTypeID', np.int32),
                               ('AreaPerilID', areaperil_int), ('VulnerabilityID', np.int32)])
        keys_tb = load_as_ndarray(input_path, 'keys', keys_dtype)
        peril_set = set(peril_filter)
        mask = np.array([p in peril_set for p in keys_tb['PerilID']])
        valid_area_peril_id = np.unique(keys_tb['AreaPerilID'][mask])
        logger.debug(
            f'Peril specific run: ({peril_filter}), '
            f'{len(valid_area_peril_id)} AreaPerilID included out of {len(keys_tb)}')
    else:
        valid_area_peril_id = None

    items = gul_get_items(input_path)
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
    if valid_area_peril_id is not None:
        items = items[np.isin(items['areaperil_id'], valid_area_peril_id)]

    # --- item map (hashmap + jagged array) -------------------------------------
    logger.debug('generate item map')
    (item_map_hm, item_map_hm_keys,
     item_map_ja_offsets) = generate_item_map(items, coverages)

    # --- correlations ----------------------------------------------------------
    logger.debug('import correlations')
    data = read_correlations(input_path, filename='correlations.bin')
    Nperil_correlation_groups = len(data)

    do_correlation = False
    if Nperil_correlation_groups > 0 and any(data['damage_correlation_value'] > 0):
        do_correlation = True

    if do_correlation:
        corr_data_by_item_id = np.ndarray(Nperil_correlation_groups + 1, dtype=correlations_dtype)
        corr_data_by_item_id[0] = (0, 0., 0., 0, 0.)
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
        # scalars
        'do_correlation': int(do_correlation),
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
    metadata = np.array([
        structures['do_correlation'],
    ], dtype=np.int64)
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
    result['do_correlation'] = int(metadata[0])

    return result
