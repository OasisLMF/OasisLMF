"""Pre-compute and persist getmodel (modelpy) read-only data structures.

Follows the same pattern as ``oasislmf.pytools.gulmc.structure``:
  - ``create_getmodel_structure`` builds all read-only numpy arrays once and
    saves them as ``.npy`` files.
  - ``load_getmodel_structure`` memory-maps them via ``np.load(mmap_mode='r')``,
    allowing multiple modelpy processes to share physical memory pages through
    the OS page cache.
"""
import logging
import os

import numpy as np
import pandas as pd
from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.getmodel.common import Keys
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel.manager import (
    get_items, get_vulns, get_mean_damage_bins, convert_vuln_id_to_index,
)

logger = logging.getLogger(__name__)

STRUCTURE_DIR = 'getmodel_structure'

ARRAY_FILES = [
    'vuln_array',
    'vulns_id',
    'areaperil_id_ind',
    'areaperil_to_vulns_idx_array',
    'areaperil_to_vulns',
    'unique_areaperil_ids',
    'mean_damage_bins',
]


def _structure_path(run_dir):
    return os.path.join(run_dir, 'input', STRUCTURE_DIR)


def getmodel_structure_exists(run_dir):
    """Check whether pre-computed getmodel structures exist."""
    return os.path.isfile(os.path.join(_structure_path(run_dir), 'metadata.npy'))


def build_structures(run_dir, ignore_file_type, peril_filter,
                     model_df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader"):
    """Build all read-only getmodel data structures from input files.

    This extracts the preparation logic from ``manager.run()`` into a
    standalone callable so that it can be invoked once (by
    ``create_getmodel_structure``) rather than repeated in every parallel
    modelpy process.

    Args:
        run_dir (str): path to the run directory.
        ignore_file_type (set[str]): file extensions to ignore when loading.
        peril_filter (list): list of perils to include (empty = all).
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
    if os.path.exists(os.path.join(input_path, 'keys.csv')):
        keys_df = pd.read_csv(os.path.join(input_path, 'keys.csv'), dtype=Keys)
        if peril_filter:
            valid_area_peril_id = np.unique(
                keys_df.loc[keys_df['PerilID'].isin(peril_filter), 'AreaPerilID'])
            logger.debug(
                f'Peril specific run: ({peril_filter}), '
                f'{len(valid_area_peril_id)} AreaPerilID included out of {len(keys_df)}')
        else:
            valid_area_peril_id = keys_df['AreaPerilID']
    else:
        valid_area_peril_id = None

    # --- items -----------------------------------------------------------------
    logger.debug('import items')
    vuln_map, vuln_map_keys, areaperil_id_ind, areaperil_to_vulns_idx_array, \
        areaperil_to_vulns, unique_areaperil_ids = get_items(
            input_path, ignore_file_type,
            valid_area_peril_id if peril_filter else None)

    # --- footprint (temporary open to get num_intensity_bins) ------------------
    logger.debug('import footprint (header only)')
    with Footprint.load(model_storage, ignore_file_type,
                        df_engine=model_df_engine,
                        areaperil_ids=list(unique_areaperil_ids)) as footprint_obj:
        num_intensity_bins = footprint_obj.num_intensity_bins

    # --- vulnerabilities -------------------------------------------------------
    logger.debug('import vulnerabilities')
    vuln_array, vulns_id, num_damage_bins = get_vulns(
        model_storage, run_dir, vuln_map, vuln_map_keys,
        num_intensity_bins, ignore_file_type, df_engine=model_df_engine)

    # convert vulnerability IDs in areaperil_to_vulns to dense indices
    convert_vuln_id_to_index(vuln_map, vuln_map_keys, areaperil_to_vulns)

    # --- mean damage bins ------------------------------------------------------
    logger.debug('import mean damage bins')
    mean_damage_bins = get_mean_damage_bins(model_storage, ignore_file_type)

    # --- pack everything into a dict -------------------------------------------
    return {
        'vuln_array': vuln_array,
        'vulns_id': vulns_id,
        'areaperil_id_ind': areaperil_id_ind,
        'areaperil_to_vulns_idx_array': areaperil_to_vulns_idx_array,
        'areaperil_to_vulns': areaperil_to_vulns,
        'unique_areaperil_ids': unique_areaperil_ids,
        'mean_damage_bins': mean_damage_bins,
        # scalars
        'num_damage_bins': num_damage_bins,
        'num_intensity_bins': num_intensity_bins,
    }


def create_getmodel_structure(run_dir, ignore_file_type, peril_filter,
                              model_df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader"):
    """Build and save all read-only getmodel data structures as ``.npy`` files.

    Args:
        run_dir (str): path to the run directory.
        ignore_file_type (set[str]): file extensions to ignore when loading.
        peril_filter (list): list of perils to include (empty = all).
        model_df_engine (str): engine for loading model dataframes.
    """
    structures = build_structures(run_dir, ignore_file_type, peril_filter,
                                  model_df_engine)

    structure_path = _structure_path(run_dir)
    os.makedirs(structure_path, exist_ok=True)

    for name in ARRAY_FILES:
        np.save(os.path.join(structure_path, name), structures[name])

    # save scalar metadata
    metadata = np.array([
        structures['num_damage_bins'],
        structures['num_intensity_bins'],
    ], dtype=np.int64)
    np.save(os.path.join(structure_path, 'metadata'), metadata)

    total_bytes = sum(
        os.path.getsize(os.path.join(structure_path, f'{name}.npy'))
        for name in ARRAY_FILES
    )
    logger.info(f"getmodel structures saved to {structure_path} ({total_bytes / 1024 / 1024:.1f} MB)")


def load_getmodel_structure(run_dir):
    """Load pre-computed getmodel structures via memory-mapped numpy files.

    Each array is loaded with ``mmap_mode='r'`` so that multiple modelpy
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
    result['num_damage_bins'] = int(metadata[0])
    result['num_intensity_bins'] = int(metadata[1])

    return result
