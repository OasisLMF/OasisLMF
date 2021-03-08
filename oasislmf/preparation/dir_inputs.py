__all__ = [
    'create_target_directory',
    'prepare_input_files_directory'
]

import filecmp
import os
import shutil

from pathlib import Path

from ..utils.exceptions import OasisException
from ..utils.path import as_path
from ..utils.defaults import store_exposure_fp


def create_target_directory(target_dir, label):
    target_dir = as_path(target_dir, label, is_dir=True, preexists=False)
    if not os.path.exists(target_dir):
        Path(target_dir).mkdir(parents=True, exist_ok=True)

    return target_dir


def prepare_input_files_directory(
    target_dir,
    exposure_fp,
    exposure_profile_fp=None,
    keys_fp=None,
    keys_errors_fp=None,
    lookup_config_fp=None,
    model_version_fp=None,
    complex_lookup_config_fp=None,
    accounts_fp=None,
    accounts_profile_fp=None,
    fm_aggregation_profile_fp=None,
    ri_info_fp=None,
    ri_scope_fp=None
):
    try:
        # Prepare the target directory and copy the source files, profiles and
        # model version file into it
        target_dir = create_target_directory(
            target_dir, 'target Oasis files directory'
        )

        # Copy preserving original filenames 
        paths = [
            (p, os.path.join(target_dir, os.path.basename(p))) for p in (
                exposure_profile_fp, accounts_profile_fp,
                fm_aggregation_profile_fp, lookup_config_fp, model_version_fp,
                keys_fp, keys_errors_fp
            ) if p
        ]

        # Copy and rename to default set in 
        # oasislmf.utils.defaults.SOURCE_FILENAMES
        paths_rename = (
            (exposure_fp, "loc"), 
            (accounts_fp, "acc"), 
            (ri_info_fp, "info"), 
            (ri_scope_fp, "scope"),
            (complex_lookup_config_fp, "complex_lookup")
        )    

        for fp, key in paths_rename:
            if fp:
                # check if exposure pre-analysis has run:
                if not os.path.exists(os.path.join(target_dir, f'epa_{store_exposure_fp(fp, key)}')):
                    paths.append((fp, os.path.join(target_dir, store_exposure_fp(fp, key))))

        for src, dst in paths:
            if src and os.path.exists(src):
                shutil.copy2(src, dst) if not (os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False)) else None
    except (FileNotFoundError, IOError, OSError, shutil.Error, TypeError, ValueError) as e:
        raise OasisException("Exception raised in 'prepare_input_files_directory'", e)

    return target_dir
