__all__ = [
    'prepare_input_files_directory'
]

import filecmp
import os
import shutil

from pathlib2 import Path

from ..utils.exceptions import OasisException
from ..utils.path import as_path
from ..utils.defaults import store_exposure_fp


def prepare_input_files_directory(
    target_dir,
    exposure_fp,
    exposure_profile_fp=None,
    keys_fp=None,
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
        target_dir = as_path(target_dir, 'target Oasis files directory', is_dir=True, preexists=False)
        if not os.path.exists(target_dir):
            Path(target_dir).mkdir(parents=True, exist_ok=True)

        paths = [
            (p, os.path.join(target_dir, os.path.basename(p))) for p in (
                exposure_profile_fp, accounts_profile_fp,
                fm_aggregation_profile_fp, lookup_config_fp, model_version_fp,
                complex_lookup_config_fp, keys_fp
            ) if p
        ]

        if exposure_fp:
            paths.append((exposure_fp, os.path.join(target_dir, store_exposure_fp(exposure_fp, 'loc'))))
        if accounts_fp:
            paths.append((accounts_fp, os.path.join(target_dir, store_exposure_fp(accounts_fp, 'acc'))))
        if ri_info_fp:
            paths.append((ri_info_fp, os.path.join(target_dir, store_exposure_fp(ri_info_fp, 'info'))))
        if ri_scope_fp:
            paths.append((ri_scope_fp, os.path.join(target_dir, store_exposure_fp(ri_scope_fp, 'scope'))))

        for src, dst in paths:
            if src and os.path.exists(src):
                shutil.copy2(src, dst) if not (os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False)) else None
    except (FileNotFoundError, IOError, OSError, shutil.Error, TypeError, ValueError) as e:
        raise OasisException("Exception raised in 'prepare_input_files_directory'", e)

    return target_dir
