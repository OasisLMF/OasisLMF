# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open
from builtins import str

from future import standard_library
standard_library.install_aliases()

__all__ = [
    'prepare_input_files_directory'
]

import filecmp
import os
import shutil

from pathlib2 import Path

from ..utils.exceptions import OasisException
from ..utils.path import as_path


def prepare_input_files_directory(
    target_dir,
    exposure_fp,
    exposure_profile_fp=None,
    keys_fp=None,
    lookup_config_fp=None,
    model_version_fp=None,
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

        paths = [p for p in (
            exposure_fp, exposure_profile_fp, accounts_fp, accounts_profile_fp,
                fm_aggregation_profile_fp, lookup_config_fp, model_version_fp,
                keys_fp, ri_info_fp, ri_scope_fp
            )
            if p
        ]
        for src in paths:
            if src and os.path.exists(src):
                dst = os.path.join(target_dir, os.path.basename(src))
                shutil.copy2(src, target_dir) if not (os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False)) else None
    except (FileNotFoundError, IOError, OSError, shutil.Error, TypeError, ValueError) as e:
        raise OasisException(e)

    return target_dir
