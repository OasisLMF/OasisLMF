# -*- coding: utf-8 -*-
from argparse import RawDescriptionHelpFormatter

from .base import OasisBaseCommand


class ConfigCmd(OasisBaseCommand):
    """
    Describes the format of the configuration (JSON) file to use the MDK
    ``model run`` command for running models end-to-end.

    One file will need to be defined per model, usually in the model repository
    and with an indicative name.

    The path-related keys should be strings, given relative to the location of
    configuration file. Optional arguments for a command are usually defaulted
    with appropriate values, but can be overridden by providing a runtime flag.

    :analysis_settings_file_path: Analysis settings (JSON) file path
    :keys_data_path: Model lookup/keys data path (optional)
    :lookup_config_file_path: Model built-in lookup config. (JSON) file path (optional)
    :lookup_package_path: Model custom lookup package path (optional)
    :model_version_file_path: Model version (CSV) file path (optional)
    :model_data_path: Model data path
    :model_package_path: Path to the directory to use as the model specific package (optional)
    :model_run_dir: Model run directory (optional)
    :ri_info_fp: Reinsurance (RI) info. file path (optional)
    :ri_scope_fp: RI scope file path (optional)
    :source_accounts_file_path: Source OED accounts (CSV) file path (optional)
    :source_accounts_profile_path: Source OED accouns (JSON) profile describing the financial terms contained in the source accounts file (optional)
    :source_exposure_file_path: Source OED exposure (CSV) file path
    :source_exposure_profile_path: Source OED exposure (JSON) profile describing the financial terms contained in the source exposure file (optional)
    :ktools_num_processes: The number of concurrent processes used by ktools during model execution - default is ``2``
    :ktools_mem_limit: Whether to force exec. failure if ktools hits the system memory limit - default is ``False``
    :ktools_fifo_relative: Whether to create ktools FIFO queues under the ``./fifo`` subfolder (in the model run directory)
    :ktools_alloc_rule: Override the allocation used in ``fmcalc`` - default is ``2``
   """
    formatter_class = RawDescriptionHelpFormatter
