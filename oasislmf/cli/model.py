__all__ = [
    'GenerateExposurePreAnalysisCmd',
    'GenerateKeysCmd',
    'GenerateLossesCmd',
    'GenerateOasisFilesCmd',
    'GeneratePerilAreasRtreeFileIndexCmd',
    'ModelCmd',
    'RunCmd'
]

import os
import json
import re

from argparse import RawDescriptionHelpFormatter

from tqdm import tqdm

from ..manager import OasisManager as om

from ..utils.exceptions import OasisException
from ..utils.path import (
    as_path,
    empty_dir,
)
from ..utils.data import (
    get_utctimestamp,
    get_analysis_settings,
    get_model_settings,
)
from ..utils.defaults import (
    get_default_exposure_profile,
    get_default_accounts_profile,
    get_default_fm_aggregation_profile,
    KEY_NAME_TO_FILE_NAME,
)
from .base import OasisBaseCommand
from .computation_command import OasisComputationCommand
from .inputs import InputValues


class GeneratePerilAreasRtreeFileIndexCmd(OasisBaseCommand):
    """
    Generates and writes an Rtree file index of peril area IDs (area peril IDs)
    and area polygon bounds from a peril areas (area peril) file.
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-m', '--lookup-config-json', default=None, help='Lookup config JSON file path')
        parser.add_argument('-d', '--lookup-data-dir', default=None, help='Keys data directory path')
        parser.add_argument('-f', '--index-output-file', default=None, help='Index file path (no file extension required)')

    def action(self, args):
        """
        Generates and writes an Rtree file index of peril area IDs (area peril IDs)
        and area polygon bounds from a peril areas (area peril) file.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        keys_data_fp = inputs.get('lookup_data_dir', required=True, is_path=True)
        rtree_index_fp = inputs.get('index_output_file', required=True, is_path=True)
        lookup_config_fp = inputs.get('lookup_config_json', required=True, is_path=True)

        _index_fp = om().generate_peril_areas_rtree_file_index(
            keys_data_fp=keys_data_fp,
            areas_rtree_index_fp=rtree_index_fp,
            lookup_config_fp=lookup_config_fp
        )
        self.logger.info('\nGenerated peril areas Rtree file index {}\n'.format(_index_fp))


class GenerateExposurePreAnalysisCmd(OasisComputationCommand):
    """
    Generate a new EOD from original one by specifying a model specific pre-analysis hook for exposure modification
    see ExposurePreAnalysis for more detail
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'HookPreAnalysis'

    def action(self, args):  # TODO remove once integrated with archi 2020
        super().action(args)

        if args.model_run_dir:
            input_dir = os.path.join(args.model_run_dir, 'input')
            for input_name in ('oed_location_csv', 'oed_accounts_csv', 'oed_info_csv', 'oed_scope_csv'):
                setattr(args, input_name, os.path.join(input_dir, KEY_NAME_TO_FILE_NAME[input_name]))


class GenerateKeysCmd(OasisComputationCommand):
    """
    Generates keys from a model lookup, and write Oasis keys and keys error files.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateKeys'


class GenerateOasisFilesCmd(OasisComputationCommand):
    """
    Generates the standard Oasis GUL input files + optionally the IL/FM input
    files and the RI input files.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateOasisFiles'


class GenerateLossesCmd(OasisComputationCommand):
    """
    Generates the standard Oasis GUL input files + optionally the IL/FM input
    files and the RI input files.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateLosses'


class RunCmd(OasisComputationCommand):
    """
    Run models end to end.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'RunModel'


class ModelCmd(OasisBaseCommand):
    """
    Model subcommands::

        * generating an Rtree spatial index for the area peril lookup component of the built-in lookup framework
        * generating keys files from model lookups
        * generating Oasis input CSV files (GUL [+ IL, RI])
        * generating losses from a preexisting set of Oasis input CSV files
        * generating deterministic losses (no model)
        * running a model end-to-end
    """
    sub_commands = {
        'generate-peril-areas-rtree-file-index': GeneratePerilAreasRtreeFileIndexCmd,
        'generate-exposure-pre-analysis': GenerateExposurePreAnalysisCmd,
        'generate-keys': GenerateKeysCmd,
        'generate-oasis-files': GenerateOasisFilesCmd,
        'generate-losses': GenerateLossesCmd,
        'run': RunCmd
    }
