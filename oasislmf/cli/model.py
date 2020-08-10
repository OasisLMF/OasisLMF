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
    computation_name = 'ExposurePreAnalysis'

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
    computation_name = 'OasisKeys'
    def action(self, args):
        super().action(args)


class GenerateOasisFilesCmd(OasisComputationCommand):
    """
    Generates the standard Oasis GUL input files + optionally the IL/FM input
    files and the RI input files.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'OasisFiles'

    def action(self, args):
        super().action(args)


class GenerateLossesCmd(OasisBaseCommand):
    """
    Generates losses using the installed ktools framework given Oasis files,
    model analysis settings JSON file, model data and model package data.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    Run ``oasislmf config --help`` for more information.

    The script creates a time-stamped folder in the model run directory and
    sets that as the new model run directory, copies the analysis settings
    JSON file into the run directory and creates the following folder
    structure
    ::

        |-- analysis_settings.json
        |-- fifo
        |-- input
        |-- output
        |-- RI_1
        |-- ri_layers.json
        |-- run_ktools.sh
        |-- static
        `-- work

    Depending on the OS type the model data is symlinked (Linux, Darwin) or
    copied (Cygwin, Windows) into the ``static`` subfolder. The input files
    are kept in the ``input`` subfolder and the losses are generated as CSV
    files in the ``output`` subfolder.
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-o', '--oasis-files-dir', default=None, help='Path to pre-existing direct Oasis files (GUL + FM input files)')
        parser.add_argument('-a', '--analysis-settings-json', default=None, help='Analysis settings JSON file path')
        parser.add_argument('-D', '--user-data-dir', default=None, help='Directory containing additional model data files which varies between analysis runs')
        parser.add_argument('-d', '--model-data-dir', default=None, help='Model data directory path')
        parser.add_argument('-r', '--model-run-dir', default=None, help='Model run directory path')
        parser.add_argument('-p', '--model-package-dir', default=None, help='Path containing model specific package')
        parser.add_argument('-B', '--model-custom-gulcalc', default=None, help='Callable custom gulcalc component to use as a drop-in replacement', type=str)
        parser.add_argument('-n', '--ktools-num-processes', default=None, help='Number of ktools calculation processes to use', type=int)
        parser.add_argument('-f', '--ktools-fifo-relative', default=None, help='Create ktools fifo queues under the ./fifo dir', action='store_true')
        parser.add_argument('-E', '--ktools-disable-guard', default=None, help='Disables error handling in the ktools run script (abort on non-zero exitcode or output on stderr)', action='store_true')
        parser.add_argument('-q', '--ktools-alloc-rule-gul', default=None, help='Override the allocation used in gulcalc', type=int)
        parser.add_argument('-u', '--ktools-alloc-rule-il', default=None, help='Override the fmcalc allocation rule used in direct insured loss', type=int)
        parser.add_argument('-U', '--ktools-alloc-rule-ri', default=None, help='Override the fmcalc allocation rule used in reinsurance', type=int)
        parser.add_argument('-X', '--ktools-legacy-stream', default=None, help='Run gulcalc using the legacy coverage/item steam, this option disables the GUL allocation rule', action='store_true')

    def action(self, args):
        """
        Generates losses using the installed ktools framework given Oasis files,
        model analysis settings JSON file, model data and model package data.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        self.logger.info('\nProcessing arguments - generating model losses')
        inputs = InputValues(args)

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        default_model_run_fp = os.path.join(os.getcwd(), 'runs', 'losses-{}'.format(utcnow))

        oasis_fp = inputs.get('oasis_files_dir', required=True, is_path=True)
        model_run_fp = inputs.get('model_run_dir', is_path=True, default=default_model_run_fp)
        analysis_settings_fp = inputs.get('analysis_settings_json', required=True, is_path=True)
        model_data_fp = inputs.get('model_data_dir', required=True, is_path=True)
        model_package_fp = as_path(inputs.get('model_package_dir', required=False, is_path=True), 'Model package path', is_dir=True)
        model_custom_gulcalc  = inputs.get('model_custom_gulcalc', required=False, is_path=False)
        user_data_dir = inputs.get('user_data_dir', required=False, is_path=True)

        ktools_num_processes = inputs.get('ktools_num_processes', default=None, required=False)
        ktools_fifo_relative = inputs.get('ktools_fifo_relative', default=False, required=False)
        ktools_error_guard = not(inputs.get('ktools_disable_guard', default=False, required=False))
        ktools_alloc_rule_gul = inputs.get('ktools_alloc_rule_gul', default=None, required=False)
        ktools_alloc_rule_il = inputs.get('ktools_alloc_rule_il', default=None, required=False)
        ktools_alloc_rule_ri = inputs.get('ktools_alloc_rule_ri', default=None, required=False)
        ktools_gul_legacy_stream = inputs.get('ktools_legacy_gul_stream', default=None, required=False)
        verbose_output = inputs.get('verbose', default=False, required=False)

        il = all(p in os.listdir(os.path.abspath(oasis_fp)) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])
        ri = False
        ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(os.path.abspath(oasis_fp)))
        self.logger.info('\nGenerating losses (GUL=True, IL={}, RIL={})'.format(il, ri))

        om().generate_model_losses(
            model_run_fp,
            oasis_fp,
            analysis_settings_fp,
            model_data_fp,
            model_package_fp=model_package_fp,
            model_custom_gulcalc=model_custom_gulcalc,
            ktools_num_processes=ktools_num_processes,
            ktools_fifo_relative=ktools_fifo_relative,
            ktools_error_guard=ktools_error_guard,
            ktools_alloc_rule_gul=ktools_alloc_rule_gul,
            ktools_alloc_rule_il=ktools_alloc_rule_il,
            ktools_alloc_rule_ri=ktools_alloc_rule_ri,
            ktools_debug=verbose_output,
            ktools_gul_legacy_stream=ktools_gul_legacy_stream,
            user_data_dir=user_data_dir,
        )

        self.logger.info('\nLosses generated in {}'.format(model_run_fp))


class RunCmd(OasisBaseCommand):
    """
    Run models end to end.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Run models end to end.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-z', '--keys-data-csv', default=None, help='Pre-generated keys CSV file path')

        parser.add_argument('-k', '--lookup-data-dir', default=None, help='Model lookup/keys data directory path')
        parser.add_argument('-l', '--lookup-module-path', default=None, help='Model lookup module path')
        parser.add_argument('-L', '--lookup-complex-config-json', default=None, help='Complex lookup config JSON file path')
        parser.add_argument('-m', '--lookup-config-json', default=None, help='Built-in lookup config JSON file path')

        parser.add_argument('-e', '--profile-loc-json', default=None, help='Source OED location profile JSON path')
        parser.add_argument('-b', '--profile-acc-json', default=None, help='Source OED accounts profile JSON path')
        parser.add_argument('-g', '--profile-fm-agg-json', default=None, help='FM OED aggregation profile JSON path')

        parser.add_argument('--exposure-pre-analysis-module', default=None,
                                help='Exposure Pre-Analysis lookup module path')
        parser.add_argument('--exposure-pre-analysis-class-name', default=None,
                                help='Name of the class to use for the exposure_pre_analysis')
        parser.add_argument('--exposure-pre-analysis-setting-json', default=None,
                                help='Exposure Pre-Analysis config JSON file path')

        parser.add_argument('-x', '--oed-location-csv', default=None, help='Source location CSV file path')
        parser.add_argument('-y', '--oed-accounts-csv', default=None, help='Source accounts CSV file path')
        parser.add_argument('-i', '--oed-info-csv', default=None, help='Reinsurance info. CSV file path')
        parser.add_argument('-s', '--oed-scope-csv', default=None, help='Reinsurance scope CSV file path')

        parser.add_argument('-a', '--analysis-settings-json', default=None, help='Model analysis settings JSON file path')
        parser.add_argument('-D', '--user-data-dir', default=None, help='Directory containing additional model data files which varies between analysis runs')

        parser.add_argument('-v', '--model-version-csv', default=None, help='Model version CSV file path')
        parser.add_argument('-M', '--model-settings-json', default=None, help='Model settings JSON file path')
        parser.add_argument('-d', '--model-data-dir', default=None, help='Model data directory path')
        parser.add_argument('-r', '--model-run-dir', default=None, help='Model run directory path')
        parser.add_argument('-p', '--model-package-dir', default=None, help='Path containing model specific package')
        parser.add_argument('-B', '--model-custom-gulcalc', default=None, help='Callable custom gulcalc component to use as a drop-in replacement', type=str)

        parser.add_argument('-n', '--ktools-num-processes', default=None, help='Number of ktools calculation processes to use', type=int)
        parser.add_argument('-f', '--ktools-fifo-relative', default=None, help='Create ktools fifo queues under the ./fifo dir', action='store_true')
        parser.add_argument('-E', '--ktools-disable-guard', default=None, help='Disables error handling in the ktools run script (abort on non-zero exitcode or output on stderr)', action='store_true')
        parser.add_argument('-q', '--ktools-alloc-rule-gul', default=None, help='Override the allocation used in gulcalc', type=int)
        parser.add_argument('-u', '--ktools-alloc-rule-il', default=None, help='Override the fmcalc allocation rule used in direct insured loss', type=int)
        parser.add_argument('-U', '--ktools-alloc-rule-ri', default=None, help='Override the fmcalc allocation rule used in reinsurance', type=int)
        parser.add_argument('--ktools-legacy-gul-stream', default=None, help='Run gulcalc using the legacy coverage/item steam, this option disables the GUL allocation rule', action='store_true')

        parser.add_argument('-S', '--disable-summarise-exposure', default=None, help='Create exposure summary report', action='store_false')
        parser.add_argument('-W', '--write-chunksize', type=int, help='Chunk size to use when writing input files from the inputs dataframes')
        parser.add_argument('-G', '--group-id-cols', nargs='+', default=None, help='Columns from loc file to set group_id')

    def action(self, args):
        """
        Generate Oasis files (items, coverages, GUL summary) for a model

        :param args: The arguments from the command line
        :type args: Namespace
        """
        self.logger.info('\nProcessing arguments - model run')
        inputs = InputValues(args)

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        default_model_run_fp = os.path.join(os.getcwd(), 'runs', 'losses-{}'.format(utcnow))

        model_run_fp = inputs.get('model_run_dir', is_path=True, default=default_model_run_fp)

        if os.path.exists(model_run_fp):
            empty_dir(model_run_fp)

        args.model_run_dir = model_run_fp

        model_package_fp = inputs.get('model_package_dir', required=False, is_path=True)

        args.model_package_path = model_package_fp

        accounts_fp = inputs.get('oed_accounts_csv', required=False, is_path=True)
        ri_info_fp = inputs.get('oed_info_csv', required=False, is_path=True)
        ri_scope_fp = inputs.get('oed_scope_csv', required=False, is_path=True)

        # Validate JSON files (Fail at entry point not after input generation)
        analysis_settings_fp = inputs.get('analysis_settings_json', required=False, is_path=True)
        if analysis_settings_fp:
            get_analysis_settings(analysis_settings_fp)
        model_settings_fp = inputs.get('model_settings_json', required=False, is_path=True)
        if model_settings_fp:
            get_model_settings(model_settings_fp)

        required_ri_paths = [ri_info_fp, ri_scope_fp]
        il = True if accounts_fp else False
        ri = all(required_ri_paths) and il

        if any(required_ri_paths) and not ri:
            raise OasisException(
                'RI option indicated by provision of some RI related assets, but other assets are missing. '
                'To generate RI inputs you need to provide all of the assets required to generate direct '
                'Oasis files (GUL + FM input files) plus all of the following assets: '
                '    reinsurance info. file path, '
                '    reinsurance scope file path.'
            )

        args.oasis_files_dir = os.path.join(model_run_fp, 'input')

        if inputs.get('exposure_pre_analysis_module', is_path=True):
            cmds = [GenerateExposurePreAnalysisCmd(args), GenerateOasisFilesCmd(args), GenerateLossesCmd(args)]
        else:
            cmds = [GenerateOasisFilesCmd(args), GenerateLossesCmd(args)]

        with tqdm(total=len(cmds)) as pbar:
            for cmd in cmds:
                cmd.action(args)
                pbar.update(1)

        self.logger.info('\nModel run completed successfully in {}'.format(model_run_fp))


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
