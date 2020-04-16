from argparse import RawDescriptionHelpFormatter

from .base import OasisBaseCommand
from .inputs import InputValues

from .. import __version__


class ConfigUpdateCmd(OasisBaseCommand):
    """
    Read in an MDK config file and writes an updated file, replacing deprecated keys
    with newer ones compatible with the current MDK release
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """

        super(self.__class__, self).add_args(parser)

        parser.add_argument('-o', '--output-config', default=None, help='File path to write an updated MDK config file')
        parser.add_argument('-y', '--no-confirm', default=False, help='No confirmation prompt before file write')

    def action(self, args):
        """
        :param args: The arguments from the command line
        :type args: Namespace
        """

        inputs = InputValues(args, update_keys=False)

        if inputs.obsolete_keys:
            # Update exec

            inputs.list_obsolete_keys(fix_warning=False)
            inputs.update_config_keys()
            new_config_fp = inputs.get('output_config') if inputs.get('output_config') else inputs.config_fp

            if inputs.get('no_confirm'):
                inputs.write_config_file(new_config_fp)
            else:
                msg = 'Write updated config file to "{}"?'.format(new_config_fp)
                if inputs.confirm_action(msg):
                    inputs.write_config_file(new_config_fp)
        else:
            self.logger.info('File "{}" is up to date with version {}'.format(
                inputs.config_fp,
                __version__,
            ))


class ConfigCmd(OasisBaseCommand):
    """
    Describes the format of the configuration (JSON) file to use the MDK
    ``model run`` command for running models end-to-end.

    One file will need to be defined per model, usually in the model repository
    and with an indicative name.

    The path-related keys should be strings, given relative to the location of
    configuration file. Optional arguments for a command are usually defaulted
    with appropriate values, but can be overridden by providing a runtime flag.

    :analysis_settings_json: Analysis settings (JSON) file path
    :lookup_data_dir: Model lookup/keys data path (optional)
    :lookup_config_json: Model built-in lookup config. (JSON) file path (optional)
    :lookup_package_dir: Model custom lookup package path (optional)
    :model_version_csv: Model version (CSV) file path (optional)
    :model_data_dir: Model data path
    :model_package_dir: Path to the directory to use as the model specific package (optional)
    :model_run_dir: Model run directory (optional)
    :oed_location_csv: Source OED exposure (CSV) file path
    :oed_accounts_csv: Source OED accounts (CSV) file path (optional)
    :oed_info_csv: Reinsurance (RI) info. file path (optional)
    :oed_scope_csv: RI scope file path (optional)
    :profile_location_csv: Source OED exposure (JSON) profile describing the financial terms contained in the source exposure file (optional)
    :profile_accounts_json: Source OED accouns (JSON) profile describing the financial terms contained in the source accounts file (optional)
    :summarise_exposure: Generates an exposure summary report in JSON
    :ktools_num_processes: The number of concurrent processes used by ktools during model execution - default is ``2``
    :ktools_fifo_relative: Whether to create ktools FIFO queues under the ``./fifo`` subfolder (in the model run directory)
    :ktools_alloc_rule_gul: Override the allocation used in ``fmcalc`` - default is ``1``
    :ktools_alloc_rule_il: Override the allocation used in ``fmcalc`` - default is ``2``
    """
    formatter_class = RawDescriptionHelpFormatter

    sub_commands = {
        'update': ConfigUpdateCmd,
    }
