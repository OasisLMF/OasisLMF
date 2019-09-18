from argparsetree import BaseCommand

from ..model_testing.validation import csv_validity_test

from ..utils.path import as_path

from .base import OasisBaseCommand
from .inputs import InputValues


class ModelValidationCmd(OasisBaseCommand):
    """
    Checks model data for validity.
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """

        super(self.__class__, self).add_args(parser)

        parser.add_argument('-d', '--model-data-path', default=None, help='Directory containing additional user-supplied model data files')

    def action(self, args):
        """
        Performs validity checks on model data csv files using ktools
        executables.

        :param args: The arguments from the command line
        :type args: Namespace
        """

        inputs = InputValues(args)

        model_data_fp = as_path(inputs.get('model_data_path', required=True, is_path=True), 'Model data path', is_dir=True)

        csv_validity_test(model_data_fp)


class TestCmd(BaseCommand):
    """
    Test models and keys servers
    """

    sub_commands = {
        'model-validation': ModelValidationCmd,
    }
