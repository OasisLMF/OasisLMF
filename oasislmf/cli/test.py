import tempfile
import os

from argparsetree import BaseCommand

from ..validation.model_data import csv_validity_test
from ..utils.path import as_path
from ..utils.inputs import InputValues

from argparse import RawDescriptionHelpFormatter
from .command import OasisBaseCommand, OasisComputationCommand



class ModelValidationCmd(OasisBaseCommand):
    """
    Checks the validity of a set of model data.
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """

        super(self.__class__, self).add_args(parser)

        parser.add_argument(
            '-d', '--model-data-path',
            default=None, help='Directory containing additional user-supplied model data files')

    def action(self, args):
        """
        Performs validity checks on model data csv files using ktools
        executables.

        :param args: The arguments from the command line
        :type args: Namespace
        """

        inputs = InputValues(args)

        model_data_fp = as_path(
            inputs.get('model_data_path', required=True, is_path=True), 'Model data path', is_dir=True)

        csv_validity_test(model_data_fp)


class FmValidationCmd(OasisComputationCommand):
    """
    Runs a set of FM tests.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'RunFmTest'


class GenerateDummyModelFilesCmd(OasisComputationCommand):
    
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateDummyModelFiles'


class GenerateDummyOasisFilesCmd(OasisComputationCommand):

    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateDummyOasisFiles'


class GenerateLossesDummyModelCmd(OasisComputationCommand):

    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateLossesDummyModel'


class TestModelCmd(BaseCommand):
    """
    Present sub-commands for creating test models
    """

    sub_commands = {
        'generate-model-files': GenerateDummyModelFilesCmd,
        'generate-oasis-files': GenerateDummyOasisFilesCmd,
        'run': GenerateLossesDummyModelCmd
    }


class TestCmd(BaseCommand):
    """
    Test models and keys servers
    """

    sub_commands = {
        'model-validation': ModelValidationCmd,
        'fm': FmValidationCmd,
        'model': TestModelCmd
    }
