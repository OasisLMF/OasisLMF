import oasislmf.cli.base
import tempfile
import os

from argparsetree import BaseCommand

from ..model_testing.validation import csv_validity_test

from ..utils.path import as_path

from ..utils.exceptions import OasisException
from .base import OasisBaseCommand
from .inputs import InputValues

from ..manager import OasisManager as om

# ! Handle multiple tests


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


class FmValidationCmd(OasisBaseCommand):
    """
    Runs a set of FM tests.
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """

        super(self.__class__, self).add_args(parser)

        parser.add_argument(
            '-c', '--test-case-name',
            type=str, default=None, required=False,
            help='Test case name - runs a specific test in the test directory. Otherwise run all tests.'
        )
        parser.add_argument(
            '-l', '--list-tests',
            action='store_true',
            help='List the valid test cases in the test directory rather than running'
        )
        parser.add_argument(
            '-t', '--test-dir',
            type=str, default=None, required=True,
            help='Test directory - should contain test directories containing OED files and expected results'
        )
        parser.add_argument(
            '-r', '--run-dir',
            type=str, default=None, required=False,
            help='Run directory - where files should be generated. If not sst, no files will be saved.'
        )

    def action(self, args):
        """
        Runs a set of FM tests.

        :param args: The arguments from the command line
        :type args: Namespace
        """

        inputs = InputValues(args)

        call_dir = os.getcwd()

        do_list_tests = inputs.get('list_tests')

        test_case_name = inputs.get('test_case_name')
        run_all_tests = test_case_name is None

        test_dir = as_path(
            inputs.get(
                'test_dir', default=call_dir, is_path=True),
            'Source files directory', is_dir=True, preexists=True)

        test_case_names = []
        if run_all_tests:
            for test_case in os.listdir(path=test_dir):
                if os.path.exists(
                    os.path.join(test_dir, test_case, "expected")
                ):
                    test_case_names.append(test_case)
        else:
            test_case_names.append(test_case_name)

        test_case_names.sort()

        status = 0

        if do_list_tests:
            for test_case_name in test_case_names:
                print(test_case_name)
            exit(0)

        failed_tests = []

        for test_case_name in test_case_names:

            test_case_dir = os.path.join(test_dir, test_case_name)
            if not os.path.exists(test_case_dir):
                raise OasisException(f"Test directory does not exist: {test_case_name}")

            run_dir = as_path(inputs.get('run_dir', is_path=True), 'Run directory', is_dir=True, preexists=False)
            if run_dir is None:
                with tempfile.TemporaryDirectory() as tmp_run_dir:
                    test_result = om().run_fm_test(test_case_dir, tmp_run_dir)
            else:
                run_dir = os.path.join(run_dir, test_case_name)
                test_result = om().run_fm_test(test_case_dir, run_dir)

            if not test_result:
                failed_tests.append(test_case_name)
                if status == 0:
                    status = 1

        if len(failed_tests) == 0:
            self.logger.info("All tests passed")
        else:
            self.logger.info("{} test failed: ".format(len(failed_tests)))
            [self.logger.info(n) for n in failed_tests]

        exit(status)


class TestCmd(BaseCommand):
    """
    Test models and keys servers
    """

    sub_commands = {
        'model-validation': ModelValidationCmd,
        'fm': FmValidationCmd
    }
