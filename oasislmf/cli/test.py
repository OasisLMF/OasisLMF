import os
import sys
import csv
import tempfile

from argparsetree import BaseCommand

from ..model_testing.validation import csv_validity_test

from ..utils.path import as_path

from ..utils.exceptions import OasisException
from .base import OasisBaseCommand
from .inputs import InputValues
from filecmp import cmp as compare_files

from pathlib2 import Path
from ..manager import OasisManager as om
from ..utils.diff import column_diff
from itertools import chain

from ..utils.defaults import (
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT,
    OASIS_FILES_PREFIXES,
)
from ..utils.exceptions import OasisException

##! Handle multiple tests

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
            help='Test case name - runs a specific test in the test directory'
        )
        parser.add_argument(
            '-l', '--list-tests', 
            type=bool, default=False,
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


    def run_test(self, 
            test_case_dir, run_dir, loss_factor, net_ri,
            il_alloc_rule, ri_alloc_rule, output_level):

            output_file = os.path.join(run_dir, 'loc_summary.csv')
            (il, ril) = om().run_exposure_wrapper(
                test_case_dir, run_dir, loss_factor, net_ri,
                il_alloc_rule, ri_alloc_rule, output_level, output_file)

            expected_data_dir = os.path.join(test_case_dir, 'expected')
            if not os.path.exists(expected_data_dir):
                raise OasisException(
                    'No subfolder named `expected` found in the input directory - '
                    'this subfolder should contain the expected set of GUL + IL '
                    'input files, optionally the RI input files, and the expected '
                    'set of GUL, IL and optionally the RI loss files'
                )

            files = ['keys.csv', 'loc_summary.csv']
            files += [
                '{}.csv'.format(fn)
                for ft, fn in chain(OASIS_FILES_PREFIXES['gul'].items(), OASIS_FILES_PREFIXES['il'].items())
            ]
            files += ['gul_summary_map.csv', 'guls.csv']
            if il:
                files += ['fm_summary_map.csv', 'ils.csv']
            if ril:
                files += ['rils.csv']

            status = 'PASS'
            test_result = True
            for f in files:
                generated = os.path.join(run_dir, f)
                expected = os.path.join(expected_data_dir, f)

                if not os.path.exists(expected):
                    continue

                file_test_result = compare_files(generated, expected)
                if not file_test_result:
                    self.logger.debug('\n FAIL: generated {} vs expected {}'.format(generated, expected))
                test_result = test_result and file_test_result

            return file_test_result


    def action(self, args):
        """
        Runs a set of FM tests.

        :param args: The arguments from the command line
        :type args: Namespace
        """

        inputs = InputValues(args)

        call_dir = os.getcwd()

        test_case_name = inputs.get('test_case_name')
        run_all_tests = test_case_name is None

        test_case_names = []
        if run_all_tests:
            pass
        else:
            test_case_names.append(test_case_name)

        for test_case_name in test_case_names:

            test_dir = as_path(
                inputs.get('test_dir', default=call_dir, is_path=True), 'Source files directory', is_dir=True, preexists=True)

            test_case_dir = os.path.join(test_dir, test_case_name)
            # ! Check that src_dir exists

            loss_percentages_fp = os.path.join(test_case_dir, 'loss_percentages.csv')
            loss_factor = []
            print(loss_percentages_fp)
            if os.path.exists(loss_percentages_fp):
                loss_factor = []
                try:
                    with open(loss_percentages_fp, 'r') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            loss_factor.append(
                                float(row['percentage_of_tiv']))

                except:
                    raise OasisException(f"Failed to read {loss_percentages_fp}")
            else:
                loss_factor.append(1.0)

            net_ri = True
            il_alloc_rule = KTOOLS_ALLOC_IL_DEFAULT
            ri_alloc_rule = KTOOLS_ALLOC_RI_DEFAULT
            output_level = 'loc'

            run_dir = as_path(inputs.get('run_dir', is_path=True), 'Run directory', is_dir=True, preexists=False)    
            if run_dir is None:
                with tempfile.TemporaryDirectory() as tmp_run_dir:
                    test_result = self.run_test( 
                        test_case_dir, tmp_run_dir, loss_factor, net_ri,
                        il_alloc_rule, ri_alloc_rule, output_level)
            else:
                run_dir = os.path.join(run_dir, test_case_name)
                test_result = self.run_test( 
                    test_case_dir, run_dir, loss_factor, net_ri,
                    il_alloc_rule, ri_alloc_rule, output_level)

            self.logger.info(
                '{}: {}'.format(test_case_name, test_result)
            )



class TestCmd(BaseCommand):
    """
    Test models and keys servers
    """

    sub_commands = {
        'model-validation': ModelValidationCmd,
        'fm': FmValidationCmd
    }
