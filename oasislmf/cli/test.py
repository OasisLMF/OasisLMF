import os, sys

from argparsetree import BaseCommand

from ..model_testing.validation import csv_validity_test

from ..utils.path import as_path

from .base import OasisBaseCommand
from .inputs import InputValues
from filecmp import cmp as compare_files

from pathlib2 import Path
from ..manager import OasisManager as om

from itertools import chain

from ..utils.defaults import (
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT,
    OASIS_FILES_PREFIXES,
)
from ..utils.exceptions import OasisException


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


class FmValidationCmd(OasisBaseCommand):
    """
    Runs FM test.
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """

        super(self.__class__, self).add_args(parser)

        parser.add_argument(
            '-t', '--test-case-name', type=str, default=None, required=False, help='Test case name'
        )
        parser.add_argument(
            '-s', '--src-dir', type=str, default=None, required=True,
            help='Source files directory - should contain the OED exposure file + optionally the accounts, and RI info. and scope files'
        )
        parser.add_argument(
            '-r', '--run-dir', type=str, default=None, required=False, help='Run directory - where files should be generated'
        )


    def action(self, args):
        """
        Performs validity checks on model data csv files using ktools
        executables.

        :param args: The arguments from the command line
        :type args: Namespace
        """

        inputs = InputValues(args)

        call_dir = os.getcwd()

        test_case_name = inputs.get('test_case_name')
        src_dir = as_path(inputs.get('src_dir', default=call_dir, is_path=True), 'Source files directory', is_dir=True, preexists=True)
        src_dir = os.path.join(src_dir, test_case_name)
        ##! Check that src_dir exists

        run_dir = as_path(inputs.get('run_dir', default=os.path.join(src_dir, 'run'), is_path=True), 'Run directory', is_dir=True, preexists=False)
        if not os.path.exists(run_dir):
            Path(run_dir).mkdir(parents=True, exist_ok=True)

        loss_factor = 1.0 
        net_ri = True
        il_alloc_rule = KTOOLS_ALLOC_IL_DEFAULT
        ri_alloc_rule = KTOOLS_ALLOC_RI_DEFAULT
        output_level = 'loc'
        output_file = os.path.join(run_dir, 'loc_summary.csv')


        (il, ril) = om().run_exposure_wrapper(
            src_dir, run_dir, loss_factor, net_ri, 
            il_alloc_rule, ri_alloc_rule, output_level, output_file)

        expected_data_dir = os.path.join(src_dir, 'expected')
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
        for f in files:
            generated = os.path.join(run_dir, f)
            expected = os.path.join(expected_data_dir, f)

            if not os.path.exists(expected):
                continue

            self.logger.info('\nComparing generated {} vs expected {}'.format(generated, expected))
            try:
                assert(compare_files(generated, expected) is True)
            except AssertionError:
                status = 'FAIL'
                self.logger.info('\n{}'.format(column_diff(generated, expected)))
                self.logger.info('\tFAIL')
            else:
                self.logger.info('\n\tPASS')

        self.logger.info(
            '\n{} validation complete: {}'.format(test_case_name, status) if test_case_name
            else 'Validation complete: {}'.format(status)
        )

        sys.exit(0 if status == 'PASS' else -1)

class TestCmd(BaseCommand):
    """
    Test models and keys servers
    """

    sub_commands = {
        'model-validation': ModelValidationCmd,
        'fm': FmValidationCmd
    }
