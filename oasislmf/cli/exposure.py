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
    'ExposureCmd',
    'RunCmd',
    'ValidateCmd'
]

import argparse
import io
import importlib
import inspect
import json
import os
import re
import shutil
import time
import sys

from argparse import RawDescriptionHelpFormatter

from pathlib2 import Path
from six import text_type as _unicode
from tabulate import tabulate

from ..manager import OasisManager as om

from ..utils.exceptions import OasisException
from ..utils.data import (
    get_json,
    print_dataframe,
)
from ..utils.defaults import get_default_deterministic_analysis_settings
from ..utils.path import (
    as_path,
    setcwd,
)
from .base import (
    InputValues,
    OasisBaseCommand,
)


class RunCmd(OasisBaseCommand):
    """
    Generates deterministic losses using the installed ktools framework given
    direct Oasis files (GUL + IL input files & optionally RI input files).

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    Run ``oasislmf config --help`` for more information.

    The script creates a time-stamped folder in the model run directory and
    sets that as the new model run directory, copies the analysis settings
    JSON file into the run directory and creates the following folder
    structure
    ::

        |── analysis_settings.json
        |── fifo/
        |── input/
        |── output/
        |── static/
        |── work/

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

        parser.add_argument(
            '-o', '--output-dir', type=str, default=None, required=True,
            help='Output directory')
        parser.add_argument(
            '-i', '--input-dir', type=str, default=None, required=True,
            help='Input directory - should contain the OED exposure file + optionally the accounts, and RI info. and scope files')
        parser.add_argument(
            '-l', '--loss-factor', type=float, default=None,
            help='Loss factor to apply to TIVs.')
        parser.add_argument(
            '-n', '--net-losses', default=False, help='Net losses', action='store_true'
            )

    def action(self, args):
        """
        Generates deterministic losses using the installed ktools framework given
        direct Oasis files (GUL + IL input files & optionally RI input files).

        :param args: The arguments from the command line
        :type args: Namespace
        """
        self.logger.info('\nProcessing arguments')
        inputs = InputValues(args)

        call_dir = os.getcwd()
        output_dir = as_path(inputs.get('output_dir', default=os.path.join(call_dir, 'output'), is_path=True), 'Output directory', is_dir=True, preexists=False)
        if not os.path.exists(output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        input_dir = as_path(inputs.get('input_dir', default=call_dir, is_path=True), 'Input directory', is_dir=True, preexists=True)

        loss_factor = inputs.get('loss_factor', default=1.0, required=False)

        net_losses = inputs.get('net_losses', default=False, required=False)

        il = ri = False
        try:
            li = [fn for fn in os.listdir(input_dir) if fn.lower().startswith('account')][0]
        except IndexError:
            pass
        else:
            il = True
            try:
                li = [fn for fn in os.listdir(input_dir) if 'reinsinfo' in fn.lower()][0]
            except IndexError:
                pass
            else:
                try:
                    li = [fn for fn in os.listdir(input_dir) if 'reinsscope' in fn.lower()][0]
                except IndexError:
                    pass
                else:
                    ri = True

        analysis_settings_fp = None
        try:
            analysis_settings_fp = [fn.lower() for fn in os.listdir(input_dir) if fn == 'analysis_settings.json'][0]
        except IndexError:
            analysis_settings_fp = get_default_deterministic_analysis_settings(path=True)

        self.logger.info('\nRunning deterministic losses (GUL=True, IL={}, RI={})'.format(il, ri))
        guls, ils, rils = om().run_deterministic(
            input_dir,
            output_dir,
            loss_percentage_of_tiv=loss_factor,
            net=net_losses
        )
        print_dataframe(guls, header='Ground-up losses', objectify_cols=guls.columns, headers='keys', tablefmt='psql', floatfmt=".2f")
        print_dataframe(ils, header='Insured losses', objectify_cols=ils.columns, headers='keys', tablefmt='psql', floatfmt=".2f")
        if rils is not None:
            print_dataframe(rils, header='Reinsurance losses', objectify_cols=rils.columns, headers='keys', tablefmt='psql', floatfmt=".2f")


class ValidateCmd(OasisBaseCommand):
    pass


class ExposureCmd(OasisBaseCommand):
    """
    Exposure subcommands::

        * generate deterministic losses (GUL, IL or RI)
        * validate OED exposure (not yet implemented)
    """
    sub_commands = {
        'run': RunCmd,
        'validate': ValidateCmd
    }
