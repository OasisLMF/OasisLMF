# -*- coding: utf-8 -*-

from __future__ import print_function

__all__ = [
    'ExposureCmd',
    'RunDeterministicCmd',
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
import subprocess
import time
import sys

from argparse import RawDescriptionHelpFormatter

from pathlib2 import Path
from six import u as _unicode
from tabulate import tabulate

from ..manager import OasisManager as om

from ..utils.exceptions import OasisException
from ..utils.data import get_json
from ..utils.path import (
    as_path,
    setcwd,
)
from .base import (
    InputValues,
    OasisBaseCommand,
)


class RunDeterministicCmd(OasisBaseCommand):
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

        ├── analysis_settings.json
        ├── fifo/
        ├── input/
        ├── output/
        ├── static/
        └── work/

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
        output_dir = as_path(inputs.get('output_dir', default=os.path.join(call_dir, 'output'), is_path=True), 'Output directory', preexists=False)
        if not os.path.exists(output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        input_dir = as_path(inputs.get('input_dir', default=call_dir, is_path=True), 'Input directory', preexists=True)

        loss_factor = inputs.get('loss_factor', default=1.0, required=False)

        net_losses = inputs.get('net_losses', default=False, required=False)

        il = all(p in os.listdir(input_dir) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])
        ri = False
        if os.path.basename(input_dir) == 'input':
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(input_dir))
        elif os.path.basename(input_dir) == 'csv':
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(os.path.dirname(input_dir)))

        self.logger.info('\nGenerating deterministic losses (GUL=True, IL={}, RI={})'.format(il, ri))
        losses_df = om().run_deterministic(input_dir, output_dir, loss_percentage_of_tiv=loss_factor, net=net_losses, print_losses=False)
        losses_df['event_id'] = losses_df['event_id'].astype(object)
        losses_df['output_id'] = losses_df['output_id'].astype(object)

        self.logger.info('\nLosses generated'.format(il, ri))
        print(tabulate(losses_df, headers='keys', tablefmt='psql', floatfmt=".2f"))


class ValidateCmd(OasisBaseCommand):
    pass


class ExposureCmd(OasisBaseCommand):
    """
    Exposure subcommands::

        * generate deterministic losses (GUL, IL or RI)
        * validate OED exposure (not yet implemented)
    """
    sub_commands = {
        'run-deterministic': RunDeterministicCmd,
        'validate': ValidateCmd
    }
