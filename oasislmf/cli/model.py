# -*- coding: utf-8 -*-

from __future__ import print_function

__all__ = [
    'GenerateKeysCmd',
    'GenerateLossesCmd',
    'GenerateOasisFilesCmd',
    'GeneratePerilAreasRtreeFileIndexCmd',
    'ModelsCmd',
    'RunCmd'
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

import pandas as pd

from pathlib2 import Path
from six import u as _unicode
from tabulate import tabulate

from ..manager import OasisManager as om

from ..model_preparation.lookup import OasisLookupFactory as olf
from ..utils.exceptions import OasisException
from ..utils.oed_profiles import (
    get_default_exposure_profile,
    get_default_accounts_profile,
    get_default_fm_aggregation_profile,
)
from ..utils.data import get_json
from ..utils.path import (
    as_path,
    setcwd,
)
from ..utils.peril import PerilAreasIndex
from ..utils.values import get_utctimestamp

from .base import OasisBaseCommand, InputValues


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

        parser.add_argument(
            '-c', '--lookup-config-file-path', default=None,
            help='Lookup config file path',
        )
        parser.add_argument(
            '-d', '--keys-data-path', default=None,
            help='Keys data path'
        )
        parser.add_argument(
            '-f', '--index-file-path', default=None,
            help='Index file path (no file extension required)',
        )

    def action(self, args):
        """
        Generates and writes an Rtree file index of peril area IDs (area peril IDs)
        and area polygon bounds from a peril areas (area peril) file.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        config_fp = as_path(inputs.get('lookup_config_file_path', required=True, is_path=True), 'Built-in lookup config file path', preexists=True)

        keys_data_fp = as_path(inputs.get('keys_data_path', required=True, is_path=True), 'Keys config file path', preexists=True)

        index_fp = as_path(inputs.get('index_file_path', required=True, is_path=True), 'Index output file path', preexists=False)

        om().generate_peril_areas_rtree_file_index(
            keys_data_fp,
            index_fp,
            lookup_config_fp=config_fp
        )


class GenerateKeysCmd(OasisBaseCommand):
    """
    Generates keys from a model lookup, and write Oasis keys and keys error files.

    The model lookup, which is normally independently implemented by the model
    supplier, should generate keys as dicts with the following format
    ::

        {
            "id": <loc. ID>,
            "peril_id": <OED sub-peril ID>,
            "coverage_type": <OED coverage type ID>,
            "area_peril_id": <area peril ID>,
            "vulnerability_id": <vulnerability ID>,
            "message": <loc. lookup status message>,
            "status": <loc. lookup status flag indicating success, failure or no-match>
        }

    The keys generation command can generate these dicts, and write them to
    file. It can also be used to write these to an Oasis keys file (which is a
    requirement for model execution), which has the following format.::

        LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID
        ..
        ..
    This file only lists the locations for which there has been a successful
    lookup. The keys errors file lists all the locations with failing or
    non-matching lookups and has the following format::

        LocID,PerilID,CoverageTypeID,Message
        ..
        ..
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-x', '--source-exposure-file-path', default=None, help='Source exposure file path')
        parser.add_argument('-k', '--keys-file-path', default=None, help='Keys file path')
        parser.add_argument('-e', '--keys-errors-file-path', default=None, help='Keys errors file path')
        parser.add_argument('-g', '--lookup-config-file-path', default=None, help='Lookup config JSON file path')
        parser.add_argument('-d', '--keys-data-path', default=None, help='Model lookup/keys data path')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')
        parser.add_argument('-l', '--lookup-package-path', default=None, help='Model lookup package path')
        parser.add_argument('-f', '--keys-format', choices=['oasis', 'json'], help='Keys files output format')

    def action(self, args):
        """
        Generates keys from a model lookup, and write Oasis keys and keys error files.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        config_fp = as_path(inputs.get('lookup_config_file_path', required=False, is_path=True), 'Lookup config JSON file path',)

        keys_data_fp = as_path(inputs.get('keys_data_path', required=False, is_path=True), 'Keys data path', preexists=False)
        model_version_fp = as_path(inputs.get('model_version_file_path', required=False, is_path=True), 'Model version file path', preexists=False)
        lookup_package_fp = as_path(inputs.get('lookup_package_path', required=False, is_path=True), 'Lookup package path', preexists=False)

        if not (config_fp or (keys_data_fp and model_version_fp and lookup_package_fp)):
            raise OasisException(
                'No lookup assets provided to generate the mandatory keys '
                'file - for built-in lookups the lookup config. JSON file '
                'path must be provided, or for custom lookups the keys data '
                'path + model version file path + lookup package path must be '
                'provided'
            )

        exposure_fp = as_path(inputs.get('source_exposure_file_path', required=True, is_path=True), 'Source exposure file path')

        keys_format = inputs.get('keys_format', default='oasis')

        f1, n1, f2, n2 = om().generate_keys(
            exposure_fp,
            lookup_config_fp=config_fp,
            keys_data_fp=keys_data_fp,
            model_version_fp=model_version_fp,
            lookup_package_fp=lookup_package_fp,
            keys_format=keys_format
        )


class GenerateOasisFilesCmd(OasisBaseCommand):
    """
    Generates the standard Oasis GUL input files + optionally the IL/FM input
    files and the RI input files.
    """
    formatter_class = RawDescriptionHelpFormatter
    static_data_fp = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, '_data'))

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-o', '--oasis-files-path', default=None, help='Path to the directory in which to generate the Oasis files')
        parser.add_argument('-c', '--lookup-config-file-path', default=None, help='Lookup config JSON file path')
        parser.add_argument('-k', '--keys-data-path', default=None, help='Model lookup/keys data path')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')
        parser.add_argument('-l', '--lookup-package-path', default=None, help='Lookup package path')
        parser.add_argument(
            '-p', '--source-exposure-profile-path', default=None,
            help='Source (OED) exposure profile path'
        )
        parser.add_argument(
            '-q', '--source-accounts-profile-path', default=None,
            help='Source (OED) accounts profile path'
        )
        parser.add_argument('-x', '--source-exposure-file-path', default=None, help='Source exposure file path')
        parser.add_argument('-y', '--source-accounts-file-path', default=None, help='Source accounts file path')

        parser.add_argument(
            '-g', '--fm-aggregation-profile-path', default=None, help='FM (OED) aggregation profile path'

        )
        parser.add_argument(
            '-i', '--ri-info-file-path', default=None,
            help='Reinsurance info. file path'
        )
        parser.add_argument(
            '-s', '--ri-scope-file-path', default=None,
            help='Reinsurance scope file path'
        )

    def action(self, args):
        """
        Generates the standard Oasis GUL input files + optionally the IL/FM input
        files and the RI input files.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        self.logger.info('\nProcessing arguments')
        inputs = InputValues(args)

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        default_oasis_fp = os.path.join(os.getcwd(), 'runs', 'OasisFiles-{}'.format(utcnow))

        call_dir = os.getcwd()

        oasis_fp = as_path(inputs.get('oasis_files_path', is_path=True, default=default_oasis_fp), 'Oasis files path', preexists=False)

        lookup_config_fp = as_path(inputs.get('lookup_config_file_path', required=False, is_path=True), 'Lookup config JSON file path', preexists=False)

        keys_data_fp = as_path(inputs.get('keys_data_path', required=False, is_path=True), 'Keys data path', preexists=False)
        model_version_fp = as_path(inputs.get('model_version_file_path', required=False, is_path=True), 'Model version file path', preexists=False)
        lookup_package_fp = as_path(inputs.get('lookup_package_path', required=False, is_path=True), 'Lookup package path', preexists=False)

        if not (lookup_config_fp or (keys_data_fp and model_version_fp and lookup_package_fp)):
            raise OasisException(
                'No lookup assets provided to generate the mandatory keys '
                'file - for built-in lookups the lookup config. JSON file '
                'path must be provided, or for custom lookups the keys data '
                'path + model version file path + lookup package path must be '
                'provided'
            )

        exposure_fp = as_path(
            inputs.get('source_exposure_file_path', required=True, is_path=True), 'Source exposure file path'
        )

        exposure_profile_fp = as_path(
            inputs.get('source_exposure_profile_path', default=os.path.join(self.static_data_fp, 'oed-loc-profile.json'), required=False, is_path=True),
            'Source OED exposure profile path'
        )

        accounts_fp = as_path(
            inputs.get('source_accounts_file_path', required=False, is_path=True), 'Source OED accounts file path'
        )
        accounts_profile_fp = as_path(
            inputs.get('source_accounts_profile_path', default=os.path.join(self.static_data_fp, 'oed-acc-profile.json'), required=False, is_path=True),
            'Source OED accounts profile path'
        )

        aggregation_profile_fp = as_path(
            inputs.get('fm_aggregation_profile_path', default=os.path.join(self.static_data_fp, 'fm-oed-agg-profile.json'), required=False, is_path=True),
            'FM OED aggregation profile path'
        )

        ri_info_fp = as_path(
            inputs.get('ri_info_file_path', required=False, is_path=True),
            'Reinsurance info. file path'
        )
        ri_scope_fp = as_path(
            inputs.get('ri_scope_file_path', required=False, is_path=True),
            'Reinsurance scope file path'
        )

        il = True if accounts_fp else False

        required_ri_paths = [ri_info_fp, ri_scope_fp]

        ri = all(required_ri_paths) and il

        self.logger.info('\nGenerating Oasis files (GUL=True, IL={}, RI={}'.format(il, ri))
        oasis_files = om().generate_oasis_files(
            oasis_fp,
            exposure_fp,
            exposure_profile_fp=exposure_profile_fp,
            lookup_config_fp=lookup_config_fp,
            keys_data_fp=keys_data_fp,
            model_version_fp=model_version_fp,
            lookup_package_fp=lookup_package_fp,
            accounts_fp=accounts_fp,
            accounts_profile_fp=accounts_profile_fp,
            aggregation_profile_fp=aggregation_profile_fp,
            ri_info_fp=ri_info_fp,
            ri_scope_fp=ri_scope_fp
        )

        self.logger.info('\nOasis files generated: {}'.format(oasis_files))


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

        parser.add_argument('-o', '--oasis-files-path', default=None, help='Path to pre-existing direct Oasis files (GUL + FM input files)')
        parser.add_argument('-a', '--analysis-settings-file-path', default=None, help='Analysis settings file path')
        parser.add_argument('-d', '--model-data-path', default=None, help='Model data path')
        parser.add_argument('-r', '--model-run-dir', default=None, help='Model run directory path')
        parser.add_argument('-p', '--model-package-path', default=None, help='Path containing model specific package')
        parser.add_argument('-n', '--ktools-num-processes', default=None, help='Number of ktools calculation processes to use', type=int)
        parser.add_argument('-m', '--ktools-mem-limit', default=False, help='Force exec failure if Ktools hits memory the system  memory limit', action='store_true')
        parser.add_argument('-f', '--ktools-fifo-relative', default=False, help='Create ktools fifo queues under the ./fifo dir', action='store_true')
        parser.add_argument('-u', '--ktools-alloc-rule', default=None, help='Override the allocation used in fmcalc', type=int)

    def action(self, args):
        """
        Generates losses using the installed ktools framework given Oasis files, 
        model analysis settings JSON file, model data and model package data.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        self.logger.info('\nProcessing arguments')
        inputs = InputValues(args)

        call_dir = os.getcwd()

        oasis_fp = as_path(
            inputs.get('oasis_files_path', required=True, is_path=True),
            'Path to direct Oasis files (GUL + optionally FM and RI input files)', preexists=True
        )

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        default_model_run_fp = os.path.join(os.getcwd(), 'runs', 'ProgOasis-{}'.format(utcnow))

        model_run_fp = as_path(inputs.get('model_run_dir', is_path=True, default=default_model_run_fp), 'Model run directory', preexists=False)

        analysis_settings_fp = as_path(
            inputs.get('analysis_settings_file_path', required=True, is_path=True),
            'Model analysis settings file path'
        )

        model_data_fp = as_path(inputs.get('model_data_path', required=True, is_path=True), 'Model data path')

        model_package_fp = as_path(inputs.get('model_package_path', required=False, is_path=True), 'Model package path')

        ktools_num_processes = inputs.get('ktools_num_processes', default=2, required=False)

        ktools_mem_limit = inputs.get('ktools_mem_limit', default=False, required=False)

        ktools_fifo_relative = inputs.get('ktools_fifo_relative', default=False, required=False)

        ktools_alloc_rule = inputs.get('ktools_alloc_rule', default=2, required=False)

        il = all(p in os.listdir(oasis_fp) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])
        ri = False
        if os.path.basename(oasis_fp) == 'input':
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(oasis_fp))
        elif os.path.basename(oasis_fp) == 'csv':
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(os.path.dirname(oasis_fp)))

        self.logger.info('\nGenerating losses (GUL=True, IL={}, RI={})'.format(il, ri))
        om().generate_losses(
            model_run_fp,
            oasis_fp,
            analysis_settings_fp,
            model_data_fp,
            model_package_fp=model_package_fp,
            ktools_num_processes=ktools_num_processes,
            ktools_mem_limit=ktools_mem_limit,
            ktools_fifo_relative=ktools_fifo_relative,
            ktools_alloc_rule=ktools_alloc_rule
        )

        self.logger.info('\nLosses generated')


class GenerateDeterministicLossesCmd(OasisBaseCommand):
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

        il = all(p in os.listdir(oasis_fp) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])
        ri = False
        if os.path.basename(oasis_fp) == 'input':
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(oasis_fp))
        elif os.path.basename(oasis_fp) == 'csv':
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(os.path.dirname(oasis_fp)))

        self.logger.info('\nGenerating deterministic losses (GUL=True, IL={}, RI={})'.format(il, ri))
        losses_df = om().generate_deterministic_losses(input_dir, output_dir, loss_percentage_of_tiv=loss_factor, net=net_losses, print_losses=False)
        losses_df['event_id'] = losses_df['event_id'].astype(object)
        losses_df['output_id'] = losses_df['output_id'].astype(object)

        self.logger.info('\nLosses generated'.format(il, ri))
        print(tabulate(losses_df, headers='keys', tablefmt='psql', floatfmt=".2f"))


class RunCmd(OasisBaseCommand):
    """
    Run models end to end.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    formatter_class = RawDescriptionHelpFormatter
    static_data_fp = os.path.join(os.path.dirname(__file__), os.path.pardir, '_data')

    def add_args(self, parser):
        """
        Run models end to end.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-k', '--keys-data-path', default=None, help='Model lookup/keys data path')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')

        parser.add_argument('-l', '--lookup-package-path', default=None, help='Model lookup package path')
        parser.add_argument('-c', '--lookup-config-file-path', default=None, help='Built-in lookup config JSON file path')

        parser.add_argument(
            '-p', '--source-exposure-profile-path', default=None,
            help='Source OED exposure profile path'
        )
        parser.add_argument(
            '-q', '--source-accounts-profile-path', default=None,
            help='Source OED accounts profile path'
        )
        
        parser.add_argument('-x', '--source-exposure-file-path', default=None, help='Source exposure file path')
        parser.add_argument('-y', '--source-accounts-file-path', default=None, help='Source accounts file path')

        parser.add_argument(
            '-g', '--fm-aggregation-profile-path', default=None,
            help='FM OED aggregation profile path'
        )
        parser.add_argument(
            '-i', '--ri-info-file-path', default=None,
            help='Reinsurance info. file path'
        )
        parser.add_argument(
            '-s', '--ri-scope-file-path', default=None,
            help='Reinsurance scope file path'
        )
        parser.add_argument(
            '-a', '--analysis-settings-file-path', default=None,
            help='Model analysis settings file path'
        )
        parser.add_argument('-d', '--model-data-path', default=None, help='Model data path')
        parser.add_argument('-r', '--model-run-dir', default=None, help='Model run directory path')
        parser.add_argument('-n', '--ktools-num-processes', default=2, help='Number of ktools calculation processes to use')
        parser.add_argument('-m', '--ktools-mem-limit', default=False, help='Force exec failure if Ktools hits memory the system  memory limit', action='store_true')
        parser.add_argument('-f', '--ktools-fifo-relative', default=False, help='Create ktools fifo queues under the ./fifo dir', action='store_true')
        parser.add_argument('-u', '--ktools-alloc-rule', default=2, help='Override the allocation used in fmcalc', type=int)

    def action(self, args):
        """
        Generate Oasis files (items, coverages, GUL summary) for a model

        :param args: The arguments from the command line
        :type args: Namespace
        """
        self.logger.info('\nProcessing arguments')
        inputs = InputValues(args)

        call_dir = os.getcwd()

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        default_model_run_fp = os.path.join(os.getcwd(), 'runs', 'ProgOasis-{}'.format(utcnow))

        model_run_fp = as_path(inputs.get('model_run_dir', is_path=True, default=default_model_run_fp), 'Model run directory', preexists=False)

        args.model_run_fp = model_run_fp

        accounts_fp = as_path(
            inputs.get('source_accounts_file_path', required=False, is_path=True), 'Source OED accounts file path'
        )

        ri_info_fp = as_path(
            inputs.get('ri_info_file_path', required=False, is_path=True),
            'Reinsurance info. file path'
        )
        ri_scope_fp = as_path(
            inputs.get('ri_scope_file_path', required=False, is_path=True),
            'Reinsurance scope file path'
        )

        il = True if accounts_fp else False

        required_ri_paths = [ri_info_fp, ri_scope_fp]

        ri = all(required_ri_paths) and il

        if any(required_ri_paths) and not ri:
            raise OasisException(
                'RI option indicated by provision of some RI related assets, but other assets are missing. '
                'To generate RI inputs you need to provide all of the assets required to generate direct '
                'Oasis files (GUL + FM input files) plus all of the following assets: '
                '    reinsurance info. file path, '
                '    reinsurance scope file path.'
            )

        args.oasis_files_path = os.path.join(model_run_fp, 'input', 'csv') if not ri else os.path.join(model_run_fp, 'input')

        self.logger.info('\nGenerating Oasis files (GUL=True, IL={}, RI={})'.format(il, ri))
        gen_oasis_files_cmd = GenerateOasisFilesCmd()
        gen_oasis_files_cmd.action(args)

        self.logger.info('\nGenerating losses (GUL=True, IL={}, RI={})'.format(il, ri))
        gen_losses_cmd = GenerateLossesCmd()
        gen_losses_cmd.action(args)


class ModelsCmd(OasisBaseCommand):
    """
    Model subcommands::

        * generating an Rtree file index for the area peril lookup component of the built-in lookup framework
        * generating keys files from model lookups
        * generating Oasis input CSV files (GUL + optionally FM and RI)
        * generating losses from a preexisting set of Oasis input CSV files
        * generating deterministic losses (no model)
        * running a model end-to-end
    """
    sub_commands = {
        'generate-peril-areas-rtree-file-index': GeneratePerilAreasRtreeFileIndexCmd,
        'generate-keys': GenerateKeysCmd,
        'generate-oasis-files': GenerateOasisFilesCmd,
        'generate-losses': GenerateLossesCmd,
        'generate-deterministic-losses': GenerateDeterministicLossesCmd,
        'run': RunCmd,
    }
