# -*- coding: utf-8 -*-

import io
import json
import os
import subprocess
import time

from argparse import RawDescriptionHelpFormatter

from pathlib2 import Path

from ..exposures.manager import OasisExposuresManager
from ..model_execution.bash import genbash
from ..model_execution.runner import run
from ..model_execution.bin import create_binary_files, prepare_model_run_directory, prepare_model_run_inputs
from ..utils.exceptions import OasisException
from ..utils.values import get_utctimestamp
from ..keys.lookup import OasisKeysLookupFactory
from .cleaners import as_path
from .base import OasisBaseCommand, InputValues


class GenerateKeysCmd(OasisBaseCommand):
    """
    Generate Oasis keys records (location records with area peril ID and
    vulnerability ID) for a model, and also writes them to file.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    Run ``oasislmf config --help`` for more information.

    Keys records returned by an Oasis keys lookup service (see the PiWind
    lookup service for reference) will be Python dicts with the following
    structure
    ::

        {
            "id": <loc. ID>,
            "peril_id": <Oasis peril type ID - oasislmf/utils/peril.py>,
            "coverage": <Oasis coverage type ID - see oasislmf/utils/coverage.py>,
            "area_peril_id": <area peril ID>,
            "vulnerability_id": <vulnerability ID>,
            "message": <lookup status message>,
            "status": <lookup status code - see oasislmf/utils/status.py>
        }

    The command can generate keys records in this format, and write them to file.

    For model loss calculations however ``ktools`` requires a keys CSV file with
    the following format
    ::

        LocID,PerilID,CoverageID,AreaPerilID,VulnerabilityID
        ..
        ..

    where the headers correspond to the relevant Oasis keys record fields.
    The command can also generate and write Oasis keys files. It will also
    optionally write an Oasis keys error file, which is a file containing
    keys records for locations with unsuccessful lookups (locations with
    a failing or non-matching lookup). It has the format
    ::

        LocID,PerilID,CoverageID,Message
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(GenerateKeysCmd, self).add_args(parser)

        parser.add_argument(
            '-k', '--keys-file-path', default=None,
            help='Keys file path',
        )
        parser.add_argument(
            '-e', '--keys-errors-file-path', default=None,
            help='Keys error file path',
        )
        parser.add_argument(
            '-d', '--keys-data-path', default=None,
            help='Keys data directory path',
        )
        parser.add_argument(
            '-v', '--model-version-file-path', default=None,
            help='Model version file path',
        )
        parser.add_argument(
            '-l', '--lookup-package-path', default=None,
            help='Keys data directory path',
        )
        parser.add_argument(
            '-f', '--keys-format', choices=['oasis', 'json'],
            help='Keys records / files output format',
        )
        parser.add_argument(
            '-x', '--model-exposures-file-path', default=None, help='Keys records file output format',
        )

    def action(self, args):
        """
        Generate and write Oasis keys (area peril ID, vulnerability ID) for a model.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)
        model_exposures_file_path = as_path(inputs.get('model_exposures_file_path', required=True, is_path=True), 'Model exposures file path')
        keys_data_path = as_path(inputs.get('keys_data_path', required=True, is_path=True), 'Keys data path')
        model_version_file_path = as_path(inputs.get('model_version_file_path', required=True, is_path=True), 'Model version file path')
        lookup_package_path = as_path(inputs.get('lookup_package_path', required=True, is_path=True), 'Lookup package path')

        keys_format = inputs.get('keys_format', default='oasis')

        start_time = time.time()
        self.logger.info('\nStarting keys files generation (@ {})'.format(get_utctimestamp()))

        self.logger.info('\nGetting model info and creating lookup service instance')
        model_info, model_klc = OasisKeysLookupFactory.create(
            model_keys_data_path=keys_data_path,
            model_version_file_path=model_version_file_path,
            lookup_package_path=lookup_package_path,
        )
        self.logger.info('\t{}, {}'.format(model_info, model_klc))

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')

        default_keys_file_name = '{}-{}-{}-keys-{}.{}'.format(model_info['supplier_id'].lower(), model_info['model_id'].lower(), model_info['model_version_id'], utcnow, 'csv' if keys_format == 'oasis' else 'json')
        default_keys_errors_file_name = '{}-{}-{}-keys-errors-{}.{}'.format(model_info['supplier_id'].lower(), model_info['model_id'].lower(), model_info['model_version_id'], utcnow, 'csv' if keys_format == 'oasis' else 'json')
           
        keys_file_path = as_path(inputs.get('keys_file_path', default=default_keys_file_name.format(utcnow), required=False, is_path=True), 'Keys file path', preexists=False)
        keys_errors_file_path = as_path(inputs.get('keys_errors_file_path', default=default_keys_errors_file_name.format(utcnow), required=False, is_path=True), 'Keys errors file path', preexists=False)

        self.logger.info('\nSaving keys records to file')
        f1, n1, f2, n2 = OasisKeysLookupFactory.save_keys(
            lookup=model_klc,
            model_exposures_file_path=model_exposures_file_path,
            keys_file_path=keys_file_path,
            keys_errors_file_path=keys_errors_file_path,
            keys_format=keys_format
        )
        self.logger.info('\n{} keys records with successful lookups saved to keys file {}'.format(n1, f1))
        self.logger.info('{} keys records with unsuccessful lookups saved to keys error file {}'.format(n2, f2))

        total_time = time.time() - start_time
        total_time_str = '{} seconds'.format(round(total_time, 3)) if total_time < 60 else '{} minutes'.format(round(total_time / 60, 3))
        self.logger.info('\nFinished keys files generation ({})'.format(total_time_str))


class GenerateOasisFilesCmd(OasisBaseCommand):
    """
    Generate Oasis files: items, coverages, GUL summary (exposure files) +
    optionally also FM files (policy TC, profile, programme, xref, summaryxref)

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(GenerateOasisFilesCmd, self).add_args(parser)

        parser.add_argument('-o', '--oasis-files-path', default=None, help='Oasis files path')
        parser.add_argument('-k', '--keys-data-path', default=None, help='Keys data path')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')
        parser.add_argument('-l', '--lookup-package-path', default=None, help='Lookup package path')
        parser.add_argument(
            '-p', '--canonical-exposures-profile-json-path', default=None,
            help='Path of the supplier canonical exposures profile JSON file'
        )
        parser.add_argument(
            '-q', '--canonical-accounts-profile-json-path', default=None,
            help='Path of the supplier canonical accounts profile JSON file'
        )
        parser.add_argument('-x', '--source-exposures-file-path', default=None, help='Source exposures file path')
        parser.add_argument('-y', '--source-accounts-file-path', default=None, help='Source accounts file path')
        parser.add_argument(
            '-a', '--source-exposures-validation-file-path', default=None,
            help='Source exposures file validation file (XSD) path'
        )
        parser.add_argument(
            '-b', '--source-accounts-validation-file-path', default=None,
            help='Source accounts file validation file (XSD) path'
        )
        parser.add_argument(
            '-c', '--source-to-canonical-exposures-transformation-file-path', default=None,
            help='Source -> canonical exposures file transformation file (XSLT) path'
        )
        parser.add_argument(
            '-d', '--source-to-canonical-accounts-transformation-file-path', default=None,
            help='Source -> canonical accounts file transformation file (XSLT) path'
        )
        parser.add_argument(
            '-e', '--canonical-exposures-validation-file-path', default=None,
            help='Canonical exposures file validation file (XSD) path'
        )
        parser.add_argument(
            '-f', '--canonical-to-model-exposures-transformation-file-path', default=None,
            help='Canonical exposures file validation file (XSD) path'
        )
        parser.add_argument('--fm', action='store_true', help='Generate FM files - False if absent')

    def action(self, args):
        """
        Generate Oasis files (items, coverages, GUL summary) for a model

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        default_oasis_files_path = os.path.join(os.getcwd(), 'runs', 'OasisFiles-{}'.format(utcnow))
        oasis_files_path = as_path(inputs.get('oasis_files_path', is_path=True, default=default_oasis_files_path), 'Oasis files path', preexists=False)
        keys_data_path = as_path(inputs.get('keys_data_path', required=True, is_path=True), 'Keys data path')
        model_version_file_path = as_path(inputs.get('model_version_file_path', required=True, is_path=True), 'Model version file path')
        lookup_package_file_path = as_path(inputs.get('lookup_package_path', required=True, is_path=True), 'Lookup package path')
        canonical_exposures_profile_json_path = as_path(
            inputs.get('canonical_exposures_profile_json_path', required=True, is_path=True),
            'Supplier canonical exposures profile JSON path'
        )
        canonical_accounts_profile_json_path = as_path(
            inputs.get('canonical_accounts_profile_json_path', required=False, is_path=True),
            'Supplier canonical accounts profile JSON path'
        )
        source_exposures_file_path = as_path(inputs.get('source_exposures_file_path', required=True, is_path=True), 'Source exposures file path')
        source_accounts_file_path = as_path(inputs.get('source_accounts_file_path', required=False, is_path=True), 'Source accounts file path')
        source_exposures_validation_file_path = as_path(
            inputs.get('source_exposures_validation_file_path', required=True, is_path=True),
            'Source exposures file validation file path'
        )
        source_accounts_validation_file_path = as_path(
            inputs.get('source_accounts_validation_file_path', required=False, is_path=True),
            'Source accounts file validation file path'
        )
        source_to_canonical_exposures_transformation_file_path = as_path(
            inputs.get('source_to_canonical_exposures_transformation_file_path', required=True, is_path=True),
            'Source to canonical exposures file transformation file path'
        )
        source_to_canonical_accounts_transformation_file_path = as_path(
            inputs.get('source_to_canonical_accounts_transformation_file_path', required=False, is_path=True),
            'Source to canonical accounts file transformation file path'
        )
        canonical_exposures_validation_file_path = as_path(
            inputs.get('canonical_exposures_validation_file_path', required=True, is_path=True),
            'Canonical exposures validation file path'
        )
        canonical_to_model_exposures_transformation_file_path = as_path(
            inputs.get('canonical_to_model_exposures_transformation_file_path', required=True, is_path=True),
            'Canonical to model exposures transformation file path'
        )

        fm = inputs.get('fm', default=False)

        start_time = time.time()
        self.logger.info('\nStarting Oasis files generation (@ {})'.format(get_utctimestamp()))

        self.logger.info('\nGenerate FM files: {}'.format(fm))

        if fm and not (canonical_accounts_profile_json_path or source_accounts_file_path):
            raise OasisException(
                'FM option indicated but missing either the canonical accounts profile JSON path or '
                'the source accounts file path'
            )

        self.logger.info('\nGetting model info and creating lookup service instance')
        model_info, model_klc = OasisKeysLookupFactory.create(
            model_keys_data_path=keys_data_path,
            model_version_file_path=model_version_file_path,
            lookup_package_path=lookup_package_file_path,
        )
        self.logger.info('\t{}, {}'.format(model_info, model_klc))

        self.logger.info('\nCreating Oasis model object')
        model = OasisExposuresManager().create_model(
            model_supplier_id=model_info['supplier_id'],
            model_id=model_info['model_id'],
            model_version_id=model_info['model_version_id'],
            resources={
                'lookup': model_klc,
                'oasis_files_path': oasis_files_path,
                'source_exposures_file_path': source_exposures_file_path,
                'source_accounts_file_path': source_accounts_file_path,
                'source_exposures_validation_file_path': source_exposures_validation_file_path,
                'source_accounts_validation_file_path': source_accounts_validation_file_path,
                'source_to_canonical_exposures_transformation_file_path': source_to_canonical_exposures_transformation_file_path,
                'source_to_canonical_accounts_transformation_file_path': source_to_canonical_accounts_transformation_file_path,
                'canonical_accounts_profile_json_path': canonical_accounts_profile_json_path,
                'canonical_exposures_profile_json_path': canonical_exposures_profile_json_path,
                'canonical_exposures_validation_file_path': canonical_exposures_validation_file_path,
                'canonical_to_model_exposures_transformation_file_path': canonical_to_model_exposures_transformation_file_path
            }
        )
        self.logger.info('\t{}'.format(model))

        self.logger.info('\nSetting up Oasis files directory for model {}'.format(model.key))
        Path(oasis_files_path).mkdir(parents=True, exist_ok=True)

        self.logger.info('\nGenerating Oasis files for model')
        oasis_files = OasisExposuresManager().start_oasis_files_pipeline(
            oasis_model=model,
            fm=fm,
            logger=self.logger
        )

        self.logger.info('\nGenerated Oasis files for model: {}'.format(oasis_files))

        total_time = time.time() - start_time
        total_time_str = '{} seconds'.format(round(total_time, 3)) if total_time < 60 else '{} minutes'.format(round(total_time / 60, 3))
        self.logger.info('\nFinished Oasis files generation ({})'.format(total_time_str))


class GenerateLossesCmd(OasisBaseCommand):
    """
    Generate losses using the installed ktools framework.

    Given Oasis files, model analysis settings JSON file, model data, and
    some other parameters. can generate losses using the installed ktools framework.

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

    By default executing the generated ktools losses script will automatically
    execute, this can be overridden by providing the ``--no-execute`` flag.
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(GenerateLossesCmd, self).add_args(parser)

        parser.add_argument('-o', '--oasis-files-path', default=None, help='Oasis files path')
        parser.add_argument('-j', '--analysis-settings-json-file-path', default=None, help='Analysis settings JSON file path')
        parser.add_argument('-m', '--model-data-path', default=None, help='Model data path')
        parser.add_argument('-r', '--model-run-dir-path', default=None, help='Model run directory path')
        parser.add_argument('-s', '--ktools-script-name', default=None, help='Ktools calc. script file path')
        parser.add_argument('-n', '--ktools-num-processes', default=-1, help='Number of ktools calculation processes to use')
        parser.add_argument('-u', '--no-execute', action='store_true', help='Whether to execute generated ktools script')

    def action(self, args):
        """
        Generate losses using the installed ktools framework.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        oasis_files_path = as_path(inputs.get('oasis_files_path', required=True, is_path=True), 'Oasis files path', preexists=True)

        model_run_dir_path = as_path(inputs.get('model_run_dir_path', required=False, is_path=True), 'Model run directory', preexists=False)

        analysis_settings_json_file_path = as_path(
            inputs.get('analysis_settings_json_file_path', required=True, is_path=True),
            'Analysis settings JSON file'
        )
        model_data_path = as_path(inputs.get('model_data_path', required=True, is_path=True), 'Model data path')

        ktools_script_name = inputs.get('ktools_script_name', default='run_ktools')
        no_execute = inputs.get('no_execute', default=False)

        start_time = time.time()
        self.logger.info('\nStarting loss generation (@ {})'.format(get_utctimestamp()))

        if not model_run_dir_path:
            utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
            model_run_dir_path = os.path.join(os.getcwd(), 'runs', 'ProgOasis-{}'.format(utcnow))
            self.logger.info('\nNo model run dir. provided - creating a timestamped run dir. in working directory as {}'.format(model_run_dir_path))
            Path(model_run_dir_path).mkdir(parents=True, exist_ok=True)
        else:
            if not os.path.exists(model_run_dir_path):
                Path(model_run_dir_path).mkdir(parents=True, exist_ok=True)

        self.logger.info(
            '\nPreparing model run directory {} - copying Oasis files, analysis settings JSON file and linking model data'.format(model_run_dir_path)
        )
        prepare_model_run_directory(
            model_run_dir_path,
            oasis_files_path,
            analysis_settings_json_file_path,
            model_data_path
        )

        self.logger.info('\nConverting Oasis files to ktools binary files')
        oasis_files_path = os.path.join(model_run_dir_path, 'input', 'csv')
        binary_files_path = os.path.join(model_run_dir_path, 'input')
        create_binary_files(oasis_files_path, binary_files_path)

        analysis_settings_json_file_path = os.path.join(model_run_dir_path, 'analysis_settings.json')
        try:
            self.logger.info('\nReading analysis settings JSON file')
            with io.open(analysis_settings_json_file_path, 'r', encoding='utf-8') as f:
                analysis_settings = json.load(f)
                if 'analysis_settings' in analysis_settings:
                    analysis_settings = analysis_settings['analysis_settings']
        except (IOError, TypeError, ValueError):
            raise OasisException('Invalid analysis settings JSON file or file path: {}.'.format(analysis_settings_json_file_path))

        self.logger.info('\nLoaded analysis settings JSON: {}'.format(analysis_settings))

        self.logger.info('\nPreparing model run inputs')
        prepare_model_run_inputs(analysis_settings, model_run_dir_path)

        script_path = os.path.join(model_run_dir_path, '{}.sh'.format(ktools_script_name))
        if no_execute:
            self.logger.info('\nGenerating ktools losses script')
            genbash(
                args.ktools_num_processes,
                analysis_settings,
                filename=script_path,
            )
            self.logger.info('\nMaking ktools losses script executable')
            subprocess.check_call("chmod +x {}".format(script_path), stderr=subprocess.STDOUT, shell=True)
        else:
            os.chdir(model_run_dir_path)
            run(analysis_settings, args.ktools_num_processes, filename=script_path)

        self.logger.info('\nLoss outputs generated in {}'.format(os.path.join(model_run_dir_path, 'output')))

        total_time = time.time() - start_time
        total_time_str = '{} seconds'.format(round(total_time, 3)) if total_time < 60 else '{} minutes'.format(round(total_time / 60, 3))
        self.logger.info('\nFinished loss generation ({})'.format(total_time_str))


class RunCmd(OasisBaseCommand):
    """
    Run models end to end.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    def add_args(self, parser):
        """
        Run models end to end.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(RunCmd, self).add_args(parser)

        parser.add_argument('-k', '--keys-data-path', default=None, help='Oasis files path')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')
        parser.add_argument('-l', '--lookup-package-file-path', default=None, help='Keys data path')
        parser.add_argument(
            '-p', '--canonical-exposures-profile-json-path', default=None,
            help='Supplier canonical exposures profile JSON path'
        )
        parser.add_argument(
            '-q', '--canonical-accounts-profile-json-path', default=None,
            help='Supplier canonical accounts profile JSON path'
        )
        
        parser.add_argument('-x', '--source-exposures-file-path', default=None, help='Source exposures file path')
        parser.add_argument('-y', '--source-accounts-file-path', default=None, help='Source accounts file path')
        parser.add_argument(
            '-a', '--source-exposures-validation-file-path', default=None,
            help='Source exposures file validation file (XSD) path'
        )
        parser.add_argument(
            '-b', '--source-accounts-validation-file-path', default=None,
            help='Source accounts file validation file (XSD) path'
        )
        parser.add_argument(
            '-c', '--source-to-canonical-exposures-transformation-file-path', default=None,
            help='Source -> canonical exposures file transformation file (XSLT) path'
        )
        parser.add_argument(
            '-d', '--source-to-canonical-accounts-transformation-file-path', default=None,
            help='Source -> canonical accounts file transformation file (XSLT) path'
        )
        parser.add_argument(
            '-e', '--canonical-exposures-validation-file-path', default=None,
            help='Canonical exposures validation file (XSD) path'
        )
        parser.add_argument(
            '-f', '--canonical-to-model-exposures-transformation-file-path', default=None,
            help='Canonical -> model exposures file validation file (XSD) path'
        )
        parser.add_argument('--fm', action='store_true', help='Generate FM files - False if absent')

        parser.add_argument(
            '-j', '--analysis-settings-json-file-path', default=None,
            help='Model analysis settings JSON file path'
        )
        parser.add_argument('-m', '--model-data-path', default=None, help='Model data path')
        parser.add_argument('-r', '--model-run-dir-path', default=None, help='Model run directory path')
        parser.add_argument(
            '-s', '--ktools-script-name', default=None,
            help='Name of the ktools output script (should not contain any filetype extension)'
        )
        parser.add_argument('-n', '--ktools-num-processes', default=2, help='Number of ktools calculation processes to use')

    def action(self, args):
        """
        Generate Oasis files (items, coverages, GUL summary) for a model

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        model_run_dir_path = as_path(inputs.get('model_run_dir_path', required=False), 'Model run path', preexists=False)

        start_time = time.time()
        self.logger.info('\nStarting model run (@ {})'.format(get_utctimestamp()))

        if not model_run_dir_path:
            utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
            model_run_dir_path = os.path.join(os.getcwd(), 'runs', 'ProgOasis-{}'.format(utcnow))
            self.logger.info('\nNo model run dir. provided - creating a timestamped run dir. in working directory as {}'.format(model_run_dir_path))
            Path(model_run_dir_path).mkdir(parents=True, exist_ok=True)
        else:
            if not os.path.exists(model_run_dir_path):
                Path(model_run_dir_path).mkdir(parents=True, exist_ok=True)

        args.model_run_dir_path = model_run_dir_path

        args.oasis_files_path = os.path.join(model_run_dir_path, 'input', 'csv')
        self.logger.info('\nCreating Oasis files directory {}'.format(args.oasis_files_path))
        Path(args.oasis_files_path).mkdir(parents=True, exist_ok=True)

        gen_oasis_files_cmd = GenerateOasisFilesCmd()
        gen_oasis_files_cmd._logger = self.logger
        gen_oasis_files_cmd.action(args)

        gen_losses_cmd = GenerateLossesCmd()
        gen_losses_cmd._logger = self.logger
        gen_losses_cmd.action(args)

        total_time = time.time() - start_time
        total_time_str = '{} seconds'.format(round(total_time, 3)) if total_time < 60 else '{} minutes'.format(round(total_time / 60, 3))
        self.logger.info('\nFinished model run ({})'.format(total_time_str))


class ModelsCmd(OasisBaseCommand):
    sub_commands = {
        'generate-keys': GenerateKeysCmd,
        'generate-oasis-files': GenerateOasisFilesCmd,
        'generate-losses': GenerateLossesCmd,
        'run': RunCmd,
    }
