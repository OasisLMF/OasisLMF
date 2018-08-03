# -*- coding: utf-8 -*-

import io
import json
import os
import subprocess

from argparse import RawDescriptionHelpFormatter

from pathlib2 import Path

from ..exposures.csv_trans import Translator
from ..exposures.manager import OasisExposuresManager

from ..model_execution.bash import genbash
from ..model_execution.runner import run
from ..model_execution.bin import create_binary_files, prepare_model_run_directory, prepare_model_run_inputs

from ..utils.exceptions import OasisException
from ..utils.peril import PerilAreasIndex
from ..utils.values import get_utctimestamp

from ..keys.lookup import OasisLookupFactory

from .cleaners import as_path

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

        lookup_config_fp = as_path(inputs.get('lookup_config_file_path', required=True, is_path=True), 'Lookup config file path', preexists=True)

        keys_data_path = as_path(inputs.get('keys_data_path', required=True, is_path=True), 'Keys config file path', preexists=True)

        index_fp = as_path(inputs.get('index_file_path', required=True, is_path=True, default=None), 'Index output file path', preexists=False)

        with io.open(lookup_config_fp, 'r', encoding='utf-8') as f:
            config = json.load(f)

        peril_config = config.get('peril')

        if not peril_config:
            raise OasisException(
                'The lookup config must contain a peril-related subdictionary with a key named '
                '`peril` defining area-peril-related model information'
            )

        areas_fp = peril_config.get('file_path')

        if not areas_fp:
            raise OasisException(
                'The lookup peril config must define the path of a peril areas '
                '(or area peril) file with the key name `file_path`'
            )

        if areas_fp.startswith('%%KEYS_DATA_PATH%%'):
            areas_fp = areas_fp.replace('%%KEYS_DATA_PATH%%', keys_data_path)

        areas_fp = as_path(areas_fp, 'areas_fp')

        src_type = str.lower(str(peril_config.get('file_type')) or '') or 'csv'

        peril_id_col = str.lower(str(peril_config.get('peril_id_col')) or '') or 'peril_id'

        coverage_config = config.get('coverage')

        if not coverage_config:
            raise OasisException(
                'The lookup config must contain a coverage-related subdictionary with a key named '
                '`coverage` defining coverage related model information'
            )

        coverage_type_col = str.lower(str(coverage_config.get('coverage_type_col')) or '') or 'coverage_type'

        peril_area_id_col = str.lower(str(peril_config.get('peril_area_id_col')) or '') or 'area_peril_id'

        area_poly_coords_cols = peril_config.get('area_poly_coords_cols')

        if not area_poly_coords_cols:
            raise OasisException(
                'The lookup peril config must define the column names of '
                'the coordinates used to define areas in the peril areas '
                '(area peril) file using the key `area_poly_coords_cols`'
            )

        non_na_cols = (
            tuple(col.lower() for col in peril_config['non_na_cols']) if peril_config.get('non_na_cols')
            else tuple(col.lower() for col in [peril_area_id_col] + area_poly_coords_cols.values())
        )

        col_dtypes = peril_config.get('col_dtypes') or {peril_area_id_col: int}

        sort_col = peril_config.get('sort_col') or peril_area_id_col

        area_poly_coords_seq_start_idx = peril_config.get('area_poly_coords_seq_start_idx') or 1

        area_reg_poly_radius = peril_config.get('area_reg_poly_radius') or 0.00166

        index_props = peril_config.get('rtree_index')
        index_props.pop('filename')

        self.logger.info(
            '\nGenerating Rtree file index {}.{{idx,dat}} from peril areas (area peril) '
            'file {}'
            .format(os.path.join(index_fp), areas_fp)
        )

        index_fp = PerilAreasIndex.create_from_peril_areas_file(
            src_fp=areas_fp,
            src_type=src_type,
            peril_id_col=peril_id_col,
            coverage_type_col=coverage_type_col,
            peril_area_id_col=peril_area_id_col,
            non_na_cols=non_na_cols,
            col_dtypes=col_dtypes,
            sort_col=sort_col,
            area_poly_coords_cols=area_poly_coords_cols,
            area_poly_coords_seq_start_idx=area_poly_coords_seq_start_idx,
            area_reg_poly_radius=area_reg_poly_radius,
            index_fp=index_fp,
            index_props=index_props
        )

        self.logger.info('\nSuccessfully generated index files {}.{{idx.dat}}'.format(index_fp))


class TransformSourceToCanonicalFileCmd(OasisBaseCommand):
    """
    Transform a supplier-specific source exposures/accounts file to a canonical
    Oasis exposures/accounts file using an XSD validation file and an XSLT
    transformation file.

    Calling syntax is::

        oasislmf model transform-source-to-canonical
            -s <source file path>
            -y <type of source file - exposures or accounts>
            -v <validation file>
            -x <transformation file>
            -o <output file path>
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
            '-s', '--source-file-path', default=None,
            help='Source file path',
        )
        parser.add_argument(
            '-y', '--source-file-type', default='exposures',
            help='Type of source file - exposures or accounts',
        )
        parser.add_argument(
            '-v', '--xsd-validation-file-path', default=None, required=False,
            help='XSD validation file path (optional argument)',
        )
        parser.add_argument(
            '-x', '--xslt-transformation-file-path', default=None,
            help='XSLT transformation file path',
        )
        parser.add_argument(
            '-o', '--output-file-path', default=None,
            help='Output file path',
        )

    def action(self, args):
        """
        Transform a source exposures/accounts file to a canonical Oasis file.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        source_file_path = as_path(inputs.get('source_file_path', required=True, is_path=True), 'Source file path', preexists=True)
        source_file_type = inputs.get('source_file_type', default='exposures')

        _sft = 'exp' if source_file_type == 'exposures' else 'acc'
        _utc = get_utctimestamp(fmt='%Y%m%d%H%M%S')

        xsd_validation_file_path = as_path(inputs.get('xsd_validation_file_path', required=False, is_path=True), 'XSD validation file path', preexists=False)
        xslt_transformation_file_path = as_path(inputs.get('xslt_transformation_file_path', required=True, is_path=True), 'XSLT transformation file path', preexists=True)
        output_file_path = as_path(inputs.get('output_file_path', required=False, is_path=True, default='can{}-{}.csv'.format(_sft, _utc)), 'Output file path', preexists=False)

        self.logger.info('\nGenerating a canonical {} file {} from source {} file {}'.format(_sft, output_file_path, _sft, source_file_path))

        translator = Translator(source_file_path, output_file_path, xslt_transformation_file_path, xsd_validation_file_path, append_row_nums=True)
        translator()

        self.logger.info('\nOutput file {} successfully generated'.format(output_file_path))


class TransformCanonicalToModelFileCmd(OasisBaseCommand):
    """
    Transform a canonical Oasis exposures file to a model Oasis exposures file
    using an XSD validation file and an XSLT transformation file. A model exposures
    file is a simplified version of the canonical exposures file and provides the
    input to an Oasis keys server for a model and its keys lookup class.

    Calling syntax is::

        oasislmf model transform-canonical-to-model
            -c <canonical exposures file path>
            -v <validation file>
            -x <transformation file>
            -o <output file path>
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
            '-c', '--canonical-exposures-file-path', default=None,
            help='Canonical file path',
        )
        parser.add_argument(
            '-v', '--xsd-validation-file-path', default=None, required=False,
            help='XSD validation file path (optional argument)',
        )
        parser.add_argument(
            '-x', '--xslt-transformation-file-path', default=None,
            help='XSLT transformation file path',
        )
        parser.add_argument(
            '-o', '--output-file-path', default=None,
            help='Output file path',
        )

    def action(self, args):
        """
        Transform a source exposures/accounts file to a canonical Oasis file.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        canonical_exposures_file_path = as_path(inputs.get('canonical_exposures_file_path', required=True, is_path=True), 'Canonical exposures file path', preexists=True)

        _utc = get_utctimestamp(fmt='%Y%m%d%H%M%S')

        xsd_validation_file_path = as_path(inputs.get('xsd_validation_file_path', required=False, is_path=True), 'XSD validation file path', preexists=False)
        xslt_transformation_file_path = as_path(inputs.get('xslt_transformation_file_path', required=True, is_path=True), 'XSLT transformation file path', preexists=True)
        output_file_path = as_path(inputs.get('output_file_path', required=False, is_path=True, default='modexp-{}.csv'.format(_utc)), 'Output file path', preexists=False)

        self.logger.info('\nGenerating a model exposures file {} from canonical exposures file {}'.format(output_file_path, canonical_exposures_file_path))

        translator = Translator(canonical_exposures_file_path, output_file_path, xslt_transformation_file_path, xsd_validation_file_path ,append_row_nums=True)
        translator()

        self.logger.info('\nOutput file {} successfully generated'.format(output_file_path))


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
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-k', '--keys-file-path', default=None, help='Keys file path')
        parser.add_argument('-e', '--keys-errors-file-path', default=None, help='Keys errors file path')
        parser.add_argument('-g', '--lookup-config-file-path', default=None, help='Lookup config JSON file path')
        parser.add_argument('-d', '--keys-data-path', default=None, help='Keys data directory path')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')
        parser.add_argument('-l', '--lookup-package-path', default=None, help='Keys data directory path')
        parser.add_argument('-f', '--keys-format', choices=['oasis', 'json'], help='Keys records / files output format')
        parser.add_argument('-x', '--model-exposures-file-path', default=None, help='Keys records file output format')

    def action(self, args):
        """
        Generate and write Oasis keys (area peril ID, vulnerability ID) for a model.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        lookup_config_fp = as_path(inputs.get('lookup_config_file_path', required=False, is_path=True), 'Lookup config JSON file path',)

        keys_data_path = as_path(inputs.get('keys_data_path', required=False, is_path=True), 'Keys data path', preexists=False)
        model_version_file_path = as_path(inputs.get('model_version_file_path', required=False, is_path=True), 'Model version file path', preexists=False)
        lookup_package_path = as_path(inputs.get('lookup_package_path', required=False, is_path=True), 'Lookup package path', preexists=False)

        if not (lookup_config_fp or (keys_data_path and model_version_file_path and lookup_package_path)):
            raise OasisException('Either the lookup config JSON file path or the keys data path + model version file path + lookup package path must be provided')

        model_exposures_file_path = as_path(inputs.get('model_exposures_file_path', required=True, is_path=True), 'Model exposures')

        keys_format = inputs.get('keys_format', default='oasis')

        self.logger.info('\nGetting model info and lookup')
        model_info, lookup = OasisLookupFactory.create(
            lookup_config_fp=lookup_config_fp,
            model_keys_data_path=keys_data_path,
            model_version_file_path=model_version_file_path,
            lookup_package_path=lookup_package_path
        )
        self.logger.info('\t{}, {}'.format(model_info, lookup))

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')

        default_keys_file_name = '{}-{}-{}-keys-{}.{}'.format(model_info['supplier_id'].lower(), model_info['model_id'].lower(), model_info['model_version'], utcnow, 'csv' if keys_format == 'oasis' else 'json')
        default_keys_errors_file_name = '{}-{}-{}-keys-errors-{}.{}'.format(model_info['supplier_id'].lower(), model_info['model_id'].lower(), model_info['model_version'], utcnow, 'csv' if keys_format == 'oasis' else 'json')

        keys_file_path = as_path(inputs.get('keys_file_path', default=default_keys_file_name.format(utcnow), required=False, is_path=True), 'Keys file path', preexists=False)
        keys_errors_file_path = as_path(inputs.get('keys_errors_file_path', default=default_keys_errors_file_name.format(utcnow), required=False, is_path=True), 'Keys errors file path', preexists=False)

        self.logger.info('\nSaving keys records to file')
        f1, n1, f2, n2 = OasisLookupFactory.save_results(
            lookup,
            keys_file_path,
            errors_fp=keys_errors_file_path,
            model_exposures_fp=model_exposures_file_path,
            format=keys_format
        )
        self.logger.info('\n{} successful results saved to keys file {}'.format(n1, f1))
        self.logger.info('\n{} unsuccessful results saved to keys errors file {}'.format(n2, f2))


class GenerateOasisFilesCmd(OasisBaseCommand):
    """
    Generate Oasis files (items, coverages, GUL summary) for a model

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-o', '--oasis-files-path', default=None, help='Path to Oasis files')
        parser.add_argument('-g', '--lookup-config-file-path', default=None, help='Lookup config JSON file path')
        parser.add_argument('-k', '--keys-data-path', default=None, help='Path to Oasis files')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')
        parser.add_argument('-l', '--lookup-package-path', default=None, help='Keys data directory path')
        parser.add_argument(
            '-p', '--canonical-exposures-profile-json-path', default=None,
            help='Path of the supplier canonical exposures profile JSON file'
        )
        parser.add_argument('-e', '--source-exposures-file-path', default=None, help='Source exposures file path')
        parser.add_argument(
            '-a', '--source-exposures-validation-file-path', default=None,
            help='Source exposures validation file (XSD) path (optional argument)'
        )
        parser.add_argument(
            '-b', '--source-to-canonical-exposures-transformation-file-path', default=None,
            help='Source -> canonical exposures transformation file (XSLT) path'
        )
        parser.add_argument(
            '-c', '--canonical-exposures-validation-file-path', default=None,
            help='Canonical exposures validation file (XSD) path (optional argument)'
        )
        parser.add_argument(
            '-d', '--canonical-to-model-exposures-transformation-file-path', default=None,
            help='Canonical exposures validation file (XSD) path, (optional argument)'
        )

    def action(self, args):
        """
        Generate Oasis files (items, coverages, GUL summary) for a model

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        default_oasis_files_path = os.path.join(os.getcwd(), 'runs', 'OasisFiles-{}'.format(utcnow))
        oasis_files_path = as_path(inputs.get('oasis_files_path', is_path=True, default=default_oasis_files_path), 'Oasis file', preexists=False)

        lookup_config_fp = as_path(inputs.get('lookup_config_file_path', required=False, is_path=True), 'Lookup config JSON file path', preexists=False)

        keys_data_path = as_path(inputs.get('keys_data_path', required=False, is_path=True), 'Keys data path', preexists=False)
        model_version_file_path = as_path(inputs.get('model_version_file_path', required=False, is_path=True), 'Model version file path', preexists=False)
        lookup_package_path = as_path(inputs.get('lookup_package_path', required=False, is_path=True), 'Lookup package path', preexists=False)

        if not (lookup_config_fp or (keys_data_path and model_version_file_path and lookup_package_path)):
            raise OasisException('Either the lookup config JSON file path or the keys data path + model version file path + lookup package path must be provided')

        canonical_exposures_profile_json_path = as_path(
            inputs.get('canonical_exposures_profile_json_path', required=True, is_path=True),
            'Canonical exposures profile json'
        )
        source_exposures_file_path = as_path(inputs.get('source_exposures_file_path', required=True, is_path=True), 'Source exposures')
        source_exposures_validation_file_path = as_path(
            inputs.get('source_exposures_validation_file_path', required=False, is_path=True),
            'Source exposures validation file',
            preexists=False
        )
        source_to_canonical_exposures_transformation_file_path = as_path(
            inputs.get('source_to_canonical_exposures_transformation_file_path', required=True, is_path=True),
            'Source to canonical exposures transformation'
        )
        canonical_exposures_validation_file_path = as_path(
            inputs.get('canonical_exposures_validation_file_path', required=False, is_path=True),
            'Canonical exposures validation file',
            preexists=False
        )
        canonical_to_model_exposures_transformation_file_path = as_path(
            inputs.get('canonical_to_model_exposures_transformation_file_path', required=True, is_path=True),
            'Canonical to model exposures transformation file'
        )

        self.logger.info('\nGetting model info and lookup')
        model_info, lookup = OasisLookupFactory.create(
                lookup_config_fp=lookup_config_fp,
                model_keys_data_path=keys_data_path,
                model_version_file_path=model_version_file_path,
                lookup_package_path=lookup_package_path
        )
        self.logger.info('\t{}, {}'.format(model_info, lookup))

        self.logger.info('\nCreating Oasis model object')
        model = OasisExposuresManager().create(
            model_supplier_id=model_info['supplier_id'],
            model_id=model_info['model_id'],
            model_version=model_info['model_version'],
            resources={
                'lookup': lookup,
                'lookup_config_fp': lookup_config_fp or None,
                'oasis_files_path': oasis_files_path,
                'source_exposures_file_path': source_exposures_file_path,
                'source_exposures_validation_file_path': source_exposures_validation_file_path,
                'source_to_canonical_exposures_transformation_file_path': source_to_canonical_exposures_transformation_file_path,
                'canonical_exposures_profile_json_path': canonical_exposures_profile_json_path,
                'canonical_exposures_validation_file_path': canonical_exposures_validation_file_path,
                'canonical_to_model_exposures_transformation_file_path': canonical_to_model_exposures_transformation_file_path
            }
        )
        self.logger.info('\t{}'.format(model))

        self.logger.info('\nSetting up Oasis files directory for model {}'.format(model.key))
        Path(oasis_files_path).mkdir(parents=True, exist_ok=True)

        self.logger.info('\nGenerating Oasis files for model')
        oasis_files = OasisExposuresManager().start_files_pipeline(
            oasis_model=model,
            logger=self.logger,
        )

        self.logger.info('\nGenerated Oasis files for model: {}'.format(oasis_files))


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
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-o', '--oasis-files-path', default=None, help='Path to Oasis files')
        parser.add_argument('-j', '--analysis-settings-json-file-path', default=None, help='Relative or absolute path of the model analysis settings JSON file')
        parser.add_argument('-m', '--model-data-path', default=None, help='Model data source path')
        parser.add_argument('-r', '--model-run-dir-path', default=None, help='Model run directory path')
        parser.add_argument('-s', '--ktools-script-name', default=None, help='Relative or absolute path of the output file')
        parser.add_argument('-n', '--ktools-num-processes', default=-1, help='Number of ktools calculation processes to use')
        parser.add_argument('-x', '--no-execute', action='store_true', help='Whether to execute generated ktools script')

    def action(self, args):
        """
        Generate losses using the installed ktools framework.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        oasis_files_path = as_path(inputs.get('oasis_files_path', required=True, is_path=True), 'Oasis files', preexists=True)

        model_run_dir_path = as_path(inputs.get('model_run_dir_path', required=False, is_path=True), 'Model run directory', preexists=False)

        analysis_settings_json_file_path = as_path(
            inputs.get('analysis_settings_json_file_path', required=True, is_path=True),
            'Analysis settings file'
        )
        model_data_path = as_path(inputs.get('model_data_path', required=True, is_path=True), 'Model data')

        ktools_script_name = inputs.get('ktools_script_name', default='run_ktools')
        no_execute = inputs.get('no_execute', default=False)

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
        super(self.__class__, self).add_args(parser)

        parser.add_argument('-k', '--keys-data-path', default=None, help='Path to Oasis files')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')
        parser.add_argument('-l', '--lookup-package-file-path', default=None, help='Keys data directory path')
        parser.add_argument('-g', '--lookup-config-file-path', default=None, help='Lookup config JSON file path')
        parser.add_argument(
            '-p', '--canonical-exposures-profile-json-path', default=None,
            help='Path of the supplier canonical exposures profile JSON file'
        )
        parser.add_argument('-e', '--source-exposures-file-path', default=None, help='Source exposures file path')
        parser.add_argument(
            '-a', '--source-exposures-validation-file-path', default=None,
            help='Source exposures validation file (XSD) path (optional argument)'
        )
        parser.add_argument(
            '-b', '--source-to-canonical-exposures-transformation-file-path', default=None,
            help='Source -> canonical exposures transformation file (XSLT) path'
        )
        parser.add_argument(
            '-c', '--canonical-exposures-validation-file-path', default=None,
            help='Canonical exposures validation file (XSD) path (optional argument)'
        )
        parser.add_argument(
            '-d', '--canonical-to-model-exposures-transformation-file-path', default=None,
            help='Canonical exposures validation file (XSD) path (optional argument)'
        )
        parser.add_argument(
            '-j', '--analysis-settings-json-file-path', default=None,
            help='Model analysis settings JSON file path'
        )
        parser.add_argument('-m', '--model-data-path', default=None, help='Model data source path')
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

        if not model_run_dir_path:
            utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
            model_run_dir_path = os.path.join(os.getcwd(), 'runs', 'ProgOasis-{}'.format(utcnow))
            self.logger.info('\nNo model run dir. provided - creating a timestamped run dir. in working directory as {}'.format(model_run_dir_path))
            Path(model_run_dir_path).mkdir(parents=True, exist_ok=True)
        else:
            if not os.path.exists(model_run_dir_path):
                Path(model_run_dir_path).mkdir(parents=True, exist_ok=True)

        args.model_run_dir_path = model_run_dir_path

        args.oasis_files_path = os.path.join(model_run_dir_path, 'tmp')
        self.logger.info('\nCreating temporary folder {} for Oasis files'.format(args.oasis_files_path))
        Path(args.oasis_files_path).mkdir(parents=True, exist_ok=True)

        gen_oasis_files_cmd = GenerateOasisFilesCmd()
        gen_oasis_files_cmd._logger = self.logger
        gen_oasis_files_cmd.action(args)

        gen_losses_cmd = GenerateLossesCmd()
        gen_losses_cmd._logger = self.logger
        gen_losses_cmd.action(args)


class ModelsCmd(OasisBaseCommand):
    sub_commands = {
        'generate-peril-areas-rtree-file-index': GeneratePerilAreasRtreeFileIndexCmd,
        'transform-source-to-canonical': TransformSourceToCanonicalFileCmd,
        'transform-canonical-to-model': TransformCanonicalToModelFileCmd,
        'generate-keys': GenerateKeysCmd,
        'generate-oasis-files': GenerateOasisFilesCmd,
        'generate-losses': GenerateLossesCmd,
        'run': RunCmd,
    }
