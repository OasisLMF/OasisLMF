"""
    Python utilities used for setting up the structure of the run directory
    in which to prepare the inputs to run a model or generate deterministic
    losses, and store the outputs.
"""
__all__ = [
    'check_binary_tar_file',
    'check_conversion_tools',
    'check_inputs_directory',
    'cleanup_bin_directory',
    'create_binary_tar_file',
    'csv_to_bin',
    'prepare_run_directory',
    'prepare_run_inputs',
    'set_footprint_set',
    'set_vulnerability_set'
]

import pathlib

import errno
import csv
import filecmp
import glob
import logging
import os
import shutil
import shutilwhich
import subprocess
import tarfile
import pandas as pd

from itertools import chain

from pathlib import Path

from oasis_data_manager.filestore.backends.base import BaseStorage
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.defaults import STATIC_DATA_FP
from .files import TAR_FILE, INPUT_FILES, GUL_INPUT_FILES, IL_INPUT_FILES
from .bash import leccalc_enabled, ord_enabled, ORD_LECCALC
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel.vulnerability import vulnerability_dataset, parquetvulnerability_meta_filename

logger = logging.getLogger(__name__)


@oasis_log
def prepare_run_directory(
    run_dir,
    oasis_src_fp,
    model_data_fp,
    analysis_settings_fp,
    inputs_archive=None,
    user_data_dir=None,
    ri=False,
    copy_model_data=False,
    model_storage_config_fp=None
):
    """
    Ensures that the model run directory has the correct folder structure in
    order for the model run script (ktools) to be executed. Without the RI
    flag the model run directory will have the following structure

    ::

        <run_directory>
        |-- fifo/
        |-- input/
        |   `-- csv/
        |-- output/
        |-- static/
        |-- work/
        |-- analysis_settings.json
        `-- run_ktools.sh


    where the direct GUL and/or FM input files exist in the ``input/csv``
    subfolder and the corresponding binaries exist in the ``input`` subfolder.

    With the RI flag the model run directory has the following structure

    ::
        <run_directory>
        |-- fifo
        |-- input
            |-- RI_1
            |-- RI_2
            |-- ri_layers.json
        |-- ...
        |-- output
        |-- static
        |-- work
        |-- analysis_settings.json
        `-- run_ktools.sh

    where the direct GUL and/or FM input files, and the corresponding binaries
    exist in the ``input`` subfolder, and the RI layer input files and binaries
    exist in the ``RI`` prefixed subfolders.

    If any subfolders are missing they are created.

    Optionally, if the path to a set of Oasis files is provided then they
    are copied into the ``input/csv`` subfolder.

    Optionally, if the path to the analysis settings JSON file is provided
    then it is copied to the base of the run directory.

    Optionally, if the path to model data is provided then the files are
    symlinked into the ``static`` subfolder provided the OS is of type
    Darwin or Linux, otherwise the source folder tree is recursively
    copied into the ``static`` subfolder.

    :param run_dir: the model run directory
    :type run_dir: str

    :param oasis_src_fp: path to a set of Oasis files
    :type oasis_src_fp: str

    :param ri: Boolean flag for RI mode
    :type ri: bool

    :param analysis_settings_fp: analysis settings JSON file path
    :type analysis_settings_fp: str

    :param model_data_fp: model data source path, if this is a file it will be loaded as a
        custom storage class
    :type model_data_fp: str

    :param inputs_archive: path to a tar file containing input files
    :type inputs_archive: str

    :param user_data_dir: path to a directory containing additional user-supplied model data
    :type user_data_dir: str

    :param model_storage_config_fp: path to the model storage configuration, if not present
        the model data will be copied to the static directory
    :type model_storage_config_fp: str
    """
    try:
        for subdir in ['fifo', 'output', 'static', 'work']:
            Path(run_dir, subdir).mkdir(parents=True, exist_ok=True)

        if not inputs_archive:
            Path(run_dir, 'input').mkdir(parents=True, exist_ok=True)
        else:
            with tarfile.open(inputs_archive) as input_tarfile:
                p = os.path.join(run_dir, 'input')

                def is_within_directory(directory, target):

                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)

                    prefix = os.path.commonprefix([abs_directory, abs_target])

                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")

                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(input_tarfile, path=p)

        oasis_dst_fp = os.path.join(run_dir, 'input')

        file_to_copy = os.listdir(oasis_src_fp)
        while file_to_copy:
            p = file_to_copy.pop()
            src = os.path.join(oasis_src_fp, p)
            if src.endswith('.tar') or src.endswith('.tar.gz'):
                continue
            if os.path.isdir(src):
                pathlib.Path(oasis_dst_fp, p).mkdir(parents=True, exist_ok=True)
                file_to_copy.extend(os.path.join(p, elm) for elm in os.listdir(src))
                continue
            dst = os.path.join(oasis_dst_fp, p)
            if not (os.path.exists(dst) and filecmp.cmp(src, dst)):
                shutil.copy2(src, pathlib.Path(dst).parent)
        dst = os.path.join(run_dir, 'analysis_settings.json')
        shutil.copy(analysis_settings_fp, dst) if not (os.path.exists(dst) and filecmp.cmp(analysis_settings_fp, dst, shallow=False)) else None

        model_data_dst_fp = os.path.join(run_dir, 'static')

        if not model_storage_config_fp:
            try:
                for sourcefile in glob.glob(os.path.join(model_data_fp, '*')):
                    destfile = os.path.join(model_data_dst_fp, os.path.basename(sourcefile))

                    if os.name == 'nt' or copy_model_data:
                        shutil.copy(sourcefile, destfile)
                    else:
                        os.symlink(sourcefile, destfile)
            except OSError as e:
                if not (e.errno == errno.EEXIST and os.path.islink(destfile) and os.name != 'nt'):
                    raise e
                else:
                    # If the link already exists, check files are different replace it
                    if os.readlink(destfile) != os.path.abspath(sourcefile):
                        os.symlink(sourcefile, destfile + ".tmp")
                        os.replace(destfile + ".tmp", destfile)

        if model_storage_config_fp:
            try:
                shutil.copy(model_storage_config_fp, os.path.join(run_dir, "model_storage.json"))
            except shutil.SameFileError as e:
                pass

        if user_data_dir and os.path.exists(user_data_dir):
            for sourcefile in glob.glob(os.path.join(user_data_dir, '*')):
                destfile = os.path.join(model_data_dst_fp, os.path.basename(sourcefile))

                try:
                    if os.name == 'nt':
                        shutil.copy(sourcefile, destfile)
                    else:
                        os.symlink(sourcefile, destfile)
                except OSError as e:
                    if not (e.errno == errno.EEXIST and os.path.islink(destfile) and os.name != 'nt'):
                        raise e
                    else:
                        # If the link already exists, check files are different replace it
                        if os.readlink(destfile) != os.path.abspath(sourcefile):
                            os.symlink(sourcefile, destfile + ".tmp")
                            os.replace(destfile + ".tmp", destfile)

    except OSError as e:
        raise OasisException("Error preparing the 'run' directory: {}".format(e))


def _create_return_period_bin(run_dir, return_periods):
    csv_fp = os.path.join(run_dir, 'input', 'returnperiods.csv')
    bin_fp = os.path.join(run_dir, 'input', 'returnperiods.bin')
    pd.DataFrame(
        return_periods,
        columns=['return_period']).sort_values(ascending=False, by=['return_period']
                                               ).to_csv(csv_fp, index=False)

    try:
        cmd_str = "returnperiodtobin < \"{}\" > \"{}\"".format(csv_fp, bin_fp)
        subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        raise OasisException("Error while converting returnperiods.csv to ktools binary format: {}".format(e))


def _create_events_bin(run_dir, event_ids):
    csv_fp = os.path.join(run_dir, 'input', 'events.csv')
    bin_fp = os.path.join(run_dir, 'input', 'events.bin')
    pd.DataFrame(
        event_ids,
        columns=['event_id']).sort_values(ascending=True, by=['event_id']
                                          ).to_csv(csv_fp, index=False)

    try:
        cmd_str = "evetobin < \"{}\" > \"{}\"".format(csv_fp, bin_fp)
        subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        raise OasisException("Error while converting events.csv to ktools binary format: {}".format(e))


def _create_quantile_bin(run_dir, quantiles):
    csv_fp = os.path.join(run_dir, 'input', 'quantile.csv')
    bin_fp = os.path.join(run_dir, 'input', 'quantile.bin')
    pd.DataFrame(
        quantiles,
        dtype='float',
        columns=['quantile']).sort_values(ascending=True, by=['quantile']
                                          ).to_csv(csv_fp, index=False)

    try:
        cmd_str = "quantiletobin < \"{}\" > \"{}\"".format(csv_fp, bin_fp)
        subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        raise OasisException("Error while converting quantile.csv to ktools binary format: {}".format(e))


def _load_default_quantile_bin(run_dir):
    default_quantile_fp = os.path.join(STATIC_DATA_FP, 'quantile.csv')
    _create_quantile_bin(run_dir,
                         pd.read_csv(default_quantile_fp)["quantile"].tolist()
                         )


def _prepare_input_bin(run_dir, bin_name, model_settings, storage: BaseStorage, setting_key=None, ri=False, extension='bin'):
    bin_fp = os.path.join(run_dir, 'input', '{}.{}'.format(bin_name, extension))
    if not os.path.exists(bin_fp):
        setting_val = model_settings.get(setting_key)

        if not setting_val:
            targets = [
                f"{bin_name}.{extension}",
            ]
        else:
            targets = [
                f"{bin_name}_{str(setting_val)}.{extension}",
                f"{bin_name}_{str(setting_val).replace(' ', '_').lower()}.{extension}"
            ]

        for fname in targets:
            if storage.exists(fname):
                storage.get(fname, bin_fp)

        if not os.path.exists(bin_fp):
            raise OasisException('Could not find {} data file: {}'.format(bin_name, targets))


def _calc_selected(analysis_settings, calc_type_list):
    """
    Return True, if any options in "calc_type_list" are set in the analysis settings file

    :param calc_type_list: List of string values or ktools outputs, e.g. `eltcalc`, `lec_output`, `aalcalc` or `pltcalc`
    :type  calc_type_list: list
    """
    gul_section = analysis_settings.get('gul_summaries')
    il_section = analysis_settings.get('il_summaries')
    ri_section = analysis_settings.get('ri_summaries')

    for calc_type in calc_type_list:

        if gul_section:
            if any(gul_summary.get(calc_type, None) or gul_summary.get('ord_output', {}).get(calc_type) for gul_summary in gul_section):
                return True
        if il_section:
            if any(il_summary.get(calc_type, None) or il_summary.get('ord_output', {}).get(calc_type) for il_summary in il_section):
                return True
        if ri_section:
            if any(ri_summary.get(calc_type, None) or ri_summary.get('ord_output', {}).get(calc_type) for ri_summary in ri_section):
                return True
    return False


def _leccalc_selected(analysis_settings):
    """ return True if either 'leccalc' or 'ordleccalc' is referenced in the analysis settings file
    """
    is_in_gul = False
    is_in_il = False
    is_in_ri = False

    gul_section = analysis_settings.get('gul_summaries')
    il_section = analysis_settings.get('il_summaries')
    ri_section = analysis_settings.get('ri_summaries')

    if gul_section:
        is_in_gul = any(leccalc_enabled(gul_summary) or ord_enabled(gul_summary, ORD_LECCALC) for gul_summary in gul_section)
    if il_section:
        is_in_il = any(leccalc_enabled(il_summary) or ord_enabled(il_summary, ORD_LECCALC) for il_summary in il_section)
    if ri_section:
        is_in_ri = any(leccalc_enabled(ri_summary) or ord_enabled(ri_summary, ORD_LECCALC) for ri_summary in ri_section)

    return any([is_in_gul, is_in_il, is_in_ri])


def get_event_range(event_range):
    """
    Parses event range string and returns a list of event ids.

    :param event_range: string representation of event range in format '1-5,10,89-100'
    :type event_range: str
    """
    lst_event_range = event_range.split(",")
    lst_events = []

    for er in lst_event_range:
        if '-' in er:
            rng = er.split('-')
            e_from = int(rng[0])
            e_to = int(rng[1])
            lst_events.extend(range(e_from, e_to + 1))
        else:
            e = int(er)
            lst_events.append(e)

    return (lst_events)


@oasis_log
def prepare_run_inputs(analysis_settings, run_dir, model_storage: BaseStorage, ri=False):
    """
    Sets up binary files in the model inputs directory.

    :param analysis_settings: model analysis settings dict
    :type analysis_settings: dict

    :param run_dir: model run directory
    :type run_dir: str
    """
    try:
        model_settings = analysis_settings.get('model_settings', {})
        # Prepare events.bin
        if analysis_settings.get('event_ids') or analysis_settings.get('event_ranges'):
            # Create events file from user input
            event_ids = []
            if analysis_settings.get('event_ids'):
                event_ids.extend(analysis_settings.get('event_ids'))
            if analysis_settings.get('event_ranges'):
                event_range = get_event_range(analysis_settings.get('event_ranges'))
                event_ids.extend(event_range)
            event_ids = list(set(event_ids))
            _create_events_bin(run_dir, event_ids)
        else:
            # copy selected event set from static
            _prepare_input_bin(run_dir, 'events', model_settings, model_storage, setting_key='event_set', ri=ri)

        # Prepare event_rates.csv
        if model_storage.exists('event_rates.csv') or model_settings.get('event_rates_set'):
            _prepare_input_bin(run_dir, 'event_rates', model_settings, model_storage, setting_key='event_rates_set', ri=ri, extension='csv')

        # Prepare quantile.bin
        if analysis_settings.get('quantiles'):
            # 1. Create quantile file from user input
            _create_quantile_bin(run_dir, analysis_settings.get('quantiles'))
        elif _calc_selected(analysis_settings, ['plt_quantile', 'elt_quantile']):
            # 2. copy quantile file from model data
            if model_storage.exists('quantile.bin'):
                _prepare_input_bin(run_dir, 'quantile', model_settings, model_storage, ri=ri)
            else:
                # 3. Create quantile file from package `_data/quantile.csv`
                _load_default_quantile_bin(run_dir)

        # Prepare occurrence / returnperiod depending on output calcs selected
        if _leccalc_selected(analysis_settings):
            if analysis_settings.get('return_periods'):
                # Create return periods from user input
                _create_return_period_bin(run_dir, analysis_settings.get('return_periods'))
            else:
                # copy return periods from static
                _prepare_input_bin(run_dir, 'returnperiods', model_settings, model_storage)

            _prepare_input_bin(run_dir, 'occurrence', model_settings, model_storage, setting_key='event_occurrence_id', ri=ri)
        elif _calc_selected(analysis_settings, [
            'pltcalc', 'aalcalc', 'aalcalcmeanonly', 'alt_period', 'elt_moment', 'elt_quantile',
            'elt_sample', 'plt_moment', 'plt_quantile', 'plt_sample'
        ]):
            _prepare_input_bin(run_dir, 'occurrence', model_settings, model_storage, setting_key='event_occurrence_id', ri=ri)

        # Prepare periods.bin
        if model_storage.exists('periods.bin'):
            _prepare_input_bin(run_dir, 'periods', model_settings, model_storage, ri=ri)

    except (OSError, IOError) as e:
        raise OasisException("Error preparing the model 'inputs' directory: {}".format(e))


@oasis_log
def set_footprint_set(setting_val, run_dir):
    """
    Create symbolic link to footprint file set that will be used for output
    calculation.

    :param setting_val: identifier for footprint set
    :type setting_val: string

    :param run_dir: model run directory
    :type run_dir: string
    """
    priorities = Footprint.get_footprint_fmt_priorities()
    setting_val = str(setting_val)

    for footprint_class in priorities:
        for filename in footprint_class.footprint_filenames:
            stem, extension = filename.split('.', 1)
            footprint_fp = os.path.join(run_dir, 'static', f'{stem}_{setting_val}.{extension}')
            footprint_target_fp = os.path.join(run_dir, 'static', filename)
            if not os.path.exists(footprint_fp):
                # 'compatibility' - Fallback name formatting to keep existing conversion
                setting_val_old = setting_val.replace(' ', '_').lower()
                footprint_fp = os.path.join(run_dir, 'static', f'{stem}_{setting_val_old}.{extension}')
                if not os.path.exists(footprint_fp):
                    logger.debug(f'{footprint_fp} not found, moving on to next footprint class')
                    break
            os.symlink(footprint_fp, footprint_target_fp)
        else:
            return

    raise OasisException(f'Could not find footprint data files with identifier "{setting_val}"')


@oasis_log
def set_vulnerability_set(setting_val, run_dir):
    """
    Create symbolic link to vulnerability file set that will be used for output
    calculation.

    :param setting_val: identifier for vulnerability set
    :type setting_val: string

    :param run_dir: model run directory
    :type run_dir: string
    """

    vulnerability_formats = ['bin', 'parquet', 'csv']
    setting_val = str(setting_val)

    for file_format in vulnerability_formats:

        if file_format == 'parquet':
            base_name, format_extension = vulnerability_dataset.split('.', 1)
            base_meta_name, _ = parquetvulnerability_meta_filename.split('.', 1)
            # For Parquet, check if it's a directory
            vulnerability_fp = os.path.join(run_dir, 'static', f'{base_name}_{setting_val}.{format_extension}')
            vulnerability_target_fp = os.path.join(run_dir, 'static', f'{vulnerability_dataset}')
            vulnerability_meta = os.path.join(run_dir, 'static', f'{base_meta_name}_{setting_val}.json')
            vulnerability_meta_target = os.path.join(run_dir, 'static', f'{parquetvulnerability_meta_filename}')
            if os.path.isdir(vulnerability_fp) and os.path.isfile(vulnerability_meta):
                os.symlink(vulnerability_fp, vulnerability_target_fp)
                os.symlink(vulnerability_meta, vulnerability_meta_target)
                return
            else:
                logger.debug(f'{vulnerability_fp} and/or {vulnerability_meta} not found, trying next format')
        else:
            # For other file formats, check if it's a file
            vulnerability_fp = os.path.join(run_dir, 'static', f'vulnerability_{setting_val}.{file_format}')
            vulnerability_target_fp = os.path.join(run_dir, 'static', f'vulnerability.{file_format}')
            if os.path.isfile(vulnerability_fp):
                os.symlink(vulnerability_fp, vulnerability_target_fp)
                return
            else:
                logger.debug(f'{vulnerability_fp} not found, trying next format')

    raise OasisException(f'Could not find vulnerability data files with identifier "{setting_val}"')


@oasis_log
def check_inputs_directory(directory_to_check, il=False, ri=False, check_binaries=True):
    """
    Check that all the required files are present in the directory.

    :param directory_to_check: directory containing the CSV files
    :type directory_to_check: string

    :param il: check insuured loss files
    :type il: bool

    :param il: check resinsurance sub-folders
    :type il: bool

    :param check_binaries: check binary files are not present
    :type check_binaries: bool
    """
    # Check the top level directory, that containes the core files and any direct FM files
    _check_each_inputs_directory(directory_to_check, il=il, check_binaries=check_binaries)

    if ri:
        for ri_directory_to_check in glob.glob('{}{}RI_\d+$'.format(directory_to_check, os.path.sep)):
            _check_each_inputs_directory(ri_directory_to_check, il=True, check_binaries=check_binaries)


def _check_each_inputs_directory(directory_to_check, il=False, check_binaries=True):
    """
    Detailed check of a specific directory
    """

    if il:
        input_files = (f['name'] for f in INPUT_FILES.values() if f['type'] != 'optional')
    else:
        input_files = (f['name'] for f in INPUT_FILES.values() if f['type'] not in ['optional', 'il'])

    for input_file in input_files:
        file_path = os.path.join(directory_to_check, input_file + ".csv")
        if not os.path.exists(file_path):
            raise OasisException("Failed to find {}".format(file_path))

        if check_binaries:
            file_path = os.path.join(directory_to_check, input_file + ".bin")
            if os.path.exists(file_path):
                raise OasisException("Binary file already exists: {}".format(file_path))


@oasis_log
def csv_to_bin(csv_directory, bin_directory, il=False, ri=False):
    """
    Create the binary files.

    :param csv_directory: the directory containing the CSV files
    :type csv_directory: str

    :param bin_directory: the directory to write the binary files
    :type bin_directory: str

    :param il: whether to create the binaries required for insured loss calculations
    :type il: bool

    :param ri: whether to create the binaries required for reinsurance calculations
    :type ri: bool

    :raises OasisException: If one of the conversions fails
    """
    csvdir = os.path.abspath(csv_directory)
    bindir = os.path.abspath(bin_directory)

    il = il or ri

    _csv_to_bin(csvdir, bindir, il)

    if ri:
        for ri_csvdir in glob.glob('{}{}RI_[0-9]*'.format(csvdir, os.sep)):
            _csv_to_bin(
                ri_csvdir, os.path.join(bindir, os.path.basename(ri_csvdir)), il=True)


def _csv_to_bin(csv_directory, bin_directory, il=False):
    """
    Create a set of binary files.
    """
    if not os.path.exists(bin_directory):
        os.mkdir(bin_directory)

    if il:
        input_files = INPUT_FILES.values()
    else:
        input_files = (f for f in INPUT_FILES.values() if f['type'] != 'il')

    for input_file in input_files:
        conversion_tool = input_file['conversion_tool']
        input_file_path = os.path.join(csv_directory, '{}.csv'.format(input_file['name']))
        if not os.path.exists(input_file_path):
            continue

        output_file_path = os.path.join(bin_directory, '{}.bin'.format(input_file['name']))

        # If input file is different for step policies, apply flag when
        # executing conversion tool should step policies be present
        step_flag = input_file.get('step_flag')
        col_names = []
        if step_flag:
            with open(input_file_path) as f:
                reader = csv.reader(f)
                col_names = next(reader)

        if 'step_id' in col_names:
            output_file_path = os.path.join(
                bin_directory, '{}{}.bin'.format(input_file['name'], '_step')
            )

            cmd_str = "{} {} < \"{}\" > \"{}\"".format(conversion_tool, step_flag, input_file_path, output_file_path)
        else:
            cmd_str = "{} < \"{}\" > \"{}\"".format(conversion_tool, input_file_path, output_file_path)

        try:
            subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            raise OasisException("Error while converting csv's to ktools binary format: {}".format(e))


@oasis_log
def check_binary_tar_file(tar_file_path, check_il=False):
    """
    Checks that all required files are present

    :param tar_file_path: Path to the tar file to check
    :type tar_file_path: str

    :param check_il: Flag whether to check insured loss files
    :type check_il: bool

    :raises OasisException: If a required file is missing

    :return: True if all required files are present, False otherwise
    :rtype: bool
    """
    expected_members = ('{}.bin'.format(f['name']) for f in GUL_INPUT_FILES.values())

    if check_il:
        expected_members = chain(expected_members, ('{}.bin'.format(f['name']) for f in IL_INPUT_FILES.values()))

    with tarfile.open(tar_file_path) as tar:
        for member in expected_members:
            try:
                tar.getmember(member)
            except KeyError:
                raise OasisException('{} is missing from the tar file {}.'.format(member, tar_file_path))

    return True


@oasis_log
def create_binary_tar_file(directory):
    """
    Package the binaries in a gzipped tar.

    :param directory: Path containing the binaries
    :type tar_file_path: str
    """
    with tarfile.open(os.path.join(directory, TAR_FILE), "w:gz") as tar:
        for f in glob.glob('{}*{}*.bin'.format(directory, os.sep)):
            logging.info("Adding {} {}".format(f, os.path.relpath(f, directory)))
            relpath = os.path.relpath(f, directory)
            tar.add(f, arcname=relpath)

        for f in glob.glob('{}*{}*{}*.bin'.format(directory, os.sep, os.sep)):
            relpath = os.path.relpath(f, directory)
            tar.add(f, arcname=relpath)


@oasis_log
def check_conversion_tools(il=False):
    """
    Check that the conversion tools are available

    :param il: Flag whether to check insured loss tools
    :type il: bool

    :return: True if all required tools are present, False otherwise
    :rtype: bool
    """
    if il:
        input_files = INPUT_FILES.values()
    else:
        input_files = (f for f in INPUT_FILES.values() if f['type'] != 'il')

    for input_file in input_files:
        tool = input_file['conversion_tool']
        if shutilwhich.which(tool) is None:
            error_message = "Failed to find conversion tool: {}".format(tool)
            logging.error(error_message)
            raise OasisException(error_message)

    return True


@oasis_log
def cleanup_bin_directory(directory):
    """
    Clean the tar and binary files.
    """
    for file in chain([TAR_FILE], (f + '.bin' for f in INPUT_FILES.keys())):
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            os.remove(file_path)
