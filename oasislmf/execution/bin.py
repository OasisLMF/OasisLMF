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
    'prepare_run_inputs'
]


import errno
import csv
import filecmp
import glob
import logging
import os
import re
import shutil
import shutilwhich
import subprocess
import tarfile
import pandas as pd

from itertools import chain

from pathlib import Path

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from .files import TAR_FILE, INPUT_FILES, GUL_INPUT_FILES, IL_INPUT_FILES


@oasis_log
def prepare_run_directory(
    run_dir,
    oasis_src_fp,
    model_data_fp,
    analysis_settings_fp,
    inputs_archive=None,
    user_data_dir=None,
    ri=False,
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
        |-- ...
        |-- output
        |-- static
        |-- work
        |-- ri_layers.json
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

    :param model_data_fp: model data source path
    :type model_data_fp: str

    :param inputs_archive: path to a tar file containing input files
    :type inputs_archive: str

    :param: user_data_dir: path to a directory containing additional user-supplied model data
    :type user_data_dir: str
    """
    try:
        for subdir in ['fifo', 'output', 'static', 'work']:
            Path(run_dir, subdir).mkdir(parents=True, exist_ok=True)

        if not inputs_archive:
            Path(run_dir, 'input').mkdir(parents=True, exist_ok=True)
        else:
            with tarfile.open(inputs_archive) as input_tarfile:
                p = os.path.join(run_dir, 'input') if not ri else os.path.join(run_dir)
                input_tarfile.extractall(path=p)

        oasis_dst_fp = os.path.join(run_dir, 'input')

        for p in os.listdir(oasis_src_fp):
            src = os.path.join(oasis_src_fp, p)
            if src.endswith('.tar') or src.endswith('.tar.gz'):
                continue
            dst = os.path.join(oasis_dst_fp, p)
            if not (re.match(r'RI_\d+$', p) or p == 'ri_layers.json'):
                shutil.copy2(src, oasis_dst_fp) if not (os.path.exists(dst) and filecmp.cmp(src, dst)) else None
            else:
                shutil.move(src, run_dir)

        dst = os.path.join(run_dir, 'analysis_settings.json')
        shutil.copy(analysis_settings_fp, dst) if not (os.path.exists(dst) and filecmp.cmp(analysis_settings_fp, dst, shallow=False)) else None

        model_data_dst_fp = os.path.join(run_dir, 'static')

        try:
            for sourcefile in glob.glob(os.path.join(model_data_fp, '*')):
                destfile = os.path.join(model_data_dst_fp, os.path.basename(sourcefile))

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
        columns =['return_period']).sort_values(ascending=False, by=['return_period']
    ).to_csv(csv_fp, index=False)

    try:
        cmd_str = "returnperiodtobin < \"{}\" > \"{}\"".format(csv_fp, bin_fp)
        subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        raise OasisException("Error while converting returnperiods.csv to ktools binary format: {}".format(e))


def _prepare_input_bin(run_dir, bin_name, model_settings, setting_key=None, ri=False):
    bin_fp = os.path.join(run_dir, 'input', '{}.bin'.format(bin_name))
    if not os.path.exists(bin_fp):
        setting_val = model_settings.get(setting_key)

        if not setting_val:
            model_data_bin_fp = os.path.join(run_dir, 'static', '{}.bin'.format(bin_name))
        else:
            # 'verbatim' -  Try setting value as given
            model_data_bin_fp = os.path.join(run_dir, 'static', '{}_{}.bin'.format(bin_name, str(setting_val)))
            if not os.path.isfile(model_data_bin_fp):
                # 'compatibility' - Fallback name formating to keep existing conversion
                setting_val = str(setting_val).replace(' ', '_').lower()
                model_data_bin_fp = os.path.join(run_dir, 'static', '{}_{}.bin'.format(bin_name, setting_val))

        if not os.path.exists(model_data_bin_fp):
            raise OasisException('Could not find {} data file: {}'.format(bin_name, model_data_bin_fp))

        shutil.copyfile(model_data_bin_fp, bin_fp)


def _calc_selected(analysis_settings, calc_type):
    """
    Return True, if "calc_type" is set in the anaylysis settings file

    :param calc_type: one of `eltcalc`, `lec_output`, `aalcalc` or `pltcalc`
    :type calc_type: str
    """
    is_in_gul = False
    is_in_il = False
    is_in_ri = False

    gul_section = analysis_settings.get('gul_summaries')
    il_section = analysis_settings.get('il_summaries')
    ri_section = analysis_settings.get('ri_summaries')

    if gul_section:
        is_in_gul = any(gul_summary.get(calc_type, None) for gul_summary in gul_section)
    if il_section:
        is_in_il = any(il_summary.get(calc_type, None) for il_summary in il_section)
    if ri_section:
        is_in_ri = any(ri_summary.get(calc_type, None) for ri_summary in ri_section)

    return any([is_in_gul, is_in_il, is_in_ri])

@oasis_log
def prepare_run_inputs(analysis_settings, run_dir, ri=False):
    """
    Sets up binary files in the model inputs directory.

    :param analysis_settings: model analysis settings dict
    :type analysis_settings: dict

    :param run_dir: model run directory
    :type run_dir: str
    """
    try:
        model_settings = analysis_settings.get('model_settings', {})
        _prepare_input_bin(run_dir, 'events', model_settings, setting_key='event_set', ri=ri)

        # Prepare occurrence / returnperiod depending on output calcs selected
        if _calc_selected(analysis_settings, 'lec_output'):
            if analysis_settings.get('return_periods'):
                # Create return periods from user input
                _create_return_period_bin(run_dir, analysis_settings.get('return_periods'))
            else:
                # copy return periods from static
                _prepare_input_bin(run_dir, 'returnperiods', model_settings)

            _prepare_input_bin(run_dir, 'occurrence', model_settings, setting_key='event_occurrence_id', ri=ri)
        elif _calc_selected(analysis_settings, 'pltcalc') or _calc_selected(analysis_settings, 'aalcalc'):
            _prepare_input_bin(run_dir, 'occurrence', model_settings, setting_key='event_occurrence_id', ri=ri)

        if os.path.exists(os.path.join(run_dir, 'static', 'periods.bin')):
            _prepare_input_bin(run_dir, 'periods', model_settings, ri=ri)
    except (OSError, IOError) as e:
        raise OasisException("Error preparing the model 'inputs' directory: {}".format(e))


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
