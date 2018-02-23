#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Python utilities used for setting up the resources needed to complete a
    model run, i.e. generating ktools outputs from Oasis files.
"""

from __future__ import print_function

import glob
import logging
import tarfile
from itertools import chain

import shutilwhich
import six
from pathlib2 import Path
from six import itervalues

__all__ = [
    'create_binary_files',
    'prepare_model_run_directory',
    'prepare_model_run_inputs'
]

import os
import shutil
import subprocess

from ..utils.exceptions import OasisException
from .files import TAR_FILE, INPUT_FILES, GUL_INPUT_FILES, IL_INPUT_FILES


def prepare_model_run_directory(
    run_dir_path,
    oasis_files_src_path=None,
    analysis_settings_json_src_file_path=None,
    model_data_src_path=None,
    inputs_archive=None,
):
    """
    Ensures that the model run directory has the correct folder structure in
    order for the model run script (ktools) to be executed.

    ::

        <run_directory>
        ├── fifo
        ├── input
        │   └── csv
        ├── output
        ├── static
        └── work

    If any subfolders are missing they are created.

    Optionally, if the path to a set of Oasis files is provided then they
    are copied into the ``input/csv`` subfolder.

    Optionally, if the path to the analysis settings JSON file is provided
    then it is copied to the base of the run directory.

    Optionally, if the path to model data is provided then the files are
    symlinked into the ``static`` subfolder provided the OS is of type
    Darwin or Linux, otherwise the source folder tree is recursively
    copied into the ``static`` subfolder.

    :param run_directory: the model run directory
    :type run_directory: str

    :param oasis_files_src_path: path to a set of Oasis files
    :type oasis_files_src_path: str

    :param analysis_settings_json_src_file_path: analysis settings JSON file path
    :type analysis_settings_json_src_file_path: str

    :param model_data_src_path: model data source path
    :type model_data_src_path: str

    :param inputs_archive: path to a tar file containing input files
    :type inputs_archive: str
    """
    try:
        for subdir in ['fifo', 'output', 'static', 'work']:
            Path(run_dir_path, subdir).mkdir(parents=True, exist_ok=True)

        if not inputs_archive:
            Path(run_dir_path, 'input', 'csv').mkdir(parents=True, exist_ok=True)
        else:
            with tarfile.open(inputs_archive) as input_tarfile:
                input_tarfile.extractall(path=(os.path.join(run_dir_path, 'input')))

        if oasis_files_src_path:
            oasis_files_destpath = os.path.join(run_dir_path, 'input', 'csv')
            for p in os.listdir(oasis_files_src_path):
                shutil.copy2(os.path.join(oasis_files_src_path, p), oasis_files_destpath)

        if analysis_settings_json_src_file_path:
            analysis_settings_json_dest_file_path = os.path.join(run_dir_path, 'analysis_settings.json')
            shutil.copyfile(analysis_settings_json_src_file_path, analysis_settings_json_dest_file_path)

        if model_data_src_path:
            model_data_dest_path = os.path.join(run_dir_path, 'static')

            for path in glob.glob(os.path.join(model_data_src_path, '*')):
                filename = os.path.basename(path)
                try:
                    os.symlink(path, os.path.join(model_data_dest_path, filename))
                except Exception:
                    shutil.copytree(model_data_src_path, os.path.join(model_data_dest_path, filename))

    except OSError as e:
        raise OasisException(e)


def _prepare_input_bin(run_directory, bin_name, model_settings, setting_key=None):
    bin_file_path = os.path.join(run_directory, 'input', '{}.bin'.format(bin_name))
    if not os.path.exists(bin_file_path):
        setting_val = model_settings.get(setting_key)

        if not setting_val:
            model_data_bin_file_path = os.path.join(run_directory, 'static', '{}.bin'.format(bin_name))
        else:
            # Format for data file names
            setting_val = setting_val.replace(' ', '_').lower()
            model_data_bin_file_path = os.path.join(run_directory, 'static', '{}_{}.bin'.format(bin_name, setting_val))

        if not os.path.exists(model_data_bin_file_path):
            raise OasisException('Could not find {} data file: {}'.format(bin_name, model_data_bin_file_path))

        shutil.copyfile(model_data_bin_file_path, bin_file_path)


def prepare_model_run_inputs(analysis_settings, run_directory):
    """
    Sets up binary files in the model inputs directory.

    :param analysis_settings: model analysis settings dict
    :type analysis_settings: dict

    :param run_directory: model run directory
    :type run_directory: str
    """
    try:
        model_settings = analysis_settings.get('model_settings', {})

        _prepare_input_bin(run_directory, 'events', model_settings, setting_key='event_set')
        _prepare_input_bin(run_directory, 'returnperiods', model_settings)
        _prepare_input_bin(run_directory, 'occurrence', model_settings, setting_key='event_occurrence_id')
        
        if os.path.exists(os.path.join(run_directory, 'static', 'periods.bin')):
            _prepare_input_bin(run_directory, 'periods', model_settings)
    except (OSError, IOError) as e:
        raise OasisException(e)


def check_inputs_directory(directory_to_check, do_il=False, check_binaries=True):
    """
    Check that all the required csv files are present in the directory.
    Args:
        ``directory`` (string): the directory containing the CSV files.
        ``do_il`` (bool): do insured loss. If True, FM file must be present.
    Returns:
        None
    """
    file_path = os.path.join(directory_to_check, TAR_FILE)
    if os.path.exists(file_path):
        raise OasisException("Inputs tar file already exists: {}".format(file_path))

    if do_il:
        input_files = (f['name'] for f in six.itervalues(INPUT_FILES) if f['type'] != 'optional')
    else:
        input_files = (f['name'] for f in six.itervalues(INPUT_FILES) if f['type'] not in ['optional', 'il'])

    for input_file in input_files:
        file_path = os.path.join(directory_to_check, input_file + ".csv")
        if not os.path.exists(file_path):
            raise OasisException("Failed to find {}".format(file_path))

        if check_binaries:
            file_path = os.path.join(directory_to_check, input_file + ".bin")
            if os.path.exists(file_path):
                raise OasisException("Binary file already exists: {}".format(file_path))


def create_binary_files(csv_directory, bin_directory, do_il=False):
    """
    Create the binary files.

    :param csv_directory: the directory containing the CSV files
    :type csv_directory: str

    :param bin_directory: the directory to write the binary files
    :type bin_directory: str

    :param do_il: whether to perform insured loss (IL) calculations; if true, FM file must be present
    :type do_il: bool
    """
    csvdir = os.path.abspath(csv_directory)
    bindir = os.path.abspath(bin_directory)

    if do_il:
        input_files = itervalues(INPUT_FILES)
    else:
        input_files = (f for f in itervalues(INPUT_FILES) if f['type'] != 'il')

    for input_file in input_files:
        conversion_tool = input_file['conversion_tool']
        input_file_path = os.path.join(csvdir, '{}.csv'.format(input_file['name']))
        if not os.path.exists(input_file_path):
            continue

        output_file_path = os.path.join(bindir, '{}.bin'.format(input_file['name']))
        cmd_str = "{} < {} > {}".format(conversion_tool, input_file_path, output_file_path)

        try:
            subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            raise OasisException(e)


def check_binary_tar_file(tar_file_path, check_il=False):
    """
    Checks that all required files are present

    :param tar_file_path: Path to the tar file to check
    :type tar_file_path: str

    :param check_il: Flag whether to check insured loss files
    :type check_il: bool

    :raises OasisException: If a required file is missing

    :return: True if all required files are present
    """
    expected_members = ('{}.bin'.format(f['name']) for f in six.itervalues(GUL_INPUT_FILES))

    if check_il:
        expected_members = chain(expected_members, ('{}.bin'.format(f['name']) for f in six.itervalues(IL_INPUT_FILES)))

    with tarfile.open(tar_file_path) as tar:
        for member in expected_members:
            try:
                tar.getmember(member)
            except KeyError:
                raise OasisException('{} is missing from the tar file {}.'.format(member, tar_file_path))

    return True


def create_binary_tar_file(directory):
    """
    Package the binaries in a gzipped tar.
    """
    original_cwd = os.getcwd()
    os.chdir(directory)

    with tarfile.open(TAR_FILE, "w:gz") as tar:
        for file in glob.glob('*.bin'):
            tar.add(file)

    os.chdir(original_cwd)


def check_conversion_tools(do_il=False):
    # Check that the conversion tools are available
    if do_il:
        input_files = six.itervalues(INPUT_FILES)
    else:
        input_files = (f for f in six.itervalues(INPUT_FILES) if f['type'] != 'il')

    for input_file in input_files:
        tool = input_file['conversion_tool']
        if shutilwhich.which(tool) is None:
            error_message = "Failed to find conversion tool: {}".format(tool)
            logging.error(error_message)
            raise OasisException(error_message)

    return True


def cleanup_bin_directory(directory):
    """
    Clean the tar and binary files.
    """
    for file in chain([TAR_FILE], (f + '.bin' for f in six.iterkeys(INPUT_FILES))):
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            os.remove(file_path)
