#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Utilities for running system or file-related commands, and other OS-related utilities.
"""

from __future__ import print_function

__all__= [
    'load_ini_file',
    'replace_in_file',
    'run_mono_executable',
]

import difflib
import io
import os
import re
import socket
import subprocess
import sys
import time
import zlib

from .exceptions import OasisException


def load_ini_file(ini_file_path):
    """
    Reads an INI file and returns it as a dictionary.
    """
    lines = None

    try:
        with io.open(ini_file_path, 'r', encoding='utf-8') as f:
            lines = map(lambda l: l.strip(), filter(lambda l: l and not l.startswith('['), f.read().split('\n')))
    except IOError as e:
        raise OasisException(str(e))

    di = dict([line.replace(' ', '').split('=') for line in lines])

    for k in di:
        if di[k] in ['True', 'False']:
            di[k] = bool(di[k])
        else:
            if di[k] != u'':
                try:
                    socket.inet_aton(di[k])
                except:
                    pass
                else:
                    continue
                if re.match(r'[-+]?\d+\.\d+', di[k]):
                    di[k] = float(di[k])
                else:
                    i = 0
                    try:
                        i = int(di[k])
                    except ValueError as e:
                        pass
                    else:
                        di[k] = i
            else:
                di[k] = None

    return di


def replace_in_file(source_file_path, target_file_path, var_names, var_values):
    """
    Replaces a list of placeholders / variable names in a source file with a
    matching set of values, and writes it out to a new target file.
    """
    if len(var_names) != len(var_values):
        raise OasisException('Number of variable names does not equal the number of variable values to replace - please check and try again.')

    try:
        with open(source_file_path, 'r') as f:
            lines = f.readlines()

        with open(target_file_path, 'w') as f:
            for i in range(len(lines)):
                outline = inline = lines[i]
                present_var_names = filter(lambda var_name: var_name in inline, var_names)
                if present_var_names:
                    for var_name in present_var_names:
                        var_value = var_values[var_names.index(var_name)]
                        outline = outline.replace(var_name, var_value)
                f.write(outline)
    except (OSError, IOError) as e:
        raise OasisException(str(e))


def run_mono_executable(
    executable_path,
    executable_args=None
):
    """
    Utility method to run executables compiled for the mono framework.
    """
    args_str = (
        ''.join(['-{} {} '.format(key, val) for key, val in executable_args.items()]).strip() if executable_args
        else ''
    )
    cmd_str = 'mono {} {}'.format(executable_path, args_str).strip()
    
    try:
        retcode = subprocess.call(cmd_str, shell=True)
        if retcode < 0:
            print('Mono executable call failed: {}'.format(-retcode), file=sys.stderr)
        else:
            print('Mono executable call succeeded: {}'.format(retcode), file=sys.stderr)
    except OSError as e:
        print('Mono executable call failed: {}'.format(str(e)), file=sys.stderr)
        raise OasisException(str(e))
