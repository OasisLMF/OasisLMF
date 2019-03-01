# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

"""
    Utilities for running system or file-related commands, and other OS-related utilities.
"""
import io
import socket

from .exceptions import OasisException

__all__ = [
    'load_ini_file',
    'replace_in_file',
]


def load_ini_file(ini_file_path):
    """
    Reads an INI file and returns it as a dictionary.
    """
    try:
        with io_open(ini_file_path, 'r', encoding='utf-8') as f:
            lines = [
                l.strip() for l in [l for l in f.read().split('\n') if l and not l.startswith('[')]
            ]
    except IOError as e:
        raise OasisException(str(e))

    di = {
        kv[0].strip(): kv[1].strip() for kv in tuple(line.split('=') for line in lines)
    }

    for k in di:
        if di[k].lower() == 'true':
            di[k] = True
        elif di[k].lower() == 'false':
            di[k] = False
        else:
            def ipf(s):
                return socket.inet_ntoa(socket.inet_aton(s))
            for conv in (int, float, ipf, str):
                try:
                    di[k] = conv(di[k])
                    break
                except:  # noqa: 722
                    continue
    return di


def replace_in_file(source_file_path, target_file_path, var_names, var_values):
    """
    Replaces a list of placeholders / variable names in a source file with a
    matching set of values, and writes it out to a new target file.
    """
    if len(var_names) != len(var_values):
        raise OasisException('Number of variable names does not equal the number of variable values to replace - please check and try again.')

    try:
        with io_open(source_file_path, 'r') as f:
            lines = f.readlines()

        with io_open(target_file_path, 'w') as f:
            for i in range(len(lines)):
                outline = inline = lines[i]

                present_var_names = [v for v in var_names if v in inline]

                if present_var_names:
                    for var_name in present_var_names:
                        var_value = var_values[var_names.index(var_name)]
                        outline = outline.replace(var_name, var_value)
                f.write(outline)
    except (OSError, IOError) as e:
        raise OasisException(str(e))
