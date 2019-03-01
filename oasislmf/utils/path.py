# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

__all__ = [
    'as_path',
    'empty_dir',
    'PathCleaner',
    'setcwd'
]

import os
import shutil

from contextlib import contextmanager
from future.utils import string_types

from .exceptions import OasisException


def as_path(path, label, is_dir=False, preexists=True):
    """
    Processes the path and returns the absolute path.

    If the path does not exist and ``preexists`` is true
    an ``OasisException`` is raised.

    :param path: The path to process
    :type path: str

    :param label: Human-readable label of the path (used for error reporting)
    :type label: str

    :param is_dir: Whether the path is a directory
    :type is_dir: bool

    :param preexists: Flag whether to raise an error if the path
        does not exist.
    :type preexists: bool

    :return: The absolute path of the input path
    """
    
    if not isinstance(path, string_types):
        return
    _path = ''.join(path)
    if not os.path.isabs(path):
        _path = os.path.abspath(_path)
    if preexists and not os.path.exists(_path):
        raise OasisException('The path {} ({}) is indicated as preexisting but does not exist'.format(_path, label))
    if is_dir and preexists and not os.path.isdir(_path):
        raise OasisException('The path {} ({}) is indicated as a preexisting directory but is not actually a directory'.format(_path, label))

    return _path


def empty_dir(dir_fp):
    """
    Empties the contents of a directory, but leaves the directory in place.

    :param dir_fp: A pre-existing directory path
    :type dir_fp: str
    """
    _dir_fp = as_path(dir_fp, dir_fp, is_dir=True)

    for p in (os.path.join(_dir_fp, fn) for fn in os.listdir(_dir_fp)):
        os.remove(p) if os.path.isfile(p) else (shutil.rmtree(p) if os.path.isdir(p) else None)


class PathCleaner(object):
    """
    A callable that generates the absolute path of the given path and checks
    that it exists if indicated as preexisting.

    :param label: A user-friendly label for the path (used for error reporting)
    :type label: str

    :param preexists: Flag whether to raise an error if the path
        does not exist.
    :type preexists: bool
    """
    def __init__(self, label, preexists=True):
        self.label = label
        self.preexists = preexists

    def __call__(self, path):
        return as_path(path, self.label, preexists=self.preexists)


@contextmanager
def setcwd(path):
    pwd = os.getcwd()
    os.chdir(str(path))
    yield path
    os.chdir(pwd)
