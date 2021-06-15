__all__ = [
    'as_path',
    'empty_dir',
    'PathCleaner',
    'import_from_string',
    'get_custom_module',
    'setcwd'
]

import os
import sys
import importlib
import re
import shutil

from contextlib import contextmanager

from .exceptions import OasisException


def as_path(path, label, is_dir=False, preexists=True, null_is_valid=True):
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

    :param null_is_valid: flag to indicate if None is a valid value
    :type null_is_valid: bool

    :return: The absolute path of the input path
    """
    if path is None and null_is_valid:
        return

    if not isinstance(path, str):
        if preexists:
            raise OasisException(f'The path {path} ({label}) is indicated as preexisting but is not a valid path')
        else:
            return
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if preexists and not os.path.exists(path):
        raise OasisException(f'The path {path} ({label}) is indicated as preexisting but does not exist')
    if is_dir and preexists and not os.path.isdir(path):
        raise OasisException(f'The path {path} ({label}) is indicated as a preexisting directory but is not actually a directory')

    return os.path.normpath(path)


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


def import_from_string(name):
    """
    return the object or module from the path given
    >>> import os.path
    >>> mod = import_from_string('os.path')
    >>> os.path is mod
    True

    >>> from os.path import isabs
    >>> cls = import_from_string('os.path.isabs')
    >>> isabs is cls
    True
    """
    components = name.split('.')
    res = __import__(components[0])
    for comp in components[1:]:
        res = getattr(res, comp)
    return res


def get_custom_module(custom_module_path, label):
    """
    return the custom module present at the custom_module_path.
    the try loop allow for the custom module to work even if it depends on other module of its package
    by testing recursively for the presence of __init__.py file

    (ex: this module "path" is using from .exceptions import OasisException so it can only be imported as part of the
    utils package => sys.path.insert(0, path_to_utils);importlib.import_module('utils.path'))
    >>> mod = get_custom_module(__file__, "test module")
    >>> mod.__name__.rsplit('.', 1)[-1]
    'path'
    """
    custom_module_path = as_path(custom_module_path, label, preexists=True, null_is_valid=False)

    package_dir = os.path.dirname(custom_module_path)
    module_name = re.sub(r'\.py$', '', os.path.basename(custom_module_path))

    while True:
        sys.path.insert(0, package_dir)
        try:
            custom_module = importlib.import_module(module_name)
            importlib.reload(custom_module)
            return custom_module
        except ImportError:
            if '__init__.py' in os.listdir(package_dir):
                module_name = os.path.basename(package_dir) + '.' + module_name
                package_dir, old_package_dir = os.path.dirname(package_dir), package_dir
                if package_dir == old_package_dir:
                    raise
            else:
                raise
        finally:
            sys.path.pop(0)


@contextmanager
def setcwd(path):
    pwd = os.getcwd()
    os.chdir(str(path))
    yield path
    os.chdir(pwd)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
