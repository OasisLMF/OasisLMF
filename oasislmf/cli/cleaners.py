import os

from future.utils import string_types

from ..utils.exceptions import OasisException


def as_path(path, label, preexists=True):
    """
    Processes the path and returns the absolute path.

    If the path does not exist and ``preexists`` is true
    an ``OasisException`` is raised.

    :param path: The path to process
    :type path: str

    :param label: The label of the path (used for error reporting)
    :type label: str

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

    return _path


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
