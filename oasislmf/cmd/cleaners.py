import os

from oasislmf.utils.exceptions import OasisException


def as_path(value, name, preexists=True):
    """
    Processes the path and returns the absolute path.

    If the path does not exist and ``preexists`` is true
    an ``OasisException`` is raised.

    :param value: The path to process
    :type value: str

    :param name: The name of the path (used for error reporting)
    :type name: str

    :param preexists: Flag whether to raise an error if the path
        does not exist.
    :type preexists: bool

    :return: The absolute path of the input path
    """
    if value is not None:
        value = os.path.abspath(value) if not os.path.isabs(value) else value

    if preexists and not (value is not None and os.path.exists(value)):
        raise OasisException('{} does not exist: {}'.format(name, value))

    return value


class PathCleaner(object):
    """
    A callable that generates the absolute path of the input and
    checks if it exists if required

    :param name: The name of the path (used for error reporting)
    :type name: str

    :param preexists: Flag whether to raise an error if the path
        does not exist.
    :type preexists: bool
    """
    def __init__(self, name, preexists=True):
        self.name = name
        self.preexists = preexists

    def __call__(self, value):
        return as_path(value, self.name, preexists=self.preexists)
