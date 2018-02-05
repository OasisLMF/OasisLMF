import os

from oasislmf.utils.exceptions import OasisException


def as_path(value, name, preexists=True):
    if value is not None:
        value = os.path.abspath(value)

    if preexists and not (value is not None and os.path.exists(value)):
        raise OasisException('{} does not exist: {}'.format(name, value))

    return value


class PathCleaner(object):
    def __init__(self, name, preexists=True):
        self.name = name
        self.preexists = preexists

    def __call__(self, value):
        return as_path(value, self.name, preexists=self.preexists)
