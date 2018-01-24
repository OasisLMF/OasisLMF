import os

from oasislmf.utils.exceptions import OasisException


class PathCleaner(object):
    def __init__(self, name, required=True):
        self.name = name
        self.required = required

    def __call__(self, value):
        if value is not None:
            value = os.path.abspath(value)

        if self.required and not (value is not None and os.path.exists(value)):
            raise OasisException('{} does not exist: {}'.format(self.name, value))

        return value
