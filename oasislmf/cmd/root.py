from ..utils.exceptions import OasisException
from .base import OasisBaseCommand
from .test import TestCmd
from .version import VersionCmd
from .bin import BinCmd


class RootCmd(OasisBaseCommand):
    sub_commands = {
        'test': TestCmd,
        'version': VersionCmd,
        'bin': BinCmd,
    }

    def run(self, args=None):
        try:
            return super(OasisBaseCommand, self).run(args=args)
        except OasisException as e:
            self.logger.error(str(e))
            return 1
