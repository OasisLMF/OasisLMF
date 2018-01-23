from oasislmf.utils.exceptions import OasisException
from .base import OasisBaseCommand
from .test import TestCmd


class RootCmd(OasisBaseCommand):
    sub_commands = {
        'test': TestCmd
    }

    def run(self, args=None):
        try:
            return super(OasisBaseCommand, self).run(args=args)
        except OasisException as e:
            self.logger.error(str(e))
            return 1
