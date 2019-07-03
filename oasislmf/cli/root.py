from ..utils.exceptions import OasisException

from .admin import AdminCmd
from .config import ConfigCmd
from .exposure import ExposureCmd
from .model import ModelCmd
from .base import OasisBaseCommand
from .test import TestCmd
from .version import VersionCmd
from .api import ApiCmd


class RootCmd(OasisBaseCommand):
    """
    Tool for manageing oasislmf models
    """
    sub_commands = {
        'admin': AdminCmd,
        'test': TestCmd,
        'version': VersionCmd,
        'model': ModelCmd,
        'exposure': ExposureCmd,
        'api': ApiCmd,
        'config': ConfigCmd
    }

    def run(self, args=None):
        """
        Runs the command passing in the parsed arguments. If an ``OasisException`` is
        raised the exception is caught, the error is logged and the process exits with
        an error code of 1.

        :param args: The arguments to run the command with. If ``None`` the arguments
            are gathered from the argument parser. This is automatically set when calling
            sub commands and in most cases should not be set for the root command.
        :type args: Namespace

        :return: The status code of the action (0 on success)
        """
        try:
            return super(OasisBaseCommand, self).run(args=args)
        except OasisException as e:
            if self.log_verbose:
                # Log with traceback
                self.logger.exception(str(e))
            else:
                self.logger.error(str(e))
            return 1
