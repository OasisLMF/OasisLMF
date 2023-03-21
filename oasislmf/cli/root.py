import sys

from oasislmf.cli.admin import AdminCmd
from oasislmf.cli.api import ApiCmd
from oasislmf.cli.command import OasisBaseCommand
from oasislmf.cli.config import ConfigCmd
from oasislmf.cli.exposure import ExposureCmd
from oasislmf.cli.model import ModelCmd
from oasislmf.cli.test import TestCmd
from oasislmf.cli.version import VersionCmd
from oasislmf.utils.exceptions import OasisException


class RootCmd(OasisBaseCommand):
    """
    Tool for managing oasislmf models.
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


def main():
    """CLI entrypoint for running the whole RootCmd"""
    sys.exit(RootCmd().run())


if __name__ == '__main__':
    main()
