from .. import __version__
from .base import OasisBaseCommand


class VersionCmd(OasisBaseCommand):
    """
    Prints the CLI version
    """

    def action(self, args):
        """
        Prints the version number to the console.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        print(__version__)
