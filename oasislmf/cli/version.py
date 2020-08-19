from .. import __version__
from .command import OasisBaseCommand


class VersionCmd(OasisBaseCommand):
    """
    Prints the installed package version
    """

    def action(self, args):
        """
        Prints the version number to the console.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        print(__version__)
