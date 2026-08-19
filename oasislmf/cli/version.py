from .. import __version__
from .command import OasisBaseCommand


class VersionCmd(OasisBaseCommand):
    """Prints the installed package version"""

    def action(self, args):
        """Prints the version number to the console.

        Args:
            args (Namespace): The arguments from the command line
        """
        print(__version__)
