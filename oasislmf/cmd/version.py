from .. import __version__
from .base import OasisBaseCommand


class VersionCmd(OasisBaseCommand):
    description = 'Prints the CLI version'

    def action(self, args):
        print(__version__)
