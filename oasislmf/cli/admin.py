__all__ = [
    'AdminCmd',
    'CreateComplexModelRepoCmd',
    'CreateSimpleModelRepoCmd',
]

import os
import re
import subprocess

from argparse import RawDescriptionHelpFormatter
from subprocess import (
    CalledProcessError,
    run
)

from ..utils.exceptions import OasisException
from ..utils.path import (
    as_path,
    empty_dir,
)
from ..utils.data import get_utctimestamp

from .base import (
    InputValues,
    OasisBaseCommand,
)


class CreateSimpleModelRepoCmd(OasisBaseCommand):
    """
    Creates a local Git repository for a "simple model" (using the
    ``cookiecutter`` package) on a "simple model" repository template
    on GitHub)
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

    def action(self, args):
        """
        Generates and writes an Rtree file index of peril area IDs (area peril IDs)
        and area polygon bounds from a peril areas (area peril) file.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        cmd_str = 'cookiecutter git+{https,ssh}://git@github.com/OasisLMF/CookiecutterOasisSimpleModel'

        try:
            run(cmd_str.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout 
        except CalledProcessError as e:
            self.logger.error(e)


class CreateComplexModelRepoCmd(OasisBaseCommand):
    """
    Creates a local Git repository for a "complex model" (using the
    ``cookiecutter`` package) on a "complex model" repository template
    on GitHub)
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        # parser.add_argument(
        #)

    def action(self, args):
        """
        Generates and writes an Rtree file index of peril area IDs (area peril IDs)
        and area polygon bounds from a peril areas (area peril) file.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

class AdminCmd(OasisBaseCommand):
    """
    Admin subcommands::

        * creates a local Git repository for a "simple model" (using the
          ``cookiecutter`` package) on a "simple model" repository template
          on GitHub)
        * creates a local Git repository for a "complex model" (using the
          ``cookiecutter`` package) on a "complex model" repository template
          on GitHub)
    """
    sub_commands = {
        'create-simple-model-repo': CreateSimpleModelRepoCmd,
        'create-complex-model-repo': CreateComplexModelRepoCmd
    }
