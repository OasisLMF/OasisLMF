__all__ = [
    'ExposureCmd',
    'RunCmd'
]

import os

from argparse import RawDescriptionHelpFormatter

from .command import OasisBaseCommand, OasisComputationCommand


class RunCmd(OasisComputationCommand):
    """
    Generates deterministic losses using the installed ktools framework given
    direct Oasis files (GUL + optionally IL and RI input files).

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    Run ``oasislmf config --help`` for more information.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'RunExposure'


class ExposureCmd(OasisBaseCommand):
    """
    Exposure subcommands::

        * generate - and optionally, validate - deterministic losses (GUL, IL or RIL)
    """
    sub_commands = {
        'run': RunCmd
    }
