__all__ = [
    'AdminCmd',
    'EnableBashCompleteCmd',
]


from argparse import RawDescriptionHelpFormatter
from .command import OasisBaseCommand, OasisComputationCommand


class EnableBashCompleteCmd(OasisComputationCommand):
    """
    Adds required command to `.bashrc` Linux or .bash_profile for mac
    so that Command autocomplete works for oasislmf CLI
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'HelperTabComplete'


class AdminCmd(OasisBaseCommand):
    """
    Admin subcommands::

    """
    sub_commands = {
        'enable-bash-complete': EnableBashCompleteCmd
    }
