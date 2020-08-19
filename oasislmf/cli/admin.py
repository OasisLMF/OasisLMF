__all__ = [
    'AdminCmd',
    'CreateComplexModelCmd',
    'CreateSimpleModelCmd',
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


class CreateSimpleModelCmd(OasisComputationCommand):
    """
    Creates a local Git repository for a "simple model" (using the
    ``cookiecutter`` package) on a "simple model" repository template
    on GitHub)
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'CreateModelRepo' 


class CreateComplexModelCmd(OasisComputationCommand):
    """
    Creates a local Git repository for a "complex model" (using the
    ``cookiecutter`` package) on a "simple model" repository template
    on GitHub)
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'CreateComplexModelRepo' 


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
        'create-simple-model': CreateSimpleModelCmd,
        'create-complex-model': CreateComplexModelCmd,
        'enable-bash-complete': EnableBashCompleteCmd
    }
