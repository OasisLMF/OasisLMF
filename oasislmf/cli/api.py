from argparse import RawDescriptionHelpFormatter
from .base import OasisBaseCommand
from .computation_command import OasisComputationCommand

class ListApiCmd(OasisComputationCommand):
    """
    Issue API GET requests via the command line
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformGetDetails'


class RunApiCmd(OasisComputationCommand):
    """
    Run a model via the Oasis Platoform API end to end 
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformRun'


class ApiCmd(OasisBaseCommand):
    sub_commands = {
        'list': ListApiCmd,
        'run': RunApiCmd
    }
