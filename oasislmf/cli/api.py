from argparse import RawDescriptionHelpFormatter
from .command import OasisBaseCommand, OasisComputationCommand

class ListApiCmd(OasisComputationCommand):
    """
    Issue API GET requests via the command line
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformList'


class RunApiCmd(OasisComputationCommand):
    """
    Run a model via the Oasis Platoform API end to end 
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformRun'

class RunInputApiCmd(OasisComputationCommand):
    """
    Run a model via the Oasis Platoform API end to end 
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformRunInputs'

class RunLossApiCmd(OasisComputationCommand):
    """
    Run a model via the Oasis Platoform API end to end 
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformRunLosses'


class DeleteApiCmd(OasisComputationCommand):
    """
    Delete items from the Platform API
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformDelete'


class GetApiCmd(OasisComputationCommand):
    """
    Download files from the Oasis Platoform API 
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformGet'


class ApiCmd(OasisBaseCommand):
    sub_commands = {
        'list': ListApiCmd,
        'run': RunApiCmd,
        'generate-oasis-files': RunInputApiCmd,
        'generate-losses': RunLossApiCmd,
        'delete': DeleteApiCmd,
        'get': GetApiCmd,
    }
