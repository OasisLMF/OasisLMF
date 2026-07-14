from argparse import RawDescriptionHelpFormatter
from .command import OasisBaseCommand, OasisComputationCommand


class ServerInfoApiCmd(OasisComputationCommand):
    """
    Print version/info details of the connected Oasis Platform API server
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformServerInfo'


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


class PostApiCmd(OasisComputationCommand):
    """
    Upload file(s) to the Platform API
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformPost'


class ValidateApiCmd(OasisComputationCommand):
    """
    Validate (or fetch the validation status of) a portfolio's OED exposure files
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformValidate'


class ExposureRunApiCmd(OasisComputationCommand):
    """
    Run `oasislmf exposure run` on the server against a portfolio's exposure files
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformExposureRun'


class ExposureTransformApiCmd(OasisComputationCommand):
    """
    Convert a portfolio's exposure data between OED and AIR
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformExposureTransform'


class CombineApiCmd(OasisComputationCommand):
    """
    Combine the ORD output of multiple RUN_COMPLETED analyses
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformCombine'


class CancelApiCmd(OasisComputationCommand):
    """
    Cancel a running analysis (input generation or execution)
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformCancel'


class ReconnectApiCmd(OasisComputationCommand):
    """
    Reconnect to an in-progress (or finished) analysis and resume polling for status
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformReconnect'


class SubTasksApiCmd(OasisComputationCommand):
    """
    List the sub-tasks of an analysis run
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformSubTasks'


class PlotApiCmd(OasisComputationCommand):
    """
    Plot a Gantt chart of an analysis's sub-tasks, with a status summary
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PlatformPlot'


class ApiCmd(OasisBaseCommand):
    sub_commands = {
        'server-info': ServerInfoApiCmd,
        'list': ListApiCmd,
        'run': RunApiCmd,
        'generate-oasis-files': RunInputApiCmd,
        'generate-losses': RunLossApiCmd,
        'delete': DeleteApiCmd,
        'get': GetApiCmd,
        'post': PostApiCmd,
        'validate': ValidateApiCmd,
        'exposure-run': ExposureRunApiCmd,
        'exposure-transform': ExposureTransformApiCmd,
        'combine': CombineApiCmd,
        'cancel': CancelApiCmd,
        'sub-tasks': SubTasksApiCmd,
        'plot': PlotApiCmd,
        'reconnect': ReconnectApiCmd,
    }
