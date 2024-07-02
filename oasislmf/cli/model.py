__all__ = [
    'GenerateExposurePreAnalysisCmd',
    'GenerateKeysCmd',
    'GeneratePostFileGenCmd',
    'GeneratePrelossCmd',
    'GenerateLossesCmd',
    'GenerateLossesPartialCmd',
    'GenerateLossesOutputCmd',
    'GenerateOasisFilesCmd',
    'ModelCmd',
    'RunCmd'
]

from argparse import RawDescriptionHelpFormatter

from .command import OasisBaseCommand, OasisComputationCommand


class GenerateExposurePreAnalysisCmd(OasisComputationCommand):
    """
    Generate a new EOD from original one by specifying a model specific pre-analysis hook for exposure modification
    see ExposurePreAnalysis for more detail
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'ExposurePreAnalysis'


class GeneratePostFileGenCmd(OasisComputationCommand):
    """
    Generate a new EOD from original one by specifying a model specific pre-analysis hook for exposure modification
    see ExposurePreAnalysis for more detail
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PostFileGen'


class GeneratePrelossCmd(OasisComputationCommand):
    """
    Generate a new EOD from original one by specifying a model specific pre-analysis hook for exposure modification
    see ExposurePreAnalysis for more detail
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PreLoss'


class GenerateKeysCmd(OasisComputationCommand):
    """
    Generates keys from a model lookup, and write Oasis keys and keys error files.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateKeys'


class GenerateOasisFilesCmd(OasisComputationCommand):
    """
    Generates the standard Oasis GUL input files + optionally the IL/FM input
    files and the RI input files.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateOasisFiles'


class GenerateLossesCmd(OasisComputationCommand):
    """
    Generates the standard Oasis GUL input files + optionally the IL/FM input
    files and the RI input files.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateOasisLosses'


class GenerateLossesPartialCmd(OasisComputationCommand):
    """
    Distributed Oasis CMD: desc todo
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateLossesPartial'


class GenerateLossesOutputCmd(OasisComputationCommand):
    """
    Distributed Oasis CMD: desc todo
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateLossesOutput'


class RunCmd(OasisComputationCommand):
    """
    Run models end to end.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'RunModel'


class RunPostAnalysisCmd(OasisComputationCommand):
    """
    Run the output postprocessing step.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PostAnalysis'


class ModelCmd(OasisBaseCommand):
    """
    Model subcommands::

        * generating keys files from model lookups
        * generating Oasis input CSV files (GUL [+ IL, RI])
        * generating losses from a preexisting set of Oasis input CSV files
        * generating deterministic losses (no model)
        * running a model end-to-end
    """
    sub_commands = {
        'generate-exposure-pre-analysis': GenerateExposurePreAnalysisCmd,
        'generate-keys': GenerateKeysCmd,
        'generate-oasis-files': GenerateOasisFilesCmd,
        'generate-post-file-gen': GeneratePostFileGenCmd,
        'generate-pre-loss': GeneratePrelossCmd,
        'generate-losses': GenerateLossesCmd,
        'generate-losses-chunk': GenerateLossesPartialCmd,
        'generate-losses-output': GenerateLossesOutputCmd,
        'run': RunCmd,
        'run-postanalysis': RunPostAnalysisCmd,
    }
