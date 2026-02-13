__all__ = [
    'GenerateExposurePreAnalysisCmd',
    'GenerateKeysCmd',
    'GeneratePostFileGenCmd',
    'GeneratePrelossCmd',
    'GenerateLossesCmd',
    'GenerateLossesPartialCmd',
    'GenerateLossesOutputCmd',
    'GenerateOasisFilesCmd',
    'GenerateDocCmd',
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
    Run a post file generation hook to modify or expand the generated Oasis files
    before loss calculation.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'PostFileGen'


class GeneratePrelossCmd(OasisComputationCommand):
    """
    Run a pre-loss hook on each worker to modify input files before loss calculation.
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
    Run loss calculations with optional pre-loss and post-analysis hooks.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateOasisLosses'


class GenerateLossesPartialCmd(OasisComputationCommand):
    """
    Runs a single analysis event chunk.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateLossesPartial'


class GenerateLossesOutputCmd(OasisComputationCommand):
    """
    Runs the output reports generation on a set of event chunks.
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateLossesOutput'


class GenerateDocumentationCmd(OasisComputationCommand):
    """
    Generate Documentation for model from the config file

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateDocumentation'


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


class GenerateComputationSettingsJsonSchema(OasisComputationCommand):
    """
    Generate a json schema to validate the computation settings part of oed settings
    """
    formatter_class = RawDescriptionHelpFormatter
    computation_name = 'GenerateComputationSettingsJsonSchema'


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
        'generate-doc': GenerateDocumentationCmd,
        'generate-computation-settings-json-schema': GenerateComputationSettingsJsonSchema,
        'run': RunCmd,
        'run-postanalysis': RunPostAnalysisCmd,
    }
