from argparse import RawDescriptionHelpFormatter

from .base import OasisBaseCommand


class ConfigCmd(OasisBaseCommand):
    """
    The ``oasislmf`` tool can be configured using using
    a json configuration file. By default this is stored
    in ``oasislmf.json`` but the path can be changed by
    specifying the ``--config`` flag.

    This configuration file should contain any or all of
    the following properties in no particular order
    ::

        keys_data_path
        model_data_path
        model_version_file_path
        model_exposures_file_path
        model_run_dir_path
        output_file_path
        output_format
        lookup_package_path
        analysis_settings_json_file_path
        ktools_script_name
        ktools_num_processes
        no_execute
        canonical_exposures_profile_json_path
        canonical_exposures_validation_file_path
        canonical_to_model_exposures_transformation_file_path
        source_exposures_file_path
        source_exposures_validation_file_path
        source_to_canonical_exposures_transformation_file_path

    The path-related keys should be string paths, given relative
    to the location of JSON file.

    As an example, this is the master script configuration file for PiWind

    ::

        {
            "keys_data_path": "OasisPiWind/keys_data/PiWind",
            "model_version_file_path": "OasisPiWind/keys_data/PiWind/ModelVersion.csv",
            "lookup_package_path": "OasisPiWind/src/keys_server",
            "canonical_exposures_profile_json_path": "OasisPiWind/oasislmf-piwind-canonical-profile.json",
            "source_exposures_file_path": "OasisPiWind/tests/data/SourceLocPiWind.csv",
            "source_exposures_validation_file_path": "OasisPiWind/flamingo/PiWind/Files/ValidationFiles/Generic_Windstorm_SourceLoc.xsd",
            "source_to_canonical_exposures_transformation_file_path": "OasisPiWind/flamingo/PiWind/Files/TransformationFiles/MappingMapToGeneric_Windstorm_CanLoc_A.xslt",
            "canonical_exposures_validation_file_path": "OasisPiWind/flamingo/PiWind/Files/ValidationFiles/Generic_Windstorm_CanLoc_B.xsd",
            "canonical_to_model_exposures_transformation_file_path": "OasisPiWind/flamingo/PiWind/Files/TransformationFiles/MappingMapTopiwind_modelloc.xslt",
            "xtrans_path": "omdk/xtrans/xtrans.exe",
            "oasis_files_path": "omdk/runs",
            "analysis_settings_json_file_path": "OasisPiWind/analysis_settings.json",
            "model_data_path": "OasisPiWind/model_data/PiWind",
            "model_run_dir_path": "omdk/runs"
        }

    It can also be obtained from
    `https://github.com/OasisLMF/OasisPiWind/blob/master/mdk-oasislmf-piwind.json <https://github.com/OasisLMF/OasisPiWind/blob/master/mdk-oasislmf-piwind.json>`_.
    """
    formatter_class = RawDescriptionHelpFormatter
