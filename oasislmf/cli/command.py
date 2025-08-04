__all__ = [
    "OasisBaseCommand",
    "OasisComputationCommand",
]

import logging
import os
import sys

from argparsetree import BaseCommand
from ods_tools.oed.settings import Settings, ROOT_USER_ROLE

from ..utils.path import PathCleaner
from ..utils.inputs import InputValues

from ..manager import OasisManager as om

from ..utils.log_config import OasisLogConfig
import warnings
from typing import Dict, Any
import json


class OasisBaseCommand(BaseCommand):
    """
    The base command to inherit from for each command.

    2 additional arguments (``--verbose`` and ``--config``) are added to
    the parser so that they are available for all commands.
    """

    def __init__(self, *args, **kwargs):
        self._logger = None
        self.args = None
        self.log_verbose = False
        super(OasisBaseCommand, self).__init__(*args, **kwargs)

    def add_args(self, parser):
        """
        Adds arguments to the argument parser. This is used to modify
        which arguments are processed by the command.

        Enhanced logging arguments (--log-level, --log-format) added.
        Legacy --verbose flag maintained for backward compatibility.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        # Create temporary log config instance for dynamic choices
        log_config = OasisLogConfig()

        # Legacy verbose flag (backward compatibility with deprecation notice)
        parser.add_argument(
            "-V",
            "--verbose",
            action="store_true",
            help="Use verbose logging. (Deprecated: use --log-level=DEBUG)",
        )

        # Enhanced logging arguments
        parser.add_argument(
            "-L",
            "--log-level",
            choices=log_config.get_available_levels(),
            help="Set logging level (default: INFO)",
        )

        parser.add_argument(
            "--log-format",
            choices=log_config.get_available_formats(),
            help="Set log format template (default: standard)",
        )

        # Configuration file argument
        parser.add_argument(
            "-C",
            "--config",
            required=False,
            type=PathCleaner("MDK config. JSON file", preexists=True),
            help="MDK config. JSON file",
            default="./oasislmf.json" if os.path.isfile("./oasislmf.json") else None,
        )

    def parse_args(self):
        """
        Parses the command line arguments and sets them in ``self.args``

        :return: The arguments taken from the command line
        """
        try:
            self.args = super(OasisBaseCommand, self).parse_args()

            # Handle backward compatibility with deprecation warning
            if self.args.verbose:
                warnings.warn(
                    "The --verbose flag is deprecated and will be removed in a future version. "
                    "Use --log-level=DEBUG instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            self.setup_logger()
            return self.args
        except Exception:
            self.setup_logger()
            raise

    def _load_config_dict(self) -> Dict[str, Any]:
        """
        Load configuration dictionary from file if available.

        Returns:
            Configuration dictionary or empty dict if loading fails
        """
        if not (
            hasattr(self, "args")
            and self.args
            and hasattr(self.args, "config")
            and self.args.config
        ):
            return {}

        try:
            with open(self.args.config, "r") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(
                f"Warning: Config file not found: {self.args.config}", file=sys.stderr
            )
            return {}
        except json.JSONDecodeError as e:
            print(
                f"Warning: Invalid JSON in config file {self.args.config}: {e}",
                file=sys.stderr,
            )
            return {}
        except Exception as e:
            print(
                f"Warning: Could not load config file {self.args.config}: {e}",
                file=sys.stderr,
            )
            return {}

    def setup_logger(self):
        """
        Setup logger using OasisLogConfig for enhanced logging configuration.

        Supports configurable log levels, formats, and maintains backward compatibility.
        """
        if not self._logger:
            # Load configuration and create log config handler
            config_dict = self._load_config_dict()
            log_config = OasisLogConfig(config_dict)

            # Validate configuration and show warnings
            warnings_list = log_config.validate_config()
            for warning in warnings_list:
                print(f"Warning: {warning}", file=sys.stderr)

            # Get effective log level and formatter
            cli_level = (
                getattr(self.args, "log_level", None)
                if hasattr(self, "args") and self.args
                else None
            )
            is_verbose = (
                getattr(self.args, "verbose", False)
                if hasattr(self, "args") and self.args
                else False
            )
            cli_format = (
                getattr(self.args, "log_format", None)
                if hasattr(self, "args") and self.args
                else None
            )

            log_level = log_config.get_log_level(cli_level, is_verbose)
            formatter = log_config.create_formatter(cli_format)
            ods_level = log_config.get_ods_tools_level(log_level)

            # Setup main oasislmf logger
            logger = logging.getLogger("oasislmf")

            # Remove existing handlers (preserve existing logic)
            for handler in list(logger.handlers):
                if handler.name == "oasislmf":
                    logger.removeHandler(handler)
                    break

            # Setup ods_tools logger
            ods_logger = logging.getLogger("ods_tools")
            ods_logger.setLevel(ods_level)
            ods_logger.propagate = False

            # Create and configure handler
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.name = "oasislmf"
            ch.setFormatter(formatter)

            # Add handler to both loggers
            logger.addHandler(ch)
            ods_logger.addHandler(ch)
            logger.setLevel(log_level)

            # Set the logger and preserve backward compatibility
            self._logger = logger
            self.log_verbose = log_level <= logging.DEBUG

            # Add debug info when running in debug mode
            if log_level <= logging.DEBUG:
                config_source = (
                    getattr(self.args, "config", None)
                    if hasattr(self, "args") and self.args
                    else None
                )
                self._logger.debug(
                    f"Effective log level: {logging.getLevelName(log_level)}"
                )
                self._logger.debug(
                    f"ods_tools level: {logging.getLevelName(ods_level)}"
                )
                self._logger.debug(
                    f"Config source: {config_source if config_source else 'default'}"
                )

    @property
    def logger(self):
        if self._logger:
            return self._logger

        return logging.getLogger("oasislmf")


class OasisComputationCommand(OasisBaseCommand):
    """
    Eventually, the Parent class for all Oasis Computation Command
    create the command line interface from parameter define in the associated computation step
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super().add_args(parser)

        for param in om.computations_params[self.computation_name]:
            add_argument_kwargs = {
                key: param.get(key)
                for key in [
                    "action",
                    "nargs",
                    "const",
                    "type",
                    "choices",
                    "help",
                    "metavar",
                    "dest",
                ]
                if param.get(key) is not None
            }
            # If 'Help' is not set then this is a function only paramter, skip
            if "help" in add_argument_kwargs:
                arg_name = f"--{param['name'].replace('_', '-')}"
                if param.get("flag"):
                    parser.add_argument(
                        param.get("flag"), arg_name, **add_argument_kwargs
                    )
                else:
                    parser.add_argument(arg_name, **add_argument_kwargs)

    @classmethod
    def get_arguments(cls, args, manager_method):
        inputs = InputValues(args)

        def get_kwargs_item(param):
            return param["name"], inputs.get(
                param["name"],
                required=param.get("required"),
                is_path=param.get("is_path"),
                dtype=param.get("type"),
            )

        settings_args = {
            param["name"] for param in manager_method.get_params(param_type="settings")
        }

        _kwargs = dict(
            get_kwargs_item(param)
            for param in manager_method.get_params()
            if param["name"] in settings_args
        )

        # read and merge computation settings files
        computation_settings = Settings()
        computation_settings.add_settings(inputs.config, ROOT_USER_ROLE)
        for settings_info in manager_method.get_params(param_type="settings"):
            setting_fp = _kwargs.get(settings_info["name"])
            if setting_fp:
                new_settings = settings_info["loader"](setting_fp)
                computation_settings.add_settings(
                    new_settings.pop("computation_settings", {}),
                    settings_info.get("user_role"),
                )
        inputs.config = computation_settings.get_settings()

        return {
            **dict(get_kwargs_item(param) for param in manager_method.get_params()),
            **_kwargs,
        }

    def action(self, args):
        """
        Generic method that call the correct manager method from the child class computation_name

        :param args: The arguments from the command line
        :type args: Namespace
        """
        manager_method = getattr(
            om(), om.computation_name_to_method(self.computation_name)
        )
        _kwargs = self.get_arguments(args, manager_method)

        self.logger.info(f"\nStarting oasislmf command - {self.computation_name}")
        manager_method(**_kwargs)
