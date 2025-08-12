import logging
import os
from typing import Optional, Dict, Any, Union, List


class OasisLogConfig:
    """
    Configuration handler for OasisLMF CLI/console logging.

    Handles log level resolution, format management, and validation for CLI-based logging.
    Designed to work alongside existing file-based logging in log.py.

    Environment Variables:
        OASISLMF_LOG_LEVEL: Override log level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

    Example:
        >>> # Basic usage with config file
        >>> config = OasisLogConfig({'logging': {'level': 'INFO', 'format': 'compact'}})
        >>> formatter = config.create_formatter()
        >>> level = config.get_log_level()

        >>> # CLI override example
        >>> level = config.get_log_level('DEBUG')  # Override config file
        >>> available_formats = config.get_available_formats()
        >>> available_levels = config.get_available_levels()
    """

    # Standard levels for validation and help text
    STANDARD_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    FORMAT_TEMPLATES = {
        "simple": "%(message)s",
        "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "detailed": "%(asctime)s - %(processName)s-%(process)d - %(name)s - %(levelname)s - %(message)s",
        "iso_timestamp": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "production": "%(asctime)s [%(process)d] %(name)s - %(levelname)s - %(message)s",
        "compact": "%(asctime)s [%(levelname)s] %(message)s",
    }

    # Date format configurations for different templates
    DATE_FORMATS = {
        "iso_timestamp": "%Y-%m-%dT%H:%M:%S",
        "standard": "%Y-%m-%d %H:%M:%S",
        "detailed": "%Y-%m-%d %H:%M:%S",
        "production": "%Y-%m-%d %H:%M:%S",
        "compact": "%H:%M:%S",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logging configuration manager.

        Args:
            config: Configuration dictionary (typically from JSON file loaded by command.py)
        """
        self.config = config or {}

    def get_log_level(
        self, cli_level: Optional[str] = None, is_verbose: bool = False
    ) -> int:
        """
        Get effective log level from various sources.

        Priority order: CLI args > env vars > config file > verbose flag > default

        Args:
            cli_level: Log level from command line (e.g., 'INFO', 'DEBUG')
            is_verbose: Legacy verbose flag for backward compatibility

        Returns:
            Numeric log level (e.g., 20 for INFO, 10 for DEBUG)

        Examples:
            >>> config = OasisLogConfig()
            >>> config.get_log_level('DEBUG')
            10
            >>> config.get_log_level(is_verbose=True)
            10
        """
        # 1. CLI argument takes highest priority
        if cli_level:
            return self._parse_level(cli_level)

        # 2. Environment variable
        env_level = os.environ.get("OASISLMF_LOG_LEVEL")
        if env_level:
            return self._parse_level(env_level)

        # 3. Config file
        config_level = self.config.get("logging", {}).get("level")
        if config_level:
            return self._parse_level(config_level)

        # 4. Verbose flag (backward compatibility)
        if is_verbose:
            return logging.DEBUG

        # 5. Default
        return logging.INFO

    def get_ods_tools_level(self, main_level: int) -> int:
        """
        Get ods_tools logger level based on main logger level.

        Args:
            main_level: Main oasislmf logger level to use as reference

        Returns:
            Numeric log level for ods_tools logger
        """
        # Check if explicitly configured
        config_level = self.config.get("logging", {}).get("ods_tools_level")
        if config_level:
            return self._parse_level(config_level)

        # Default behavior: WARNING unless main level is DEBUG
        return logging.DEBUG if main_level <= logging.DEBUG else logging.WARNING

    def get_format_string(self, format_name: Optional[str] = None) -> str:
        """
        Get log format string.

        Args:
            format_name: Format template name from CLI or None for config/default

        Returns:
            Format string for logging.Formatter
        """
        # CLI format name takes priority
        if format_name and format_name in self.FORMAT_TEMPLATES:
            return self.FORMAT_TEMPLATES[format_name]

        # Check config file
        config_format = self.config.get("logging", {}).get("format")
        if config_format:
            if config_format in self.FORMAT_TEMPLATES:
                return self.FORMAT_TEMPLATES[config_format]
            else:
                # Custom format string from config
                return config_format

        # Default: standard format
        return self.FORMAT_TEMPLATES["standard"]

    def get_date_format(self, format_name: Optional[str] = None) -> Optional[str]:
        """
        Get date format string for the specified template.

        Args:
            format_name: Format template name

        Returns:
            Date format string or None for logging default
        """
        # Check CLI format name first
        if format_name and format_name in self.DATE_FORMATS:
            return self.DATE_FORMATS[format_name]

        # Check config format
        config_format = self.config.get("logging", {}).get("format")
        if config_format and config_format in self.DATE_FORMATS:
            return self.DATE_FORMATS[config_format]

        # Default: None (uses logging module default)
        return None

    def create_formatter(self, format_name: Optional[str] = None) -> logging.Formatter:
        """
        Create a logging formatter with appropriate format and date format.

        Args:
            format_name: Format template name from CLI

        Returns:
            Configured logging.Formatter instance
        """
        format_str = self.get_format_string(format_name)
        date_format = self.get_date_format(format_name)

        return logging.Formatter(format_str, datefmt=date_format)

    def get_available_formats(self) -> List[str]:
        """
        Get list of available format template names.

        Returns:
            List of format template names that can be used with get_format_string()

        Example:
            >>> config = OasisLogConfig()
            >>> formats = config.get_available_formats()
            >>> 'standard' in formats
            True
        """
        return list(self.FORMAT_TEMPLATES.keys())

    def get_available_levels(self) -> List[str]:
        """
        Get list of available log level names.

        Returns:
            List of standard log level names that can be used with get_log_level()

        Example:
            >>> config = OasisLogConfig()
            >>> levels = config.get_available_levels()
            >>> 'DEBUG' in levels
            True
        """
        return self.STANDARD_LEVELS.copy()

    def validate_config(self) -> List[str]:
        """
        Validate logging configuration and return any issues.

        Returns:
            List of warning messages about configuration issues
        """
        warnings = []
        logging_config = self.config.get("logging", {})

        # Validate main log level
        level = logging_config.get("level")
        if level:
            try:
                self._parse_level(level)
            except ValueError as e:
                warnings.append(f"Invalid log level in config: {e}")

        # Validate ods_tools level
        ods_level = logging_config.get("ods_tools_level")
        if ods_level:
            try:
                self._parse_level(ods_level)
            except ValueError as e:
                warnings.append(f"Invalid ods_tools log level in config: {e}")

        # Validate format (lenient - custom format strings are allowed)
        format_name = logging_config.get("format")
        if format_name and format_name not in self.FORMAT_TEMPLATES:
            # Only warn if it doesn't look like a format string
            if not ("%(message)s" in str(format_name) or "%(" in str(format_name)):
                available = ", ".join(self.get_available_formats())
                warnings.append(
                    f"Format '{format_name}' may not be a valid format template. "
                    f"Available templates: {available}"
                )

        return warnings

    def _parse_level(self, level: Union[str, int, None]) -> int:
        """
        Parse log level using Python's logging module directly.

        Args:
            level: Log level as string ('DEBUG'), integer (10), or None

        Returns:
            Numeric log level

        Raises:
            ValueError: If level is invalid
        """
        if level is None:
            return logging.INFO

        if isinstance(level, str):
            # Use logging module's own level names
            numeric_level = getattr(logging, level.upper(), None)
            if numeric_level is None:
                available = ", ".join(self.get_available_levels())
                raise ValueError(
                    f"Invalid log level: '{level}'. " f"Available levels: {available}"
                )
            return numeric_level

        elif isinstance(level, int):
            # Validate integer is in reasonable range
            if 0 <= level <= 50:  # Standard logging range
                return level
            else:
                raise ValueError(
                    f"Invalid log level: {level}. Use 0-50 or standard level names."
                )

        else:
            raise ValueError(f"Log level must be string or integer, got {type(level)}")
