Logging Configuration
=====================

Enhanced logging configuration system for OasisLMF CLI and console output.

.. versionadded:: 2.4.6
   Configurable logging levels and enhanced formatting support.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

**What's New**

OasisLMF now supports configurable logging levels and enhanced output formatting, 
replacing the previous binary INFO/DEBUG system with full user control.

**Key Benefits**

* Configurable log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
* Multiple format templates with timestamps and process information  
* Configuration via CLI arguments, environment variables, and JSON config files
* Full backward compatibility with existing ``--verbose`` flag

**Performance Considerations**

* DEBUG level may impact performance in high-throughput scenarios
* File logging (if configured) has minimal performance overhead  
* Format templates have negligible performance differences
* Process ID inclusion in detailed formats adds minimal overhead

**Related Configuration**

See :doc:`options_config_file` for general configuration options and model-specific settings.

Quick Start
-----------

**Basic Usage**

.. code-block:: bash

   # Set log level
   oasislmf --log-level=DEBUG model run
   
   # Set format template  
   oasislmf --log-format=compact model run
   
   # Combine both
   oasislmf --log-level=WARNING --log-format=detailed model run

**Environment Variable**

.. code-block:: bash

   export OASISLMF_LOG_LEVEL=DEBUG
   oasislmf model run

**Configuration File**

Add to your ``oasislmf.json``:

.. code-block:: json

   {
     "logging": {
       "level": "INFO",
       "format": "standard"
     }
   }

CLI Arguments
-------------

Enhanced logging arguments available for all commands:

``--log-level``
   Set logging level. 
   
   **Choices:** DEBUG, INFO, WARNING, ERROR, CRITICAL
   
   **Default:** INFO
   
   .. versionadded:: 2.4.6
   
   **Examples:**
   
   .. code-block:: bash
   
      oasislmf --log-level=DEBUG model run
      oasislmf --log-level=ERROR model run

``--log-format``  
   Set log format template.
   
   **Choices:** simple, standard, detailed, iso_timestamp, production, compact
   
   **Default:** standard
   
   .. versionadded:: 2.4.6
   
   **Examples:**
   
   .. code-block:: bash
   
      oasislmf --log-format=compact model run
      oasislmf --log-format=detailed model run

``--verbose`` *(deprecated)*
   Use verbose logging. 
   
   .. deprecated:: 2.4.6
      Use ``--log-level=DEBUG`` instead.
   
   **Note:** Will show deprecation warning when used.

Configuration File
------------------

Add logging configuration to your JSON config file (``oasislmf.json`` by default):

.. code-block:: json

   {
     "logging": {
       "level": "INFO",
       "format": "standard",
       "ods_tools_level": "WARNING"
     }
   }

**Configuration Options**

``level``
   Main logging level. Accepts string names (DEBUG, INFO, WARNING, ERROR, CRITICAL) 
   or numeric values (10, 20, 30, 40, 50).

``format``  
   Format template name (simple, standard, detailed, iso_timestamp, production, compact) 
   or custom format string.

``ods_tools_level``
   Separate level for ods_tools logger. If not specified, defaults to WARNING 
   unless main level is DEBUG.

**Custom Format Strings**

You can specify custom format strings in the configuration file:

.. code-block:: json

   {
     "logging": {
       "format": "%(asctime)s [%(process)d] %(levelname)s: %(message)s"
     }
   }

See :doc:`options_config_file` for more configuration file examples and general settings.

Environment Variables
---------------------

``OASISLMF_LOG_LEVEL``
   Override log level for all commands.
   
   .. versionadded:: 2.4.6
   
   **Examples:**
   
   .. code-block:: bash
   
      export OASISLMF_LOG_LEVEL=ERROR
      export OASISLMF_LOG_LEVEL=DEBUG
      export OASISLMF_LOG_LEVEL=20  # INFO level

Format Templates
----------------

Available format templates with examples:

**simple**
   Basic message-only output. Fastest performance.
   
   .. code-block:: text
   
      Starting model execution...
      Model completed successfully

**standard** *(default)*
   Timestamp, logger name, level, and message.
   
   .. code-block:: text
   
      2024-01-15 10:30:45 - oasislmf.model - INFO - Starting model execution...
      2024-01-15 10:35:22 - oasislmf.model - INFO - Model completed successfully

**compact**
   Compact format with time and level. Good for development.
   
   .. code-block:: text
   
      10:30:45 [INFO] Starting model execution...
      10:35:22 [INFO] Model completed successfully

**detailed**
   Full details including process name and ID. Best for debugging.
   
   .. code-block:: text
   
      2024-01-15 10:30:45 - MainProcess-12345 - oasislmf.model - INFO - Starting model execution...
      2024-01-15 10:35:22 - MainProcess-12345 - oasislmf.model - INFO - Model completed successfully

**iso_timestamp**
   ISO 8601 timestamp format. Good for log parsing.
   
   .. code-block:: text
   
      2024-01-15T10:30:45 - oasislmf.model - INFO - Starting model execution...
      2024-01-15T10:35:22 - oasislmf.model - INFO - Model completed successfully

**production**
   Production-ready format with process ID. Recommended for production.
   
   .. code-block:: text
   
      2024-01-15 10:30:45 [12345] oasislmf.model - INFO - Starting model execution...
      2024-01-15 10:35:22 [12345] oasislmf.model - INFO - Model completed successfully

Configuration Priority
-----------------------

The system uses this priority order (highest to lowest):

1. **CLI arguments** (``--log-level``, ``--log-format``)
2. **Environment variables** (``OASISLMF_LOG_LEVEL``)  
3. **Configuration file** (``logging`` section)
4. **Legacy verbose flag** (``--verbose``)
5. **Default values** (INFO level, standard format)

**Example Priority Resolution**

.. code-block:: bash

   # Config file has level: "WARNING"
   # Environment has OASISLMF_LOG_LEVEL=INFO  
   # CLI argument --log-level=DEBUG
   # Result: DEBUG (CLI takes precedence)

Migration Guide
---------------

**From --verbose to New System**

.. list-table::
   :header-rows: 1
   :widths: 40 40 20
   
   * - Old Command
     - New Command
     - Notes
   * - ``oasislmf --verbose model run``
     - ``oasislmf --log-level=DEBUG model run``
     - Recommended migration
   * - ``oasislmf model run``
     - ``oasislmf --log-level=INFO model run``
     - Default behavior (no change needed)
   * - ``oasislmf --verbose --config=my.json model run``
     - ``oasislmf --log-level=DEBUG --config=my.json model run``
     - CLI args override config

**Batch Migration**

For scripts using ``--verbose``, you can:

1. **Immediate:** Continue using ``--verbose`` (shows deprecation warning)
2. **Recommended:** Replace with ``--log-level=DEBUG``
3. **Long-term:** Move to configuration file for consistency

**Performance Migration Notes**

* DEBUG level in production may impact performance - consider INFO or WARNING
* For high-throughput scenarios, use ERROR or CRITICAL levels
* Simple format offers best performance if log content is less important

Troubleshooting
---------------

**Common Issues**

*Configuration file not loading*
   * Check file path and JSON syntax
   * Warnings are displayed for invalid configurations
   * Use ``--log-level=DEBUG`` to see config loading details

*Log level not changing*  
   * Verify configuration priority (CLI > env > config > verbose > default)
   * Check for typos in level names
   * Environment variables override config file settings

*Format not applied*
   * Ensure format name is spelled correctly
   * Use ``--help`` to see available options
   * Custom format strings must be valid Python logging format strings

*Performance issues with DEBUG level*
   * Consider using INFO level for production workloads
   * Use WARNING or ERROR for high-throughput scenarios
   * Profile your specific use case to determine optimal level

*Deprecation warnings appearing*
   * Replace ``--verbose`` with ``--log-level=DEBUG``
   * Update scripts and documentation

**Debug Information**

Enable debug mode to see effective configuration:

.. code-block:: bash

   oasislmf --log-level=DEBUG --log-format=detailed model run

Debug output will show:

.. code-block:: text

   DEBUG - oasislmf - Effective log level: DEBUG
   DEBUG - oasislmf - ods_tools level: DEBUG  
   DEBUG - oasislmf - Config source: ./oasislmf.json

Advanced Usage
--------------

**Programmatic Configuration**

For Python scripts using OasisLMF as a library:

.. code-block:: python

   from oasislmf.utils.log_config import OasisLogConfig
   
   # Create configuration
   config = {'logging': {'level': 'DEBUG', 'format': 'compact'}}
   log_config = OasisLogConfig(config)
   
   # Get effective settings
   level = log_config.get_log_level()
   formatter = log_config.create_formatter()
   
   # Validate configuration
   warnings = log_config.validate_config()
   for warning in warnings:
       print(f"Warning: {warning}")

**Available API Methods**

.. code-block:: python

   # Get available options
   formats = log_config.get_available_formats()
   levels = log_config.get_available_levels()
   
   # Parse levels
   numeric_level = log_config._parse_level('DEBUG')  # Returns 10
   
   # Create formatter with specific template
   formatter = log_config.create_formatter('production')

**Integration with Custom Loggers**

.. code-block:: python

   import logging
   from oasislmf.utils.log_config import OasisLogConfig
   
   # Setup custom logger with OasisLMF configuration
   log_config = OasisLogConfig({'logging': {'level': 'INFO', 'format': 'detailed'}})
   
   custom_logger = logging.getLogger('my_app')
   custom_logger.setLevel(log_config.get_log_level())
   
   handler = logging.StreamHandler()
   handler.setFormatter(log_config.create_formatter())
   custom_logger.addHandler(handler)

See Also
--------

* :doc:`options_config_file` - General configuration file options
* :doc:`building-and-running-models` - Model execution commands
* :doc:`installation` - Installation and setup guide

**Related CLI Commands**

* ``oasislmf model run`` - Run complete model with logging
* ``oasislmf config`` - Configuration management  
* ``oasislmf --help`` - Show all available options