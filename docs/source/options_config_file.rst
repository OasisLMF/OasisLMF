Options for the JSON Configuration File
=======================================

This page contains a list of all the options that can be set in the JSON configuration file.
Click on an option on the left to see a description of the option and the default values.

.. toctree::
   :maxdepth: 1
   :hidden:

   generated_options

Logging Configuration
---------------------

.. versionadded:: 2.4.6

Enhanced logging configuration is available with configurable levels and output formats.

**Basic Example:**

.. code-block:: json

   {
     "logging": {
       "level": "INFO",
       "format": "standard",
       "ods_tools_level": "WARNING"
     }
   }

**Available Options:**

* ``level`` - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
* ``format`` - Format template or custom format string  
* ``ods_tools_level`` - Separate level for ods_tools logger

Related Configuration
---------------------

See :doc:`logging-configuration` for detailed logging setup, including:

* CLI arguments (``--log-level``, ``--log-format``)
* Environment variables (``OASISLMF_LOG_LEVEL``)
* Format templates and custom formatting
* Migration from ``--verbose`` flag