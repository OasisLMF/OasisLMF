Environment Variables
=====================

OasisLMF uses environment variables to control runtime behaviour that must be
set before the package is imported. These are distinct from the JSON
configuration file options (see :doc:`options_config_file`).

Numeric Type Configuration
--------------------------

The following variables control the numeric precision of internal data types.
They are read once at import time, so they must be set in the shell before
running any ``oasislmf`` command or importing the package.

``OASIS_AREAPERIL_TYPE``
    Controls the integer type used for AreaPeril IDs.
    Default: ``u4`` (uint32, max value 4,294,967,295).
    Set to ``u8`` (uint64) for models with AreaPeril IDs exceeding this limit.

    .. code-block:: bash

        export OASIS_AREAPERIL_TYPE=u8
        oasislmf model run ...

``OASIS_FLOAT``
    Controls floating-point precision for loss values.
    Default: ``f4`` (float32). Set to ``f8`` (float64) for higher numerical
    precision at the cost of increased memory usage and slower computation.

    .. code-block:: bash

        export OASIS_FLOAT=f8

``OASIS_INT``
    Controls integer precision for internal Oasis fields for example `SummaryId`.
    Default: ``i4`` (int32). Set to ``i8`` (int64) for extended range.

    .. code-block:: bash

        export OASIS_INT=i8

.. note::

   All three variables must be consistent with the binary model data files
   on disk. Changing them without regenerating model data will produce
   incorrect results. See issue `#1990
   <https://github.com/OasisLMF/OasisLMF/issues/1990>`_ for details on
   cache-related pitfalls when changing these values between runs.

Logging
-------

See :doc:`logging-configuration` for the ``OASIS_PACKAGE_LOG_LEVEL`` and
related logging environment variables.
