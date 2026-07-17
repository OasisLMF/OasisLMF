OasisLMF Package
================

On this page:
-------------

* :ref:`intro_package`
* :ref:`features_package`
* :ref:`requirements_package`
* :ref:`installation_package`
* :ref:`bash_enable_package`
* :ref:`dependencies_package`
* :ref:`testing_package`
* :ref:`links_package`

|

.. _intro_package:

Introduction
------------

----

The ``oasislmf`` Python package, loosely called the model development kit (MDK) or the MDK package, provides a command line 
toolkit for developing, testing and running Oasis models end-to-end locally, or remotely via the Oasis API. It can generate 
ground-up losses (GUL), direct/insured losses (IL) and reinsurance losses (RIL). It can also generate deterministic losses 
at all these levels.

|

.. _features_package:

Features
********

----

For running models locally the CLI provides a ``model`` subcommand with the following options:

* ``model generate-exposure-pre-analysis``: generate new Exposure input using user custom code (e.g. geo-coding, exposure
  enhancement, or disaggregation).
* ``model generate-keys``: generates Oasis keys files from model lookups; these are essentially line items of (location ID,
  peril ID, coverage type ID, area peril ID, vulnerability ID) where peril ID and coverage type ID span the full set of
  perils and coverage types that the model supports; if the lookup is for a complex/custom model the keys file will have
  the same format except that area peril ID and vulnerability ID are replaced by a model data JSON string.
  Keys can be output in ``oasis``, ``json``, or ``parquet`` format via the ``--keys-format`` flag.
* ``model generate-oasis-files``: generates the Oasis input files for losses (GUL, GUL + IL, or GUL + IL + RIL); it
  requires the provision of source exposure and optionally source accounts and reinsurance info and scope files (in OED
  format), as well as assets for instantiating model lookups and generating keys files.
* ``model generate-pre-loss``: runs pre-loss hooks before the main loss calculation. Custom code can be injected via
  ``--pre-loss-module`` / ``--pre-loss-class-name``.
* ``model generate-post-file-gen``: runs post-file-generation hooks after Oasis input files are created but before losses
  are computed. Custom code injected via ``--post-file-gen-module`` / ``--post-file-gen-class-name``.
* ``model generate-losses``: generates losses (GUL, or GUL + IL, or GUL + IL + RIL) from a set of pre-existing Oasis files.
* ``model generate-losses-chunk``: generates losses for a single chunk (used internally by the platform worker).
* ``model generate-losses-output``: post-processes and collects output from chunked loss generation.
* ``model run``: runs the model from start to finish — exposure pre-analysis → keys → Oasis files → losses — from the
  source OED exposure, and optionally source accounts and reinsurance info and scope files.
* ``model run-postanalysis``: runs the post-analysis hook on a completed set of results without re-running the full model.
* ``model generate-doc``: prints the analysis settings JSON schema documentation.
* ``model generate-computation-settings-json-schema``: outputs the computation settings JSON schema for tooling.

|

The optional ``--summarise-exposure`` flag can be issued with ``model generate-oasis-files`` and ``model run`` to generate
a summary of Total Insured Values (TIVs) grouped by coverage type and peril. This produces the
``exposure_summary_report.json`` file.

For remote model execution the ``api`` subcommand provides the following subcommands:

* ``api run``: runs the model remotely (same as ``model run``) but via the Oasis API
* ``api generate-oasis-files``: remotely generates Oasis input files via the API
* ``api generate-losses``: remotely generates losses via the API
* ``api list``: lists analyses available on the remote API server
* ``api get``: retrieves results from a remote analysis
* ``api delete``: deletes a remote analysis

See :doc:`/how-to/api-client` for a full guide, including authentication options, a step-by-step
workflow, and advice on diagnosing platform-specific failures.


For generating deterministic losses an ``exposure run`` subcommand is available:

* ``exposure run``: generates deterministic losses (GUL, or GUL + IL, or GUL + IL + RIL)

For utility and maintenance:

* ``warmup``: pre-compiles all Numba JIT functions to eliminate cold-start overhead on the first model run. Recommended
  after installation — especially in Docker images — to avoid a 2–6 minute compilation delay on first use.
* ``config``: describes the format of the MDK configuration JSON file.
* ``config update``: updates a config JSON file with new values.
* ``version``: prints the installed oasislmf version.
* ``admin enable-bash-complete``: activates bash tab-completion (see :ref:`bash_enable_package`).
* ``test``: runs a regression test against an expected set of outputs.

|

The reusable libraries are organised into several sub-packages, the most relevant of which from a model developer or user's
perspective are:

* ``api_client``
* ``model_preparation``
* ``model_execution``
* ``utils``

|

.. _requirements_package:

Minimum Python Requirements
***************************

-----

Starting from 1st January 2019, Pandas will no longer be supporting Python 2. As Pandas is a key dependency of the MDK we 
are **dropping Python 2 (2.7) support** as of this release (1.3.4). The last version which still supports Python 2.7 is 
version ``1.3.3`` (published 12/03/2019).

Also for this release (and all future releases) a **minimum of Python 3.10 is required**.

|

.. _installation_package:

Installation
************

----

The latest released version of the package, or a specific package version, can be installed using ``pip``:

.. code-block::

    pip install oasislmf
    pip install oasislmf==<version string>

|

Alternatively you can install the latest development version using:

.. code-block::

    pip install git+{https,ssh}://git@github.com/OasisLMF/OasisLMF

|

You can also install from a specific branch ``<branch name>`` using:

.. code-block::

    pip install [-v] git+{https,ssh}://git@github.com/OasisLMF/OasisLMF.git@<branch name>#egg=oasislmf

|

macOS Apple Silicon (M1/M2/M3/M4)
##################################

OasisLMF installs natively on Apple Silicon Macs via ``pip install oasislmf``. Ensure you have:

* **Python 3.10+** — the system Python on macOS is 3.9; install a newer version via ``brew install python@3.12`` or ``pyenv``.
* **macOS 12 (Monterey) or later** — required for scipy ARM64 wheels.

For optional geospatial extras (``pip install oasislmf[extra]``), also install:

.. code-block::

    brew install spatialindex geos

|

JIT Cache Warmup
################

OasisLMF uses Numba JIT compilation for performance-critical calculations. The first run after installation incurs a
one-time compilation overhead of 2–6 minutes. Pre-compile all JIT functions to eliminate this delay:

.. code-block::

    oasislmf warmup

In Docker images, you can bake the cache in at build time:

.. code-block::

    RUN pip install oasislmf && oasislmf warmup

.. note::
    JIT caches are CPU-architecture-specific. ``oasislmf warmup`` in a Docker image is most effective when the build
    machine and the runtime machine share the same CPU architecture.

|

.. _bash_enable_package:

Enable Bash completion
**********************

----

Bash completion is a functionality which bash helps users type their commands by presenting possible options when users 
press the tab key while typing a command.

Once oasislmf is installed you'll need to be activate the feature by sourcing a bash file (only needs to be run once).

|

Local
#####

.. code-block::

    oasislmf admin enable-bash-complete

|

Global
######

.. code-block::

    echo 'complete -C completer_oasislmf oasislmf' | sudo tee /usr/share/bash-completion/completions/oasislmf

|

.. _dependencies_package:

Dependencies
************

----

System
######

The package provides a built-in lookup framework (``oasislmf.model_preparation.lookup.OasisLookup``) which uses the Rtree 
Python package, which in turn requires the ``libspatialindex`` spatial indexing C library.

https://libspatialindex.github.io/index.html

|

Linux users can install the development version of ``libspatialindex`` from the command line using ``apt``.

.. code-block::

    [sudo] apt install -y libspatialindex-dev

|

and OS X users can do the same via ``brew``.

.. code-block::

    brew install spatialindex

|

The PiWind demonstration model uses the built-in lookup framework, therefore running PiWind or any model which uses the 
built-in lookup, requires that you install ``libspatialindex``.

|

**GNU/Linux**

For GNU/Linux the following is a specific list of required system libraries

* **Debian**: g++ compiler build-essential, libtool, zlib1g-dev autoconf on debian distros

.. code-block::

    sudo apt install g++ build-essential libtool zlib1g-dev autoconf

* **Red Hat**: 'Development Tools' and zlib-devel

|

Python
######

Package Python dependencies are controlled by ``pip-tools``. To install the development dependencies first, install 
``pip-tools`` using:

.. code-block::

    pip install pip-tools

and run:

.. code-block::

    pip-sync

|

To add new dependencies to the development requirements add the package name to ``requirements.in`` or to add a new 
dependency to the installed package add the package name to ``requirements-package.in``. Version specifiers can be supplied 
to the packages but these should be kept as loose as possible so thatall packages can be easily updated and there will be 
fewer conflict when installing.

|

After adding packages to either ``*.in`` file:

.. code-block::

    pip-compile && pip-sync

This should be ran ensuring the development dependencies are kept up to date.

|

ods_tools
#########

OasisLMF uses the ods_tools package to read exposure files and the settings files. The compatible version for each
OasisLMF release is managed in the requirements files. Below is the current summary:

* OasisLMF 1.23.x or before => no ods_tools
* OasisLMF 1.26.x => use ods_tools 2.3.2
* OasisLMF 1.27.0 => use ods_tools 3.0.0 or later
* OasisLMF 1.27.1 => use ods_tools 3.0.0 or later
* OasisLMF 1.27.2 => use ods_tools 3.0.4 or later
* OasisLMF 2.3.x => use ods_tools 3.2.x or later
* OasisLMF 2.4.x => use ods_tools 4.0.x or later
* OasisLMF 2.5.x => use ods_tools 5.0.x or later

|

pandas
######

Pandas has released its major version number 2 breaking some of the compatibility with the 1st version. Therefore, for all 
version of OasisLMF ``<= 1.27.2``, the latest supported version for pandas is ``1.5.3``. Support for pandas 2, starts from 
version ``1.27.3``.

|

.. _testing_package:

Testing
*******

----

To test the code style run:

.. code-block::

    flake8

|

To test against all supported python versions run:

.. code-block::

    tox

|

To test against your currently installed version of python run:

.. code-block::

    py.test

|

To run the full test suite run:

.. code-block::

    ./runtests.sh

|

Publishing
**********

----

Version management and PyPI releases are handled automatically by the CI pipeline (``version.yml`` and ``publish.yml``
workflows). Manually publishing is not normally required.

To build and upload manually (using modern ``build`` + ``twine``):

.. code-block::

    pip install build twine
    python -m build
    twine upload dist/*

The ``__version__`` value in ``oasislmf/__init__.py`` must be incremented before building.

|

.. _links_package:

Links for further information
*****************************

* the release notes
* :doc:`/how-to/model-development-kit`
* `OasisLMF Github repository <https://github.com/OasisLMF/OasisLMF>`_