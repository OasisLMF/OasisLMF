API Client CLI
==============

On this page:
-------------

* :ref:`why_api_client`
* :ref:`prerequisites_api_client`
* :ref:`auth_api_client`
* :ref:`commands_api_client`
* :ref:`workflow_api_client`
* :ref:`config_api_client`

|

.. _why_api_client:

Why use the API client?
-----------------------

----

The ``oasislmf model run`` command runs an analysis entirely in-process. The Oasis Platform,
by contrast, serialises intermediate data to files between stages — for example, writing
generated input files to shared storage before the losses stage reads them back.

This difference means a model can pass ``oasislmf model run`` but fail on the platform
when a serialisation or file-format issue is exposed. The ``oasislmf api`` subcommand runs
an analysis through a live Platform instance using its real worker containers, so it is a
faithful test of how the model will behave in a deployed environment.

Using the API client is the recommended way for model developers to validate a model against
a worker container before distributing it to users.

|

.. _prerequisites_api_client:

Prerequisites
-------------

----

* A running Oasis Platform instance accessible over HTTP. The
  `OasisEvaluation <https://github.com/OasisLMF/OasisEvaluation>`_ repository provides a
  Docker Compose configuration that brings up the full stack locally.
* The ``oasislmf`` package installed (``pip install oasislmf``).
* OED exposure files and an ``analysis_settings.json`` for the model you want to test.

When using the OasisEvaluation compose file the server is available at
``http://localhost:8000`` with the default credentials ``admin`` / ``password``. The
``oasislmf api`` commands will try these defaults automatically, so no extra auth
configuration is needed for a local evaluation deployment.

|

.. _auth_api_client:

Authentication
--------------

----

Three authentication modes are supported, selected via ``--auth-type``.

.. list-table::
   :widths: 15 30 55
   :header-rows: 1

   * - Mode
     - ``--auth-type``
     - Credentials required
   * - Simple JWT
     - ``simple``
     - ``username`` and ``password``
   * - OIDC via platform
     - ``oidc``
     - ``client_id`` and ``client_secret`` (token exchange handled by the platform)
   * - M2M direct to IdP
     - ``m2m``
     - ``client_id``, ``client_secret``, and ``--oidc-token-url``; optionally ``--oidc-scope``

Credentials can be supplied in three ways:

**1. Credentials JSON file** (recommended for scripting)

Create a JSON file and pass it with ``--server-login-json``.

For ``simple`` auth:

.. code-block:: json

    {
        "username": "admin",
        "password": "password"
    }

For ``oidc`` or ``m2m`` auth:

.. code-block:: json

    {
        "client_id": "my-client",
        "client_secret": "my-secret"
    }

For ``m2m`` with a token URL baked in:

.. code-block:: json

    {
        "client_id": "my-client",
        "client_secret": "my-secret",
        "token_url": "https://idp.example.com/oauth2/token",
        "scope": "oasis/m2m"
    }

**2. MDK config file** (``-C oasislmf.json``)

Any ``api`` option can be stored in the MDK config file so you do not have to repeat it on every command:

.. code-block:: json

    {
        "server_url": "http://localhost:8000",
        "server_version": "v2",
        "server_login_json": "./credentials.json"
    }

**3. Interactive prompt**

If no credentials file is provided and the default local credentials fail, the CLI
prompts for the auth type and then the required fields.

|

.. _commands_api_client:

Commands
--------

----

All ``api`` subcommands share a common set of connection options:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Option
     - Description
   * - ``--server-url URL``
     - Base URL of the platform (default: ``http://localhost:8000``)
   * - ``--server-version {v1,v2}``
     - ``v1`` = single-server run mode; ``v2`` = distributed/chunked run mode (default: ``v2``)
   * - ``--server-login-json FILE``
     - Path to a JSON file containing credentials (see :ref:`auth_api_client`)
   * - ``--auth-type {simple,oidc,m2m}``
     - Force a specific authentication mode
   * - ``-C FILE``
     - MDK config JSON file; all options can be stored here

|

``api run``
***********

Runs the model end-to-end: uploads exposure files, triggers input generation, triggers losses
generation, and downloads results.

.. code-block:: bash

    oasislmf api run \
        --server-url http://localhost:8000 \
        --server-version v1 \
        -x location.csv \
        -y accounts.csv \
        -a analysis_settings.json \
        -o ./results/

Key options:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Option
     - Description
   * - ``-x / --oed-location-csv``
     - OED location file
   * - ``-y / --oed-accounts-csv``
     - OED accounts file (optional)
   * - ``-i / --oed-info-csv``
     - Reinsurance info file (optional)
   * - ``-s / --oed-scope-csv``
     - Reinsurance scope file (optional)
   * - ``-a / --analysis-settings-json``
     - Analysis settings JSON
   * - ``-o / --output-dir``
     - Directory to write results to
   * - ``--model-id ID``
     - Use an existing model (skips model upload)
   * - ``--portfolio-id ID``
     - Use an existing portfolio (skips exposure upload)
   * - ``--analysis-id ID``
     - Re-use an existing analysis object
   * - ``--analysis-chunks N``
     - Number of loss-generation chunks for ``v2`` runs
   * - ``--lookup-chunks N``
     - Number of input-generation chunks for ``v2`` runs

|

``api generate-oasis-files``
****************************

Runs only the input-generation stage (keys lookup + Oasis file preparation) on the platform.
Useful for isolating lookup or exposure-pre-analysis issues.

.. code-block:: bash

    oasislmf api generate-oasis-files \
        --server-url http://localhost:8000 \
        --model-id 1 \
        -x location.csv \
        -a analysis_settings.json

|

``api generate-losses``
***********************

Runs only the losses-generation stage for an analysis whose inputs have already been
generated. Requires an existing ``--analysis-id``.

.. code-block:: bash

    oasislmf api generate-losses \
        --server-url http://localhost:8000 \
        --analysis-id 42 \
        -o ./results/

|

``api list``
************

Lists models, portfolios, and/or analyses on the server. Pass one or more IDs to print
full detail for those objects; omit IDs to list all.

.. code-block:: bash

    # List everything
    oasislmf api list --server-url http://localhost:8000

    # Show detail for specific objects
    oasislmf api list --server-url http://localhost:8000 \
        --models 1 2 \
        --analyses 10 11

|

``api get``
***********

Downloads files associated with a model, portfolio, or analysis. Useful for retrieving
intermediate files (generated inputs, lookup logs) after a run.

.. code-block:: bash

    # Download outputs for analysis 42
    oasislmf api get --server-url http://localhost:8000 \
        --analyses-output-file 42 \
        -o ./results/

    # Download the traceback log after a failed run
    oasislmf api get --server-url http://localhost:8000 \
        --analyses-run-traceback-file 42 \
        -o ./logs/

Available file types: ``--analyses-output-file``, ``--analyses-input-file``,
``--analyses-run-traceback-file``, ``--analyses-run-log-file``,
``--analyses-input-generation-traceback-file``, ``--analyses-lookup-errors-file``,
``--analyses-lookup-success-file``, ``--analyses-lookup-validation-file``,
``--analyses-settings-file``, ``--analyses-summary-levels-file``,
``--model-settings``, ``--model-versions``,
``--portfolio-location-file``, ``--portfolio-accounts-file``,
``--portfolio-reinsurance-info-file``, ``--portfolio-reinsurance-scope-file``.

|

``api delete``
**************

Deletes models, portfolios, and/or analyses from the server.

.. code-block:: bash

    oasislmf api delete --server-url http://localhost:8000 \
        --analyses 42 43 \
        --portfolios 5

|

.. _workflow_api_client:

Typical workflow
----------------

----

The following walkthrough shows how a model developer would test a new model against a
local Platform instance.

**Step 1 — start the platform**

.. code-block:: bash

    git clone https://github.com/OasisLMF/OasisEvaluation
    cd OasisEvaluation
    docker compose -f oasis-platform.yml up -d

Wait for the server to be ready (``http://localhost:8000/healthcheck/`` returns ``200``).

**Step 2 — run the model end-to-end**

.. code-block:: bash

    oasislmf api run \
        --server-url http://localhost:8000 \
        --server-version v1 \
        -x /path/to/location.csv \
        -y /path/to/accounts.csv \
        -a /path/to/analysis_settings.json \
        -o ./output/

The command creates a portfolio, uploads the exposure, creates an analysis, triggers input
generation and then losses generation, and finally downloads the results to ``./output/``.

**Step 3 — inspect the results**

.. code-block:: bash

    ls ./output/

**Step 4 — diagnose a failure**

If the run fails, retrieve the traceback:

.. code-block:: bash

    # First find the analysis ID
    oasislmf api list --server-url http://localhost:8000 --analyses

    # Then download the log
    oasislmf api get --server-url http://localhost:8000 \
        --analyses-run-traceback-file <analysis-id> \
        -o ./logs/

**Step 5 — clean up**

.. code-block:: bash

    oasislmf api delete --server-url http://localhost:8000 \
        --analyses <analysis-id> \
        --portfolios <portfolio-id>

|

.. _config_api_client:

Using a config file
-------------------

----

For repeated runs it is more convenient to store connection and file path settings in an
``oasislmf.json`` config file and pass it with ``-C``:

.. code-block:: json

    {
        "server_url": "http://localhost:8000",
        "server_version": "v1",
        "server_login_json": "./credentials.json",
        "oed_location_csv": "location.csv",
        "oed_accounts_csv": "accounts.csv",
        "analysis_settings_json": "analysis_settings.json",
        "output_dir": "./results/"
    }

.. code-block:: bash

    oasislmf api run -C oasislmf.json

Any command-line flag overrides the corresponding key in the config file.

|

.. seealso::

    * :doc:`/reference/OasisLMF-package` — local model run commands (``oasislmf model run``)
    * the platform REST API — the full platform REST API reference
    * `OasisEvaluation <https://github.com/OasisLMF/OasisEvaluation>`_ — Docker Compose deployment for local testing
