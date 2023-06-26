====
Command Line Interface (oasislmf)
====

The package comes with a command line interface for creating, testing and managing models.
The tool is split into several namespaces that group similar commands. For a full list of
namespaces use ``oasislmf --help``, and ``oasislmf <namespace> --help`` for a full list of
commands available in each namespace.

**bin**
=======

``oasislmf bin build``
----------------------

.. autocli:: oasislmf.cmd.bin.BuildCmd
   :noindex:

``oasislmf bin check``
----------------------

.. autocli:: oasislmf.cmd.bin.CheckCmd
   :noindex:

``oasislmf bin clean``
----------------------

.. autocli:: oasislmf.cmd.bin.CleanCmd
   :noindex:

**config**
==========


``oasislmf model config``
---

.. autocli:: oasislmf.cli.config.ConfigCmd
   :noindex:

**model**
=========

``oasislmf model generate-exposure-pre-analysis``
=================================================
.. autocli:: oasislmf.cli.model.GenerateExposurePreAnalysisCmd
   :noindex:

``oasislmf model generate-keys``
================================
.. autocli:: oasislmf.cli.model.GenerateKeysCmd
   :noindex:

``oasislmf model generate-oasis-files``
=======================================
.. autocli:: oasislmf.cli.model.GenerateOasisFilesCmd
   :noindex:

``oasislmf model generate-losses``
==================================
.. autocli:: oasislmf.cli.model.GenerateLossesCmd
   :noindex:

``oasislmf model generate-losses-chunk``
========================================
.. autocli:: oasislmf.cli.model.GenerateLossesPartialCmd
   :noindex:

``oasislmf model generate-losses-output``
=========================================
.. autocli:: oasislmf.cli.model.GenerateLossesOutputCmd
   :noindex:

``oasislmf model run``
======================
.. autocli:: oasislmf.cli.model.RunCmd
   :noindex:

**test**
========

``oasislmf test gen-model-tester-dockerfile``
---------------------------------------------

.. autocli:: oasislmf.cmd.test.GenerateModelTesterDockerFileCmd
   :noindex:

``oasislmf bin model-api``
--------------------------

.. autocli:: oasislmf.cmd.test.TestModelApiCmd
   :noindex:

version
=======

.. autocli:: oasislmf.cmd.version.VersionCmd
   :noindex:


