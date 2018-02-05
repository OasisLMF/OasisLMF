=================================
Command Line Interface (oasislmf)
=================================

The package comes with a command line interface for creating, testing and managing models.
The tool is split into several namespaces that group similar commands. For a full list of
namespaces use ``oasislmf --help``, and ``oasislmf <namespace> --help`` for a full list of
commands available in each namespace.

bin
===

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

model
=====

``oasislmf model generate-keys``
================================


.. autocli:: oasislmf.cmd.model.GenerateKeysCmd
   :noindex:

``oasislmf model generate-losses``
==================================


.. autocli:: oasislmf.cmd.model.GenerateLossesCmd
   :noindex:

``oasislmf model generate-oasis-files``
=======================================


.. autocli:: oasislmf.cmd.model.GenerateOasisFilesCmd
   :noindex:

``oasislmf model run``
======================


.. autocli:: oasislmf.cmd.model.RunCmd
   :noindex:

test
====

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
