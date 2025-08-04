.. Oasis LMF documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`oasislmf` |version| - Oasis Loss Modelling Framework
===========================================

This repository provides a Python toolkit for building, running and
testing Oasis LMF models end-to-end, including performing individual steps
in this process. It includes:

-  an api client for interacting with the api server (in the
   ``api_client`` submodule)
-  a Python class framework for working with Oasis models and model
   resources as Python objects (the ``models`` submodule)
-  a Python class framework for managing model exposures and resources,
   and also for generating Oasis files from these (the ``exposures``
   submodule)
-  a Python factory class for instantiating keys lookup services for
   models, and generating and saving keys outputs from these lookup
   services (the ``keys`` submodule)
-  a command line interface (CLI) for creating and testing models. App sub-commands
   can be found by running ``oasislmf --help``. 
   CLI documentation can be found in the API Reference pages, under `oasislmf.cli`. 
   Each sub-command, e.g., `oasislmf model` is described in the corresponding documentation 
   page `oasislmf.model`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   building-and-running-models
   logging-configuration
   options_config_file

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
