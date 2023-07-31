.. Oasis LMF documentation master file, created by
   sphinx-quickstart on Thu Feb  1 12:22:30 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Oasis Loss Modelling Framework
==============================

The repository provides a Python toolkit for building, running and
testing Oasis models end-to-end, including performing individual steps
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
-  a command line interface for creating and testing models. App options
   can be found by running ``oasiscli --help``

.. mdinclude:: ../README.md


test

.. toctree::
   :titlesonly:
   :caption: Use Cases:

   model-developer
   model-users
   installing-deploying-Oasis
   support

.. toctree::
   :maxdepth: 0
   :caption: Sections:

   installation
   sphinx
   building-and-running-models
   cli
   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
