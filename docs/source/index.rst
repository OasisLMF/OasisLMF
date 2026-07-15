.. Oasis LMF documentation master file.

`oasislmf` |version| — Oasis Loss Modelling Framework
=====================================================

This repository provides a Python toolkit (the Model Development Kit) for
building, running and testing Oasis LMF models end-to-end. It includes an API
client, a class framework for working with models and exposures, a keys/lookup
factory, and a command line interface (run ``oasislmf --help``).

The documentation is organised using the `Diátaxis <https://diataxis.fr/>`_
framework — **Tutorials**, **How-to guides**, **Reference** and **Explanation** —
so each page has one clear job. See ``DOCS_STRATEGY.md`` in the repository root
for the documentation strategy this structure is part of.

Start where you fit
-------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 🧭 New here?
      :link: tutorials/index
      :link-type: doc

      Learn by doing. Run a first analysis end-to-end.
      **Tutorials.**

   .. grid-item-card:: 🛠️ Model developer
      :link: building-and-running-models
      :link-type: doc

      Build, run and test models with the MDK.
      **How-to guides.**

   .. grid-item-card:: 📖 Need a fact?
      :link: reference/index
      :link-type: doc

      CLI options, environment variables and the Python API,
      generated from source. **Reference.**

   .. grid-item-card:: 💡 Want the why?
      :link: explanation/index
      :link-type: doc

      How the Financial Module and the rest of the framework work,
      and why. **Explanation.**


.. toctree::
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :caption: How-to guides
   :hidden:

   installation
   building-and-running-models
   logging-configuration

.. toctree::
   :caption: Reference
   :hidden:

   reference/index

.. toctree::
   :caption: Explanation
   :hidden:

   explanation/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
