OasisLMF
========

Core loss modelling framework.

Development
===========

Dependencies
------------

Dependencies are controlled by ``pip-tools``. To install the development dependencies
first, install ``pip-tools`` using::

    pip install pip-tools

and run::

    pip-sync

To add new dependencies to the development requirements add the package name to ``requirements.in`` or
to add a new dependency to the installed package add the package name to ``requirements-package.in``.
Version specifiers can be supplied to the packages but these should be kept as loose as possible so that
all packages can be easily updated and there will be fewer conflict when installing.

After adding packages to either ``*.in`` file::

    pip-compile && pip-sync

should be ran ensuring the development dependencies are kept up to

Testing
-------

To test the code style run::

    flake8

To test against all supported python versions run::

    tox

To test against your currently installed version of python run::

    py.test

To run the full test suite run::

    ./runtests.sh
