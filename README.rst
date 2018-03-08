OasisLMF
========

Core loss modelling framework.

Installation
============

Dependencies
------------

-----
Linux
-----

* **Debian**: g++ compiler build-essential, libtool, zlib1g-dev autoconf on debian distros
* **Red Hat**: 'Development Tools' and zlib-devel

-------
Windows
-------

Cygwin 64-bit is required for the Windows native build.  Cygwin is a Linux environment running in Windows.
http://www.cygwin.com/

Download and run the set-up program for Cygwin.
The following Cygwin add-in packages are required;

* gcc-g++
* gcc-core
* make
* diffutils
* automake
* libtools
* zlib-devel
* git


To build native Windows 64-bit executables;

* mingw64-x86_64-gcc-g++
* mingw64-x86_64-gcc-core
* mingw64-x86_64-zlib

Search for 'mingw', gcc', 'make' and 'diffutils' to find all of the relevant packages (Only 'gcc' illustrated below).
![alt text](docs/img/cygwin1.jpg "Add-in packages")

Install With Pip
----------------

The latest released version of the package can be installed using pip
by running::

    pip install oasislmf

Alternatively you can install the latest development version using::

    pip install git+https://git@github.com/OasisLMF/OasisLMF

(over HTTPS) or::

    pip install git+ssh://git@github.com/OasisLMF/OasisLMF

(over SSH).

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

Model keys lookup dependencies
------------------------------

When using the Oasis model development kit (MDK) subcommands::

    oasislmf model {generate-keys, generate-oasis-files, generate-losses, run}

for a specific model (e.g. PiWind) you may first need to install the Python requirements specific to the
keys lookup service for that model, e.g. Pandas and Shapely for the `PiWind keys lookup <https://github.com/OasisLMF/OasisPiWind/blob/master/src/keys_server/PiWind/requirements.txt>`_, otherwise you will encounter command failures.

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
