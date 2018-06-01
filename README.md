<img src="https://oasislmf.org/packages/oasis_theme_package/themes/oasis_theme/assets/src/oasis-lmf-colour.png" alt="Oasis LMF logo" width="250"/>

[![PyPI version](https://badge.fury.io/py/oasislmf.svg)](https://badge.fury.io/py/oasislmf)

# OasisLMF

Core loss modelling framework, deployed as a PYPL Python package.
The repository provides a Python toolkit for building, running and testing Oasis models end-to-end. 
It includes:
-  an api client for interacting with the api server (in the ``api_client`` submodule)
-  a Python class framework for working with Oasis models and model resources as Python objects (the ``models`` submodule)
-  a Python class framework for managing model exposures and resources, and also for generating Oasis files from these (the ``exposures`` submodule)
-  a Python factory class for instantiating keys lookup services for models, and generating and saving keys outputs from these lookup services (the ``keys`` submodule)
-  a command line interface for creating and testing models. App optionscan be found by running ``oasiscli --help``

## Installation

### Dependencies

#### Linux

 * **Debian**: g++ compiler build-essential, libtool, zlib1g-dev autoconf on debian distros
 * **Red Hat**: 'Development Tools' and zlib-devel

#### Windows

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

#### Install With ``pip``

The latest released version of the package can be installed using pip
by running::

    pip install oasislmf

Alternatively you can install the latest development version using::

    pip install git+https://git@github.com/OasisLMF/OasisLMF

(over HTTPS) or::

    pip install git+ssh://git@github.com/OasisLMF/OasisLMF

(over SSH).

You can also install from a specific branch ``<branch name>`` using::

    pip install -e {git+ssh,git+https}://git@github.com/OasisLMF/OasisLMF.git@<branch name>#egg=oasislmf

## Development

### Dependencies

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

### Model keys lookup dependencies

When using the Oasis model development kit (MDK) subcommands::

    oasislmf model {generate-keys, generate-oasis-files, generate-losses, run}

for a specific model you may first need to install the Python requirements specific to the
keys lookup service for that model, e.g. Pandas and Shapely for the `PiWind keys lookup <https://github.com/OasisLMF/OasisPiWind/blob/master/src/keys_server/PiWind/requirements.txt>`_, otherwise you will encounter command failures.

## Testing

To test the code style run::

    flake8

To test against all supported python versions run::

    tox

To test against your currently installed version of python run::

    py.test

To run the full test suite run::

    ./runtests.sh

## Publishing

Before publishing the latest version of the package make you sure increment the ``__version__`` value in ``oasislmf/__init__.py``, and commit the change. You'll also need to install the ``twine`` Python package which ``setuptools`` uses for publishing packages on PyPI. If publishing wheels then you'll also need to install the ``wheel`` Python package.

### Using the ``publish`` subcommand in ``setup.py``

The distribution format can be either a source distribution or a platform-specific wheel. To publish the source distribution package run::

    python setup.py publish --sdist

or to publish the platform specific wheel run::

    python setup.py publish --wheel

### Manually publishing, with a GPG signature

The first step is to create the distribution package with the desired format: for the source distribution run::

    python setup.py sdist

which will create a ``.tar.gz`` file in the ``dist`` subfolder, or for the platform specific wheel run::

    python setup.py bdist_wheel

which will create ``.whl`` file in the ``dist`` subfolder. To attach a GPG signature using your default private key you can then run::

    gpg --detach-sign -a dist/<package file name>.{tar.gz,whl}

This will create ``.asc`` signature file named ``<package file name>.{tar.gz,whl}.asc`` in ``dist``. You can just publish the package with the signature using::

    twine upload dist/<package file name>.{tar.gz,whl} dist/<package file name>.{tar.gz,whl}.asc
    
## Documentation
* <a href="https://github.com/OasisLMF/OasisLMF/issues">Issues</a>
* <a href="https://github.com/OasisLMF/OasisLMF/releases">Releases</a>
* <a href="https://oasislmf.github.io">General Oasis documentation</a>
* <a href="http://localhost:8000/html/docs/oasis_cli.html">OasisLMF CLI</a>
* <a href="http://localhost:8000/html/docs/oasis_cli.html">Model Developerment Kit (MDK)</a>
* <a href="http://localhost:8000/html/OasisLmf/modules.html">Modules</a>

## License
The code in this project is licensed under BSD 3-clause license.
    
    
    
