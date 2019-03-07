<img src="https://oasislmf.org/packages/oasis_theme_package/themes/oasis_theme/assets/src/oasis-lmf-colour.png" alt="Oasis LMF logo" width="250"/>

[![PyPI version](https://badge.fury.io/py/oasislmf.svg)](https://badge.fury.io/py/oasislmf)  [![Build Status](https://ci.oasislmfdev.org/buildStatus/icon?job=oasis_oasislmf/master)](https://ci.oasislmfdev.org/job/oasis_oasislmf/job/master/)

# OasisLMF

The `oasislmf` Python package, loosely called the *model development kit (MDK)* or the *MDK package*, provides a command line interface and reusable libraries primarly for developing and running Oasis models end-to-end locally, or remotely via the Oasis API, for the purpose of generating group-up losses (GUL), direct/insured losses (IL) and reinsurance losses (RIL). The package also provides end users with a way to generate deterministic losses at all levels, GUL, IL or RIL.

For running models locally the CLI provides a `model` subcommand with the following main subcommands:

* `model generate-keys`: generates Oasis keys files that model lookups would generate; these are essentially line items of (location ID, peril ID, coverage type ID, area peril ID, vulnerability ID) where peril ID and coverage type ID span the full set of perils and coverage types that the model supports
* `model generate-oasis-files`: generates the Oasis input CSV files for losses (GUL, GUL + IL, or GUL + IL + RIL); it requires the provision of source exposure and optionally source accounts and reinsurance info. and scope files (in OED format), as well as assets for instantiating model lookups and generating keys files
* `model generate-losses`: generates losses (GUL, or GUL + IL, or GUL + IL + RIL) from a set of pre-existing Oasis files
* `model run`: runs the model from start to finish by generating losses (GUL, or GUL + IL, or GUL + IL + RIL) from the source exposure, and optionally source accounts and reinsurance info. and scope files (in OED or RMS format), as well as assets for instantiating model lookups and generating keys files

For remote model execution the `api` subcommand provides the following main subcommand:

* `api run`: runs the model remotely (same as `model run`) but via the Oasis API

For generating deterministic losses (GUL, or GUL + IL, or GUL + IL + RIL) the CLI provides an `exposure run` subcommand.

The reusable libraries are organised into several sub-packages, the most relevant of which from a model developer or user's perspective are:

* `api_client`
* `model_preparation`
* `model_execution`
* `utils`

## Installation

The latest released version of the package can be installed using `pip` (or `pip3` if using Python 3):

    pip install oasislmf

Alternatively you can install the latest development version using:

    pip install git+{https,ssh}://git@github.com/OasisLMF/OasisLMF

You can also install from a specific branch `<branch name>` using:

    pip install [-v] git+{https,ssh}://git@github.com/OasisLMF/OasisLMF.git@<branch name>#egg=oasislmf

## Dependencies

### System

The package provides a built-in lookup framework (`oasislmf.model_preparation.lookup.OasisLookup`) which uses the Rtree Python package, which in turn requires the `libspatialindex` spatial indexing C library.

https://libspatialindex.github.io/index.html

The PiWind demonstration model uses the built-in lookup framework, therefore running PiWind or any model which uses the built-in lookup, requires that you install `libspatialindex`.

#### GNU/Linux

For GNU/Linux the following is a specific list of required system libraries

 * unixodbc unixodbc-dev
 * **Debian**: g++ compiler build-essential, libtool, zlib1g-dev autoconf on debian distros
 * **Red Hat**: 'Development Tools' and zlib-devel

### Python

Package Python dependencies are controlled by `pip-tools`. To install the development dependencies first, install `pip-tools` using:

    pip install pip-tools

and run:

    pip-sync

To add new dependencies to the development requirements add the package name to `requirements.in` or
to add a new dependency to the installed package add the package name to `requirements-package.in`.
Version specifiers can be supplied to the packages but these should be kept as loose as possible so that
all packages can be easily updated and there will be fewer conflict when installing.

After adding packages to either `*.in` file:

    pip-compile && pip-sync

should be ran ensuring the development dependencies are kept up to date.

## Testing

To test the code style run:

    flake8

To test against all supported python versions run:

    tox

To test against your currently installed version of python run:

    py.test

To run the full test suite run:

    ./runtests.sh

## Publishing

Before publishing the latest version of the package make you sure increment the `__version__` value in `oasislmf/__init__.py`, and commit the change. You'll also need to install the `twine` Python package which `setuptools` uses for publishing packages on PyPI. If publishing wheels then you'll also need to install the `wheel` Python package.

### Using the `publish` subcommand in `setup.py`

The distribution format can be either a source distribution or a platform-specific wheel. To publish the source distribution package run:

    python setup.py publish --sdist

or to publish the platform specific wheel run:

    python setup.py publish --wheel

### Manually publishing, with a GPG signature

The first step is to create the distribution package with the desired format: for the source distribution run:

    python setup.py sdist

which will create a `.tar.gz` file in the `dist` subfolder, or for the platform specific wheel run:

    python setup.py bdist_wheel

which will create `.whl` file in the `dist` subfolder. To attach a GPG signature using your default private key you can then run:

    gpg --detach-sign -a dist/<package file name>.{tar.gz,whl}

This will create `.asc` signature file named `<package file name>.{tar.gz,whl}.asc` in `dist`. You can just publish the package with the signature using:

    twine upload dist/<package file name>.{tar.gz,whl} dist/<package file name>.{tar.gz,whl}.asc
    
## Documentation
* <a href="https://github.com/OasisLMF/OasisLMF/issues">Issues</a>
* <a href="https://github.com/OasisLMF/OasisLMF/releases">Releases</a>
* <a href="https://oasislmf.github.io">General Oasis documentation</a>
* <a href="https://oasislmf.github.io/docs/oasis_mdk.html">Model Development Kit (MDK)</a>
* <a href="https://oasislmf.github.io/OasisLmf/modules.html">Modules</a>

## License
The code in this project is licensed under BSD 3-clause license.
