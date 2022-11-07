<img src="https://oasislmf.org/packages/oasis_theme_package/themes/oasis_theme/assets/src/oasis-lmf-colour.png" alt="Oasis LMF logo" width="250"/>

[![ktools version](https://img.shields.io/github/tag/Oasislmf/ktools?label=ktools)](https://github.com/OasisLMF/ktools/releases) [![PyPI version](https://badge.fury.io/py/oasislmf.svg)](https://badge.fury.io/py/oasislmf) [![Build Status](https://ci.oasislmfdev.org/buildStatus/icon?job=oasis_oasislmf/master)](https://ci.oasislmfdev.org/job/oasis_oasislmf/job/master/)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OasisLMF/OasisLMF/develop?filepath=FmTesting.ipynb)

[![Oasislmf Testing](https://github.com/OasisLMF/OasisLMF/actions/workflows/unittest.yml/badge.svg?branch=master&event=push)](https://github.com/OasisLMF/OasisLMF/actions/workflows/unittest.yml)
[![PiWind check output](https://github.com/OasisLMF/OasisLMF/actions/workflows/piwind-results.yml/badge.svg?branch=master&event=push)](https://github.com/OasisLMF/OasisLMF/actions/workflows/piwind-results.yml)
[![PiWind MDK](https://github.com/OasisLMF/OasisLMF/actions/workflows/piwind-mdk.yml/badge.svg?branch=master&event=push)](https://github.com/OasisLMF/OasisLMF/actions/workflows/piwind-mdk.yml)



# OasisLMF

The `oasislmf` Python package, loosely called the *model development kit (MDK)* or the *MDK package*, provides a command line toolkit for developing, testing and running Oasis models end-to-end locally, or remotely via the Oasis API. It can generate ground-up losses (GUL), direct/insured losses (IL) and reinsurance losses (RIL). It can also generate deterministic losses at all these levels.


## Releases and maintenance
Releases are published on a monthly cadence which tracks our team's development cycle. The planned fixes, enhancements and features can be seen on the [project development board](https://github.com/orgs/OasisLMF/projects/35) before each release.

### Release Cycle
Release candidates are published on the last Thursday of each month, and tagged as `<version>rc<candidate-version>`, if no changes are required a candidate release is switched from `pre-release` to a `montly release` on the following Thursday (First Thursday of the next month)
Within the release candidate testing week a pre-release version of oasislmf can be installed using `pip install oasislmf --pre`

### Long(er) term support release
Starting from Oasis version `1.15.x` Oasis will backport fixes to the (Long Term support) LTS version of Oasis. This is a single version of oasis that will be updated along with the main release cycle.
The LTS release with switch to track a newer version of monthly release as high priority feature milestones are met. We expect that to happen on a quarterly basis.


## Features

For running models locally the CLI provides a `model` subcommand with the following options:

* `model generate-exposure-pre-analysis`: generate new Exposure input using user custom code (ex: geo-coding, exposure enhancement, or dis-aggregation...)
* `model generate-keys`: generates Oasis keys files from model lookups; these are essentially line items of (location ID, peril ID, coverage type ID, area peril ID, vulnerability ID) where peril ID and coverage type ID span the full set of perils and coverage types that the model supports; if the lookup is for a complex/custom model the keys file will have the same format except that area peril ID and vulnerability ID are replaced by a model data JSON string
* `model generate-oasis-files`: generates the Oasis input CSV files for losses (GUL, GUL + IL, or GUL + IL + RIL); it requires the provision of source exposure and optionally source accounts and reinsurance info. and scope files (in OED format), as well as assets for instantiating model lookups and generating keys files
* `model generate-losses`: generates losses (GUL, or GUL + IL, or GUL + IL + RIL) from a set of pre-existing Oasis files
* `model run`: runs the model from start to finish by generating losses (GUL, or GUL + IL, or GUL + IL + RIL) from the source exposure, and optionally source accounts and reinsurance info. and scope files (in OED or RMS format), as well as assets related to lookup instantiation and keys file generation

The optional `--summarise-exposure` flag can be issued with `model generate-oasis-files` and `model run` to generate a summary of Total Insured Values (TIVs) grouped by coverage type and peril. This produces the `exposure_summary_report.json` file.

For remote model execution the `api` subcommand provides the following main subcommand:

* `api run`: runs the model remotely (same as `model run`) but via the Oasis API

For generating deterministic losses an `exposure run` subcommand is available:

* `exposure run`: generates deterministic losses (GUL, or GUL + IL, or GUL + IL + RIL)

The reusable libraries are organised into several sub-packages, the most relevant of which from a model developer or user's perspective are:

* `api_client`
* `model_preparation`
* `model_execution`
* `utils`

## Minimum Python Requirements

Starting from 1st January 2019, Pandas will no longer be supporting Python 2. As Pandas is a key dependency of the MDK we are **dropping Python 2 (2.7) support** as of this release (1.3.4). The last version which still supports Python 2.7 is version `1.3.3` (published 12/03/2019).

Also for this release (and all future releases) a **minimum of Python 3.7 is required**.


## Installation

The latest released version of the package, or a specific package version, can be installed using `pip`:

    pip install oasislmf[==<version string>]

Alternatively you can install the latest development version using:

    pip install git+{https,ssh}://git@github.com/OasisLMF/OasisLMF

You can also install from a specific branch `<branch name>` using:

    pip install [-v] git+{https,ssh}://git@github.com/OasisLMF/OasisLMF.git@<branch name>#egg=oasislmf

## Enable Bash completion

Bash completion is a functionality which bash helps users type their commands by presenting possible options when users press the tab key while typing a command.

Once oasislmf is installed you'll need to be activate the feature by sourcing a bash file. (only needs to be run once)

### Local

    oasislmf admin enable-bash-complete

### Global

    echo 'complete -C completer_oasislmf oasislmf' | sudo tee /usr/share/bash-completion/completions/oasislmf


## Dependencies

### System

The package provides a built-in lookup framework (`oasislmf.model_preparation.lookup.OasisLookup`) which uses the Rtree Python package, which in turn requires the `libspatialindex` spatial indexing C library.

https://libspatialindex.github.io/index.html

Linux users can install the development version of `libspatialindex` from the command line using `apt`.

    [sudo] apt install -y libspatialindex-dev

and OS X users can do the same via `brew`.

    brew install spatialindex

The PiWind demonstration model uses the built-in lookup framework, therefore running PiWind or any model which uses the built-in lookup, requires that you install `libspatialindex`.

#### GNU/Linux

For GNU/Linux the following is a specific list of required system libraries

 * **Debian**: g++ compiler build-essential, libtool, zlib1g-dev autoconf on debian distros

     sudo apt install g++ build-essential libtool zlib1g-dev autoconf


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

### Creating a bdist for another platform

To create a distribution for a non-host platform use the `--plat-name` flag:

     python setup.py bdist_wheel --plat-name Linux_x86_64

     or

     python setup.py bdist_wheel --plat-name Darwin_x86_64


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
