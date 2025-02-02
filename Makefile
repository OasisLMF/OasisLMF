SHELL := /bin/bash

REPO := https://github.com/OasisLMF/OasisLMF

PACKAGE_NAME := oasislmf
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
HEAD := $(shell git rev-parse --short=8 HEAD)
PACKAGE_VERSION := $(shell grep __version__ oasislmf/__init__.py | cut -d '=' -f 2 | xargs)

ROOT := $(PWD)

TESTS_ROOT := $(ROOT)/tests

DOCS_ROOT := $(PROJECT_ROOT)/docs
DOCS_BUILD := $(DOCS_ROOT)/_build
DOCS_BUILD_HTML := $(DOCS_ROOT)/_build/html


# Make everything (possible)
all:

# Housekeeping
clean:
	@echo "\n$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Deleting all temporary files\n"
	rm -fr docs/_build/* .pytest_cache *.pyc *__pycache__* ./dist/* ./build/* *.egg-info*

# A simple version check for the installed package (local, sdist or wheel)
version_check:
	@echo "\n$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Checking installed package version (if it is installed)\n"
	python3 -c "import os; os.chdir('oasislmf'); from __init__ import __version__; print(__version__); os.chdir('../')"

version_extract:
	echo "$(PACKAGE_VERSION)"

unittests_jit: clean
	@echo "\n$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Running unit tests (with JIT) + measuring coverage\n"
	cd "$(PROJECT_ROOT)" && \
	python3 -m pytest \
				-p no:flaky \
				--cache-clear \
				--capture=no \
				--code-highlight=yes \
				--color=yes \
				--cov-config=tox.ini \
				--cov=oasislmf \
				--cov-report=xml \
				--cov-report=term-missing:skip-covered \
				--dist worksteal \
				--ignore=fm_testing_tool \
				--ignore=validation \
				--numprocesses=auto \
				--tb=native \
				--verbosity=3 \
				tests

unittests_nojit: clean
	@echo "\n$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Running unit tests (no JIT) + measuring coverage\n"
	cd "$(PROJECT_ROOT)" && \
	python3 -m pytest \
				-p no:flaky \
				--cache-clear \
				--capture=no \
				--code-highlight=yes \
				--color=yes \
				--cov-config=tox.ini \
				--cov=oasislmf \
				--cov-report=xml \
				--cov-report=term-missing:skip-covered \
				--dist worksteal \
				--ignore=fm_testing_tool \
				--ignore=validation \
				--numprocesses=auto \
				--tb=native \
				--verbosity=3 \
				--gul-rtol 1e-3 \
				--gul-atol 1e-3 \
				tests
