#!/bin/bash

LOG_DIR='/var/log/oasis/'
LOG_BUILD=$LOG_DIR'install_oasislmf.log'
LOG_TOX=$LOG_DIR'test_oasislmf.log'
LOG_FLAKE=$LOG_DIR'pep8_oasislmf.log'
LOG_COV=$LOG_DIR'coverage_oasislmf.log'

BUILD_OUTPUT_DIR='/tmp/output/'


# Install requirements && build
    set -exu
    pip install pip-tools
    pip-compile && pip-sync
    python setup.py sdist

# Test install
    VER_PKG=$(cat ./oasislmf/__init__.py | awk -F"'" ' {print $2} ')
    python setup.py bdist_wheel --verbose > >(tee -a $LOG_BUILD) 2> >(tee -a ${LOG_BUILD} >&2)
    WHL_PKG=$(find ./dist/ -name "oasislmf-${VER_PKG}*.whl")
    pip install --verbose $WHL_PKG 

    # Create OSX wheel 
    python setup.py bdist_wheel --plat-name Darwin_x86_64 

# Unit testing
    find /home/ -name __pycache__ | xargs -r rm -rfv
    find /home/ -name "*.pyc" | xargs -r rm -rfv
    tox | tee $LOG_TOX

    set +exu
    TOX_FAILED=$(cat $LOG_TOX | grep -ci 'ERROR: InvocationError')
    set -exu
    if [ $TOX_FAILED -ne 0 ]; then 
        echo "Unit testing failed - Exiting build"
        exit 1
    fi 
 
# Code Standards report
    flake8 oasislmf/ --ignore=E501,E402 | tee -a $LOG_FLAKE 
    flake8 tests/ --ignore=E501,E402 --exclude=tests/model_preparation/test_reinsurance.py | tee -a $LOG_FLAKE

# Coverate report 
    coverage combine
    coverage report -i oasislmf/*/*.py oasislmf/*.py > $LOG_COV
