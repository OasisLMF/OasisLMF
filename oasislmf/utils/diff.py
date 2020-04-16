__all__ = [
    'column_diff',
    'unified_diff'
]

import difflib
import subprocess

from subprocess import (
    CalledProcessError,
    run,
)

import io

from .exceptions import OasisException


def column_diff(a, b, width=130):
    cmd_str = 'diff -y {} {} -W {}'.format(a, b, width)
    try:
        res = run(cmd_str.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except CalledProcessError as e:
        return 'Error in generating file diff: {}'.format(e)

    return res.stdout.decode('utf-8')


def unified_diff(a, b, as_string=False):
    """
    Generates a unified diff of two files: ``a`` and ``b``. The files must
    be passed in as absolute paths.
    """
    try:
        with io.open(a, 'r') as f1:
            with io.open(b, 'r') as f2:
                diff = difflib.unified_diff(
                    f1.readlines(),
                    f2.readlines(),
                    fromfile=f1.name,
                    tofile=f2.name,
                )
    except (OSError, IOError) as e:
        raise OasisException("Exception raised in 'unified_diff'", e)

    if as_string:
        return ''.join(diff)
    return diff
