from __future__ import print_function
import subprocess

import sys

from .exceptions import OasisException


def run_mono_executable(executable_path, executable_args=None):
    """
    Utility method to run executables compiled for the mono framework.
    """
    executable_args = executable_args or {}

    args_str = ''.join(['-{} {} '.format(key, val) for key, val in executable_args.items()]).strip()
    cmd_str = 'mono {} {}'.format(executable_path, args_str).strip()

    try:
        retcode = subprocess.call(cmd_str, shell=True)
        if retcode < 0:
            print('Mono executable call failed: {}'.format(-retcode), file=sys.stderr)
        else:
            print('Mono executable call succeeded: {}'.format(retcode), file=sys.stderr)
    except OSError as e:
        print('Mono executable call failed: {}'.format(str(e)), file=sys.stderr)
        raise OasisException(str(e))
