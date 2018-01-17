import logging
import subprocess

from .exceptions import OasisException


def run_mono_executable(executable_path, executable_args=None):
    """
    Utility method to run executables compiled for the mono framework.
    """
    logger = logging.getLogger()
    executable_args = executable_args or {}

    args_str = ''.join(['-{} {} '.format(key, val) for key, val in executable_args.items()]).strip()
    cmd_str = 'mono {} {}'.format(executable_path, args_str).strip()

    try:
        retcode = subprocess.call(cmd_str, shell=True)
        if retcode < 0:
            logger.error('Mono executable call failed: {}'.format(-retcode))
        else:
            logger.info('Mono executable call succeeded: {}'.format(retcode))
    except OSError as e:
        logger.error('Mono executable call failed: {}'.format(str(e)))
        raise OasisException(str(e))

    return retcode
