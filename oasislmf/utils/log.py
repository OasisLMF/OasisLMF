__all__ = [
    'oasis_log',
    'read_log_config',
    'set_rotating_logger'
]

"""
Logging utils.
"""
import inspect
import logging
import os
import time

from functools import wraps
from logging.handlers import RotatingFileHandler


def getargspec(func):
    if hasattr(inspect, 'getfullargspec'):
        return inspect.getfullargspec(func)
    else:
        return inspect.getargspec(func)


def set_rotating_logger(
    log_file_path=inspect.stack()[1][1],
    log_level=logging.INFO,
    max_file_size=10**7,
    max_backups=5
):
    _log_fp = log_file_path
    if not os.path.isabs(_log_fp):
        _log_fp = os.path.abspath(_log_fp)

    log_dir = os.path.dirname(_log_fp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('oasislmf')
    for handler in list(logger.handlers):
        if handler.name == 'oasislmf':
            logger.removeHandler(handler)
            break
    handler = RotatingFileHandler(
        _log_fp,
        maxBytes=max_file_size,
        backupCount=max_backups
    )
    handler.name='oasislmf'
    logging.getLogger('oasislmf').setLevel(log_level)
    logging.getLogger('oasislmf').addHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)


def read_log_config(config_parser):
    """
    Read an Oasis standard logging config
    """
    log_file = config_parser['LOG_FILE']
    log_level = config_parser['LOG_LEVEL']
    log_max_size_in_bytes = int(config_parser['LOG_MAX_SIZE_IN_BYTES'])
    log_backup_count = int(config_parser['LOG_BACKUP_COUNT'])

    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('oasislmf')
    for handler in list(logger.handlers):
        if handler.name == 'oasislmf':
            logger.removeHandler(handler)
            break
    handler = RotatingFileHandler(
        log_file, maxBytes=log_max_size_in_bytes,
        backupCount=log_backup_count)
    handler.name = 'oasislmf'
    logging.getLogger('oasislmf').setLevel(log_level)
    logging.getLogger('oasislmf').addHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)


def oasis_log(*args, **kwargs):
    """
    Decorator that logs the entry, exit and execution time.
    """
    logger = logging.getLogger('oasislmf')

    def actual_oasis_log(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            caller_module_name = func.__globals__.get('__name__')

            if func_name == '__init__':
                logger.debug("RUNNING: {}.{}".format(
                    caller_module_name, func_name))
            else:
                logger.info("RUNNING: {}.{}".format(
                    caller_module_name, func_name))

            args_name = getargspec(func)[0]
            args_dict = dict(zip(args_name, args))

            for key, value in args_dict.items():
                if key == "self":
                    continue
                logger.debug("    {} == {}".format(key, value))

            if len(args) > len(args_name):
                for i in range(len(args_name), len(args)):
                    logger.debug("    {}".format(args[i]))
            for key, value in kwargs.items():
                logger.debug("    {} == {}".format(key, value))

            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            # Only log timestamps on functions which took longer than 10ms
            if (end-start) > 0.01:
                logger.info(
                    "COMPLETED: {}.{} in {}s".format(
                        caller_module_name, func_name, round(end - start, 2)))
            else:
                logger.debug(
                    "COMPLETED: {}.{} in {}s".format(
                        caller_module_name, func_name, round(end - start, 2)))
            return result
        return wrapper

    if len(args) == 1 and callable(args[0]):
        return actual_oasis_log(args[0])
    else:
        return actual_oasis_log
