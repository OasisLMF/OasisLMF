# -*- coding: utf-8 -*-

"""
Logging utils.
"""

__all__ = [
    'oasis_log',
    'read_log_config'
]

import inspect
import itertools
import logging
import time

from functools import wraps
from logging.handlers import RotatingFileHandler


def read_log_config(config_parser):
    '''
    Read an Oasis standard logging config
    '''
    log_file = config_parser['LOG_FILE']
    log_level = config_parser['LOG_LEVEL']
    log_max_size_in_bytes = int(config_parser['LOG_MAX_SIZE_IN_BYTES'])
    log_backup_count = int(config_parser['LOG_BACKUP_COUNT'])
    handler = RotatingFileHandler(
        log_file, maxBytes=log_max_size_in_bytes,
        backupCount=log_backup_count)
    logging.getLogger().setLevel(log_level)
    logging.getLogger().addHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)


def oasis_log():
    '''
    Decorator that logs the entry, exit and execution time.
    '''
    logger = logging.getLogger()

    def actual_oasis_log(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            caller_module_name = func.func_globals['__name__']
            logger.info("STARTED: {}.{}".format(
                caller_module_name, func_name))

            args_name = inspect.getargspec(func)[0]
            args_dict = dict(itertools.izip(args_name, args))

            for key, value in args_dict.iteritems():
                if key == "self":
                    continue
                logger.debug("    {} == {}".format(key, value))

            if len(args) > len(args_name):
                for i in range(len(args_name), len(args)):
                    logger.debug("    {}".format(args[i]))
            for key, value in kwargs.iteritems():
                logger.debug("    {} == {}".format(key, value))

            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(
                "COMPLETED: {}.{} in {}s".format(
                    caller_module_name, func_name, round(end - start, 2)))
            return result
        return wrapper
    return actual_oasis_log
