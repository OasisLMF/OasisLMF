"""
This file contains general-purpose utilities.
"""
import logging
import numpy as np
import os
import sys



def setup_redirect_logging(exec_name, log_dir='./log', log_level=logging.WARNING):
    """
    This option redirects messages from `warnings` to `logging`
    Without this, importing numba JIT functions can raise NumbaDeprecationWarning
    or NumbaPendingDeprecationWarning which are not caught (crashing the execution)
    """
    logging.captureWarnings(True)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Set Error handler --> only for failues which should terminate execution 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)

    # Set File handler --> All other log messages go here 'set from log_level' 
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{exec_name}_{os.getpid()}.log'))
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Attach handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)



def log_process_start():
    """
    Decorator that redirects logging output to a file.

    Apply to the main run function of a python exec from the pytools directory.
    Only errors will be send to STDERR, all other logging is stored in a file named:
       "<log_dir>/<exec_name>_<PID>.log"

    Each log file is timestamped with start / finish times
        ❯ cat log/fmpy_112820.log
        2023-03-01 13:48:31,286 - oasislmf - INFO - starting process
        2023-03-01 13:48:36,476 - oasislmf - INFO - finishing process

    Args:
        exec_name (str): The name of the script or function being executed. This will be used as part of the log file name.
        log_dir (str, optional): The path to the directory where log files will be stored. Defaults to './log'.
        log_level (int or str, optional): The logging level to use. Can be an integer or a string. Defaults to logging.INFO.

    Returns:
        function: The decorated function.

    Example:
        @redirect_logging(exec_name='my_script', log_dir='./logs', log_level=logging.DEBUG)
        def my_run_function():
            # code here

    """
    def inner(func):
        def wrapper(*args, **kwargs):
            #import ipdb; ipdb.set_trace()
            #import logging_tree; logging_tree.printout()
            
            
            #logging_config = logging.root.manager.loggerDict.keys()
            #root_handlers = logging.getLogger().handlers


            #

            ## https://docs.python.org/3/library/logging.html#logging.lastResort
            ## logging.lastResort.setLevel(logging.ERROR)

            ## Set all logger handlers to level ERROR
            #for lg_name in logging_config:
            #    logger = logging.getLogger(lg_name)

            #    # set all handlers to ERROR
            #    for handler in logger.handlers:
            #        handler.setLevel(logging.ERROR)
            #    # set children oasislmf loggers to 'log_level'
            #    if 'oasislmf.' in lg_name:
            #        for handler in root_handlers:
            #            logger.addHandler(handler)
            #        logger.propagate = False
            #        logger.setLevel(logging.INFO)

            # Set root oasislmf logger to INFO
            logger = logging.getLogger('oasislmf')
            try:
                #import ipdb; ipdb.set_trace()
                logger.info(kwargs)
                logger.info('starting process')

                # Run the wrapped function
                retval = func(*args, **kwargs)
                logger.info('finishing process')
                return retval
            except Exception as err:
                logger.exception(err)
                raise err
            finally:
                logger.handlers.clear()
                logging.shutdown()
        return wrapper
    return inner


def assert_allclose(x, y, rtol=1e-10, atol=1e-8, x_name="x", y_name="y"):
    """
    Drop in replacement for `numpy.testing.assert_allclose` that also shows
    the nonmatching elements in a nice human-readable format.

    Args:
        x (np.array or scalar): first input to compare.
        y (np.array or scalar): second input to compare.
        rtol (float, optional): relative tolreance. Defaults to 1e-10.
        atol (float, optional): absolute tolerance. Defaults to 1e-8.
        x_name (str, optional): header to print for x if x and y do not match. Defaults to "x".
        y_name (str, optional): header to print for y if x and y do not match. Defaults to "y".

    Raises:
        AssertionError: if x and y shapes do not match.
        AssertionError: if x and y data do not match.

    """
    if np.isscalar(x) and np.isscalar(y) == 1:
        return np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

    if x.shape != y.shape:
        raise AssertionError("Shape mismatch: %s vs %s" % (str(x.shape), str(y.shape)))

    d = ~np.isclose(x, y, rtol, atol)

    if np.any(d):
        miss = np.where(d)[0]
        msg = f"Mismatch of {len(miss):d} elements ({len(miss) / x.size * 100:g} %) at the level of rtol={rtol:g}, atol={atol:g},\n" \
            f"{repr(miss)}\n" \
            f"x: {x_name}\n{str(x[d])}\n\n" \
            f"y: {y_name}\n{str(y[d])}"\

        raise AssertionError(msg)
