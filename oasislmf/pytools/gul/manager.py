"""
This file is the entry point for the gul command for the package.

"""

import logging
logger = logging.getLogger(__name__)

print("[gulpy] Hello world!")


def run(*args, **kwargs):
    """
    Runs the main process of the gul calculation.

    """
    logger.info("Hello world!")

    return 0
