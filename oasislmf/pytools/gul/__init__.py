import logging
from logging import NullHandler

from oasislmf import __version__

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())
