__all__ = ["AggReports"]

import logging
from logging import NullHandler
from .aggreports import AggReports

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())
