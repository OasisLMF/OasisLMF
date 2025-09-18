# evepy/manager.py

import logging

from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


@redirect_logging(exec_name='evepy')
def main(**kwargs):
    pass
