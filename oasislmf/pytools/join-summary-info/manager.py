
# join-summary-info/manager.py

import logging
from contextlib import ExitStack

from oasislmf.pytools.utils import redirect_logging


logger = logging.getLogger(__name__)


def run(
    summaryinfo_file,
    data_file,
    output_file,
):
    with ExitStack() as stack:
        pass


@redirect_logging(exec_name='join-summary-info')
def main(
    summaryinfo=None,
    data=None,
    output=None,
    **kwargs
):
    run(
        summaryinfo_file=summaryinfo,
        data_file=data,
        output_file=output,
    )