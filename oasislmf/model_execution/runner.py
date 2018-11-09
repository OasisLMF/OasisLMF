import multiprocessing

import subprocess

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from .bash import genbash


@oasis_log()
def run(analysis_settings, number_of_processes=-1, num_reinsurance_iterations=0, ktools_mem_limit=False, filename='run_ktools.sh'):
    if number_of_processes == -1:
        number_of_processes = multiprocessing.cpu_count()

    genbash(number_of_processes, analysis_settings, filename, num_reinsurance_iterations=num_reinsurance_iterations, mem_limit=ktools_mem_limit)
    try:
        subprocess.check_call(['bash', filename])
    except subprocess.CalledProcessError as e:
        raise OasisException('Error running ktools: {}'.format(str(e)))
