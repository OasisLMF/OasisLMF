import logging
import multiprocessing
import os
import shutil

import subprocess

from ..model_preparation.oed import ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from .bash import genbash


@oasis_log()
def run(
    analysis_settings,
    number_of_processes=-1,
    num_reinsurance_iterations=0,
    ktools_mem_limit=False,
    set_alloc_rule=ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID,
    fifo_tmp_dir=True,
    run_debug=False,
    filename='run_ktools.sh'
):
    if number_of_processes == -1:
        number_of_processes = multiprocessing.cpu_count()

    custom_gulcalc_cmd = "{}_{}_gulcalc".format(
        analysis_settings.get('module_supplier_id'),
        analysis_settings.get('model_version_id'))

    # Check for custom gulcalc
    if shutil.which(custom_gulcalc_cmd):

        def custom_get_getmodel_cmd(
            number_of_samples,
            gul_threshold,
            use_random_number_file,
            coverage_output,
            item_output,
            process_id,
            max_process_id,
            **kwargs
        ):

            cmd = "{} -e {} {} -a {} -p {}".format(
                custom_gulcalc_cmd,
                process_id,
                max_process_id,
                os.path.abspath("analysis_settings.json"),
                "input")
            if coverage_output != '':
                cmd = '{} -c {}'.format(cmd, coverage_output)
            if item_output != '':
                cmd = '{} -i {}'.format(cmd, item_output)

            return cmd

        genbash(
            number_of_processes,
            analysis_settings,
            num_reinsurance_iterations=num_reinsurance_iterations,
            fifo_tmp_dir=fifo_tmp_dir,
            mem_limit=ktools_mem_limit,
            alloc_rule=set_alloc_rule,
            bash_trace=run_debug,
            filename=filename,
            _get_getmodel_cmd=custom_get_getmodel_cmd,
        )
    else:
        genbash(
            number_of_processes,
            analysis_settings,
            num_reinsurance_iterations=num_reinsurance_iterations,
            fifo_tmp_dir=fifo_tmp_dir,
            mem_limit=ktools_mem_limit,
            alloc_rule=set_alloc_rule,
            bash_trace=run_debug,
            filename=filename
        )

    try:
        bash_trace = subprocess.check_output(['bash', filename], stderr=subprocess.STDOUT)
        logging.info(bash_trace.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        raise OasisException('Error running ktools: {}'.format(e.output.decode('utf-8').strip()))
