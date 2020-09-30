import logging
import multiprocessing
import os
import shutil

import subprocess

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from .bash import genbash


@oasis_log()
def run(analysis_settings,
        number_of_processes=-1,
        set_alloc_rule_gul=None,
        set_alloc_rule_il=None,
        set_alloc_rule_ri=None,
        gul_legacy_stream=False,
        run_debug=False,
        custom_gulcalc_cmd=None,
        filename='run_ktools.sh',
        **kwargs
):
    # TODO: need clearer responsibility between runner.py and manager.py
    #  ie: why is cpu count and custom_gulcalc_cmd checks here and not in manager
    #      as manager does a lot of alike env check.
    if number_of_processes == -1:
        number_of_processes = multiprocessing.cpu_count()

    # If `given_gulcalc_cmd` is set then always run as a complex model
    # and raise an exception when not found in PATH
    if custom_gulcalc_cmd:
        if not shutil.which(custom_gulcalc_cmd):
            raise OasisException(
                'Run error: Custom Gulcalc command "{}" explicitly set but not found in path.'.format(custom_gulcalc_cmd)
            )
    # when not set then fallback to previous behaviour:
    # Check if a custom binary `<supplier>_<model>_gulcalc` exists in PATH
    else:
        inferred_gulcalc_cmd = "{}_{}_gulcalc".format(
            analysis_settings.get('module_supplier_id'),
            analysis_settings.get('model_version_id'))
        if shutil.which(inferred_gulcalc_cmd):
            custom_gulcalc_cmd = inferred_gulcalc_cmd

    # TODO: should be integrated into bash.py
    if custom_gulcalc_cmd:
        def custom_get_getmodel_cmd(
            number_of_samples,
            gul_threshold,
            use_random_number_file,
            coverage_output,
            item_output,
            process_id,
            max_process_id,
            gul_alloc_rule,
            stderr_guard,
            **kwargs
        ):

            cmd = "{} -e {} {} -a {} -p {}".format(
                custom_gulcalc_cmd,
                process_id,
                max_process_id,
                os.path.abspath("analysis_settings.json"),
                "input")
            if gul_legacy_stream and coverage_output != '':    
                cmd = '{} -c {}'.format(cmd, coverage_output)
            if item_output != '':
                cmd = '{} -i {}'.format(cmd, item_output)
            if stderr_guard:
                cmd = '({}) 2>> log/gul_stderror.err'.format(cmd)

            return cmd
    else:
        custom_get_getmodel_cmd = None

    genbash(
        number_of_processes,
        analysis_settings,
        gul_alloc_rule=set_alloc_rule_gul,
        il_alloc_rule=set_alloc_rule_il,
        ri_alloc_rule=set_alloc_rule_ri,
        gul_legacy_stream=gul_legacy_stream,
        bash_trace=run_debug,
        filename=filename,
        _get_getmodel_cmd=custom_get_getmodel_cmd,
        **kwargs,
    )

    bash_trace = subprocess.check_output(['bash', filename])
    logging.info(bash_trace.decode('utf-8'))
