import logging
import os
import shutil

import subprocess

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from .bash import genbash, genbash_wrapper, genbash_params, genbash_outputs, genbash_analysis
from ..utils.defaults import (
    KTOOLS_ALLOC_GUL_DEFAULT,
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT,
)


@oasis_log()
def run(
    analysis_settings,
    number_of_processes=-1,
    process_number=None,
    num_reinsurance_iterations=0,
    set_alloc_rule_gul=KTOOLS_ALLOC_GUL_DEFAULT,
    set_alloc_rule_il=KTOOLS_ALLOC_IL_DEFAULT,
    set_alloc_rule_ri=KTOOLS_ALLOC_RI_DEFAULT,
    fifo_tmp_dir=True,
    stderr_guard=True,
    run_debug=False,
    custom_gulcalc_cmd=None,
    filename='run_ktools.sh'
):
    params = genbash_params(
        analysis_settings=analysis_settings,
        max_process_id=number_of_processes,
        process_number=process_number,
        num_reinsurance_iterations=num_reinsurance_iterations,
        fifo_tmp_dir=fifo_tmp_dir,
        gul_alloc_rule=set_alloc_rule_gul,
        il_alloc_rule=set_alloc_rule_il,
        ri_alloc_rule=set_alloc_rule_ri,
        stderr_guard=stderr_guard,
        bash_trace=run_debug,
        filename=filename,
        _get_getmodel_cmd=custom_gulcalc_cmd,
    )
    params['fifo_queue_dir'], params['analysis_bash_trace'] = run_analysis(**params)
    params['output_bash_trace'] = run_outputs(**params)
    return params


@oasis_log()
def run_analysis(
    max_process_id=None,
    analysis_settings=None,
    full_correlation=False,
    gul_output=False,
    il_output=False,
    ri_output=False,
    fifo_queue_dir="",
    fifo_full_correlation_dir="",
    work_dir='work/',
    work_kat_dir='work/kat/',
    work_full_correlation_dir='work/full_correlation/',
    work_full_correlation_kat_dir='work/full_correlation/kat/',
    output_dir='output/',
    output_full_correlation_dir='output/full_correlation/',
    gul_alloc_rule=None,
    gul_threshold=0,
    num_reinsurance_iterations=0,
    fifo_tmp_dir=True,
    il_alloc_rule=None,
    ri_alloc_rule=None,
    stderr_guard=True,
    filename='run_kools.sh',
    custom_args=None,
    process_number=None,
    process_counter=None,
    gul_item_stream=None,
    use_random_number_file=False,
    number_of_samples=0,
    bash_trace=False,
    _get_getmodel_cmd=None,
    **extra
):
    # If `given_gulcalc_cmd` is set then always run as a complex model
    # and raise an exception when not found in PATH
    if _get_getmodel_cmd:
        if not shutil.which(_get_getmodel_cmd):
            raise OasisException(
                'Run error: Custom Gulcalc command "{}" explicitly set but not found in path.'.format(_get_getmodel_cmd)
            )

        def _custom_get_getmodel_cmd(
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
            if coverage_output != '' and not gul_alloc_rule:
                cmd = '{} -c {}'.format(cmd, coverage_output)
            if item_output != '':
                cmd = '{} -i {}'.format(cmd, item_output)
            if stderr_guard:
                cmd = '({}) 2>> log/gul_stderror.err'.format(cmd)

            return cmd

        custom_gulcalc_cmd = _custom_get_getmodel_cmd
    # when not set then fallback to previous behaviour:
    # Check if a custom binary `<supplier>_<model>_gulcalc` exists in PATH
    else:
        inferred_gulcalc_cmd = "{}_{}_gulcalc".format(
            analysis_settings.get('module_supplier_id'),
            analysis_settings.get('model_version_id'))
        if shutil.which(inferred_gulcalc_cmd):
            custom_gulcalc_cmd = inferred_gulcalc_cmd

    with genbash_wrapper(filename, bash_trace, stderr_guard):
        fifo_queue_dir = genbash_analysis(
            max_process_id=max_process_id,
            analysis_settings=analysis_settings,
            full_correlation=full_correlation,
            gul_output=gul_output,
            il_output=il_output,
            ri_output=ri_output,
            fifo_queue_dir=fifo_queue_dir,
            fifo_full_correlation_dir=fifo_full_correlation_dir,
            work_dir=work_dir,
            work_kat_dir=work_kat_dir,
            work_full_correlation_dir=work_full_correlation_dir,
            work_full_correlation_kat_dir=work_full_correlation_kat_dir,
            output_dir=output_dir,
            output_full_correlation_dir=output_full_correlation_dir,
            gul_alloc_rule=gul_alloc_rule,
            gul_threshold=gul_threshold,
            num_reinsurance_iterations=num_reinsurance_iterations,
            fifo_tmp_dir=fifo_tmp_dir,
            il_alloc_rule=il_alloc_rule,
            ri_alloc_rule=ri_alloc_rule,
            stderr_guard=stderr_guard,
            filename=filename,
            _get_getmodel_cmd=_get_getmodel_cmd,
            custom_args=custom_args,
            process_number=process_number,
            process_counter=process_counter,
            gul_item_stream=gul_item_stream,
            use_random_number_file=use_random_number_file,
            number_of_samples=number_of_samples,
        )

    bash_trace = subprocess.check_output(['bash', filename]).decode('utf-8')
    logging.info(bash_trace)

    return fifo_queue_dir, bash_trace


@oasis_log()
def run_outputs(**params):
    with genbash_wrapper(params['filename'], params['bash_trace'], params['stderr_guard']):
        genbash_outputs(
            **{**params, 'fifo_queue_dir': params['fifo_queue_dir']},
        )

    bash_trace = subprocess.check_output(['bash', params['filename']]).decode('utf-8')
    logging.info(bash_trace)

    return bash_trace
