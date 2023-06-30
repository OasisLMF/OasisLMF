"""
This file mimics the older format for overriding the default model runner. 

Its now obsolete, however there is a compatibility fallback in the code which 
loading this module aims to test.
"""
from oasislmf.execution.runner import run as oasislmf_run


def run(analysis_settings,
        number_of_processes,
        filename,
        num_reinsurance_iterations,
        run_debug,
        stderr_guard,
        gul_legacy_stream,
        set_alloc_rule_gul=None,
        set_alloc_rule_il=None,
        set_alloc_rule_ri=None,
        num_gul_per_lb=None,
        num_fm_per_lb=None,
        fifo_tmp_dir=True,
        custom_gulcalc_cmd=None,
        ):

    oasislmf_run(analysis_settings,
                 number_of_processes=number_of_processes,
                 filename=filename,
                 num_reinsurance_iterations=number_of_processes,
                 run_debug=run_debug,
                 stderr_guard=stderr_guard,
                 gul_legacy_stream=gul_legacy_stream,
                 set_alloc_rule_gul=set_alloc_rule_gul,
                 set_alloc_rule_il=set_alloc_rule_il,
                 set_alloc_rule_ri=set_alloc_rule_ri,
                 num_gul_per_lb=num_gul_per_lb,
                 num_fm_per_lb=num_fm_per_lb,
                 fifo_tmp_dir=fifo_tmp_dir,
                 custom_gulcalc_cmd=custom_gulcalc_cmd)
