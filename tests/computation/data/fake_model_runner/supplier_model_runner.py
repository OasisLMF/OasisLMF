from oasislmf.execution.runner import run as oasislmf_run
from oasislmf.execution.runner import run_analysis as oasislmf_run_analysis_chunk

def run(analysis_settings, **params):
    oasislmf_run(analysis_settings, **params)

def run_analysis(**params):
    oasislmf_run_analysis_chunk(**params)
