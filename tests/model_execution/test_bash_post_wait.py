import re
import shlex
import shutil
import sys
from pathlib import Path

import pandas as pd

from oasislmf.execution.bash import do_post_wait_processing
from oasislmf.pytools.aal.cli import main as aalpy_main

AALPY_ASSETS_DIR = Path(__file__).parent.parent / "assets" / "test_aalpy" / "all_files"


def _aalpy_args(ord_output, tmp_path):
    """Generate the post-wait commands for one gul summary set and return the aalpy arguments."""
    script = tmp_path / "run_ktools.sh"
    process_counter = {'lpid_monitor_count': 0, 'jpid_monitor_count': 0}
    analysis_settings = {'gul_summaries': [{'id': 1, 'ord_output': ord_output}]}
    do_post_wait_processing('gul', analysis_settings, str(script), process_counter, stderr_guard=False)

    aalpy_cmds = [line for line in script.read_text().splitlines() if line.startswith('aalpy ')]
    assert len(aalpy_cmds) == 1
    return shlex.split(re.sub(r' & lpid\d+=\$!$', '', aalpy_cmds[0]))[1:]


def test_alt_meanonly_command_passes_meanonly_flag(tmp_path):
    args = _aalpy_args({'alt_meanonly': True}, tmp_path)
    assert '-M' in args
    assert '-Kgul_S1_summary_altmeanonly' in args


def test_alt_period_command_does_not_pass_meanonly_flag(tmp_path):
    args = _aalpy_args({'alt_period': True}, tmp_path)
    assert '-M' not in args


def test_alt_meanonly_command_writes_mean_only_table(tmp_path, monkeypatch):
    """Run the generated aalpy command and check it writes the ALT mean-only schema."""
    run_dir = tmp_path / "run"
    shutil.copytree(AALPY_ASSETS_DIR / "input", run_dir / "input")
    shutil.copytree(AALPY_ASSETS_DIR / "work" / "gul", run_dir / "work" / "gul_S1_summary_altmeanonly")
    (run_dir / "output").mkdir()

    args = _aalpy_args({'alt_meanonly': True}, tmp_path)
    monkeypatch.chdir(run_dir)
    monkeypatch.setattr(sys, 'argv', ['aalpy'] + args)
    aalpy_main()

    actual = pd.read_csv(run_dir / "output" / "gul_S1_altmeanonly.csv")
    expected = pd.read_csv(AALPY_ASSETS_DIR / "py_aalmeanonlygul.csv")
    assert list(actual.columns) == ['SummaryId', 'SampleType', 'MeanLoss']
    pd.testing.assert_frame_equal(actual, expected)
