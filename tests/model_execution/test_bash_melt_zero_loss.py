"""ord_output.elt_moment_exclude_zero_loss adds eltpy's -Z flag (issue #2168)."""
import pytest

from oasislmf.execution.bash import do_ord


def _eltpy_commands(tmp_path, ord_output):
    script = tmp_path / 'run.sh'
    script.write_text('')
    settings = {'gul_summaries': [{'id': 1, 'ord_output': ord_output}]}
    do_ord('gul', settings, 1, str(script), {'pid_monitor_count': 0}, fifo_dir='fifo/', work_dir='work/')
    return [line for line in script.read_text().splitlines() if 'eltpy' in line]


def test_melt_exclude_zero_loss_adds_flag(tmp_path):
    [cmd] = _eltpy_commands(tmp_path, {'elt_moment': True, 'elt_quantile': True, 'elt_moment_exclude_zero_loss': True})

    assert ' -Z ' in cmd and ' -m ' in cmd


@pytest.mark.parametrize('ord_output', [
    {'elt_moment': True},
    {'elt_moment': True, 'elt_moment_exclude_zero_loss': False},
    {'elt_quantile': True, 'elt_moment_exclude_zero_loss': True},  # no MELT requested
], ids=['unset', 'false', 'no-melt'])
def test_melt_exclude_zero_loss_off(tmp_path, ord_output):
    [cmd] = _eltpy_commands(tmp_path, ord_output)

    assert ' -Z ' not in cmd
