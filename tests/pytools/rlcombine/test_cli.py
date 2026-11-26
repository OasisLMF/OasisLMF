"""Tests for the rlcombine command line: argument parsing, and the argument checks run() makes
before it opens any stream."""
import pytest

from oasislmf.pytools.rlcombine import cli
from oasislmf.pytools.rlcombine.manager import parse_intermediate_outputs, run
from oasislmf.utils.exceptions import OasisException

REQUIRED_ARGS = ['-i', 'rl_IP1_S1_P1', 'rl_IP2_S1_P1', '-p', 'input/RI_1', 'input/RI_2', '-o', 'rl_S1_summary_P1']


def test_parse_args_defaults():
    """Only -i, -p and -o are needed; the other options take their defaults."""
    args = vars(cli.parser.parse_args(REQUIRED_ARGS))
    assert args['low_memory'] is False
    assert args['output_zeros'] is False
    assert args['intermediate_outputs'] is None


@pytest.mark.parametrize('missing, long_name, n_values', [
    ('-i', '--summary-files-in', 2), ('-p', '--static-paths', 2), ('-o', '--output-summary-file', 1),
])
def test_missing_required_args(missing, long_name, n_values, capsys):
    """Leaving out -i, -p or -o stops argparse with a message naming the missing argument."""
    i = REQUIRED_ARGS.index(missing)
    argv = REQUIRED_ARGS[:i] + REQUIRED_ARGS[i + 1 + n_values:]

    with pytest.raises(SystemExit) as exc_info:
        cli.parser.parse_args(argv)

    assert exc_info.value.code == 2
    assert f'the following arguments are required: {missing}/{long_name}' in capsys.readouterr().err


def run_args(**overrides):
    """Keyword arguments for run() with placeholder paths: the checks below fail before any file is opened."""
    kwargs = {
        'summary_files_in': ['rl_IP1_S1_P1', 'rl_IP2_S1_P1'],
        'output_summary_file': 'rl_S1_summary_P1',
        'static_paths': ['input/RI_1', 'input/RI_2'],
        'low_memory': False,
        'intermediate_outputs': None,
        'output_zeros': False,
    }
    kwargs.update(overrides)
    return kwargs


def test_inputs_static_paths_mismatch():
    """Each input summary stream needs its own static path."""
    with pytest.raises(OasisException, match='each summary stream file needs a corresponding static path'):
        run(**run_args(static_paths=['input/RI_1']))


def test_parse_intermediate_outputs():
    """-I values map an input's 0-based position in -i to its pass-through path; a path may contain '='."""
    assert parse_intermediate_outputs(None) == {}
    assert parse_intermediate_outputs([]) == {}
    assert parse_intermediate_outputs(['0=a', '2=dir/b=c']) == {0: 'a', 2: 'dir/b=c'}


@pytest.mark.parametrize('value', ['0:rl_IP1_S1_summary_P1', 'x=rl_IP1_S1_summary_P1', 'rl_IP1_S1_summary_P1'])
def test_intermediate_outputs_malformed(value):
    """A -I value that is not input_index=path is rejected."""
    with pytest.raises(Exception, match='intermediate-outputs expected input of form input=path'):
        run(**run_args(intermediate_outputs=[value]))


def test_intermediate_outputs_out_of_range():
    """A -I index must refer to one of the -i inputs."""
    with pytest.raises(ValueError, match='out of bound'):
        run(**run_args(intermediate_outputs=['2=rl_IP3_S1_summary_P1']))


def test_intermediate_outputs_negative_index():
    """A negative -I index does not refer to an input and is rejected."""
    with pytest.raises(ValueError, match='out of bound'):
        run(**run_args(intermediate_outputs=['-1=rl_IP0_S1_summary_P1']))
