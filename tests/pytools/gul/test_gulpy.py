
"""
This file tests gulpy functionality
"""
import filecmp
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import pandas as pd
import pytest
from unittest.mock import patch
from oasislmf.pytools.gul.manager import run as gulpy_run
from oasislmf.pytools.utils import assert_allclose
from oasislmf.pytools.converters.bintocsv.manager import bintocsv

# get tests dirs
TESTS_DIR = Path(__file__).parent.parent.parent
TESTS_ASSETS_DIR = TESTS_DIR.joinpath("assets")

# get all available test models
# gulpy only supports model data in test_model_1
test_models_dirs = [(x.name, x) for x in TESTS_ASSETS_DIR.glob("test_model_1") if x.is_dir()]

# JIT-aware sample sizes (mirrors test_gulmc): a loop edge + a large/scale size under
# JIT, and just a small fast size when the JIT is disabled (the pure-Python coverage
# job), since the positive sample sizes all exercise the same sampling code. gulpy has
# no sample_size=0 path.
if os.environ.get("NUMBA_DISABLE_JIT", "0") != "0":
    sample_sizes = [10]
else:
    sample_sizes = [1, 1000]
alloc_rules = [1, 2, 3]
ignore_correlations = [True, False]
random_generators = [0, 1, 2]


@pytest.fixture
def gul_rtol(request):
    """Fixture to get the value of the `--gul-rtol` command line argument."""
    return request.config.getoption('--gul-rtol')


@pytest.fixture
def gul_atol(request):
    """Fixture to get the value of the `--gul-atol` command line argument."""
    return request.config.getoption('--gul-atol')


@pytest.mark.parametrize("random_generator", random_generators, ids=lambda x: f"random_generator={x} ")
@pytest.mark.parametrize("ignore_correlation", ignore_correlations, ids=lambda x: f"ignore_correlation={str(x):5} ")
@pytest.mark.parametrize("alloc_rule", alloc_rules, ids=lambda x: f"a{x} ")
@pytest.mark.parametrize("sample_size", sample_sizes, ids=lambda x: f"S{x:<6} ")
@pytest.mark.parametrize("test_model", test_models_dirs, ids=lambda x: x[0])
def test_gulpy(test_model: Tuple[str, str], sample_size: int, alloc_rule: int, ignore_correlation: bool,
               random_generator: int, gul_rtol: float, gul_atol: float):

    test_model_name, test_model_dir_str = test_model
    test_model_dir = Path(test_model_dir_str)

    with TemporaryDirectory() as tmp_result_dir_str:

        tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")

        # link to test model data and expected results (copy would be too slow)
        os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)

        ref_out_bin_fname = tmp_result_dir.joinpath("expected").joinpath(
            f'exp_res_{test_model_name}_a{alloc_rule}_S{sample_size}_L0_ign_corr{ignore_correlation}_'
            f'rng{random_generator}_eff_damagTrue.bin'
        )

        # run modelpy + gulpy
        test_out_bin_fname = tmp_result_dir.joinpath(f'gulmc_{test_model_name}.bin')

        test_cmd = 'evepy 1 1 | modelpy | gulpy -a{} -S{} -L0 {} --random-generator={} > {}'.format(
            alloc_rule,
            sample_size,
            "--ignore-correlation" if ignore_correlation else "",
            random_generator,
            test_out_bin_fname)

        subprocess.run(test_cmd, cwd=test_model_dir, shell=True, capture_output=True, check=True)
        # compare the test results to the expected results
        try:
            assert filecmp.cmp(test_out_bin_fname, ref_out_bin_fname, shallow=False)

        except AssertionError:
            # if the test fails, convert the binaries to csv, compare them, and print the differences in the `loss` column

            # convert to csv
            bintocsv(ref_out_bin_fname.with_suffix('.bin'), ref_out_bin_fname.with_suffix('.csv'), 'gul')
            bintocsv(test_out_bin_fname.with_suffix('.bin'), test_out_bin_fname.with_suffix('.csv'), 'gul')

            df_ref = pd.read_csv(ref_out_bin_fname.with_suffix('.csv'))
            df_test = pd.read_csv(test_out_bin_fname.with_suffix('.csv'))

            # compare the `loss` columns
            assert_allclose(df_ref['loss'], df_test['loss'], rtol=gul_rtol, atol=gul_atol, x_name='expected', y_name='test')

        finally:
            # remove temporary files; the csv files exist only if the bitwise comparison
            # failed, and are written through the symlink into the real assets dir
            test_out_bin_fname.with_suffix('.bin').unlink()
            ref_out_bin_fname.with_suffix('.csv').unlink(missing_ok=True)
            test_out_bin_fname.with_suffix('.csv').unlink(missing_ok=True)


@pytest.mark.parametrize("socket_server,ping_expected,port_expected", [
    ('False', False, None),   # ping disabled
    ('True', True, None),     # ping enabled, non-numeric value -> no port override
    ('8888', True, 8888),     # ping enabled, numeric value -> port override
], ids=lambda v: f"socket_server={v}")
def test_gulpy_ping(socket_server, ping_expected, port_expected):
    """gulpy's socket_server value controls the end-of-run progress ping (oasis_ping).

    The ping is emitted only when socket_server != 'False', and a numeric value is passed
    through as a `port_override`. The model stream is mocked empty, so only the
    end-of-run ping fires (the periodic in-loop ping is covered separately).
    """
    test_model_dir = Path(test_models_dirs[0][1])
    with (patch('oasislmf.pytools.gul.manager.oasis_ping') as mock_ping,
          patch('oasislmf.pytools.gul.manager.read_getmodel_stream', return_value=[]),
          TemporaryDirectory() as tmp_result_dir_str):
        tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")
        os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)
        gulpy_run(run_dir=tmp_result_dir, ignore_file_type="", sample_size=1, loss_threshold=1,
                  alloc_rule=1, debug=False, random_generator=1, socket_server=socket_server)
        if ping_expected:
            mock_ping.assert_called()
            payloads = [c.args[0] for c in mock_ping.call_args_list]
            assert all(('port_override' in p) == (port_expected is not None) for p in payloads)
            if port_expected is not None:
                assert all(p['port_override'] == port_expected for p in payloads)
        else:
            mock_ping.assert_not_called()


def test_gulpy_periodic_ping():
    """The in-loop periodic ping fires once more than SERVER_UPDATE_TIME elapses between events.

    Feeds gulpy a real getmodel stream (so the event loop runs) and forces the threshold
    negative, so the periodic ping fires per event -> oasis_ping is called more than once
    (periodic pings + the end-of-run ping).
    """
    test_model_dir = Path(test_models_dirs[0][1])
    with TemporaryDirectory() as tmp_result_dir_str:
        tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")
        os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)
        stream = tmp_result_dir.joinpath("getmodel_stream.bin")
        out_bin = tmp_result_dir.joinpath("out.bin")
        try:
            # capture a real getmodel stream (evepy | modelpy) so gulpy processes events
            with open(stream, 'wb') as fout:
                subprocess.run("evepy 1 1 | modelpy", cwd=test_model_dir, shell=True, check=True, stdout=fout)
            with (patch('oasislmf.pytools.gul.manager.oasis_ping') as mock_ping,
                  patch('oasislmf.pytools.gul.manager.SERVER_UPDATE_TIME', -1)):
                gulpy_run(run_dir=tmp_result_dir, ignore_file_type=set(), sample_size=10, loss_threshold=0.,
                          alloc_rule=1, debug=False, random_generator=1,
                          file_in=str(stream), file_out=str(out_bin),
                          socket_server='True')
            assert mock_ping.call_count > 1
        finally:
            for scratch in (stream, out_bin):
                if scratch.exists():
                    scratch.unlink()
