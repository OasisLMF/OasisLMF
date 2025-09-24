
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

# get tests dirs
TESTS_DIR = Path(__file__).parent.parent.parent
TESTS_ASSETS_DIR = TESTS_DIR.joinpath("assets")

# get all available test models
# gulpy only supports model data in test_model_1
test_models_dirs = [(x.name, x) for x in TESTS_ASSETS_DIR.glob("test_model_1") if x.is_dir()]

sample_sizes = [1, 10, 100, 1000]
alloc_rules = [1, 2, 3]
ignore_correlations = [True, False]
random_generators = [0, 1]
socket_server = ["True", "3.0", "False"]


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

        test_cmd = 'eve 1 1 | modelpy | gulpy -a{} -S{} -L0 {} --random-generator={} > {}'.format(
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
            subprocess.run(f"gultocsv < {ref_out_bin_fname.with_suffix('.bin')} > {ref_out_bin_fname.with_suffix('.csv')}",
                           check=True, capture_output=False, shell=True)

            subprocess.run(f"gultocsv < {test_out_bin_fname.with_suffix('.bin')} > {test_out_bin_fname.with_suffix('.csv')}",
                           check=True, capture_output=False, shell=True)

            df_ref = pd.read_csv(ref_out_bin_fname.with_suffix('.csv'))
            df_test = pd.read_csv(test_out_bin_fname.with_suffix('.csv'))

            # compare the `loss` columns
            assert_allclose(df_ref['loss'], df_test['loss'], rtol=gul_rtol, atol=gul_atol, x_name='expected', y_name='test')

            # remove temporary files
            ref_out_bin_fname.with_suffix('.csv').unlink()
            test_out_bin_fname.with_suffix('.csv').unlink()

        finally:
            # remove temporary files
            test_out_bin_fname.with_suffix('.bin').unlink()


@pytest.mark.parametrize("socket_server", socket_server)
@pytest.mark.parametrize("test_model", test_models_dirs, ids=lambda x: x[0])
def test_gulpy_ping(test_model, socket_server):
    _, test_model_dir_str = test_model
    test_model_dir = Path(test_model_dir_str)
    with (patch('oasislmf.pytools.gul.manager.oasis_ping') as mock_ping,
          patch('oasislmf.pytools.gul.manager.read_getmodel_stream', return_value=[]),
          TemporaryDirectory() as tmp_result_dir_str):
        tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")
        os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)
        gulpy_run(run_dir=tmp_result_dir, ignore_file_type="", sample_size=1, loss_threshold=1,
                  alloc_rule=1, debug=False, random_generator=1, socket_server=socket_server)
        if socket_server != 'False':
            mock_ping.assert_called()
        else:
            mock_ping.assert_not_called()
