"""
This file tests gulmc functionality
"""
import filecmp
import os
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
from typing import Tuple

from oasislmf.pytools.gulmc.manager import run as run_gulmc

# get tests dirs
TESTS_DIR = Path(__file__).parent.parent.parent
TESTS_ASSETS_DIR = TESTS_DIR.joinpath("assets")

# get all available test models
test_models_dirs = [(x.name, x) for x in TESTS_ASSETS_DIR.iterdir() if x.is_dir()]

sample_sizes = [1, 10, 100, 1000]
alloc_rules = [1, 2, 3]
ignore_correlations = [True, False]
random_generators = [0, 1]
effective_damageabilities = [True, False]


@pytest.mark.parametrize("test_model", test_models_dirs)
@pytest.mark.parametrize("sample_size", sample_sizes)
@pytest.mark.parametrize("alloc_rule", alloc_rules)
@pytest.mark.parametrize("ignore_correlation", ignore_correlations)
@pytest.mark.parametrize("random_generator", random_generators)
@pytest.mark.parametrize("effective_damageability", effective_damageabilities)
def test_gulmc(test_model: Tuple[str, str], sample_size: int, alloc_rule: int, ignore_correlation: bool,
               random_generator: int, effective_damageability: bool):

    test_model_name, test_model_dir_str = test_model
    test_model_dir = Path(test_model_dir_str)

    with TemporaryDirectory() as tmp_result_dir_str:

        tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")

        # link to test model data and expected results (copy would be too slow)
        os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)

        ref_out_bin_fname = tmp_result_dir.joinpath("expected").joinpath(
            f'exp_res_{test_model_name}_a{alloc_rule}_S{sample_size}_L0_ign_corr{ignore_correlation}_'
            f'rng{random_generator}_eff_damag{effective_damageability}.bin'
        )

        # run gulmc
        test_out_bin_fname = tmp_result_dir.joinpath(f'gulmc_{test_model_name}.bin')
        run_gulmc(
            run_dir=tmp_result_dir,
            ignore_file_type=set(),
            file_in=tmp_result_dir.joinpath('input').joinpath('events.bin'),
            file_out=test_out_bin_fname,
            sample_size=sample_size,
            loss_threshold=0.,
            alloc_rule=alloc_rule,
            debug=False,
            random_generator=random_generator,
            ignore_correlation=ignore_correlation,
            effective_damageability=effective_damageability,
        )

        assert filecmp.cmp(test_out_bin_fname, ref_out_bin_fname, shallow=False)
