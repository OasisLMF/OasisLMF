"""
This file tests gulmc functionality
"""
import filecmp
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import pandas as pd
import pytest

from oasislmf.pytools.gulmc.manager import run as run_gulmc
from oasislmf.pytools.utils import assert_allclose

# get tests dirs
TESTS_DIR = Path(__file__).parent.parent.parent
TESTS_ASSETS_DIR = TESTS_DIR.joinpath("assets")

# get all available test models
test_models_dirs = [(x.name, x) for x in sorted(TESTS_ASSETS_DIR.glob("test_model_*")) if x.is_dir()]

# define the grid of model parameters to test
sample_sizes = [0, 1, 10, 100, 1000]
alloc_rules = [1, 2, 3]
ignore_correlations = [True, False]
random_generators = [0, 1]
effective_damageabilities = [True, False]


@pytest.fixture
def generate_expected(request):
    """Fixture to get the value of the `--generate-expected` command line argument."""
    return request.config.getoption('--generate-expected')


@pytest.mark.parametrize("effective_damageability", effective_damageabilities, ids=lambda x: f"effective_damageability={str(x):5} ")
@pytest.mark.parametrize("random_generator", random_generators, ids=lambda x: f"random_generator={x} ")
@pytest.mark.parametrize("ignore_correlation", ignore_correlations, ids=lambda x: f"ignore_correlation={str(x):5} ")
@pytest.mark.parametrize("alloc_rule", alloc_rules, ids=lambda x: f"a{x} ")
@pytest.mark.parametrize("sample_size", sample_sizes, ids=lambda x: f"S{x:<6} ")
@pytest.mark.parametrize("test_model", test_models_dirs, ids=lambda x: x[0])
def test_gulmc(test_model: Tuple[str, str],
               sample_size: int,
               alloc_rule: int,
               ignore_correlation: bool,
               random_generator: int,
               effective_damageability: bool,
               generate_expected: bool):
    """Test gulmc functionality.

    Args:
        test_model (Tuple[str, str]): test model name and directory.
        sample_size (int): number of samples.
        alloc_rule (int): back allocation rule.
        ignore_correlation (bool): if True, ignore peril correlation groups.
        random_generator (int): random generator (0: Mersenne-Twister, 1: Latin Hypercube).
        effective_damageability (bool): if True, draw loss samples from the effective damageability.
        generate_expected (bool): If True, produce the expected outputs and store them in the expected/ directory.
            If False, run the test.

    Notes:
        For more information on the definitions of gulmc parameters, refer to gulmc documentation.
        To produce the expected outputs, run:
        ```
            pytest --generate-expected tests/pytools/gulmc/test_gulmc.py`
        ```
    """
    test_model_name, test_model_dir_str = test_model
    test_model_dir = Path(test_model_dir_str)

    with TemporaryDirectory() as tmp_result_dir_str:

        tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")

        # link to test model data and expected results (copy would be too slow)
        os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)

        ref_out_bin_fname = tmp_result_dir.joinpath("expected").joinpath(
            f'exp_res_{test_model_name}_a{alloc_rule}_S{sample_size}_L0_ign_corr{ignore_correlation}_'
            f'rng{random_generator}_eff_damag{effective_damageability}'
        )

        # run gulmc
        test_out_bin_fname = tmp_result_dir.joinpath(f'gulmc_{test_model_name}')

        if generate_expected:
            # generate the expected results

            if not ref_out_bin_fname.with_suffix('.bin').exists():
                # generate the expected results only if they don't exist yet
                run_gulmc(
                    run_dir=tmp_result_dir,
                    ignore_file_type=set(),
                    file_in=tmp_result_dir.joinpath('input').joinpath('events.bin'),
                    file_out=ref_out_bin_fname.with_suffix('.bin'),
                    sample_size=sample_size,
                    loss_threshold=0.,
                    alloc_rule=alloc_rule,
                    debug=0,
                    random_generator=random_generator,
                    ignore_correlation=ignore_correlation,
                    effective_damageability=effective_damageability,
                )

        else:
            # run the test case
            file_out = test_out_bin_fname.with_suffix('.bin')
            run_gulmc(
                run_dir=tmp_result_dir,
                ignore_file_type=set(),
                file_in=tmp_result_dir.joinpath('input').joinpath('events.bin'),
                file_out=file_out,
                sample_size=sample_size,
                loss_threshold=0.,
                alloc_rule=alloc_rule,
                debug=0,
                random_generator=random_generator,
                ignore_correlation=ignore_correlation,
                effective_damageability=effective_damageability,
            )

            # compare the test results to the expected results
            try:
                assert filecmp.cmp(file_out, ref_out_bin_fname.with_suffix('.bin'), shallow=False)

            except AssertionError:
                # if the test fails, convert the binaries to csv, compare them, and print the differences in the `loss` column

                # convert to csv
                subprocess.run(f"gultocsv < {ref_out_bin_fname.with_suffix('.bin')} > {ref_out_bin_fname.with_suffix('.csv')}",
                               check=True, capture_output=False, shell=True)

                subprocess.run(f"gultocsv < {file_out} > {test_out_bin_fname.with_suffix('.csv')}",
                               check=True, capture_output=False, shell=True)

                df_ref = pd.read_csv(ref_out_bin_fname.with_suffix('.csv'))
                df_test = pd.read_csv(test_out_bin_fname.with_suffix('.csv'))

                # compare the `loss` columns
                assert_allclose(df_ref['loss'], df_test['loss'], x_name='expected', y_name='test')

                # remove temporary files
                ref_out_bin_fname.with_suffix('.csv').unlink()
                test_out_bin_fname.with_suffix('.csv').unlink()

            finally:
                # remove temporary files
                file_out.unlink()


@pytest.mark.parametrize("effective_damageability", effective_damageabilities, ids=lambda x: f"effective_damageability={str(x):5} ")
@pytest.mark.parametrize("random_generator", random_generators, ids=lambda x: f"random_generator={x} ")
@pytest.mark.parametrize("ignore_correlation", ignore_correlations, ids=lambda x: f"ignore_correlation={str(x):5} ")
@pytest.mark.parametrize("alloc_rule", alloc_rules, ids=lambda x: f"a{x} ")
@pytest.mark.parametrize("sample_size", sample_sizes, ids=lambda x: f"S{x:<6} ")
@pytest.mark.parametrize("test_model", [("test_model_1", TESTS_ASSETS_DIR.joinpath("test_model_1"))], ids=lambda x: x[0])
def test_debug_flag(test_model: Tuple[str, str],
                    sample_size: int,
                    alloc_rule: int,
                    ignore_correlation: bool,
                    random_generator: int,
                    effective_damageability: bool):
    """Test gulmc to raise ValueError if debug is 1 or 2 (i.e., the user wants to print out the random values used
    for the sampling), but alloc_rule is 1, 2, or 3, which does not make sense as it applies back-allocation rules
    on the random values.

    With debug=1 or debug=2, alloc_rule must be 0 or it raises ValueError.

    Args:
        test_model (Tuple[str, str]): test model name and directory.
        sample_size (int): number of samples.
        alloc_rule (int): back allocation rule.
        ignore_correlation (bool): if True, ignore peril correlation groups.
        random_generator (int): random generator (0: Mersenne-Twister, 1: Latin Hypercube).
        effective_damageability (bool): if True, draw loss samples from the effective damageability.
    """
    _, test_model_dir_str = test_model
    test_model_dir = Path(test_model_dir_str)

    with pytest.raises(ValueError) as e:

        with TemporaryDirectory() as tmp_result_dir_str:

            tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")

            # link to test model data and expected results (copy would be too slow)
            os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)

            # run gulmc
            file_out = tmp_result_dir.joinpath('tmp.bin')
            run_gulmc(
                run_dir=tmp_result_dir,
                ignore_file_type=set(),
                file_in=tmp_result_dir.joinpath('input').joinpath('events.bin'),
                file_out=file_out,
                sample_size=sample_size,
                loss_threshold=0.,
                alloc_rule=alloc_rule,
                debug=1,
                random_generator=random_generator,
                ignore_correlation=ignore_correlation,
                effective_damageability=effective_damageability,
            )

    assert f"Expect alloc_rule to be 0 if debug is 1 or 2, got {alloc_rule}" == str(e.value)

    # remove temporary file, if it was created
    if file_out.exists():
        file_out.unlink()


@pytest.mark.parametrize("effective_damageability", effective_damageabilities, ids=lambda x: f"effective_damageability={str(x):5} ")
@pytest.mark.parametrize("random_generator", random_generators, ids=lambda x: f"random_generator={x} ")
@pytest.mark.parametrize("ignore_correlation", ignore_correlations, ids=lambda x: f"ignore_correlation={str(x):5} ")
@pytest.mark.parametrize("alloc_rule", [-2, -1, 4, 5, 6, 7, 8, 9, 10], ids=lambda x: f"a{x} ")
@pytest.mark.parametrize("sample_size", sample_sizes, ids=lambda x: f"S{x:<6} ")
@pytest.mark.parametrize("test_model", [("test_model_1", TESTS_ASSETS_DIR.joinpath("test_model_1"))], ids=lambda x: x[0])
def test_alloc_rule_value(test_model: Tuple[str, str],
                          sample_size: int,
                          alloc_rule: int,
                          ignore_correlation: bool,
                          random_generator: int,
                          effective_damageability: bool):
    """Test gulmc to raise ValueError if alloc rule does not have a valid value.

    Args:
        test_model (Tuple[str, str]): test model name and directory.
        sample_size (int): number of samples.
        alloc_rule (int): back allocation rule.
        ignore_correlation (bool): if True, ignore peril correlation groups.
        random_generator (int): random generator (0: Mersenne-Twister, 1: Latin Hypercube).
        effective_damageability (bool): if True, draw loss samples from the effective damageability.
    """
    _, test_model_dir_str = test_model
    test_model_dir = Path(test_model_dir_str)

    with pytest.raises(ValueError) as e:

        with TemporaryDirectory() as tmp_result_dir_str:

            tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")

            # link to test model data and expected results (copy would be too slow)
            os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)

            # run gulmc
            file_out = tmp_result_dir.joinpath('tmp.bin')
            run_gulmc(
                run_dir=tmp_result_dir,
                ignore_file_type=set(),
                file_in=tmp_result_dir.joinpath('input').joinpath('events.bin'),
                file_out=file_out,
                sample_size=sample_size,
                loss_threshold=0.,
                alloc_rule=alloc_rule,
                debug=0,
                random_generator=random_generator,
                ignore_correlation=ignore_correlation,
                effective_damageability=effective_damageability,
            )

    assert f"Expect alloc_rule to be 0, 1, 2, or 3, got {alloc_rule}" == str(e.value)

    # remove temporary file, if it was created
    if file_out.exists():
        file_out.unlink()
