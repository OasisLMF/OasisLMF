"""
This file tests gulmc functionality
"""
import filecmp
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.getmodel.common import FootprintHeader
from oasislmf.pytools.getmodel.footprint import SHARED_MANIFEST, Footprint
from oasislmf.pytools.gulmc.items import get_dynamic_footprint_adjustments
from oasislmf.pytools.gulmc.manager import run as run_gulmc
from oasislmf.pytools.gulmc.structure import create_gulmc_structure, dynamic_footprint_path
from oasislmf.pytools.utils import assert_allclose
from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from tests.pytools.utils import copy_test_model, read_gul_stream, set_coverage_buildings

# get tests dirs
TESTS_DIR = Path(__file__).parent.parent.parent
TESTS_ASSETS_DIR = TESTS_DIR.joinpath("assets")

# get all available test models
test_models_dirs = [(x.name, x) for x in sorted(TESTS_ASSETS_DIR.glob("test_model_*")) if x.is_dir()]

# define the grid of model parameters to test
# Sample sizes are JIT-aware: the positive sizes all exercise the same sampling code,
# so we cover S=0 (the distinct no-sampling path) plus a loop edge and a large/scale
# case under JIT, and just a small fast size when the JIT is disabled (the pure-Python
# coverage CI job, where large sample sizes are slow for no extra coverage).
if os.environ.get("NUMBA_DISABLE_JIT", "0") != "0":
    sample_sizes = [0, 10]
else:
    sample_sizes = [0, 1, 1000]
alloc_rules = [1, 2, 3]
ignore_correlations = [True, False]
random_generators = [0, 1, 2]
effective_damageabilities = [True, False]


@pytest.fixture
def gulmc_generate_missing_expected(request):
    """Fixture to get the value of the `--gulmc-generate-missing-expected` command line argument."""
    return request.config.getoption('--gulmc-generate-missing-expected')


@pytest.fixture
def update_expected(request):
    """Fixture to get the value of the `--update-expected` command line argument."""
    return request.config.getoption('--update-expected')


@pytest.fixture
def gul_rtol(request):
    """Fixture to get the value of the `--gul-rtol` command line argument."""
    return request.config.getoption('--gul-rtol')


@pytest.fixture
def gul_atol(request):
    """Fixture to get the value of the `--gul-atol` command line argument."""
    return request.config.getoption('--gul-atol')


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
               gulmc_generate_missing_expected: bool,
               update_expected: bool,
               gul_rtol: float,
               gul_atol: float):
    """Test gulmc functionality.

    Args:
        test_model (Tuple[str, str]): test model name and directory.
        sample_size (int): number of samples.
        alloc_rule (int): back allocation rule.
        ignore_correlation (bool): if True, ignore peril correlation groups.
        random_generator (int): random generator (0: Mersenne-Twister, 1: Latin Hypercube).
        effective_damageability (bool): if True, draw loss samples from the effective damageability.
        gulmc_generate_missing_expected (bool): If True, produce the expected outputs and store them in the expected/ directory.
            If False, run the test.
        gulmc_generate_missing_expected (bool): If True, overwrite the expected outputs even if they exist already.

    Notes:
        For more information on the definitions of gulmc parameters, refer to gulmc documentation.
        To produce the expected outputs, run:
        ```
            pytest --gulmc-generate-missing-expected tests/pytools/gulmc/test_gulmc.py
        ```
        By default, the above command only produces the expected outputs that are missing in the /assets/ directory.
        To overwrite the expected results:
        ```
            pytest --gulmc-generate-missing-expected --update-expected tests/pytools/gulmc/test_gulmc.py
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

        if (gulmc_generate_missing_expected and not ref_out_bin_fname.with_suffix('.bin').exists()) or update_expected:
            # generate the expected results only if they don't exist yet or if update_expected is True
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
                bintocsv(ref_out_bin_fname.with_suffix('.bin'), ref_out_bin_fname.with_suffix('.csv'), 'gul')
                bintocsv(test_out_bin_fname.with_suffix('.bin'), test_out_bin_fname.with_suffix('.csv'), 'gul')

                df_ref = pd.read_csv(ref_out_bin_fname.with_suffix('.csv'))
                df_test = pd.read_csv(test_out_bin_fname.with_suffix('.csv'))

                # compare the `loss` columns
                assert_allclose(df_ref['loss'], df_test['loss'], rtol=gul_rtol, atol=gul_atol, x_name='expected', y_name='test')

            finally:
                # remove temporary files; the csv files exist only if the bitwise comparison
                # failed, and are written through the symlink into the real assets dir
                file_out.unlink()
                ref_out_bin_fname.with_suffix('.csv').unlink(missing_ok=True)
                test_out_bin_fname.with_suffix('.csv').unlink(missing_ok=True)


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

    with (pytest.raises(ValueError) as e,
          TemporaryDirectory() as tmp_result_dir_str):

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


@pytest.mark.parametrize("n_buildings", [1, 3], ids=lambda x: f"buildings={x} ")
@pytest.mark.parametrize("debug", [1, 2], ids=lambda x: f"debug={x} ")
@pytest.mark.parametrize("sample_size", [1, 7], ids=lambda x: f"S{x} ")
def test_debug_writes_the_random_values_and_leaves_the_specials_alone(sample_size, debug, n_buildings, tmp_path):
    """debug 1/2 replaces the sample losses with the random values, and nothing else.

    The special (negative) sidx records are derived from the cdf, not from the draw, so they
    must be identical to a debug=0 run. This is only reachable with alloc_rule 0.

    Each building draws its own values, so a coverage whose buildings are summed at source keeps
    them separate under debug: summed, the values would land anywhere in [0, N].

    Regression guard: the write target used to be sized ``S + NUM_IDX`` while the debug branch
    assigns a length-``S`` slice, so debug raised ValueError for S > 1 and overwrote the
    specials for S == 1. Nothing ran debug with alloc_rule 0, so neither was noticed.
    """
    def run(debug_value, n_building, name):
        run_dir = copy_test_model(tmp_path / name)
        set_coverage_buildings(run_dir / "input", n_building)
        out = run_dir / "out.bin"
        run_gulmc(
            run_dir=run_dir,
            ignore_file_type=set(),
            file_in=run_dir / "input" / "events.bin",
            file_out=out,
            sample_size=sample_size,
            loss_threshold=0.,
            alloc_rule=0,
            debug=debug_value,
            random_generator=2,
            ignore_correlation=True,
            effective_damageability=False,
        )
        return read_gul_stream(out)[2]

    # summed at source under debug, compared with the same buildings kept separate
    plain = run(0, -n_buildings, "plain")
    dbg = run(debug, n_buildings, "dbg")

    specials_plain = plain[plain["sidx"] < 0].reset_index(drop=True)
    specials_dbg = dbg[dbg["sidx"] < 0].reset_index(drop=True)
    assert not specials_plain.empty, "no special records to compare"
    pd.testing.assert_frame_equal(specials_plain, specials_dbg)

    # and the sample records are the random values, one per building and sample
    samples_plain = plain[plain["sidx"] > 0]
    samples_dbg = dbg[dbg["sidx"] > 0]
    assert samples_dbg["sidx"].tolist() == samples_plain["sidx"].tolist()
    assert samples_dbg["loss"].between(0, 1).all(), "debug output should be random values in [0, 1]"


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


@pytest.mark.parametrize("test_model", [("test_adjustments_1", TESTS_ASSETS_DIR.joinpath("test_adjustments_1"))], ids=lambda x: x[0])
def test_adjustments(test_model: Tuple[str, str]):
    """Test gulmc to check that the adjustments are applied correctly.
    """
    _, test_model_dir_str = test_model
    test_model_dir = Path(test_model_dir_str)
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
            sample_size=10,
            loss_threshold=0.,
            alloc_rule=0,
            debug=0,
            random_generator=0,
            ignore_correlation=False,
            effective_damageability=False,
        )

        # assert that file_out exists
        assert file_out.exists()

        if file_out.exists():
            # convert to csv
            bintocsv(file_out, file_out.with_suffix('.csv'), 'gul')

            df_test = pd.read_csv(file_out.with_suffix('.csv'))
            df_compare = pd.read_csv(Path(tmp_result_dir).joinpath("expected").joinpath("rng_adj.csv"))
            # assert that the adjustments were applied correctly

            assert_allclose(df_test['loss'], df_compare['loss'], rtol=1e-10, atol=1e-8, x_name='test', y_name='expected')

            # remove temporary file
            file_out.unlink()
            file_out.with_suffix('.csv').unlink()


@pytest.mark.parametrize("socket_server,ping_expected,port_expected", [
    ('False', False, None),   # ping disabled (default / real loss runs)
    ('True', True, None),     # ping enabled, non-numeric value -> no port override
    ('8888', True, 8888),     # ping enabled, numeric value -> port override (matrix never hit this)
], ids=lambda v: f"socket_server={v}")
def test_gulmc_socket_server_ping(socket_server, ping_expected, port_expected):
    """The socket_server value controls the end-of-run progress ping (oasis_ping).

    The ping is emitted only when socket_server != 'False', and a numeric value is
    passed through as a `port_override`. Kept separate from test_gulmc so this small,
    loss-independent feature isn't multiplied across the whole loss-regression matrix.
    """
    test_model_dir = TESTS_ASSETS_DIR.joinpath("test_model_1")
    with (TemporaryDirectory() as tmp_result_dir_str,
          patch('oasislmf.pytools.gulmc.manager.oasis_ping') as mock_ping):
        tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")
        os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)
        file_out = tmp_result_dir.joinpath('tmp.bin')
        try:
            run_gulmc(
                run_dir=tmp_result_dir,
                ignore_file_type=set(),
                file_in=tmp_result_dir.joinpath('input').joinpath('events.bin'),
                file_out=file_out,
                sample_size=10,
                loss_threshold=0.,
                alloc_rule=1,
                debug=0,
                random_generator=1,
                ignore_correlation=False,
                effective_damageability=False,
                socket_server=socket_server,
            )
            if ping_expected:
                mock_ping.assert_called()
                ping_payloads = [c.args[0] for c in mock_ping.call_args_list]
                assert all(('port_override' in p) == (port_expected is not None) for p in ping_payloads)
                if port_expected is not None:
                    assert all(p['port_override'] == port_expected for p in ping_payloads)
            else:
                mock_ping.assert_not_called()
        finally:
            if file_out.exists():
                file_out.unlink()


def test_gulmc_socket_server_periodic_ping():
    """The in-loop periodic ping fires once more than SERVER_UPDATE_TIME elapses between events.

    Forcing the threshold below zero makes the periodic ping (distinct from the
    end-of-run ping) fire on every event. The periodic ping is dispatched through
    `oasis_ping_async` (fire-and-forget, so a stuck ping target can't stall the compute
    loop), while the end-of-run ping still calls `oasis_ping` directly.
    """
    test_model_dir = TESTS_ASSETS_DIR.joinpath("test_model_1")
    with (TemporaryDirectory() as tmp_result_dir_str,
          patch('oasislmf.pytools.gulmc.manager.oasis_ping') as mock_ping,
          patch('oasislmf.pytools.gulmc.manager.oasis_ping_async') as mock_ping_async,
          patch('oasislmf.pytools.gulmc.manager.SERVER_UPDATE_TIME', -1)):
        tmp_result_dir = Path(tmp_result_dir_str).joinpath("assets")
        os.symlink(test_model_dir, tmp_result_dir, target_is_directory=True)
        file_out = tmp_result_dir.joinpath('tmp.bin')
        try:
            run_gulmc(
                run_dir=tmp_result_dir,
                ignore_file_type=set(),
                file_in=tmp_result_dir.joinpath('input').joinpath('events.bin'),
                file_out=file_out,
                sample_size=10,
                loss_threshold=0.,
                alloc_rule=1,
                debug=0,
                random_generator=1,
                ignore_correlation=False,
                effective_damageability=False,
                socket_server='True',
            )
            # periodic ping (per event, threshold forced negative) fires more than once
            assert mock_ping_async.call_count > 1
            mock_ping.assert_called_once()
        finally:
            if file_out.exists():
                file_out.unlink()


def test_get_dynamic_footprint_adjustments_without_item_adjustments_bin_only():
    """Regression test for issue #2015.

    When the keys stage does not run a `dynamic_model_adjustment` step,
    `item_adjustments.csv` is absent and the items file only exists as
    `items.bin`. The fallback must read the binary items file (rather than
    hard-coding `items.csv`) and return zero adjustments per item.
    """
    src_items_bin = TESTS_ASSETS_DIR.joinpath("test_adjustments_1", "input", "items.bin")
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        # only items.bin is present: no item_adjustments.csv and no items.csv
        shutil.copy(src_items_bin, tmp_dir.joinpath("items.bin"))

        adjustments = get_dynamic_footprint_adjustments(str(tmp_dir))

        assert len(adjustments) == 1
        assert adjustments[0]["item_id"] == 1
        assert np.all(adjustments["intensity_adjustment"] == 0)
        assert np.all(adjustments["return_period"] == 0)


def make_dynamic_test_model(run_dir, confluences=False, partitioned=False, labels=False):
    """Copy test_model_1 to run_dir with its binary footprint replaced by a dynamic one over the same areaperils.

    Args:
        confluences (bool): if True, a second section holds every other areaperil with other intensities, and
            every other event affects both sections with different brackets
        partitioned (bool): if True, the hazard case and event definition are partitioned by section_id
        labels (bool): if True, the hazard case and event definition have a string label column the footprint
            build does not use
    """
    shutil.copytree(TESTS_ASSETS_DIR / 'test_model_1', run_dir, ignore=shutil.ignore_patterns('log', 'expected', '*.log'))
    static = run_dir / 'static'
    num_intensity_bins = int(np.fromfile(static / 'footprint.bin', dtype=FootprintHeader, count=1)['num_intensity_bins'][0])
    for footprint_file in static.glob('footprint.*'):
        footprint_file.unlink()
    areaperils = np.unique(pd.read_csv(run_dir / 'input' / 'items.csv')['areaperil_id'])
    event_ids = pd.read_csv(run_dir / 'input' / 'events.csv')['event_id']
    hazard_case = [pd.DataFrame({'section_id': 1, 'areaperil_id': np.repeat(areaperils, 2),
                                 'return_period': np.tile([10, 20], len(areaperils)),
                                 'intensity': np.tile([1, 2], len(areaperils))})]
    event_definition = [pd.DataFrame({'event_id': event_ids, 'section_id': 1, 'rp_from': 10, 'rp_to': 20,
                                      'interpolation': 0.5})]
    sections = [1]
    if confluences:
        in_two = areaperils[::2]
        hazard_case.append(pd.DataFrame({'section_id': 2, 'areaperil_id': np.repeat(in_two, 2),
                                         'return_period': np.tile([10, 20], len(in_two)),
                                         'intensity': np.tile([2, 3], len(in_two))}))
        event_definition.append(pd.DataFrame({'event_id': event_ids[::2], 'section_id': 2, 'rp_from': 10, 'rp_to': 20,
                                              'interpolation': 0.8}))
        sections.append(2)
    hazard_case, event_definition = pd.concat(hazard_case), pd.concat(event_definition)
    if labels:
        hazard_case['label'] = 'rp' + hazard_case['return_period'].astype(str)
        event_definition['label'] = 'event' + event_definition['event_id'].astype(str)
    partition_cols = ['section_id'] if partitioned else None
    hazard_case.to_parquet(static / 'hazard_case.parquet', index=False, partition_cols=partition_cols)
    event_definition.to_parquet(static / 'event_definition.parquet', index=False, partition_cols=partition_cols)
    # a dynamic footprint maps intensities to bins per peril; here intensity n is bin n
    pd.DataFrame({'peril_id': 'WTC', 'intensity_bin': np.arange(num_intensity_bins + 1),
                  'intensity': np.arange(num_intensity_bins + 1)}).to_csv(static / 'intensity_bin_dict.csv', index=False)
    (static / 'footprint_parquet_meta.json').write_text(
        f'{{"num_intensity_bins": {num_intensity_bins}, "has_intensity_uncertainty": 0}}')
    pd.DataFrame({'section_id': sections}).to_csv(run_dir / 'input' / 'sections.csv', index=False)
    items = pd.read_csv(run_dir / 'input' / 'items.csv')
    pd.DataFrame({'LocID': np.arange(1, len(items) + 1), 'PerilID': 'WTC', 'CoverageTypeID': 1,
                  'AreaPerilID': items['areaperil_id'], 'VulnerabilityID': items['vulnerability_id']}).to_csv(
        run_dir / 'input' / 'keys.csv', index=False)
    summary_map = pd.read_csv(run_dir / 'input' / 'gul_summary_map.csv')
    summary_map.assign(peril_id='WTC').to_csv(run_dir / 'input' / 'gul_summary_map.csv', index=False)


@pytest.mark.parametrize("partitioned,labels", [(False, False), (True, False), (True, True)],
                         ids=["flat", "partitioned", "partitioned-labels"])
@pytest.mark.parametrize("sample_size", [0, 10])
def test_dynamic_footprint_shared_between_processes_gives_the_same_losses(tmp_path, monkeypatch, partitioned, labels,
                                                                          sample_size):
    """gulmc memory-maps the dynamic footprint's hazard that create_gulmc_structure saved, rather than reading the
    model data in every process, with the same losses, confluences included. A column the footprint build does not
    use, here a string label, is not saved and does not stop the sharing.
    """
    run_dir = tmp_path / 'run'
    make_dynamic_test_model(run_dir, confluences=True, partitioned=partitioned, labels=labels)
    monkeypatch.chdir(run_dir)

    def gulmc(file_out):
        run_gulmc(run_dir=str(run_dir), ignore_file_type=set(), file_in=run_dir / 'input' / 'events.bin',
                  file_out=file_out, sample_size=sample_size, loss_threshold=0., alloc_rule=1, debug=0,
                  random_generator=1, dynamic_footprint=True)

    gulmc(tmp_path / 'read.bin')  # no gulmc structures: the footprint is read from the model data
    create_gulmc_structure(str(run_dir), set(), [], True, 'oasis_data_manager.df_reader.reader.OasisPandasReader')
    manifest_path = Path(dynamic_footprint_path(str(run_dir))) / SHARED_MANIFEST
    assert manifest_path.is_file()
    assert 'label' not in manifest_path.read_text()
    gulmc(tmp_path / 'shared.bin')

    assert (tmp_path / 'read.bin').stat().st_size > 1000
    assert filecmp.cmp(tmp_path / 'read.bin', tmp_path / 'shared.bin', shallow=False)


def test_dynamic_footprint_incomplete_save_is_not_used(tmp_path, monkeypatch):
    """A save without its manifest, written last, did not finish: the footprint is read from the model data."""
    run_dir = tmp_path / 'run'
    make_dynamic_test_model(run_dir, confluences=True, partitioned=True)
    monkeypatch.chdir(run_dir)
    create_gulmc_structure(str(run_dir), set(), [], True, 'oasis_data_manager.df_reader.reader.OasisPandasReader')
    shared_dir = dynamic_footprint_path(str(run_dir))
    storage = get_storage_from_config_path(str(run_dir / 'model_storage.json'), str(run_dir / 'static'))

    with Footprint.load(storage, shared_dir=shared_dir) as footprint:
        assert isinstance(footprint.shared_hazard['intensity'], np.memmap)
        shared_event = footprint.get_event(1)

    (Path(shared_dir) / SHARED_MANIFEST).unlink()
    with Footprint.load(storage, shared_dir=shared_dir) as footprint:
        assert footprint.shared_hazard is None and footprint.df_hazard_case is not None
        np.testing.assert_array_equal(footprint.get_event(1), shared_event)
