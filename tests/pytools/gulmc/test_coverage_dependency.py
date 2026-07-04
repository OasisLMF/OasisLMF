"""Tests for the gulmc coverage dependency feature.

A coverage can be configured to depend on a source coverage at the same location: in full
Monte Carlo the dependent coverage's hazard sampling is driven by the source coverage's
per-sample damage ratio; under effective damageability the dependent's effective CDF is
built by combining the source's effective-damage distribution with the dependent's
vulnerability. Dependency is opt-in via model_settings and carried per item on the
correlations file, so with nothing configured behaviour is identical to before.
"""
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.data import correlations_dtype, damagebin_dtype
from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.gulmc.manager import build_dependent_haz_pdf, run as run_gulmc
from oasislmf.pytools.gulmc.structure import (
    build_coverage_dependency_forest, _validate_acyclic_coverage_dependency,
)
from oasislmf.preparation.correlations import get_coverage_dependency_settings
from oasislmf.utils.exceptions import OasisException

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets")
SRC_MODEL = TESTS_ASSETS_DIR.joinpath("test_model_1")

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# settings parsing
# --------------------------------------------------------------------------------------
def test_get_coverage_dependency_settings():
    # valid pairs, including a chain (4 -> 3 -> 1)
    data = {"model_settings": {"coverage_dependency_settings": [
        {"source_coverage_type": 1, "dependent_coverage_type": 3},
        {"source_coverage_type": 3, "dependent_coverage_type": 4},
    ]}}
    assert get_coverage_dependency_settings(data, logger) == [(1, 3), (3, 4)]
    assert get_coverage_dependency_settings(None, logger) == []
    assert get_coverage_dependency_settings({}, logger) == []

    # malformed / self-reference / duplicate-dependent all raise (fail-loud)
    bad_settings = [
        [{"source_coverage_type": "x", "dependent_coverage_type": 3}],   # not an int
        [{"dependent_coverage_type": 3}],                                # missing source
        [{"source_coverage_type": 2, "dependent_coverage_type": 2}],     # self reference
        [{"source_coverage_type": 1, "dependent_coverage_type": 4},
         {"source_coverage_type": 3, "dependent_coverage_type": 4}],     # duplicate dependent
    ]
    for bad in bad_settings:
        with pytest.raises(OasisException):
            get_coverage_dependency_settings({"model_settings": {"coverage_dependency_settings": bad}}, logger)


# --------------------------------------------------------------------------------------
# dependency forest
# --------------------------------------------------------------------------------------
def test_build_coverage_dependency_forest():
    # coverage 100 root; 101 -> 100; 102 -> 101 (chain); 200 root; 201 -> 200
    items = np.array([(100, 0), (101, 100), (102, 101), (200, 0), (201, 200)],
                     dtype=[('coverage_id', 'u4'), ('source_coverage_id', 'u4')])
    src, off, data = build_coverage_dependency_forest(items, 203)
    assert [int(src[i]) for i in (100, 101, 102, 200, 201)] == [0, 100, 101, 0, 200]

    def children(p):
        return list(data[off[p]:off[p + 1]])
    assert children(100) == [101]
    assert children(101) == [102]
    assert children(200) == [201]
    assert children(102) == []


def test_forest_shared_source():
    # a single source (100) may drive multiple dependents (101, 102): a branch, not a cycle.
    items = np.array([(100, 0), (101, 100), (102, 100)],
                     dtype=[('coverage_id', 'u4'), ('source_coverage_id', 'u4')])
    src, off, data = build_coverage_dependency_forest(items, 103)
    assert [int(src[i]) for i in (100, 101, 102)] == [0, 100, 100]
    assert sorted(data[off[100]:off[101]].tolist()) == [101, 102]


def test_forest_rejects_cycles():
    bad = np.zeros(5, dtype='i4')
    bad[1], bad[2] = 2, 1  # 1 -> 2 -> 1
    with pytest.raises(OasisException):
        _validate_acyclic_coverage_dependency(bad)


# --------------------------------------------------------------------------------------
# effective-damageability combined hazard pdf
# --------------------------------------------------------------------------------------
def test_build_dependent_haz_pdf():
    # source effective-damage CDF over 3 damage bins: pmf [0.5, 0.3, 0.2]
    parent_eff_cdf = np.array([0.5, 0.8, 1.0], dtype='f8')
    # damage bin means (representative damage ratios) used as hazard-CDF percentiles
    damage_bins = np.zeros(3, dtype=damagebin_dtype)
    damage_bins['interpolation'] = [0.1, 0.5, 0.9]
    # dependent hazard CDF over 2 bins: bin 0 covers percentiles < 0.4, bin 1 the rest
    haz_cdf_prob = np.array([0.4, 1.0], dtype='f8')
    out = np.zeros(2, dtype='f8')
    dep = build_dependent_haz_pdf(parent_eff_cdf, damage_bins, haz_cdf_prob, 2, out)
    # 0.1 -> bin 0 (mass 0.5); 0.5, 0.9 -> bin 1 (mass 0.3 + 0.2) -> normalised [0.5, 0.5]
    np.testing.assert_allclose(dep, [0.5, 0.5])


# --------------------------------------------------------------------------------------
# end-to-end behaviour
# --------------------------------------------------------------------------------------
def _write_correlations(run_dir, dependent_to_source):
    """Write a correlations file for the model, linking dependent coverages to their source.

    Args:
        run_dir (Path): run directory containing input/items.csv.
        dependent_to_source (dict[int, int]): mapping dependent coverage_id -> source coverage_id.
    """
    items = pd.read_csv(run_dir / 'input' / 'items.csv')
    corr = np.zeros(len(items), dtype=correlations_dtype)
    corr['item_id'] = items['item_id'].to_numpy()
    cov = items['coverage_id'].to_numpy()
    for dep_cov, src_cov in dependent_to_source.items():
        corr['source_coverage_id'][cov == dep_cov] = src_cov
    corr.tofile(run_dir / 'input' / 'correlations.bin')
    pd.DataFrame({k: corr[k] for k in corr.dtype.names}).to_csv(run_dir / 'input' / 'correlations.csv', index=False)


def _setup(tmp, dependent_to_source):
    run_dir = Path(tmp) / 'assets'
    shutil.copytree(SRC_MODEL, run_dir)
    # force a fresh structure build so the dependency forest is (re)computed from correlations
    shutil.rmtree(run_dir / 'input' / 'gulmc_structure', ignore_errors=True)
    _write_correlations(run_dir, dependent_to_source)
    return run_dir


def _run(run_dir, effective_damageability, coverage_dependency_mode='percentile'):
    out = run_dir / 'out.bin'
    run_gulmc(run_dir=run_dir, ignore_file_type=set(),
              file_in=run_dir / 'input' / 'events.bin', file_out=out,
              sample_size=2000, loss_threshold=0., alloc_rule=1, debug=0,
              random_generator=0, ignore_correlation=False,
              effective_damageability=effective_damageability,
              coverage_dependency_mode=coverage_dependency_mode)
    bintocsv(out, run_dir / 'out.csv', 'gul')
    return pd.read_csv(run_dir / 'out.csv')


def _samples(df, item_id):
    d = df[(df['item_id'] == item_id) & (df['sidx'] > 0)].sort_values(['event_id', 'sidx'])
    return d['loss'].to_numpy()


@pytest.mark.parametrize("effective_damageability", [False, True], ids=lambda x: f"eff_damag={x}")
def test_dependency_end_to_end(effective_damageability):
    """Configuring coverage 2 to depend on coverage 1 must:

    - leave the source coverage (1) bit-for-bit unchanged,
    - change the dependent coverage (2), whose hazard is now source-driven,
    - increase the per-sample correlation between source and dependent.
    """
    with tempfile.TemporaryDirectory() as t_base, tempfile.TemporaryDirectory() as t_dep:
        base = _run(_setup(t_base, {}), effective_damageability)
        dep = _run(_setup(t_dep, {2: 1}), effective_damageability)

        # item 1 belongs to source coverage 1, item 3 to dependent coverage 2 (same areaperil)
        s1_base, s1_dep = _samples(base, 1), _samples(dep, 1)
        s3_base, s3_dep = _samples(base, 3), _samples(dep, 3)

        assert np.allclose(s1_base, s1_dep), "source coverage must be unaffected by dependency"
        assert not np.allclose(s3_base, s3_dep), "dependent coverage must change under dependency"

        corr_base = np.corrcoef(s1_base, s3_base)[0, 1]
        corr_dep = np.corrcoef(s1_dep, s3_dep)[0, 1]
        assert corr_dep > corr_base + 0.01, f"dependency should raise correlation (base={corr_base:.3f}, dep={corr_dep:.3f})"


def test_dependency_chain_runs():
    """A dependency chain (3 -> 2 -> 1) must run and leave the chain root unchanged."""
    with tempfile.TemporaryDirectory() as t_base, tempfile.TemporaryDirectory() as t_dep:
        base = _run(_setup(t_base, {}), False)
        dep = _run(_setup(t_dep, {2: 1, 3: 2}), False)

        assert np.allclose(_samples(base, 1), _samples(dep, 1)), "chain root must be unaffected"
        # coverage 2 (item 3) and coverage 3 (item 5) are both dependents and must change
        assert not np.allclose(_samples(base, 3), _samples(dep, 3))
        assert not np.allclose(_samples(base, 5), _samples(dep, 5))


# --------------------------------------------------------------------------------------
# conditional mode
# --------------------------------------------------------------------------------------
def test_conditional_mode_requires_damage_bin_indexed_vuln():
    """Conditional mode requires each dependent vulnerability to be authored with one intensity
    bin per damage bin. test_model_1's dependent vuln is a normal hazard-indexed curve, so
    conditional mode must fail loud (this also confirms the mode flag reaches the engine).
    Percentile mode on the same model is unaffected.
    """
    with tempfile.TemporaryDirectory() as t_pct, tempfile.TemporaryDirectory() as t_cond:
        # percentile mode works on any hazard-indexed vuln
        _run(_setup(t_pct, {2: 1}), effective_damageability=False, coverage_dependency_mode='percentile')
        # conditional mode rejects a vuln that is not 1:1 with the damage bins
        with pytest.raises(OasisException):
            _run(_setup(t_cond, {2: 1}), effective_damageability=False, coverage_dependency_mode='conditional')


def test_conditional_convolution_reference():
    """The conditional eff-dam kernel reuses `calc_eff_damage_cdf(dependent_vuln, source_pmf)`.
    Lock that convolution against a hand-computed reference (the walked-through example)."""
    from oasislmf.pytools.gulmc.manager import calc_eff_damage_cdf

    # source damage pmf and a dependent conditional vuln (rows = source damage bin, cols = dep bin)
    source_pmf = np.array([0.10, 0.20, 0.30, 0.20, 0.10, 0.10], dtype='f8')
    dependent_vuln = np.array([
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.5, 0.2, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.4, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.4, 0.2, 0.1],
        [0.0, 0.0, 0.1, 0.2, 0.5, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
    ], dtype='f8')
    eff_cdf = calc_eff_damage_cdf(dependent_vuln, source_pmf, np.zeros(dependent_vuln.shape[1], dtype='f8'))
    eff_pmf = np.diff(np.concatenate(([0.0], eff_cdf)))
    np.testing.assert_allclose(eff_pmf, [0.18, 0.19, 0.21, 0.16, 0.13, 0.13], atol=1e-9)
