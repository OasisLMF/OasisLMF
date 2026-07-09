"""Tests for the gulmc coverage dependency feature.

A coverage can be configured to depend on a source coverage at the same location: the source
coverage's sampled damage bin indexes the dependent coverage's (damage-bin-authored)
vulnerability directly, so the dependent's damage is conditioned on how badly the source was
damaged. Dependency is opt-in via model_settings and carried per item on the correlations
file, so with nothing configured behaviour is identical to before.
"""
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ods_tools.oed import OedExposure

from oasislmf.pytools.common.data import correlations_dtype
from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.gulmc.manager import run as run_gulmc
from oasislmf.pytools.gulmc.structure import (
    build_coverage_dependency_forest, _validate_acyclic_coverage_dependency,
)
from oasislmf.preparation.correlations import get_coverage_dependency_settings
from oasislmf.preparation.gul_inputs import get_gul_input_items
from oasislmf.utils.data import prepare_oed_exposure
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
    # read only from the canonical nested model_settings block (no legacy top-level form)
    top_level_only = {"coverage_dependency_settings": [{"source_coverage_type": 1, "dependent_coverage_type": 3}]}
    assert get_coverage_dependency_settings(top_level_only, logger) == []

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
# preparation: zero-TIV driver retention & per-location activation by keys
# --------------------------------------------------------------------------------------
def _gul_inputs_for_keys(keys_rows):
    """Build gul_inputs with 3 locations (building type 1, contents type 3) from keys rows.

    loc 1: building TIV 0 (uninsured), contents insured -> building must be retained driver-only.
    loc 2: both insured. loc 3: both insured (used to test per-location ap_id activation).

    Args:
        keys_rows (list[dict]): keys with loc_id, coverage_type_id, area_peril_id.

    Returns:
        pd.DataFrame: gul_inputs_df with a coverage_dependency_settings of {source 1 -> dep 3}.
    """
    loc_df = pd.DataFrame({
        'PortNumber': ['1', '1', '1'], 'AccNumber': ['1', '2', '3'], 'LocNumber': ['1', '2', '3'],
        'CountryCode': ['GB', 'GB', 'GB'], 'LocCurrency': ['GBP', 'GBP', 'GBP'],
        'LocPerilsCovered': ['WTC', 'WTC', 'WTC'],
        'BuildingTIV': [0.0, 5000.0, 3000.0], 'ContentsTIV': [1000.0, 2000.0, 1500.0],
        'OtherTIV': [0.0, 0.0, 0.0], 'BITIV': [0.0, 0.0, 0.0],
    })
    exposure = OedExposure(location=loc_df, use_field=True)
    prepare_oed_exposure(exposure)
    loc_df = exposure.location.dataframe

    keys_df = pd.DataFrame([
        {'peril_id': 'WTC', 'vulnerability_id': 1, 'status': 'success', 'message': '', **r} for r in keys_rows
    ])
    return get_gul_input_items(loc_df, keys_df, damage_group_id_cols=['loc_id'],
                               coverage_dependency_settings=[(1, 3)])


def test_zero_tiv_source_retained_as_driver():
    """A zero-TIV source (building) at a location with an insured dependent (contents) is kept
    as a driver-only coverage, and its dependent links to it — so an uninsured structure can
    still drive its contents."""
    # every location: building (1) and contents (3) at the same areaperil -> dependency active
    keys = []
    for loc in (1, 2, 3):
        keys.append({'loc_id': loc, 'coverage_type_id': 1, 'areaperil_id': 1})
        keys.append({'loc_id': loc, 'coverage_type_id': 3, 'areaperil_id': 1})
    gul = _gul_inputs_for_keys(keys)

    building = gul[gul['coverage_type_id'] == 1]
    contents = gul[gul['coverage_type_id'] == 3]

    # loc 1 building is uninsured (tiv 0) but retained as driver-only; loc 2/3 buildings insured
    loc1_building = building[building['loc_id'] == 1]
    assert (loc1_building['tiv'] == 0).all() and loc1_building['driver_only'].all()
    assert not building[building['loc_id'].isin([2, 3])]['driver_only'].any()

    # every contents links to its building (same location, same areaperil)
    for loc in (1, 2, 3):
        src = int(building[building['loc_id'] == loc]['coverage_id'].iloc[0])
        assert (contents[contents['loc_id'] == loc]['source_coverage_id'] == src).all()


def test_per_location_activation_by_areaperil():
    """Dependency is active only where the key server returns the source at the same areaperil as
    the dependent. Where contents' areaperil differs from building's, the dependent is independent."""
    keys = []
    for loc in (1, 2, 3):
        keys.append({'loc_id': loc, 'coverage_type_id': 1, 'areaperil_id': 1})
        # loc 3 contents is geocoded to a different areaperil -> must be independent
        keys.append({'loc_id': loc, 'coverage_type_id': 3, 'areaperil_id': 2 if loc == 3 else 1})
    gul = _gul_inputs_for_keys(keys)
    contents = gul[gul['coverage_type_id'] == 3]

    assert (contents[contents['loc_id'].isin([1, 2])]['source_coverage_id'] > 0).all(), \
        "matching areaperil -> dependent"
    assert (contents[contents['loc_id'] == 3]['source_coverage_id'] == 0).all(), \
        "different areaperil -> independent"


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


def _run(run_dir, effective_damageability):
    out = run_dir / 'out.bin'
    run_gulmc(run_dir=run_dir, ignore_file_type=set(),
              file_in=run_dir / 'input' / 'events.bin', file_out=out,
              sample_size=2000, loss_threshold=0., alloc_rule=1, debug=0,
              random_generator=0, ignore_correlation=False,
              effective_damageability=effective_damageability)
    bintocsv(out, run_dir / 'out.csv', 'gul')
    return pd.read_csv(run_dir / 'out.csv')


def test_dependency_requires_damage_bin_indexed_vuln():
    """Coverage dependency requires each dependent vulnerability to be authored with one
    intensity bin per damage bin (the source damage bin indexes the dependent's vuln directly).
    test_model_1's dependent vuln is a normal hazard-indexed curve, so configuring a dependency
    on it must fail loud (this also confirms the forest reaches the engine).
    """
    with tempfile.TemporaryDirectory() as t_cond:
        with pytest.raises(OasisException):
            _run(_setup(t_cond, {2: 1}), effective_damageability=False)


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
