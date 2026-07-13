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
from oasislmf.pytools.gulmc.structure import build_coverage_dependency_forest
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
# conditional vulnerability file (damage-transition matrix)
# --------------------------------------------------------------------------------------
def _write_conditional_vuln_csv(dir_path, rows):
    # columns match vulnerability_dtype order: vulnerability_id, intensity_bin_id (= source damage
    # bin), damage_bin_id (= dependent damage bin), probability
    path = Path(dir_path) / 'conditional_vulnerability.csv'
    with open(path, 'w') as f:
        f.write('vulnerability_id,source_damage_bin,damage_bin,probability\n')
        for r in rows:
            f.write('{},{},{},{}\n'.format(*r))


def test_get_conditional_vulns():
    """The conditional vulnerability file loads into an [n_cond, ndmg, ndmg] transition matrix,
    indexed [cond, dependent_damage_bin-1, source_damage_bin-1]."""
    from oasislmf.pytools.gulmc.structure import get_conditional_vulns
    from oasis_data_manager.filestore.backends.local import LocalStorage
    with tempfile.TemporaryDirectory() as d:
        _write_conditional_vuln_csv(d, [
            (7, 1, 1, 0.8), (7, 1, 2, 0.2),   # source bin 1 -> dependent {1:0.8, 2:0.2}
            (7, 2, 2, 0.5), (7, 2, 3, 0.5),   # source bin 2 -> dependent {2:0.5, 3:0.5}
            (7, 3, 3, 1.0),                   # source bin 3 -> dependent {3:1.0}
        ])
        arr, ids = get_conditional_vulns(LocalStorage(d), num_damage_bins=3)
    assert arr.shape == (1, 3, 3)
    assert ids.tolist() == [7]
    # [dependent-1, source-1]
    np.testing.assert_allclose(arr[0, :, 0], [0.8, 0.2, 0.0])  # source bin 1
    np.testing.assert_allclose(arr[0, :, 1], [0.0, 0.5, 0.5])  # source bin 2
    np.testing.assert_allclose(arr[0, :, 2], [0.0, 0.0, 1.0])  # source bin 3


def test_get_conditional_vulns_absent_is_empty():
    from oasislmf.pytools.gulmc.structure import get_conditional_vulns
    from oasis_data_manager.filestore.backends.local import LocalStorage
    with tempfile.TemporaryDirectory() as d:
        arr, ids = get_conditional_vulns(LocalStorage(d), num_damage_bins=3)
    assert arr.shape == (0, 3, 3) and ids.shape == (0,)


def test_get_conditional_vulns_allows_missing_source_bins():
    """A source damage bin may be left undefined: its column is all-zero, which the kernel samples
    as no dependent damage. This is a valid modelling choice, so the loader must not require every
    source bin to be defined."""
    from oasislmf.pytools.gulmc.structure import get_conditional_vulns
    from oasis_data_manager.filestore.backends.local import LocalStorage
    with tempfile.TemporaryDirectory() as d:
        # only source bins 1 and 2 defined; source bin 3 left undefined (num_damage_bins=3)
        _write_conditional_vuln_csv(d, [(7, 1, 1, 1.0), (7, 2, 2, 1.0)])
        arr, ids = get_conditional_vulns(LocalStorage(d), num_damage_bins=3)
    assert arr.shape == (1, 3, 3) and ids.tolist() == [7]
    assert np.all(arr[0, :, 2] == 0.0), "undefined source bin 3 -> all-zero column (no dependent damage)"


# --------------------------------------------------------------------------------------
# validation guards (validate_coverage_dependency contract)
# --------------------------------------------------------------------------------------
def _dependency_arrays():
    """A minimal valid coverage-dependency setup: source coverage 1 (hazard-indexed), dependent
    coverage 2 (conditional). Returns the args for validate_coverage_dependency, which individual
    tests perturb to trip one guard."""
    items = np.zeros(2, dtype=[('coverage_id', 'i4'), ('vulnerability_id', 'i4'),
                               ('vulnerability_idx', 'i4'), ('areaperil_agg_vuln_idx', 'i4')])
    items['coverage_id'] = [1, 2]
    items['vulnerability_id'] = [10, 20]
    items['vulnerability_idx'] = [0, 1]
    items['areaperil_agg_vuln_idx'] = [-1, -1]          # both non-aggregate
    coverage_source_id = np.array([0, 0, 1], dtype='u4')  # indexed by coverage_id; cov 2 -> source 1
    vuln_idx_to_cond_idx = np.array([-1, 0], dtype='i8')  # vuln 0 normal, vuln 1 conditional
    return items, coverage_source_id, vuln_idx_to_cond_idx


def test_validate_coverage_dependency_accepts_valid_setup():
    from oasislmf.pytools.gulmc.manager import validate_coverage_dependency
    validate_coverage_dependency(*_dependency_arrays())  # must not raise


def test_validate_rejects_aggregate_dependent():
    from oasislmf.pytools.gulmc.manager import validate_coverage_dependency
    items, csid, v2c = _dependency_arrays()
    items['areaperil_agg_vuln_idx'][1] = 0  # dependent uses an aggregate vulnerability
    with pytest.raises(OasisException):
        validate_coverage_dependency(items, csid, v2c)


def test_validate_rejects_dependent_without_conditional_vuln():
    from oasislmf.pytools.gulmc.manager import validate_coverage_dependency
    items, csid, v2c = _dependency_arrays()
    v2c[1] = -1  # dependent's vuln is not in the conditional file
    with pytest.raises(OasisException):
        validate_coverage_dependency(items, csid, v2c)


def test_validate_rejects_independent_with_conditional_vuln():
    from oasislmf.pytools.gulmc.manager import validate_coverage_dependency
    items, csid, v2c = _dependency_arrays()
    csid[2] = 0  # coverage 2 has no source (independent) but still carries a conditional vuln
    with pytest.raises(OasisException):
        validate_coverage_dependency(items, csid, v2c)


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
    (rather than dropped as an empty coverage) so an uninsured structure can still drive its
    contents; its dependent links to it. It is not special-cased — it stays an ordinary
    zero-TIV coverage."""
    # every location: building (1) and contents (3) at the same areaperil -> dependency active
    keys = []
    for loc in (1, 2, 3):
        keys.append({'loc_id': loc, 'coverage_type_id': 1, 'areaperil_id': 1})
        keys.append({'loc_id': loc, 'coverage_type_id': 3, 'areaperil_id': 1})
    gul = _gul_inputs_for_keys(keys)

    building = gul[gul['coverage_type_id'] == 1]
    contents = gul[gul['coverage_type_id'] == 3]

    # loc 1 building is uninsured (tiv 0) but retained as a driver; loc 2/3 buildings insured.
    # All three buildings are present (the zero-TIV loc-1 source was not dropped).
    assert set(building['loc_id']) == {1, 2, 3}
    assert (building[building['loc_id'] == 1]['tiv'] == 0).all()

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


def test_source_multiplicity_demotes_dependent():
    """A dependent must line up one-to-one with its source. Here the source (building) has two
    items at one areaperil (two perils geocoded to the same cell) while the dependent (contents)
    has one — same areaperil SET but different multiset — so the dependent must be demoted to
    independent (a set-only check would have missed this and silently misaligned)."""
    loc_df = pd.DataFrame({
        'PortNumber': ['1'], 'AccNumber': ['1'], 'LocNumber': ['1'],
        'CountryCode': ['GB'], 'LocCurrency': ['GBP'], 'LocPerilsCovered': ['WTC;WSS'],
        'BuildingTIV': [5000.0], 'ContentsTIV': [2000.0], 'OtherTIV': [0.0], 'BITIV': [0.0],
    })
    exposure = OedExposure(location=loc_df, use_field=True)
    prepare_oed_exposure(exposure)
    loc_df = exposure.location.dataframe
    keys_df = pd.DataFrame([
        {'loc_id': 1, 'peril_id': 'WTC', 'coverage_type_id': 1, 'areaperil_id': 1, 'vulnerability_id': 1, 'status': 'success', 'message': ''},
        {'loc_id': 1, 'peril_id': 'WSS', 'coverage_type_id': 1, 'areaperil_id': 1, 'vulnerability_id': 1, 'status': 'success', 'message': ''},
        {'loc_id': 1, 'peril_id': 'WTC', 'coverage_type_id': 3, 'areaperil_id': 1, 'vulnerability_id': 1, 'status': 'success', 'message': ''},
    ])
    gul = get_gul_input_items(loc_df, keys_df, damage_group_id_cols=['loc_id'],
                              coverage_dependency_settings=[(1, 3)])
    building = gul[gul['coverage_type_id'] == 1]
    contents = gul[gul['coverage_type_id'] == 3]
    assert len(building) == 2 and len(contents) == 1, "source should have 2 items, dependent 1"
    assert (contents['source_coverage_id'] == 0).all(), "multiplicity mismatch -> demoted to independent"


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
    # a cyclic dependency (coverage 1 -> 2 -> 1) must be rejected when the forest is built
    items = np.array([(1, 2), (2, 1)],
                     dtype=[('coverage_id', 'u4'), ('source_coverage_id', 'u4')])
    with pytest.raises(OasisException):
        build_coverage_dependency_forest(items, 3)


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
              sample_size=500, loss_threshold=0., alloc_rule=1, debug=0,
              random_generator=0, ignore_correlation=False,
              effective_damageability=effective_damageability)
    bintocsv(out, run_dir / 'out.csv', 'gul')
    return pd.read_csv(run_dir / 'out.csv')


def test_dependent_without_conditional_vulnerability_fails_loud():
    """A dependent coverage must have a conditional (damage-transition) vulnerability in the
    conditional_vulnerability file. test_model_1 ships no such file, so configuring coverage 2 to
    depend on coverage 1 must fail loud rather than silently mis-sample (this also confirms the
    dependency forest reaches the engine).
    """
    with tempfile.TemporaryDirectory() as t_cond:
        with pytest.raises(OasisException):
            _run(_setup(t_cond, {2: 1}), effective_damageability=False)


@pytest.mark.parametrize("source_damage_type", [0, 2], ids=["relative", "absolute"])
def test_conditional_dependency_end_to_end(source_damage_type):
    """End-to-end: coverage 2 depends on coverage 1 via a conditional_vulnerability file. With an
    identity transition matrix (source damage bin k -> dependent damage bin k), the dependent must
    land in the SAME damage bin as its source on every sample.

    Parametrised over the model's damage type: test_model_1's native (relative) bins, and an
    absolute rewrite with currency-scale bin_to (> 1). The absolute case is what the earlier
    ratio-based recovery got wrong (it clamped the ratio to [0, 1]); driving from the stored
    source damage bin makes a source of any damage type work, which this asserts.
    """
    with tempfile.TemporaryDirectory() as t:
        run_dir = Path(t) / 'assets'
        shutil.copytree(SRC_MODEL, run_dir)
        shutil.rmtree(run_dir / 'input' / 'gulmc_structure', ignore_errors=True)

        # give dependent coverage 2 its own conditional vulnerability ids (per areaperil)
        items = pd.read_csv(run_dir / 'input' / 'items.csv')
        items.loc[(items.coverage_id == 2) & (items.areaperil_id == 154), 'vulnerability_id'] = 101
        items.loc[(items.coverage_id == 2) & (items.areaperil_id == 54), 'vulnerability_id'] = 102
        items.to_csv(run_dir / 'input' / 'items.csv', index=False)
        (run_dir / 'input' / 'items.bin').unlink()  # force the edited csv to be read

        dbd_path = run_dir / 'static' / 'damage_bin_dict.csv'
        dbd = pd.read_csv(dbd_path)
        n_damage_bins = len(dbd)
        if source_damage_type == 2:  # rewrite to absolute, currency-scale bins (bin_to > 1)
            to = np.arange(n_damage_bins, dtype='f8') * 1000.0
            frm = np.concatenate([[0.0], to[:-1]])
            dbd['bin_from'], dbd['bin_to'], dbd['interpolation'] = frm, to, (frm + to) / 2
            dbd['damage_type'] = 2
            dbd.to_csv(dbd_path, index=False)
            (run_dir / 'static' / 'damage_bin_dict.bin').unlink()  # force the edited csv to be read

        with open(run_dir / 'static' / 'conditional_vulnerability.csv', 'w') as f:
            f.write('vulnerability_id,source_damage_bin,damage_bin,probability\n')
            for vid in (101, 102):  # identity: source bin k -> dependent bin k
                for k in range(1, n_damage_bins + 1):
                    f.write(f'{vid},{k},{k},1.0\n')

        _write_correlations(run_dir, {2: 1})
        df = _run(run_dir, effective_damageability=False)

        cov_tiv = pd.read_csv(run_dir / 'input' / 'coverages.csv').set_index('coverage_id')['tiv']
        bin_to = dbd['bin_to'].to_numpy()

        def sampled_bins(item_id, coverage_id):
            d = df[(df['item_id'] == item_id) & (df['sidx'] > 0)].sort_values(['event_id', 'sidx'])
            loss = d['loss'].to_numpy()
            # relative losses are a fraction of TIV; absolute losses are already in bin units
            value = loss / cov_tiv[coverage_id] if source_damage_type == 0 else loss
            return np.searchsorted(bin_to, value, side='left')  # damage-bin index per sample

        # item 1 = source (coverage 1, areaperil 154); item 3 = dependent (coverage 2, areaperil 154)
        src_bins, dep_bins = sampled_bins(1, 1), sampled_bins(3, 2)
        n = min(len(src_bins), len(dep_bins))
        diff = np.abs(src_bins[:n].astype(int) - dep_bins[:n].astype(int))
        # dependent tracks the source's damage bin: (near-)exact match, with rare off-by-one at bin
        # boundaries. Independent sampling over 12 bins would give a mean difference of several bins.
        assert n > 0 and (diff == 0).mean() > 0.9 and diff.mean() < 0.15, \
            f"identity conditional => dependent damage bin should follow source's (exact {(diff == 0).mean():.3f}, mean|d| {diff.mean():.3f})"


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
