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
from oasislmf.pytools.common.event_stream import MAX_LOSS_IDX, MEAN_IDX, STD_DEV_IDX, TIV_IDX
from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.gulmc.manager import run as run_gulmc
from oasislmf.pytools.gulmc.structure import build_coverage_dependency_forest
from oasislmf.preparation.gul_inputs import get_gul_input_items
from oasislmf.utils.data import prepare_oed_exposure
from oasislmf.utils.exceptions import OasisException

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets")
SRC_MODEL = TESTS_ASSETS_DIR.joinpath("test_model_1")


# --------------------------------------------------------------------------------------
# conditional vulnerability file (damage-transition matrix)
# --------------------------------------------------------------------------------------
def _damage_bins(n, first_bin_is_zero_damage=True):
    """A damage_bin_dict with n bins: bin 1 is the no-damage point bin [0, 0] by convention,
    then n-1 equal bins up to 1.0. `first_bin_is_zero_damage=False` drops that convention."""
    from oasislmf.pytools.common.data import damagebin_dtype
    dbd = np.zeros(n, dtype=damagebin_dtype)
    dbd['bin_index'] = np.arange(1, n + 1)
    edges = np.linspace(0., 1., n if first_bin_is_zero_damage else n + 1)
    if first_bin_is_zero_damage:
        dbd[0]['bin_from'] = dbd[0]['bin_to'] = dbd[0]['interpolation'] = 0.
        dbd[1:]['bin_from'], dbd[1:]['bin_to'] = edges[:-1], edges[1:]
    else:
        dbd['bin_from'], dbd['bin_to'] = edges[:-1], edges[1:]
    dbd['interpolation'] = (dbd['bin_from'] + dbd['bin_to']) / 2
    return dbd


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
        arr, ids = get_conditional_vulns(LocalStorage(d), _damage_bins(3))
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
        arr, ids = get_conditional_vulns(LocalStorage(d), _damage_bins(3))
    assert arr.shape == (0, 3, 3) and ids.shape == (0,)


def test_get_conditional_vulns_allows_missing_source_bins():
    """A source damage bin may be left undefined (the source may never reach it), meaning "no
    dependent damage". The loader makes that explicit as a point mass on damage bin 1, the
    no-damage bin: left as an all-zero column the sampled loss would be undefined, since a
    zero-height cdf bin cannot be interpolated within."""
    from oasislmf.pytools.gulmc.structure import get_conditional_vulns
    from oasis_data_manager.filestore.backends.local import LocalStorage
    with tempfile.TemporaryDirectory() as d:
        # only source bins 1 and 2 defined; source bin 3 left undefined (num_damage_bins=3)
        _write_conditional_vuln_csv(d, [(7, 1, 1, 1.0), (7, 2, 2, 1.0)])
        arr, ids = get_conditional_vulns(LocalStorage(d), _damage_bins(3))
    assert arr.shape == (1, 3, 3) and ids.tolist() == [7]
    np.testing.assert_allclose(arr[0, :, 2], [1.0, 0.0, 0.0]), "undefined source bin -> no damage"
    # the defined columns are untouched
    np.testing.assert_allclose(arr[0, :, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(arr[0, :, 1], [0.0, 1.0, 0.0])


def test_get_conditional_vulns_rejects_undefined_bin_without_a_no_damage_bin():
    """"No dependent damage" has to be expressible: if damage bin 1 of the damage_bin_dict is not a
    zero-damage point bin, an undefined source damage bin cannot mean no damage, so it must be
    authored explicitly rather than silently mapped to a damaging bin."""
    from oasislmf.pytools.gulmc.structure import get_conditional_vulns
    from oasis_data_manager.filestore.backends.local import LocalStorage
    with tempfile.TemporaryDirectory() as d:
        _write_conditional_vuln_csv(d, [(7, 1, 1, 1.0), (7, 2, 2, 1.0)])
        with pytest.raises(OasisException, match="zero-damage"):
            get_conditional_vulns(LocalStorage(d), _damage_bins(3, first_bin_is_zero_damage=False))


# NB a column that is DEFINED but does not sum to 1, and a duplicated
# (vulnerability_id, source_damage_bin, damage_bin) triple, are both rejected by the csv -> bin
# converter -- see tests/pytools/converters/test_converters.py, where the equivalent
# vulnerability.csv checks live. The loader does not re-check them.
def test_get_conditional_vulns_bin_matches_csv():
    """The binary loader (fixed 4-byte int32 header, then vulnerability_dtype records) yields the
    same transition matrix as the CSV loader."""
    from oasislmf.pytools.gulmc.structure import get_conditional_vulns
    from oasislmf.pytools.common.data import vulnerability_dtype
    from oasis_data_manager.filestore.backends.local import LocalStorage
    # (vulnerability_id, source damage bin = intensity_bin_id, dependent damage bin, probability)
    rows = [(7, 1, 1, 0.8), (7, 1, 2, 0.2), (7, 2, 2, 0.5), (7, 2, 3, 0.5), (7, 3, 3, 1.0)]
    with tempfile.TemporaryDirectory() as d_csv, tempfile.TemporaryDirectory() as d_bin:
        _write_conditional_vuln_csv(d_csv, rows)
        arr_csv, ids_csv = get_conditional_vulns(LocalStorage(d_csv), _damage_bins(3))

        recs = np.array(rows, dtype=vulnerability_dtype)
        with open(Path(d_bin) / 'conditional_vulnerability.bin', 'wb') as f:
            np.array([3], dtype=np.int32).tofile(f)  # 4-byte header: num_damage_bins
            recs.tofile(f)
        arr_bin, ids_bin = get_conditional_vulns(LocalStorage(d_bin), _damage_bins(3))

    np.testing.assert_array_equal(ids_bin, ids_csv)
    np.testing.assert_allclose(arr_bin, arr_csv)


# --------------------------------------------------------------------------------------
# dependent-damage axis alignment (conditional matrix vs vulnerability array width)
# --------------------------------------------------------------------------------------
def _cond_matrix(num_damage_bins, num_source_bins=None, top_bin_prob=0.0):
    """A single conditional vulnerability: source bin 1 -> dependent bin 1, plus an optional
    probability on the top dependent damage bin."""
    from oasislmf.pytools.common.data import oasis_float
    arr = np.zeros((1, num_damage_bins, num_source_bins or num_damage_bins), dtype=oasis_float)
    arr[0, 0, 0] = 1.0
    if top_bin_prob:
        arr[0, num_damage_bins - 1, 0] = top_bin_prob
    return arr


def test_align_conditional_damage_axis_is_identity_when_widths_match():
    from oasislmf.pytools.gulmc.structure import align_conditional_damage_axis
    arr = _cond_matrix(3)
    assert align_conditional_damage_axis(arr, 3) is arr
    # an empty matrix (no conditional file) is passed through whatever the widths
    empty = np.zeros((0, 3, 3), dtype=arr.dtype)
    assert align_conditional_damage_axis(empty, 2) is empty


def test_align_conditional_damage_axis_drops_unreachable_top_bins():
    """The vulnerability data may declare fewer damage bins than the damage_bin_dict (e.g. a
    vulnerability.csv whose top damage bin is unused). The kernel copies a conditional column into
    a vuln_pdf row of that width, so the unused tail must be dropped, not left to mismatch."""
    from oasislmf.pytools.gulmc.structure import align_conditional_damage_axis
    arr = _cond_matrix(3)
    aligned = align_conditional_damage_axis(arr, 2)
    assert aligned.shape == (1, 2, 3)
    np.testing.assert_allclose(aligned[0, :, 0], [1.0, 0.0])


def test_align_conditional_damage_axis_rejects_dropping_probability():
    from oasislmf.pytools.gulmc.structure import align_conditional_damage_axis
    arr = _cond_matrix(3, top_bin_prob=0.5)  # dependent damage bin 3 carries probability
    with pytest.raises(OasisException, match="damage bin"):
        align_conditional_damage_axis(arr, 2)


def test_align_conditional_damage_axis_rejects_oversized_vulnerability_axis():
    """If the vulnerability data declares more damage bins than the damage_bin_dict, a source
    coverage can sample a damage bin with no column in the conditional matrix."""
    from oasislmf.pytools.gulmc.structure import align_conditional_damage_axis
    with pytest.raises(OasisException, match="damage_bin_dict"):
        align_conditional_damage_axis(_cond_matrix(2), 3)


# --------------------------------------------------------------------------------------
# dense vuln index -> conditional row mapping
# --------------------------------------------------------------------------------------
def _items_for_cond_idx(rows):
    """rows: (vulnerability_id, vulnerability_idx, areaperil_agg_vuln_idx)."""
    return np.array(rows, dtype=[('vulnerability_id', 'i4'), ('vulnerability_idx', 'i4'),
                                 ('areaperil_agg_vuln_idx', 'i4')])


def test_build_vuln_idx_to_cond_idx_maps_conditional_vulns():
    from oasislmf.pytools.gulmc.structure import build_vuln_idx_to_cond_idx
    items = _items_for_cond_idx([(50, 0, -1), (101, 1, -1), (102, 2, -1)])
    got = build_vuln_idx_to_cond_idx(items, np.array([101, 102], dtype=np.int32), n_vulns=3)
    assert got.tolist() == [-1, 0, 1]


def test_build_vuln_idx_to_cond_idx_ignores_aggregate_items():
    """An aggregate item has no vulnerability_idx (generate_item_map assigns it only in the
    non-aggregate branch), so an aggregate vulnerability id colliding with a conditional one must
    not scatter through that unassigned index."""
    from oasislmf.pytools.gulmc.structure import build_vuln_idx_to_cond_idx
    # the aggregate item's id collides with conditional id 101; its vulnerability_idx is unset (0)
    items = _items_for_cond_idx([(101, 0, 7), (50, 0, -1), (102, 1, -1)])
    got = build_vuln_idx_to_cond_idx(items, np.array([101, 102], dtype=np.int32), n_vulns=2)
    assert got.tolist() == [-1, 1], "dense index 0 belongs to normal vuln 50, not to the aggregate"


def test_build_vuln_idx_to_cond_idx_no_conditional_vulns():
    from oasislmf.pytools.gulmc.structure import build_vuln_idx_to_cond_idx
    items = _items_for_cond_idx([(50, 0, -1)])
    got = build_vuln_idx_to_cond_idx(items, np.zeros(0, dtype=np.int32), n_vulns=2)
    assert got.tolist() == [-1, -1]


# --------------------------------------------------------------------------------------
# validation guards (validate_coverage_dependency contract)
# --------------------------------------------------------------------------------------
def _dependency_arrays():
    """A minimal valid coverage-dependency setup: item 1 a hazard-indexed source, item 2 a
    dependent linked to it with a conditional vulnerability. Returns the args for
    validate_coverage_dependency, which individual tests perturb to trip one guard."""
    items = np.zeros(2, dtype=[('item_id', 'i4'), ('coverage_id', 'i4'), ('vulnerability_id', 'i4'),
                               ('vulnerability_idx', 'i4'), ('areaperil_agg_vuln_idx', 'i4'),
                               ('source_item_id', 'i4')])
    items['item_id'] = [1, 2]
    items['coverage_id'] = [1, 2]
    items['vulnerability_id'] = [10, 20]
    items['vulnerability_idx'] = [0, 1]
    items['areaperil_agg_vuln_idx'] = [-1, -1]          # both non-aggregate
    items['source_item_id'] = [0, 1]                    # item 2 is driven by item 1
    vuln_idx_to_cond_idx = np.array([-1, 0], dtype='i8')  # vuln 0 normal, vuln 1 conditional
    return items, vuln_idx_to_cond_idx


def test_validate_coverage_dependency_accepts_valid_setup():
    from oasislmf.pytools.gulmc.manager import validate_coverage_dependency
    validate_coverage_dependency(*_dependency_arrays())  # must not raise


def test_validate_rejects_aggregate_dependent():
    from oasislmf.pytools.gulmc.manager import validate_coverage_dependency
    items, v2c = _dependency_arrays()
    items['areaperil_agg_vuln_idx'][1] = 0  # dependent uses an aggregate vulnerability
    with pytest.raises(OasisException):
        validate_coverage_dependency(items, v2c)


def test_validate_rejects_dependent_without_conditional_vuln():
    from oasislmf.pytools.gulmc.manager import validate_coverage_dependency
    items, v2c = _dependency_arrays()
    v2c[1] = -1  # dependent's vuln is not in the conditional file
    with pytest.raises(OasisException):
        validate_coverage_dependency(items, v2c)


def test_validate_rejects_unpaired_item_with_conditional_vuln():
    """An item with no source item cannot use a conditional vulnerability: there is no source
    damage bin to index it with. This is the check that catches a dependent the key server placed
    in a different areaperil from its source, which file generation leaves unpaired."""
    from oasislmf.pytools.gulmc.manager import validate_coverage_dependency
    items, v2c = _dependency_arrays()
    items['source_item_id'][1] = 0  # item 2 found no source but still carries a conditional vuln
    with pytest.raises(OasisException, match="no source item"):
        validate_coverage_dependency(items, v2c)


def test_validate_accepts_unpaired_item_with_hazard_indexed_vuln():
    """An item with no source and an ordinary hazard-indexed vulnerability is simply computed
    independently — file generation leaves it unpaired on purpose, and this is where that is
    resolved. A coverage may hold a mix of paired and unpaired items."""
    from oasislmf.pytools.gulmc.manager import validate_coverage_dependency
    items, v2c = _dependency_arrays()
    items['source_item_id'][1] = 0   # no source ...
    v2c[1] = -1                      # ... and its vulnerability is hazard-indexed
    validate_coverage_dependency(items, v2c)  # must not raise


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
        assert (contents[contents['loc_id'] == loc]['source_item_id'] == src).all()


def _gul_inputs_chain(building_tiv, contents_tiv, bi_tiv):
    """One location with a configured dependency CHAIN building (1) -> contents (3) -> BI (4)."""
    loc_df = pd.DataFrame({
        'PortNumber': ['1'], 'AccNumber': ['1'], 'LocNumber': ['1'],
        'CountryCode': ['GB'], 'LocCurrency': ['GBP'], 'LocPerilsCovered': ['WTC'],
        'BuildingTIV': [building_tiv], 'ContentsTIV': [contents_tiv],
        'OtherTIV': [0.0], 'BITIV': [bi_tiv],
    })
    exposure = OedExposure(location=loc_df, use_field=True)
    prepare_oed_exposure(exposure)
    keys_df = pd.DataFrame([
        {'loc_id': 1, 'peril_id': 'WTC', 'coverage_type_id': ct, 'areaperil_id': 1,
         'vulnerability_id': 1, 'status': 'success', 'message': ''} for ct in (1, 3, 4)
    ])
    gul = get_gul_input_items(exposure.location.dataframe, keys_df, damage_group_id_cols=['loc_id'],
                              coverage_dependency_settings=[(1, 3), (3, 4)])
    return gul.set_index('coverage_type_id')


def _gul_inputs_zero_tiv_building(perils_building, perils_contents, areaperil_contents):
    """One location, uninsured building (TIV 0) and insured contents, with the perils and the
    contents' areaperil under test."""
    loc_df = pd.DataFrame({
        'PortNumber': ['1'], 'AccNumber': ['1'], 'LocNumber': ['1'], 'CountryCode': ['GB'],
        'LocCurrency': ['GBP'], 'LocPerilsCovered': ['WTC;WSS'],
        'BuildingTIV': [0.0], 'ContentsTIV': [1000.0], 'OtherTIV': [0.0], 'BITIV': [0.0]})
    exposure = OedExposure(location=loc_df, use_field=True)
    prepare_oed_exposure(exposure)
    rows = [{'loc_id': 1, 'peril_id': p, 'coverage_type_id': 1, 'areaperil_id': 1,
             'vulnerability_id': 8} for p in perils_building]
    rows += [{'loc_id': 1, 'peril_id': p, 'coverage_type_id': 3,
              'areaperil_id': areaperil_contents, 'vulnerability_id': 101} for p in perils_contents]
    keys_df = pd.DataFrame(rows)
    keys_df['status'], keys_df['message'] = 'success', ''
    return get_gul_input_items(exposure.location.dataframe, keys_df, damage_group_id_cols=['loc_id'],
                               coverage_dependency_settings=[(1, 3)])


def test_zero_tiv_retention_only_keeps_rows_that_drive_something():
    """A zero-TIV source is retained to drive its dependent, so it is only worth retaining where a
    dependent can actually pair with it — same location, peril AND areaperil. Retaining more would
    add zero-TIV, zero-loss items that drive nothing, and they would flow on into IL and the
    summaries."""
    # every retained row must be named as a source by some dependent item
    for perils_b, perils_c, ap_c in [(['WTC', 'WSS'], ['WTC', 'WSS'], 1),
                                     (['WTC', 'WSS'], ['WTC'], 1),
                                     (['WTC'], ['WTC', 'WSS'], 1)]:
        gul = _gul_inputs_zero_tiv_building(perils_b, perils_c, ap_c)
        building = gul[gul['coverage_type_id'] == 1]
        named = {int(v) for v in gul['source_item_id'] if int(v) > 0}
        assert set(building['item_id']) <= named, \
            f"retained a zero-TIV source row that drives nothing ({perils_b}, {perils_c})"
        # ... and the perils that CAN pair are still retained
        assert set(building['peril_id']) == set(perils_b) & set(perils_c)


def test_zero_tiv_source_dropped_when_dependent_is_at_another_areaperil():
    """With the contents in a different cell the dependency cannot be built, so the uninsured
    building drives nothing and is dropped as an ordinary empty coverage — exactly as it would be
    with the feature switched off."""
    gul = _gul_inputs_zero_tiv_building(['WTC', 'WSS'], ['WTC', 'WSS'], areaperil_contents=2)
    assert gul[gul['coverage_type_id'] == 1].empty, "uninsured building drives nothing -> dropped"
    assert (gul[gul['coverage_type_id'] == 3]['source_item_id'] == 0).all()


def test_zero_tiv_retention_follows_a_dependency_chain():
    """Retention has to resolve from the insured end backwards. With a configured chain
    building -> contents -> BI and only BI insured, contents is retained as BI's source — and the
    building must then be retained as *contents'* source. Keeping contents while dropping the
    building would leave contents with a conditional vulnerability and no source, which gulmc
    rejects outright (validate_coverage_dependency).
    """
    cov = _gul_inputs_chain(building_tiv=0.0, contents_tiv=0.0, bi_tiv=200.0)
    assert set(cov.index) == {1, 3, 4}, "the whole chain is retained, not just the last zero-TIV link"
    # every dependent in the chain resolves to its source
    assert int(cov.loc[3, 'source_item_id']) == int(cov.loc[1, 'item_id'])
    assert int(cov.loc[4, 'source_item_id']) == int(cov.loc[3, 'item_id'])
    assert float(cov.loc[1, 'tiv']) == 0.0 and float(cov.loc[3, 'tiv']) == 0.0


def test_zero_tiv_retention_stops_where_the_chain_is_not_kept():
    """Retention is not unconditional: a zero-TIV source with no kept dependent anywhere down the
    chain is still dropped as an empty coverage."""
    # nothing insured below the building: contents and BI are both zero-TIV, so none is retained
    cov = _gul_inputs_chain(building_tiv=5000.0, contents_tiv=0.0, bi_tiv=0.0)
    assert set(cov.index) == {1}, "only the insured building survives"


def test_duplicate_keys_rows_do_not_break_link_resolution():
    """A keys file may hold several rows for one (loc_id, peril_id, coverage_type_id) — they share
    an item_id and get deduped at the end of get_gul_input_items. Link resolution must work off the
    same first-occurrence view: otherwise a duplicated source row fans its merge out and breaks the
    positional assignment, and duplicate item_id labels cannot be reindexed. Configuring the feature
    must not turn an input the pipeline already tolerates into a crash."""
    def build(rows):
        loc_df = pd.DataFrame({
            'PortNumber': ['1'], 'AccNumber': ['1'], 'LocNumber': ['1'], 'CountryCode': ['GB'],
            'LocCurrency': ['GBP'], 'LocPerilsCovered': ['WTC'], 'BuildingTIV': [220000.0],
            'ContentsTIV': [50000.0], 'OtherTIV': [0.0], 'BITIV': [0.0]})
        exposure = OedExposure(location=loc_df, use_field=True)
        prepare_oed_exposure(exposure)
        keys_df = pd.DataFrame(rows)
        keys_df['status'], keys_df['message'] = 'success', ''
        return exposure.location.dataframe, keys_df

    src = {'loc_id': 1, 'peril_id': 'WTC', 'coverage_type_id': 1, 'vulnerability_id': 8}
    dep = {'loc_id': 1, 'peril_id': 'WTC', 'coverage_type_id': 3, 'vulnerability_id': 101}
    cases = {
        # two source rows for one (loc, peril, coverage type) -> one duplicated item_id
        "duplicate source row": [{**src, 'areaperil_id': 1}, {**src, 'areaperil_id': 2},
                                 {**dep, 'areaperil_id': 1}],
        "duplicate dependent row": [{**src, 'areaperil_id': 1}, {**dep, 'areaperil_id': 1},
                                    {**dep, 'areaperil_id': 2}],
    }
    for label, rows in cases.items():
        loc_df, keys_df = build(rows)
        configured = get_gul_input_items(loc_df, keys_df, damage_group_id_cols=['loc_id'],
                                         coverage_dependency_settings=[(1, 3)])
        loc_df, keys_df = build(rows)
        unconfigured = get_gul_input_items(loc_df, keys_df, damage_group_id_cols=['loc_id'],
                                           coverage_dependency_settings=None)
        assert len(configured) == len(unconfigured), \
            f"{label}: configuring the dependency changed how duplicates are deduped"
        assert configured['item_id'].is_unique, f"{label}: duplicate item_id survived"


def test_areaperil_mismatch_leaves_dependent_unpaired(caplog):
    """A dependent must share its source's areaperil: its damage is driven by the source's, which
    belongs to the source's cell. Where the key server places the configured pair in different
    cells the item is left unpaired and computed independently, logged at INFO — a coverage type
    may deliberately carry a conditional vulnerability where the cells align and a hazard-indexed
    one where they do not. Only the broken combination (unpaired AND conditional) is refused, by
    gulmc, which is the stage that knows which vulnerabilities are conditional."""
    keys = []
    for loc in (1, 2, 3):
        keys.append({'loc_id': loc, 'coverage_type_id': 1, 'areaperil_id': 1})
        # loc 3 contents is geocoded to a different areaperil than its building
        keys.append({'loc_id': loc, 'coverage_type_id': 3, 'areaperil_id': 2 if loc == 3 else 1})
    # at_level must name the logger that actually emits, not an ancestor: any gulmc/gulpy run in
    # the same process leaves every 'oasislmf.*' logger pinned at WARNING
    # (pytools.utils.logging_reset_handlers restores their handlers and propagate flag but not
    # their level), so raising only the parent's level leaves this record filtered at source.
    emitter = 'oasislmf.preparation.gul_inputs'
    with caplog.at_level(logging.INFO, logger=emitter):
        gul = _gul_inputs_for_keys(keys)
    contents = gul[gul['coverage_type_id'] == 3].set_index('loc_id')['source_item_id']
    assert (contents.loc[[1, 2]] > 0).all(), "matching areaperil -> paired"
    assert int(contents.loc[3]) == 0, "different areaperil -> left unpaired"
    from_prep = [r for r in caplog.records if r.name == emitter]
    assert any("different areaperils" in r.getMessage() for r in from_prep), \
        f"expected an INFO record about the mismatch, got {[r.getMessage() for r in from_prep]}"
    assert not [r for r in from_prep if r.levelno >= logging.WARNING], \
        "a supported configuration must not warn"


def test_dependency_active_where_areaperils_match():
    """With the source and dependent at the same areaperil everywhere, every dependent item pairs
    with the source item for its own peril."""
    keys = []
    for loc in (1, 2, 3):
        keys.append({'loc_id': loc, 'coverage_type_id': 1, 'areaperil_id': 1})
        keys.append({'loc_id': loc, 'coverage_type_id': 3, 'areaperil_id': 1})
    gul = _gul_inputs_for_keys(keys)
    contents = gul[gul['coverage_type_id'] == 3]
    building = gul[gul['coverage_type_id'] == 1].set_index('loc_id')['item_id']
    assert (contents['source_item_id'] > 0).all()
    for row in contents.itertuples():
        assert int(row.source_item_id) == int(building.loc[row.loc_id])


def test_dependent_without_a_source_coverage_is_left_independent():
    """A location may hold the dependent coverage and not the source (contents but no building).
    gul_inputs cannot know whether that coverage's vulnerability is a conditional one — only the
    model's static data says so — so it leaves the item unpaired and lets gulmc validate it."""
    keys = [{'loc_id': 1, 'coverage_type_id': 1, 'areaperil_id': 1},
            {'loc_id': 1, 'coverage_type_id': 3, 'areaperil_id': 1},
            {'loc_id': 2, 'coverage_type_id': 3, 'areaperil_id': 1}]  # loc 2: contents only
    gul = _gul_inputs_for_keys(keys)
    contents = gul[gul['coverage_type_id'] == 3].set_index('loc_id')['source_item_id']
    assert int(contents.loc[1]) > 0, "loc 1 has a building -> paired"
    assert int(contents.loc[2]) == 0, "loc 2 has no building -> left independent, no error"


def test_source_multiplicity_pairs_on_peril_not_position():
    """The source (building) has two items at ONE areaperil — two perils geocoded to the same cell
    — while the dependent (contents) has one. The link is per item, so the dependent pairs with
    the source item for its own peril; the source's other item simply drives nothing. Pairing by
    position within the areaperil could not distinguish the two, which is why the link is
    resolved here rather than inferred in the engine."""
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
    # the dependent is WTC: it must pair with the building's WTC item, not its WSS one
    wtc_building = int(building[building['peril_id'] == 'WTC']['item_id'].iloc[0])
    assert int(contents['source_item_id'].iloc[0]) == wtc_building, \
        "dependent must pair with the source item for its own peril"


# --------------------------------------------------------------------------------------
# dependency forest
# --------------------------------------------------------------------------------------
def test_build_coverage_dependency_forest():
    # coverage 100 root; 101 -> 100; 102 -> 101 (chain); 200 root; 201 -> 200.
    # one item per coverage, item_id = coverage_id, so the item links mirror the coverage links.
    items = np.array([(100, 100, 0, 7), (101, 101, 100, 7), (102, 102, 101, 7), (200, 200, 0, 7), (201, 201, 200, 7)],
                     dtype=[('item_id', 'i4'), ('coverage_id', 'u4'), ('source_item_id', 'i4'),
                            ('areaperil_id', 'u4')])
    src, off, data, source_item_idx = build_coverage_dependency_forest(items, 203)
    # the resolved source item index points at the row holding that item
    assert source_item_idx.tolist() == [-1, 0, 1, -1, 3]
    assert [int(src[i]) for i in (100, 101, 102, 200, 201)] == [0, 100, 101, 0, 200]

    def children(p):
        return list(data[off[p]:off[p + 1]])
    assert children(100) == [101]
    assert children(101) == [102]
    assert children(200) == [201]
    assert children(102) == []


def test_forest_shared_source():
    # a single source (100) may drive multiple dependents (101, 102): a branch, not a cycle.
    items = np.array([(100, 100, 0, 7), (101, 101, 100, 7), (102, 102, 100, 7)],
                     dtype=[('item_id', 'i4'), ('coverage_id', 'u4'), ('source_item_id', 'i4'),
                            ('areaperil_id', 'u4')])
    src, off, data, _ = build_coverage_dependency_forest(items, 103)
    assert [int(src[i]) for i in (100, 101, 102)] == [0, 100, 100]
    assert sorted(data[off[100]:off[101]].tolist()) == [101, 102]


def test_forest_rejects_cycles():
    # a cyclic dependency (coverage 1 -> 2 -> 1) must be rejected when the forest is built
    items = np.array([(1, 1, 2, 7), (2, 2, 1, 7)],
                     dtype=[('item_id', 'i4'), ('coverage_id', 'u4'), ('source_item_id', 'i4'),
                            ('areaperil_id', 'u4')])
    with pytest.raises(OasisException):
        build_coverage_dependency_forest(items, 3)


def test_forest_rejects_a_source_item_at_another_areaperil():
    """A dependent item must sit at its source item's areaperil. The gulmc kernel reuses its
    per-event item position map and its depth-indexed source stacks without clearing them, which is
    only sound because a linked pair is present or absent together; and fix for an absent source
    coverage treats such a coverage as a root on the same grounds. Violated silently, losses become
    event-order dependent, so it is rejected like every other link malformation."""
    items = np.array([(1, 1, 0, 7), (2, 2, 1, 9)],
                     dtype=[('item_id', 'i4'), ('coverage_id', 'u4'), ('source_item_id', 'i4'),
                            ('areaperil_id', 'u4')])
    with pytest.raises(OasisException, match="different areaperil"):
        build_coverage_dependency_forest(items, 3)


def test_forest_rejects_a_coverage_with_two_source_coverages():
    """The dependency forest is coverage-level, so all the linked items of one coverage must resolve
    to the same source coverage. File generation guarantees it; malformed input would otherwise be
    absorbed by the scatter, silently driving some items from the wrong coverage's depth row."""
    # coverage 3's two items are linked to items of coverage 1 and coverage 2 respectively
    items = np.array([(1, 1, 0, 7), (2, 2, 0, 7), (3, 3, 1, 7), (4, 3, 2, 7)],
                     dtype=[('item_id', 'i4'), ('coverage_id', 'u4'), ('source_item_id', 'i4'),
                            ('areaperil_id', 'u4')])
    with pytest.raises(OasisException, match="more than one source coverage"):
        build_coverage_dependency_forest(items, 4)


def test_forest_rejects_nonexistent_source_item():
    # a source_item_id that is not in the items table is malformed/stale input -> fail loud, not
    # silently demote the dependent to independent. An OasisException, not an assert: the check
    # must survive python -O, where a bad id would reach the njit depth walk and index out of
    # bounds with no boundscheck.
    items = np.array([(1, 1, 0, 7), (2, 2, 99, 7)],
                     dtype=[('item_id', 'i4'), ('coverage_id', 'u4'), ('source_item_id', 'i4'),
                            ('areaperil_id', 'u4')])
    with pytest.raises(OasisException, match="do not exist"):
        build_coverage_dependency_forest(items, 3)


def test_forest_rejects_self_reference():
    # a coverage listing itself as its own source is malformed input -> fail loud
    items = np.array([(1, 1, 0, 7), (2, 2, 2, 7)],
                     dtype=[('item_id', 'i4'), ('coverage_id', 'u4'), ('source_item_id', 'i4'),
                            ('areaperil_id', 'u4')])
    with pytest.raises(OasisException, match="itself"):
        build_coverage_dependency_forest(items, 3)


# --------------------------------------------------------------------------------------
# end-to-end behaviour
# --------------------------------------------------------------------------------------
def _write_correlations(run_dir, dependent_to_source):
    """Write a correlations file for the model, linking each dependent ITEM to its source item.

    The link is per item: a dependent item pairs with the source coverage's item at the same
    areaperil. Expressed as coverage pairs for brevity in the tests, then resolved here.

    Args:
        run_dir (Path): run directory containing input/items.csv.
        dependent_to_source (dict[int, int]): mapping dependent coverage_id -> source coverage_id.
    """
    items = pd.read_csv(run_dir / 'input' / 'items.csv')
    corr = np.zeros(len(items), dtype=correlations_dtype)
    corr['item_id'] = items['item_id'].to_numpy()
    for dep_cov, src_cov in dependent_to_source.items():
        for row in items[items['coverage_id'] == dep_cov].itertuples():
            match = items[(items['coverage_id'] == src_cov)
                          & (items['areaperil_id'] == row.areaperil_id)]['item_id']
            if len(match):
                corr['source_item_id'][corr['item_id'] == row.item_id] = int(match.iloc[0])
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


@pytest.mark.parametrize("source_damage_type,absolute_bins", [
    (0, False), (0, True), (1, False), (2, True), (3, False),
], ids=["default-relative-bins", "default-absolute-bins", "relative", "absolute", "duration"])
def test_zero_tiv_source_reports_no_loss_but_still_drives_dependent(source_damage_type, absolute_bins):
    """A source retained with zero TIV must report no loss of its own while still driving its
    dependent, for every damage type.

    Relative and duration functions get the zero from their tiv factor, but absolute functions
    carry currency directly in the damage bins, so without an explicit zero an uninsured building
    would emit non-zero gul (and the tiv split that caps absolute losses at the tiv is itself
    skipped when tiv is 0). That covers both ways a scaling of 1 is reached: damage_type 2, and
    damage_type 0 (default) with currency-scale bins, whose last bin_to exceeds 1. The dependent is
    unaffected either way: it is driven by the source's sampled damage bin, which does not depend
    on the loss scaling.
    """
    with tempfile.TemporaryDirectory() as t:
        run_dir = Path(t) / 'assets'
        shutil.copytree(SRC_MODEL, run_dir)
        shutil.rmtree(run_dir / 'input' / 'gulmc_structure', ignore_errors=True)

        items = pd.read_csv(run_dir / 'input' / 'items.csv')
        items.loc[(items.coverage_id == 2) & (items.areaperil_id == 154), 'vulnerability_id'] = 101
        items.loc[(items.coverage_id == 2) & (items.areaperil_id == 54), 'vulnerability_id'] = 102
        items.to_csv(run_dir / 'input' / 'items.csv', index=False)
        (run_dir / 'input' / 'items.bin').unlink()

        dbd_path = run_dir / 'static' / 'damage_bin_dict.csv'
        dbd = pd.read_csv(dbd_path)
        n_damage_bins = len(dbd)
        if absolute_bins:  # currency-scale bins (bin_to > 1)
            to = np.arange(n_damage_bins, dtype='f8') * 1000.0
            frm = np.concatenate([[0.0], to[:-1]])
            dbd['bin_from'], dbd['bin_to'], dbd['interpolation'] = frm, to, (frm + to) / 2
        dbd['damage_type'] = source_damage_type
        dbd.to_csv(dbd_path, index=False)
        (run_dir / 'static' / 'damage_bin_dict.bin').unlink()

        with open(run_dir / 'static' / 'conditional_vulnerability.csv', 'w') as f:
            f.write('vulnerability_id,source_damage_bin,damage_bin,probability\n')
            for vid in (101, 102):  # identity: source bin k -> dependent bin k
                for k in range(1, n_damage_bins + 1):
                    f.write(f'{vid},{k},{k},1.0\n')

        # coverage 1 is the source: make it uninsured (the retained zero-TIV driver)
        cov = pd.read_csv(run_dir / 'input' / 'coverages.csv')
        cov.loc[cov.coverage_id == 1, 'tiv'] = 0.0
        cov.to_csv(run_dir / 'input' / 'coverages.csv', index=False)
        (run_dir / 'input' / 'coverages.bin').unlink()

        _write_correlations(run_dir, {2: 1})
        df = _run(run_dir, effective_damageability=False)

    src = df[df['item_id'] == 1]           # coverage 1, tiv 0
    dep = df[df['item_id'] == 3]           # coverage 2, tiv > 0, dependent on coverage 1
    assert len(src) > 0 and len(dep) > 0

    # the uninsured source reports no loss anywhere: samples and the loss-valued special sidx
    # (mean, std dev, tiv, max loss). sidx -4 is a probability, not a loss, so it is left alone.
    assert (src[src['sidx'] > 0]['loss'] == 0).all()
    loss_sidx = src[src['sidx'].isin([MEAN_IDX, STD_DEV_IDX, TIV_IDX, MAX_LOSS_IDX])]
    assert (loss_sidx['loss'] == 0).all(), f"zero-tiv source reported loss on {loss_sidx.to_dict('records')}"

    # ... and still drives its dependent, which is insured and does have losses
    assert (dep[dep['sidx'] > 0]['loss'] > 0).any()


def test_dependent_coverage_runs_when_its_source_coverage_is_absent_from_the_event():
    """A dependent coverage can be present in an event through unpaired items alone, with its
    source coverage contributing nothing — the source's areaperil simply is not in that event's
    footprint. Its present items are all unpaired (a paired item shares its source item's
    areaperil, so the two are present or absent together), so the coverage is computed as a root.
    It must not abort the run.
    """
    with tempfile.TemporaryDirectory() as t:
        run_dir = Path(t) / 'assets'
        shutil.copytree(SRC_MODEL, run_dir)
        shutil.rmtree(run_dir / 'input' / 'gulmc_structure', ignore_errors=True)

        # coverage 1's only item sits at areaperil 1, which event 1's footprint does not contain;
        # coverage 2 has a paired item there and an unpaired one at areaperil 4, which it does
        items = pd.DataFrame({'item_id': [1, 2, 3], 'coverage_id': [1, 2, 2],
                              'areaperil_id': [1, 1, 4], 'vulnerability_id': [8, 101, 2],
                              'group_id': [11, 22, 22]})
        items.to_csv(run_dir / 'input' / 'items.csv', index=False)
        (run_dir / 'input' / 'items.bin').unlink()
        pd.DataFrame({'coverage_id': [1, 2], 'tiv': [220000.0, 790000.0]}).to_csv(
            run_dir / 'input' / 'coverages.csv', index=False)
        (run_dir / 'input' / 'coverages.bin').unlink()

        n_damage_bins = len(pd.read_csv(run_dir / 'static' / 'damage_bin_dict.csv'))
        with open(run_dir / 'static' / 'conditional_vulnerability.csv', 'w') as f:
            f.write('vulnerability_id,source_damage_bin,damage_bin,probability\n')
            for k in range(1, n_damage_bins + 1):
                f.write(f'101,{k},{k},1.0\n')

        corr = np.zeros(3, dtype=correlations_dtype)
        corr['item_id'] = [1, 2, 3]
        corr['source_item_id'] = [0, 1, 0]          # item 2 paired to item 1; item 3 unpaired
        corr.tofile(run_dir / 'input' / 'correlations.bin')
        pd.DataFrame({k: corr[k] for k in corr.dtype.names}).to_csv(
            run_dir / 'input' / 'correlations.csv', index=False)

        # event 1's footprint holds areaperil 4 but not areaperil 1
        fp = pd.read_csv(run_dir / 'static' / 'footprint.csv')
        present = set(fp[fp.event_id == 1]['areaperil_id'])
        assert 4 in present and 1 not in present, "event 1 is expected to miss areaperil 1"
        np.array([1], dtype='i4').tofile(run_dir / 'input' / 'events.bin')

        df = _run(run_dir, effective_damageability=False)   # must not raise

    assert sorted(df['item_id'].unique()) == [3], "only the unpaired item is in this event"
    assert (df[(df['item_id'] == 3) & (df['sidx'] > 0)]['loss'] > 0).any(), \
        "the unpaired item is sampled from the footprint hazard, not zeroed"


def test_conditional_vulnerability_csv_header_is_detected():
    """A headerless conditional_vulnerability.csv must load identically to a headered one.
    `bintocsv conditionalvulnerability --noheader` writes one, and unconditionally skipping the
    first row would silently drop a whole source-bin -> damage-bin transition."""
    from oasislmf.pytools.gulmc.structure import get_conditional_vulns
    from oasislmf.pytools.getmodel.manager import get_damage_bins
    from oasis_data_manager.filestore.backends.local import LocalStorage
    rows = "7,1,2,1.0\n7,2,2,1.0\n7,3,3,1.0\n"
    header = "vulnerability_id,source_damage_bin,damage_bin,probability\n"
    loaded = {}
    for label, text in (("headered", header + rows), ("headerless", rows)):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'damage_bin_dict.csv').write_text(
                (SRC_MODEL / 'static' / 'damage_bin_dict.csv').read_text())
            (d / 'conditional_vulnerability.csv').write_text(text)
            arr, _ = get_conditional_vulns(LocalStorage(str(d)), get_damage_bins(LocalStorage(str(d))))
            loaded[label] = arr
    np.testing.assert_array_equal(loaded["headered"], loaded["headerless"])
    # source damage bin 1 maps to dependent damage bin 2, not bin 1 (which the fill would give)
    np.testing.assert_allclose(loaded["headerless"][0, :3, 0], [0.0, 1.0, 0.0])


def test_mixed_conditional_and_hazard_indexed_on_one_coverage_type():
    """One coverage type may carry a conditional vulnerability where the dependency applies and a
    hazard-indexed one where it does not, in the same run. Coverage 2's item at areaperil 154 uses
    conditional vuln 101 and is driven by coverage 1; its item at areaperil 54 uses the
    hazard-indexed vuln 2 and is sampled from the footprint hazard.

    The conditional item's tracking is asserted against the independent item's tracking of the same
    source, measured in the same run: that is the chance baseline, so the test does not depend on
    how precisely a damage bin can be recovered from a loss.
    """
    with tempfile.TemporaryDirectory() as t:
        run_dir = Path(t) / 'assets'
        shutil.copytree(SRC_MODEL, run_dir)
        shutil.rmtree(run_dir / 'input' / 'gulmc_structure', ignore_errors=True)

        items = pd.read_csv(run_dir / 'input' / 'items.csv')
        items = items[items['coverage_id'].isin([1, 2])].copy()
        items.loc[(items.coverage_id == 2) & (items.areaperil_id == 154), 'vulnerability_id'] = 101
        items.loc[(items.coverage_id == 2) & (items.areaperil_id == 54), 'vulnerability_id'] = 2
        items.to_csv(run_dir / 'input' / 'items.csv', index=False)
        (run_dir / 'input' / 'items.bin').unlink()

        n_damage_bins = len(pd.read_csv(run_dir / 'static' / 'damage_bin_dict.csv'))
        with open(run_dir / 'static' / 'conditional_vulnerability.csv', 'w') as f:
            f.write('vulnerability_id,source_damage_bin,damage_bin,probability\n')
            for k in range(1, n_damage_bins + 1):   # identity: the dependent mirrors its source
                f.write(f'101,{k},{k},1.0\n')

        # pair only the areaperil-154 item of coverage 2; leave its areaperil-54 item independent
        source_item = int(items[(items.coverage_id == 1) & (items.areaperil_id == 154)]['item_id'].iloc[0])
        cond_item = int(items[(items.coverage_id == 2) & (items.areaperil_id == 154)]['item_id'].iloc[0])
        indep_item = int(items[(items.coverage_id == 2) & (items.areaperil_id == 54)]['item_id'].iloc[0])
        corr = np.zeros(len(items), dtype=correlations_dtype)
        corr['item_id'] = items['item_id'].to_numpy()
        corr['source_item_id'][corr['item_id'] == cond_item] = source_item
        corr.tofile(run_dir / 'input' / 'correlations.bin')
        pd.DataFrame({k: corr[k] for k in corr.dtype.names}).to_csv(
            run_dir / 'input' / 'correlations.csv', index=False)

        out = run_dir / 'out.bin'
        run_gulmc(run_dir=run_dir, ignore_file_type=set(),
                  file_in=run_dir / 'input' / 'events.bin', file_out=out,
                  sample_size=500, loss_threshold=0., alloc_rule=0,  # no tiv split to distort ratios
                  debug=0, random_generator=0, ignore_correlation=False,
                  effective_damageability=False)
        bintocsv(out, run_dir / 'out.csv', 'gul')
        df = pd.read_csv(run_dir / 'out.csv')
        bin_to = pd.read_csv(run_dir / 'static' / 'damage_bin_dict.csv')['bin_to'].to_numpy()

    cov_tiv = pd.read_csv(SRC_MODEL / 'input' / 'coverages.csv').set_index('coverage_id')['tiv']

    def bins(item_id, coverage_id):
        d = df[(df['item_id'] == item_id) & (df['sidx'] > 0)].sort_values(['event_id', 'sidx'])
        return np.searchsorted(bin_to, d['loss'].to_numpy() / cov_tiv[coverage_id], side='left')

    src, cond, indep = bins(source_item, 1), bins(cond_item, 2), bins(indep_item, 2)

    def tracking(a, b):
        n = min(len(a), len(b))
        diff = np.abs(a[:n].astype(int) - b[:n].astype(int))
        return (diff == 0).mean(), diff.mean()

    cond_exact, cond_mean = tracking(src, cond)
    chance_exact, chance_mean = tracking(src, indep)   # same coverage type, no source
    assert cond_exact > 0.7 and cond_mean < 0.5, \
        f"conditional item should follow its source (exact {cond_exact:.3f}, mean|d| {cond_mean:.3f})"
    assert cond_exact > 2 * chance_exact and cond_mean < chance_mean / 2, \
        (f"tracking must beat chance (conditional {cond_exact:.3f}/{cond_mean:.3f} vs "
         f"chance {chance_exact:.3f}/{chance_mean:.3f})")
    assert (df[(df['item_id'] == indep_item) & (df['sidx'] > 0)]['loss'] > 0).any(), \
        "the hazard-indexed item on the same coverage type must still be sampled, not zeroed"


def test_conditional_dependency_eff_dam_marginal():
    """Effective damageability supports dependents, but marginal-only. With an identity transition
    matrix the dependent's eff-damage distribution equals the source's, so their sampled-bin
    DISTRIBUTIONS match — yet the per-sample comonotonic tie that full Monte Carlo produces is
    absent. Contrast the two modes on the same setup: full MC follows the source's bin per sample
    (>0.9 exact), eff-dam does not.
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

        dbd = pd.read_csv(run_dir / 'static' / 'damage_bin_dict.csv')
        n_damage_bins = len(dbd)
        with open(run_dir / 'static' / 'conditional_vulnerability.csv', 'w') as f:
            f.write('vulnerability_id,source_damage_bin,damage_bin,probability\n')
            for vid in (101, 102):  # identity: source bin k -> dependent bin k
                for k in range(1, n_damage_bins + 1):
                    f.write(f'{vid},{k},{k},1.0\n')
        _write_correlations(run_dir, {2: 1})

        cov_tiv = pd.read_csv(run_dir / 'input' / 'coverages.csv').set_index('coverage_id')['tiv']
        bin_to = dbd['bin_to'].to_numpy()

        def sampled_bins(df, item_id, coverage_id):
            d = df[(df['item_id'] == item_id) & (df['sidx'] > 0)].sort_values(['event_id', 'sidx'])
            return np.searchsorted(bin_to, d['loss'].to_numpy() / cov_tiv[coverage_id], side='left').astype(int)

        def source_and_dependent_bins(effective_damageability):
            df = _run(run_dir, effective_damageability=effective_damageability)
            # item 1 = source (coverage 1, areaperil 154); item 3 = dependent (coverage 2, areaperil 154)
            src, dep = sampled_bins(df, 1, 1), sampled_bins(df, 3, 2)
            n = min(len(src), len(dep))
            return src[:n], dep[:n]

        src_e, dep_e = source_and_dependent_bins(True)   # effective damageability
        src_f, dep_f = source_and_dependent_bins(False)  # full Monte Carlo
        assert len(src_e) > 0 and len(src_f) > 0

        # eff-dam dependent is correct at the MARGINAL level: identity matrix => the dependent's
        # damage distribution equals the source's, so sorted samples (order statistics) coincide
        # even though the per-sample pairing does not.
        order_stat_diff = np.abs(np.sort(src_e) - np.sort(dep_e)).mean()
        assert order_stat_diff < 0.3, \
            f"eff-dam dependent marginal should equal the source's (order-stat |d| {order_stat_diff:.3f})"

        # full MC is comonotonic (dependent follows the source's bin per sample); eff-dam is
        # marginal-only (independent per sample), so its per-sample match rate is markedly lower.
        match_full = float((src_f == dep_f).mean())
        match_eff = float((src_e == dep_e).mean())
        assert match_full > 0.9, f"full MC dependency should be comonotonic per sample (match {match_full:.3f})"
        assert match_full - match_eff > 0.3, \
            f"eff-dam should NOT be comonotonic per sample (full {match_full:.3f} vs eff-dam {match_eff:.3f})"


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
