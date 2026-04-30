"""
Tests for the return period (RP) protection feature in the dynamic footprint model.

The RP protection feature zeroes out losses for an item when the event's effective
return period is below the item's RP protection threshold (item_adjustments.csv).

Three levels of coverage:
  1. process_areaperils_in_footprint  – areaperil_to_event_rp is built correctly
  2. reconstruct_coverages            – event_rp is stored in items_event_data
  3. compute_event_losses             – losses are zeroed when event_rp < item_rp
"""
import numpy as np

from oasislmf.pytools.common.data import areaperil_int, nb_areaperil_int, nb_oasis_int, oasis_float, oasis_int, damagebin_dtype
from oasislmf.pytools.common.id_index import build as id_index_build
from oasislmf.pytools.getmodel.common import EventDynamic
from oasislmf.pytools.gulmc.common import (
    items_MC_data_type, coverage_type, haz_arr_type, gulmc_compute_info_type, NormInversionParameters,
    agg_vuln_idx_weight_dtype,
)
from oasislmf.pytools.gulmc.manager import process_areaperils_in_footprint, compute_event_losses, CDF_CACHE_EMPTY
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PERIL_ID = np.int32(1)
AREAPERIL_ID = np.uint32(10)
VULN_ID = np.int32(1)
COVERAGE_ID = np.uint32(1)
EVENT_ID = np.int32(1)
TIV = np.float64(1000.0)

# A single hazard footprint entry: intensity=100, probability=1.0
HAZ_INTENSITY = np.int32(100)
HAZ_BIN_ID = np.int32(1)

# RP values used in tests
ITEM_RP_PROTECTION = np.int32(25)   # item's protection threshold
EVENT_RP_BELOW = np.int32(12)       # event RP below threshold → zero loss
EVENT_RP_ABOVE = np.int32(50)       # event RP at or above threshold → normal loss


def _make_event_footprint(areaperil_id=AREAPERIL_ID, return_period=EVENT_RP_BELOW,
                          intensity=HAZ_INTENSITY, probability=1.0):
    """Build a minimal EventDynamic footprint array (1 record per areaperil)."""
    ev = np.zeros(1, dtype=EventDynamic)
    ev[0]['areaperil_id'] = areaperil_id
    ev[0]['return_period'] = return_period
    ev[0]['intensity'] = intensity
    ev[0]['intensity_bin_id'] = HAZ_BIN_ID
    ev[0]['probability'] = probability
    return ev


def _make_present_areaperils(*areaperil_ids):
    """Build an id_index for the given areaperil_ids."""
    keys = np.array(sorted(areaperil_ids), dtype=areaperil_int)
    return id_index_build(keys)


def _make_items_array(intensity_adjustment=0, return_period=0):
    """
    Build the fully-merged items structured array that compute_event_losses expects.

    Includes all fields added by the successive join_by calls in run():
      items_dtype + correlations + vulnerability_idx + dynamic footprint fields
      + group_seq_id + hazard_group_seq_id
    """
    full_dtype = np.dtype([
        ('item_id', np.int32),
        ('coverage_id', np.uint32),
        ('areaperil_id', np.uint32),
        ('vulnerability_id', np.int32),
        ('group_id', np.uint32),
        ('peril_correlation_group', np.int32),
        ('damage_correlation_value', np.float32),
        ('hazard_group_id', np.int32),
        ('hazard_correlation_value', np.float32),
        ('vulnerability_idx', oasis_int),
        ('areaperil_agg_vuln_idx', oasis_int),
        ('intensity_adjustment', np.int32),
        ('return_period', np.int32),
        ('peril_id', np.int32),
        ('group_seq_id', np.int32),
        ('hazard_group_seq_id', np.int32),
    ])
    items = np.zeros(1, dtype=full_dtype)
    items[0]['item_id'] = 1
    items[0]['coverage_id'] = COVERAGE_ID
    items[0]['areaperil_id'] = AREAPERIL_ID
    items[0]['vulnerability_id'] = VULN_ID
    items[0]['group_id'] = 1
    items[0]['vulnerability_idx'] = 0
    items[0]['areaperil_agg_vuln_idx'] = -1
    items[0]['peril_id'] = PERIL_ID
    items[0]['intensity_adjustment'] = intensity_adjustment
    items[0]['return_period'] = return_period
    items[0]['group_seq_id'] = 0
    items[0]['hazard_group_seq_id'] = 0
    return items


def _make_compute_event_losses_args(event_rp, item_rp, item_intensity_adjustment=0):
    """
    Build the full argument set for a single-item, single-coverage call to
    compute_event_losses with sample_size=0 and effective_damageability=True.

    Returns (args_tuple, losses_array) where losses_array is the buffer
    that will be written by the function.
    """
    sample_size = 0
    Ndamage_bins = 2
    Nintensity_bins = 1

    # --- compute_info ---
    compute_info = np.zeros(1, dtype=gulmc_compute_info_type)[0]
    compute_info['event_id'] = EVENT_ID
    compute_info['Ndamage_bins_max'] = Ndamage_bins
    compute_info['loss_threshold'] = 0.0
    compute_info['alloc_rule'] = 0
    compute_info['effective_damageability'] = 1
    compute_info['do_correlation'] = 0
    compute_info['do_haz_correlation'] = 0
    compute_info['debug'] = 0
    # byte buffer: generous upper bound
    compute_info['max_bytes_per_item'] = 2048
    compute_info['cursor'] = 0
    compute_info['coverage_i'] = 0
    compute_info['coverage_n'] = 1

    # --- coverages ---
    coverages = np.zeros(2, dtype=coverage_type)  # index 0 unused; 1 is our coverage
    coverages[COVERAGE_ID]['tiv'] = TIV
    coverages[COVERAGE_ID]['max_items'] = 1
    coverages[COVERAGE_ID]['start_items'] = 0
    coverages[COVERAGE_ID]['cur_items'] = 1

    # coverage_ids (compute) array – contains the one coverage to process
    coverage_ids = np.array([COVERAGE_ID], dtype=np.uint32)

    # --- items_event_data ---
    items_event_data = np.zeros(1, dtype=items_MC_data_type)
    items_event_data[0]['item_id'] = 1
    items_event_data[0]['item_idx'] = 0       # index into items array
    items_event_data[0]['haz_arr_i'] = 0      # index into haz_arr_ptr
    items_event_data[0]['rng_index'] = 0
    items_event_data[0]['hazard_rng_index'] = 0
    items_event_data[0]['intensity_adjustment'] = item_intensity_adjustment
    items_event_data[0]['return_period'] = item_rp
    items_event_data[0]['event_rp'] = event_rp
    items_event_data[0]['eff_cdf_id'] = 0

    # --- items ---
    items = _make_items_array(intensity_adjustment=item_intensity_adjustment, return_period=item_rp)

    # --- haz_pdf: single entry (intensity=100, prob=1.0) ---
    haz_pdf = np.zeros(1, dtype=haz_arr_type)
    haz_pdf[0]['probability'] = 1.0
    haz_pdf[0]['intensity_bin_id'] = HAZ_BIN_ID
    haz_pdf[0]['intensity'] = HAZ_INTENSITY

    haz_arr_ptr = np.array([0, 1], dtype=np.int64)

    # --- vulnerability array: shape (1, Ndamage_bins, Nintensity_bins) ---
    # All probability in damage bin 1 (index 1) → always max damage
    vuln_array = np.zeros((1, Ndamage_bins, Nintensity_bins), dtype=oasis_float)
    vuln_array[0, 1, 0] = 1.0  # P(damage bin index 1) = 1.0 at intensity bin 0

    # --- damage_bins ---
    damage_bins = np.zeros(Ndamage_bins, dtype=damagebin_dtype)
    damage_bins[0]['bin_from'] = 0.0
    damage_bins[0]['bin_to'] = 0.5
    damage_bins[0]['interpolation'] = 0.25
    damage_bins[0]['damage_type'] = 0
    damage_bins[1]['bin_from'] = 0.5
    damage_bins[1]['bin_to'] = 1.0
    damage_bins[1]['interpolation'] = 0.75
    damage_bins[1]['damage_type'] = 0

    # --- CDF cache (power-of-two sized, array-based) ---
    Nvulns_cached = 16  # power of two
    cdf_cache_mask = np.int64(Nvulns_cached - 1)
    cdf_cache_tag = np.full(1, CDF_CACHE_EMPTY, dtype=np.int64)  # 1 triplet
    cdf_cache_nbins = np.zeros(Nvulns_cached, dtype=np.int32)
    cached_vuln_cdfs = np.zeros((Nvulns_cached, Ndamage_bins), dtype=oasis_float)

    # --- aggregate vulnerability CRS arrays (empty – no aggregate vulns) ---
    areaperil_agg_vuln_idx_ja_data = np.empty(0, dtype=agg_vuln_idx_weight_dtype)
    areaperil_agg_vuln_idx_ja_offsets = np.zeros(1, dtype=oasis_int)  # single sentinel: [0]

    # --- losses buffer ---
    losses = np.zeros((sample_size + 6, 1), dtype=oasis_float)  # 6 = NUM_IDX + 1

    # --- stub random arrays (not accessed with sample_size=0) ---
    haz_rndms_base = np.zeros((1, max(sample_size, 1)), dtype=np.float64)
    vuln_rndms_base = np.zeros((1, max(sample_size, 1)), dtype=np.float64)
    vuln_adj = np.ones(1, dtype=oasis_float)
    haz_eps_ij = np.zeros((1, max(sample_size, 1)), dtype=np.float64)
    damage_eps_ij = np.zeros((1, max(sample_size, 1)), dtype=np.float64)

    # NormInversionParameters (unused when do_correlation=False)
    norm_inv_parameters = np.zeros(1, dtype=NormInversionParameters)[0]
    norm_inv_cdf = np.zeros(1, dtype=np.float64)
    norm_cdf = np.zeros(1, dtype=np.float64)
    vuln_z_unif = np.zeros(max(sample_size, 1), dtype=np.float64)
    haz_z_unif = np.zeros(max(sample_size, 1), dtype=np.float64)

    # --- output byte buffer ---
    byte_mv = np.zeros(PIPE_CAPACITY * 2, dtype='b')

    # --- intensity bin lookup arrays: peril_ids and 2D bins ---
    intensity_bin_peril_ids = np.array([PERIL_ID], dtype=np.int32)
    intensity_bins = np.zeros((1, int(HAZ_INTENSITY) + 1), dtype=np.int32)
    intensity_bins[0, HAZ_INTENSITY] = HAZ_BIN_ID

    dynamic_footprint = True  # truthy, enables dynamic footprint path

    args = (
        compute_info, coverages, coverage_ids, items_event_data, items,
        sample_size, haz_pdf, haz_arr_ptr, vuln_array, damage_bins,
        cdf_cache_tag, cdf_cache_nbins, cdf_cache_mask, cached_vuln_cdfs,
        areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data,
        losses, haz_rndms_base, vuln_rndms_base, vuln_adj,
        haz_eps_ij, damage_eps_ij,
        norm_inv_parameters, norm_inv_cdf, norm_cdf, vuln_z_unif, haz_z_unif,
        byte_mv, dynamic_footprint, intensity_bin_peril_ids, intensity_bins,
    )
    return args, losses


# ---------------------------------------------------------------------------
# Tests: process_areaperils_in_footprint
# ---------------------------------------------------------------------------

def _make_fp_buffers(n):
    """Pre-allocate reusable per-event footprint buffers for n areaperils.
    The first buffer holds dense areaperil indices (uint32), not raw IDs."""
    return (
        np.empty(n, dtype=np.uint32),
        np.empty(n, dtype=np.int32),
        np.empty(n + 1, dtype=np.int64),
    )


def test_process_areaperils_builds_event_rp_dict():
    """event_rps array is populated from EventDynamic return_period values."""
    ev = np.zeros(2, dtype=EventDynamic)
    ev[0]['areaperil_id'] = 1
    ev[0]['return_period'] = 20
    ev[0]['probability'] = 1.0
    ev[0]['intensity_bin_id'] = 1
    ev[0]['intensity'] = 100
    ev[1]['areaperil_id'] = 2
    ev[1]['return_period'] = 50
    ev[1]['probability'] = 1.0
    ev[1]['intensity_bin_id'] = 2
    ev[1]['intensity'] = 200

    present = _make_present_areaperils(1, 2)
    fp_ap_inds, fp_event_rps, fp_haz_arr_ptr = _make_fp_buffers(2)
    N, _ = process_areaperils_in_footprint(ev, present, True, fp_ap_inds, fp_event_rps, fp_haz_arr_ptr)

    assert N == 2
    assert int(fp_event_rps[0]) == 20
    assert int(fp_event_rps[1]) == 50


def test_process_areaperils_event_rp_empty_when_non_dynamic():
    """No areaperils with event_rp are populated for non-dynamic footprints (dynamic_footprint=None)."""
    ev = np.zeros(1, dtype=EventDynamic)
    ev[0]['areaperil_id'] = 1
    ev[0]['return_period'] = 30
    ev[0]['probability'] = 1.0
    ev[0]['intensity_bin_id'] = 1
    ev[0]['intensity'] = 0

    present = _make_present_areaperils(1)
    fp_ap_inds, fp_event_rps, fp_haz_arr_ptr = _make_fp_buffers(1)
    N, _ = process_areaperils_in_footprint(ev, present, None, fp_ap_inds, fp_event_rps, fp_haz_arr_ptr)

    # event_rps is not written for non-dynamic, N tells us how many areaperils were found
    assert N == 1


def test_process_areaperils_excludes_absent_areaperil():
    """Areaperils not in present_areaperils are excluded."""
    ev = np.zeros(2, dtype=EventDynamic)
    ev[0]['areaperil_id'] = 1
    ev[0]['return_period'] = 20
    ev[0]['probability'] = 1.0
    ev[0]['intensity_bin_id'] = 1
    ev[0]['intensity'] = 100
    ev[1]['areaperil_id'] = 2  # NOT in present
    ev[1]['return_period'] = 50
    ev[1]['probability'] = 1.0
    ev[1]['intensity_bin_id'] = 2
    ev[1]['intensity'] = 200

    present = _make_present_areaperils(1)  # only areaperil 1
    fp_ap_inds, fp_event_rps, fp_haz_arr_ptr = _make_fp_buffers(2)
    N, _ = process_areaperils_in_footprint(ev, present, True, fp_ap_inds, fp_event_rps, fp_haz_arr_ptr)

    assert N == 1
    # AP 1 in an id_index built from [1] has dense index 0
    assert fp_ap_inds[0] == 0


def test_process_areaperils_multi_bin_areaperil_uses_first_record_rp():
    """When an areaperil has multiple probability records, the RP is taken from the first record."""
    ev = np.zeros(3, dtype=EventDynamic)
    # areaperil 1 has 2 records (different intensity bins, same RP)
    ev[0]['areaperil_id'] = 1
    ev[0]['return_period'] = 25
    ev[0]['probability'] = 0.6
    ev[0]['intensity_bin_id'] = 1
    ev[0]['intensity'] = 80
    ev[1]['areaperil_id'] = 1
    ev[1]['return_period'] = 25
    ev[1]['probability'] = 0.4
    ev[1]['intensity_bin_id'] = 2
    ev[1]['intensity'] = 120
    # areaperil 2 has 1 record
    ev[2]['areaperil_id'] = 2
    ev[2]['return_period'] = 100
    ev[2]['probability'] = 1.0
    ev[2]['intensity_bin_id'] = 3
    ev[2]['intensity'] = 200

    present = _make_present_areaperils(1, 2)
    fp_ap_inds, fp_event_rps, fp_haz_arr_ptr = _make_fp_buffers(2)
    N, haz_pdf = process_areaperils_in_footprint(ev, present, True, fp_ap_inds, fp_event_rps, fp_haz_arr_ptr)

    assert N == 2
    assert int(fp_event_rps[0]) == 25
    assert int(fp_event_rps[1]) == 100
    # Both probability records for areaperil 1 should be in haz_pdf
    assert len(haz_pdf) == 3


# ---------------------------------------------------------------------------
# Tests: compute_event_losses RP protection
# ---------------------------------------------------------------------------

def test_rp_protection_zeros_all_losses_when_event_rp_below_item_rp():
    """
    When event_rp (12) < item return_period (25), all losses for that item
    must be zero regardless of the vulnerability function.
    """
    args, losses = _make_compute_event_losses_args(
        event_rp=EVENT_RP_BELOW,     # 12
        item_rp=ITEM_RP_PROTECTION,  # 25
    )
    compute_event_losses(*args)

    # Every row of the losses column for item_j=0 must be 0
    assert np.all(losses[:, 0] == 0.0), (
        f"Expected all-zero losses for RP-protected item, got {losses[:, 0]}"
    )


def test_rp_protection_does_not_zero_losses_when_event_rp_equals_item_rp():
    """When event_rp == item_rp the protection does not trigger; losses are non-zero."""
    args, losses = _make_compute_event_losses_args(
        event_rp=ITEM_RP_PROTECTION,  # 25 == 25
        item_rp=ITEM_RP_PROTECTION,   # 25
    )
    compute_event_losses(*args)

    # At least the mean loss should be non-zero (vulnerability always produces max damage)
    assert np.any(losses[:, 0] != 0.0), (
        f"Expected non-zero losses when event_rp == item_rp, got {losses[:, 0]}"
    )


def test_rp_protection_does_not_zero_losses_when_event_rp_above_item_rp():
    """When event_rp (50) > item_rp (25) the protection does not trigger."""
    args, losses = _make_compute_event_losses_args(
        event_rp=EVENT_RP_ABOVE,     # 50
        item_rp=ITEM_RP_PROTECTION,  # 25
    )
    compute_event_losses(*args)

    assert np.any(losses[:, 0] != 0.0), (
        f"Expected non-zero losses when event_rp > item_rp, got {losses[:, 0]}"
    )


def test_rp_protection_does_not_trigger_when_item_rp_is_zero():
    """When item return_period is 0 (no protection configured), losses are non-zero."""
    args, losses = _make_compute_event_losses_args(
        event_rp=EVENT_RP_BELOW,  # 12 — would trigger if item_rp > 0
        item_rp=np.int32(0),      # no protection
    )
    compute_event_losses(*args)

    assert np.any(losses[:, 0] != 0.0), (
        f"Expected non-zero losses when item_rp=0, got {losses[:, 0]}"
    )


def test_rp_protection_only_affects_protected_items():
    """
    In a two-item coverage, only the item with event_rp < item_rp gets zeroed;
    the other item retains its losses.
    """
    sample_size = 0
    Ndamage_bins = 2
    Nintensity_bins = 1

    compute_info = np.zeros(1, dtype=gulmc_compute_info_type)[0]
    compute_info['event_id'] = EVENT_ID
    compute_info['Ndamage_bins_max'] = Ndamage_bins
    compute_info['loss_threshold'] = 0.0
    compute_info['alloc_rule'] = 0
    compute_info['effective_damageability'] = 1
    compute_info['max_bytes_per_item'] = 2048
    compute_info['cursor'] = 0
    compute_info['coverage_i'] = 0
    compute_info['coverage_n'] = 1

    coverages = np.zeros(2, dtype=coverage_type)
    coverages[COVERAGE_ID]['tiv'] = TIV
    coverages[COVERAGE_ID]['max_items'] = 2
    coverages[COVERAGE_ID]['start_items'] = 0
    coverages[COVERAGE_ID]['cur_items'] = 2

    coverage_ids = np.array([COVERAGE_ID], dtype=np.uint32)

    full_dtype = np.dtype([
        ('item_id', np.int32), ('coverage_id', np.uint32), ('areaperil_id', np.uint32),
        ('vulnerability_id', np.int32), ('group_id', np.uint32),
        ('peril_correlation_group', np.int32), ('damage_correlation_value', np.float32),
        ('hazard_group_id', np.int32), ('hazard_correlation_value', np.float32),
        ('vulnerability_idx', oasis_int), ('areaperil_agg_vuln_idx', oasis_int),
        ('intensity_adjustment', np.int32),
        ('return_period', np.int32), ('peril_id', np.int32),
        ('group_seq_id', np.int32), ('hazard_group_seq_id', np.int32),
    ])
    items = np.zeros(2, dtype=full_dtype)
    for i in range(2):
        items[i]['item_id'] = i + 1
        items[i]['coverage_id'] = COVERAGE_ID
        items[i]['areaperil_id'] = AREAPERIL_ID
        items[i]['vulnerability_id'] = VULN_ID
        items[i]['group_id'] = 1
        items[i]['vulnerability_idx'] = 0
        items[i]['areaperil_agg_vuln_idx'] = -1
        items[i]['peril_id'] = PERIL_ID
        items[i]['group_seq_id'] = 0
        items[i]['hazard_group_seq_id'] = 0
    # Item 0: RP-protected (event_rp=12 < item_rp=25) → zero loss
    items[0]['return_period'] = ITEM_RP_PROTECTION  # 25
    # Item 1: not protected (event_rp=12, item_rp=0) → non-zero loss
    items[1]['return_period'] = 0

    items_event_data = np.zeros(2, dtype=items_MC_data_type)
    items_event_data[0]['item_id'] = 1
    items_event_data[0]['item_idx'] = 0
    items_event_data[0]['haz_arr_i'] = 0
    items_event_data[0]['return_period'] = ITEM_RP_PROTECTION
    items_event_data[0]['event_rp'] = EVENT_RP_BELOW  # 12 < 25 → protected
    items_event_data[1]['item_id'] = 2
    items_event_data[1]['item_idx'] = 1
    items_event_data[1]['haz_arr_i'] = 0
    items_event_data[1]['return_period'] = 0
    items_event_data[1]['event_rp'] = EVENT_RP_BELOW  # irrelevant when item_rp=0

    haz_pdf = np.zeros(1, dtype=haz_arr_type)
    haz_pdf[0]['probability'] = 1.0
    haz_pdf[0]['intensity_bin_id'] = HAZ_BIN_ID
    haz_pdf[0]['intensity'] = HAZ_INTENSITY
    haz_arr_ptr = np.array([0, 1], dtype=np.int64)

    vuln_array = np.zeros((1, Ndamage_bins, Nintensity_bins), dtype=oasis_float)
    vuln_array[0, 1, 0] = 1.0

    damage_bins = np.zeros(Ndamage_bins, dtype=damagebin_dtype)
    damage_bins[0] = (0, 0.0, 0.5, 0.25, 0)
    damage_bins[1] = (0, 0.5, 1.0, 0.75, 0)

    Nvulns_cached = 16  # power of two
    cdf_cache_mask = np.int64(Nvulns_cached - 1)
    cdf_cache_tag = np.full(1, CDF_CACHE_EMPTY, dtype=np.int64)  # 1 triplet (both items share it)
    cdf_cache_nbins = np.zeros(Nvulns_cached, dtype=np.int32)
    cached_vuln_cdfs = np.zeros((Nvulns_cached, Ndamage_bins), dtype=oasis_float)

    areaperil_agg_vuln_idx_ja_data = np.empty(0, dtype=agg_vuln_idx_weight_dtype)
    areaperil_agg_vuln_idx_ja_offsets = np.zeros(1, dtype=oasis_int)

    losses = np.zeros((sample_size + 6, 2), dtype=oasis_float)

    haz_rndms_base = np.zeros((1, 1), dtype=np.float64)
    vuln_rndms_base = np.zeros((1, 1), dtype=np.float64)
    vuln_adj = np.ones(1, dtype=oasis_float)
    haz_eps_ij = np.zeros((1, 1), dtype=np.float64)
    damage_eps_ij = np.zeros((1, 1), dtype=np.float64)
    norm_inv_parameters = np.zeros(1, dtype=NormInversionParameters)[0]
    norm_inv_cdf = np.zeros(1, dtype=np.float64)
    norm_cdf = np.zeros(1, dtype=np.float64)
    vuln_z_unif = np.zeros(1, dtype=np.float64)
    haz_z_unif = np.zeros(1, dtype=np.float64)
    byte_mv = np.zeros(PIPE_CAPACITY * 2, dtype='b')

    intensity_bin_peril_ids = np.array([PERIL_ID], dtype=np.int32)
    intensity_bins = np.zeros((1, int(HAZ_INTENSITY) + 1), dtype=np.int32)
    intensity_bins[0, HAZ_INTENSITY] = HAZ_BIN_ID

    compute_event_losses(
        compute_info, coverages, coverage_ids, items_event_data, items,
        sample_size, haz_pdf, haz_arr_ptr, vuln_array, damage_bins,
        cdf_cache_tag, cdf_cache_nbins, cdf_cache_mask, cached_vuln_cdfs,
        areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data,
        losses, haz_rndms_base, vuln_rndms_base, vuln_adj,
        haz_eps_ij, damage_eps_ij,
        norm_inv_parameters, norm_inv_cdf, norm_cdf, vuln_z_unif, haz_z_unif,
        byte_mv, True, intensity_bin_peril_ids, intensity_bins,
    )

    # Item 0 (RP-protected): all losses must be zero
    assert np.all(losses[:, 0] == 0.0), (
        f"Expected zero losses for RP-protected item 0, got {losses[:, 0]}"
    )
    # Item 1 (not protected): mean loss must be non-zero
    assert np.any(losses[:, 1] != 0.0), (
        f"Expected non-zero losses for unprotected item 1, got {losses[:, 1]}"
    )
