"""
Tests for an areaperil whose event footprint holds more hazard records than the model has
intensity bins.

A dynamic footprint keeps one record per (intensity, return period), so an areaperil lying in
several sections of one event can carry many more records than there are intensity bins.
compute_event_losses must size its per-hazard-bin scratch buffers, and treat the CDF cache, on
the records actually present rather than on the number of intensity bins.
"""
import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import MEAN_IDX
from oasislmf.pytools.gulmc.manager import CDF_CACHE_EMPTY, compute_event_losses

from .test_rp_protection import _make_compute_event_losses_args

# the helper's model has 1 intensity bin, 2 damage bins and a 16-slot CDF cache. The helper shares
# the probability evenly between the records, so powers of two keep each share exact in float32 and
# the comparison with one record exact with the JIT disabled too.
N_HAZ_RECORDS_ABOVE_BINS = 8     # above the per-hazard-bin buffers (2), within the cache
N_HAZ_RECORDS_ABOVE_CACHE = 64   # a CDF block (1 + records) larger than the 16-slot cache


@pytest.mark.parametrize("n_haz_records", [N_HAZ_RECORDS_ABOVE_BINS, N_HAZ_RECORDS_ABOVE_CACHE],
                         ids=["above-intensity-bins", "above-cdf-cache"])
@pytest.mark.parametrize("sample_size", [0, 8], ids=["effective-damageability", "full-monte-carlo"])
def test_more_hazard_records_than_intensity_bins(n_haz_records, sample_size):
    """Splitting one hazard record into many records of the same intensity, with the probability
    shared between them, describes the same hazard, so the losses must not change.
    """
    args, losses_one, *_ = _make_compute_event_losses_args(
        event_rp=0, item_rp=0, sample_size=sample_size, n_haz_records=1)
    assert compute_event_losses(*args) is True

    args, losses_many, *_ = _make_compute_event_losses_args(
        event_rp=0, item_rp=0, sample_size=sample_size, n_haz_records=n_haz_records)
    assert compute_event_losses(*args) is True

    assert losses_one[MEAN_IDX, 0] > 0
    np.testing.assert_allclose(losses_many, losses_one, rtol=1e-6)


@pytest.mark.parametrize("n_haz_records, cached", [(N_HAZ_RECORDS_ABOVE_BINS, True), (N_HAZ_RECORDS_ABOVE_CACHE, False)],
                         ids=["block-fits-cache", "block-wider-than-cache"])
def test_cdf_block_wider_than_cache_is_not_cached(n_haz_records, cached):
    """In full Monte Carlo an item caches 1 + Nhaz_bins CDFs as one block. A block wider than the
    cache would wrap onto itself and a later cache hit would read overwritten CDFs, so such an item
    is computed but left out of the cache.
    """
    args, *_ = _make_compute_event_losses_args(event_rp=0, item_rp=0, sample_size=8, n_haz_records=n_haz_records)
    cdf_cache_tag = args[12]
    assert compute_event_losses(*args) is True
    assert (cdf_cache_tag[0] != CDF_CACHE_EMPTY) == cached
