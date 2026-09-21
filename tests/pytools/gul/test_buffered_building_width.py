"""The per-building buffer is sized by the coverages that actually buffer.

A coverage only needs every building held at once when the alloc-rule cap reduces across
items at a fixed building. Anything else emits each building as it is computed, so one
outsized single-item location must not set the width for the whole run.
"""
import numpy as np
import pytest

from oasislmf.pytools.gul.manager import buffered_building_width


def make(coverage_ids, packed_buildings, max_items):
    """Items on the given coverages, plus a coverages table indexed by coverage_id.

    Only the two fields the rule reads are modelled; the run's real items array carries them
    alongside the rest of items_dtype and the correlations columns.
    """
    items = np.zeros(len(coverage_ids), dtype=[('coverage_id', 'i4'), ('packed_buildings', 'i4')])
    items['coverage_id'] = coverage_ids
    items['packed_buildings'] = packed_buildings
    coverages = np.zeros(max(coverage_ids) + 1, dtype=[('max_items', 'i4')])
    for cid, n in max_items.items():
        coverages['max_items'][cid] = n
    return items['coverage_id'], items['packed_buildings'], coverages['max_items']


def test_alloc_rule_zero_never_buffers():
    cov_ids, packed, max_by_cov = make([1, 1, 2], [-630510, -630510, -4], {1: 2, 2: 1})
    assert buffered_building_width(0, 100, cov_ids, packed, max_by_cov) == 1


def test_single_item_coverages_never_buffer():
    """The case the change is for: one huge location, one item on it."""
    cov_ids, packed, max_by_cov = make([1, 2], [-630510, -4], {1: 1, 2: 1})
    for alloc_rule in (1, 2, 3):
        assert buffered_building_width(alloc_rule, 100, cov_ids, packed, max_by_cov) == 1


def test_multi_item_coverage_sets_the_width():
    cov_ids, packed, max_by_cov = make([1, 1, 2], [-16, -16, -630510], {1: 2, 2: 1})
    # coverage 2 is huge but single-item, so only coverage 1's 16 buildings need holding
    assert buffered_building_width(2, 100, cov_ids, packed, max_by_cov) == 16


def test_width_uses_the_magnitude_of_the_signed_count():
    """packed_buildings is signed; a summed item must not contribute a negative width."""
    cov_ids, packed, max_by_cov = make([1, 1], [9, 9], {1: 2})
    assert buffered_building_width(1, 100, cov_ids, packed, max_by_cov) == 9


def test_no_samples_needs_no_building_axis():
    """With sample_size 0 write_losses emits only specials and never indexes the building axis."""
    cov_ids, packed, max_by_cov = make([1, 1], [-630510, -630510], {1: 2})
    assert buffered_building_width(2, 0, cov_ids, packed, max_by_cov) == 1


def test_no_items():
    cov_ids, packed, max_by_cov = make([1], [-4], {1: 1})
    assert buffered_building_width(2, 100, cov_ids[:0], packed[:0], max_by_cov) == 1


@pytest.mark.parametrize("alloc_rule", [0, 1, 2, 3])
def test_width_is_at_least_one(alloc_rule):
    cov_ids, packed, max_by_cov = make([1, 1], [-1, -1], {1: 2})
    assert buffered_building_width(alloc_rule, 100, cov_ids, packed, max_by_cov) >= 1
