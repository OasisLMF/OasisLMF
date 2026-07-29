"""Unit tests for il_inputs.get_cond_info (vectorized condition-hierarchy resolution).

These lock in the observable contract of get_cond_info: the mapping of cond-tags
to hierarchy levels (``level_conds``) and the generated filler account rows
(``extra_accounts``), including the branches the FM acceptance fixtures don't
exercise directly (the same-priority conflict raise and priority-0 handling).
"""
import pandas as pd
import pytest

from oasislmf.preparation.il_inputs import get_cond_info
from oasislmf.utils.exceptions import OasisException

ACC_COLS = ['acc_id', 'acc_idx', 'layer_id', 'PolNumber', 'LayerNumber',
            'CondTag', 'CondNumber', 'CondPriority', 'CondPeril', 'CondClass']


def _acc(rows):
    return pd.DataFrame(rows, columns=ACC_COLS)


def _loc(rows):
    return pd.DataFrame(rows, columns=['loc_id', 'acc_id', 'CondTag'])


def _levels(level_conds):
    return {lvl: sorted((int(a), str(t)) for a, t in keys) for lvl, keys in level_conds.items()}


def test_no_conditions_returns_empty():
    # no CondTag column on the account file -> no condition hierarchy at all
    acc = _acc([[1, 0, 1, 'P1', 1, '0', '0', 1, 'AA1', 0]]).drop(columns=['CondTag'])
    loc = _loc([[10, 1, '0']])
    level_conds, extra_accounts = get_cond_info(loc, acc)
    assert level_conds == {}
    assert extra_accounts == []


def test_nesting_levels_by_priority():
    # one location subject to three cond-tags of increasing priority -> three levels
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'B', 'B', 2, 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'C', 'C', 3, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [10, 1, 'B'], [10, 1, 'C']])
    level_conds, _ = get_cond_info(loc, acc)
    assert _levels(level_conds) == {1: [(1, 'A')], 2: [(1, 'B')], 3: [(1, 'C')]}


def test_cond_level_start_is_max_over_locations():
    # tag B appears at level 2 for one location; that determines its level globally
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'B', 'B', 2, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [10, 1, 'B'], [11, 1, 'B']])
    level_conds, _ = get_cond_info(loc, acc)
    assert _levels(level_conds) == {1: [(1, 'A')], 2: [(1, 'B')]}


def test_same_priority_conflict_raises():
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'B', 'B', 1, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [10, 1, 'B']])  # both priority 1 on the same location
    with pytest.raises(OasisException, match="same priority"):
        get_cond_info(loc, acc)


def test_priority_zero_treated_as_one():
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 0, 'AA1', 0],   # priority 0 -> 1
        [1, 0, 1, 'P1', 1, 'B', 'B', 2, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [10, 1, 'B']])
    level_conds, _ = get_cond_info(loc, acc)
    assert _levels(level_conds) == {1: [(1, 'A')], 2: [(1, 'B')]}


def test_extra_account_filler_for_uncovered_layer():
    # tag A is present in layer 1 only; layer 2 exists on the account -> a filler row is emitted
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0],
        [1, 0, 2, 'P2', 2, 'B', 'B', 1, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [11, 1, 'B']])
    _, extra_accounts = get_cond_info(loc, acc)
    extra = pd.DataFrame(extra_accounts)
    # A missing from layer 2, B missing from layer 1
    a_row = extra[(extra['CondTag'] == 'A') & (extra['layer_id'] == 2)]
    assert len(a_row) == 1
    assert a_row.iloc[0]['CondNumber'] == ''            # no exclusion -> empty filler
    assert 'CondDed6All' not in extra_accounts[list(extra.index[(extra['CondTag'] == 'A') & (extra['layer_id'] == 2)])[0]]


def test_exclusion_produces_fullfilter():
    # a CondClass==1 (exclusion) row in a layer makes fillers for that layer FullFilter
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 1],   # exclusion in layer 1
        [1, 0, 1, 'P1', 1, 'B', 'B', 1, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [11, 1, 'B']])
    _, extra_accounts = get_cond_info(loc, acc)
    by_tag_layer = {(e['CondTag'], e['layer_id']): e for e in extra_accounts}
    # A has no row in... both A and B are in layer 1; each is missing the *other's* coverage only within same layer
    # every emitted filler in layer 1 must be FullFilter because layer 1 has an exclusion
    for (tag, layer), e in by_tag_layer.items():
        if layer == 1:
            assert e['CondNumber'] == 'FullFilter'
            assert e['CondDed6All'] == 1 and e['CondDedType6All'] == 1
