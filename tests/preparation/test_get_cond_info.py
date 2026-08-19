"""Unit tests for il_inputs.get_cond_info (vectorized condition-hierarchy resolution).

These lock in the observable contract of get_cond_info: the mapping of cond-tags
to hierarchy levels (``level_conds``) and the generated filler account rows
(``extra_accounts``), including the branches the FM acceptance fixtures don't
exercise directly (the same-priority conflict raise and priority-0 handling).
"""
import numpy as np
import pandas as pd
import pytest
from ods_tools.oed import fill_empty

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
    # tags sit in different layers so each one is missing the other's layer and a filler is emitted;
    # layer 1 holds a CondClass==1 exclusion, layer 2 does not
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 1],   # exclusion in layer 1
        [1, 0, 2, 'P2', 2, 'B', 'B', 1, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [11, 1, 'B']])
    _, extra_accounts = get_cond_info(loc, acc)
    by_tag_layer = {(e['CondTag'], e['layer_id']): e for e in extra_accounts}

    assert set(by_tag_layer) == {('A', 2), ('B', 1)}

    b_in_excluded_layer = by_tag_layer[('B', 1)]
    assert b_in_excluded_layer['CondNumber'] == 'FullFilter'
    assert b_in_excluded_layer['CondDed6All'] == 1
    assert b_in_excluded_layer['CondDedType6All'] == 1

    a_in_plain_layer = by_tag_layer[('A', 2)]
    assert a_in_plain_layer['CondNumber'] == ''
    assert 'CondDed6All' not in a_in_plain_layer


# ---------------------------------------------------------------------------
# Differential test against the pre-vectorisation implementation
# ---------------------------------------------------------------------------

def reference_get_cond_info(locations_df, accounts_df):
    """The loop implementation this replaced, copied verbatim from main as the reference."""
    pol_info = {}
    level_conds = {}
    extra_accounts = []
    default_cond_tag = '0'
    if 'CondTag' in locations_df.columns:
        fill_empty(locations_df, 'CondTag', default_cond_tag)
        loc_condkey_df = locations_df.loc[locations_df['CondTag'] != default_cond_tag, ['acc_id', 'CondTag']].drop_duplicates()
    else:
        loc_condkey_df = pd.DataFrame([], columns=['acc_id', 'CondTag'])

    if 'CondTag' in accounts_df.columns:
        fill_empty(accounts_df, 'CondTag', default_cond_tag)
        acc_condkey_df = accounts_df.loc[accounts_df['CondTag'] != '', ['acc_id', 'CondTag']].drop_duplicates()
        condkey_match_df = acc_condkey_df.merge(loc_condkey_df, how='outer', indicator=True)
        missing_condkey_df = condkey_match_df.loc[condkey_match_df['_merge'] == 'right_only', ['acc_id', 'CondTag']]
    else:
        acc_condkey_df = pd.DataFrame([], columns=['acc_id', 'CondTag'])
        missing_condkey_df = loc_condkey_df

    if missing_condkey_df.shape[0]:
        raise OasisException(f'Those condtag are present in locations but missing in the account file:\n{missing_condkey_df}')

    if acc_condkey_df.shape[0]:
        if 'CondTag' not in locations_df.columns:
            locations_df['CondTag'] = default_cond_tag
        # we get information about cond from accounts_df
        cond_tags = {}  # information about each cond tag
        account_layer_exclusion = {}  # for each account and layer, store info about cond class exclusion
        if 'CondPriority' in accounts_df.columns:
            fill_empty(accounts_df, 'CondPriority', 1)
        else:
            accounts_df['CondPriority'] = 1
        if 'CondPeril' in accounts_df.columns:
            fill_empty(accounts_df, 'CondPeril', '')
        else:
            accounts_df['CondPeril'] = ''
        for acc_rec in accounts_df.to_dict(orient="records"):
            cond_tag_key = (acc_rec['acc_id'], acc_rec['CondTag'])
            cond_number_key = (acc_rec['acc_id'], acc_rec['CondTag'], acc_rec['CondNumber'])
            cond_tag = cond_tags.setdefault(cond_tag_key, {'CondPriority': acc_rec['CondPriority'] or 1, 'CondPeril': acc_rec['CondPeril']})
            cond_tag.setdefault('layers', {})[acc_rec['layer_id']] = {'CondNumber': cond_number_key}
            exclusion_cond_tags = account_layer_exclusion.setdefault(acc_rec['acc_id'], {}).setdefault(acc_rec['layer_id'],
                                                                                                       set())
            pol_info[(acc_rec['acc_id'], acc_rec['layer_id'])] = [acc_rec['PolNumber'], acc_rec['LayerNumber'], acc_rec['acc_idx']]
            if acc_rec.get('CondClass') == 1:
                exclusion_cond_tags.add(acc_rec['CondTag'])

        # we get the list of loc for each cond_tag
        loc_conds = {}
        KEY_INDEX = 0
        PRIORITY_INDEX = 1
        for loc_rec in locations_df.to_dict(orient="records"):
            loc_key = loc_rec['loc_id']
            cond_key = (loc_rec['acc_id'], loc_rec.get('CondTag', default_cond_tag))
            if cond_key in cond_tags:
                cond_tag = cond_tags[cond_key]
            else:
                cond_tag = {'CondPriority': 1, 'layers': {}}
                cond_tags[cond_key] = cond_tag

            cond_location = cond_tag.setdefault('locations', set())
            cond_location.add(loc_key)
            cond_tag_priority = cond_tag['CondPriority']
            conds = loc_conds.setdefault(loc_key, [])

            for i, cond in enumerate(conds):
                if cond_tag_priority < cond[PRIORITY_INDEX]:
                    conds.insert(i, (cond_key, cond_tag_priority))
                    break
                elif cond_tag_priority == cond[PRIORITY_INDEX] and cond_key != cond[KEY_INDEX]:
                    raise OasisException(f"{cond_key} and {cond[KEY_INDEX]} have same priority in {loc_key}")
            else:
                conds.append((cond_key, cond_tag_priority))

        # at first we just want condtag for each level
        for cond_key, cond_info in cond_tags.items():
            acc_id, cond_tag = cond_key
            cond_level_start = 1
            for loc_key in cond_info.get('locations', set()):
                for i, (cond_key_i, _) in enumerate(loc_conds[loc_key]):
                    if cond_key_i == cond_key:
                        cond_level_start = max(cond_level_start, i + 1)
                        break
            cond_info['cond_level_start'] = cond_level_start
            cond_peril = cond_info.get('CondPeril') or 'AA1'
            for layer_id, exclusion_conds in account_layer_exclusion[acc_id].items():
                if layer_id not in cond_info['layers']:
                    PolNumber, LayerNumber, acc_idx = pol_info[(acc_id, layer_id)]
                    if exclusion_conds:
                        extra_accounts.append({
                            'acc_idx': acc_idx,
                            'acc_id': acc_id,
                            'PolNumber': PolNumber,
                            'LayerNumber': LayerNumber,
                            'CondTag': cond_tag,
                            'layer_id': layer_id,
                            'CondNumber': 'FullFilter',
                            'CondDed6All': 1,
                            'CondDedType6All': 1,
                            'CondPeril': cond_peril,
                        })
                    else:
                        extra_accounts.append({
                            'acc_idx': acc_idx,
                            'acc_id': acc_id,
                            'PolNumber': PolNumber,
                            'LayerNumber': LayerNumber,
                            'CondTag': cond_tag,
                            'layer_id': layer_id,
                            'CondNumber': '',
                            'CondPeril': cond_peril,
                        })
            level_conds.setdefault(cond_level_start, set()).add(cond_key)
    return level_conds, extra_accounts


def _normalise(fn, loc, acc):
    """Run fn and reduce its result to a comparable form, treating a raise as an outcome."""
    try:
        level_conds, extra_accounts = fn(loc.copy(), acc.copy())
    except OasisException:
        return 'raised'
    return (
        _levels(level_conds),
        # order differs between the two implementations and does not matter to the caller
        sorted(tuple(sorted((k, str(v)) for k, v in extra.items())) for extra in extra_accounts),
    )


def _random_case(rng):
    """An account/location pair shaped like the real ones: one row per (loc_id, acc_id, CondTag).

    The trailing ``.drop_duplicates()`` on the locations frame is a deliberate exclusion, not an
    oversight: exactly-duplicated (loc_id, acc_id, CondTag) rows are the one input class where the
    two implementations are *meant* to differ, so feeding them to the reference comparison below
    would assert the old answer. They are covered instead by the two duplicate-row tests, which
    pin the new answer directly.
    """
    tags = ['A', 'B', 'C', '0']
    acc_rows = []
    for _ in range(int(rng.integers(1, 7))):
        acc_id = int(rng.integers(1, 3))
        acc_rows.append([
            acc_id, acc_id - 1, int(rng.integers(1, 3)), f'P{acc_id}', int(rng.integers(1, 3)),
            tags[int(rng.integers(0, len(tags)))], 'C1',
            int(rng.integers(0, 4)),                          # CondPriority, 0 included
            ['AA1', '', 'WW1'][int(rng.integers(0, 3))],      # CondPeril, blank included
            int(rng.integers(0, 2)),                          # CondClass, exclusion included
        ])
    acc = _acc(acc_rows)

    acc_tags = acc[['acc_id', 'CondTag']].drop_duplicates().to_numpy().tolist()
    loc_rows = []
    for _ in range(int(rng.integers(1, 6))):
        acc_id, cond_tag = acc_tags[int(rng.integers(0, len(acc_tags)))]
        loc_rows.append([int(rng.integers(10, 13)), int(acc_id), cond_tag])
    return _loc(loc_rows).drop_duplicates(), acc


def test_matches_reference_implementation():
    """The vectorized result equals the loop one over randomly generated account/location pairs."""
    rng = np.random.default_rng(20260812)
    compared = 0

    for _ in range(120):
        loc, acc = _random_case(rng)
        expected = _normalise(reference_get_cond_info, loc, acc)
        assert _normalise(get_cond_info, loc, acc) == expected, f"\nacc:\n{acc}\nloc:\n{loc}"
        compared += expected != 'raised'

    # the generator must be producing real comparisons, not just conflicting priorities
    assert compared > 60


def test_duplicate_location_rows_do_not_inflate_levels():
    """Repeated (loc_id, acc_id, CondTag) rows count once, so they cannot invent nesting levels.

    The loop implementation appended one entry per location row, so a tag repeated behind a
    higher priority one was pushed down a level for every duplicate. get_levels de-duplicates
    these same three columns before use, so counting them once is what the caller assumes.
    """
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'B', 'B', 2, 'AA1', 0],
    ])
    duplicated = _loc([[10, 1, 'A'], [10, 1, 'A'], [10, 1, 'A'], [10, 1, 'B']])
    deduplicated = _loc([[10, 1, 'A'], [10, 1, 'B']])

    level_conds, _ = get_cond_info(duplicated, acc)
    assert _levels(level_conds) == {1: [(1, 'A')], 2: [(1, 'B')]}
    assert _levels(level_conds) == _levels(get_cond_info(deduplicated, acc)[0])


def test_duplicate_location_rows_can_merge_two_tags_into_one_level():
    """Two tags of equal priority on different locations share an FM level once duplicates count once.

    This is the one deliberate behaviour change against the loop implementation, and it moves loss
    numbers rather than only renumbering levels. B and C both have priority 2 but sit on different
    locations, so no same-priority conflict is raised. The loop pushed B down to level 3 because
    location 10's duplicated 'A' row occupied a slot; de-duplicating first leaves B and C tied at
    the same dense rank, so they are applied at one aggregation node instead of nested.

    Old (loop): {1: [(1, 'A')], 2: [(1, 'C')], 3: [(1, 'B')]} -> three cond FM levels.
    New:        {1: [(1, 'A')], 2: [(1, 'B'), (1, 'C')]}      -> two, with B and C sharing one.
    """
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'B', 'B', 2, 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'C', 'C', 2, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [10, 1, 'A'], [10, 1, 'B'], [11, 1, 'A'], [11, 1, 'C']])

    level_conds, _ = get_cond_info(loc, acc)
    assert _levels(level_conds) == {1: [(1, 'A')], 2: [(1, 'B'), (1, 'C')]}

    # the loop's answer, kept here so the divergence stays visible if either side is touched
    assert _levels(reference_get_cond_info(loc.copy(), acc.copy())[0]) == {
        1: [(1, 'A')], 2: [(1, 'C')], 3: [(1, 'B')]}


def test_duplicate_location_rows_match_the_deduplicated_frame():
    """Duplicated location rows are equivalent to the de-duplicated frame, over random cases.

    This is the invariant the new semantics claims, and it is the input class
    test_matches_reference_implementation deliberately excludes. Asserting it against
    get_cond_info's own de-duplicated answer (rather than against the loop) is the point:
    the loop does not hold this invariant.
    """
    rng = np.random.default_rng(20260819)
    compared = 0

    for _ in range(120):
        loc, acc = _random_case(rng)
        # repeat a random subset of the location rows, so the frame carries real duplicates
        repeated = pd.concat([loc, loc.sample(frac=0.5, random_state=int(rng.integers(0, 2**31)))])
        expected = _normalise(get_cond_info, loc, acc)
        assert _normalise(get_cond_info, repeated, acc) == expected, f"\nacc:\n{acc}\nloc:\n{repeated}"
        compared += expected != 'raised'

    assert compared > 60
