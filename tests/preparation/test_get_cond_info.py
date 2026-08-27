"""Unit tests for il_inputs.get_cond_info (vectorized condition-hierarchy resolution).

These lock in the observable contract of get_cond_info: the mapping of cond-tags
to hierarchy levels (``level_conds``) and the generated filler account rows
(``extra_accounts``), including the branches the FM acceptance fixtures don't
exercise directly (the same-priority conflict raise and priority-0 handling).
"""
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype
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
    # the full message is pinned, not just "same priority": support scripts grep this text, and the
    # operand order is the row loop's -- the arriving tag first, then the one already held
    with pytest.raises(OasisException, match=r"^\(1, 'B'\) and \(1, 'A'\) have same priority in 10$"):
        get_cond_info(loc, acc)


def test_same_priority_conflict_names_the_first_conflict_in_row_order():
    """Several conflicting locations report the one that collides first in the file, not the lowest id.

    The row loop raised at the first offending location *row*, so a conflict on location 12 that
    appears before a conflict on location 11 is the one reported. Selecting by a sorted groupby key
    would name 11 instead.
    """
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'B', 'B', 1, 'AA1', 0],
    ])
    loc = _loc([[12, 1, 'A'], [11, 1, 'A'], [12, 1, 'B'], [11, 1, 'B']])

    with pytest.raises(OasisException, match=r"have same priority in 12$"):
        get_cond_info(loc, acc)
    assert _normalise(get_cond_info, loc, acc) == _normalise(reference_get_cond_info, loc, acc)


def test_empty_locations_against_a_populated_account_file():
    """No locations means no cond keys from the location side, but the account tags still emit."""
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0],
        [1, 0, 2, 'P2', 2, 'A', 'A', 1, 'AA1', 0],
    ])
    loc = _loc([])

    assert _normalise(get_cond_info, loc, acc) == _normalise(reference_get_cond_info, loc, acc)


def test_extra_account_dtypes_survive_the_frame_the_caller_builds():
    """get_levels does pd.DataFrame(extra_accounts) and concatenates it onto accounts_df.

    The values come from itertuples, so they are numpy scalars where the row loop produced python
    ones. That must not change the dtypes the caller ends up concatenating, in particular it must
    not upcast the integer id columns.
    """
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0],
        [1, 0, 2, 'P2', 2, 'B', 'B', 1, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [11, 1, 'B']])

    _, extra_accounts = get_cond_info(loc.copy(), acc.copy())
    _, reference_extra = reference_get_cond_info(loc.copy(), acc.copy())
    frame, reference_frame = pd.DataFrame(extra_accounts), pd.DataFrame(reference_extra)

    assert frame.columns.to_list() == reference_frame.columns.to_list()
    assert frame.dtypes.to_dict() == reference_frame.dtypes.to_dict()
    for column in ['acc_id', 'acc_idx', 'layer_id']:
        assert frame[column].dtype.kind == 'i', f'{column} must stay an integer, got {frame[column].dtype}'


def test_acc_id_missing_from_the_account_file_raises():
    """A location acc_id with no account rows is rejected, not dropped.

    The existing CondTag check above only covers non-default tags, so a location carrying the
    default '0' tag reaches here. There are no layers to attach its conds to: the row loop died on
    a bare ``account_layer_exclusion[acc_id]`` lookup, and dropping it on the merge instead would
    leave the key in level_conds for get_levels to left-merge into an all-null account row.
    """
    acc = _acc([[1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0]])
    loc = _loc([[10, 1, 'A'], [11, 2, '0']])  # acc_id 2 is not in the account file

    with pytest.raises(OasisException, match='present in locations but missing in the account file'):
        get_cond_info(loc.copy(), acc.copy())
    with pytest.raises(KeyError):  # what the loop implementation did with the same input
        reference_get_cond_info(loc.copy(), acc.copy())


def test_unresolved_acc_id_raises():
    """An unresolved location->account match leaves a null acc_id, which is the same failure."""
    acc = _acc([[1, 0, 1, 'P1', 1, 'A', 'A', 1, 'AA1', 0]])
    loc = _loc([[10, 1, 'A'], [11, np.nan, '0']])

    with pytest.raises(OasisException, match='present in locations but missing in the account file'):
        get_cond_info(loc, acc)


def test_non_numeric_priority_raises():
    """A malformed CondPriority fails loudly rather than defaulting to 1.

    fill_empty has already mapped every blank value to 1 by this point, so a value that survives
    and will not convert is not a missing priority but a bad one. Coercing it to 1 would silently
    reassign the condition's place in the hierarchy.
    """
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', 'N/A', 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'B', 'B', 2, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [10, 1, 'B']])

    with pytest.raises(OasisException, match='CondPriority values in the account file are not numeric'):
        get_cond_info(loc, acc)


def test_blank_priority_still_defaults_to_one():
    """The values fill_empty treats as empty keep defaulting to priority 1, unaffected by the above."""
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', '', 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'B', 'B', np.nan, 'AA1', 0],
    ])
    loc = _loc([[10, 1, 'A'], [11, 1, 'B']])  # different locations, so no same-priority conflict

    level_conds, _ = get_cond_info(loc, acc)
    assert _levels(level_conds) == {1: [(1, 'A'), (1, 'B')]}


def test_blank_priority_defaults_to_one_on_a_text_dtype_column():
    """A text-dtype CondPriority still defaults its blanks, rather than rejecting the int default.

    Both pandas' string dtypes refuse an int written into them, so `fill_empty(.., 1)` raises on a
    CondPriority column held as text. pandas 3 reaches that on ordinary input: a column of priorities
    read as strings, or one left entirely blank, is inferred as `str` rather than object.
    """
    acc = _acc([
        [1, 0, 1, 'P1', 1, 'A', 'A', '', 'AA1', 0],
        [1, 0, 1, 'P1', 1, 'B', 'B', '2', 'AA1', 0],
    ])
    acc['CondPriority'] = acc['CondPriority'].astype('string')
    loc = _loc([[10, 1, 'A'], [10, 1, 'B']])

    level_conds, _ = get_cond_info(loc, acc)
    assert _levels(level_conds) == {1: [(1, 'A')], 2: [(1, 'B')]}


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
    filler = next(e for e in extra_accounts if e['CondTag'] == 'A' and e['layer_id'] == 2)
    assert 'CondDed6All' not in filler


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


def _nesting_holds(level_conds, loc, acc):
    """True when every location's cond tags have strictly increasing levels in priority order.

    This is the property the FM needs -- two conditions on one location must sit on different
    levels, innermost first -- and it is a property of the answer alone, so it can be asserted
    without reference to either implementation.
    """
    level = {key: lvl for lvl, keys in level_conds.items() for key in keys}

    # mirrors get_cond_info's own resolution, and the order matters: it fills blanks *before*
    # taking the first row per tag, so a tag whose first row is blank resolves to 1. Reading
    # .first() off the unfilled frame instead would skip that row and pick up the next one.
    acc = acc.copy()
    if 'CondPriority' in acc.columns:
        if not is_numeric_dtype(acc['CondPriority']):
            acc['CondPriority'] = acc['CondPriority'].astype('object')
        fill_empty(acc, 'CondPriority', 1)
    else:
        acc['CondPriority'] = 1
    first = acc.groupby(['acc_id', 'CondTag'], sort=False, observed=True)['CondPriority'].first()
    numeric = pd.to_numeric(first, errors='coerce').fillna(1)
    priority = {(int(a), str(t)): float(v) for (a, t), v in numeric.mask(numeric == 0, 1).items()}

    for _, rows in loc.groupby('loc_id'):
        keys = [(int(r.acc_id), str(r.CondTag)) for r in rows.itertuples()]
        ordered = sorted(keys, key=lambda k: priority.get(k, 1))
        levels = [level.get(k, 1) for k in ordered]
        if any(inner >= outer for inner, outer in zip(levels, levels[1:])):
            return False
    return True


def _normalise(fn, loc, acc):
    """Run fn and reduce its result to a comparable form, treating a raise as an outcome.

    extra_accounts row order is load-bearing and is not sorted away: it is concatenated onto
    accounts_df by the caller, so it becomes FM row order. The exception message is compared too,
    since two implementations raising for different reasons are not the same outcome.

    level_conds is compared by level, not by insertion order. get_levels iterates it and never
    reads the key, so insertion order is what assigns cond FM levels -- but it is now depth order
    by construction, whereas the loop emitted a level the first time a cond_tag carrying it was
    seen, i.e. account-file order, which nests an outer condition inside an inner one whenever the
    two disagree. That divergence is deliberate and is pinned end to end by the
    validation/insurance_conditions unit sc26, which fails on the losses if it regresses.
    """
    try:
        level_conds, extra_accounts = fn(loc.copy(), acc.copy())
    except OasisException as exc:
        return f'raised: {exc}'
    return (
        sorted(_levels(level_conds).items()),
        [tuple(sorted((k, str(v)) for k, v in extra.items())) for extra in extra_accounts],
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

    # PolNumber/LayerNumber are never fill_empty'd, so nulls reach the filler rows the account
    # aggregation emits; CondClass/CondPriority/CondPeril are all optional on the account file
    for column in ['PolNumber', 'LayerNumber']:
        nulls = rng.random(len(acc)) < 0.2
        if nulls.any():
            acc.loc[nulls, column] = np.nan
    for column in ['CondClass', 'CondPriority', 'CondPeril']:
        if rng.random() < 0.15:
            acc = acc.drop(columns=[column])

    acc_tags = acc[['acc_id', 'CondTag']].drop_duplicates().to_numpy().tolist()
    loc_rows = []
    for _ in range(int(rng.integers(1, 6))):
        acc_id, cond_tag = acc_tags[int(rng.integers(0, len(acc_tags)))]
        loc_rows.append([int(rng.integers(10, 13)), int(acc_id), cond_tag])
    return _loc(loc_rows).drop_duplicates(), acc


def test_matches_reference_implementation():
    """The vectorized result matches the loop one wherever the loop's answer was usable.

    The loop can assign one level to two conditions that share a location and must therefore
    nest, which the FM cannot express. Where it does, the two implementations are meant to
    differ, so the levels are held to the nesting invariant rather than to the loop's answer;
    everything else -- the extra_accounts rows and their order, and raising -- must match
    exactly. Seeds are only a sampling strategy here, so a case that diverges is a case the
    generator found, not a test that has drifted.
    """
    rng = np.random.default_rng(16)
    compared = diverged = 0

    for _ in range(120):
        loc, acc = _random_case(rng)
        expected = _normalise(reference_get_cond_info, loc, acc)
        actual = _normalise(get_cond_info, loc, acc)
        if isinstance(expected, str) or isinstance(actual, str):
            assert actual == expected, f"\nacc:\n{acc}\nloc:\n{loc}"
            continue

        compared += 1
        assert actual[1] == expected[1], f"extra_accounts differ\nacc:\n{acc}\nloc:\n{loc}"
        if actual[0] != expected[0]:
            diverged += 1
            assert _nesting_holds(get_cond_info(loc.copy(), acc.copy())[0], loc, acc), (
                f"new levels break the nesting invariant\nacc:\n{acc}\nloc:\n{loc}")
            assert not _nesting_holds(reference_get_cond_info(loc.copy(), acc.copy())[0], loc, acc), (
                f"levels differ but the loop's answer was already valid\nacc:\n{acc}\nloc:\n{loc}")

    # the generator must be producing real comparisons, not just conflicting priorities
    assert compared > 60
    # and it must still be reaching the divergent branch: without this the seed can drift to one
    # where the two implementations never disagree, leaving the invariant assertions above dead
    # and the test passing with the level fix reverted
    assert diverged > 0


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
        compared += not isinstance(expected, str)

    assert compared > 60
