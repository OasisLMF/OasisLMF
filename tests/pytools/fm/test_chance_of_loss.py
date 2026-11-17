"""The chance of loss (sidx -4) that fmpy passes from each input item to its outputs.

fmpy does not compute the chance of loss; it copies the value each item arrives with onto that
item's outputs under the back-allocation rules. The value belongs to one item in one event, so it
must reach that item's outputs only, and the output of an event must not depend on which events
the same process read before it.
"""
import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from oasislmf.pytools.common.data import (fm_policytc_dtype, fm_profile_dtype,
                                          fm_programme_dtype, fm_xref_dtype, oasis_float, oasis_int)
from oasislmf.pytools.common.event_stream import (CHANCE_OF_LOSS_IDX, ITEM_STREAM, LOSS_STREAM_ID,
                                                  MAX_LOSS_IDX, MEAN_IDX, TIV_IDX, stream_info_to_bytes)
from oasislmf.pytools.fm.manager import run as run_fm

S = 2
N_ITEMS = 3
LAYERS = (1, 2)
OUTPUT_ID = {(item_id, layer_id): 2 * (item_id - 1) + layer_id for item_id in range(1, N_ITEMS + 1) for layer_id in LAYERS}


def _write_structure(d, level_1):
    """Items -> level 1 (``level_1`` maps item_id to agg_id) -> one level-2 node with two layers.

    Level 1 takes a deductible of 10, the two layers at level 2 a deductible of 0 and 50.
    """
    profile = np.zeros(3, dtype=fm_profile_dtype)
    for profile_id, deductible in ((1, 10.), (2, 0.), (3, 50.)):
        profile[profile_id - 1]['profile_id'] = profile_id
        profile[profile_id - 1]['calcrule_id'] = 12
        profile[profile_id - 1]['deductible1'] = deductible
    n_level_1 = max(level_1.values())
    programme = ([(item_id, 1, agg_id) for item_id, agg_id in level_1.items()]
                 + [(agg_id, 2, 1) for agg_id in range(1, n_level_1 + 1)])
    policytc = ([(1, agg_id, 1, 1) for agg_id in range(1, n_level_1 + 1)]
                + [(2, 1, 1, 2), (2, 1, 2, 3)])
    xref = [(output_id, item_id, layer_id) for (item_id, layer_id), output_id in OUTPUT_ID.items()]
    for name, arr in (('fm_programme', np.array(programme, dtype=fm_programme_dtype)),
                      ('fm_policytc', np.array(policytc, dtype=fm_policytc_dtype)),
                      ('fm_profile', profile),
                      ('fm_xref', np.array(xref, dtype=fm_xref_dtype))):
        arr.tofile(os.path.join(d, f'{name}.bin'))


STRUCTURES = {
    # level 1 is one node per item, so fmpy starts at level 1
    'single_peril': {1: 1, 2: 2, 3: 3},
    # items 1 and 2 share a level-1 node, so fmpy starts at the item level
    'multi_peril': {1: 1, 2: 1, 3: 2},
}


def _chance_of_loss(event_id, item_id):
    return 0.125 * item_id + 0.03125 * event_id


# event_id: {item_id: has a chance-of-loss record}. Event 3 reads items 1 and 2 again without
# one, and leaves item 3 out: stale values from events 1 and 2 would show up on it.
EVENTS = {
    1: {1: True, 2: True, 3: True},
    2: {2: True, 3: True},
    3: {1: False, 2: False},
    4: {3: True},
}


def _write_gul_stream(path, event_ids):
    with open(path, 'wb') as f:
        f.write(stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM))
        f.write(np.int32(S).tobytes())
        for event_id in event_ids:
            for item_id, has_chance_of_loss in EVENTS[event_id].items():
                loss = 100. * item_id
                records = [(MAX_LOSS_IDX, 2 * loss)]
                if has_chance_of_loss:
                    records.append((CHANCE_OF_LOSS_IDX, _chance_of_loss(event_id, item_id)))
                records += [(TIV_IDX, 10 * loss), (MEAN_IDX, loss), (1, loss), (2, loss / 2), (0, 0.)]
                f.write(np.array([event_id, item_id], dtype=oasis_int).tobytes())
                for sidx, value in records:
                    f.write(np.array([sidx], dtype=oasis_int).tobytes())
                    f.write(np.array([value], dtype=oasis_float).tobytes())


def _read_fm_stream(path):
    """{(event_id, output_id): {sidx: loss}}"""
    raw = open(path, 'rb').read()[8:]
    out, pos = {}, 0
    while pos < len(raw):
        event_id, output_id = (int(v) for v in np.frombuffer(raw[pos:pos + 8], dtype=oasis_int))
        pos += 8
        records = out.setdefault((event_id, output_id), {})
        while True:
            sidx = int(np.frombuffer(raw[pos:pos + 4], dtype=oasis_int)[0])
            loss = float(np.frombuffer(raw[pos + 4:pos + 8], dtype=oasis_float)[0])
            pos += 8
            if sidx == 0:
                break
            records[sidx] = loss
    return out


def _run(structure, event_ids, allocation_rule, net_loss=None):
    with TemporaryDirectory() as d:
        _write_structure(d, STRUCTURES[structure])
        gul, out = os.path.join(d, 'gul.bin'), os.path.join(d, 'fm.bin')
        _write_gul_stream(gul, event_ids)
        run_fm(True, allocation_rule=allocation_rule, static_path=d)
        run_fm(False, allocation_rule=allocation_rule, static_path=d, files_in=[gul], files_out=[out],
               net_loss=net_loss, storage_method='sparse', low_memory=False, sort_output=False)
        return _read_fm_stream(out)


MODES = [(structure, allocation_rule, net_loss)
         for structure in STRUCTURES
         for allocation_rule in (0, 1, 2, 3)
         for net_loss in ((None,) if allocation_rule == 0 else (None, ''))]


@pytest.mark.parametrize('structure, allocation_rule, net_loss', MODES)
def test_an_event_gives_the_same_output_whatever_was_read_before_it(structure, allocation_rule, net_loss):
    together = _run(structure, list(EVENTS), allocation_rule, net_loss)
    for event_id in EVENTS:
        alone = _run(structure, [event_id], allocation_rule, net_loss)
        assert alone
        assert {k: v for k, v in together.items() if k[0] == event_id} == alone, f'event {event_id}'


@pytest.mark.parametrize('structure, allocation_rule, net_loss', [mode for mode in MODES if mode[1] != 0])
def test_each_output_carries_the_chance_of_loss_of_its_own_item(structure, allocation_rule, net_loss):
    """Under back-allocation every output is one item's share of a layer, so it takes the value
    that item arrived with, and none when the item arrived without one."""
    output = _run(structure, list(EVENTS), allocation_rule, net_loss)
    for event_id, items in EVENTS.items():
        for item_id, has_chance_of_loss in items.items():
            for layer_id in LAYERS:
                key = (event_id, OUTPUT_ID[item_id, layer_id])
                expected = _chance_of_loss(event_id, item_id) if has_chance_of_loss else None
                assert output[key].get(CHANCE_OF_LOSS_IDX) == expected, key
