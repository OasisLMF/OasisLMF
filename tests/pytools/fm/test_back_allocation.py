"""End to end tests of the fm back-allocation rules on a small multi layer structure."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from oasislmf.pytools.common.data import oasis_float, oasis_int
from oasislmf.pytools.common.event_stream import LOSS_STREAM_ID, stream_info_to_bytes
from oasislmf.pytools.fm import manager

# two items in one aggregation, one level, two layers limited to 3000 and 1000
FM_PROGRAMME = """from_agg_id,level_id,to_agg_id
1,1,1
2,1,1
"""

FM_POLICYTC = """level_id,agg_id,layer_id,profile_id
1,1,1,2
1,1,2,3
"""

FM_PROFILE = """profile_id,calcrule_id,deductible1,deductible2,deductible3,attachment1,limit1,share1,share2,share3
1,100,0,0,0,0,0,0,0,0
2,14,0,0,0,0,3000,0,0,0
3,14,0,0,0,0,1000,0,0,0
"""

# output ids in item then layer order
FM_XREF = """output,agg_id,layer_id
1,1,1
2,1,2
3,2,1
4,2,2
"""

# (event_id, item_id, [(sidx, loss), ...])
GULS = [
    (1, 1, [(1, 1000.0)]),
    (1, 2, [(1, 3000.0)]),
]

MAX_SAMPLE_INDEX = 1


def write_static(path):
    for name, content in [('fm_programme', FM_PROGRAMME), ('fm_policytc', FM_POLICYTC),
                          ('fm_profile', FM_PROFILE), ('fm_xref', FM_XREF)]:
        (path / f'{name}.csv').write_text(content)


def write_loss_stream(path, records):
    with open(path, 'wb') as f:
        f.write(stream_info_to_bytes(LOSS_STREAM_ID, 1))
        f.write(np.array([MAX_SAMPLE_INDEX], dtype=oasis_int).tobytes())
        for event_id, item_id, sidx_losses in records:
            f.write(np.array([event_id, item_id], dtype=oasis_int).tobytes())
            for sidx, loss in sidx_losses + [(0, 0.)]:
                f.write(np.array([sidx], dtype=oasis_int).tobytes())
                f.write(np.array([loss], dtype=oasis_float).tobytes())


def read_loss_stream(path):
    """Return [(output_id, sidx, loss), ...] from a loss stream file."""
    data = open(path, 'rb').read()[8:]  # skip stream type and max sample index
    int_size, float_size = oasis_int.itemsize, oasis_float.itemsize
    records, cursor, output_id = [], 0, None
    while cursor < len(data):
        if output_id is None:
            output_id = int(np.frombuffer(data, oasis_int, 1, cursor + int_size)[0])
            cursor += 2 * int_size
            continue
        sidx = int(np.frombuffer(data, oasis_int, 1, cursor)[0])
        loss = float(np.frombuffer(data, oasis_float, 1, cursor + int_size)[0])
        cursor += int_size + float_size
        if sidx == 0:
            output_id = None
        else:
            records.append((output_id, sidx, loss))
    return records


def run_fm(tmp_path, allocation_rule):
    """Return {(output_id, sidx): loss} for the sampled sidx only."""
    write_static(tmp_path)

    gul_path = tmp_path / 'guls.bin'
    write_loss_stream(gul_path, GULS)

    out_path = tmp_path / 'fm.bin'
    manager.run(create_financial_structure_files=True, allocation_rule=allocation_rule,
                static_path=str(tmp_path))
    manager.run(create_financial_structure_files=False, allocation_rule=allocation_rule,
                static_path=str(tmp_path), files_in=[str(gul_path)], files_out=[str(out_path)],
                net_loss=None, storage_method='sparse', low_memory=False, sort_output=True,
                stepped=None)

    return {(output_id, sidx): loss for output_id, sidx, loss in read_loss_stream(out_path) if sidx > 0}


@pytest.mark.parametrize('allocation_rule, expected', [
    # layer 1 loss is min(4000, 3000) = 3000, layer 2 loss is min(4000, 1000) = 1000.
    # rule 1 splits each layer by the ground up losses of 1000 and 3000, so both layers
    # split 1:3. rules 2 and 3 have nothing below them here, so they agree.
    (1, {1: 750., 2: 250., 3: 2250., 4: 750.}),
    (2, {1: 750., 2: 250., 3: 2250., 4: 750.}),
    (3, {1: 750., 2: 250., 3: 2250., 4: 750.}),
])
def test_back_allocation_is_per_layer(tmp_path, allocation_rule, expected):
    losses = run_fm(tmp_path, allocation_rule)

    assert_allclose([losses[(output_id, 1)] for output_id in sorted(expected)],
                    [expected[output_id] for output_id in sorted(expected)], rtol=1e-6)


@pytest.mark.parametrize('allocation_rule', [1, 2, 3])
def test_back_allocated_losses_sum_to_the_layer_loss(tmp_path, allocation_rule):
    """Each layer's item losses must add back up to that layer's loss.

    Allocation rule 1 read the ground up losses with a layer offset that the storage does
    not have, so every layer above the first was allocated with another node's losses and
    did not add back up. See https://github.com/OasisLMF/OasisLMF/issues/2131.
    """
    losses = run_fm(tmp_path, allocation_rule)

    layer_1 = losses[(1, 1)] + losses[(3, 1)]  # output ids 1 and 3 are layer 1
    layer_2 = losses[(2, 1)] + losses[(4, 1)]  # output ids 2 and 4 are layer 2

    assert layer_1 == pytest.approx(3000., rel=1e-6)
    assert layer_2 == pytest.approx(1000., rel=1e-6)
