"""End to end tests of the fm back-allocation rules on a small multi layer structure."""
import struct

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oasislmf.pytools.common.event_stream import ITEM_STREAM, LOSS_STREAM_ID, stream_info_to_bytes
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

GULS = {1: 1000.0, 2: 3000.0}


def write_static(path):
    for name, content in [('fm_programme', FM_PROGRAMME), ('fm_policytc', FM_POLICYTC),
                          ('fm_profile', FM_PROFILE), ('fm_xref', FM_XREF)]:
        (path / f'{name}.csv').write_text(content)


def write_gul_stream(path, guls, max_sidx=1):
    """Write a loss stream holding one event with a single sample per item."""
    with open(path, 'wb') as stream:
        stream.write(stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM))
        stream.write(np.int32(max_sidx).tobytes())
        for item_id, loss in guls.items():
            stream.write(struct.pack('=ii', 1, item_id))
            for sidx in range(1, max_sidx + 1):
                stream.write(struct.pack('=if', sidx, loss))
            stream.write(struct.pack('=if', 0, 0.))  # delimiter
    return path


def read_loss_stream(path):
    """Return {(output_id, sidx): loss} for the sampled sidx only."""
    data = np.fromfile(path, dtype='b')
    losses = {}
    cursor = 8  # stream_type + max_sidx
    while cursor < data.nbytes:
        _, output_id = struct.unpack_from('=ii', data, cursor)
        cursor += 8
        while cursor < data.nbytes:
            sidx, loss = struct.unpack_from('=if', data, cursor)
            cursor += 8
            if sidx == 0:
                break
            if sidx > 0:
                losses[(output_id, sidx)] = loss
    return losses


def run_fm(tmp_path, allocation_rule):
    write_static(tmp_path)
    gul_path = write_gul_stream(tmp_path / 'guls.bin', GULS)
    out_path = tmp_path / 'fm.bin'
    manager.run(create_financial_structure_files=True, allocation_rule=allocation_rule,
                static_path=str(tmp_path))
    manager.run(create_financial_structure_files=False, allocation_rule=allocation_rule,
                static_path=str(tmp_path), files_in=[str(gul_path)], files_out=[str(out_path)],
                net_loss=None, storage_method='sparse', low_memory=False, sort_output=True,
                stepped=None)
    return read_loss_stream(out_path)


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
