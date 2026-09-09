"""End to end tests of the fm back-allocation rules on a small multi layer structure."""
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from oasislmf.pytools.common.event_stream import LOSS_STREAM_ID
from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
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

GULS = """event_id,item_id,sidx,loss
1,1,1,1000.0
1,2,1,3000.0
"""

MAX_SAMPLE_INDEX = 1


def write_static(path):
    for name, content in [('fm_programme', FM_PROGRAMME), ('fm_policytc', FM_POLICYTC),
                          ('fm_profile', FM_PROFILE), ('fm_xref', FM_XREF)]:
        (path / f'{name}.csv').write_text(content)


def run_fm(tmp_path, allocation_rule):
    """Return {(output_id, sidx): loss} for the sampled sidx only."""
    write_static(tmp_path)

    gul_path = tmp_path / 'guls.bin'
    (tmp_path / 'guls.csv').write_text(GULS)
    csvtobin(tmp_path / 'guls.csv', gul_path, 'gul',
             stream_type=LOSS_STREAM_ID, max_sample_index=MAX_SAMPLE_INDEX)

    out_path = tmp_path / 'fm.bin'
    manager.run(create_financial_structure_files=True, allocation_rule=allocation_rule,
                static_path=str(tmp_path))
    manager.run(create_financial_structure_files=False, allocation_rule=allocation_rule,
                static_path=str(tmp_path), files_in=[str(gul_path)], files_out=[str(out_path)],
                net_loss=None, storage_method='sparse', low_memory=False, sort_output=True,
                stepped=None)

    bintocsv(out_path, tmp_path / 'fm.csv', 'fm')
    losses = pd.read_csv(tmp_path / 'fm.csv')
    losses = losses[losses['sidx'] > 0]
    return {(row.output_id, row.sidx): row.loss for row in losses.itertuples()}


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
