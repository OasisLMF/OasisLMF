"""End to end tests of the fm back-allocation rules on a small multi layer structure."""
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
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


def write_static(path, fm_profile=FM_PROFILE):
    for name, content in [('fm_programme', FM_PROGRAMME), ('fm_policytc', FM_POLICYTC),
                          ('fm_profile', fm_profile), ('fm_xref', FM_XREF)]:
        (path / f'{name}.csv').write_text(content)


def run_fm(tmp_path, allocation_rule, guls=GULS, fm_profile=FM_PROFILE, max_sample_index=MAX_SAMPLE_INDEX):
    """Return {(output_id, sidx): loss} for the sampled sidx only."""
    write_static(tmp_path, fm_profile)

    gul_path = tmp_path / 'guls.bin'
    (tmp_path / 'guls.csv').write_text(guls)
    csvtobin(tmp_path / 'guls.csv', gul_path, 'gul',
             stream_type=LOSS_STREAM_ID, max_sample_index=max_sample_index)

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


@settings(max_examples=15, deadline=None)
@given(allocation_rule=st.sampled_from([1, 2, 3]),
       guls=st.lists(st.tuples(st.floats(0, 1e5, width=32), st.floats(0, 1e5, width=32)), min_size=1, max_size=5),
       limits=st.tuples(st.floats(1, 1e5, width=32), st.floats(1, 1e5, width=32)))
def test_back_allocated_losses_sum_to_the_layer_loss_for_any_losses_and_limits(allocation_rule, guls, limits):
    """For any ground up losses and layer limits, each layer's item losses add up to that layer's loss."""
    gul_csv = 'event_id,item_id,sidx,loss\n' + ''.join(
        f'1,{item_id},{sidx},{item_losses[item_id - 1]!r}\n'
        for item_id in (1, 2) for sidx, item_losses in enumerate(guls, start=1))
    fm_profile = FM_PROFILE.replace(',3000,', f',{limits[0]!r},').replace(',1000,', f',{limits[1]!r},')

    with TemporaryDirectory() as tmp_dir:
        losses = run_fm(Path(tmp_dir), allocation_rule, gul_csv, fm_profile, max_sample_index=len(guls))

    for sidx, (loss_1, loss_2) in enumerate(guls, start=1):
        total_gul = float(loss_1) + float(loss_2)
        for layer_outputs, limit in [((1, 3), limits[0]), ((2, 4), limits[1])]:
            allocated = sum(losses.get((output_id, sidx), 0.) for output_id in layer_outputs)
            assert allocated == pytest.approx(min(total_gul, float(limit)), rel=1e-5, abs=1e-2)


def test_allocation_rule_0_builds_with_item_level_outputs(tmp_path):
    """fm_xref maps item ids to outputs whatever the allocation rule, but rule 0 outputs at the top level,
    which has fewer nodes. The structure build must not index past the top level. It runs in a subprocess
    with the JIT disabled so the index is always checked: under the JIT an out of bounds write corrupts
    memory silently or aborts the process.
    See https://github.com/OasisLMF/OasisLMF/issues/2256.
    """
    write_static(tmp_path)

    build = subprocess.run(
        [sys.executable, '-c', 'import sys; from oasislmf.pytools.fm.financial_structure import create_financial_structure; '
                               'create_financial_structure(0, sys.argv[1])', str(tmp_path)],
        capture_output=True, text=True, env={**os.environ, 'NUMBA_DISABLE_JIT': '1'})

    assert build.returncode == 0, build.stderr[-2000:]
