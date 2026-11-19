"""Tests of the fmpy net loss output modes: net to a separate path, net instead of gross, and net to stdout."""
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from oasislmf.pytools.common.event_stream import LOSS_STREAM_ID
from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.fm import manager

from .test_back_allocation import FM_PROGRAMME

# two items in one aggregation under a single limited layer
FM_POLICYTC = """level_id,agg_id,layer_id,profile_id
1,1,1,2
"""

FM_XREF = """output,agg_id,layer_id
1,1,1
2,2,1
"""


def fm_profile(limit):
    return ("profile_id,calcrule_id,deductible1,deductible2,deductible3,attachment1,limit1,share1,share2,share3\n"
            "1,100,0,0,0,0,0,0,0,0\n"
            f"2,14,0,0,0,0,{limit!r},0,0,0\n")


def write_inputs(path, guls, limit):
    """Write the static files and the ground up loss stream; `guls` is a list of per-sample (item 1, item 2) losses."""
    for name, content in [('fm_programme', FM_PROGRAMME), ('fm_policytc', FM_POLICYTC),
                          ('fm_profile', fm_profile(limit)), ('fm_xref', FM_XREF)]:
        (path / f'{name}.csv').write_text(content)
    (path / 'guls.csv').write_text('event_id,item_id,sidx,loss\n' + ''.join(
        f'1,{item_id},{sidx},{item_losses[item_id - 1]!r}\n'
        for item_id in (1, 2) for sidx, item_losses in enumerate(guls, start=1)))
    csvtobin(path / 'guls.csv', path / 'guls.bin', 'gul', stream_type=LOSS_STREAM_ID, max_sample_index=len(guls))


def read_losses(path, bin_name):
    """Return {(output_id, sidx): loss} for the sampled sidx of an fm stream."""
    bintocsv(path / bin_name, path / f'{bin_name}.csv', 'fm')
    losses = pd.read_csv(path / f'{bin_name}.csv')
    losses = losses[losses['sidx'] > 0]
    return {(row.output_id, row.sidx): row.loss for row in losses.itertuples()}


def run_fm(path, allocation_rule, net_loss):
    manager.run(create_financial_structure_files=True, allocation_rule=allocation_rule, static_path=str(path))
    manager.run(create_financial_structure_files=False, allocation_rule=allocation_rule, static_path=str(path),
                files_in=[str(path / 'guls.bin')], files_out=[str(path / 'fm.bin')], net_loss=net_loss,
                storage_method='sparse', low_memory=False, sort_output=True, stepped=None)


def assert_gross_plus_net_is_the_ground_up_loss(guls, gross, net):
    for sidx, item_losses in enumerate(guls, start=1):
        for output_id, gul in enumerate(item_losses, start=1):
            key = (output_id, sidx)
            assert gross.get(key, 0.) + net.get(key, 0.) == pytest.approx(float(gul), rel=1e-5, abs=1e-2)


guls_strategy = st.lists(st.tuples(st.floats(0, 1e5, width=32), st.floats(0, 1e5, width=32)), min_size=1, max_size=4)
limits = st.floats(1, 1e5, width=32)


@settings(max_examples=15, deadline=None)
@given(allocation_rule=st.sampled_from([1, 2, 3]), guls=guls_strategy, limit=limits)
def test_net_loss_path_holds_what_the_gross_does_not(allocation_rule, guls, limit):
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)
        write_inputs(path, guls, limit)
        run_fm(path, allocation_rule, net_loss=str(path / 'net.bin'))
        gross, net = read_losses(path, 'fm.bin'), read_losses(path, 'net.bin')

    assert_gross_plus_net_is_the_ground_up_loss(guls, gross, net)


@settings(max_examples=15, deadline=None)
@given(allocation_rule=st.sampled_from([1, 2, 3]), guls=guls_strategy, limit=limits)
def test_empty_net_loss_writes_the_net_instead_of_the_gross(allocation_rule, guls, limit):
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)
        write_inputs(path, guls, limit)
        run_fm(path, allocation_rule, net_loss=None)
        gross = read_losses(path, 'fm.bin')
        run_fm(path, allocation_rule, net_loss='')
        net = read_losses(path, 'fm.bin')

    assert_gross_plus_net_is_the_ground_up_loss(guls, gross, net)


def test_dash_net_loss_writes_the_net_to_stdout():
    guls = [(1000., 3000.), (200., 50.)]
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)
        write_inputs(path, guls, 1000.)
        manager.run(create_financial_structure_files=True, allocation_rule=1, static_path=str(path))
        net_stream = subprocess.run(
            ['fmpy', '-a1', '-p', str(path), '-i', str(path / 'guls.bin'), '-o', str(path / 'fm.bin'), '-n', '-'],
            cwd=path, capture_output=True, check=True).stdout
        (path / 'net.bin').write_bytes(net_stream)
        gross, net = read_losses(path, 'fm.bin'), read_losses(path, 'net.bin')

    assert net, "nothing was written to stdout"
    assert_gross_plus_net_is_the_ground_up_loss(guls, gross, net)


def test_net_loss_is_rejected_for_allocation_rule_0():
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)
        write_inputs(path, [(1000., 3000.)], 1000.)
        with pytest.raises(NotImplementedError):
            manager.run(create_financial_structure_files=False, allocation_rule=0, static_path=str(path),
                        files_in=[str(path / 'guls.bin')], files_out=[str(path / 'fm.bin')], net_loss=str(path / 'net.bin'),
                        storage_method='sparse', low_memory=False, sort_output=True, stepped=None)
