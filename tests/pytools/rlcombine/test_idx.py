"""Tests for the rlcombine .idx files (-m): one (summary_id, offset) entry per record of the combined
output and of each pass-through, offset being where the record starts in the stream."""
import os

import numpy as np
import pytest

from oasislmf.pytools.common.data import summary_stream_index_dtype
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.rlcombine.manager import run
from oasislmf.pytools.summary.manager import run as summarypy_run

from .common import (COMBINED_FILE, IP1, IP2, SUMMARYPY_ASSETS, event_bytes, make_summary_stream, passthrough_file,
                     run_rlcombine, write_summarypy_rl_static)


def idx_entries(entries):
    """``[(summary_id, offset), ...]`` as .idx records."""
    return np.array(entries, dtype=summary_stream_index_dtype)


def read_idx(data):
    """The records of a .idx file."""
    return np.frombuffer(data, dtype=summary_stream_index_dtype)


def test_idx_contents(tmp_path):
    """With -m, a .idx is written next to the combined output and the pass-through, with an entry
    for every record. The stream header is 12 bytes, and a record is 12 bytes of header plus
    8 bytes per (sidx, loss) pair, terminator included."""
    written = run_rlcombine(tmp_path, [make_summary_stream(ip) for ip in (IP1, IP2)], passthrough=[0],
                            low_memory=True, ext='.bin')

    assert set(written) == {f'{name}.{ext}' for name in (COMBINED_FILE, passthrough_file(0)) for ext in ('bin', 'idx')}
    # IP1_IP2_COMBINED: event 10 summary_ids 1 and 3 (3 pairs each), event 7 summary_id 1 (2 pairs),
    # event 12 summary_id 2
    np.testing.assert_array_equal(read_idx(written[f'{COMBINED_FILE}.idx']),
                                  idx_entries([(1, 12), (3, 56), (1, 100), (2, 136)]))
    # IP1: event 10 summary_ids 1 (3 pairs) and 3 (2 pairs), nothing for event 7, event 12 summary_id 2
    np.testing.assert_array_equal(read_idx(written[f'{passthrough_file(0)}.idx']),
                                  idx_entries([(1, 12), (3, 56), (2, 92)]))


def test_idx_offsets_across_flushes(tmp_path):
    """Offsets stay right when an event is written across several flushes. 4,000 records of 36 bytes
    (max loss and mean only) fill PIPE_CAPACITY up to 16 bytes short, so the flushes fall between records."""
    n_records = 4000
    records = [(summary_id, 1.0, [(-5, 1.0), (-1, 1.0)]) for summary_id in range(1, n_records + 1)]
    assert len(event_bytes(1, records, sentinel=False)) > 2 * PIPE_CAPACITY

    written = run_rlcombine(tmp_path, [make_summary_stream([(1, records)], len_sample=1)] * 2, passthrough=[0],
                            low_memory=True, ext='.bin', max_summary_id=n_records)

    expected = idx_entries([(summary_id, 12 + 36 * (summary_id - 1)) for summary_id in range(1, n_records + 1)])
    np.testing.assert_array_equal(read_idx(written[f'{COMBINED_FILE}.idx']), expected)
    np.testing.assert_array_equal(read_idx(written[f'{passthrough_file(0)}.idx']), expected)


@pytest.mark.parametrize('file_name', [pytest.param('rl_S1_summary_P1.bin', id='with-extension'),
                                       pytest.param('rl_S1_summary_P1', id='without-extension')])
def test_idx_path_matches_summarypy(tmp_path, file_name):
    """rlcombine writes the .idx of its combined output and of a pass-through to the same place as
    summarypy -m does for an output of the same name."""
    static_dir = write_summarypy_rl_static(tmp_path / 'RI_1')
    out_dirs = {tool: tmp_path / tool for tool in ('summarypy', 'rlcombine', 'passthrough')}
    for out_dir in out_dirs.values():
        out_dir.mkdir()

    summarypy_run(str(SUMMARYPY_ASSETS / 'ri.bin'), static_dir, 'rl', True, False,
                  summary_sets_output=['-1', str(out_dirs['summarypy'] / file_name)])
    sentinel_stream = tmp_path / 'rl_S1_summary_sentinel'
    summarypy_run(str(SUMMARYPY_ASSETS / 'ri.bin'), static_dir, 'rl', False, False, sentinel_event=True,
                  summary_sets_output=['-1', str(sentinel_stream)])
    run([str(sentinel_stream)], str(out_dirs['rlcombine'] / file_name), [static_dir], True,
        [f"0={out_dirs['passthrough'] / file_name}"], False)

    expected = sorted(os.listdir(out_dirs['summarypy']))
    assert len(expected) == 2
    assert sorted(os.listdir(out_dirs['rlcombine'])) == expected
    assert sorted(os.listdir(out_dirs['passthrough'])) == expected
