"""Tests for the rlcombine pass-through outputs (-I): each requested input is written out unchanged,
less its E:S0 sentinel records."""
from oasislmf.pytools.rlcombine.manager import run
from oasislmf.pytools.summary.manager import run as summarypy_run

from .common import (COMBINED_FILE, IP1, IP1_IP2_COMBINED, IP2, IP3, SUMMARYPY_ASSETS, make_summary_stream,
                     passthrough_file, run_rlcombine, write_summarypy_rl_static)


def test_passthrough_unchanged_without_sentinels(tmp_path):
    """With -I 0=... 2=..., inputs 0 and 2 are written out byte for byte, header included, without
    their E:S0 sentinels. Nothing is written for input 1, and no .idx is written without -m."""
    ips = [IP1, IP2, IP3]
    written = run_rlcombine(tmp_path, [make_summary_stream(ip) for ip in ips], passthrough=[0, 2])

    assert set(written) == {COMBINED_FILE, passthrough_file(0), passthrough_file(2)}
    for file_idx in (0, 2):
        assert written[passthrough_file(file_idx)] == make_summary_stream(ips[file_idx], sentinel=False)


def test_combined_output_unchanged_by_passthrough(tmp_path):
    """Passing every input through doesn't change the combined output."""
    written = run_rlcombine(tmp_path, [make_summary_stream(ip) for ip in (IP1, IP2)], passthrough=[0, 1])
    assert written[COMBINED_FILE] == make_summary_stream(IP1_IP2_COMBINED, sentinel=False)


def test_passthrough_matches_plain_summarypy(tmp_path):
    """summarypy -s only adds the E:S0 sentinels: passing its output through rlcombine gives the same
    bytes as summarypy without -s."""
    static_dir = write_summarypy_rl_static(tmp_path / 'RI_1')

    summary_files = {}
    for sentinel_event in (False, True):
        summary_files[sentinel_event] = tmp_path / f'rl_S1_summary_sentinel_{sentinel_event}'
        summarypy_run(str(SUMMARYPY_ASSETS / 'ri.bin'), static_dir, 'rl', False, False,
                      sentinel_event=sentinel_event, summary_sets_output=['-1', str(summary_files[sentinel_event])])
    plain = summary_files[False].read_bytes()
    assert summary_files[True].read_bytes() != plain

    passthrough = tmp_path / passthrough_file(0)
    run([str(summary_files[True])], str(tmp_path / COMBINED_FILE), [static_dir], False,
        [f'0={passthrough}'], False)
    assert passthrough.read_bytes() == plain
