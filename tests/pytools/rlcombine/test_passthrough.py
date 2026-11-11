"""Tests for the rlcombine pass-through outputs (-I): each requested input is written out unchanged,
less its E:S0 sentinel records."""
import shutil
from pathlib import Path

from oasislmf.pytools.rlcombine.manager import run
from oasislmf.pytools.summary.manager import run as summarypy_run
from oasislmf.utils.defaults import SUMMARY_OUTPUT

from .common import IP1, IP1_IP2_COMBINED, IP2, IP3, make_summary_stream, write_rl_static, write_stream_files

SUMMARYPY_ASSETS = Path(__file__).parent.parent.parent / 'assets' / 'test_summarypy'
COMBINED_FILE = 'rl_S1_summary_P1'


def passthrough_file(file_idx):
    """Name of the pass-through output of input file_idx."""
    return f'rl_IP{file_idx + 1}_S1_summary_P1'


def run_passthrough(tmp_path, streams, passthrough):
    """Run rlcombine on the summary streams, passing through the inputs whose indices are in passthrough.

    Returns:
        dict: ``{file name: bytes}`` of every file rlcombine wrote
    """
    files = write_stream_files(tmp_path, streams)
    static_paths = [write_rl_static(tmp_path / f'RI_{i + 1}', 5) for i in range(len(streams))]
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    run(files, str(out_dir / COMBINED_FILE), static_paths, False,
        [f'{i}={out_dir / passthrough_file(i)}' for i in passthrough], False)
    return {path.name: path.read_bytes() for path in out_dir.iterdir()}


def test_passthrough_unchanged_without_sentinels(tmp_path):
    """With -I 0=... 2=..., inputs 0 and 2 are written out byte for byte, header included, without
    their E:S0 sentinels, and nothing is written for input 1."""
    ips = [IP1, IP2, IP3]
    written = run_passthrough(tmp_path, [make_summary_stream(ip) for ip in ips], passthrough=[0, 2])

    assert set(written) == {COMBINED_FILE, passthrough_file(0), passthrough_file(2)}
    for file_idx in (0, 2):
        assert written[passthrough_file(file_idx)] == make_summary_stream(ips[file_idx], sentinel=False)


def test_combined_output_unchanged_by_passthrough(tmp_path):
    """Passing every input through doesn't change the combined output."""
    written = run_passthrough(tmp_path, [make_summary_stream(ip) for ip in (IP1, IP2)], passthrough=[0, 1])
    assert written[COMBINED_FILE] == make_summary_stream(IP1_IP2_COMBINED, sentinel=False)


def test_passthrough_matches_plain_summarypy(tmp_path):
    """summarypy -s only adds the E:S0 sentinels: passing its output through rlcombine gives the same
    bytes as summarypy without -s."""
    static_dir = tmp_path / 'RI_1'
    static_dir.mkdir()
    shutil.copy(SUMMARYPY_ASSETS / 'single_summary_set' / 'RI_1' / f"{SUMMARY_OUTPUT['ri']}.bin",
                static_dir / f"{SUMMARY_OUTPUT['rl']}.bin")

    summary_files = {}
    for sentinel_event in (False, True):
        summary_files[sentinel_event] = tmp_path / f'rl_S1_summary_sentinel_{sentinel_event}'
        summarypy_run(str(SUMMARYPY_ASSETS / 'ri.bin'), str(static_dir), 'rl', False, False,
                      sentinel_event=sentinel_event, summary_sets_output=['-1', str(summary_files[sentinel_event])])
    plain = summary_files[False].read_bytes()
    assert summary_files[True].read_bytes() != plain

    passthrough = tmp_path / passthrough_file(0)
    run([str(summary_files[True])], str(tmp_path / COMBINED_FILE), [str(static_dir)], False,
        [f'0={passthrough}'], False)
    assert passthrough.read_bytes() == plain
