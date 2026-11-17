import json
import os
import shutil
import sys

import numpy as np
import pytest

from oasislmf.pytools.common.data import oasis_float
from oasislmf.pytools.common.event_stream import ITEM_STREAM, LOSS_STREAM_ID, stream_info_to_bytes
from tests.orchestrator.tools import compare_streams, profile_kernel, reference_streams
from tests.orchestrator.tools.compare_streams import (compare_dirs_bytes, compare_dirs_events, compare_files, decode_event_rows,
                                                      list_stream_files, scan_file)

PAIR_DTYPE = np.dtype([('sidx', '<i4'), ('loss', oasis_float)])
SAMPLES = 4


def stream_header(samples=SAMPLES, source=LOSS_STREAM_ID, agg=ITEM_STREAM):
    return stream_info_to_bytes(source, agg) + np.int32(samples).tobytes()


def item_record(event_id, item_id, losses, specials=((-3, 1000.0), (-1, 10.0))):
    pairs = list(specials) + [(sidx, loss) for sidx, loss in enumerate(losses, start=1)] + [(0, 0.0)]
    return np.array([event_id, item_id], '<i4').tobytes() + np.array(pairs, PAIR_DTYPE).tobytes()


def event_block(event_id, n_items=3, samples=SAMPLES):
    rng = np.random.default_rng(event_id)
    return b''.join(item_record(event_id, item_id, rng.random(samples).astype(oasis_float) * 100)
                    for item_id in range(1, n_items + 1))


def write_run(directory, partitions, tap='gul', header=None, blocks=None):
    """Write <tap>_P<k>.bin for each list of event ids in partitions. blocks overrides the bytes of some events."""
    directory.mkdir(parents=True, exist_ok=True)
    blocks = blocks or {}
    for k, events in enumerate(partitions, start=1):
        data = (header or stream_header()) + b''.join(blocks.get(e) or event_block(e, n_items=1 + e % 4) for e in events)
        (directory / f'{tap}_P{k}.bin').write_bytes(data)
    return directory


EVENTS = list(range(1, 13))


def split(events, n):
    return [events[k::n] for k in range(n)]


def test_compare_files_identical_and_first_difference(tmp_path):
    a, b = tmp_path / 'a.bin', tmp_path / 'b.bin'
    a.write_bytes(b'0123456789')
    b.write_bytes(b'0123456789')
    assert compare_files(a, b).ok
    b.write_bytes(b'0123x56789')
    result = compare_files(a, b, chunk_size=3)
    assert result.status == 'different' and result.first_difference == 4
    b.write_bytes(b'01234567')
    result = compare_files(a, b, chunk_size=3)
    assert result.status == 'different' and result.first_difference == 8 and (result.size_a, result.size_b) == (10, 8)


def test_compare_dirs_bytes(tmp_path):
    run_a = write_run(tmp_path / 'a', split(EVENTS, 3))
    run_b = write_run(tmp_path / 'b', split(EVENTS, 3))
    results = compare_dirs_bytes(run_a, run_b)
    assert [r.name for r in results] == ['gul_P1.bin', 'gul_P2.bin', 'gul_P3.bin'] and all(r.ok for r in results)

    (run_b / 'gul_P3.bin').unlink()
    (run_b / 'il_P1.bin').write_bytes(stream_header())
    status = {r.name: r.status for r in compare_dirs_bytes(run_a, run_b)}
    assert status == {'gul_P1.bin': 'identical', 'gul_P2.bin': 'identical', 'gul_P3.bin': 'missing in b', 'il_P1.bin': 'missing in a'}
    assert [r.name for r in compare_dirs_bytes(run_a, run_b, taps=['gul'])] == ['gul_P1.bin', 'gul_P2.bin', 'gul_P3.bin']


def test_list_stream_files_ignores_other_names(tmp_path):
    write_run(tmp_path, split(EVENTS, 2))
    (tmp_path / 'gul_P1.bin.tmp').write_bytes(b'')
    (tmp_path / 'notes.txt').write_bytes(b'')
    files = list_stream_files(tmp_path)
    assert list(files) == ['gul'] and sorted(files['gul']) == [1, 2]


def test_events_identical(tmp_path):
    run_a = write_run(tmp_path / 'a', split(EVENTS, 3))
    run_b = write_run(tmp_path / 'b', split(EVENTS, 3))
    result = compare_dirs_events(run_a, run_b)['gul']
    assert result.ok, result.summary()
    assert sorted(result.a.events) == EVENTS


def test_events_reordered_across_partitions(tmp_path):
    run_a = write_run(tmp_path / 'a', split(EVENTS, 2))
    shuffled = list(np.random.default_rng(0).permutation(EVENTS))
    run_b = write_run(tmp_path / 'b', split(shuffled, 5))
    assert not all(r.ok for r in compare_dirs_bytes(run_a, run_b))
    result = compare_dirs_events(run_a, run_b)['gul']
    assert result.ok, result.summary()
    assert len(result.b.files) == 5


def test_events_one_loss_changed(tmp_path):
    run_a = write_run(tmp_path / 'a', split(EVENTS, 3))
    block = bytearray(event_block(7, n_items=1 + 7 % 4))
    loss_offset = 8 + PAIR_DTYPE.itemsize * 2 + 4
    value = np.frombuffer(bytes(block[loss_offset:loss_offset + oasis_float.itemsize]), oasis_float)[0]
    block[loss_offset:loss_offset + oasis_float.itemsize] = np.array([value + 1], oasis_float).tobytes()
    run_b = write_run(tmp_path / 'b', split(EVENTS, 4), blocks={7: bytes(block)})
    result = compare_dirs_events(run_a, run_b)['gul']
    assert not result.ok
    assert [d.event_id for d in result.different] == [7]
    assert not result.only_a and not result.only_b and result.header_equal
    description = result.different[0].describe()
    assert 'losses differing on common pairs: 1' in description
    assert 'item 1 sidx 1' in description


def test_events_missing_and_reordered_rows(tmp_path):
    run_a = write_run(tmp_path / 'a', split(EVENTS, 3))
    records = [item_record(5, item_id, [1.0, 2.0, 3.0, 4.0]) for item_id in (1, 2)]
    run_b_events = split([e for e in EVENTS if e != 9], 3)
    run_b = write_run(tmp_path / 'b', run_b_events, blocks={5: records[1] + records[0]})
    write_run(tmp_path / 'a2', split(EVENTS, 3), blocks={5: records[0] + records[1]})
    result = compare_dirs_events(tmp_path / 'a2', run_b)['gul']
    assert result.only_a == [9] and not result.only_b
    assert [d.event_id for d in result.different] == [5]
    assert 'same (item, sidx, loss) rows, in a different order' in result.different[0].describe()
    assert compare_dirs_events(run_a, tmp_path / 'a2')['gul'].different[0].event_id == 5


def test_events_truncated(tmp_path):
    run_a = write_run(tmp_path / 'a', split(EVENTS, 3))
    run_b = write_run(tmp_path / 'b', split(EVENTS, 3))
    last = run_b / 'gul_P3.bin'
    data = last.read_bytes()
    last.write_bytes(data[:-PAIR_DTYPE.itemsize])
    result = compare_dirs_events(run_a, run_b)['gul']
    assert not result.ok
    assert any('truncated' in e and 'gul_P3.bin' in e for e in result.b.errors)
    truncated_event = split(EVENTS, 3)[2][-1]
    assert truncated_event in result.only_a

    last.write_bytes(data + b'\x01\x02')
    result = compare_dirs_events(run_a, run_b)['gul']
    assert not result.ok and any('trailing bytes' in e for e in result.b.errors)

    last.write_bytes(data[:5])
    scan = scan_file(last)
    assert 'shorter than' in scan.error


def test_events_bad_record(tmp_path):
    bad = item_record(3, 0, [1.0])
    run = write_run(tmp_path / 'a', [[1, 2, 3]], blocks={3: bad})
    scan = scan_file(run / 'gul_P1.bin')
    assert [e for e, _ in scan.events] == [1]
    assert 'bad item record' in scan.error


def test_events_header_mismatch(tmp_path):
    run_a = write_run(tmp_path / 'a', split(EVENTS, 2))
    run_b = write_run(tmp_path / 'b', split(EVENTS, 2), header=stream_header(samples=SAMPLES + 1))
    result = compare_dirs_events(run_a, run_b)['gul']
    assert not result.header_equal and not result.ok
    assert not result.different
    assert 'HEADERS DIFFER' in result.summary()

    (run_b / 'gul_P1.bin').write_bytes(stream_header() + (run_b / 'gul_P1.bin').read_bytes()[8:])
    result = compare_dirs_events(run_a, run_b)['gul']
    assert not result.ok and any('disagree on the stream header' in e for e in result.b.errors)


def test_events_header_only_files(tmp_path):
    run_a = write_run(tmp_path / 'a', [EVENTS])
    run_b = write_run(tmp_path / 'b', [EVENTS, [], []])
    result = compare_dirs_events(run_a, run_b)['gul']
    assert result.ok, result.summary()


def test_events_duplicate(tmp_path):
    run_a = write_run(tmp_path / 'a', split(EVENTS, 2))
    run_b = write_run(tmp_path / 'b', [EVENTS[:7], EVENTS[6:]])
    result = compare_dirs_events(run_a, run_b)['gul']
    assert not result.ok
    assert len(result.b.duplicates) == 1 and result.b.duplicates[0].startswith('event 7 ')


def test_scan_resumes_when_run_buffer_is_full(tmp_path):
    run = write_run(tmp_path / 'a', [EVENTS])
    full = scan_file(run / 'gul_P1.bin')
    for capacity in (1, 2, 5):
        small = scan_file(run / 'gul_P1.bin', run_capacity=capacity)
        assert small.error is None
        assert [(e, i.digest, i.offset, i.nbytes) for e, i in small.events] == [(e, i.digest, i.offset, i.nbytes) for e, i in full.events]
    assert [e for e, _ in full.events] == EVENTS


def test_decode_event_rows():
    data = item_record(4, 2, [5.0, 6.0]) + item_record(4, 9, [7.0])
    rows = decode_event_rows(data)
    assert rows['item_id'].tolist() == [2] * 5 + [9] * 4
    assert rows['sidx'].tolist() == [-3, -1, 1, 2, 0, -3, -1, 1, 0]
    assert rows['loss'].tolist() == [1000.0, 10.0, 5.0, 6.0, 0.0, 1000.0, 10.0, 7.0, 0.0]
    assert set(rows['event_id'].tolist()) == {4}
    with pytest.raises(ValueError):
        decode_event_rows(data[:-PAIR_DTYPE.itemsize])


def test_compare_streams_cli(tmp_path, capsys):
    run_a = write_run(tmp_path / 'a', split(EVENTS, 2))
    run_b = write_run(tmp_path / 'b', split(EVENTS, 3))
    assert compare_streams.main(['bytes', str(run_a), str(run_a)]) == 0
    assert compare_streams.main(['bytes', str(run_a), str(run_b)]) == 1
    out_json = tmp_path / 'events.json'
    assert compare_streams.main(['events', str(run_a), str(run_b), '--json', str(out_json)]) == 0
    assert json.loads(out_json.read_text())['results']['gul']['ok']
    assert compare_streams.main(['events', str(run_a), str(tmp_path / 'a'), '--taps', 'il']) == 1
    assert 'no <tap>_P<k>.bin files found' in capsys.readouterr().out


@pytest.mark.parametrize('cmdline, expected', [
    (['/usr/bin/gulmc', '-S10'], 'gulmc'),
    (['/usr/bin/python3', '/home/u/.local/bin/fmpy', '-a2'], 'fmpy'),
    (['python3.12', '-u', '-m', 'oasislmf.pytools.eve.cli', '1', '2'], 'oasislmf.pytools.eve.cli'),
    (['python3', '-c', "import sys; from oasislmf.pytools.gulmc.cli import main; sys.argv[0] = 'gulmc'; sys.exit(main())"], 'gulmc'),
    (['python3', '-c', 'from oasislmf.pytools.fm.cli import main; main()'], 'fm'),
    (['python', '-c', 'print(1)'], 'python -c'),
    ([], None),
])
def test_profile_tool_name(cmdline, expected):
    assert profile_kernel.tool_name(cmdline) == expected


def test_profile_command(tmp_path):
    code = "import sys; sys.argv[0] = 'busy'; sum(range(3_000_000))"
    report = profile_kernel.profile_command([sys.executable, '-c', code], interval=0.01)
    assert report['returncode'] == 0
    assert report['wall'] > 0 and report['n_processes'] >= 1
    assert 'busy' in report['tools']
    assert report['peak_rss_sum'] > 0
    json.dumps(report)
    assert 'busy' in profile_kernel.format_report(report)
    assert profile_kernel.main(['--json', str(tmp_path / 'p.json'), '--', sys.executable, '-c', 'raise SystemExit(3)']) == 3
    assert json.loads((tmp_path / 'p.json').read_text())['returncode'] == 3


def fake_tools(fail=None):
    """Stand-in tools: evepy writes 'E<k>/<K>', gulmc and fmpy append their name and arguments to their input."""
    def tool(name):
        code = ("import sys; data = sys.stdin.buffer.read(); "
                f"sys.stdout.buffer.write(data + b'|{name}:' + ' '.join(sys.argv[1:]).encode()); "
                f"sys.exit(3 if {name!r} == {fail!r} else 0)")
        return [sys.executable, '-c', code]
    evepy = [sys.executable, '-c', "import sys; sys.stdout.buffer.write(('E' + sys.argv[1] + '/' + sys.argv[2]).encode())"]
    return reference_streams.Tools(evepy, tool('gulmc'), tool('fmpy'))


def test_pipeline_commands():
    tools = reference_streams.Tools(['eve'], ['gul'], ['fm'])
    commands = reference_streams.pipeline_commands(2, 5, tools, samples=7, gul_args=('-a1',), ri_layers=2)
    assert commands == [
        (None, ['eve', '2', '5']),
        ('gul', ['gul', '-S7', '-a1']),
        ('il', ['fm', '-a2']),
        (None, ['fm', '-a3', '-p', os.path.join('input', 'RI_1'), '-n', '-']),
        ('ri', ['fm', '-a3', '-p', os.path.join('input', 'RI_2'), '-n', '-']),
    ]
    assert [tap for tap, _ in reference_streams.pipeline_commands(1, 1, tools, ri_layers=0)] == [None, 'gul', 'il']
    assert [tap for tap, _ in reference_streams.pipeline_commands(1, 1, tools, il_alloc_rule=None)] == [None, 'gul']


def test_make_reference_streams_with_fake_tools(tmp_path):
    results = reference_streams.make_reference_streams(tmp_path, 3, tmp_path / 'out', tools=fake_tools(), gul_args=('-a1',),
                                                       prepare=False)
    assert sorted(results) == ['gul', 'il', 'ri'] and all(len(paths) == 3 for paths in results.values())
    assert (tmp_path / 'out' / 'gul_P2.bin').read_bytes() == b'E2/3|gulmc:-S10 -a1'
    assert (tmp_path / 'out' / 'il_P2.bin').read_bytes() == b'E2/3|gulmc:-S10 -a1|fmpy:-a2'
    ri_1 = os.path.join('input', 'RI_1')
    assert (tmp_path / 'out' / 'ri_P3.bin').read_bytes() == f'E3/3|gulmc:-S10 -a1|fmpy:-a2|fmpy:-a3 -p {ri_1} -n -'.encode()


def test_make_reference_streams_reports_failures(tmp_path):
    with pytest.raises(reference_streams.PipelineError, match='exited 3'):
        reference_streams.make_reference_streams(tmp_path, 2, tmp_path / 'out', tools=fake_tools(fail='fmpy'), prepare=False)


def test_kernel_run_fixture(kernel_run):
    input_dir = kernel_run.run_dir / 'input'
    for name in ('items.bin', 'coverages.bin', 'events.bin', 'fm_programme.bin', 'compute_info_2.npy'):
        assert (input_dir / name).exists(), name
    for layer in range(1, kernel_run.ri_layers + 1):
        assert (input_dir / f'RI_{layer}' / 'compute_info_2.npy').exists()
    for name in ('footprint.bin', 'footprint.idx', 'vulnerability.bin', 'damage_bin_dict.bin'):
        assert (kernel_run.run_dir / 'static' / name).exists(), name
    assert kernel_run.event_ids.tolist() == list(range(1, 31))


def test_kernel_reference_fixture(kernel_run, kernel_reference, tmp_path):
    ref_1 = kernel_reference(1)
    ref_3 = kernel_reference(3)
    assert kernel_reference(3) == ref_3
    files = list_stream_files(ref_3)
    assert {tap: sorted(chunks) for tap, chunks in files.items()} == {'gul': [1, 2, 3], 'il': [1, 2, 3], 'ri': [1, 2, 3]}

    seen = {}
    for tap, chunks in files.items():
        for k, path in chunks.items():
            scan = scan_file(path)
            assert scan.error is None, scan.error
            assert scan.header == stream_header(samples=kernel_run.samples)
            for event_id, _ in scan.events:
                assert seen.setdefault((tap, event_id), k) == k
    gul_events = {e for (tap, e) in seen if tap == 'gul'}
    assert gul_events and gul_events <= set(kernel_run.event_ids.tolist())

    copy = tmp_path / 'copy'
    shutil.copytree(ref_3, copy)
    assert all(r.ok for r in compare_dirs_bytes(ref_3, copy))
    result = compare_dirs_events(ref_1, ref_3, taps=['gul'])['gul']
    assert result.ok, result.summary(details=3)


@pytest.mark.xfail(strict=True, reason=(
    "fmpy's chance-of-loss (sidx -4) output for an event depends on the events processed before it in the same process: "
    "FMReader writes pass_through[compute position] and never resets it between events, while the writer reads "
    "pass_through[node_id]. Only -4 records differ; every other (item, sidx, loss) is identical."))
@pytest.mark.parametrize('tap', ['il', 'ri'])
def test_kernel_fm_streams_do_not_depend_on_partitioning(kernel_reference, tap):
    result = compare_dirs_events(kernel_reference(1), kernel_reference(3), taps=[tap])[tap]
    assert result.ok, result.summary(details=3)
