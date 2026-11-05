"""Summaries of an event must not depend on the events processed before it in the same summarypy run.

summarypy keeps per-event state in arrays sized by the portfolio (losses per summary id, summary ids
present in the event, risks already affected) and resets only the entries an event used. This checks
that a stream of consecutive events touching different items gives, for each event, the same bytes as
running that event alone.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from oasislmf.pytools.summary.cli import parser, manager

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_summarypy")
STREAM_HEADER_SIZE = 8  # loss stream header: stream type (int32) + sample size (int32)

# run type: (input stream, static directory, summary sets, output zeros)
RUN_TYPES = {
    manager.RUNTYPE_GROUNDUP_LOSS: ('gul.bin', 'multiple_summary_set', [1, 2], False),
    manager.RUNTYPE_INSURED_LOSS: ('il.bin', 'multiple_summary_set', [1, 2, 3], False),
    manager.RUNTYPE_REINSURANCE_LOSS: ('ri.bin', 'multiple_summary_set/RI_1', [1, 2], True),
}


def read_records(stream_path):
    """Split a loss stream into its header and its item records (event_id, item_id, bytes of the record)."""
    buf = Path(stream_path).read_bytes()
    cursor = STREAM_HEADER_SIZE
    records = []
    while cursor < len(buf):
        start = cursor
        event_id, item_id = np.frombuffer(buf, '<i4', 2, cursor)
        cursor += 8
        while True:
            sidx = np.frombuffer(buf, '<i4', 1, cursor)[0]
            cursor += 8
            if sidx == 0:
                break
        records.append((int(event_id), int(item_id), buf[start + 4:cursor]))  # record without its event_id
    return buf[:STREAM_HEADER_SIZE], records


def build_events(records):
    """Four events made from the item records of the asset stream, each with a different set of items."""
    by_event = {}
    for event_id, item_id, body in records:
        by_event.setdefault(event_id, []).append((item_id, body))
    first, second = (sorted(items) for items in by_event.values())
    n = len(first)
    events = {
        1: first[: n // 2],
        2: second[n // 2:],
        3: first[n // 4: 3 * n // 4],
        4: second,
    }
    item_sets = [frozenset(i for i, _ in items) for items in events.values()]
    assert len(set(item_sets)) == len(item_sets)
    return events


def write_stream(path, header, events, sequence):
    with open(path, 'wb') as f:
        f.write(header)
        for event_id in sequence:
            for _, body in events[event_id]:
                f.write(np.int32(event_id).tobytes())
                f.write(body)


def run_summarypy(static_path, run_type, stream_path, summary_sets, output_zeros, out_dir):
    cmd = f"-t {run_type}{' -z' if output_zeros else ''} -p {static_path} -i {stream_path}".split()
    kwargs = vars(parser.parse_args(cmd))
    kwargs.pop('logging_level')
    paths = {i: Path(out_dir, f'S{i}.bin') for i in summary_sets}
    kwargs['summary_sets_output'] = [arg for i, path in paths.items() for arg in (f'-{i}', str(path))]
    manager.main(**kwargs)
    return {i: path.read_bytes() for i, path in paths.items()}


@pytest.mark.parametrize('run_type', list(RUN_TYPES))
def test_event_summaries_independent_of_previous_events(run_type, tmp_path):
    stream_name, static_name, summary_sets, output_zeros = RUN_TYPES[run_type]
    static_path = tmp_path / 'static'
    shutil.copytree(TESTS_ASSETS_DIR / static_name, static_path,
                    ignore=shutil.ignore_patterns('*.bin', '*.idx', 'error_files', 'RI_1'))
    header, records = read_records(TESTS_ASSETS_DIR / stream_name)
    events = build_events(records)

    def run(sequence, name):
        out_dir = tmp_path / name
        out_dir.mkdir()
        write_stream(out_dir / 'in.bin', header, events, sequence)
        return run_summarypy(static_path, run_type, out_dir / 'in.bin', summary_sets, output_zeros, out_dir)

    alone = {e: run([e], f'alone_{e}') for e in events}
    summary_header_size = manager.SUMMARY_META_SIZE
    for e in events:
        for i in summary_sets:
            assert len(alone[e][i]) > summary_header_size
    sequences = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4, 1, 2, 3, 4], [1, 3, 1, 2, 4, 2]]
    for k, sequence in enumerate(sequences):
        result = run(sequence, f'sequence_{k}')
        for i in summary_sets:
            expected = alone[sequence[0]][i][:summary_header_size] + b''.join(
                alone[e][i][summary_header_size:] for e in sequence)
            assert result[i] == expected, f'summary set {i}, sequence {sequence}'
