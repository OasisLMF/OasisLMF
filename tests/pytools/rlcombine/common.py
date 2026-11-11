"""Common data and helpers for the rlcombine tests: build summary streams and run the reader on them."""
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from oasislmf.pytools.common.data import fm_summary_xref_dtype
from oasislmf.pytools.common.event_stream import ITEM_STREAM, SUMMARY_STREAM_ID, stream_info_to_bytes
from oasislmf.pytools.rlcombine.manager import init_summary_streams_in, run
from oasislmf.pytools.rlcombine.reader import (SENTINEL_SUMMARY_ID, SummaryCombineReader, event_id_dtype, loss_dtype,
                                               sidx_dtype, summary_id_dtype, summaryset_id_dtype)
from oasislmf.utils.defaults import SUMMARY_OUTPUT

SUMMARYPY_ASSETS = Path(__file__).parent.parent.parent / 'assets' / 'test_summarypy'
COMBINED_FILE = 'rl_S1_summary_P1'  # name of the combined output written by run_rlcombine

# a summary record header and a (sidx, loss) pair, as written in a summary stream
record_header_dtype = np.dtype([('event_id', event_id_dtype), ('summary_id', summary_id_dtype), ('expval', loss_dtype)])
sidx_loss_dtype = np.dtype([('sidx', sidx_dtype), ('loss', loss_dtype)])
TERMINATOR = (0, 0.0)  # (sidx, loss) pair closing every record


def record_bytes(event_id, summary_id, expval, pairs):
    """Bytes of one summary record: its header, its (sidx, loss) pairs and the (0, 0) terminator."""
    header = np.array([(event_id, summary_id, expval)], dtype=record_header_dtype)
    body = np.array(list(pairs) + [TERMINATOR], dtype=sidx_loss_dtype)
    return header.tobytes() + body.tobytes()


def sentinel_bytes(event_id):
    """Bytes of the E:S0 sentinel record that closes an event."""
    return record_bytes(event_id, SENTINEL_SUMMARY_ID, 0.0, [])


def event_bytes(event_id, records, sentinel=True):
    """Bytes of one event's summary records, by default closed with an E:S0 sentinel record.

    Args:
        event_id (int): event the records belong to
        records (list): ``[(summary_id, expval, [(sidx, loss), ...]), ...]``
        sentinel (bool): if True close the event with an ``E:S0`` sentinel record

    Returns:
        bytes: the event's records
    """
    out = b''.join(record_bytes(event_id, summary_id, expval, pairs) for summary_id, expval, pairs in records)
    return out + sentinel_bytes(event_id) if sentinel else out


def make_summary_stream(events, len_sample=3, summary_set_id=1, sentinel=True):
    """Build a summary stream, by default with an E:S0 sentinel after every event.

    Args:
        events (list): ``[(event_id, [(summary_id, expval, [(sidx, loss), ...]), ...]), ...]``
        len_sample (int): number of samples written in the stream header
        summary_set_id (int): summary set written in the stream header
        sentinel (bool): if True close every event with an ``E:S0`` sentinel record

    Returns:
        bytes: the summary stream
    """
    out = bytearray(stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM))
    out += np.int32(len_sample).tobytes() + summaryset_id_dtype.type(summary_set_id).tobytes()
    for event_id, records in events:
        out += event_bytes(event_id, records, sentinel)
    return bytes(out)


def write_stream_files(tmp_path, blobs):
    """Write each summary stream to its own file and return the paths, in input order."""
    paths = []
    for i, blob in enumerate(blobs):
        path = tmp_path / f'rl_IP{i + 1}_S1_rlcombine_P1'
        path.write_bytes(blob)
        paths.append(str(path))
    return paths


def buffer_to_records(event_buffer):
    """Turn an event buffer yielded by SummaryCombineReader.read_events back into
    ``[(summary_id, expval, [(sidx, loss), ...]), ...]``, the form make_summary_stream takes."""
    offsets = event_buffer['offsets']
    return [(int(summary_id), float(event_buffer['expval'][i]),
             list(zip(event_buffer['sidx'][offsets[i]:offsets[i + 1]].tolist(),
                      event_buffer['loss'][offsets[i]:offsets[i + 1]].tolist())))
            for i, summary_id in enumerate(event_buffer['summary_ids'])]


def write_rl_static(static_dir, n_summary_ids, summary_set_id=1):
    """Write an rlsummaryxref mapping outputs 1..n_summary_ids to the same summary_ids, so
    get_max_summary_id finds n_summary_ids for the summary set."""
    static_dir.mkdir(parents=True, exist_ok=True)
    xref = np.zeros(n_summary_ids, dtype=fm_summary_xref_dtype)
    xref['output'] = np.arange(1, n_summary_ids + 1)
    xref['summary_id'] = np.arange(1, n_summary_ids + 1)
    xref['summaryset_id'] = summary_set_id
    xref.tofile(static_dir / f"{SUMMARY_OUTPUT['rl']}.bin")
    return str(static_dir)


def write_summarypy_rl_static(static_dir):
    """Copy the risummaryxref of the summarypy test assets to static_dir as an rlsummaryxref, so
    summarypy -t rl and rlcombine can run on SUMMARYPY_ASSETS/ri.bin."""
    static_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SUMMARYPY_ASSETS / 'single_summary_set' / 'RI_1' / f"{SUMMARY_OUTPUT['ri']}.bin",
                static_dir / f"{SUMMARY_OUTPUT['rl']}.bin")
    return str(static_dir)


def passthrough_file(file_idx):
    """Name of the pass-through output of input file_idx written by run_rlcombine."""
    return f'rl_IP{file_idx + 1}_S1_summary_P1'


def run_rlcombine(tmp_path, streams, passthrough=(), low_memory=False, ext='', max_summary_id=5):
    """Run rlcombine on the summary streams, writing to tmp_path/out.

    Args:
        tmp_path (Path): directory for the inputs, static files and outputs
        streams (list): summary stream of each input, as bytes
        passthrough (list): indices of the inputs to pass through (-I)
        low_memory (bool): if True write .idx files (-m)
        ext (str): extension of the output files, e.g. '.bin'
        max_summary_id (int): number of summary_ids in the rlsummaryxref of each input

    Returns:
        dict: ``{file name: bytes}`` of every file rlcombine wrote, named COMBINED_FILE and
            passthrough_file(i), plus ext
    """
    files = write_stream_files(tmp_path, streams)
    static_paths = [write_rl_static(tmp_path / f'RI_{i + 1}', max_summary_id) for i in range(len(streams))]
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    run(files, str(out_dir / (COMBINED_FILE + ext)), static_paths, low_memory,
        [f'{i}={out_dir / (passthrough_file(i) + ext)}' for i in passthrough], False)
    return {path.name: path.read_bytes() for path in out_dir.iterdir()}


def read_events(files_in, max_summary_id=10):
    """Read summary streams with SummaryCombineReader.

    Returns:
        list: ``(file_idx, event_id, event_buffer)`` for every event, in the order they were yielded
    """
    with ExitStack() as stack:
        streams_in, len_sample, _ = init_summary_streams_in(files_in, stack)
        reader = SummaryCombineReader(len(streams_in), max_summary_id, len_sample)
        return list(reader.read_events(streams_in))


def read_all(files_in, max_summary_id=10):
    """Read summary streams with SummaryCombineReader.

    Returns:
        dict: ``{file_idx: [(event_id, records), ...]}`` in the order each input's events were yielded,
            with records as returned by buffer_to_records
    """
    events = {i: [] for i in range(len(files_in))}
    for file_idx, event_id, event_buffer in read_events(files_in, max_summary_id):
        events[file_idx].append((int(event_id), buffer_to_records(event_buffer)))
    return events


# Two inuring priorities with the same events in the same order. IP1 has nothing for event 7
# and IP2 has nothing for event 12: those events are only an E:S0 sentinel on that input.
IP1 = [(10, [(1, 100, [(-5, 50), (-1, 5), (1, 4)]), (3, 100, [(-5, 9), (-1, 1)])]),
       (7, []),
       (12, [(2, 30, [(-5, 30), (-1, 3), (3, 2.5)])])]
IP2 = [(10, [(3, 10, [(-5, 1), (-1, 1), (2, 7)])]),
       (7, [(1, 5, [(-5, 2), (-1, 2)])]),
       (12, [])]
# A third inuring priority, for passing through a subset of the inputs. It has nothing for event 10.
IP3 = [(10, []),
       (7, [(2, 20, [(-5, 6), (-1, 4), (1, 6), (3, 2)])]),
       (12, [(1, 8, [(-5, 8), (-1, 8), (2, 8)])])]

# IP1 and IP2 combined as rlcombine writes them with len_sample 3: records with the same summary_id
# summed, in summary_id order, max loss and mean first, then the non-zero samples.
IP1_IP2_COMBINED = [(10, [(1, 100, [(-5, 50), (-1, 5), (1, 4)]), (3, 110, [(-5, 10), (-1, 2), (2, 7)])]),
                    (7, [(1, 5, [(-5, 2), (-1, 2)])]),
                    (12, [(2, 30, [(-5, 30), (-1, 3), (3, 2.5)])])]
