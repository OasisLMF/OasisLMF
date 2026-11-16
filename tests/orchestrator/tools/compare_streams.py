"""Compare the per-chunk loss streams ``<tap>_P<k>.bin`` written by two runs.

Two comparisons are provided:

* **byte comparison** (:func:`compare_dirs_bytes`): both directories hold the same files and each
  pair is identical byte for byte. This is the test for two runs with the same number of chunks.
* **event comparison** (:func:`compare_dirs_events`): the files of each tap are cut into events,
  the bytes of every event are digested, and the two runs are equal when they hold the same
  events with the same bytes and the same stream header, whichever file an event was written
  to. This is the test for two runs with a different number of chunks.

Loss stream layout, as written by gulmc and fmpy and read by
``oasislmf.pytools.fm.stream_sparse.read_buffer`` and ``oasislmf.pytools.common.event_stream``:
an 8-byte header (the stream type, an int32 holding the aggregation type in its low three bytes
and the source type in its high byte, then the sample count as int32), followed by item records.
An item record is ``event_id`` (i4) and ``item_id`` (i4), then ``(sidx, loss)`` pairs, sidx i4
and loss ``oasis_float``, the last pair having sidx 0. The records of one event are contiguous.

Command line::

    python tests/orchestrator/tools/compare_streams.py bytes DIR_A DIR_B [--taps gul,il,ri]
    python tests/orchestrator/tools/compare_streams.py events DIR_A DIR_B [--taps gul,il,ri] [--jobs N] [--details N]

The exit status is 0 when the runs are equal and 1 otherwise.
"""
import argparse
import hashlib
import json
import mmap
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np

from oasislmf.pytools.common.data import oasis_float

STREAM_HEADER_SIZE = 8
ITEM_HEADER_WORDS = 2
TAPS = ('gul', 'il', 'ri')
FILE_PATTERN = re.compile(r'^(?P<tap>[A-Za-z0-9]+)_P(?P<k>\d+)\.bin$')
DIGEST_SIZE = 16
MAX_JOBS = 6

SCAN_DONE = 0
SCAN_FULL = 1
SCAN_TRUNCATED = 2
SCAN_BAD_RECORD = 3
RUN_CAPACITY = 1 << 16


def pair_words(loss_dtype=oasis_float):
    """Number of 4-byte words in one ``(sidx, loss)`` pair.

    Args:
        loss_dtype (np.dtype): dtype of the loss, ``oasis_float`` unless the stream was written with another.

    Returns:
        int: 2 for a float32 loss, 3 for a float64 loss.
    """
    return (4 + np.dtype(loss_dtype).itemsize) // 4


def list_stream_files(directory, taps=None):
    """Find the ``<tap>_P<k>.bin`` files of a directory.

    Args:
        directory (str | Path): directory to search (not recursive).
        taps (iterable[str] | None): taps to keep, all found taps if None.

    Returns:
        dict: tap -> {k: Path}, k the 1-based chunk number.
    """
    found = {}
    for path in sorted(Path(directory).iterdir()):
        match = FILE_PATTERN.match(path.name)
        if match is None or not path.is_file():
            continue
        tap = match['tap']
        if taps is not None and tap not in taps:
            continue
        found.setdefault(tap, {})[int(match['k'])] = path
    return found


@dataclass
class FileComparison:
    """Result of comparing one file of two runs byte for byte."""
    name: str
    status: str
    size_a: int = None
    size_b: int = None
    first_difference: int = None

    @property
    def ok(self):
        return self.status == 'identical'

    def __str__(self):
        if self.status == 'identical':
            return f'{self.name}: identical ({self.size_a} bytes)'
        if self.status == 'different':
            return (f'{self.name}: DIFFERENT, first differing byte at offset {self.first_difference}, '
                    f'sizes {self.size_a} and {self.size_b}')
        return f'{self.name}: {self.status.upper()} (sizes {self.size_a} and {self.size_b})'


def compare_files(path_a, path_b, chunk_size=1 << 24):
    """Compare two files byte for byte.

    Args:
        path_a (str | Path): first file.
        path_b (str | Path): second file.
        chunk_size (int): bytes read from each file at a time.

    Returns:
        FileComparison: 'identical', or 'different' with the offset of the first difference. When one
            file is a prefix of the other, the first difference is the size of the shorter one.
    """
    path_a, path_b = Path(path_a), Path(path_b)
    size_a, size_b = path_a.stat().st_size, path_b.stat().st_size
    offset = 0
    with open(path_a, 'rb') as file_a, open(path_b, 'rb') as file_b:
        while True:
            data_a = file_a.read(chunk_size)
            data_b = file_b.read(chunk_size)
            if data_a != data_b:
                n = min(len(data_a), len(data_b))
                diff = np.flatnonzero(np.frombuffer(data_a, np.uint8, n) != np.frombuffer(data_b, np.uint8, n))
                first = offset + (int(diff[0]) if diff.size else n)
                return FileComparison(path_a.name, 'different', size_a, size_b, first)
            if not data_a:
                return FileComparison(path_a.name, 'identical', size_a, size_b)
            offset += len(data_a)


def compare_dirs_bytes(dir_a, dir_b, taps=None):
    """Compare every ``<tap>_P<k>.bin`` file of two directories byte for byte.

    Args:
        dir_a (str | Path): first directory.
        dir_b (str | Path): second directory.
        taps (iterable[str] | None): taps to compare, all taps found in either directory if None.

    Returns:
        list[FileComparison]: one entry per file name present in either directory; a file present
            in only one is reported as 'missing in a' or 'missing in b'.
    """
    files_a = list_stream_files(dir_a, taps)
    files_b = list_stream_files(dir_b, taps)
    results = []
    for tap in sorted(set(files_a) | set(files_b)):
        chunks_a, chunks_b = files_a.get(tap, {}), files_b.get(tap, {})
        for k in sorted(set(chunks_a) | set(chunks_b)):
            name = f'{tap}_P{k}.bin'
            if k not in chunks_a:
                results.append(FileComparison(name, 'missing in a', None, chunks_b[k].stat().st_size))
            elif k not in chunks_b:
                results.append(FileComparison(name, 'missing in b', chunks_a[k].stat().st_size, None))
            else:
                results.append(compare_files(chunks_a[k], chunks_b[k]))
    return results


@nb.njit(cache=True, nogil=True)
def _scan_event_runs(words, n_pair_words, pos, state, run_event, run_start, run_end):
    """Find the runs of consecutive item records sharing an event id.

    Resumable: when the run buffers are full it returns SCAN_FULL and the position to call again
    from. ``state`` carries the open run between calls: [event id, start word, 1 if open].

    Args:
        words (np.ndarray[int32]): the stream after its header.
        n_pair_words (int): words per (sidx, loss) pair.
        pos (int): word to start at, at an item record boundary.
        state (np.ndarray[int64]): open run, updated in place.
        run_event (np.ndarray[int64]): event id of each closed run, filled in.
        run_start (np.ndarray[int64]): first word of each closed run, filled in.
        run_end (np.ndarray[int64]): word after the last of each closed run, filled in.

    Returns:
        (n_runs, pos, status): runs written, where the scan stopped, and a SCAN_* code. On
            SCAN_TRUNCATED and SCAN_BAD_RECORD, pos is the start of the faulty item record.
    """
    n = words.shape[0]
    capacity = run_event.shape[0]
    n_runs = 0
    while pos < n:
        if pos + ITEM_HEADER_WORDS > n:
            return n_runs, pos, SCAN_TRUNCATED
        event_id = words[pos]
        if event_id <= 0 or words[pos + 1] <= 0:
            return n_runs, pos, SCAN_BAD_RECORD
        if state[2] == 0 or event_id != state[0]:
            if state[2]:
                if n_runs == capacity:
                    return n_runs, pos, SCAN_FULL
                run_event[n_runs] = state[0]
                run_start[n_runs] = state[1]
                run_end[n_runs] = pos
                n_runs += 1
            state[0] = event_id
            state[1] = pos
            state[2] = 1
        item_start = pos
        pos += ITEM_HEADER_WORDS
        while True:
            if pos + n_pair_words > n:
                return n_runs, item_start, SCAN_TRUNCATED
            sidx = words[pos]
            pos += n_pair_words
            if sidx == 0:
                break
    if state[2]:
        if n_runs == capacity:
            return n_runs, pos, SCAN_FULL
        run_event[n_runs] = state[0]
        run_start[n_runs] = state[1]
        run_end[n_runs] = pos
        n_runs += 1
        state[2] = 0
    return n_runs, pos, SCAN_DONE


@nb.njit(cache=True)
def _locate_pairs(words, n_pair_words):
    """Word position of every (sidx, loss) pair of an event block and the item id it belongs to.

    Args:
        words (np.ndarray[int32]): whole item records, without stream header.
        n_pair_words (int): words per (sidx, loss) pair.

    Returns:
        (item_ids, positions, complete): arrays of the same length, and False if the block ends inside a record.
    """
    n = words.shape[0]
    item_ids = np.empty(n // n_pair_words + 1, np.int32)
    positions = np.empty(n // n_pair_words + 1, np.int64)
    n_pairs = 0
    pos = 0
    while pos < n:
        if pos + ITEM_HEADER_WORDS > n:
            return item_ids[:n_pairs], positions[:n_pairs], False
        item_id = words[pos + 1]
        pos += ITEM_HEADER_WORDS
        while True:
            if pos + n_pair_words > n:
                return item_ids[:n_pairs], positions[:n_pairs], False
            item_ids[n_pairs] = item_id
            positions[n_pairs] = pos
            n_pairs += 1
            sidx = words[pos]
            pos += n_pair_words
            if sidx == 0:
                break
    return item_ids[:n_pairs], positions[:n_pairs], True


def decode_event_rows(data, loss_dtype=oasis_float):
    """Decode a block of item records into one row per (sidx, loss) pair, delimiters included.

    Args:
        data (bytes | np.ndarray): whole item records, without stream header.
        loss_dtype (np.dtype): dtype of the loss.

    Returns:
        np.ndarray: structured array with fields event_id, item_id, sidx and loss.

    Raises:
        ValueError: if the block ends inside an item record.
    """
    raw = np.frombuffer(data, np.uint8)
    words = raw[:raw.size - raw.size % 4].view('<i4')
    n_pair_words = pair_words(loss_dtype)
    item_ids, positions, complete = _locate_pairs(words, n_pair_words)
    if not complete:
        raise ValueError('event block ends inside an item record')
    loss_dtype = np.dtype(loss_dtype)
    byte_pos = positions * 4 + 4
    loss_bytes = raw[byte_pos[:, None] + np.arange(loss_dtype.itemsize)]
    rows = np.empty(positions.size, dtype=[('event_id', 'i4'), ('item_id', 'i4'), ('sidx', 'i4'), ('loss', loss_dtype)])
    rows['event_id'] = words[0] if words.size else 0
    rows['item_id'] = item_ids
    rows['sidx'] = words[positions]
    rows['loss'] = loss_bytes.reshape(-1).view(loss_dtype) if positions.size else []
    return rows


@dataclass
class EventInfo:
    """Where an event was found and the digest of its bytes."""
    digest: bytes
    nbytes: int
    path: Path
    offset: int


@dataclass
class FileScan:
    """Events and header of one stream file."""
    path: Path
    header: bytes
    events: list = field(default_factory=list)
    error: str = None


def scan_file(path, loss_dtype=oasis_float, run_capacity=RUN_CAPACITY):
    """Cut one stream file into events and digest each event's bytes.

    The file is memory-mapped, so its pages are file-backed and can be dropped by the OS; nothing
    beyond the run buffers is allocated per file.

    Args:
        path (str | Path): stream file.
        loss_dtype (np.dtype): dtype of the loss.
        run_capacity (int): size of the run buffers handed to the scanning kernel.

    Returns:
        FileScan: header, list of (event_id, EventInfo) in file order, and an error message if the
            file is not a well-formed loss stream. After a fault only the events that ended before
            the fault's event began are listed, as the event in progress cannot be known to be complete.
    """
    path = Path(path)
    size = path.stat().st_size
    with open(path, 'rb') as stream:
        header = stream.read(STREAM_HEADER_SIZE)
        if size < STREAM_HEADER_SIZE:
            return FileScan(path, header, error=f'{size} bytes, shorter than the {STREAM_HEADER_SIZE}-byte stream header')
        if size == STREAM_HEADER_SIZE:
            return FileScan(path, header)
        scan = FileScan(path, header)
        n_words = (size - STREAM_HEADER_SIZE) // 4
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped, memoryview(mapped) as view:
            words = np.frombuffer(mapped, dtype='<i4', count=n_words, offset=STREAM_HEADER_SIZE)
            state = np.zeros(3, np.int64)
            run_event = np.empty(run_capacity, np.int64)
            run_start = np.empty(run_capacity, np.int64)
            run_end = np.empty(run_capacity, np.int64)
            pos, status = 0, SCAN_FULL
            while status == SCAN_FULL:
                n_runs, pos, status = _scan_event_runs(words, pair_words(loss_dtype), pos, state, run_event, run_start, run_end)
                for event_id, start, end in zip(run_event[:n_runs].tolist(), run_start[:n_runs].tolist(), run_end[:n_runs].tolist()):
                    begin, stop = STREAM_HEADER_SIZE + 4 * start, STREAM_HEADER_SIZE + 4 * end
                    digest = hashlib.sha256(view[begin:stop]).digest()[:DIGEST_SIZE]
                    scan.events.append((event_id, EventInfo(digest, stop - begin, path, begin)))
            del words
        offset = STREAM_HEADER_SIZE + 4 * pos
        if status == SCAN_TRUNCATED:
            scan.error = f'truncated: the item record at byte {offset} is incomplete (file size {size})'
        elif status == SCAN_BAD_RECORD:
            scan.error = f'bad item record at byte {offset}: event_id and item_id must be positive'
        elif (size - STREAM_HEADER_SIZE) % 4:
            scan.error = f'truncated: {(size - STREAM_HEADER_SIZE) % 4} trailing bytes after the last record'
    return scan


@dataclass
class EventMap:
    """Every event of one tap of one run, keyed by event id."""
    tap: str
    files: list = field(default_factory=list)
    headers: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    duplicates: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def header(self):
        """The stream header shared by all files, or None if there are none or they disagree."""
        values = set(self.headers.values())
        return values.pop() if len(values) == 1 else None

    @property
    def nbytes(self):
        return sum(info.nbytes for info in self.events.values())


def build_event_map(paths, tap='', loss_dtype=oasis_float, jobs=None):
    """Scan the files of one tap and map each event id to the digest of its bytes.

    Args:
        paths (iterable[str | Path]): the tap's stream files.
        tap (str): tap name, for reporting.
        loss_dtype (np.dtype): dtype of the loss.
        jobs (int | None): files scanned in parallel (threads; the kernel and the hash release the GIL).

    Returns:
        EventMap: the events, the header of each file, events found more than once (in two files, or
            in two separate runs of one file) and file format errors.
    """
    paths = [Path(p) for p in paths]
    jobs = jobs or min(MAX_JOBS, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(paths) or 1))) as pool:
        scans = list(pool.map(lambda p: scan_file(p, loss_dtype), paths))
    event_map = EventMap(tap, files=paths)
    for scan in scans:
        event_map.headers[scan.path.name] = scan.header
        if scan.error:
            event_map.errors.append(f'{scan.path.name}: {scan.error}')
        for event_id, info in scan.events:
            if event_id in event_map.events:
                first = event_map.events[event_id]
                event_map.duplicates.append(
                    f'event {event_id} in {first.path.name} at byte {first.offset} and again in {info.path.name} at byte {info.offset}')
            else:
                event_map.events[event_id] = info
    if len(set(event_map.headers.values())) > 1:
        event_map.errors.append(f'files disagree on the stream header: {describe_headers(event_map.headers)}')
    return event_map


def describe_header(header):
    """Human-readable form of an 8-byte loss stream header.

    Args:
        header (bytes): the header.

    Returns:
        str: hex bytes with the decoded source type, aggregation type and sample count.
    """
    if header is None or len(header) < STREAM_HEADER_SIZE:
        return f'{header!r} (incomplete)'
    stream_type, samples = np.frombuffer(header[:STREAM_HEADER_SIZE], '<i4')
    return f'{header.hex()} (source type {stream_type >> 24}, aggregation type {stream_type & 0xFFFFFF}, samples {samples})'


def describe_headers(headers):
    groups = {}
    for name, header in headers.items():
        groups.setdefault(header, []).append(name)
    return '; '.join(f'{describe_header(h)} in {", ".join(names)}' for h, names in groups.items())


@dataclass
class EventDifference:
    """One event whose bytes differ between two runs."""
    event_id: int
    a: EventInfo
    b: EventInfo

    def describe(self, loss_dtype=oasis_float, max_rows=5):
        """Decode both versions of the event and say how they differ.

        Args:
            loss_dtype (np.dtype): dtype of the loss.
            max_rows (int): differing rows to show.

        Returns:
            str: multi-line description.
        """
        rows_a = decode_event_rows(_read_bytes(self.a), loss_dtype)
        rows_b = decode_event_rows(_read_bytes(self.b), loss_dtype)
        lines = [f'event {self.event_id}: {self.a.nbytes} bytes in {self.a.path.name}@{self.a.offset}, '
                 f'{self.b.nbytes} bytes in {self.b.path.name}@{self.b.offset}; {rows_a.size} and {rows_b.size} pairs']
        keys_a, keys_b = _row_keys(rows_a), _row_keys(rows_b)
        common, idx_a, idx_b = np.intersect1d(keys_a, keys_b, assume_unique=False, return_indices=True)
        only_a, only_b = np.setdiff1d(keys_a, keys_b), np.setdiff1d(keys_b, keys_a)
        items_a, items_b = np.unique(rows_a['item_id']), np.unique(rows_b['item_id'])
        lines.append(f'  items: {items_a.size} and {items_b.size}, {np.setdiff1d(items_a, items_b).size} only in a, '
                     f'{np.setdiff1d(items_b, items_a).size} only in b')
        lines.append(f'  (item, sidx) pairs: {common.size} in both, {only_a.size} only in a, {only_b.size} only in b')
        loss_a, loss_b = rows_a['loss'][idx_a], rows_b['loss'][idx_b]
        differ = ~((loss_a == loss_b) | (np.isnan(loss_a) & np.isnan(loss_b)))
        if common.size:
            abs_diff = np.abs(loss_a.astype('f8') - loss_b.astype('f8'))
            scale = np.maximum(np.abs(loss_a.astype('f8')), np.abs(loss_b.astype('f8')))
            rel = np.divide(abs_diff, scale, out=np.zeros_like(abs_diff), where=scale > 0)
            lines.append(f'  losses differing on common pairs: {int(differ.sum())}, max abs diff {np.nanmax(abs_diff):.6g}, '
                         f'max rel diff {np.nanmax(rel):.3g}')
        if not differ.any() and not only_a.size and not only_b.size:
            lines.append('  same (item, sidx, loss) rows, in a different order')
        for i in np.flatnonzero(differ)[:max_rows]:
            ra, rb = rows_a[idx_a[i]], rows_b[idx_b[i]]
            lines.append(f'    item {ra["item_id"]} sidx {ra["sidx"]}: {ra["loss"]!r} vs {rb["loss"]!r}')
        n = min(rows_a.size, rows_b.size)
        mismatch = np.flatnonzero(rows_a[:n] != rows_b[:n])
        first = int(mismatch[0]) if mismatch.size else n
        if first < max(rows_a.size, rows_b.size):
            show_a = tuple(rows_a[first].tolist()) if first < rows_a.size else None
            show_b = tuple(rows_b[first].tolist()) if first < rows_b.size else None
            lines.append(f'  first differing pair in stream order: #{first}: a={show_a} b={show_b}')
        return '\n'.join(lines)


def _read_bytes(info):
    with open(info.path, 'rb') as stream:
        stream.seek(info.offset)
        return stream.read(info.nbytes)


def _row_keys(rows):
    return (rows['item_id'].astype(np.int64) << 32) | (rows['sidx'].astype(np.int64) & 0xFFFFFFFF)


@dataclass
class EventMapComparison:
    """Result of comparing the event maps of one tap of two runs."""
    tap: str
    a: EventMap
    b: EventMap
    only_a: list
    only_b: list
    different: list
    header_equal: bool

    @property
    def ok(self):
        return (self.header_equal and not self.only_a and not self.only_b and not self.different
                and not self.a.errors and not self.b.errors and not self.a.duplicates and not self.b.duplicates)

    def summary(self, details=0, loss_dtype=oasis_float):
        """Multi-line report.

        Args:
            details (int): number of differing events to decode and describe.
            loss_dtype (np.dtype): dtype of the loss.

        Returns:
            str: the report.
        """
        status = 'EQUAL' if self.ok else 'DIFFERENT'
        lines = [f'{self.tap}: {status}: {len(self.a.events)} events in {len(self.a.files)} files ({self.a.nbytes} bytes) vs '
                 f'{len(self.b.events)} events in {len(self.b.files)} files ({self.b.nbytes} bytes)']
        if self.header_equal:
            lines.append(f'  header {describe_header(self.a.header)}')
        else:
            lines.append(f'  HEADERS DIFFER: a: {describe_headers(self.a.headers)}; b: {describe_headers(self.b.headers)}')
        for side, event_map in (('a', self.a), ('b', self.b)):
            lines.extend(f'  {side}: {message}' for message in event_map.errors)
            lines.extend(f'  {side}: duplicate {message}' for message in event_map.duplicates[:20])
        if self.only_a:
            lines.append(f'  {len(self.only_a)} events only in a: {_preview(self.only_a)}')
        if self.only_b:
            lines.append(f'  {len(self.only_b)} events only in b: {_preview(self.only_b)}')
        if self.different:
            lines.append(f'  {len(self.different)} events with different bytes: {_preview([d.event_id for d in self.different])}')
            for difference in self.different[:details]:
                lines.append(difference.describe(loss_dtype))
        return '\n'.join(lines)


def _preview(values, n=20):
    shown = ', '.join(str(v) for v in values[:n])
    return shown + (', ...' if len(values) > n else '')


def compare_event_maps(map_a, map_b):
    """Compare two event maps of the same tap.

    Args:
        map_a (EventMap): first run.
        map_b (EventMap): second run.

    Returns:
        EventMapComparison: the events present in only one run, those whose bytes differ, and whether
            the stream headers agree.
    """
    ids_a, ids_b = set(map_a.events), set(map_b.events)
    different = [EventDifference(e, map_a.events[e], map_b.events[e]) for e in sorted(ids_a & ids_b)
                 if map_a.events[e].digest != map_b.events[e].digest]
    header_equal = (map_a.header is not None and map_a.header == map_b.header) or (not map_a.headers and not map_b.headers)
    return EventMapComparison(map_a.tap, map_a, map_b, sorted(ids_a - ids_b), sorted(ids_b - ids_a), different, header_equal)


def compare_dirs_events(dir_a, dir_b, taps=None, loss_dtype=oasis_float, jobs=None):
    """Partition-independent comparison of the streams of two directories.

    Args:
        dir_a (str | Path): first directory.
        dir_b (str | Path): second directory.
        taps (iterable[str] | None): taps to compare, all taps found in either directory if None.
        loss_dtype (np.dtype): dtype of the loss.
        jobs (int | None): files scanned in parallel.

    Returns:
        dict: tap -> EventMapComparison. A tap with files on one side only compares against an empty map.
    """
    files_a = list_stream_files(dir_a, taps)
    files_b = list_stream_files(dir_b, taps)
    results = {}
    for tap in sorted(set(files_a) | set(files_b), key=lambda t: (TAPS.index(t) if t in TAPS else len(TAPS), t)):
        map_a = build_event_map(files_a.get(tap, {}).values(), tap, loss_dtype, jobs)
        map_b = build_event_map(files_b.get(tap, {}).values(), tap, loss_dtype, jobs)
        results[tap] = compare_event_maps(map_a, map_b)
    return results


def _parse_taps(value):
    return None if value in (None, '', 'all') else tuple(t for t in value.split(',') if t)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode, help_text in (('bytes', 'compare files of the same name byte for byte'),
                            ('events', 'compare the events of each tap, whatever file they are in')):
        p = sub.add_parser(mode, help=help_text)
        p.add_argument('dir_a')
        p.add_argument('dir_b')
        p.add_argument('--taps', default=None, help='comma-separated taps (default: every tap found)')
        p.add_argument('--json', dest='json_path', default=None, help='also write the result as JSON to this path')
        if mode == 'events':
            p.add_argument('--jobs', type=int, default=None, help=f'files scanned in parallel (default min({MAX_JOBS}, CPUs))')
            p.add_argument('--details', type=int, default=3, help='differing events to decode and describe per tap')
    args = parser.parse_args(argv)
    taps = _parse_taps(args.taps)
    start = time.perf_counter()
    if args.mode == 'bytes':
        results = compare_dirs_bytes(args.dir_a, args.dir_b, taps)
        for result in results:
            print(result)
        ok = bool(results) and all(r.ok for r in results)
        print(f'{"OK" if ok else "FAILED"}: {sum(r.ok for r in results)}/{len(results)} files identical '
              f'({time.perf_counter() - start:.1f}s)')
        payload = [vars(r) for r in results]
    else:
        results = compare_dirs_events(args.dir_a, args.dir_b, taps, jobs=args.jobs)
        for result in results.values():
            print(result.summary(details=args.details))
        ok = bool(results) and all(r.ok for r in results.values())
        print(f'{"OK" if ok else "FAILED"}: {sum(r.ok for r in results.values())}/{len(results)} taps equal '
              f'({time.perf_counter() - start:.1f}s)')
        payload = {tap: {'ok': r.ok, 'events_a': len(r.a.events), 'events_b': len(r.b.events), 'only_a': r.only_a,
                         'only_b': r.only_b, 'different': [d.event_id for d in r.different], 'header_equal': r.header_equal,
                         'errors_a': r.a.errors, 'errors_b': r.b.errors, 'duplicates_a': r.a.duplicates,
                         'duplicates_b': r.b.duplicates} for tap, r in results.items()}
    if not results:
        print('no <tap>_P<k>.bin files found')
    if args.json_path:
        with open(args.json_path, 'w') as out:
            json.dump({'ok': ok, 'mode': args.mode, 'results': payload}, out, indent=2, default=str)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
