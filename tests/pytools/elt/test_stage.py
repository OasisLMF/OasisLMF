"""EltStage: same bytes as eltpy, whatever the chunks and blocks the summary stream is fed in."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import GUL_STREAM_ID, stream_info_to_bytes
from oasislmf.pytools.elt.manager import main as eltpy
from oasislmf.pytools.elt.stage import EltStage

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "test_eltpy"
SUMMARY_STREAM = ASSETS_DIR / "summarypy.bin"
SUMMARY_HEADER_SIZE = 12

OUTPUT_FORMATS = [("bin", False), ("csv", False), ("csv", True), ("parquet", False)]


def split_summary_stream(data):
    """Split a summary stream into its 12-byte header and its records.

    Args:
        data (bytes): the whole stream, header included.

    Returns:
        tuple: ``(header, records)``; records is a list of ``(event_id, bytes)``, one per summary
        record (summary header, its (sidx, loss) pairs and the closing (0, 0) pair).
    """
    header, body = data[:SUMMARY_HEADER_SIZE], data[SUMMARY_HEADER_SIZE:]
    ints = np.frombuffer(body, dtype="<i4")
    records = []
    i = 0
    while i < ints.shape[0]:
        start = i
        event_id = int(ints[i])
        i += 3
        while ints[i] != 0:
            i += 2
        i += 2
        records.append((event_id, body[4 * start:4 * i]))
    return header, records


def group_events(records):
    """Group the records of a summary stream by event.

    Returns:
        list: ``(event_id, bytes)`` per event, in stream order.
    """
    events = []
    for event_id, record in records:
        if events and events[-1][0] == event_id:
            events[-1] = (event_id, events[-1][1] + record)
        else:
            events.append((event_id, record))
    return events


def to_block(data, writable=True):
    """Return ``data`` as a 1-D int8 block, read-only unless ``writable``."""
    block = np.frombuffer(data, dtype=np.int8)
    return block.copy() if writable else block


def read_outputs(paths):
    """Read each output file of a {out_type: path} dict."""
    return {out_type: Path(path).read_bytes() for out_type, path in paths.items()}


def out_paths(directory, tag, out_types, ext):
    """Output path of each out_type, inside ``directory``."""
    return {out_type: Path(directory, f"{tag}_{out_type}.{ext}") for out_type in out_types}


def run_stage(stage, chunks, directory, tag, ext):
    """Feed each chunk (a list of blocks) through one stage and return the outputs of each chunk."""
    outputs = []
    for k, blocks in enumerate(chunks):
        paths = out_paths(directory, f"{tag}_P{k}", stage.active_types, ext)
        stage.start_chunk(paths)
        for block in blocks:
            stage.process_block(block)
        stage.finish_chunk()
        outputs.append(read_outputs(paths))
    return outputs


@pytest.fixture
def elt_run_dir(tmp_path):
    """A run directory with quantiles and event rates."""
    run_dir = tmp_path / "run"
    (run_dir / "input").mkdir(parents=True)
    shutil.copy(ASSETS_DIR / "input" / "quantile.bin", run_dir / "input" / "quantile.bin")
    shutil.copy(ASSETS_DIR / "input" / "er.csv", run_dir / "input" / "event_rates.csv")
    return run_dir


def eltpy_outputs(run_dir, stream_path, out_types, ext, noheader, directory, tag):
    """Run the eltpy command on a stream file and return its outputs."""
    paths = out_paths(directory, tag, out_types, ext)
    eltpy(run_dir=run_dir, files_in=[str(stream_path)], noheader=noheader, ext=ext,
          **{out_type: str(path) for out_type, path in paths.items()})
    return read_outputs(paths)


def make_stage(run_dir, header, out_types, ext="bin", noheader=False):
    return EltStage(run_dir, header, output_format=ext, noheader=noheader, **{out_type: True for out_type in out_types})


@pytest.mark.parametrize("ext,noheader", OUTPUT_FORMATS)
@pytest.mark.parametrize("out_types", [("selt",), ("melt",), ("qelt",), ("selt", "melt", "qelt")])
def test_stage_matches_eltpy(elt_run_dir, tmp_path, out_types, ext, noheader):
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    expected = eltpy_outputs(elt_run_dir, SUMMARY_STREAM, out_types, ext, noheader, tmp_path, "cli")
    assert all(len(v) > 0 for v in expected.values())

    with make_stage(elt_run_dir, header, out_types, ext, noheader) as stage:
        [actual] = run_stage(stage, [[to_block(data[SUMMARY_HEADER_SIZE:])]], tmp_path, "stage", ext)
    assert actual == expected


@pytest.mark.parametrize("ext,noheader", OUTPUT_FORMATS)
def test_chunks_through_one_stage_equal_fresh_stages_and_eltpy(elt_run_dir, tmp_path, ext, noheader):
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    n_chunks = 3
    chunks = [[to_block(b"".join(ev for _, ev in events[k::n_chunks]))] for k in range(n_chunks)]
    out_types = ("selt", "melt", "qelt")

    expected = []
    for k in range(n_chunks):
        chunk_path = tmp_path / f"chunk_{k}.bin"
        chunk_path.write_bytes(header + chunks[k][0].tobytes())
        expected.append(eltpy_outputs(elt_run_dir, chunk_path, out_types, ext, noheader, tmp_path, f"cli_P{k}"))

    fresh = []
    for k in range(n_chunks):
        with make_stage(elt_run_dir, header, out_types, ext, noheader) as stage:
            fresh += run_stage(stage, [chunks[k]], tmp_path, f"fresh{k}", ext)
    assert fresh == expected

    with make_stage(elt_run_dir, header, out_types, ext, noheader) as stage:
        assert run_stage(stage, chunks, tmp_path, "one", ext) == expected
        order = [2, 0, 1]
        scrambled = run_stage(stage, [chunks[k] for k in order], tmp_path, "scrambled", ext)
    assert scrambled == [expected[k] for k in order]


def test_block_splitting_does_not_change_output(elt_run_dir, tmp_path):
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    out_types = ("selt", "melt", "qelt")
    with make_stage(elt_run_dir, header, out_types, "csv") as stage:
        [expected] = run_stage(stage, [[to_block(data[SUMMARY_HEADER_SIZE:])]], tmp_path, "whole", "csv")

        one_event_per_block = [to_block(ev) for _, ev in events]
        rng = np.random.default_rng(42)
        cuts = np.sort(rng.choice(np.arange(1, len(events)), size=7, replace=False))
        random_groups = [to_block(b"".join(ev for _, ev in group)) for group in np.split(np.array(events, dtype=object), cuts)]
        one_record_per_block = [to_block(rec) for _, rec in records]

        splits = [one_event_per_block, random_groups, one_record_per_block]
        for i, blocks in enumerate(splits):
            [actual] = run_stage(stage, [blocks], tmp_path, f"split{i}", "csv")
            assert actual == expected, f"split {i}"


def test_read_only_blocks(elt_run_dir, tmp_path):
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    out_types = ("selt", "melt", "qelt")
    expected = eltpy_outputs(elt_run_dir, SUMMARY_STREAM, out_types, "bin", False, tmp_path, "cli")
    blocks = [to_block(ev, writable=False) for _, ev in events]
    assert not blocks[0].flags.writeable
    with make_stage(elt_run_dir, header, out_types) as stage:
        [actual] = run_stage(stage, [blocks], tmp_path, "ro", "bin")
    assert actual == expected


def test_reader_state_is_clean_at_event_boundaries(elt_run_dir):
    """After whole events, only the id of the last event is left in the reader state (it yields once more, with no rows)."""
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    with make_stage(elt_run_dir, header, ("selt", "melt", "qelt")) as stage:
        fresh = stage.reader.state.copy()
        previous_event_id = None
        for event_id, ev in events[:5]:
            yielded = list(stage.reader.read_block(to_block(ev)))
            assert yielded == ([event_id] if previous_event_id is None else [previous_event_id, event_id])
            for out_type in stage.out_types:
                stage.reader.get_data_idx(out_type)[0] = 0
            state = stage.reader.state
            assert state["current_event_id"][0] == event_id
            for field in state.dtype.names:
                if field in ("current_event_id", "summary_id", "impacted_exposure"):
                    continue
                assert np.array_equal(state[field], fresh[field]), field
            previous_event_id = event_id


def test_start_chunk_resets_a_chunk_abandoned_mid_event(elt_run_dir, tmp_path):
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    out_types = ("selt", "melt", "qelt")
    rest = [to_block(ev) for _, ev in events[3:]]

    with make_stage(elt_run_dir, header, out_types) as stage:
        [expected] = run_stage(stage, [rest], tmp_path, "fresh", "bin")

    with make_stage(elt_run_dir, header, out_types) as stage:
        truncated = records[0][1][:12 + 8 * 5]
        stage.start_chunk(out_paths(tmp_path, "aborted", out_types, "bin"))
        stage.process_block(to_block(truncated))
        assert stage.reader.state["reading_losses"][0]
        stage.finish_chunk()
        [actual] = run_stage(stage, [rest], tmp_path, "after", "bin")
    assert actual == expected


def test_header_validation(elt_run_dir):
    data = SUMMARY_STREAM.read_bytes()
    header = data[:SUMMARY_HEADER_SIZE]
    with make_stage(elt_run_dir, header[:8], ("selt",)) as stage:
        assert stage.len_sample == 100 and stage.summary_set_id is None
    with make_stage(elt_run_dir, header, ("selt",)) as stage:
        assert stage.len_sample == 100 and stage.summary_set_id == 1
    with pytest.raises(ValueError):
        make_stage(elt_run_dir, header[:10], ("selt",))
    gul_header = stream_info_to_bytes(GUL_STREAM_ID, 1) + header[4:]
    with pytest.raises(Exception, match="unsupported stream type"):
        make_stage(elt_run_dir, gul_header, ("selt",))


def test_start_chunk_checks_paths(elt_run_dir, tmp_path):
    header = SUMMARY_STREAM.read_bytes()[:SUMMARY_HEADER_SIZE]
    with make_stage(elt_run_dir, header, ("selt", "qelt"), "csv") as stage:
        with pytest.raises(ValueError, match="paths must be given"):
            stage.start_chunk({"selt": tmp_path / "a.csv"})
        with pytest.raises(ValueError, match="Invalid file extension"):
            stage.start_chunk({"selt": tmp_path / "a.bin", "qelt": tmp_path / "b.bin"})
        with pytest.raises(ValueError, match="no chunk is open"):
            stage.process_block(np.zeros(0, dtype=np.int8))
        stage.start_chunk({"selt": tmp_path / "a.csv", "qelt": tmp_path / "b.csv"})
        with pytest.raises(ValueError, match="already open"):
            stage.start_chunk({"selt": tmp_path / "c.csv", "qelt": tmp_path / "d.csv"})
    assert (tmp_path / "a.csv").read_text().startswith("EventId,SummaryId,SampleId")


def test_stage_lifecycle_errors(elt_run_dir, tmp_path):
    header = SUMMARY_STREAM.read_bytes()[:SUMMARY_HEADER_SIZE]
    stage = make_stage(elt_run_dir, header, ("selt", "qelt"), "csv")
    with pytest.raises(ValueError, match="no chunk is open"):
        stage.process_streams([])
    with pytest.raises(ValueError, match="no chunk is open"):
        stage.finish_chunk()
    with pytest.raises(FileNotFoundError):
        stage.start_chunk({"selt": tmp_path / "a.csv", "qelt": tmp_path / "missing" / "b.csv"})
    stage.start_chunk({"selt": tmp_path / "a.csv", "qelt": tmp_path / "b.csv"})
    stage.finish_chunk()
    stage.close()
    stage.close()
    with pytest.raises(ValueError, match="stage is closed"):
        stage.start_chunk({"selt": tmp_path / "c.csv", "qelt": tmp_path / "d.csv"})
