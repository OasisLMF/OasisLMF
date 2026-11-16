"""PltStage: same bytes as pltpy, whatever the chunks and blocks the summary stream is fed in."""
from pathlib import Path

import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import GUL_STREAM_ID, stream_info_to_bytes
from oasislmf.pytools.plt.manager import main as pltpy
from oasislmf.pytools.plt.stage import PltStage
from tests.pytools.elt.test_stage import (OUTPUT_FORMATS, SUMMARY_HEADER_SIZE, group_events, out_paths, read_outputs, run_stage,
                                          split_summary_stream, to_block)

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "test_pltpy"
SUMMARY_STREAM = ASSETS_DIR / "summarypy.bin"
RUN_DIRS = ["all_files", "no_files", "occ_gran_files"]
ALL_TYPES = ("splt", "mplt", "qplt")


def pltpy_outputs(run_dir, stream_path, out_types, ext, noheader, directory, tag):
    """Run the pltpy command on a stream file and return its outputs."""
    paths = out_paths(directory, tag, out_types, ext)
    pltpy(run_dir=run_dir, files_in=[str(stream_path)], noheader=noheader, ext=ext,
          **{out_type: str(path) for out_type, path in paths.items()})
    return read_outputs(paths)


def make_stage(run_dir, header, out_types, ext="bin", noheader=False):
    return PltStage(run_dir, header, output_format=ext, noheader=noheader, **{out_type: True for out_type in out_types})


@pytest.mark.parametrize("run_dir_name", RUN_DIRS)
@pytest.mark.parametrize("ext,noheader", OUTPUT_FORMATS)
@pytest.mark.parametrize("out_types", [("splt",), ("mplt",), ("qplt",), ALL_TYPES])
def test_stage_matches_pltpy(tmp_path, run_dir_name, out_types, ext, noheader):
    run_dir = ASSETS_DIR / run_dir_name
    data = SUMMARY_STREAM.read_bytes()
    header = data[:SUMMARY_HEADER_SIZE]
    expected = pltpy_outputs(run_dir, SUMMARY_STREAM, out_types, ext, noheader, tmp_path, "cli")
    assert all(len(v) > 0 for v in expected.values())

    with make_stage(run_dir, header, out_types, ext, noheader) as stage:
        [actual] = run_stage(stage, [[to_block(data[SUMMARY_HEADER_SIZE:])]], tmp_path, "stage", ext)
    assert actual == expected


@pytest.mark.parametrize("ext,noheader", OUTPUT_FORMATS)
def test_chunks_through_one_stage_equal_fresh_stages_and_pltpy(tmp_path, ext, noheader):
    run_dir = ASSETS_DIR / "all_files"
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    n_chunks = 3
    chunks = [[to_block(b"".join(ev for _, ev in events[k::n_chunks]))] for k in range(n_chunks)]

    expected = []
    for k in range(n_chunks):
        chunk_path = tmp_path / f"chunk_{k}.bin"
        chunk_path.write_bytes(header + chunks[k][0].tobytes())
        expected.append(pltpy_outputs(run_dir, chunk_path, ALL_TYPES, ext, noheader, tmp_path, f"cli_P{k}"))

    fresh = []
    for k in range(n_chunks):
        with make_stage(run_dir, header, ALL_TYPES, ext, noheader) as stage:
            fresh += run_stage(stage, [chunks[k]], tmp_path, f"fresh{k}", ext)
    assert fresh == expected

    with make_stage(run_dir, header, ALL_TYPES, ext, noheader) as stage:
        assert run_stage(stage, chunks, tmp_path, "one", ext) == expected
        order = [2, 0, 1]
        scrambled = run_stage(stage, [chunks[k] for k in order], tmp_path, "scrambled", ext)
    assert scrambled == [expected[k] for k in order]


def test_block_splitting_does_not_change_output(tmp_path):
    run_dir = ASSETS_DIR / "all_files"
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    with make_stage(run_dir, header, ALL_TYPES, "csv") as stage:
        [expected] = run_stage(stage, [[to_block(data[SUMMARY_HEADER_SIZE:])]], tmp_path, "whole", "csv")

        one_event_per_block = [to_block(ev) for _, ev in events]
        rng = np.random.default_rng(42)
        cuts = np.sort(rng.choice(np.arange(1, len(events)), size=7, replace=False))
        random_groups = [to_block(b"".join(ev for _, ev in group)) for group in np.split(np.array(events, dtype=object), cuts)]
        one_record_per_block = [to_block(rec) for _, rec in records]

        for i, blocks in enumerate([one_event_per_block, random_groups, one_record_per_block]):
            [actual] = run_stage(stage, [blocks], tmp_path, f"split{i}", "csv")
            assert actual == expected, f"split {i}"


def test_read_only_blocks(tmp_path):
    run_dir = ASSETS_DIR / "all_files"
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    expected = pltpy_outputs(run_dir, SUMMARY_STREAM, ALL_TYPES, "bin", False, tmp_path, "cli")
    blocks = [to_block(ev, writable=False) for _, ev in events]
    assert not blocks[0].flags.writeable
    with make_stage(run_dir, header, ALL_TYPES) as stage:
        [actual] = run_stage(stage, [blocks], tmp_path, "ro", "bin")
    assert actual == expected


def test_reader_state_is_clean_at_event_boundaries():
    """After whole events, only the id of the last event is left in the reader state (it yields once more, with no rows)."""
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    with make_stage(ASSETS_DIR / "all_files", header, ALL_TYPES) as stage:
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
                if field in ("current_event_id", "summary_id", "exposure_value"):
                    continue
                assert np.array_equal(state[field], fresh[field]), field
            previous_event_id = event_id


def test_start_chunk_resets_a_chunk_abandoned_mid_event(tmp_path):
    run_dir = ASSETS_DIR / "all_files"
    data = SUMMARY_STREAM.read_bytes()
    header, records = split_summary_stream(data)
    events = group_events(records)
    rest = [to_block(ev) for _, ev in events[3:]]

    with make_stage(run_dir, header, ALL_TYPES) as stage:
        [expected] = run_stage(stage, [rest], tmp_path, "fresh", "bin")

    with make_stage(run_dir, header, ALL_TYPES) as stage:
        truncated = records[0][1][:12 + 8 * 5]
        stage.start_chunk(out_paths(tmp_path, "aborted", ALL_TYPES, "bin"))
        stage.process_block(to_block(truncated))
        assert stage.reader.state["reading_losses"][0]
        stage.finish_chunk()
        [actual] = run_stage(stage, [rest], tmp_path, "after", "bin")
    assert actual == expected


def test_header_validation():
    run_dir = ASSETS_DIR / "all_files"
    header = SUMMARY_STREAM.read_bytes()[:SUMMARY_HEADER_SIZE]
    with make_stage(run_dir, header[:8], ("splt",)) as stage:
        assert stage.len_sample == 100 and stage.summary_set_id is None
    with pytest.raises(ValueError):
        make_stage(run_dir, header[:10], ("splt",))
    gul_header = stream_info_to_bytes(GUL_STREAM_ID, 1) + header[4:]
    with pytest.raises(Exception, match="unsupported stream type"):
        make_stage(run_dir, gul_header, ("splt",))
