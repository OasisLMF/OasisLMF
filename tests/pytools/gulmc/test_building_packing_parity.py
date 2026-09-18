"""End-to-end parity test for gulmc building-packing.

Building-packing multiplexes N buildings per item into the sample dimension of a single
stream item (see ``write_losses`` / ``encode_sidx``). Every random generator gives
building 1 the unpacked draw byte-for-byte, so with alloc_rule=0 building 1 of a packed run
must reproduce a legacy run value-for-value on all three; the remaining buildings draw from
their own stream coordinate and must be present and genuinely distinct.

Run for each generator, including 2 (Latin Hypercube on Philox), which is the default.

The test runs gulmc twice on a real copy of test_model_1: once with no side file (legacy),
once with a per-item building count on correlations assigning two buildings to every item.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from oasislmf.pytools.gulmc.manager import run as run_gulmc
from oasislmf.pytools.common.event_stream import decode_building, decode_local_sidx
from oasislmf.pytools.common.data import oasis_float, correlations_dtype
from oasislmf.pytools.common.input_files import read_correlations

TESTS_DIR = Path(__file__).parent.parent.parent
SRC_MODEL = TESTS_DIR.joinpath("assets", "test_model_1")
SAMPLE_SIZE = 100


def _parse_stream(path):
    """Parse a gul binary stream into {(event_id, item_id): [(sidx, loss), ...]}."""
    raw = Path(path).read_bytes()
    pos = 8  # skip 4-byte magic header + 4-byte sample_size
    out = {}
    while pos < len(raw):
        event_id, item_id = np.frombuffer(raw[pos:pos + 8], dtype='<i4')
        pos += 8
        recs = []
        while True:
            sidx = np.frombuffer(raw[pos:pos + 4], dtype='<i4')[0]
            loss = np.frombuffer(raw[pos + 4:pos + 8], dtype=oasis_float)[0]
            pos += 8
            if sidx == 0:
                break
            recs.append((int(sidx), float(loss)))
        out[(int(event_id), int(item_id))] = recs
    return out


def _run(run_dir, file_out, random_generator):
    run_gulmc(
        run_dir=str(run_dir),
        ignore_file_type=set(),
        file_in=str(Path(run_dir) / "input" / "events.bin"),
        file_out=str(file_out),
        sample_size=SAMPLE_SIZE,
        loss_threshold=0.,
        alloc_rule=0,  # building-packing requires alloc_rule == 0
        debug=0,
        random_generator=random_generator,
        ignore_correlation=False,
        effective_damageability=False,
    )


def _fresh_copy(dst):
    """Copy the model and drop any stale cached structures so the side file is read."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC_MODEL, dst, symlinks=False)
    cache = dst / "input" / "gulmc_structure"
    if cache.exists():
        shutil.rmtree(cache)


@pytest.mark.skipif(not SRC_MODEL.exists(), reason="test_model_1 assets not available")
@pytest.mark.parametrize("random_generator", [0, 1, 2], ids=["mersenne", "latin_hypercube", "lh_philox"])
def test_building_packing_building1_matches_legacy(tmp_path, random_generator):
    """Building 1 of an N=2 packed run reproduces the legacy run; building 2 is present and distinct."""
    # legacy run (one building per item)
    legacy_dir = tmp_path / "legacy"
    _fresh_copy(legacy_dir)
    legacy_out = legacy_dir / "legacy.bin"
    _run(legacy_dir, legacy_out, random_generator)
    legacy = _parse_stream(legacy_out)
    assert legacy, "legacy run produced no output"

    # packed run with two buildings for every item, kept separate. The correlations table (1:1
    # with items, joined into the items array downstream) carries this as one signed field:
    # magnitude 2 buildings, negative sign meaning "emit them as separate blocks". Positive would
    # make gulmc sum the buildings at source instead.
    packed_dir = tmp_path / "packed"
    _fresh_copy(packed_dir)
    corr = np.array(read_correlations(packed_dir / "input"), dtype=correlations_dtype)
    corr['packed_buildings'] = -2
    corr.tofile(packed_dir / "input" / "correlations.bin")
    packed_out = packed_dir / "packed.bin"
    _run(packed_dir, packed_out, random_generator)
    packed = _parse_stream(packed_out)

    assert set(packed.keys()) == set(legacy.keys())

    distinct_positive = 0
    for key, lrecs in legacy.items():
        ldict = dict(lrecs)
        b1, b2 = {}, {}
        for sidx, loss in packed[key]:
            b = decode_building(sidx, SAMPLE_SIZE)
            local = decode_local_sidx(sidx, SAMPLE_SIZE)
            (b1 if b == 1 else b2)[local] = loss

        # building 1 must reproduce the legacy stream exactly (same sidx set and values)
        assert set(b1.keys()) == set(ldict.keys()), f"sidx set mismatch for {key}"
        for sidx, lval in ldict.items():
            assert b1[sidx] == lval, f"building-1 value mismatch for {key} sidx={sidx}"

        # building 2 must be present with a full special + sample set
        assert b2, f"building 2 missing for {key}"
        for sidx in ldict:
            if sidx > 0 and ldict[sidx] != b2.get(sidx, 0.0):
                distinct_positive += 1

    # the second building draws the continuation of the seed sequence, so across the whole
    # run its samples must not be a verbatim copy of building 1 (which would signal a slice bug)
    assert distinct_positive > 0, "building 2 samples are identical to building 1 everywhere"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
