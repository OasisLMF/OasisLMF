"""Losses of an event must not depend on the events processed before it in the same gulmc run.

gulmc keeps per-event state in arrays sized by the portfolio (coverage item counters, group -> rng
index maps, CDF cache tags) and resets only the entries an event used. This checks that running
consecutive events that touch different items gives, for each event, the same bytes as running
that event alone.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.gulmc.manager import run as run_gulmc

TEST_MODEL = Path(__file__).parent.parent.parent.joinpath("assets", "test_model_1")
HEADER_SIZE = 8  # gul stream header: stream type (int32) + sample size (int32)

# footprint: (event_id, areaperil_id, intensity_bin_id, probability); events 1 and 2 touch disjoint
# areaperils, event 3 a subset of event 1, event 4 a mix; several areaperils carry intensity uncertainty
FOOTPRINT = [
    (1, 1, 3, 0.6), (1, 1, 7, 0.4), (1, 2, 12, 1.0),
    (2, 3, 5, 1.0), (2, 4, 9, 0.5), (2, 4, 20, 0.5),
    (3, 2, 15, 1.0),
    (4, 1, 25, 1.0), (4, 3, 2, 0.3), (4, 3, 18, 0.7), (4, 4, 11, 1.0),
]
N_COVERAGES = 8


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("event_sequence")
    run_dir.joinpath("input").mkdir()
    static = run_dir.joinpath("static")
    static.mkdir()
    for f in ("vulnerability.csv", "damage_bin_dict.csv", "intensity_bin_dict.csv"):
        shutil.copy(TEST_MODEL.joinpath("static", f), static.joinpath(f))
    pd.DataFrame(FOOTPRINT, columns=["event_id", "areaperil_id", "intensity_bin_id", "probability"]).to_csv(
        static.joinpath("footprint.csv"), index=False)
    # two items per coverage, on two different areaperils, with two vulnerability functions
    cov = np.repeat(np.arange(1, N_COVERAGES + 1), 2)
    ap = np.where(np.arange(len(cov)) % 2 == 0, (cov - 1) % 4 + 1, cov % 4 + 1)
    items = pd.DataFrame({"item_id": np.arange(1, len(cov) + 1), "coverage_id": cov, "areaperil_id": ap,
                          "vulnerability_id": np.where(np.arange(len(cov)) % 2 == 0, 2, 3), "group_id": cov * 7})
    items.to_csv(run_dir.joinpath("input", "items.csv"), index=False)
    pd.DataFrame({"coverage_id": np.arange(1, N_COVERAGES + 1), "tiv": 1000.0 * np.arange(1, N_COVERAGES + 1)}).to_csv(
        run_dir.joinpath("input", "coverages.csv"), index=False)
    pd.DataFrame({"item_id": items["item_id"], "peril_correlation_group": 1, "damage_correlation_value": 0.4,
                  "hazard_group_id": items["coverage_id"], "hazard_correlation_value": 0.3, "source_item_id": 0}).to_csv(
        run_dir.joinpath("input", "correlations.csv"), index=False)
    fp = pd.DataFrame(FOOTPRINT, columns=["event_id", "areaperil_id", "intensity_bin_id", "probability"])
    touched = {e: frozenset(items.loc[items["areaperil_id"].isin(g["areaperil_id"]), "item_id"]) for e, g in fp.groupby("event_id")}
    events = sorted(touched)
    assert all(0 < len(touched[e]) < len(items) for e in events)
    assert all(touched[a] != touched[b] for a, b in zip(events[:-1], events[1:]))
    return run_dir, events


def _run(run_dir, events, sample_size, ignore_correlation, effective_damageability, alloc_rule):
    events_bin = run_dir.joinpath("events_seq.bin")
    np.asarray(events, dtype="<i4").tofile(events_bin)
    out = run_dir.joinpath("gul_seq.bin")
    run_gulmc(run_dir=run_dir, ignore_file_type=set(), file_in=events_bin, file_out=out,
              sample_size=sample_size, loss_threshold=0., alloc_rule=alloc_rule, debug=0, random_generator=1,
              ignore_correlation=ignore_correlation, effective_damageability=effective_damageability)
    return out.read_bytes()


@pytest.mark.parametrize("alloc_rule", [1, 2])
@pytest.mark.parametrize("effective_damageability", [False, True])
@pytest.mark.parametrize("ignore_correlation", [False, True])
@pytest.mark.parametrize("sample_size", [0, 10])
def test_event_losses_independent_of_previous_events(model, sample_size, ignore_correlation, effective_damageability,
                                                     alloc_rule):
    run_dir, events = model
    args = (sample_size, ignore_correlation, effective_damageability, alloc_rule)
    alone = {e: _run(run_dir, [e], *args) for e in events}
    assert all(len(alone[e]) > HEADER_SIZE for e in events)
    header = alone[events[0]][:HEADER_SIZE]
    for sequence in (events, events[::-1], events + events, [1, 3, 1, 2, 4, 2]):
        expected = header + b"".join(alone[e][HEADER_SIZE:] for e in sequence)
        assert _run(run_dir, sequence, *args) == expected
