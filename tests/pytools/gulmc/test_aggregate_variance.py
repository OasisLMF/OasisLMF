"""A packed location's aggregate must keep its sample-to-sample variance.

A location of N buildings is the sum of N draws, so its total varies between samples with a
coefficient of variation falling as 1/sqrt(N) -- falling, never vanishing, and never jumping at
a particular N.

This exists because a building pool did exactly that. Buildings read a shared pool of M values
through a rotation, and a rotation is a bijection: at any sample the N buildings covered the
same M entries the same number of times, so a location with N a multiple of M produced a
*constant* total, identical in every sample. The per-building marginals were fine and every
test in the suite passed; only the aggregate gave it away.
"""
import functools
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.gulmc.manager import run as run_gulmc
from tests.pytools.utils import copy_test_model, read_gul_stream, set_coverage_buildings, write_correlations

# Statistical, not functional: the assertions need up to 20,000 buildings, which is seconds once compiled and
# hours interpreted. The four JIT-enabled CI legs run them; the coverage leg, which sets
# NUMBA_DISABLE_JIT, does not.
pytestmark = pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT", "0") != "0",
    reason="statistical test: up to 20,000 buildings is hours with the JIT disabled",
)


@functools.lru_cache(maxsize=None)
def _location_totals(n_buildings, sample_size):
    """Per-sample total for one summed item carrying n_buildings, correlation off.

    Cached: the tests below ask for the same runs.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = copy_test_model(Path(tmp) / 'assets')
        items = pd.read_csv(run_dir / 'input' / 'items.csv')
        write_correlations(run_dir / 'input', items['item_id'].to_numpy(), 0.)
        # positive: summed, so the item IS the total
        set_coverage_buildings(run_dir / 'input', n_buildings)
        out = run_dir / 'out.bin'
        run_gulmc(run_dir=run_dir, ignore_file_type=set(),
                  file_in=run_dir / 'input' / 'events.bin', file_out=out,
                  sample_size=sample_size, loss_threshold=-1e30, alloc_rule=0, debug=0,
                  random_generator=2, ignore_correlation=True, effective_damageability=True)
        df = read_gul_stream(out)[2]

    item_1 = df[(df['item_id'] == 1) & (df['sidx'] > 0)]
    first_event = item_1[item_1['event_id'] == item_1['event_id'].min()]
    return first_event.sort_values('sidx')['loss'].to_numpy()


# 8192 was the old pooling gate and 16384 a multiple of the old pool size -- the two counts at
# which the total went exactly constant. 8191 is the control just below.
BUILDING_COUNTS = [8191, 8192, 16384, 20000]


@pytest.mark.parametrize("sample_size", [10, 100])
@pytest.mark.parametrize("n_buildings", BUILDING_COUNTS)
def test_the_total_is_not_the_same_in_every_sample(n_buildings, sample_size):
    """The failure mode, stated directly: a location must not be a constant."""
    totals = _location_totals(n_buildings, sample_size)
    assert len(np.unique(totals)) > 1, (
        f"N={n_buildings}, S={sample_size}: the location total is identical in every sample "
        f"({totals[0]:,.0f}) -- its buildings are not being drawn independently")
    assert totals.std(ddof=1) / totals.mean() > 1e-3, (
        f"N={n_buildings}, S={sample_size}: the total barely varies between samples")


@pytest.mark.parametrize("sample_size", [10, 100])
def test_the_spread_falls_as_one_over_root_n(sample_size):
    """A sum of N independent draws has CoV proportional to 1/sqrt(N).

    Checked as a ratio so it does not depend on the model's own loss distribution: pooling
    showed up here as CoV*sqrt(N) collapsing toward zero as N grew.
    """
    scaled = {}
    for n in BUILDING_COUNTS:
        totals = _location_totals(n, sample_size)
        scaled[n] = (totals.std(ddof=1) / totals.mean()) * np.sqrt(n)
    lo, hi = min(scaled.values()), max(scaled.values())
    assert hi / lo < 2.0, (
        f"S={sample_size}: CoV*sqrt(N) should be roughly constant across building counts, got "
        + ", ".join(f"N={n}: {v:.3f}" for n, v in scaled.items()))


def test_a_single_sample_still_produces_a_loss():
    """S=1 has no spread to check, but it must still run and produce a positive total."""
    for n in (8192, 20000):
        totals = _location_totals(n, 1)
        assert totals.shape == (1,)
        assert totals[0] > 0
