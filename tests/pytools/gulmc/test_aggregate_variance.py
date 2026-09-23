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
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.data import correlations_dtype
from oasislmf.pytools.gulmc.manager import run as run_gulmc

SRC_MODEL = Path(__file__).parents[2].joinpath("assets", "test_model_1")


def _location_totals(n_buildings, sample_size):
    """Per-sample total for one summed item carrying n_buildings, correlation off."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / 'assets'
        shutil.copytree(SRC_MODEL, run_dir)
        shutil.rmtree(run_dir / 'input' / 'gulmc_structure', ignore_errors=True)
        items = pd.read_csv(run_dir / 'input' / 'items.csv')
        corr = np.zeros(len(items), dtype=correlations_dtype)
        corr['item_id'] = items['item_id'].to_numpy()
        corr['packed_buildings'] = n_buildings        # positive: summed, so the item IS the total
        corr['peril_correlation_group'] = 1
        corr['damage_correlation_value'] = 0.
        corr['hazard_group_id'] = 1
        corr.tofile(run_dir / 'input' / 'correlations.bin')
        pd.DataFrame({k: corr[k] for k in corr.dtype.names}).to_csv(
            run_dir / 'input' / 'correlations.csv', index=False)
        out = run_dir / 'out.bin'
        run_gulmc(run_dir=run_dir, ignore_file_type=set(),
                  file_in=run_dir / 'input' / 'events.bin', file_out=out,
                  sample_size=sample_size, loss_threshold=-1e30, alloc_rule=0, debug=0,
                  random_generator=2, ignore_correlation=True, effective_damageability=True)
        raw = np.fromfile(out, dtype=np.int32)

    rawf = raw.view(np.float32)
    i, totals = 2, {}
    while i + 1 < len(raw):
        key = (int(raw[i]), int(raw[i + 1]))
        i += 2
        rec = {}
        while i + 1 < len(raw):
            sidx = int(raw[i])
            i += 2
            if sidx == 0:
                break
            rec[sidx] = float(rawf[i - 1])
        totals[key] = rec
    first = sorted(k for k in totals if k[1] == 1)[0]
    return np.array([totals[first][s] for s in range(1, sample_size + 1)])


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


@pytest.mark.parametrize("sample_size", [10, 100])
def test_no_discontinuity_at_the_old_pooling_gate(sample_size):
    """One building more must not change how a location is sampled.

    The pool switched behaviour at a building count, so 8191 and 8192 buildings were drawn by
    different schemes and gave visibly different spreads.
    """
    below = _location_totals(8191, sample_size)
    above = _location_totals(8192, sample_size)
    cov_below = below.std(ddof=1) / below.mean()
    cov_above = above.std(ddof=1) / above.mean()
    assert cov_above == pytest.approx(cov_below, rel=0.25), (
        f"S={sample_size}: 8191 buildings gives CoV {cov_below:.4%} but 8192 gives "
        f"{cov_above:.4%} -- one building should not change the sampling scheme")


def test_a_single_sample_still_produces_a_loss():
    """S=1 has no spread to check, but it must still run and produce a positive total."""
    for n in (8192, 20000):
        totals = _location_totals(n, 1)
        assert totals.shape == (1,)
        assert totals[0] > 0
