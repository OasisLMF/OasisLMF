"""The analytic specials a summed packed item reports must match the samples it emits.

A positive ``n_building`` on the coverage is summed at source, so the item's five special sidx describe the
distribution of that SUM. ``mean``, ``tiv`` and ``max_loss`` are additive and scale by N;
``chance_of_loss`` is a probability and is taken once. ``std_dev`` is the one that needs the
correlation: the buildings of an item share ``damage_eps_ij[peril_correlation_group]``, so their
variances do not simply add.
"""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.data import correlations_dtype
from oasislmf.pytools.common.event_stream import MEAN_IDX, STD_DEV_IDX
from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.gulmc.manager import run as run_gulmc
from tests.pytools.utils import set_coverage_buildings

# Statistical, not functional: the assertions need 2,000 samples per run, which is seconds once compiled and
# hours interpreted. The four JIT-enabled CI legs run them; the coverage leg, which sets
# NUMBA_DISABLE_JIT, does not.
pytestmark = pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT", "0") != "0",
    reason="statistical test: 2,000 samples per run is hours with the JIT disabled",
)

SRC_MODEL = Path(__file__).parents[2].joinpath("assets", "test_model_1")
SAMPLE_SIZE = 2000


def _run(n_buildings, rho):
    """Run one packed, summed item and return (reported std_dev, empirical std, reported mean)."""
    with tempfile.TemporaryDirectory() as t:
        run_dir = Path(t) / 'assets'
        shutil.copytree(SRC_MODEL, run_dir)
        shutil.rmtree(run_dir / 'input' / 'gulmc_structure', ignore_errors=True)

        items = pd.read_csv(run_dir / 'input' / 'items.csv')
        corr = np.zeros(len(items), dtype=correlations_dtype)
        corr['item_id'] = items['item_id'].to_numpy()
        corr['peril_correlation_group'] = 1
        corr['damage_correlation_value'] = rho
        corr['hazard_group_id'] = 1
        corr.tofile(run_dir / 'input' / 'correlations.bin')
        pd.DataFrame({k: corr[k] for k in corr.dtype.names}).to_csv(
            run_dir / 'input' / 'correlations.csv', index=False)
        # positive: summed at source. The count lives on the coverage.
        set_coverage_buildings(run_dir / 'input', n_buildings,
                               item_to_coverage=items['coverage_id'].to_numpy())

        run_gulmc(run_dir=run_dir, ignore_file_type=set(),
                  file_in=run_dir / 'input' / 'events.bin', file_out=run_dir / 'o.bin',
                  sample_size=SAMPLE_SIZE, loss_threshold=-1e30, alloc_rule=0, debug=0,
                  random_generator=2, ignore_correlation=False, effective_damageability=False)
        bintocsv(run_dir / 'o.bin', run_dir / 'o.csv', 'gul')
        df = pd.read_csv(run_dir / 'o.csv')

    event = df['event_id'].iloc[0]
    d = df[(df['event_id'] == event) & (df['item_id'] == 1)]
    return (float(d[d['sidx'] == STD_DEV_IDX]['loss'].iloc[0]),
            float(d[d['sidx'] > 0]['loss'].std(ddof=0)),
            float(d[d['sidx'] == MEAN_IDX]['loss'].iloc[0]))


def test_uncorrelated_buildings_still_scale_by_root_n():
    """With no correlation the buildings really are independent, so the variances add and the old
    sqrt(N) is right. This pins that the correlation term does not disturb the uncorrelated case."""
    one_std, _, one_mean = _run(1, 0.0)
    for n in (16, 64):
        std, empirical, mean = _run(n, 0.0)
        assert std == pytest.approx(one_std * np.sqrt(n), rel=1e-5), f"N={n}: not sqrt(N) scaling"
        assert mean == pytest.approx(one_mean * n, rel=1e-5), f"N={n}: mean must stay additive"
        assert std == pytest.approx(empirical, rel=0.15), f"N={n}: disagrees with its own samples"


@pytest.mark.parametrize("rho", [0.3, 0.7, 0.9])
def test_correlated_buildings_combine_variances(rho):
    """The buildings share the correlation group's common factor, so var(sum) is
    sigma^2 * (N + N(N-1)*r), not N*sigma^2. sqrt(N) alone understates sigma by about
    sqrt(N*r) -- a factor of 5.6 at N=64, r=0.7 -- and the error grows with the count.

    ``r`` is the correlation between two buildings' LOSSES, which the damage curve attenuates
    away from the copula value: substituting the copula value overstated sigma by up to ~39%.
    This checks the property rather than the formula -- the reported sigma has to agree with the
    spread of the samples the engine actually emitted -- and that it beats the copula value at
    doing so, which is the specific thing that changed.
    """
    one_std, _, _ = _run(1, rho)
    for n in (16, 64):
        std, empirical, _ = _run(n, rho)

        naive = one_std * np.sqrt(n)
        copula = one_std * np.sqrt(n + n * (n - 1) * rho)
        assert naive < std < copula, (
            f"N={n}, rho={rho}: the loss correlation must sit strictly between no correlation "
            f"and the copula value, which it can never exceed")

        assert std == pytest.approx(empirical, rel=0.08), (
            f"N={n}, rho={rho}: reported sigma {std:,.0f} disagrees with the samples' "
            f"{empirical:,.0f}")
        assert abs(std - empirical) < abs(copula - empirical), (
            f"N={n}, rho={rho}: no better than substituting the copula correlation")
