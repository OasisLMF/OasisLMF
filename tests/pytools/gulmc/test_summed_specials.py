"""The analytic specials a summed packed item reports must match the samples it emits.

A positive ``packed_buildings`` is summed at source, so the item's five special sidx describe the
distribution of that SUM. ``mean``, ``tiv`` and ``max_loss`` are additive and scale by N;
``chance_of_loss`` is a probability and is taken once. ``std_dev`` is the one that needs the
correlation: the buildings of an item share ``damage_eps_ij[peril_correlation_group]``, so their
variances do not simply add.
"""
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
        corr['packed_buildings'] = n_buildings          # positive: summed at source
        corr['peril_correlation_group'] = 1
        corr['damage_correlation_value'] = rho
        corr['hazard_group_id'] = 1
        corr.tofile(run_dir / 'input' / 'correlations.bin')
        pd.DataFrame({k: corr[k] for k in corr.dtype.names}).to_csv(
            run_dir / 'input' / 'correlations.csv', index=False)

        run_gulmc(run_dir=run_dir, ignore_file_type=set(),
                  file_in=run_dir / 'input' / 'events.bin', file_out=run_dir / 'o.bin',
                  sample_size=SAMPLE_SIZE, loss_threshold=-1e30, alloc_rule=0, debug=0,
                  random_generator=0, ignore_correlation=False, effective_damageability=False)
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
    sigma^2 * (N + N(N-1)*rho), not N*sigma^2. sqrt(N) alone understates sigma by about
    sqrt(N*rho) -- a factor of 5.6 at N=64, rho=0.7 -- and the error grows with the count.

    rho here is the copula correlation, while the sum needs the correlation it induces between
    two buildings' losses, which the marginal attenuates. So the reported value runs ~15-30%
    HIGH. The bound below is deliberately one-sided about how far low it may be: being under is
    the failure this test exists to catch.
    """
    one_std, _, _ = _run(1, rho)
    for n in (16, 64):
        std, empirical, _ = _run(n, rho)

        expected = one_std * np.sqrt(n + n * (n - 1) * rho)
        assert std == pytest.approx(expected, rel=1e-5), f"N={n}: not the combined-variance form"

        naive = one_std * np.sqrt(n)
        assert std > naive, f"N={n}: no better than the uncorrelated scaling"
        assert empirical / std > 0.6, f"N={n}: reported sigma wildly above the samples"
        assert empirical / std < 1.25, (
            f"N={n}, rho={rho}: reported sigma {std:,.0f} is far below the samples' "
            f"{empirical:,.0f} -- the correlation term is missing or wrong")
