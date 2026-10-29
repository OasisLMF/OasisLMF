"""How a location's buildings are encoded must not change any loss.

FM validation cases are rerun with every location given several buildings, under each
disaggregation mode. ``test_fmpy.py`` checks the packed encoding on one purpose-built unit; this
reaches further because each defect packing has had so far sat where it meets another feature -- a
%TIV step policy, a tree with no location-level terms, per-coverage policy terms -- and the
validation suites are where those features already have cases.

``IsAggregate`` decides what the buildings are:

* 0 -- one risk. The buildings are summed before any term applies, so ``none``, ``items`` and
  ``samples`` must all give the original expected ``loc_summary.csv``, which was made with one
  building per location.
* 1 -- each building is its own risk and the site terms are split across them. No expected file
  describes that, so ``samples`` is checked against ``items``, at item level.

Running every suite this way costs about 2.5 times ``test_fmpy.py``, so it runs a selection
instead: the most involved units of each insurance suite plus a simple one, merged into one
portfolio, and two small suites whole -- business interruption, which has loss factors of its own,
and a reinsurance one, which cannot be cut down without changing its expected net losses.

Allocation rule 2, the default, throughout, and rules 0 and 1 on the insurance case, at account
level: rule 0 reports one loss per policy layer, and the expected files were made under rule 2,
whose account totals the other rules must reproduce. That is where they risk going wrong -- a
level linked past (``root_start``, which the per-coverage policy units produce) read the items'
loss from before the terms below, and under rule 0 the items reached it still packed.

Allocation rule 3 is not run: fmpy runs it as rule 2.

The one-location cases give a location many more buildings than the structure has nodes, which
is what makes the arena's packed allowance the part that matters. Run with ``NUMBA_BOUNDSCHECK=1``
(CI does): an arena overrun is otherwise a segfault, or worse a silent wrong answer, rather than a
failure.
"""
import os
import shutil
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from oasislmf.computation.run.exposure import RunExposure
from oasislmf.utils.defaults import DISAGGREGATION_ITEMS, DISAGGREGATION_NONE, DISAGGREGATION_SAMPLES

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'validation')

# Cycled over the locations rather than one count for all: the arena budget is worked out per
# node from the counts beneath it, so only a mix of counts tests that it adds up.
BUILDING_COUNTS = [1, 2, 5, 8]
PERILS = ['WTC']

# The units merged into the 'insurance' case, by FlexiLocUnit or, where a suite has none, by
# AccNumber. Each brings its whole accounts, so account, condition and policy terms see the same
# locations they do in the full suite.
SELECTED_UNITS = {
    'insurance_and_step': [
        'fm58',   # %TIV step triggers with a payout limit
        'fm64',   # step policy over location terms
        'fm12',   # seven policies on one account, min/max deductibles
        'Q4_14',  # two policies on one account
        'fm6',    # a plain one
    ],
    'insurance_conditions': [
        'sc14',  # conditions under two policies, min/max deductibles
        'sc3',   # four policies
        'fm41',  # conditions over six locations
        'sc25',  # one location under several conditions, by priority
        'sc18',  # a plain one
    ],
    'insurance_policy_coverage': ['Q5_22', 'Q5_14', 'Q5_7'],  # per-coverage policy terms
    'issues': [
        '828',   # two policies
        '1824',  # already several buildings with IsAggregate=1
    ],
    'issue_1816': ['1'],
    'issue_1953_layerparticipation': ['1'],  # layer participation
}
WHOLE_SUITES = ['insurance_bi', 'reinsurance3']
CASES = ['insurance'] + WHOLE_SUITES

INPUT_FILES = ('account.csv', 'ri_info.csv', 'ri_scope.csv', 'loss_factors.csv')
LOSS_COLS = ['loss_gul', 'loss_il', 'loss_ri']

# Partial losses as well as a total one: at 100% every %TIV trigger fires whatever the TIV, so a
# step policy measuring 1/N of it would pass unnoticed. Kept off round fractions, which is where the
# step suites put their triggers: a loss landing exactly on one fires or not depending on how float32
# rounds TIV/N, so the modes can disagree there by a whole payout without either being wrong.
DEFAULT_LOSS_FACTORS = [0.07, 0.23, 0.53, 1.0]

# A loss is a float32 sum of N per-building losses under ``items`` and one loss under ``samples``
# or ``none``, so they differ in the last bits -- 0.35 against 0.33 on a near-zero location, which
# a relative tolerance alone rejects. Real defects so far are thousands or more.
ABS_TOL = 1.0
REL_TOL = 1e-4


def _read(suite, fn):
    return pd.read_csv(os.path.join(VALIDATION_DIR, suite, fn), dtype=str, keep_default_na=False)


def _selected_insurance():
    """The selected units' locations, accounts and expected location losses, as one portfolio.

    PortNumber is prefixed with the suite, since the suites reuse port and account numbers.
    """
    locs, accs, expected = [], [], []
    for suite, units in SELECTED_UNITS.items():
        loc, acc, exp = _read(suite, 'location.csv'), _read(suite, 'account.csv'), _read(suite, 'expected/loc_summary.csv')
        unit_col = 'FlexiLocUnit' if 'FlexiLocUnit' in loc.columns else 'AccNumber'
        accounts = set(zip(*loc.loc[loc[unit_col].isin(units), ['PortNumber', 'AccNumber']].T.values))
        for df, out in ((loc, locs), (acc, accs), (exp, expected)):
            df = df[[key in accounts for key in zip(df['PortNumber'], df['AccNumber'])]].copy()
            df['PortNumber'] = suite + '/' + df['PortNumber']
            out.append(df)
    return pd.concat(locs).fillna(''), pd.concat(accs).fillna(''), pd.concat(expected)


def _write_case(case, dest, set_buildings):
    """Write ``case``'s exposure to ``dest`` with its locations' buildings set by ``set_buildings``.

    Returns the expected location losses and the loss factors they were made at.
    """
    os.makedirs(dest)
    if case == 'insurance':
        loc, acc, expected = _selected_insurance()
        acc.to_csv(os.path.join(dest, 'account.csv'), index=False)
    else:
        loc, expected = _read(case, 'location.csv'), _read(case, 'expected/loc_summary.csv')
        for fn in INPUT_FILES:
            if os.path.exists(os.path.join(VALIDATION_DIR, case, fn)):
                shutil.copy(os.path.join(VALIDATION_DIR, case, fn), dest)
    if 'IsAggregate' not in loc.columns:
        loc['IsAggregate'] = '0'
    set_buildings(loc)
    loc.to_csv(os.path.join(dest, 'location.csv'), index=False)

    fp = os.path.join(dest, 'loss_factors.csv')
    loss_factor = pd.read_csv(fp)['loss_factor'].astype(float).tolist() if os.path.exists(fp) else None
    return _summed(expected), loss_factor


def _losses(src_dir, run_dir, disaggregation, output_level, loss_factor, alloc_rule_il=2):
    """Run the deterministic exposure step and return its losses at ``output_level``."""
    out = os.path.join(run_dir, 'summary.csv')
    RunExposure(
        src_dir=src_dir,
        run_dir=run_dir,
        loss_factor=loss_factor,
        include_loss_factor=len(loss_factor) > 1,
        output_level=output_level,
        output_file=out,
        model_perils_covered=PERILS,
        keys_format='oasis',
        disaggregation=disaggregation,
        kernel_alloc_rule_il=alloc_rule_il,
        print_summary=False,
        portfolio_complexity=False,
    ).run()
    return _summed(pd.read_csv(out, dtype=str, keep_default_na=False))


def _summed(df, drop=()):
    """Sum the losses over everything but the output id and ``drop``.

    At item level ``items`` reports a row per building where ``samples`` reports one per coverage.
    """
    loss_cols = [c for c in LOSS_COLS if c in df.columns]
    keys = [c for c in df.columns if c not in loss_cols and c != 'output_id' and c not in drop]
    df[loss_cols] = df[loss_cols].astype(float)
    return df.groupby(keys, as_index=False)[loss_cols].sum()


def _assert_same_losses(ref, other, ref_name, other_name):
    # Only the losses both report: the merged insurance case carries an expected loss_ri from a
    # suite whose reinsurance it does not run.
    loss_cols = [c for c in LOSS_COLS if c in ref.columns and c in other.columns]
    ref = ref.drop(columns=[c for c in LOSS_COLS if c in ref.columns and c not in loss_cols])
    keys = [c for c in ref.columns if c not in loss_cols]
    merged = ref.merge(other, on=keys, how='outer', suffixes=('_ref', '_other'), indicator=True)
    assert (merged['_merge'] == 'both').all(), (
        f"rows reported by only one of {ref_name}/{other_name}:\n"
        f"{merged[merged['_merge'] != 'both'][keys + ['_merge']]}")

    for col in loss_cols:
        a, b = merged[f'{col}_ref'].fillna(0), merged[f'{col}_other'].fillna(0)
        bad = ~((a - b).abs() <= ABS_TOL + REL_TOL * a.abs())
        assert not bad.any(), (
            f"{col}: {int(bad.sum())} of {len(merged)} rows differ; "
            f"total {ref_name}={a.sum():,.2f} {other_name}={b.sum():,.2f}\n"
            f"{merged.loc[bad, keys + [f'{col}_ref', f'{col}_other']].head(10).to_string()}"
        )


def _building_counts(n):
    return [str(BUILDING_COUNTS[i % len(BUILDING_COUNTS)]) for i in range(n)]


def _one_risk(loc):
    # Locations already marked IsAggregate=1 are left alone: their expected losses split the site
    # terms across buildings, which is not what this case is about.
    single_risk = loc['IsAggregate'] != '1'
    loc.loc[single_risk, 'NumberOfBuildings'] = _building_counts(int(single_risk.sum()))
    loc.loc[single_risk, 'IsAggregate'] = '0'


def _separate_risks(loc):
    loc['NumberOfBuildings'] = _building_counts(len(loc))
    loc['IsAggregate'] = '1'


@pytest.mark.parametrize('mode', [DISAGGREGATION_NONE, DISAGGREGATION_ITEMS, DISAGGREGATION_SAMPLES])
@pytest.mark.parametrize('case', CASES)
def test_buildings_of_one_risk_give_the_original_location_losses(case, mode):
    with TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'src')
        expected, loss_factor = _write_case(case, src, _one_risk)
        # the loss factors the expected file was made at
        losses = _losses(src, os.path.join(tmp, 'run'), mode, 'loc', loss_factor or [1.0])

    _assert_same_losses(expected, losses, 'expected', mode)


@pytest.mark.parametrize('mode', [DISAGGREGATION_NONE, DISAGGREGATION_ITEMS, DISAGGREGATION_SAMPLES])
@pytest.mark.parametrize('alloc_rule', [0, 1])
def test_buildings_of_one_risk_give_the_original_account_losses_under_rules_0_and_1(alloc_rule, mode):
    with TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'src')
        expected, loss_factor = _write_case('insurance', src, _one_risk)
        losses = _losses(src, os.path.join(tmp, 'run'), mode, 'acc', loss_factor or [1.0], alloc_rule)

    _assert_same_losses(_summed(expected, drop=['LocNumber']), losses, 'expected', f'{mode} -a{alloc_rule}')


@pytest.mark.parametrize('case', CASES)
def test_separate_buildings_give_the_same_item_losses_packed(case):
    with TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'src')
        _, loss_factor = _write_case(case, src, _separate_risks)
        items, samples = (
            _losses(src, os.path.join(tmp, mode), mode, 'item', loss_factor or DEFAULT_LOSS_FACTORS)
            for mode in (DISAGGREGATION_ITEMS, DISAGGREGATION_SAMPLES)
        )

    _assert_same_losses(items, samples, DISAGGREGATION_ITEMS, DISAGGREGATION_SAMPLES)


def test_separate_buildings_give_the_same_account_losses_packed_under_rules_0_and_1():
    # Rule 0 alone used to leave a location's items packed, which a level linked past then read
    # building by building; rule 1 always merged them.
    with TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'src')
        _write_case('insurance', src, _separate_risks)
        rule_0, rule_1 = (
            _losses(src, os.path.join(tmp, f'a{rule}'), DISAGGREGATION_SAMPLES, 'acc', DEFAULT_LOSS_FACTORS, alloc_rule_il=rule)
            for rule in (0, 1)
        )

    _assert_same_losses(rule_1, rule_0, '-a1', '-a0')


EVERY_COVERAGE = {'BuildingTIV': '5000000', 'OtherTIV': '300000', 'ContentsTIV': '500000', 'BITIV': '200000'}

# One location with many more buildings than the structure has nodes, which is what makes the
# arena's packed allowance the part that matters: an under-reserved one runs off the end of an
# array, which NUMBA_BOUNDSCHECK turns into a failure.
ONE_LOCATION = {
    # Under a policy deductible only, no level applies terms per building and fm sums a
    # location's buildings as it reads them.
    'policy_terms/building_and_contents': ({'BuildingTIV': '5000000', 'ContentsTIV': '500000'}, {}, 50),
    # one item, so nothing above it merges anything
    'policy_terms/building_only': ({'BuildingTIV': '5000000'}, {}, 50),
    'policy_terms/every_coverage': (EVERY_COVERAGE, {}, 10),
    # Under location terms the buildings stay apart up to the location level. Every coverage
    # gives a multi-peril structure, whose item nodes sit below the levels generation counts.
    'location_terms/deductible': (EVERY_COVERAGE, {'LocPeril': 'WTC', 'LocDed6All': '20000', 'LocDedType6All': '0'}, 50),
    # a minimum deductible needs the extras arena, which has a budget of its own
    'location_terms/min_deductible': (EVERY_COVERAGE, {'LocPeril': 'WTC', 'LocDed6All': '20000', 'LocDedType6All': '0',
                                                       'LocMinDed6All': '10000'}, 50),
}


@pytest.mark.parametrize('alloc_rule', [0, 1, 2])
@pytest.mark.parametrize('layout', ONE_LOCATION, ids=str)
def test_one_location_of_many_separate_buildings_gives_the_same_item_losses_packed(layout, alloc_rule):
    tivs, loc_terms, n_buildings = ONE_LOCATION[layout]
    location = pd.DataFrame({
        'PortNumber': ['1'], 'AccNumber': ['1'], 'LocNumber': ['1'], 'CountryCode': ['US'],
        'LocPerilsCovered': ['WTC'], 'BuildingTIV': ['0'], 'OtherTIV': ['0'],
        'ContentsTIV': ['0'], 'BITIV': ['0'], 'LocCurrency': ['USD'],
        'NumberOfBuildings': [str(n_buildings)], 'IsAggregate': ['1'],
    })
    for col, value in {**tivs, **loc_terms}.items():
        location[col] = value
    account = pd.DataFrame({
        'PortNumber': ['1'], 'AccNumber': ['1'], 'PolNumber': ['1'], 'PolPerilsCovered': ['WTC'],
        'PolPeril': ['WTC'], 'PolDed6All': ['100000'], 'PolDedType6All': ['0'], 'AccCurrency': ['USD'],
    })
    with TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'src')
        os.makedirs(src)
        location.to_csv(os.path.join(src, 'location.csv'), index=False)
        account.to_csv(os.path.join(src, 'account.csv'), index=False)
        items, samples = (
            _losses(src, os.path.join(tmp, mode), mode, 'item', DEFAULT_LOSS_FACTORS, alloc_rule)
            for mode in (DISAGGREGATION_ITEMS, DISAGGREGATION_SAMPLES)
        )

    _assert_same_losses(items, samples, DISAGGREGATION_ITEMS, f'{DISAGGREGATION_SAMPLES} -a{alloc_rule}')
