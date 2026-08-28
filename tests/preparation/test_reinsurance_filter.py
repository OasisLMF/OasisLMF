# ---------------------------------------------------------------------------
# Reinsurance filter level scope matching
#
# match_filter_level_scope replaced a per row `filter_df.apply(_match, axis=1)`.
# The row wise version is kept here as the reference the vectorized one is pinned against.
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

from oasislmf.preparation.reinsurance_layer import (
    FILTER_LEVEL_EXTRA_FIELDS,
    RISK_LEVEL_ALL_FIELDS,
    RISK_LEVEL_FIELD_MAP,
    match_filter_level_scope,
)
from oasislmf.utils import oed
from oasislmf.utils.exceptions import OasisException

RI_FILTER_FIELDS = RISK_LEVEL_ALL_FIELDS + FILTER_LEVEL_EXTRA_FIELDS


def row_wise_match(filter_df, ri_filter_fields, merge_on):
    """The original row wise implementation, kept as the reference (only its guard clause reshaped)."""
    def _match(row):
        for field in ri_filter_fields:
            if field in merge_on:
                continue
            if row[f'{field}_valid'] and row[f'{field}_x'] != row[f'{field}_y']:
                return False

        # Risk Attaching filter for reinsurance
        if "AttachmentBasis" in row and row["AttachmentBasis"] == "RA":
            if row["ReinsInceptionDate"] == "" or row["ReinsExpiryDate"] == "":
                error_msg = "Error: ReinsInceptionDate/ReinsExpiryDate missing, cannot use AttachmentBasis [RA]. Please check the ri_info file"
                raise OasisException(error_msg)
            elif row["PolInceptionDate"] == "":
                acc_info = {
                    field: row[f'{field}_x'] if f'{field}_x' in row else row[f'{field}']
                    for field in RISK_LEVEL_FIELD_MAP[oed.REINS_RISK_LEVEL_ACCOUNT]
                    if f'{field}_x' in row or f'{field}' in row
                }
                error_msg = f"Error: PolInceptionDate missing for {acc_info}, cannot use AttachmentBasis [RA]. Please check the account file"
                raise OasisException(error_msg)
            else:
                if row["PolInceptionDate"] < row["ReinsInceptionDate"] or row["ReinsExpiryDate"] < row["PolInceptionDate"]:
                    return False

        return True

    return filter_df.apply(_match, axis=1).to_numpy(dtype=bool)


def make_filter_df(n, rng, attachment='LO', categorical=False, n_values=4):
    """Build a frame shaped like filter_df is after the profile map / ri scope merge."""
    data = {'index': np.arange(n)}
    for field in RI_FILTER_FIELDS:
        left = rng.integers(0, n_values, n).astype(str)
        right = rng.integers(0, n_values, n).astype(str)
        if categorical and field in ('CountryCode', 'LOB'):
            # deliberately different category sets, as two independently read OED files give
            data[f'{field}_x'] = pd.Categorical(left)
            data[f'{field}_y'] = pd.Categorical(right, categories=[str(i) for i in range(n_values + 2)])
        else:
            data[f'{field}_x'] = left
            data[f'{field}_y'] = right
        data[f'{field}_valid'] = rng.random(n) < 0.4  # a scope only specifies some of its fields

    basis = np.full(n, 'LO', dtype=object)
    if attachment == 'RA':
        basis[rng.random(n) < 0.5] = 'RA'
    data['AttachmentBasis'] = basis
    data['ReinsInceptionDate'] = np.full(n, '2024-01-01', dtype=object)
    data['ReinsExpiryDate'] = np.full(n, '2024-12-31', dtype=object)
    data['PolInceptionDate'] = rng.choice(
        ['2023-06-01', '2024-03-01', '2024-11-30', '2025-02-01'], n).astype(object)
    df = pd.DataFrame(data)
    if categorical:
        # OED date columns arrive as unordered categoricals, which cannot be order compared as a
        # Series even though their values order fine - this is what the real RI fixtures produce
        for field in ['AttachmentBasis', 'ReinsInceptionDate', 'ReinsExpiryDate', 'PolInceptionDate']:
            df[field] = df[field].astype('category')
    return df


def make_one_row(categorical=False, **overrides):
    """A single filter_df row that matches on every field unless overridden."""
    row = {'index': 0}
    for field in RI_FILTER_FIELDS:
        row[f'{field}_x'] = 'same'
        row[f'{field}_y'] = 'same'
        row[f'{field}_valid'] = True
    row['AttachmentBasis'] = 'LO'
    row['ReinsInceptionDate'] = '2024-01-01'
    row['ReinsExpiryDate'] = '2024-12-31'
    row['PolInceptionDate'] = '2024-06-01'
    row.update(overrides)
    df = pd.DataFrame([row])
    if categorical:
        for field in ['AttachmentBasis', 'ReinsInceptionDate', 'ReinsExpiryDate', 'PolInceptionDate']:
            if field in df:
                df[field] = df[field].astype('category')
    return df


@pytest.mark.parametrize("attachment,categorical,n_values,n", [
    ('LO', False, 4, 2000),      # losses occurring only, so the date branch never runs
    ('RA', False, 4, 2000),      # mixed risk attaching / losses occurring
    ('RA', True, 4, 2000),       # categorical columns with differing category sets
    ('RA', False, 1, 500),       # one distinct value, so most rows match
    ('LO', False, 4, 0),         # empty frame
])
def test_matches_row_wise_reference(attachment, categorical, n_values, n):
    """The vectorized mask equals the row wise one across the shapes filter_df takes."""
    rng = np.random.default_rng(20260810)
    filter_df = make_filter_df(n, rng, attachment=attachment, categorical=categorical, n_values=n_values)

    expected = row_wise_match(filter_df, RI_FILTER_FIELDS, ['layer_id'])
    result = match_filter_level_scope(filter_df, RI_FILTER_FIELDS, ['layer_id'])

    assert result.dtype == np.dtype('bool')
    assert result.tolist() == expected.tolist()


def test_matches_row_wise_reference_with_exact_match_keys():
    """Fields already used as merge keys are skipped, as they are equal by construction."""
    rng = np.random.default_rng(11)
    filter_df = make_filter_df(2000, rng, attachment='RA')
    merge_on = ['layer_id', 'PortNumber', 'AccNumber']

    expected = row_wise_match(filter_df, RI_FILTER_FIELDS, merge_on)
    result = match_filter_level_scope(filter_df, RI_FILTER_FIELDS, merge_on)

    assert result.tolist() == expected.tolist()
    # the merge keys are ignored, so more rows survive than with the plain key set
    assert result.sum() > match_filter_level_scope(filter_df, RI_FILTER_FIELDS, ['layer_id']).sum()


def test_field_only_filters_when_flagged_valid():
    """A field the scope does not specify is not compared, even when the values differ."""
    unflagged = make_one_row(LocNumber_x='1', LocNumber_y='2', LocNumber_valid=False)
    flagged = make_one_row(LocNumber_x='1', LocNumber_y='2', LocNumber_valid=True)

    assert match_filter_level_scope(unflagged, RI_FILTER_FIELDS, ['layer_id']).tolist() == [True]
    assert match_filter_level_scope(flagged, RI_FILTER_FIELDS, ['layer_id']).tolist() == [False]


@pytest.mark.parametrize("categorical", [False, True], ids=['object', 'categorical'])
@pytest.mark.parametrize("pol_inception,expected", [
    ('2023-12-31', False),   # before the treaty incepts
    ('2024-01-01', True),    # on the inception date
    ('2024-06-01', True),    # within the treaty period
    ('2024-12-31', True),    # on the expiry date
    ('2025-01-01', False),   # after the treaty expires
])
def test_risk_attaching_policy_inception_window(pol_inception, expected, categorical):
    """A risk attaching treaty only applies to policies incepting inside the treaty period."""
    filter_df = make_one_row(categorical=categorical, AttachmentBasis='RA', PolInceptionDate=pol_inception)

    result = match_filter_level_scope(filter_df, RI_FILTER_FIELDS, ['layer_id'])

    assert result.tolist() == [expected]
    assert result.tolist() == row_wise_match(filter_df, RI_FILTER_FIELDS, ['layer_id']).tolist()


def test_losses_occurring_ignores_the_inception_window():
    """The date window is only applied for AttachmentBasis 'RA'."""
    filter_df = make_one_row(AttachmentBasis='LO', PolInceptionDate='2019-01-01')

    assert match_filter_level_scope(filter_df, RI_FILTER_FIELDS, ['layer_id']).tolist() == [True]


@pytest.mark.parametrize("missing_field", ['ReinsInceptionDate', 'ReinsExpiryDate'])
def test_risk_attaching_without_reins_dates_raises(missing_field):
    """A risk attaching treaty with no treaty dates cannot be applied."""
    filter_df = make_one_row(AttachmentBasis='RA', **{missing_field: ''})

    with pytest.raises(OasisException, match="ReinsInceptionDate/ReinsExpiryDate missing"):
        match_filter_level_scope(filter_df, RI_FILTER_FIELDS, ['layer_id'])


def test_risk_attaching_without_pol_inception_date_raises():
    """A risk attaching treaty needs the policy inception date, and the error names the account."""
    filter_df = make_one_row(AttachmentBasis='RA', PolInceptionDate='',
                             PortNumber_x='Port1', PortNumber_y='Port1',
                             AccNumber_x='Acc1', AccNumber_y='Acc1')

    with pytest.raises(OasisException, match="PolInceptionDate missing") as raised:
        match_filter_level_scope(filter_df, RI_FILTER_FIELDS, ['layer_id'])

    assert "'PortNumber': 'Port1'" in str(raised.value)
    assert "'AccNumber': 'Acc1'" in str(raised.value)


def test_row_excluded_by_filter_fields_does_not_raise():
    """A row the filter fields already rejected never reaches the date validation."""
    filter_df = make_one_row(AttachmentBasis='RA', PolInceptionDate='',
                             LocNumber_x='1', LocNumber_y='2', LocNumber_valid=True)

    assert match_filter_level_scope(filter_df, RI_FILTER_FIELDS, ['layer_id']).tolist() == [False]


def test_first_offending_row_is_reported():
    """With several bad rows the error describes the first one, as the row wise loop did."""
    good = make_one_row()
    missing_pol = make_one_row(AttachmentBasis='RA', PolInceptionDate='')
    missing_reins = make_one_row(AttachmentBasis='RA', ReinsInceptionDate='')

    pol_first = pd.concat([good, missing_pol, missing_reins], ignore_index=True)
    reins_first = pd.concat([good, missing_reins, missing_pol], ignore_index=True)

    with pytest.raises(OasisException, match="PolInceptionDate missing"):
        match_filter_level_scope(pol_first, RI_FILTER_FIELDS, ['layer_id'])
    with pytest.raises(OasisException, match="ReinsInceptionDate/ReinsExpiryDate missing"):
        match_filter_level_scope(reins_first, RI_FILTER_FIELDS, ['layer_id'])


def test_missing_attachment_basis_column_skips_the_date_branch():
    """filter_df without an AttachmentBasis column is matched on the filter fields alone."""
    filter_df = make_one_row().drop(columns=['AttachmentBasis', 'PolInceptionDate'])

    result = match_filter_level_scope(filter_df, RI_FILTER_FIELDS, ['layer_id'])

    assert result.tolist() == [True]
    assert result.tolist() == row_wise_match(filter_df, RI_FILTER_FIELDS, ['layer_id']).tolist()
