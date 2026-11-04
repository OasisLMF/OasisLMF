"""Pin get_exposure_summary against the implementation it replaced.

The reference is built from the two implementations that came before:

- `reference_summary_frame` is the first half of `get_exposure_summary` as of 3eaa93fe8. It builds
  one row per exposure row, TIV column and peril, and left-merges the keys onto them, so every
  location gets a status for every coverage type and peril, `noreturn` where it has no keys row.
- `reference_exposure_summary_fields` and `reference_exposure_totals` are the row-wise summary of
  that frame as of cefcce6d, filtering it to one field value, status and coverage type at a time.

The current implementation builds neither the frame nor the per-group filters, so the portfolios
here are built to hit the cases where the frame's first-row deduplication decides the numbers:
loc_ids shared by several exposure rows, keys rows repeated or carrying several statuses, keys
outside the exposure, and missing values.
"""

import numpy as np
import pandas as pd
import pytest

from oasislmf.preparation.summaries import convert_col_name, get_exposure_summary, select_keys_columns
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.defaults import get_default_exposure_profile
from oasislmf.utils.status import OASIS_KEYS_STATUS, OASIS_KEYS_STATUS_MODELLED

PERILS = ['WTC', 'WSS', 'ORF', 'OO1']
STATUSES = list(OASIS_KEYS_STATUS)
FIELDS = ['CountryCode', 'OccupancyCode', 'NumberOfBuildings', 'IsAggregate']


def reference_summary_frame(exposure_df, keys_df, exposure_profile, additional_fields):
    """The `df_summary_peril` frame and field values of `get_exposure_summary` as of 3eaa93fe8."""
    keys_df = select_keys_columns(keys_df, ['loc_id', 'peril_id', 'coverage_type_id', 'status'])
    # get location tivs by coveragetype
    df_summary = []

    exposure_fields = ['loc_id']
    exposure_col_names = ['loc_id']

    for field_name in additional_fields:
        if field_name in exposure_df.columns:
            exposure_fields += [field_name]
            exposure_col_names += [convert_col_name(field_name)]
        else:
            pass

    for field_name in exposure_profile:
        if 'FMTermType' in exposure_profile[field_name].keys():
            if exposure_profile[field_name]['FMTermType'] == 'TIV' and exposure_profile[field_name]['ProfileElementName'] in exposure_df.columns:
                cov_name = exposure_profile[field_name]['ProfileElementName']
                coverage_type_id = exposure_profile[field_name]['CoverageTypeID']

                fields = exposure_fields + [cov_name]
                column_names = exposure_col_names + ['tiv']

                tmp_df = exposure_df[fields].copy()
                tmp_df.columns = column_names
                tmp_df['coverage_type_id'] = coverage_type_id

                # Add number_of_buildings column
                if 'NumberOfBuildings' in exposure_df:
                    tmp_df['number_of_buildings'] = exposure_df['NumberOfBuildings']
                else:
                    tmp_df['number_of_buildings'] = 1

                # Add number_of_risks column
                if 'IsAggregate' in exposure_df:
                    tmp_df['number_of_risks'] = tmp_df['number_of_buildings']
                    tmp_df.loc[exposure_df['IsAggregate'] == 0, 'number_of_risks'] = 1
                else:
                    tmp_df['number_of_risks'] = 1
                df_summary.append(tmp_df)
    df_summary = pd.concat(df_summary)

    # fix 0 number_of_buildings and risks
    df_summary = df_summary.replace({'number_of_buildings': 0, 'number_of_risks': 0}, 1)

    # get all perils
    peril_list = keys_df['peril_id'].drop_duplicates().to_list()

    # Initialise sub categories
    oed_categories = {'peril_id': peril_list}
    for field_name in exposure_col_names[1:]:  # ignore 'loc_id'
        field_list = df_summary[field_name].drop_duplicates().to_list()
        oed_categories[field_name] = field_list

    df_summary_peril = []
    for peril_id in peril_list:
        tmp_df = df_summary.copy()
        tmp_df['peril_id'] = peril_id
        df_summary_peril.append(tmp_df)
    df_summary_peril = pd.concat(df_summary_peril)

    df_summary_peril = df_summary_peril.merge(keys_df, how='left', on=['loc_id', 'coverage_type_id', 'peril_id'])
    no_return = OASIS_KEYS_STATUS['noreturn']['id']
    df_summary_peril['status'] = df_summary_peril['status'].fillna(no_return)

    return df_summary_peril, oed_categories


def reference_exposure_summary_field(df, exposure_summary, field_name, field_value, status):
    dedupe_cols_tiv = ['loc_id', 'peril_id']
    useful_cols = ['tiv', 'loc_id', 'peril_id', 'coverage_type_id',
                   'number_of_buildings', 'number_of_risks']
    df_field = df.loc[df[field_name] == field_value, useful_cols]

    for coverage_type in SUPPORTED_COVERAGE_TYPES:
        df_cov = df_field.loc[df_field['coverage_type_id'] == SUPPORTED_COVERAGE_TYPES[coverage_type]['id']]
        df_cov = df_cov.drop_duplicates(subset=dedupe_cols_tiv)
        tiv_sum = float(df_cov['tiv'].sum())
        exposure_summary[field_name][field_value][status]['tiv_by_coverage'][coverage_type] = tiv_sum
        exposure_summary[field_name][field_value][status]['tiv'] += tiv_sum

        df_num = df_cov.drop_duplicates(subset='loc_id')
        exposure_summary[field_name][field_value][status]['number_of_locations_by_coverage'][coverage_type] = len(df_num)
        exposure_summary[field_name][field_value][status]['number_of_buildings_by_coverage'][coverage_type] = int(df_num['number_of_buildings'].sum())
        exposure_summary[field_name][field_value][status]['number_of_risks_by_coverage'][coverage_type] = int(df_num['number_of_risks'].sum())

    num_df = df_field.drop_duplicates(subset='loc_id')
    exposure_summary[field_name][field_value][status]['number_of_locations'] = len(num_df['loc_id'])
    exposure_summary[field_name][field_value][status]['number_of_buildings'] = int(num_df['number_of_buildings'].sum())
    exposure_summary[field_name][field_value][status]['number_of_risks'] = int(num_df['number_of_risks'].sum())

    return exposure_summary


def reference_exposure_summary_fields(df, oed_categories):
    exposure_summary = {}
    for field_name, field_list in oed_categories.items():
        exposure_summary[field_name] = {}
        for value in field_list:
            exposure_summary[field_name][value] = {}
            for status in ['all'] + STATUSES:
                exposure_summary[field_name][value][status] = {
                    'tiv': 0.0, 'tiv_by_coverage': {},
                    'number_of_locations': 0, 'number_of_locations_by_coverage': {},
                    'number_of_buildings': 0, 'number_of_buildings_by_coverage': {},
                    'number_of_risks': 0, 'number_of_risks_by_coverage': {},
                }

    for status in ['all'] + STATUSES:
        if status != 'all':
            df_status = df[df['status'] == status]
        else:
            df_status = df.copy()

        for field_name, field_list in oed_categories.items():
            for field_value in field_list:
                # a missing value matches no row, and `==` would give pd.NA a mask of NAs
                missing = pd.api.types.is_scalar(field_value) and pd.isna(field_value)
                exposure_summary = reference_exposure_summary_field(
                    df_status.iloc[:0] if missing else df_status, exposure_summary, field_name, field_value, status
                )

    return exposure_summary


def reference_exposure_totals(df):
    dedupe_cols = ['loc_id', 'coverage_type_id']

    within_scope_tiv = df[df.status.isin(OASIS_KEYS_STATUS_MODELLED)].drop_duplicates(subset=dedupe_cols)['tiv'].sum()
    within_scope_num = len(df[df.status.isin(OASIS_KEYS_STATUS_MODELLED)]['loc_id'].unique())
    within_scope_num_buildings = int(
        df[df.status.isin(OASIS_KEYS_STATUS_MODELLED)].drop_duplicates(subset='loc_id')['number_of_buildings'].sum())
    within_scope_num_risks = int(
        df[df.status.isin(OASIS_KEYS_STATUS_MODELLED)].drop_duplicates(subset='loc_id')['number_of_risks'].sum())

    outside_scope_tiv = df[~df.status.isin(OASIS_KEYS_STATUS_MODELLED)].drop_duplicates(subset=dedupe_cols)['tiv'].sum()
    outside_scope_num = len(df[~df.status.isin(OASIS_KEYS_STATUS_MODELLED)]['loc_id'].unique())
    outside_scope_num_buildings = int(
        df[~df.status.isin(OASIS_KEYS_STATUS_MODELLED)].drop_duplicates(subset='loc_id')['number_of_buildings'].sum())
    outside_scope_num_risks = int(
        df[~df.status.isin(OASIS_KEYS_STATUS_MODELLED)].drop_duplicates(subset='loc_id')['number_of_risks'].sum())

    portfolio_tiv = df.drop_duplicates(subset=dedupe_cols)['tiv'].sum()
    portfolio_num = len(df['loc_id'].unique())
    portfolio_num_buildings = int(df.drop_duplicates(subset='loc_id')['number_of_buildings'].sum())
    portfolio_num_risks = int(df.drop_duplicates(subset='loc_id')['number_of_risks'].sum())

    return {
        "modelled": {"tiv": within_scope_tiv, "number_of_locations": within_scope_num,
                     "number_of_buildings": within_scope_num_buildings, "number_of_risks": within_scope_num_risks},
        "not-modelled": {"tiv": outside_scope_tiv, "number_of_locations": outside_scope_num,
                         "number_of_buildings": outside_scope_num_buildings, "number_of_risks": outside_scope_num_risks},
        "portfolio": {"tiv": portfolio_tiv, "number_of_locations": portfolio_num,
                      "number_of_buildings": portfolio_num_buildings, "number_of_risks": portfolio_num_risks},
    }


def reference_exposure_summary(exposure_df, keys_df, exposure_profile=get_default_exposure_profile(),
                               additional_fields=[]):
    df, oed_categories = reference_summary_frame(exposure_df, keys_df, exposure_profile, additional_fields)
    return {'total': reference_exposure_totals(df), **reference_exposure_summary_fields(df, oed_categories)}


def flatten(summary, prefix=()):
    """Flatten the nested summary so the leaves can be compared in one go, keeping their order."""
    flat = {}
    for key, value in summary.items():
        if pd.api.types.is_scalar(key) and pd.isna(key):
            key = 'missing'  # NaN != NaN, and pd.NA is not even a bool
        if isinstance(value, dict):
            flat.update(flatten(value, prefix + (key,)))
        else:
            flat[prefix + (key,)] = value
    return flat


def assert_summaries_equal(summary, expected):
    """Compare two summaries leaf by leaf, allowing for float summation order."""
    flat, flat_expected = flatten(summary), flatten(expected)
    # the report is written as JSON in dict order
    assert list(flat) == list(flat_expected)
    assert flat == pytest.approx(flat_expected)
    # counts are integers in the report, and must stay exact
    for key, value in flat_expected.items():
        if 'tiv' not in key[-2:] and 'tiv_by_coverage' not in key[-2:]:
            assert flat[key] == value and isinstance(flat[key], int), key


def make_portfolio(seed, n_locs=40, shared_loc_ids=False, missing_values=False, extra_keys=True,
                   tiv_covs=('BuildingTIV', 'OtherTIV', 'ContentsTIV', 'BITIV')):
    """Exposure and keys frames for a random portfolio.

    Args:
        seed (int): random seed
        n_locs (int): number of exposure rows
        shared_loc_ids (bool): give some exposure rows the loc_id of an earlier row, with the same or
            a different country code, TIVs and building count
        missing_values (bool): blank some TIVs, building counts, country codes and statuses
        extra_keys (bool): add keys for loc_ids outside the exposure and for a coverage type without
            a TIV column, with a peril only those keys return
        tiv_covs (tuple): the TIV columns of the exposure

    Returns:
        tuple: ``(exposure_df, keys_df)``, the keys holding both successes and errors
    """
    rng = np.random.default_rng(seed)
    exposure_df = pd.DataFrame({
        'loc_id': np.arange(1, n_locs + 1),
        'CountryCode': rng.choice(['GB', 'US', 'FR'], n_locs),
        'OccupancyCode': rng.choice([1050, 1100, 1150], n_locs),
        'NumberOfBuildings': rng.integers(0, 5, n_locs).astype(float),
        'IsAggregate': rng.integers(0, 2, n_locs),
    })
    for col in tiv_covs:
        exposure_df[col] = rng.uniform(0, 1e5, n_locs).round(2)

    if shared_loc_ids:
        shared = rng.choice(n_locs, n_locs // 4, replace=False)
        exposure_df.loc[shared, 'loc_id'] = rng.integers(1, n_locs // 2, len(shared))
        exposure_df.loc[shared[::2], 'CountryCode'] = 'ZA'

    if missing_values:
        for col in [tiv_covs[0], 'NumberOfBuildings', 'CountryCode']:
            exposure_df.loc[rng.random(n_locs) < 0.15, col] = np.nan

    # every loc_id, coverage and peril gets 0 to 3 keys rows, so some are repeated or carry
    # several statuses, and some get none and so are noreturn
    cells = all_cells(exposure_df, covs=[1, 2, 3])
    keys = cells.loc[np.repeat(cells.index, rng.choice([0, 1, 1, 1, 2, 3], len(cells)))].reset_index(drop=True)
    keys['status'] = rng.choice(STATUSES, len(keys), p=[0.4, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05])

    if extra_keys:
        keys = pd.concat([keys, pd.DataFrame({
            'loc_id': [n_locs + 100, n_locs + 101, int(exposure_df['loc_id'].iloc[0])],
            'coverage_type_id': [1, 1, 5],
            'peril_id': [PERILS[3], PERILS[0], PERILS[0]],
            'status': ['success', 'fail', 'success'],
        })], ignore_index=True)

    if missing_values:
        keys.loc[rng.random(len(keys)) < 0.05, 'status'] = np.nan

    keys = keys.sample(frac=1, random_state=seed).reset_index(drop=True)
    return exposure_df, keys


def all_cells(exposure_df, covs=(1, 2, 3, 4), perils=PERILS[:3]):
    """One row per loc_id of the exposure, coverage type and peril."""
    return pd.MultiIndex.from_product([exposure_df['loc_id'].unique(), covs, perils],
                                      names=['loc_id', 'coverage_type_id', 'peril_id']).to_frame(index=False)


def assert_matches_reference(exposure_df, keys_df, additional_fields=(), keys_input=None):
    expected = reference_exposure_summary(exposure_df, keys_df, additional_fields=list(additional_fields))
    summary = get_exposure_summary(exposure_df, keys_df if keys_input is None else keys_input,
                                   additional_fields=list(additional_fields))
    assert_summaries_equal(summary, expected)
    return summary


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('shared_loc_ids', [False, True])
@pytest.mark.parametrize('missing_values', [False, True])
def test_matches_reference(seed, shared_loc_ids, missing_values):
    exposure_df, keys_df = make_portfolio(seed, shared_loc_ids=shared_loc_ids, missing_values=missing_values)
    assert_matches_reference(exposure_df, keys_df, FIELDS)


def test_matches_reference_without_additional_fields():
    exposure_df, keys_df = make_portfolio(11, shared_loc_ids=True)
    assert_matches_reference(exposure_df, keys_df)


def test_matches_reference_with_tiv_columns_missing():
    exposure_df, keys_df = make_portfolio(12, tiv_covs=('ContentsTIV', 'BuildingTIV'))
    assert_matches_reference(exposure_df, keys_df, FIELDS)


def test_matches_reference_with_a_categorical_field():
    exposure_df, keys_df = make_portfolio(18, shared_loc_ids=True)
    exposure_df['CountryCode'] = exposure_df['CountryCode'].astype('category')
    assert_matches_reference(exposure_df, keys_df, FIELDS)


def test_matches_reference_when_no_key_is_in_the_exposure():
    exposure_df, keys_df = make_portfolio(13)
    keys_df['loc_id'] += 1000
    summary = assert_matches_reference(exposure_df, keys_df, FIELDS)
    assert summary['total']['modelled']['number_of_locations'] == 0


def test_an_empty_exposure_gives_a_zeroed_summary():
    exposure_df, keys_df = make_portfolio(19)
    summary = assert_matches_reference(exposure_df.iloc[:0], keys_df, FIELDS)

    assert all(value == 0 for value in flatten(summary).values())
    assert list(summary['peril_id']) == keys_df['peril_id'].drop_duplicates().to_list()


def test_matches_reference_with_an_explicit_noreturn_status():
    exposure_df, keys_df = make_portfolio(14)
    keys_df.loc[::7, 'status'] = OASIS_KEYS_STATUS['noreturn']['id']
    assert_matches_reference(exposure_df, keys_df, FIELDS)


def test_matches_reference_with_a_status_outside_the_report():
    """An unknown status is not reported by itself, but still counts as not modelled."""
    exposure_df, keys_df = make_portfolio(15)
    keys_df.loc[::5, 'status'] = 'something_else'
    assert_matches_reference(exposure_df, keys_df, FIELDS)


def test_split_and_categorical_keys_with_file_headers_match_reference():
    """Keys passed as GenerateFiles holds them: successes and errors apart, lowercased file headers,
    and the successes' peril id as a categorical."""
    exposure_df, keys_df = make_portfolio(16, shared_loc_ids=True)
    success = keys_df['status'] == 'success'
    keys_success = keys_df[success].drop(columns='status').rename(
        columns={'loc_id': 'locid', 'peril_id': 'perilid', 'coverage_type_id': 'coveragetypeid'})
    keys_success['perilid'] = keys_success['perilid'].astype('category')
    keys_success['areaperilid'] = 1
    keys_success['status'] = 'success'
    keys_errors = keys_df[~success].rename(columns={'loc_id': 'LocID', 'peril_id': 'PerilID',
                                                    'coverage_type_id': 'CoverageTypeID', 'status': 'Status'})
    keys_errors['Message'] = 'msg'

    # the reference sees them concatenated, the way write_exposure_summary used to concatenate them
    assert_matches_reference(exposure_df, pd.concat([keys_df[success], keys_df[~success]], ignore_index=True), FIELDS,
                             keys_input=[keys_success, keys_errors])


@pytest.mark.parametrize('missing', [np.nan, None, pd.NA])
def test_a_missing_field_value_is_reported_empty(missing):
    """A missing field value is listed, but matches no location, and is not coded -1 by the back door."""
    exposure_df, keys_df = make_portfolio(20)
    exposure_df['CountryCode'] = exposure_df['CountryCode'].astype(object)
    exposure_df.loc[:9, 'CountryCode'] = missing

    summary = assert_matches_reference(exposure_df, keys_df, ['CountryCode'])

    [missing_key] = [key for key in summary['country_code'] if pd.api.types.is_scalar(key) and pd.isna(key)]
    assert all(value == 0 for value in flatten(summary['country_code'][missing_key]).values())
    by_country = sum(summary['country_code'][value]['all']['number_of_locations'] for value in summary['country_code'])
    assert by_country == len(exposure_df) - 10


def test_one_loc_id_carrying_two_field_values_counts_under_both():
    """A loc_id shared by exposure rows with different country codes lands in both country buckets.

    The field value is part of the deduplication, so the country breakdown deliberately sums to more
    than the peril breakdown, which counts the location once, from its first exposure row.
    """
    exposure_df, keys_df = make_portfolio(21, n_locs=8, extra_keys=False)
    shared = exposure_df.iloc[[0]].assign(CountryCode='ZA', BuildingTIV=exposure_df['BuildingTIV'].iloc[0] * 2)
    exposure_df = pd.concat([exposure_df, shared], ignore_index=True)

    summary = assert_matches_reference(exposure_df, keys_df, ['CountryCode'])

    assert summary['country_code']['ZA']['all']['number_of_locations'] == 1
    by_country = sum(summary['country_code'][value]['all']['tiv'] for value in summary['country_code'])
    by_peril = sum(summary['peril_id'][value]['all']['tiv'] for value in summary['peril_id'])
    assert by_country > by_peril
    assert summary['total']['portfolio']['number_of_locations'] == 8


def test_statuses_sum_to_the_all_status_with_one_status_per_cell():
    exposure_df, _ = make_portfolio(22)
    keys_df = all_cells(exposure_df, covs=[1, 3])
    keys_df['status'] = np.random.default_rng(22).choice(STATUSES, len(keys_df))

    summary = assert_matches_reference(exposure_df, keys_df, ['CountryCode'])

    for section in ['peril_id', 'country_code']:
        for value, by_status in summary[section].items():
            assert sum(by_status[status]['tiv'] for status in STATUSES) == pytest.approx(by_status['all']['tiv'])


def test_every_cell_modelled_leaves_nothing_not_modelled():
    exposure_df, _ = make_portfolio(23)
    keys_df = all_cells(exposure_df)
    keys_df['status'] = np.random.default_rng(23).choice(OASIS_KEYS_STATUS_MODELLED, len(keys_df))

    totals = assert_matches_reference(exposure_df, keys_df, FIELDS)['total']

    assert totals['not-modelled']['number_of_locations'] == 0
    assert totals['not-modelled']['tiv'] == 0
    assert totals['modelled']['tiv'] == pytest.approx(totals['portfolio']['tiv'])


def test_tiv_is_attributed_by_coverage_type_id():
    """Each TIV column lands under its own coverage type, whatever the column order."""
    exposure_df, keys_df = make_portfolio(24)
    exposure_df = exposure_df[list(exposure_df.columns[::-1])]

    summary = assert_matches_reference(exposure_df, keys_df)

    for coverage_type, column in [('buildings', 'BuildingTIV'), ('other', 'OtherTIV'),
                                  ('contents', 'ContentsTIV'), ('bi', 'BITIV')]:
        assert summary['peril_id']['WTC']['all']['tiv_by_coverage'][coverage_type] == pytest.approx(exposure_df[column].sum())
    assert set(summary['peril_id']['WTC']['all']['tiv_by_coverage']) == set(SUPPORTED_COVERAGE_TYPES)


@pytest.mark.parametrize('column', ['Status', 'PerilId', 'CoverageTypeId', 'LocId', 'Tiv', 'loc_id'])
def test_field_named_after_an_internal_column_is_ignored(column):
    """A summary field named, once converted, after a column the summary uses itself is skipped.

    No OED column converts to one of these, but a custom column or the internal loc_id can be listed
    in summary_report_fields, and used to fail or overwrite the peril breakdown.
    """
    exposure_df, keys_df = make_portfolio(17)
    if column not in exposure_df:
        exposure_df[column] = 'X'

    summary = get_exposure_summary(exposure_df, keys_df, additional_fields=[column, 'CountryCode'])
    expected = reference_exposure_summary(exposure_df, keys_df, additional_fields=['CountryCode'])

    assert_summaries_equal(summary, expected)
    assert convert_col_name(column) not in summary or convert_col_name(column) == 'peril_id'
