"""Pin get_exposure_summary_fields against the row-wise implementation it replaced."""

import numpy as np
import pandas as pd
import pytest

from oasislmf.preparation.summaries import get_exposure_summary_fields
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.status import OASIS_KEYS_STATUS

STATUSES = ['all'] + list(OASIS_KEYS_STATUS.keys())


def reference_exposure_summary_field(df, exposure_summary, field_name, field_value, status):
    """The implementation replaced by get_exposure_summary_stats, kept here as the reference."""
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
    """The row-wise get_exposure_summary loop, filtering the frame once per field value and status."""
    exposure_summary = {}
    for field_name, field_list in oed_categories.items():
        exposure_summary[field_name] = {}
        for value in field_list:
            exposure_summary[field_name][value] = {}
            for status in STATUSES:
                exposure_summary[field_name][value][status] = {
                    'tiv': 0.0, 'tiv_by_coverage': {},
                    'number_of_locations': 0, 'number_of_locations_by_coverage': {},
                    'number_of_buildings': 0, 'number_of_buildings_by_coverage': {},
                    'number_of_risks': 0, 'number_of_risks_by_coverage': {},
                }

    for status in STATUSES:
        if status != 'all':
            df_status = df[df['status'] == status]
        else:
            df_status = df.copy()

        for field_name, field_list in oed_categories.items():
            for field_value in field_list:
                exposure_summary = reference_exposure_summary_field(
                    df_status, exposure_summary, field_name, field_value, status
                )

    return exposure_summary


def flatten(summary, prefix=()):
    """Flatten the nested summary so the leaves can be compared in one go."""
    flat = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            flat.update(flatten(value, prefix + (key,)))
        else:
            flat[prefix + (key,)] = value
    return flat


def assert_summaries_equal(summary, expected):
    """Compare two summaries leaf by leaf, allowing for float summation order."""
    flat, flat_expected = flatten(summary), flatten(expected)
    assert flat.keys() == flat_expected.keys()
    assert flat == pytest.approx(flat_expected)
    # counts are integers in the report, and must stay exact
    for key, value in flat_expected.items():
        if isinstance(value, int):
            assert flat[key] == value and isinstance(flat[key], int), key


def make_summary_peril_df(seed, num_locations=60, duplicate_keys=True, categorical=False):
    """Build a frame shaped like df_summary_peril in get_exposure_summary."""
    rng = np.random.default_rng(seed)
    coverage_type_ids = [info['id'] for info in SUPPORTED_COVERAGE_TYPES.values()][:4]
    perils = ['WTC', 'WSS', 'ORF']
    countries = ['GB', 'US', 'FR']

    loc_id = np.arange(1, num_locations + 1)
    rows = pd.DataFrame({
        'loc_id': np.repeat(loc_id, len(coverage_type_ids)),
        'coverage_type_id': np.tile(coverage_type_ids, num_locations),
        'tiv': rng.uniform(1e3, 1e6, num_locations * len(coverage_type_ids)).round(2),
    })
    rows['number_of_buildings'] = rng.integers(1, 5, len(rows))[np.argsort(np.argsort(rows['loc_id']))]
    rows['number_of_risks'] = rows['number_of_buildings']
    rows['country_code'] = np.repeat(rng.choice(countries, num_locations), len(coverage_type_ids))

    df = pd.concat([rows.assign(peril_id=peril) for peril in perils], ignore_index=True)
    df['status'] = rng.choice(list(OASIS_KEYS_STATUS.keys()), len(df))

    if duplicate_keys:
        # the keys file can hold several rows per loc/peril/coverage, which is what the
        # deduplication in the summary exists for
        df = pd.concat([df, df.sample(frac=0.3, random_state=seed)], ignore_index=True)

    if categorical:
        for col in ['peril_id', 'country_code', 'status']:
            df[col] = df[col].astype('category')

    return df


def oed_categories_for(df):
    return {
        'peril_id': df['peril_id'].drop_duplicates().to_list(),
        'country_code': df['country_code'].drop_duplicates().to_list(),
    }


@pytest.mark.parametrize('seed', range(5))
def test_matches_reference_implementation(seed):
    df = make_summary_peril_df(seed)
    categories = oed_categories_for(df)

    assert_summaries_equal(
        get_exposure_summary_fields(df, categories),
        reference_exposure_summary_fields(df, categories),
    )


def test_matches_reference_implementation_without_duplicate_keys():
    df = make_summary_peril_df(11, duplicate_keys=False)
    categories = oed_categories_for(df)

    assert_summaries_equal(
        get_exposure_summary_fields(df, categories),
        reference_exposure_summary_fields(df, categories),
    )


def test_matches_reference_implementation_with_categoricals():
    df = make_summary_peril_df(12, categorical=True)
    categories = oed_categories_for(df)

    assert_summaries_equal(
        get_exposure_summary_fields(df, categories),
        reference_exposure_summary_fields(df, categories),
    )


def test_field_value_absent_from_the_data_is_zeroed():
    df = make_summary_peril_df(13)
    categories = oed_categories_for(df)
    categories['country_code'] = categories['country_code'] + ['ZZ']

    summary = get_exposure_summary_fields(df, categories)

    for status in STATUSES:
        absent = summary['country_code']['ZZ'][status]
        assert absent['tiv'] == 0.0
        assert absent['number_of_locations'] == 0
        assert absent['number_of_buildings'] == 0
        assert absent['number_of_risks'] == 0
        assert set(absent['tiv_by_coverage']) == set(SUPPORTED_COVERAGE_TYPES)
        assert set(absent['tiv_by_coverage'].values()) == {0.0}
        assert set(absent['number_of_locations_by_coverage'].values()) == {0}


def test_empty_frame_gives_a_fully_zeroed_summary():
    df = make_summary_peril_df(14).iloc[:0]
    categories = {'peril_id': ['WTC'], 'country_code': ['GB']}

    summary = get_exposure_summary_fields(df, categories)

    assert_summaries_equal(summary, reference_exposure_summary_fields(df, categories))


def make_shared_loc_id_df(seed=21, field_value='ZA'):
    """A frame where one loc_id carries two different country codes.

    loc_id identifies the location (PortNumber, AccNumber, LocNumber) rather than the row, so two
    exposure rows with the same LocNumber share one, and ods_tools has no uniqueness rule that
    rejects them. Any deduplication shared across summary fields drops one of the two, taking the
    whole of its field bucket to zero while the portfolio total stays put.
    """
    df = make_summary_peril_df(seed, num_locations=8, duplicate_keys=False)
    shared = df.loc[df['loc_id'] == 1].copy()
    shared['country_code'] = field_value
    shared['tiv'] = shared['tiv'] * 2
    shared['number_of_buildings'] = shared['number_of_buildings'] + 1
    shared['number_of_risks'] = shared['number_of_buildings']

    return pd.concat([df, shared], ignore_index=True)


def test_one_loc_id_carrying_two_field_values():
    """A field value reachable only through a shared loc_id keeps its own TIV and counts."""
    df = make_shared_loc_id_df()
    categories = oed_categories_for(df)

    summary = get_exposure_summary_fields(df, categories)

    # the shared row's bucket is populated, not zeroed
    assert summary['country_code']['ZA']['all']['tiv'] > 0.0
    assert summary['country_code']['ZA']['all']['number_of_locations'] == 1
    assert_summaries_equal(summary, reference_exposure_summary_fields(df, categories))


def test_one_loc_id_carrying_two_field_values_counts_it_under_both():
    """A shared loc_id lands in the bucket of each of its field values, as it did before.

    The field value is part of the deduplication key, so a location carrying two country codes
    contributes to both, and the country breakdown deliberately sums to more than the peril
    breakdown. That asymmetry is the pre-existing behaviour of the row-wise implementation, not
    something the deduplication may quietly resolve one way or the other.
    """
    df = make_shared_loc_id_df(seed=23)
    categories = oed_categories_for(df)
    shared_values = sorted(df.loc[df['loc_id'] == 1, 'country_code'].unique())
    assert len(shared_values) == 2, 'fixture must give loc_id 1 two country codes'

    summary = get_exposure_summary_fields(df, categories)
    expected = reference_exposure_summary_fields(df, categories)

    for value in shared_values:
        assert summary['country_code'][value]['all']['tiv'] > 0.0

    by_country = sum(summary['country_code'][value]['all']['tiv'] for value in categories['country_code'])
    by_peril = sum(summary['peril_id'][value]['all']['tiv'] for value in categories['peril_id'])
    assert by_country > by_peril
    assert by_country == pytest.approx(
        sum(expected['country_code'][value]['all']['tiv'] for value in categories['country_code']))
    assert by_peril == pytest.approx(
        sum(expected['peril_id'][value]['all']['tiv'] for value in categories['peril_id']))


@pytest.mark.parametrize('field_name', ['number_of_buildings', 'number_of_risks', 'coverage_type_id', 'loc_id'])
def test_field_named_after_a_summed_or_key_column(field_name):
    """A summary field named after a column the summary sums or groups by does not shadow it.

    summary_report_fields comes from the model settings and is converted with convert_col_name, so
    an ordinary OED column like NumberOfBuildings arrives here as 'number_of_buildings' -- the same
    name as one of the columns COUNT_AGGREGATION sums.
    """
    df = make_summary_peril_df(24, num_locations=12)
    categories = {
        'peril_id': df['peril_id'].drop_duplicates().to_list(),
        field_name: df[field_name].drop_duplicates().to_list(),
    }

    assert_summaries_equal(
        get_exposure_summary_fields(df, categories),
        reference_exposure_summary_fields(df, categories),
    )


def test_statuses_sum_to_the_all_status():
    df = make_summary_peril_df(15, duplicate_keys=False)
    categories = oed_categories_for(df)

    summary = get_exposure_summary_fields(df, categories)

    for peril_id in categories['peril_id']:
        per_status = sum(summary['peril_id'][peril_id][status]['tiv'] for status in OASIS_KEYS_STATUS)
        assert per_status == pytest.approx(summary['peril_id'][peril_id]['all']['tiv'])
