"""Pin get_exposure_summary_fields and get_exposure_totals against the implementation they replaced.

The expected summaries in exposure_summary_expected/ were produced by the row-wise implementation
at cefcce6d and committed, so these compare the new code against the old rather than against
itself. generate_exposure_summary_expected.py holds that implementation and rewrites them.
"""

import json
import os

import pytest

from oasislmf.preparation.summaries import get_exposure_summary_fields, get_exposure_totals
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.status import OASIS_KEYS_STATUS

from .exposure_summary_cases import (COLLIDING_FIELD_NAMES, MISSING_VALUES, STATUSES, case,
                                     make_missing_field_value_df, oed_categories_for)

EXPECTED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exposure_summary_expected')


def load_expected(name, categories=None):
    """Read one case's expected summary, restoring the field values JSON turned into strings.

    A summary field is named after an OED column and can hold any value that column holds, so
    `loc_id` or `NumberOfBuildings` gives a summary keyed by integers.
    """
    with open(os.path.join(EXPECTED_DIR, f'{name}.json')) as expected_file:
        expected = json.load(expected_file)

    if categories is not None and 'summary' in expected:
        expected['summary'] = {
            field_name: {value: expected['summary'][field_name][str(value)] for value in field_list}
            for field_name, field_list in categories.items()
        }

    return expected


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


def assert_matches_expected(name):
    """Summarise one case and compare it against the committed expected summary."""
    df, categories = case(name)
    expected = load_expected(name, categories)

    assert_summaries_equal(get_exposure_summary_fields(df, categories), expected['summary'])


@pytest.mark.parametrize('seed', range(5))
def test_matches_reference_implementation(seed):
    assert_matches_expected(f'base_seed_{seed}')


def test_matches_reference_implementation_without_duplicate_keys():
    assert_matches_expected('no_duplicate_keys')


def test_matches_reference_implementation_with_categoricals():
    assert_matches_expected('categorical')


def test_field_value_absent_from_the_data_is_zeroed():
    df, categories = case('absent_field_value')

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
    assert_matches_expected('empty')


def test_one_loc_id_carrying_two_field_values():
    """A field value reachable only through a shared loc_id keeps its own TIV and counts."""
    df, categories = case('shared_loc_id')

    summary = get_exposure_summary_fields(df, categories)

    # the shared row's bucket is populated, not zeroed
    assert summary['country_code']['ZA']['all']['tiv'] > 0.0
    assert summary['country_code']['ZA']['all']['number_of_locations'] == 1
    assert_summaries_equal(summary, load_expected('shared_loc_id', categories)['summary'])


def test_one_loc_id_carrying_two_field_values_counts_it_under_both():
    """A shared loc_id lands in the bucket of each of its field values, as it did before.

    The field value is part of the deduplication key, so a location carrying two country codes
    contributes to both, and the country breakdown deliberately sums to more than the peril
    breakdown. That asymmetry is the pre-existing behaviour of the row-wise implementation, not
    something the deduplication may quietly resolve one way or the other.
    """
    df, categories = case('shared_loc_id_both_buckets')
    shared_values = sorted(df.loc[df['loc_id'] == 1, 'country_code'].unique())
    assert len(shared_values) == 2, 'fixture must give loc_id 1 two country codes'

    summary = get_exposure_summary_fields(df, categories)
    expected = load_expected('shared_loc_id_both_buckets', categories)['summary']

    for value in shared_values:
        assert summary['country_code'][value]['all']['tiv'] > 0.0

    by_country = sum(summary['country_code'][value]['all']['tiv'] for value in categories['country_code'])
    by_peril = sum(summary['peril_id'][value]['all']['tiv'] for value in categories['peril_id'])
    assert by_country > by_peril
    assert by_country == pytest.approx(
        sum(expected['country_code'][value]['all']['tiv'] for value in categories['country_code']))
    assert by_peril == pytest.approx(
        sum(expected['peril_id'][value]['all']['tiv'] for value in categories['peril_id']))


@pytest.mark.parametrize('field_name', COLLIDING_FIELD_NAMES)
def test_field_named_after_a_summed_or_key_column(field_name):
    """A summary field named after a column the summary sums or groups by does not shadow it.

    summary_report_fields comes from the model settings and is converted with convert_col_name, so
    an ordinary OED column like NumberOfBuildings arrives here as 'number_of_buildings' -- the same
    name as one of the columns the summary sums. 'status' and 'peril_id' name the two columns the
    summary itself groups by, so a field of either name is grouped by the same coded column twice.
    """
    assert_matches_expected(f'colliding_field_{field_name}')


def test_a_status_field_intersects_itself_on_the_diagonal_only():
    """A summary field named 'status' is the column the summary already groups by.

    So the (field value, status) pair is empty off the diagonal, and on it holds the whole of that
    status -- which is the identity the collapsed group key used to break, silently zeroing every
    non-'all' bucket rather than only the off-diagonal ones.
    """
    df, categories = case('status_field')

    summary = get_exposure_summary_fields(df, categories)

    for field_value in categories['status']:
        on_diagonal = summary['status'][field_value][field_value]
        assert on_diagonal['tiv'] > 0.0
        assert on_diagonal['number_of_locations'] > 0
        assert on_diagonal['tiv'] == pytest.approx(summary['status'][field_value]['all']['tiv'])
        assert on_diagonal['number_of_locations'] == summary['status'][field_value]['all']['number_of_locations']

        for status in categories['status']:
            if status != field_value:
                assert summary['status'][field_value][status]['tiv'] == 0.0
                assert summary['status'][field_value][status]['number_of_locations'] == 0


def test_statuses_sum_to_the_all_status():
    df, categories = case('statuses_sum')

    summary = get_exposure_summary_fields(df, categories)

    for peril_id in categories['peril_id']:
        per_status = sum(summary['peril_id'][peril_id][status]['tiv'] for status in OASIS_KEYS_STATUS)
        assert per_status == pytest.approx(summary['peril_id'][peril_id]['all']['tiv'])


def assert_totals_match_expected(name):
    df, _ = case(name)

    assert_summaries_equal(get_exposure_totals(df), load_expected(name)['totals'])


@pytest.mark.parametrize('seed', range(5))
def test_totals_match_the_reference_implementation(seed):
    assert_totals_match_expected(f'base_seed_{seed}')


def test_totals_match_the_reference_implementation_with_a_shared_loc_id():
    assert_totals_match_expected('shared_loc_id')


def test_totals_match_the_reference_implementation_when_every_status_is_modelled():
    """The not-modelled scope is empty, so its sums come from an empty frame rather than a filter."""
    df, _ = case('all_modelled')

    totals = get_exposure_totals(df)

    assert_summaries_equal(totals, load_expected('all_modelled')['totals'])
    assert totals['not-modelled']['number_of_locations'] == 0
    assert totals['not-modelled']['tiv'] == 0
    assert totals['modelled']['tiv'] == pytest.approx(totals['portfolio']['tiv'])


def test_totals_match_the_reference_implementation_on_an_empty_frame():
    assert_totals_match_expected('empty_totals')


def test_duplicate_keys_differing_in_tiv_and_status_keep_the_first_row():
    assert_matches_expected('heterogeneous_duplicates')


def test_totals_for_duplicate_keys_differing_in_tiv_and_status():
    assert_totals_match_expected('heterogeneous_duplicates_totals')


@pytest.mark.parametrize('missing', MISSING_VALUES)
def test_a_missing_field_value_is_bucketed_to_zero(missing):
    """A null field value matches no bucket, and must not become one via factorize's -1 sentinel.

    pd.factorize codes a null as -1, which is a legal index, so this is exactly the sort of
    mechanism worth pinning rather than reasoning about. The three null sentinels are summarised
    alike, so they share one expected summary.
    """
    df = make_missing_field_value_df(missing)
    categories = oed_categories_for(df)
    categories['country_code'] = [value for value in categories['country_code'] if isinstance(value, str)]

    assert_summaries_equal(
        get_exposure_summary_fields(df, categories),
        load_expected('missing_field_value', categories)['summary'],
    )


def test_every_supported_coverage_type_is_attributed_by_id():
    """The coverage attribution is driven by coverage_type_id, not by position in the fixture.

    The other cases here only ever use the first four of the thirteen supported coverage types, so
    the remaining nine are only exercised as zero-filled buckets.
    """
    df, categories = case('all_coverage_types')

    summary = get_exposure_summary_fields(df, categories)

    assert_summaries_equal(summary, load_expected('all_coverage_types', categories)['summary'])
    # every coverage type carries TIV, so none of the thirteen is only ever a zero bucket
    for coverage_type in SUPPORTED_COVERAGE_TYPES:
        assert sum(summary['country_code'][value]['all']['tiv_by_coverage'][coverage_type]
                   for value in categories['country_code']) > 0, coverage_type
