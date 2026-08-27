"""Write the expected exposure summaries in exposure_summary_expected/.

The functions below are the implementation `get_exposure_summary_stats` and `get_exposure_totals`
replaced, taken verbatim from oasislmf/preparation/summaries.py at cefcce6d. They live here rather
than in the test suite so that the expected summaries are produced by the old code once, committed,
and compared against from then on.

Regenerating from a later version of oasislmf would compare the new code against itself, so run
this only to add a case -- and never to make a failing test pass.

    python tests/preparation/generate_exposure_summary_expected.py
"""

import json
import os

from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.status import OASIS_KEYS_STATUS_MODELLED

from exposure_summary_cases import CASES, STATUSES, case

EXPECTED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exposure_summary_expected')

SUMMARY_CASES = [
    *[f'base_seed_{seed}' for seed in range(5)],
    'no_duplicate_keys',
    'categorical',
    'empty',
    'shared_loc_id',
    'shared_loc_id_both_buckets',
    *[name for name in CASES if name.startswith('colliding_field_')],
    'heterogeneous_duplicates',
    'missing_field_value',
    'all_coverage_types',
]

TOTALS_CASES = [
    *[f'base_seed_{seed}' for seed in range(5)],
    'shared_loc_id',
    'all_modelled',
    'empty_totals',
    'heterogeneous_duplicates_totals',
]


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


def main():
    os.makedirs(EXPECTED_DIR, exist_ok=True)
    for name in sorted(set(SUMMARY_CASES) | set(TOTALS_CASES)):
        df, categories = case(name)
        expected = {}
        if name in SUMMARY_CASES:
            expected['summary'] = reference_exposure_summary_fields(df, categories)
        if name in TOTALS_CASES:
            expected['totals'] = {scope: {key: float(value) if key == 'tiv' else int(value)
                                          for key, value in totals.items()}
                                  for scope, totals in reference_exposure_totals(df).items()}

        path = os.path.join(EXPECTED_DIR, f'{name}.json')
        with open(path, 'w') as expected_file:
            json.dump(expected, expected_file, indent=0, sort_keys=True)
        print(f'{path} ({os.path.getsize(path) / 1024:.0f} KiB)')


if __name__ == '__main__':
    main()
