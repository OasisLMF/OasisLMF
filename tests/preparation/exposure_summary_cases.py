"""The frames get_exposure_summary_fields and get_exposure_totals are pinned against.

Shared by test_exposure_summary_fields.py and by generate_exposure_summary_expected.py, which
writes the expected summaries in exposure_summary_expected/ from the implementation this one
replaced. A case is a name and a callable returning ``(df, oed_categories)``, so the frames stay
built rather than serialised and the expected summaries alone are committed.
"""

import numpy as np
import pandas as pd

from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.status import OASIS_KEYS_STATUS, OASIS_KEYS_STATUS_MODELLED

STATUSES = ['all'] + list(OASIS_KEYS_STATUS.keys())


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


def make_heterogeneous_duplicates_df(seed=41):
    """Duplicate keys whose rows differ, so which row survives the deduplication is observable.

    make_summary_peril_df duplicates whole rows, so every duplicate group is identical in tiv,
    status and number_of_buildings and the tests cannot tell 'keeps the first row' from 'keeps an
    arbitrary row' -- even though the summed TIV depends on the answer.
    """
    df = make_summary_peril_df(seed, num_locations=12, duplicate_keys=False)
    later = df.copy()
    later['tiv'] = later['tiv'] * 10
    later['number_of_buildings'] = later['number_of_buildings'] + 7
    later['number_of_risks'] = later['number_of_buildings']
    later['status'] = list(OASIS_KEYS_STATUS.keys())[-1]

    return pd.concat([df, later], ignore_index=True)


def make_all_coverage_types_df(seed=44, num_locations=6):
    """A frame carrying TIV on every supported coverage type, not only the first four.

    The other frames here use four of the thirteen, so the remaining nine are only ever exercised
    as zero-filled buckets.
    """
    coverage_type_ids = [info['id'] for info in SUPPORTED_COVERAGE_TYPES.values()]
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame({
        'loc_id': np.repeat(np.arange(1, num_locations + 1), len(coverage_type_ids)),
        'coverage_type_id': np.tile(coverage_type_ids, num_locations),
        'tiv': rng.uniform(1e3, 1e6, num_locations * len(coverage_type_ids)).round(2),
    })
    rows['number_of_buildings'] = rng.integers(1, 5, len(rows))
    rows['number_of_risks'] = rows['number_of_buildings']
    rows['country_code'] = np.repeat(rng.choice(['GB', 'US'], num_locations), len(coverage_type_ids))
    df = pd.concat([rows.assign(peril_id=peril) for peril in ['WTC', 'WSS']], ignore_index=True)
    df['status'] = rng.choice(list(OASIS_KEYS_STATUS.keys()), len(df))

    return df


def make_missing_field_value_df(missing, seed=43):
    """A frame where some rows have no country code at all.

    pd.factorize codes a null as -1, which is a legal index, so a null must not become a bucket of
    its own by the back door.
    """
    df = make_summary_peril_df(seed, num_locations=12, duplicate_keys=False)
    df.loc[df['loc_id'] <= 4, 'country_code'] = missing

    return df


def oed_categories_for(df):
    return {
        'peril_id': df['peril_id'].drop_duplicates().to_list(),
        'country_code': df['country_code'].drop_duplicates().to_list(),
    }


def categories_named_after(df, field_name):
    return {
        'peril_id': df['peril_id'].drop_duplicates().to_list(),
        field_name: df[field_name].drop_duplicates().to_list(),
    }


COLLIDING_FIELD_NAMES = ['number_of_buildings', 'number_of_risks', 'coverage_type_id',
                         'loc_id', 'status', 'peril_id']

MISSING_VALUES = [np.nan, None, pd.NA]


def _base(seed):
    def build():
        df = make_summary_peril_df(seed)
        return df, oed_categories_for(df)
    return build


def _shared_loc_id(seed):
    def build():
        df = make_shared_loc_id_df(seed=seed)
        return df, oed_categories_for(df)
    return build


def _colliding_field(field_name):
    def build():
        df = make_summary_peril_df(24, num_locations=12)
        return df, categories_named_after(df, field_name)
    return build


def _no_duplicate_keys():
    df = make_summary_peril_df(11, duplicate_keys=False)
    return df, oed_categories_for(df)


def _categorical():
    df = make_summary_peril_df(12, categorical=True)
    return df, oed_categories_for(df)


def _absent_field_value():
    df = make_summary_peril_df(13)
    categories = oed_categories_for(df)
    categories['country_code'] = categories['country_code'] + ['ZZ']
    return df, categories


def _empty():
    df = make_summary_peril_df(14).iloc[:0]
    return df, {'peril_id': ['WTC'], 'country_code': ['GB']}


def _statuses_sum():
    df = make_summary_peril_df(15, duplicate_keys=False)
    return df, oed_categories_for(df)


def _status_field():
    df = make_summary_peril_df(25, num_locations=12)
    return df, {'status': df['status'].drop_duplicates().to_list()}


def _all_modelled():
    """Every status is modelled, so the not-modelled scope is empty."""
    df = make_summary_peril_df(31)
    df['status'] = list(OASIS_KEYS_STATUS_MODELLED)[0]
    return df, oed_categories_for(df)


def _empty_totals():
    df = make_summary_peril_df(32).iloc[:0]
    return df, oed_categories_for(df)


def _heterogeneous_duplicates(seed):
    def build():
        df = make_heterogeneous_duplicates_df(seed=seed)
        return df, oed_categories_for(df)
    return build


def _missing_field_value(missing):
    def build():
        df = make_missing_field_value_df(missing)
        categories = oed_categories_for(df)
        categories['country_code'] = [value for value in categories['country_code'] if isinstance(value, str)]
        return df, categories
    return build


def _all_coverage_types():
    df = make_all_coverage_types_df()
    return df, oed_categories_for(df)


CASES = {
    **{f'base_seed_{seed}': _base(seed) for seed in range(5)},
    'no_duplicate_keys': _no_duplicate_keys,
    'categorical': _categorical,
    'absent_field_value': _absent_field_value,
    'empty': _empty,
    'statuses_sum': _statuses_sum,
    'shared_loc_id': _shared_loc_id(21),
    'shared_loc_id_both_buckets': _shared_loc_id(23),
    **{f'colliding_field_{name}': _colliding_field(name) for name in COLLIDING_FIELD_NAMES},
    'status_field': _status_field,
    'all_modelled': _all_modelled,
    'empty_totals': _empty_totals,
    'heterogeneous_duplicates': _heterogeneous_duplicates(41),
    'heterogeneous_duplicates_totals': _heterogeneous_duplicates(42),
    'missing_field_value': _missing_field_value(np.nan),
    'all_coverage_types': _all_coverage_types,
}


def case(name):
    """Build the frame and categories for one case."""
    return CASES[name]()
