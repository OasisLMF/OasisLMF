from __future__ import unicode_literals

__all__ = [
    'canonical_exposure_data',
    'coverage_type_ids',
    'keys_data',
    'keys_status_flags',
    'peril_ids',
    'write_input_files'
]

import string

from collections import OrderedDict

import pandas as pd

from hypothesis import given
from hypothesis.strategies import (
    booleans,
    fixed_dictionaries,
    integers,
    just,
    lists,
    sampled_from,
    text,
    tuples,
)

from oasislmf.utils.coverage import (
    BUILDING_COVERAGE_CODE,
    CONTENTS_COVERAGE_CODE,
    OTHER_STRUCTURES_COVERAGE_CODE,
    TIME_COVERAGE_CODE,
)
from oasislmf.utils.peril import (
    PERIL_ID_FLOOD,
    PERIL_ID_QUAKE,
    PERIL_ID_SURGE,
    PERIL_ID_WIND,
)
from oasislmf.utils.status import (
    KEYS_STATUS_FAIL,
    KEYS_STATUS_NOMATCH,
    KEYS_STATUS_SUCCESS,
)

coverage_type_ids = [BUILDING_COVERAGE_CODE, CONTENTS_COVERAGE_CODE, OTHER_STRUCTURES_COVERAGE_CODE, TIME_COVERAGE_CODE]

keys_status_flags = [KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH, KEYS_STATUS_SUCCESS]

peril_ids = [PERIL_ID_FLOOD, PERIL_ID_QUAKE, PERIL_ID_QUAKE, PERIL_ID_WIND]


def keys_data(peril_id=PERIL_ID_WIND, coverage_type_id=BUILDING_COVERAGE_CODE, status=None, size=None, min_size=1, max_size=10):

    def keys_records(
        from_ids=integers(min_value=1, max_value=(size if size else max_size)),
        from_peril_ids=sampled_from(peril_ids),
        from_coverage_type_ids=sampled_from(coverage_type_ids),
        from_area_peril_ids=integers(min_value=1, max_value=(size if size else max_size)),
        from_vulnerability_ids=integers(min_value=1, max_value=(size if size else max_size)),
        from_statuses=sampled_from(keys_status_flags),
        from_messages=text(min_size=1, max_size=100, alphabet=string.ascii_letters),
        min_size=1,
        max_size=10
    ):
        return lists(fixed_dictionaries({
            'id': from_ids,
            'peril_id': from_peril_ids,
            'coverage': from_coverage_type_ids,
            'area_peril_id': from_area_peril_ids,
            'vulnerability_id': from_vulnerability_ids,
            'status': from_statuses,
            'message': from_messages
        }), min_size=min_size, max_size=max_size)

    if status and status in keys_status_flags and peril_id and coverage_type_id:
        return keys_records(from_peril_ids=just(peril_id), from_coverage_type_ids=just(coverage_type_id), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), from_statuses=just(status), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif status and status in keys_status_flags and peril_id and not coverage_type_id:
        return keys_records(from_peril_ids=just(peril_id), from_coverage_type_ids=sampled_from(coverage_type_ids), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), from_statuses=just(status), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif status and status in keys_status_flags and not peril_id and coverage_type_id:
        return keys_records(from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=coverage_type_id, from_area_peril_ids=just(0), from_vulnerability_ids=just(0), from_statuses=just(status), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif status and status in keys_status_flags and not (peril_id or coverage_type_id):
        return keys_records(from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=sampled_from(coverage_type_ids), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif status and status not in keys_status_flags and peril_id and coverage_type_id:
        return keys_records(from_peril_ids=just(peril_id), from_coverage_type_ids=just(coverage_type_id), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif status and status not in keys_status_flags and peril_id and not coverage_type_id:
        return keys_records(from_peril_ids=just(peril_id), from_coverage_type_ids=sampled_from(coverage_type_ids), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif status and status not in keys_status_flags and not peril_id and coverage_type_id:
        return keys_records(from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=just(coverage_type_id), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif status and status not in keys_status_flags and not (peril_id or coverage_type_id):
        return keys_records(from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=sampled_from(coverage_type_ids), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif not status and peril_id and coverage_type_id:
        return keys_records(from_peril_ids=just(peril_id), from_coverage_type_ids=just(coverage_type_id), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif not status and peril_id and not coverage_type_id:
        return keys_records(from_peril_ids=just(peril_id), from_coverage_type_ids=sampled_from(coverage_type_ids), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif not status and not peril_id and coverage_type_id:
        return keys_records(from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=just(coverage_type_id), from_area_peril_ids=just(0), from_vulnerability_ids=just(0), min_size=(size if size else min_size), max_size=(size if size else max_size))

    elif not (status or peril_id or coverage_type_id):
        return keys_records(from_area_peril_ids=just(0), from_vulnerability_ids=just(0), min_size=(size if size else min_size), max_size=(size if size else max_size))


def canonical_exposure_data(num_rows=10, min_value=None, max_value=None):
    return lists(tuples(integers(min_value=min_value, max_value=max_value)), min_size=num_rows, max_size=num_rows).map(
        lambda l: [(i + 1, 1.0) for i, _ in enumerate(l)]
    )


def write_input_files(
    keys,
    keys_file_path,
    exposures,
    exposures_file_path,
    profile_element_name='profile_element'
):
    heading_row = OrderedDict([
        ('id', 'LocID'),
        ('peril_id', 'PerilID'),
        ('coverage', 'CoverageID'),
        ('area_peril_id', 'AreaPerilID'),
        ('vulnerability_id', 'VulnerabilityID'),
    ])

    pd.DataFrame(
        columns=heading_row.keys(),
        data=[heading_row]+keys,
    ).to_csv(
        path_or_buf=keys_file_path,
        index=False,
        encoding='utf-8',
        header=False
    )

    heading_row = OrderedDict([
        ('row_id', 'ROW_ID'),
        (profile_element_name, 'ProfileElementName'),
    ])
    exposures_df = pd.DataFrame(
        columns=heading_row.keys(),
        data=exposures
    )
    exposures_df[profile_element_name] = exposures_df[profile_element_name].astype(float)
    exposures_df.to_csv(
        path_or_buf=exposures_file_path,
        index=False,
        encoding='utf-8',
        header=True
    )
