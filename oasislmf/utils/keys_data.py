from __future__ import unicode_literals

__all__ = [
    'coverage_type_ids',
    'keys_data',
    'keys_status_flags',
    'peril_ids',
    'oasis_keys_file_to_record_metadict',
    'oasis_keys_error_file_to_record_metadict'
]

import string

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

from .coverage import (
    BUILDING_COVERAGE_CODE,
    CONTENTS_COVERAGE_CODE,
    OTHER_STRUCTURES_COVERAGE_CODE,
    TIME_COVERAGE_CODE,
)
from .peril import (
    PERIL_ID_FLOOD,
    PERIL_ID_QUAKE,
    PERIL_ID_SURGE,
    PERIL_ID_WIND,
)
from .status import (
    KEYS_STATUS_FAIL,
    KEYS_STATUS_NOMATCH,
    KEYS_STATUS_SUCCESS,
)

coverage_type_ids = [BUILDING_COVERAGE_CODE, CONTENTS_COVERAGE_CODE, OTHER_STRUCTURES_COVERAGE_CODE, TIME_COVERAGE_CODE]

peril_ids = [PERIL_ID_FLOOD, PERIL_ID_QUAKE, PERIL_ID_QUAKE, PERIL_ID_WIND]

keys_status_flags = [KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH, KEYS_STATUS_SUCCESS]


def keys_data(peril_id=PERIL_ID_WIND, coverage_type_id=BUILDING_COVERAGE_CODE, status=None, min_size=1, max_size=10):

    def keys_records(
        from_ids=integers(),
        from_peril_ids=sampled_from(peril_ids),
        from_coverage_type_ids=sampled_from(coverage_type_ids),
        from_area_peril_ids=integers(),
        from_vulnerability_ids=integers(),
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
        return keys_records(from_statuses=just(status), from_peril_ids=just(peril_id), from_coverage_type_ids=just(coverage_type_id), min_size=min_size, max_size=max_size)

    elif status and status in keys_status_flags and peril_id and not coverage_type_id:
        return keys_records(from_statuses=just(status), from_peril_ids=just(peril_id), from_coverage_type_ids=sampled_from(coverage_type_ids), min_size=min_size, max_size=max_size)

    elif status and status in keys_status_flags and not peril_id and coverage_type_id:
        return keys_records(from_statuses=just(status), from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=coverage_type_id, min_size=min_size, max_size=max_size)

    elif status and status in keys_status_flags and not (peril_id or coverage_type_id):
        return keys_records(from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=sampled_from(coverage_type_ids), min_size=min_size, max_size=max_size)

    elif status and status not in keys_status_flags and peril_id and coverage_type_id:
        return keys_records(from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), from_peril_ids=just(peril_id), from_coverage_type_ids=just(coverage_type_id), min_size=min_size, max_size=max_size)

    elif status and status not in keys_status_flags and peril_id and not coverage_type_id:
        return keys_records(from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), from_peril_ids=just(peril_id), from_coverage_type_ids=sampled_from(coverage_type_ids), min_size=min_size, max_size=max_size)

    elif status and status not in keys_status_flags and not peril_id and coverage_type_id:
        return keys_records(from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=just(coverage_type_id), min_size=min_size, max_size=max_size)

    elif status and status not in keys_status_flags and not (peril_id or coverage_type_id):
        return keys_records(from_statuses=sampled_from([KEYS_STATUS_FAIL, KEYS_STATUS_NOMATCH]), from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=sampled_from(coverage_type_ids), min_size=min_size, max_size=max_size)

    elif not status and peril_id and coverage_type_id:
        return keys_records(from_peril_ids=just(peril_id), from_coverage_type_ids=just(coverage_type_id) min_size=min_size, max_size=max_size)

    elif not status and peril_id and not coverage_type_id:
        return keys_records(from_peril_ids=just(peril_id), from_coverage_type_ids=sampled_from(coverage_type_ids) min_size=min_size, max_size=max_size)

    elif not status and not peril_id and coverage_type_id:
        return keys_records(from_peril_ids=sampled_from(peril_ids), from_coverage_type_ids=just(coverage_type_id) min_size=min_size, max_size=max_size)

    elif not (status or peril_id or coverage_type_id):
        return keys_records(min_size=min_size, max_size=max_size)
