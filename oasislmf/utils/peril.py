__all__ = [
    'PERILS',
    'PERIL_GROUPS',
]

import pandas as pd
from collections import defaultdict

from ods_tools.oed import OedSchema


def get_peril_info_from_schema(oed_version='latest version'):
    '''
    Get perils and peril_groups info from OedSchema.

    Args:
        oed_version (str): The version of OedSchema, default to `latest version`.
    '''
    oed_schema = OedSchema.from_oed_schema_info(oed_version)

    peril_info = oed_schema.schema['perils']['info']
    peril_covered = oed_schema.schema['perils']['covered']

    peril_groups = {}
    perils = {}
    for peril_code, peril in peril_info.items():
        if peril['Grouped PerilCode'] == 'Yes':
            peril_groups[peril_code] = {'id': peril_code, 'desc': peril['Peril Description'],
                                        'peril_ids': peril_covered[peril_code]}
        else:
            perils[peril_code] = {'id': peril_code,
                                  'desc': peril['Peril Description']}

    return perils, peril_groups


PERILS, PERIL_GROUPS = get_peril_info_from_schema()


def get_peril_groups_df():
    res = []
    for peril in PERILS.values():
        res.append((peril['id'], peril['id']))

    for peril_group in PERIL_GROUPS.values():
        for peril in peril_group['peril_ids']:
            res.append((peril_group['id'], peril))

    return pd.DataFrame(res, columns=['peril_group_id', 'peril_id'])
