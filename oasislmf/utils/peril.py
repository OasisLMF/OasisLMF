__all__ = [
    'PERILS',
    'PERIL_GROUPS',
]

import pandas as pd
from collections import defaultdict

from ods_tools.oed import OedSchema


def get_peril_info_from_schema(oed_version='latest version'):
    oed_schema = OedSchema.from_oed_schema_info(oed_version)

    peril_info = oed_schema.schema['perils']['info']
    peril_covered = oed_schema.schema['perils']['covered']

    peril_to_group = defaultdict(list)

    peril_groups = {}
    for p, v in peril_info.items():
        if v['Grouped PerilCode'] != 'Yes':
            continue

        peril_groups[p] = {'id': p, 'desc': v['Peril Description'],
                           'peril_ids': peril_covered[p]}

        for peril in peril_covered[p]:
            peril_to_group[peril].append(p)

    perils = {}
    group_count = defaultdict(int)
    for peril_code, peril in peril_info.items():
        if peril_code in peril_groups:
            continue

        curr_peril = {'id': peril_code,
                      'desc': peril['Peril Description']
                      }

        group_codes = peril_to_group.get(peril_code, ['AA1'])
        curr_peril['group_perils'] = group_codes
        perils[peril_code] = curr_peril

        for group_code in group_codes:
            group_count[group_code] += 1

    for peril_code, peril in perils.items():
        perils[peril_code]['group_perils'] = sorted(perils[peril_code]['group_perils'],
                                                    key=lambda x: group_count[x])

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
