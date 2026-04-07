__all__ = [
    'PERILS',
    'PERIL_GROUPS',
]

import pandas as pd

from oasislmf.utils.defaults import get_default_peril_groups, get_default_perils

PERILS = get_default_perils()
PERIL_GROUPS = get_default_peril_groups()


def get_peril_groups_df():
    res = []
    for peril in PERILS.values():
        res.append((peril['id'], peril['id']))

    for peril_group in PERIL_GROUPS.values():
        for peril in peril_group['peril_ids']:
            res.append((peril_group['id'], peril))

    return pd.DataFrame(res, columns=['peril_group_id', 'peril_id'])
