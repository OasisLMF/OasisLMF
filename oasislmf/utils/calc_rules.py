__all__ = [
    'get_calc_rules',
    'get_step_calc_rules'
]

import os

from .data import (
    fast_zip_arrays,
    get_dataframe,
)
from .defaults import (
    STATIC_DATA_FP,
)


# Ktools calc. rules
def get_calc_rules(policy_layer=False):
    calc_rules_filename = 'calc_rules_direct'
    if policy_layer:
        calc_rules_filename += '_layer'
    calc_rules_filename += '.csv'
    fp = os.path.join(STATIC_DATA_FP, calc_rules_filename)
    return get_dataframe(src_fp=fp)


def get_step_calc_rules():
    """
    Get dataframe of calc rules for step policies

    :return: dataframe of calc rules for step policies
    :rtype: pandas.DataFrame
    """
    fp = os.path.join(STATIC_DATA_FP, 'calc_rules_step.csv')
    return get_dataframe(src_fp=fp)
