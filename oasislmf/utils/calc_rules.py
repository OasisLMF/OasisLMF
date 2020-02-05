__all__ = [
    'get_calc_rules',
    'get_step_calc_rules',
    'update_calc_rules'
]

import os

from .data import (
    fast_zip_arrays,
    get_dataframe,
)
from .defaults import (
    STATIC_DATA_FP,
)


def update_calc_rules():
    fp = os.path.join(STATIC_DATA_FP, 'calc_rules.csv')

    terms = ['deductible', 'deductible_min', 'deductible_max', 'limit', 'share', 'attachment']
    terms_indicators = ['{}_gt_0'.format(t) for t in terms]
    types_and_codes = ['ded_type', 'ded_code', 'lim_type', 'lim_code']

    dtypes = {
        'id_key': 'str',
        **{t: 'int32' for t in ['calcrule_id'] + terms_indicators + types_and_codes}
    }
    calc_rules = get_dataframe(
        src_fp=fp,
        col_dtypes=dtypes
    )

    calc_rules['id_key'] = [t for t in fast_zip_arrays(*calc_rules[terms_indicators + types_and_codes].transpose().values)]

    calc_rules.to_csv(path_or_buf=fp, index=False, encoding='utf-8')


# Ktools calc. rules
def get_calc_rules(path=False, update=False):
    fp = os.path.join(STATIC_DATA_FP, 'calc_rules.csv')
    if path:
        return fp

    if update:
        update_calc_rules()

    return get_dataframe(src_fp=fp)


def get_step_calc_rules():
    """
    Get dataframe of calc rules for step policies

    :return: dataframe of calc rules for step policies
    :rtype: pandas.DataFrame
    """
    fp = os.path.join(STATIC_DATA_FP, 'calc_rules_step.csv')
    return get_dataframe(src_fp=fp)
