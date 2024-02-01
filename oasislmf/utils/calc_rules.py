__all__ = [
    'get_calc_rules',
]

import os

from .data import get_dataframe
from .defaults import (
    STATIC_DATA_FP,
)
CALC_RULE_FILES = {
    'base': 'calc_rules_direct.csv',
    'policy_layer': 'calc_rules_direct_layer.csv',
    'step': 'calc_rules_step.csv',
}

CALC_RULE_TERMS_INFO = {
    'base': {
        'terms': ['deductible', 'deductible_min', 'deductible_max', 'limit', 'share'],
        'types_and_codes': ['ded_type', 'ded_code', 'lim_type', 'lim_code'],
    },
    'policy_layer': {
        'terms': ['limit', 'share', 'attachment'],
        'types_and_codes': []
    },
    'step': {
        'terms': ['deductible1', 'payout_start', 'payout_end', 'limit1', 'limit2'],
        'types_and_codes': ['trigger_type', 'payout_type']
    }
}

# Ktools calc. rules


def get_calc_rules(calc_rule_type):
    return get_dataframe(src_fp=os.path.join(STATIC_DATA_FP, CALC_RULE_FILES[calc_rule_type])), CALC_RULE_TERMS_INFO[calc_rule_type]
