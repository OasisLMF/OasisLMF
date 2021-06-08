__all__ = [
    'OASIS_KEYS_STATUS',
    'OASIS_TASK_STATUS',
    'OASIS_KEYS_STATUS_MODELLED'
]


OASIS_KEYS_SC = 'success'
OASIS_KEYS_FL = 'fail'
OASIS_KEYS_NM = 'nomatch'
OASIS_KEYS_FA = 'fail_ap'
OASIS_KEYS_FV = 'fail_v'
OASIS_KEYS_NR = 'notatrisk'
OASIS_KEYS_XX = 'noreturn'

OASIS_KEYS_STATUS = {
    'success': {'id': OASIS_KEYS_SC, 'desc': 'Success'},
    'fail': {'id': OASIS_KEYS_FL, 'desc': 'Failure'},
    'nomatch': {'id': OASIS_KEYS_NM, 'desc': 'No match'},
    'fail_ap': {'id': OASIS_KEYS_FA, 'desc': 'Failure areaperil'},
    'fail_v': {'id': OASIS_KEYS_FV, 'desc': 'Failure vulnerability'},
    'notatrisk': {'id': OASIS_KEYS_NR, 'desc': 'Modelled but not at risk'},
    'noreturn': {'id': OASIS_KEYS_XX, 'desc': 'No key returned from lookup'}
}

OASIS_UNKNOWN_ID = -1

# list of statuses classed as "modelled"
OASIS_KEYS_STATUS_MODELLED = [OASIS_KEYS_SC,OASIS_KEYS_NR]

OASIS_TASK_PN = 'PENDING'
OASIS_TASK_RN = 'RUNNING'
OASIS_TASK_SC = 'SUCCESS'
OASIS_TASK_FL = 'FAILURE'

OASIS_TASK_STATUS = {
    'pending': {'id': OASIS_TASK_PN, 'desc': 'Pending'},
    'running': {'id': OASIS_TASK_RN, 'desc': 'Running'},
    'success': {'id': OASIS_TASK_SC, 'desc': 'Success'},
    'failure': {'id': OASIS_TASK_FL, 'desc': 'Failure'}
}
