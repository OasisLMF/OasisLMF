__all__ = [
    'OASIS_KEYS_STATUS',
    'OASIS_TASK_STATUS'
]


OASIS_KEYS_SC = 'success'
OASIS_KEYS_FL = 'fail'
OASIS_KEYS_NM = 'nomatch'

OASIS_KEYS_STATUS = {
    'success': {'id': OASIS_KEYS_SC, 'desc': 'Success'},
    'fail': {'id': OASIS_KEYS_FL, 'desc': 'Failure'},
    'nomatch': {'id': OASIS_KEYS_NM, 'desc': 'No match'}
}

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
