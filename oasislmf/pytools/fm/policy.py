from numba import njit

class UnknownCalcrule(Exception):
    pass


@njit(cache=True, fastmath=True)
def calcrule_1(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    Deductible and limit
    """
    lim = policy['limit_1'] + policy['deductible_1']

    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            over_limit[i] = 0
            under_limit[i] = policy['limit_1']
            deductible[i] += loss_in[i]
            loss_out[i] = 0
        elif loss_in[i] <= lim:
            over_limit[i] = 0
            under_limit[i] = lim - loss_in[i]
            deductible[i] += policy['deductible_1']
            loss_out[i] = loss_in[i] - policy['deductible_1']
        else:
            over_limit[i] = loss_in[i] - lim
            under_limit[i] = 0
            deductible[i] += policy['deductible_1']
            loss_out[i] = policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_2(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    Deductible, attachment, limit and share

    """
    ded_att = policy['deductible_1'] + policy['attachment_1']
    lim = policy['limit_1'] + ded_att
    maxi = policy['limit_1'] * policy['share_1']
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            over_limit[i] = 0
            under_limit[i] = maxi
            deductible[i] += loss_in[i]
            loss_out[i] = 0
        elif loss_in[i] <= ded_att:
            over_limit[i] = 0
            under_limit[i] = maxi
            deductible[i] += policy['deductible_1']
            loss_out[i] = 0
        elif loss_in[i] <= lim:
            over_limit[i] = 0
            under_limit[i] = (lim - loss_in[i]) * policy['share_1']
            deductible[i] += policy['deductible_1']
            loss_out[i] = (loss_in[i] - ded_att) * policy['share_1']
        else:
            over_limit[i] = (loss_in[i] - lim) * policy['share_1']
            under_limit[i] = 0
            deductible[i] += policy['deductible_1']
            loss_out[i] = maxi


@njit(cache=True, fastmath=True)
def calcrule_3(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    Franchise deductible and limit
    """
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            over_limit[i] = 0
            under_limit[i] = policy['limit_1']
            deductible[i] += loss_in[i]
            loss_out[i] = 0
        elif loss_in[i] <= policy['limit_1']:
            over_limit[i] = 0
            under_limit[i] = policy['limit_1'] - loss_in[i]
            # no deductible change
            loss_out[i] = loss_in[i]
        else:
            over_limit[i] = loss_in[i] - policy['limit_1']
            under_limit[i] = 0
            # no deductible change
            loss_out[i] = policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_4(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible % TIV and limit
    """

    maxi = policy['limit_1'] / (1 - policy['deductible_1'])
    for i in range(loss_in.shape[0]):
        effective_deductible = loss_in[i] * policy['deductible_1']
        deductible[i] += effective_deductible
        if loss_in[i] <= maxi:
            loss_out[i] = loss_in[i] - effective_deductible
            over_limit[i] = 0
            under_limit[i] = policy['limit_1'] - loss_out[i]
        else:
            over_limit[i] = loss_in[i] - effective_deductible - policy['limit_1']
            under_limit[i] = 0
            loss_out[i] = policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_5(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    Deductible and limit as a proportion of loss
    """
    effective_deductible = loss_in * policy['deductible_1']
    effective_limit = loss_in * policy['limit_1']
    deductible += effective_deductible
    if policy['deductible_1'] + policy['limit_1'] >= 1 : # always under limit
        loss_out = loss_in - effective_deductible
        over_limit = 0
        under_limit = effective_limit - loss_out
    else: # always over limit
        loss_out = effective_limit
        over_limit = loss_in - effective_deductible - effective_limit
        under_limit = 0



@njit(cache=True, fastmath=True)
def calcrule_6(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible % TIV
    """
    effective_deductible = loss_in * policy['deductible_1']
    deductible += effective_deductible
    loss_out = loss_in - effective_deductible
    over_limit = 0
    under_limit = 0


@njit(cache=True, fastmath=True)
def calcrule_7(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    limit and maximum deductible
    """
    pass



@njit(cache=True)
def calc(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    if policy['calcrule_id'] == 1:
        calcrule_1(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 2:
        calcrule_2(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 3:
        calcrule_3(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 4:
        calcrule_4(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 5:
        calcrule_5(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 6:
        calcrule_6(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 100:
        loss_out = loss_in
    else:
        raise UnknownCalcrule()
