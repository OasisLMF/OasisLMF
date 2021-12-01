"""
TODO: should I check values are valid in the financial structure percentage not between 0 and 1 (ex: deductible, limit ...)
TODO: validate max and min ded implementation
TODO: It seems that if a policy with share is used, subsequent policy using min or max deductible will be wrong
     so it make no sense to compute deductible, over_limit, under_limit

"""


from numba import njit


class UnknownCalcrule(Exception):
    pass

@njit(cache=True)
def min2(a, b):
    return a if a < b else b


@njit(cache=True, fastmath=True)
def calcrule_1(policy, loss_out, loss_in):
    """
    Deductible and limit
    """
    lim = policy['limit_1'] + policy['deductible_1']
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            loss_out[i] = 0
        elif loss_in[i] <= lim:
            loss_out[i] = loss_in[i] - policy['deductible_1']
        else:
            loss_out[i] = policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_2(policy, loss_out, loss_in):
    """
    Deductible, attachment, limit and share

    """
    ded_att = policy['deductible_1'] + policy['attachment_1']
    lim = policy['limit_1'] + ded_att
    maxi = policy['limit_1'] * policy['share_1']
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= ded_att:
            loss_out[i] = 0
        elif loss_in[i] <= lim:
            loss_out[i] = (loss_in[i] - ded_att) * policy['share_1']
        else:
            loss_out[i] = maxi


@njit(cache=True, fastmath=True)
def calcrule_3(policy, loss_out, loss_in):
    """
    Franchise deductible and limit
    """
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            loss_out[i] = 0
        elif loss_in[i] <= policy['limit_1']:
            loss_out[i] = loss_in[i]
        else:
            loss_out[i] = policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_5(policy, loss_out, loss_in):
    """
    Deductible and limit as a proportion of loss
    """
    effective_deductible = loss_in * policy['deductible_1']
    effective_limit = loss_in * policy['limit_1']
    if policy['deductible_1'] + policy['limit_1'] >= 1 : # always under limit
        for i in range(loss_in.shape[0]):
            loss_out[i] = loss_in[i] - effective_deductible[i]

    else: # always over limit
        loss_out[:] = effective_limit


@njit(cache=True, fastmath=True)
def calcrule_12(policy, loss_out, loss_in):
    """
    Deductible only
    """
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            loss_out[i] = 0
        else:
            loss_out[i] = loss_in[i] - policy['deductible_1']


@njit(cache=True, fastmath=True)
def calcrule_14(policy, loss_out, loss_in):
    """
    Limit only
    """
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['limit_1']:
            loss_out[i] = loss_in[i]
        else:
            loss_out[i] = policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_15(policy, loss_out, loss_in):
    """
    deductible and limit % loss
    """
    effective_limit = policy['deductible_1']/(1 - policy['limit_1'])
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            loss_out[i] = 0
        elif loss_in[i] <= effective_limit:
            loss_out[i] = loss_in[i] - policy['deductible_1']
        else:
            loss_out[i] = loss_in[i] * policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_16(policy, loss_out, loss_in):
    """
    deductible % loss
    """
    loss_out[:] = loss_in * (1 - policy['deductible_1'])


@njit(cache=True, fastmath=True)
def calcrule_17(policy, loss_out, loss_in):
    """
    deductible % loss with attachment, limit and share
    """
    if policy['deductible_1'] >= 1:
        loss_out.fill(0)
    else:
        post_ded_attachment = policy['attachment_1'] / (1- policy['deductible_1'])
        post_ded_attachment_limit = (policy['attachment_1'] + policy['limit_1']) / (1 - policy['deductible_1'])
        maxi = policy['limit_1'] * policy['share_1']
        for i in range(loss_in.shape[0]):
            if loss_in[i] <= post_ded_attachment:
                loss_out[i] = 0
            elif loss_in[i] <= post_ded_attachment_limit:
                loss_out[i] = (loss_in[i] * (1 - policy['deductible_1']) - policy['attachment_1']) * policy['share_1']
            else:
                loss_out[i] = maxi


@njit(cache=True, fastmath=True)
def calcrule_20(policy, loss_out, loss_in):
    """
    reverse franchise deductible
    """
    for i in range(loss_in.shape[0]):
        if loss_in[i] > policy['deductible_1']:
            loss_out[i] = 0
        else:
            loss_out[i] = loss_in[i]


@njit(cache=True, fastmath=True)
def calcrule_22(policy, loss_out, loss_in):
    """
    reinsurance % ceded, limit and % placed
    """
    if policy['share_1'] == 0:
        loss_out.fill(0)
    else:
        pre_share_limit = policy['limit_1'] / policy['share_1']
        all_share = policy['share_1'] * policy['share_2'] * policy['share_3']
        maxi = policy['limit_1'] * policy['share_2'] * policy['share_3']
        for i in range(loss_in.shape[0]):
            if loss_in[i] <= pre_share_limit:
                loss_out[i] = loss_in[i] * all_share
            else:
                loss_out[i] = maxi


@njit(cache=True, fastmath=True)
def calcrule_23(policy, loss_out, loss_in):
    """
    reinsurance limit and % placed
    """
    all_share = policy['share_2'] * policy['share_3']
    maxi = policy['limit_1'] * all_share
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['limit_1']:
            loss_out[i] = loss_in[i] * all_share
        else:
            loss_out[i] = maxi


@njit(cache=True, fastmath=True)
def calcrule_24(policy, loss_out, loss_in):
    """
    reinsurance excess terms
    """
    if policy['share_1'] == 0:
        loss_out.fill(0)
    else:
        pre_share_attachment = policy['attachment_1'] / policy['share_1']
        pre_share_attachment_limit = (policy['limit_1'] + policy['attachment_1']) / policy['share_1']
        attachment_share = policy['attachment_1'] * policy['share_2'] * policy['share_3']
        all_share = policy['share_1'] * policy['share_2'] * policy['share_3']
        maxi = policy['limit_1'] * policy['share_2'] * policy['share_3']
        for i in range(loss_in.shape[0]):
            if loss_in[i] <= pre_share_attachment:
                loss_out[i] = 0
            elif loss_in[i] <= pre_share_attachment_limit:
                loss_out[i] = loss_in[i] * all_share - attachment_share
            else:
                loss_out[i] = maxi


@njit(cache=True, fastmath=True)
def calcrule_25(policy, loss_out, loss_in):
    """
    reinsurance proportional terms
    """
    loss_out[:] = loss_in * (policy['share_1'] * policy['share_2'] * policy['share_3'])


@njit(cache=True, fastmath=True)
def calcrule_28(policy, loss_out, loss_in):
    """
    % loss step payout
    """
    if policy['step_id'] == 1:
        loss_out.fill(0)
    for i in range(loss_in.shape[0]):
        if policy['trigger_start'] <= loss_in[i] < policy['trigger_end']:
            loss = max(policy['payout_start'] * loss_in[i] - policy['deductible_1'], 0)
            loss_out[i] = (loss + min(loss * policy['scale_2'], policy['limit_2'])) * policy['scale_1']


@njit(cache=True, fastmath=True)
def calcrule_281(policy, loss_out, loss_in):
    """
    conditional coverage
    """
    if policy['step_id'] == 1:
        loss_out.fill(0)
    for i in range(loss_in.shape[0]):
        if policy['trigger_start'] <= loss_in[i] < policy['trigger_end']:
            loss_out[i] += min(loss_out[i] * policy['scale_2'], policy['limit_2']) * policy['scale_1']


@njit(cache=True, fastmath=True)
def calcrule_32(policy, loss_out, loss_in):
    """
    monetary amount trigger and % loss step payout with limit
    """
    if policy['step_id'] == 1:
        loss_out.fill(0)
    for i in range(loss_in.shape[0]):
        if policy['trigger_start'] <= loss_in[i]:
            loss = min(policy['payout_start'] * loss_in[i], policy['limit_1'])
            loss_out[i] += (loss + min(loss * policy['scale_2'], policy['limit_2'])) * policy['scale_1']


@njit(cache=True, fastmath=True)
def calcrule_33(policy, loss_out, loss_in):
    """
    deductible % loss with limit

    """
    if policy['deductible_1'] >= 1:
        loss_out.fill(0)
    else:
        post_ded_limit = policy['limit_1'] / (1 - policy['deductible_1'])
        for i in range(loss_in.shape[0]):
            if loss_in[i] <= post_ded_limit:
                loss_out[i] = loss_in[i] * (1 - policy['deductible_1'])
            else:
                loss_out[i] = policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_34(policy, loss_out, loss_in):
    """
    deductible with attachment and share

    TODO: compare to the cpp, as there is shares, deductible won't be use later on so no need to compute it
    """
    ded_att = policy['deductible_1'] + policy['attachment_1']
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= ded_att:
            loss_out[i] = 0
        else:
            loss_out[i] = (loss_in[i] - ded_att) * policy['share_1']


@njit(cache=True, fastmath=True)
def calcrule_37(policy, loss_out, loss_in):
    """
    % loss step payout
    """
    if policy['step_id'] == 1:
        loss_out.fill(0)
    for i in range(loss_in.shape[0]):
        if policy['trigger_start'] <= loss_in[i] < policy['trigger_end']:
            loss = min(max(policy['payout_start'] * loss_in[i] - policy['deductible_1'], 0), policy['limit_1'])
            loss_out[i] = (loss + min(loss * policy['scale_2'], policy['limit_2'])) * policy['scale_1']


@njit(cache=True, fastmath=True)
def calcrule_38(policy, loss_out, loss_in):
    """
    conditional coverage
    """
    if policy['step_id'] == 1:
        loss_out.fill(0)
    for i in range(loss_in.shape[0]):
        if policy['trigger_start'] <= loss_in[i] < policy['trigger_end']:
            loss_out[i] = (loss_out[i] + min(loss_out[i] * policy['scale_2'], policy['limit_2'])) * (policy['scale_1'])


@njit(cache=True)
def calc(policy, loss_out, loss_in, stepped):
    if policy['calcrule_id'] == 1:
        calcrule_1(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 2:
        calcrule_2(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 3:
        calcrule_3(policy, loss_out, loss_in)
    # calcrule_4 (deductible % TIV and limit) is redirected to 1 when building financial structure
    elif policy['calcrule_id'] == 5:
        calcrule_5(policy, loss_out, loss_in)
    # calcrule_6 (deductible % TIV) is redirected to 12 when building financial structure
    # calcrule_9 (limit with deductible % limit) is redirected to 1 when building financial structure
    elif policy['calcrule_id'] == 12:
        calcrule_12(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 14:
        calcrule_14(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 15:
        calcrule_15(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 16:
        calcrule_16(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 17:
        calcrule_17(policy, loss_out, loss_in)
    # calcrule_18 (deductible % tiv with attachment, limit and share) is redirected to 2 when building financial structure
    elif policy['calcrule_id'] == 20:
        calcrule_20(policy, loss_out, loss_in)
    # calcrule_21 (deductible % tiv with min and max deductible) is redirected to 13 when building financial structure
    elif policy['calcrule_id'] == 22:
        calcrule_22(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 23:
        calcrule_23(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 24:
        calcrule_24(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 25:
        calcrule_25(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 33:
        calcrule_33(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 34:
        calcrule_34(policy, loss_out, loss_in)
    elif policy['calcrule_id'] == 100:
        loss_out[:] = loss_in
    elif stepped is not None:
        if policy['calcrule_id'] == 28:
            calcrule_28(policy, loss_out, loss_in)
        elif policy['calcrule_id'] == 32:
            calcrule_32(policy, loss_out, loss_in)
        elif policy['calcrule_id'] == 37:
            calcrule_37(policy, loss_out, loss_in)
        elif policy['calcrule_id'] == 38:
            calcrule_38(policy, loss_out, loss_in)
        else:
            raise UnknownCalcrule()
    else:
        raise UnknownCalcrule()
