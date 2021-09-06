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


@njit(cache=True)
def deductible_over_max(i, loss_out, loss_in, deductible, over_limit, under_limit, max_deductible):
    """
    deductible is over maximum deductible, we reduce the loss, therefore increase the loss up to under_limit

    under limit is always the minimum between the limit - loss and  the sub_node under_limit + the applied deductible
    so we are sure that if deductible[i] > max_ded_left, we are sure that under_limit is the good cap
    we are sure that is there is no sub node with limit loss_delta < under_limit

    loss delta can be negative so in this case we have to be careful it is not bigger than loss_in

    """
    loss_delta = deductible[i] - max_deductible
    if loss_delta > under_limit[i]:
        loss_out[i] = loss_in[i] + under_limit[i]
        over_limit[i] += loss_delta - under_limit[i]
        deductible[i] = max_deductible
        under_limit[i] = 0
    elif loss_in[i] >= -loss_delta :
        loss_out[i] = loss_in[i] + loss_delta
        under_limit[i] -= loss_delta
        deductible[i] = max_deductible
    else:
        loss_out[i] = 0
        deductible[i] += loss_in[i]
        under_limit[i] += loss_in[i]


@njit(cache=True)
def deductible_under_min(i, loss_out, loss_in, effective_deductible, over_limit, under_limit, min_deductible, deductible):
    """
    Deductible is under the minimum, we raise the deductible from over_limit first then loss if over_limit is not enough

    """
    loss_delta = min_deductible - deductible - effective_deductible[i]
    if loss_delta <= over_limit[i]:  # we have enough over_limit to cover loss_delta
        if loss_in[i] > deductible:  # we have enough loss to cover deductible
            loss_out[i] = loss_in[i]
            over_limit[i] -= loss_delta
            effective_deductible[i] = min_deductible
        elif (over_limit[i] - loss_delta) + loss_in[i] > deductible:  # not enough loss, we also reduce the over_limit
            loss_out[i] = 0
            over_limit[i] -= loss_delta + deductible
            effective_deductible[i] = min_deductible
            under_limit[i] += loss_in[i]
        else:
            effective_deductible[i] += loss_in[i] + over_limit[i]
            loss_out[i] = 0
            over_limit[i] = 0
            under_limit[i] += loss_in[i]

    else:
        loss_not_over_limit = loss_delta - over_limit[i]
        if loss_in[i] > loss_not_over_limit + deductible:  # we have enough loss after we use over_limit pool
            loss_out[i] = loss_in[i] - loss_not_over_limit - deductible
            over_limit[i] = 0
            effective_deductible[i] = min_deductible
            under_limit[i] += loss_not_over_limit
        else:
            loss_out[i] = 0
            effective_deductible[i] += loss_in[i] + over_limit[i]
            under_limit[i] += loss_in[i]
            over_limit[i] = 0


@njit(cache=True, fastmath=True)
def calcrule_1(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    Deductible and limit
    """
    lim = policy['limit_1'] + policy['deductible_1']
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            under_limit[i] = min2(under_limit[i] + loss_in[i], policy['limit_1'])
            deductible[i] += loss_in[i]
            loss_out[i] = 0
        elif loss_in[i] <= lim:
            under_limit[i] = min2(under_limit[i] + policy['deductible_1'], lim - loss_in[i])
            deductible[i] += policy['deductible_1']
            loss_out[i] = loss_in[i] - policy['deductible_1']
        else:
            over_limit[i] += loss_in[i] - lim
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
        if loss_in[i] <= ded_att:
            loss_out[i] = 0
        elif loss_in[i] <= lim:
            loss_out[i] = (loss_in[i] - ded_att) * policy['share_1']
        else:
            loss_out[i] = maxi


@njit(cache=True, fastmath=True)
def calcrule_3(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    Franchise deductible and limit
    """
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            under_limit[i] = min2(under_limit[i] + loss_in[i], policy['limit_1'])
            deductible[i] += loss_in[i]
            loss_out[i] = 0
        elif loss_in[i] <= policy['limit_1']:
            under_limit[i] = min2(under_limit[i], policy['limit_1'] - loss_in[i])
            loss_out[i] = loss_in[i]
        else:
            under_limit[i] = 0
            over_limit[i] += loss_in[i] - policy['limit_1']
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
        for i in range(loss_in.shape[0]):
            loss_out[i] = loss_in[i] - effective_deductible[i]
            under_limit[i] = min2(effective_limit[i] - loss_out[i], under_limit[i] + effective_deductible[i])

    else: # always over limit
        loss_out[:] = effective_limit
        over_limit += loss_in - effective_deductible - effective_limit
        under_limit[:] = 0


@njit(cache=True, fastmath=True)
def calcrule_7(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible, minimum and maximum deductible, with limit
    """

    max_ded_left = policy['deductible_3'] - policy['deductible_1']
    min_ded_left = policy['deductible_2'] - policy['deductible_1']

    for i in range(loss_in.shape[0]):
        if deductible[i] > max_ded_left:
            deductible_over_max(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_3'])
        elif deductible[i] < min_ded_left:
            deductible_under_min(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_2'], policy['deductible_1'])
        else:
            if loss_in[i] > policy['deductible_1']:
                loss_out[i] = loss_in[i] - policy['deductible_1']
                deductible[i] += policy['deductible_1']
                under_limit[i] += policy['deductible_1']
            else:
                loss_out[i] = 0
                deductible[i] += loss_in[i]
                under_limit[i] += loss_in[i]

        if loss_out[i] > policy['limit_1']:
            over_limit[i] += loss_out[i] - policy['limit_1']
            under_limit[i] = 0
            loss_out[i] = policy['limit_1']
        else:
            under_limit[i] = min2(policy['limit_1'] - loss_out[i], under_limit[i])


@njit(cache=True, fastmath=True)
def calcrule_8(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible and minimum deductible, with limit
    """
    min_ded_left = policy['deductible_2'] - policy['deductible_1']
    for i in range(loss_in.shape[0]):
        if deductible[i] < min_ded_left:
            deductible_under_min(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_2'], policy['deductible_1'])
        else:
            if loss_in[i] > policy['deductible_1']:
                loss_out[i] = loss_in[i] - policy['deductible_1']
                deductible[i] += policy['deductible_1']
                under_limit[i] += policy['deductible_1']
            else:
                loss_out[i] = 0
                deductible[i] += loss_in[i]
                under_limit[i] += loss_in[i]

        if loss_out[i] > policy['limit_1']:
            over_limit[i] += loss_out[i] - policy['limit_1']
            under_limit[i] = 0
            loss_out[i] = policy['limit_1']
        else:
            under_limit[i] = min2(policy['limit_1'] - loss_out[i], under_limit[i])


@njit(cache=True, fastmath=True)
def calcrule_10(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible and maximum deductible
    """
    max_ded_left = policy['deductible_3'] - policy['deductible_1']

    for i in range(loss_in.shape[0]):
        if deductible[i] > max_ded_left:
            deductible_over_max(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_3'])
        else:
            if loss_in[i] > policy['deductible_1']:
                loss_out[i] = loss_in[i] - policy['deductible_1']
                deductible[i] += policy['deductible_1']
                under_limit[i] += policy['deductible_1']
            else:
                loss_out[i] = 0
                deductible[i] += loss_in[i]
                under_limit[i] += loss_in[i]


@njit(cache=True, fastmath=True)
def calcrule_11(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible and minimum deductible
    """
    min_ded_left = policy['deductible_2'] - policy['deductible_1']

    for i in range(loss_in.shape[0]):
        if deductible[i] < min_ded_left:
            deductible_under_min(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_2'], policy['deductible_1'])
        else:
            if loss_in[i] > policy['deductible_1']:
                loss_out[i] = loss_in[i] - policy['deductible_1']
                deductible[i] += policy['deductible_1']
                under_limit[i] += policy['deductible_1']
            else:
                loss_out[i] = 0
                deductible[i] += loss_in[i]
                under_limit[i] += loss_in[i]


@njit(cache=True, fastmath=True)
def calcrule_12(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    Deductible only
    """
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            under_limit[i] += loss_in[i]
            deductible[i] += loss_in[i]
            loss_out[i] = 0
        else:
            under_limit[i] += policy['deductible_1']
            deductible[i] += policy['deductible_1']
            loss_out[i] = loss_in[i] - policy['deductible_1']


@njit(cache=True, fastmath=True)
def calcrule_13(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible, minimum and maximum deductible
    """

    max_ded_left = policy['deductible_3'] - policy['deductible_1']
    min_ded_left = policy['deductible_2'] - policy['deductible_1']

    for i in range(loss_in.shape[0]):
        if deductible[i] > max_ded_left:
            deductible_over_max(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_3'])
        elif deductible[i] < min_ded_left:
            deductible_under_min(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_2'], policy['deductible_1'])
        else:
            if loss_in[i] > policy['deductible_1']:
                loss_out[i] = loss_in[i] - policy['deductible_1']
                deductible[i] += policy['deductible_1']
                under_limit[i] += policy['deductible_1']
            else:
                loss_out[i] = 0
                deductible[i] += loss_in[i]
                under_limit[i] += loss_in[i]


@njit(cache=True, fastmath=True)
def calcrule_14(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    Limit only
    """
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['limit_1']:
            under_limit[i] = min2(policy['limit_1'] - loss_in[i], under_limit[i])
            loss_out[i] = loss_in[i]
        else:
            over_limit[i] += loss_in[i] - policy['limit_1']
            under_limit[i] = 0
            loss_out[i] = policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_15(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible and limit % loss
    """
    effective_limit = policy['deductible_1']/(1 - policy['limit_1'])
    for i in range(loss_in.shape[0]):
        if loss_in[i] <= policy['deductible_1']:
            under_limit[i] = min2(effective_limit, under_limit[i] + loss_in[i])
            loss_out[i] = 0
            deductible[i] += loss_in[i]
        elif loss_in[i] <= effective_limit:
            under_limit[i] = min2(effective_limit - loss_in[i], under_limit[i] + policy['deductible_1'])
            loss_out[i] = loss_in[i] - policy['deductible_1']
            deductible[i] += policy['deductible_1']
        else:
            loss_out[i] = loss_in[i] * policy['limit_1']
            deductible[i] += policy['deductible_1']
            over_limit[i] += loss_in[i] - loss_out[i] - policy['deductible_1']
            under_limit[i] = 0


@njit(cache=True, fastmath=True)
def calcrule_16(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible % loss
    """
    effective_deductible = loss_in * policy['deductible_1']
    deductible += effective_deductible
    under_limit += effective_deductible
    loss_out[:] = loss_in - effective_deductible


@njit(cache=True, fastmath=True)
def calcrule_17(policy, loss_out, loss_in, deductible, over_limit, under_limit):
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
            effective_deductible = loss_in[i] * policy['deductible_1']
            if loss_in[i] <= post_ded_attachment:
                loss_out[i] = 0
            elif loss_in[i] <= post_ded_attachment_limit:
                loss_out[i] = (loss_in[i] - effective_deductible - policy['attachment_1']) * policy['share_1']
            else:
                loss_out[i] = maxi


@njit(cache=True, fastmath=True)
def calcrule_19(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible % loss with min and/or max deductible

    TODO: check if we can assume 0 <= policy['deductible_1'] <= 1
    """

    for i in range(loss_in.shape[0]):
        effective_deductible = loss_in[i] * policy['deductible_1']
        if effective_deductible + deductible[i] > policy['deductible_3'] > 0:
            deductible_over_max(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_3'])
        elif effective_deductible + deductible[i] < policy['deductible_2']:
            deductible_under_min(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_2'], policy['deductible_1'])
        else:
            if loss_in[i] > effective_deductible:
                loss_out[i] = loss_in[i] - effective_deductible
                deductible[i] += effective_deductible
                under_limit[i] += effective_deductible
            else:
                loss_out[i] = 0
                deductible[i] += loss_in[i]
                under_limit[i] += loss_in[i]


@njit(cache=True, fastmath=True)
def calcrule_20(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    reverse franchise deductible
    """
    for i in range(loss_in.shape[0]):
        if loss_in[i] > policy['deductible_1']:
            loss_out[i] = 0
        else:
            loss_out[i] = loss_in[i]


@njit(cache=True, fastmath=True)
def calcrule_22(policy, loss_out, loss_in, deductible, over_limit, under_limit):
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
def calcrule_23(policy, loss_out, loss_in, deductible, over_limit, under_limit):
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
def calcrule_24(policy, loss_out, loss_in, deductible, over_limit, under_limit):
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
def calcrule_25(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    reinsurance proportional terms
    """
    loss_out[:] = loss_in * (policy['share_1'] * policy['share_2'] * policy['share_3'])


@njit(cache=True, fastmath=True)
def calcrule_26(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible % loss with min and/or max deductible and limit

    TODO: check if we can assume 0 <= policy['deductible_1'] <= 1
    """

    for i in range(loss_in.shape[0]):
        effective_deductible = loss_in[i] * policy['deductible_1']
        if effective_deductible + deductible[i] > policy['deductible_3'] > 0:
            deductible_over_max(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_3'])
        elif effective_deductible + deductible[i] < policy['deductible_2']:
            deductible_under_min(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_2'], effective_deductible)
        else:
            if loss_in[i] > effective_deductible:
                loss_out[i] = loss_in[i] - effective_deductible
                deductible[i] += effective_deductible
                under_limit[i] += effective_deductible
            else:
                loss_out[i] = 0
                deductible[i] += loss_in[i]
                under_limit[i] += loss_in[i]

        if loss_out[i] > policy['limit_1']:
            over_limit[i] += loss_out[i] - policy['limit_1']
            under_limit[i] = 0
            loss_out[i] = policy['limit_1']
        else:
            under_limit[i] = min2(policy['limit_1'] - loss_out[i], under_limit[i])


@njit(cache=True, fastmath=True)
def calcrule_27(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    step payout with limit
    """
    for i in range(loss_in.shape[0]):
        if (0 < loss_in[i] or 0 < deductible[i]) and policy['trigger_start'] <= loss_in[i] < policy['trigger_end']:
            loss_out[i] += policy['payout_start']


@njit(cache=True, fastmath=True)
def calcrule_28(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    % loss step payout
    """
    for i in range(loss_in.shape[0]):
        if policy['trigger_start'] <= loss_in[i] < policy['trigger_end']:
            loss = max(policy['payout_start'] * loss_in[i] - policy['deductible_1'], 0)
            loss_out[i] = (loss + min(loss * policy['scale_2'], policy['limit_2'])) * policy['scale_1']


@njit(cache=True, fastmath=True)
def calcrule_281(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    conditional coverage
    """
    for i in range(loss_in.shape[0]):
        if policy['trigger_start'] <= loss_in[i] < policy['trigger_end']:
            loss_out[i] += min(loss_out[i] * policy['scale_2'], policy['limit_2']) * policy['scale_1']


@njit(cache=True, fastmath=True)
def calcrule_32(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    monetary amount trigger and % loss step payout with limit
    """
    for i in range(loss_in.shape[0]):
        if policy['trigger_start'] <= loss_in[i]:
            loss = min(policy['payout_start'] * loss_in[i], policy['limit_1'])
            loss_out[i] += (loss + min(loss * policy['scale_2'], policy['limit_2'])) * policy['scale_1']


@njit(cache=True, fastmath=True)
def calcrule_33(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible % loss with limit

    """
    if policy['deductible_1'] >= 1:
        loss_out.fill(0)
        deductible += loss_in
    else:
        post_ded_limit = policy['limit_1'] / (1 - policy['deductible_1'])
        for i in range(loss_in.shape[0]):
            effective_deductible = loss_in[i] * policy['deductible_1']
            deductible[i] += effective_deductible
            if loss_in[i] <= post_ded_limit:
                loss_out[i] = loss_in[i] - effective_deductible
                under_limit[i] = min2(under_limit[i] + effective_deductible, policy['limit_1'] - loss_out[i])
            else:
                over_limit[i] += loss_in[i] - effective_deductible - policy['limit_1']
                under_limit[i] = 0
                loss_out[i] = policy['limit_1']


@njit(cache=True, fastmath=True)
def calcrule_34(policy, loss_out, loss_in, deductible, over_limit, under_limit):
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
def calcrule_35(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible % loss with min and/or max deductible and limit % loss

    TODO: check if we can assume 0 <= policy['deductible_1'] <= 1
    """

    for i in range(loss_in.shape[0]):
        effective_deductible = loss_in[i] * policy['deductible_1']
        if effective_deductible + deductible[i] > policy['deductible_3'] > 0:
            deductible_over_max(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_3'])
        elif effective_deductible + deductible[i] < policy['deductible_2']:
            deductible_under_min(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_2'], effective_deductible)
        else:
            if loss_in[i] > effective_deductible:
                loss_out[i] = loss_in[i] - effective_deductible
                deductible[i] += effective_deductible
                under_limit[i] += effective_deductible
            else:
                loss_out[i] = 0
                deductible[i] += loss_in[i]
                under_limit[i] += loss_in[i]
        limit = loss_in[i] * policy['limit_1']
        if loss_out[i] > limit:
            over_limit[i] += loss_out[i] - limit
            under_limit[i] = 0
            loss_out[i] = limit
        else:
            under_limit[i] = min2(limit - loss_out[i], under_limit[i])


@njit(cache=True, fastmath=True)
def calcrule_36(policy, loss_out, loss_in, deductible, over_limit, under_limit):
    """
    deductible with min and/or max deductible and limit % loss
    """

    max_ded_left = policy['deductible_3'] - policy['deductible_1']
    min_ded_left = policy['deductible_2'] - policy['deductible_1']

    for i in range(loss_in.shape[0]):
        if deductible[i] > max_ded_left > 0:
            deductible_over_max(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_3'])
        elif deductible[i] < min_ded_left:
            deductible_under_min(i, loss_out, loss_in, deductible, over_limit, under_limit, policy['deductible_2'], policy['deductible_1'])
        else:
            if loss_in[i] > policy['deductible_1']:
                loss_out[i] = loss_in[i] - policy['deductible_1']
                deductible[i] += policy['deductible_1']
                under_limit[i] += policy['deductible_1']
            else:
                loss_out[i] = 0
                deductible[i] += loss_in[i]
                under_limit[i] += loss_in[i]

        limit = loss_in[i] * policy['limit_1']
        if loss_out[i] > limit:
            over_limit[i] += loss_out[i] - limit
            under_limit[i] = 0
            loss_out[i] = limit
        else:
            under_limit[i] = min2(limit - loss_out[i], under_limit[i])


@njit(cache=True)
def calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, stepped):
    if policy['calcrule_id'] == 1:
        calcrule_1(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 2:
        calcrule_2(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 3:
        calcrule_3(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    # calcrule_4 (deductible % TIV and limit) is redirected to 1 when building financial structure
    elif policy['calcrule_id'] == 5:
        calcrule_5(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    # calcrule_6 (deductible % TIV) is redirected to 12 when building financial structure
    elif policy['calcrule_id'] == 7:
        calcrule_7(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 8:
        calcrule_8(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    # calcrule_9 (limit with deductible % limit) is redirected to 1 when building financial structure
    elif policy['calcrule_id'] == 10:
        calcrule_10(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 11:
        calcrule_11(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 12:
        calcrule_12(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 13:
        calcrule_13(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 14:
        calcrule_14(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 15:
        calcrule_15(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 16:
        calcrule_16(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 17:
        calcrule_17(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    # calcrule_18 (deductible % tiv with attachment, limit and share) is redirected to 2 when building financial structure
    elif policy['calcrule_id'] == 19:
        calcrule_19(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 20:
        calcrule_20(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    # calcrule_21 (deductible % tiv with min and max deductible) is redirected to 13 when building financial structure
    elif policy['calcrule_id'] == 22:
        calcrule_22(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 23:
        calcrule_23(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 24:
        calcrule_24(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 25:
        calcrule_25(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 26:
        calcrule_26(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif stepped is not None:
        if policy['calcrule_id'] == 27:
            calcrule_27(policy, loss_out, loss_in, deductible, over_limit, under_limit)
        elif policy['calcrule_id'] == 28:
            calcrule_28(policy, loss_out, loss_in, deductible, over_limit, under_limit)
        elif policy['calcrule_id'] == 281:
            calcrule_281(policy, loss_out, loss_in, deductible, over_limit, under_limit)
        elif policy['calcrule_id'] == 32:
            calcrule_32(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 33:
        calcrule_33(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 34:
        calcrule_34(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 35:
        calcrule_35(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 36:
        calcrule_36(policy, loss_out, loss_in, deductible, over_limit, under_limit)
    elif policy['calcrule_id'] == 100:
        loss_out[:] = loss_in
    else:
        raise UnknownCalcrule()
