from numba import njit
from .common import DEDUCTIBLE, UNDERLIMIT, OVERLIMIT


@njit(cache=True, fastmath=True)
def back_alloc_extra_a2(base_children_len, temp_children_queue, nodes_array, p,
                        node_val_len, node_sidx, sidx_indptr, sidx_indexes, sidx_val,
                        loss_in, loss_out, temp_node_loss, loss_indptr, loss_val,
                        extra, temp_node_extras, extras_indptr, extras_val):
    """
    back allocation of loss and extra to the base children,
    The function modifies in-place array loss_in and items loss and extra in loss_val and extra val
    Args:
        base_children_len: number of base children
        temp_children_queue: array of base children
        nodes_array: array of all node info
        p: profile index
        node_val_len: number of actual values in sidx
        node_sidx: sidx for this node
        sidx_indptr: index to sidx pointer
        sidx_indexes: index of sidx for nodes
        sidx_val: sidx values
        loss_in: loss before applying profile
        loss_out: loss after applying profile
        temp_node_loss: array to store loss factor
        loss_indptr: index to loss pointer
        loss_val: loss values
        extra: extra after applying profile
        temp_node_extras: extra after applying profile(dense)
        extras_indptr: index to extra pointer
        extras_val: extra values
    """
    if base_children_len == 1:  # this is a base children, we only need to assign loss_in
        loss_in[:] = loss_out
    else:
        # back allocation rules:
        # if deductible grows, deductible and loss are allocated based on loss
        # else it means it is reallocated to loss,
        #   if underlimit is still >0 then extra loss and deductible are allocated based on underlimit
        #   else                           extra loss and deductible are allocated based on deductible
        # if overlimit grows, more loss is over limit so it is reallocated based on loss
        # else it is reallocated based on overlimit
        # if underlimit grows, more loss has been deducted so we reallocate based on loss
        # else it is reallocated based on underlimit

        for i in range(node_val_len):
            diff = extra[i, DEDUCTIBLE] - temp_node_extras[p, node_sidx[i], DEDUCTIBLE]
            if diff >= 0:
                realloc = 0
                if loss_in[i] > 0:
                    temp_node_extras[p, node_sidx[i], DEDUCTIBLE] = diff / loss_in[i]
                    temp_node_loss[p, node_sidx[i]] = loss_out[i] / loss_in[i]
                else:
                    temp_node_extras[p, node_sidx[i], DEDUCTIBLE] = 0
                    temp_node_loss[p, node_sidx[i]] = 0
            else:
                realloc = diff  # to loss or to over

                if extra[i, UNDERLIMIT] > 0:
                    temp_node_extras[p, node_sidx[i], DEDUCTIBLE] = diff / temp_node_extras[p, node_sidx[i], UNDERLIMIT]
                else:
                    temp_node_extras[p, node_sidx[i], DEDUCTIBLE] = diff / temp_node_extras[p, node_sidx[i], DEDUCTIBLE]
                temp_node_loss[p, node_sidx[i]] = loss_out[i] / (loss_in[i] - diff)

            diff = extra[i, OVERLIMIT] - temp_node_extras[p, node_sidx[i], OVERLIMIT]
            if diff > 0:
                temp_node_extras[p, node_sidx[i], OVERLIMIT] = diff / (loss_in[i] - realloc)
            elif diff == 0:
                temp_node_extras[p, node_sidx[i], OVERLIMIT] = 0
            else:  # we set it to <0 to be able to check it later
                temp_node_extras[p, node_sidx[i], OVERLIMIT] = - extra[i, OVERLIMIT] / temp_node_extras[
                    p, node_sidx[i], OVERLIMIT]

            diff = extra[i, UNDERLIMIT] - temp_node_extras[p, node_sidx[i], UNDERLIMIT]
            if diff > 0:
                temp_node_extras[p, node_sidx[i], UNDERLIMIT] = diff / loss_in[i]
            elif diff == 0:
                temp_node_extras[p, node_sidx[i], UNDERLIMIT] = 0
            else:  # we set it to <0 to be able to check it later
                temp_node_extras[p, node_sidx[i], UNDERLIMIT] = - extra[i, UNDERLIMIT] / temp_node_extras[
                    p, node_sidx[i], UNDERLIMIT]

            loss_in[i] = loss_out[i]

        for base_child_i in range(base_children_len):
            child = nodes_array[temp_children_queue[base_child_i]]

            child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
            child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
            child_val_len = child_sidx_end - child_sidx_start

            child_sidx = sidx_val[child_sidx_start: child_sidx_end]
            child_loss = loss_val[loss_indptr[child['loss'] + p]: loss_indptr[child['loss'] + p] + child_val_len]
            child_extra = extras_val[
                extras_indptr[child['extra'] + p]: extras_indptr[child['extra'] + p] + child_val_len]

            for i in range(child_val_len):
                if temp_node_extras[p, child_sidx[i], DEDUCTIBLE] < 0:  # realloc loss
                    if temp_node_extras[p, child_sidx[i], UNDERLIMIT] == 0:
                        realloc = temp_node_extras[p, child_sidx[i], DEDUCTIBLE] * child_extra[i, DEDUCTIBLE]
                    else:
                        realloc = temp_node_extras[p, child_sidx[i], DEDUCTIBLE] * child_extra[i, UNDERLIMIT]
                    if temp_node_extras[p, child_sidx[i], OVERLIMIT] >= 0:
                        child_extra[i, OVERLIMIT] = child_extra[i, OVERLIMIT] + temp_node_extras[
                            p, child_sidx[i], OVERLIMIT] * (child_loss[i] - realloc)
                    else:
                        child_extra[i, OVERLIMIT] = - temp_node_extras[p, child_sidx[i], OVERLIMIT] * child_extra[
                            i, OVERLIMIT]

                    child_loss[i] = (child_loss[i] - realloc) * temp_node_loss[p, child_sidx[i]]
                    child_extra[i, DEDUCTIBLE] = child_extra[i, DEDUCTIBLE] + realloc
                    child_extra[i, UNDERLIMIT] = - temp_node_extras[p, child_sidx[i], UNDERLIMIT] * child_extra[
                        i, UNDERLIMIT]

                else:
                    if temp_node_extras[p, child_sidx[i], OVERLIMIT] >= 0:
                        child_extra[i, OVERLIMIT] = child_extra[i, OVERLIMIT] + temp_node_extras[
                            p, child_sidx[i], OVERLIMIT] * child_loss[i]
                    else:
                        child_extra[i, OVERLIMIT] = - temp_node_extras[p, child_sidx[i], OVERLIMIT] * child_extra[
                            i, OVERLIMIT]

                    if temp_node_extras[p, child_sidx[i], UNDERLIMIT] >= 0:
                        child_extra[i, UNDERLIMIT] = child_extra[i, UNDERLIMIT] + temp_node_extras[
                            p, child_sidx[i], UNDERLIMIT] * child_loss[i]
                    else:
                        child_extra[i, UNDERLIMIT] = - temp_node_extras[p, child_sidx[i], UNDERLIMIT] * child_extra[
                            i, UNDERLIMIT]

                    child_extra[i, DEDUCTIBLE] = child_extra[i, DEDUCTIBLE] + temp_node_extras[
                        p, child_sidx[i], DEDUCTIBLE] * child_loss[i]
                    child_loss[i] = child_loss[i] * temp_node_loss[p, child_sidx[i]]
            # print('ba', child['level_id'], child['agg_id'], p, loss_indptr[child['loss'] + p], child_loss[0], temp_node_loss[p, -3],
            # extras_indptr[child['extra'] + p], child_extra[0])


@njit(cache=True, fastmath=True)
def back_alloc_a2(base_children_len, temp_children_queue, nodes_array, p,
                  node_val_len, node_sidx, sidx_indptr, sidx_indexes, sidx_val,
                  loss_in, loss_out, temp_node_loss, loss_indptr, loss_val):
    """
    back allocation of loss to the base children. The function modifies in-place array loss_in and items loss in loss_val.
    Args:
        base_children_len: number of base children
        temp_children_queue: array of base children
        nodes_array: array of all node info
        p: profile index
        node_val_len: number of actual values in sidx
        node_sidx: sidx for this node
        sidx_indptr: index to sidx pointer
        sidx_indexes: index of sidx for nodes
        sidx_val: sidx values
        loss_in: loss before applying profile
        loss_out: loss after applying profile
        temp_node_loss: array to store loss factor
        loss_indptr: index to loss pointer
        loss_val: loss values

    """
    if base_children_len == 1:
        loss_in[:] = loss_out
    else:
        for i in range(node_val_len):
            if loss_out[i]:
                temp_node_loss[p, node_sidx[i]] = loss_out[i] / loss_in[i]
            else:
                temp_node_loss[p, node_sidx[i]] = 0
            loss_in[i] = loss_out[i]

        for base_child_i in range(base_children_len):
            child = nodes_array[temp_children_queue[base_child_i]]

            child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
            child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
            child_val_len = child_sidx_end - child_sidx_start

            child_sidx = sidx_val[child_sidx_start: child_sidx_end]
            child_loss = loss_val[loss_indptr[child['loss'] + p]: loss_indptr[child['loss'] + p] + child_val_len]

            for i in range(child_val_len):
                child_loss[i] = child_loss[i] * temp_node_loss[p, child_sidx[i]]
            # print('ba', child['level_id'], child['agg_id'], p, loss_indptr[child['loss'] + p], child_loss[0], temp_node_loss[p, -3])


@njit(cache=True, fastmath=True)
def back_alloc_layer(layer_len, node_val_len, node_loss_indptr,
                     loss_in, loss_out, loss_indptr, loss_val,
                     temp_node_loss_layer_ba):
    """
    Args:
        layer_len: number of layers in the node
        node_val_len: number of actual values in sidx
        node_loss_indptr: index of the loss pointer of the node
        loss_in: loss before applying profile
        loss_out: loss after applying profile
        loss_indptr: index to loss pointer
        loss_val: loss values
        temp_node_loss_layer_ba: sparse array to loss after layer back alloc
    """
    for i in range(node_val_len):
        if loss_out[i]:
            loss_factor = loss_out[i] / loss_in[i]
        else:
            loss_factor = 0

        for l in range(layer_len):
            layer_loss_indptr = loss_indptr[node_loss_indptr + l]
            temp_node_loss_layer_ba[l, i] = loss_val[layer_loss_indptr + i] * loss_factor


@njit(cache=True, fastmath=True)
def back_alloc_layer_extra(layer_len, node_val_len, node_loss_indptr, node_extra_indptr,
                           loss_in, loss_out, loss_indptr, loss_val,
                           temp_node_loss_layer_ba,
                           extras_indptr, extras_val,
                           temp_node_extras_layer_merge, temp_node_extras_layer_merge_save
                           ):
    """

    Args:
        layer_len: number of layers in the node
        node_val_len: number of actual values in sidx
        node_loss_indptr: index of the loss pointer of the node
        node_extra_indptr: index of the extra pointer of the node
        loss_in: loss before applying profile
        loss_out: loss after applying profile
        loss_indptr: index to loss pointer
        loss_val: loss values
        temp_node_loss_layer_ba: sparse array to loss after layer back alloc
        extras_indptr: index to extra pointer
        extras_val: extra values
        temp_node_extras_layer_merge: node extra after profile
        temp_node_extras_layer_merge_save: node extra before profile

    """
    for i in range(node_val_len):
        deductible_delta = temp_node_extras_layer_merge[i, DEDUCTIBLE] - temp_node_extras_layer_merge_save[i, DEDUCTIBLE]
        if deductible_delta >= 0:  # deductible increase no loss reallocation, deductible and loss are allocated based on loss
            if loss_in[i] > 0:
                ded_factor = deductible_delta / loss_in[i]
                loss_factor = loss_out[i] / loss_in[i]
            else:
                ded_factor = 0
                loss_factor = 0
            realloc = 0
        else:
            realloc = deductible_delta
            realloc_numerator = UNDERLIMIT if temp_node_extras_layer_merge[i, UNDERLIMIT] > 0 else DEDUCTIBLE
            ded_factor = realloc / temp_node_extras_layer_merge_save[i, realloc_numerator]
            loss_factor = loss_out[i] / (loss_in[i] - realloc)

        overlimit_delta = (temp_node_extras_layer_merge[i, OVERLIMIT]
                           - temp_node_extras_layer_merge_save[i, OVERLIMIT])
        if overlimit_delta > 0:
            overlimit_factor = overlimit_delta / (loss_in[i] - realloc)
        elif overlimit_delta == 0:
            overlimit_factor = 0
        else:
            overlimit_factor = - temp_node_extras_layer_merge[i, OVERLIMIT] / temp_node_extras_layer_merge_save[i, OVERLIMIT]

        underlimit_delta = (temp_node_extras_layer_merge[i, UNDERLIMIT]
                            - temp_node_extras_layer_merge_save[i, UNDERLIMIT])

        if underlimit_delta > 0:
            underlimit_factor = underlimit_delta / loss_in[i]
        elif underlimit_delta == 0:
            underlimit_factor = 0
        else:
            underlimit_factor = - temp_node_extras_layer_merge[i, UNDERLIMIT] / temp_node_extras_layer_merge_save[i, UNDERLIMIT]

        for l in range(layer_len):
            layer_loss_indptr = loss_indptr[node_loss_indptr + l]
            layer_extra_indptr = extras_indptr[node_extra_indptr + l]
            if ded_factor < 0:
                if underlimit_factor == 0:
                    l_realloc = ded_factor * extras_val[layer_extra_indptr + i, DEDUCTIBLE]
                else:
                    l_realloc = ded_factor * extras_val[layer_extra_indptr + i, UNDERLIMIT]

                if overlimit_factor >= 0:
                    extras_val[layer_extra_indptr + i, OVERLIMIT] += (overlimit_factor *
                                                                      (loss_val[layer_loss_indptr + i] - l_realloc))
                else:
                    extras_val[layer_extra_indptr + i, OVERLIMIT] *= - overlimit_factor
                temp_node_loss_layer_ba[l, i] = (loss_val[layer_loss_indptr + i] - l_realloc) * loss_factor
                extras_val[layer_extra_indptr + i, DEDUCTIBLE] += l_realloc
                extras_val[layer_extra_indptr + i, UNDERLIMIT] *= -underlimit_factor
            else:
                if overlimit_factor >= 0:
                    extras_val[layer_extra_indptr + i, OVERLIMIT] += overlimit_factor * loss_val[layer_loss_indptr + i]
                else:
                    extras_val[layer_extra_indptr + i, OVERLIMIT] *= - overlimit_factor

                if underlimit_factor >= 0:
                    extras_val[layer_extra_indptr + i, UNDERLIMIT] += underlimit_factor * loss_val[layer_loss_indptr + i]
                else:
                    extras_val[layer_extra_indptr + i, UNDERLIMIT] *= - underlimit_factor

                extras_val[layer_extra_indptr + i, DEDUCTIBLE] += ded_factor * loss_val[layer_loss_indptr + i]
                temp_node_loss_layer_ba[l, i] = loss_val[layer_loss_indptr + i] * loss_factor
