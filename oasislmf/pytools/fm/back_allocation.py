from numba import njit
from .common import DEDUCTIBLE, UNDERLIMIT, OVERLIMIT


@njit(cache=True, fastmath=True, error_model="numpy")
def back_alloc_extra_a2(base_children_count, temp_children_queue, nodes_array, profile_i,
                        node_val_count, node_sidx, sidx_indptr, sidx_indexes, sidx_val,
                        loss_in, loss_out, temp_node_loss, loss_indptr, loss_val,
                        extra, temp_node_extras, extras_indptr, extras_val):
    """
    back allocation of loss and extra to the base children,
    The function modifies in-place array loss_in and items loss and extra in loss_val and extra val
    Args:
        base_children_count: number of base children
        temp_children_queue: array of base children
        nodes_array: array of all node info
        profile_i: profile index
        node_val_count: number of actual values in sidx
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
    if base_children_count == 1:  # this is a base children, we only need to assign loss_in
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

        for val_i in range(node_val_count):
            diff = extra[val_i, DEDUCTIBLE] - temp_node_extras[profile_i, node_sidx[val_i], DEDUCTIBLE]
            if diff >= 0:
                realloc = 0
                if loss_in[val_i] > 0:
                    temp_node_extras[profile_i, node_sidx[val_i], DEDUCTIBLE] = diff / loss_in[val_i]
                    temp_node_loss[profile_i, node_sidx[val_i]] = loss_out[val_i] / loss_in[val_i]
                else:
                    temp_node_extras[profile_i, node_sidx[val_i], DEDUCTIBLE] = 0
                    temp_node_loss[profile_i, node_sidx[val_i]] = 0
            else:
                realloc = diff  # to loss or to over

                if extra[val_i, UNDERLIMIT] > 0:
                    temp_node_extras[profile_i, node_sidx[val_i], DEDUCTIBLE] = diff / temp_node_extras[profile_i, node_sidx[val_i], UNDERLIMIT]
                else:
                    temp_node_extras[profile_i, node_sidx[val_i], DEDUCTIBLE] = diff / temp_node_extras[profile_i, node_sidx[val_i], DEDUCTIBLE]
                temp_node_loss[profile_i, node_sidx[val_i]] = loss_out[val_i] / (loss_in[val_i] - diff)

            diff = extra[val_i, OVERLIMIT] - temp_node_extras[profile_i, node_sidx[val_i], OVERLIMIT]
            if diff > 0:
                temp_node_extras[profile_i, node_sidx[val_i], OVERLIMIT] = diff / (loss_in[val_i] - realloc)
            elif diff == 0:
                temp_node_extras[profile_i, node_sidx[val_i], OVERLIMIT] = 0
            else:  # we set it to <0 to be able to check it later
                temp_node_extras[profile_i, node_sidx[val_i], OVERLIMIT] = - extra[val_i, OVERLIMIT] / temp_node_extras[
                    profile_i, node_sidx[val_i], OVERLIMIT]

            diff = extra[val_i, UNDERLIMIT] - temp_node_extras[profile_i, node_sidx[val_i], UNDERLIMIT]
            if diff > 0:
                temp_node_extras[profile_i, node_sidx[val_i], UNDERLIMIT] = diff / loss_in[val_i]
            elif diff == 0:
                temp_node_extras[profile_i, node_sidx[val_i], UNDERLIMIT] = 0
            else:  # we set it to <0 to be able to check it later
                temp_node_extras[profile_i, node_sidx[val_i], UNDERLIMIT] = - extra[val_i, UNDERLIMIT] / temp_node_extras[
                    profile_i, node_sidx[val_i], UNDERLIMIT]

            loss_in[val_i] = loss_out[val_i]

        for base_child_i in range(base_children_count):
            child = nodes_array[temp_children_queue[base_child_i]]

            child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
            child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
            child_val_count = child_sidx_end - child_sidx_start

            child_sidx = sidx_val[child_sidx_start: child_sidx_end]
            child_loss = loss_val[loss_indptr[child['loss'] + profile_i]: loss_indptr[child['loss'] + profile_i] + child_val_count]
            child_extra = extras_val[
                extras_indptr[child['extra'] + profile_i]: extras_indptr[child['extra'] + profile_i] + child_val_count]

            for val_i in range(child_val_count):
                if temp_node_extras[profile_i, child_sidx[val_i], DEDUCTIBLE] < 0:  # realloc loss
                    if temp_node_extras[profile_i, child_sidx[val_i], UNDERLIMIT] == 0:
                        realloc = temp_node_extras[profile_i, child_sidx[val_i], DEDUCTIBLE] * child_extra[val_i, DEDUCTIBLE]
                    else:
                        realloc = temp_node_extras[profile_i, child_sidx[val_i], DEDUCTIBLE] * child_extra[val_i, UNDERLIMIT]
                    if temp_node_extras[profile_i, child_sidx[val_i], OVERLIMIT] >= 0:
                        child_extra[val_i, OVERLIMIT] = child_extra[val_i, OVERLIMIT] + temp_node_extras[
                            profile_i, child_sidx[val_i], OVERLIMIT] * (child_loss[val_i] - realloc)
                    else:
                        child_extra[val_i, OVERLIMIT] = - temp_node_extras[profile_i, child_sidx[val_i], OVERLIMIT] * child_extra[
                            val_i, OVERLIMIT]

                    child_loss[val_i] = (child_loss[val_i] - realloc) * temp_node_loss[profile_i, child_sidx[val_i]]
                    child_extra[val_i, DEDUCTIBLE] = child_extra[val_i, DEDUCTIBLE] + realloc
                    child_extra[val_i, UNDERLIMIT] = - temp_node_extras[profile_i, child_sidx[val_i], UNDERLIMIT] * child_extra[
                        val_i, UNDERLIMIT]

                else:
                    if temp_node_extras[profile_i, child_sidx[val_i], OVERLIMIT] >= 0:
                        child_extra[val_i, OVERLIMIT] = child_extra[val_i, OVERLIMIT] + temp_node_extras[
                            profile_i, child_sidx[val_i], OVERLIMIT] * child_loss[val_i]
                    else:
                        child_extra[val_i, OVERLIMIT] = - temp_node_extras[profile_i, child_sidx[val_i], OVERLIMIT] * child_extra[
                            val_i, OVERLIMIT]

                    if temp_node_extras[profile_i, child_sidx[val_i], UNDERLIMIT] >= 0:
                        child_extra[val_i, UNDERLIMIT] = child_extra[val_i, UNDERLIMIT] + temp_node_extras[
                            profile_i, child_sidx[val_i], UNDERLIMIT] * child_loss[val_i]
                    else:
                        child_extra[val_i, UNDERLIMIT] = - temp_node_extras[profile_i, child_sidx[val_i], UNDERLIMIT] * child_extra[
                            val_i, UNDERLIMIT]

                    child_extra[val_i, DEDUCTIBLE] = child_extra[val_i, DEDUCTIBLE] + temp_node_extras[
                        profile_i, child_sidx[val_i], DEDUCTIBLE] * child_loss[val_i]
                    child_loss[val_i] = child_loss[val_i] * temp_node_loss[profile_i, child_sidx[val_i]]
            # print('ba', child['level_id'], child['agg_id'], profile_i, loss_indptr[child['loss'] + profile_i], child_loss[0], temp_node_loss[profile_i, -3],
            # extras_indptr[child['extra'] + profile_i], child_extra[0])


@njit(cache=True, fastmath=True, error_model="numpy")
def back_alloc_a2(base_children_count, temp_children_queue, nodes_array, profile_i,
                  node_val_count, node_sidx, sidx_indptr, sidx_indexes, sidx_val,
                  loss_in, loss_out, temp_node_loss, loss_indptr, loss_val):
    """
    back allocation of loss to the base children. The function modifies in-place array loss_in and items loss in loss_val.
    Args:
        base_children_count: number of base children
        temp_children_queue: array of base children
        nodes_array: array of all node info
        profile_i: profile index
        node_val_count: number of actual values in sidx
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
    if base_children_count == 1:
        loss_in[:] = loss_out
    else:
        for val_i in range(node_val_count):
            if loss_out[val_i]:
                temp_node_loss[profile_i, node_sidx[val_i]] = loss_out[val_i] / loss_in[val_i]
            else:
                temp_node_loss[profile_i, node_sidx[val_i]] = 0
            loss_in[val_i] = loss_out[val_i]

        for base_child_i in range(base_children_count):
            child = nodes_array[temp_children_queue[base_child_i]]

            child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
            child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
            child_val_count = child_sidx_end - child_sidx_start

            child_sidx = sidx_val[child_sidx_start: child_sidx_end]
            child_loss = loss_val[loss_indptr[child['loss'] + profile_i]: loss_indptr[child['loss'] + profile_i] + child_val_count]

            for val_i in range(child_val_count):
                child_loss[val_i] = child_loss[val_i] * temp_node_loss[profile_i, child_sidx[val_i]]
            # print('ba', child['level_id'], child['agg_id'], profile_i, loss_indptr[child['loss'] + profile_i], child_loss[0], temp_node_loss[profile_i, -3])


@njit(cache=True, fastmath=True, error_model="numpy")
def back_alloc_layer(layer_count, node_val_count, node_loss_ptr_i,
                     loss_in, loss_out, loss_indptr, loss_val,
                     temp_node_loss_layer_ba):
    """
    Args:
        layer_count: number of layers in the node
        node_val_count: number of actual values in sidx
        node_loss_ptr_i: index of the loss pointer of the node
        loss_in: loss before applying profile
        loss_out: loss after applying profile
        loss_indptr: index to loss pointer
        loss_val: loss values
        temp_node_loss_layer_ba: sparse array to loss after layer back alloc
    """
    for val_i in range(node_val_count):
        if loss_out[val_i]:
            loss_factor = loss_out[val_i] / loss_in[val_i]
        else:
            loss_factor = 0

        for layer_i in range(layer_count):
            layer_loss_ptr_i = loss_indptr[node_loss_ptr_i + layer_i]
            temp_node_loss_layer_ba[layer_i, val_i] = loss_val[layer_loss_ptr_i + val_i] * loss_factor


@njit(cache=True, fastmath=True, error_model="numpy")
def back_alloc_layer_extra(layer_count, node_val_count, node_loss_ptr_i, node_extra_ptr_i,
                           loss_in, loss_out, loss_indptr, loss_val,
                           temp_node_loss_layer_ba,
                           extras_indptr, extras_val,
                           temp_node_extras_layer_merge, temp_node_extras_layer_merge_save
                           ):
    """

    Args:
        layer_count: number of layers in the node
        node_val_count: number of actual values in sidx
        node_loss_ptr_i: index of the loss pointer of the node
        node_extra_ptr_i: index of the extra pointer of the node
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
    for val_i in range(node_val_count):
        deductible_delta = temp_node_extras_layer_merge[val_i, DEDUCTIBLE] - temp_node_extras_layer_merge_save[val_i, DEDUCTIBLE]
        if deductible_delta >= 0:  # deductible increase no loss reallocation, deductible and loss are allocated based on loss
            if loss_in[val_i] > 0:
                ded_factor = deductible_delta / loss_in[val_i]
                loss_factor = loss_out[val_i] / loss_in[val_i]
            else:
                ded_factor = 0
                loss_factor = 0
            realloc = 0
        else:
            realloc = deductible_delta
            realloc_numerator = UNDERLIMIT if temp_node_extras_layer_merge[val_i, UNDERLIMIT] > 0 else DEDUCTIBLE
            ded_factor = realloc / temp_node_extras_layer_merge_save[val_i, realloc_numerator]
            loss_factor = loss_out[val_i] / (loss_in[val_i] - realloc)

        overlimit_delta = (temp_node_extras_layer_merge[val_i, OVERLIMIT]
                           - temp_node_extras_layer_merge_save[val_i, OVERLIMIT])
        if overlimit_delta > 0:
            overlimit_factor = overlimit_delta / (loss_in[val_i] - realloc)
        elif overlimit_delta == 0:
            overlimit_factor = 0
        else:
            overlimit_factor = - temp_node_extras_layer_merge[val_i, OVERLIMIT] / temp_node_extras_layer_merge_save[val_i, OVERLIMIT]

        underlimit_delta = (temp_node_extras_layer_merge[val_i, UNDERLIMIT]
                            - temp_node_extras_layer_merge_save[val_i, UNDERLIMIT])

        if underlimit_delta > 0:
            underlimit_factor = underlimit_delta / loss_in[val_i]
        elif underlimit_delta == 0:
            underlimit_factor = 0
        else:
            underlimit_factor = - temp_node_extras_layer_merge[val_i, UNDERLIMIT] / temp_node_extras_layer_merge_save[val_i, UNDERLIMIT]

        for layer_i in range(layer_count):
            layer_loss_ptr_i = loss_indptr[node_loss_ptr_i + layer_i]
            layer_extra_ptr_i = extras_indptr[node_extra_ptr_i + layer_i]
            if ded_factor < 0:
                if underlimit_factor == 0:
                    layer_realloc = ded_factor * extras_val[layer_extra_ptr_i + val_i, DEDUCTIBLE]
                else:
                    layer_realloc = ded_factor * extras_val[layer_extra_ptr_i + val_i, UNDERLIMIT]

                if overlimit_factor >= 0:
                    extras_val[layer_extra_ptr_i + val_i, OVERLIMIT] += (overlimit_factor *
                                                                      (loss_val[layer_loss_ptr_i + val_i] - layer_realloc))
                else:
                    extras_val[layer_extra_ptr_i + val_i, OVERLIMIT] *= - overlimit_factor
                temp_node_loss_layer_ba[layer_i, val_i] = (loss_val[layer_loss_ptr_i + val_i] - layer_realloc) * loss_factor
                extras_val[layer_extra_ptr_i + val_i, DEDUCTIBLE] += layer_realloc
                extras_val[layer_extra_ptr_i + val_i, UNDERLIMIT] *= -underlimit_factor
            else:
                if overlimit_factor >= 0:
                    extras_val[layer_extra_ptr_i + val_i, OVERLIMIT] += overlimit_factor * loss_val[layer_loss_ptr_i + val_i]
                else:
                    extras_val[layer_extra_ptr_i + val_i, OVERLIMIT] *= - overlimit_factor

                if underlimit_factor >= 0:
                    extras_val[layer_extra_ptr_i + val_i, UNDERLIMIT] += underlimit_factor * loss_val[layer_loss_ptr_i + val_i]
                else:
                    extras_val[layer_extra_ptr_i + val_i, UNDERLIMIT] *= - underlimit_factor

                extras_val[layer_extra_ptr_i + val_i, DEDUCTIBLE] += ded_factor * loss_val[layer_loss_ptr_i + val_i]
                temp_node_loss_layer_ba[layer_i, val_i] = loss_val[layer_loss_ptr_i + val_i] * loss_factor
