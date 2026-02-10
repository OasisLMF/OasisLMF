"""
Back Allocation Functions for FM Computation
=============================================

This module handles the distribution of computed losses back to the original items
(base children) after financial terms have been applied at an aggregate level.

Back Allocation Overview
------------------------
When financial terms (deductibles, limits) are applied to an aggregated loss,
we need to determine how the resulting insured loss should be attributed back
to each contributing item. This is called "back allocation".

Allocation Rules:
- Rule 0: No back allocation - output only at aggregate level
- Rule 1: Proportional to original (ground-up) loss
- Rule 2: Proportional to computed loss at each level (pro-rata)

For Rule 2, the allocation factor is: output_loss / input_loss
Each child's loss is multiplied by this factor.

Extras Back Allocation
----------------------
When extras (deductible, overlimit, underlimit) are tracked, their allocation
is more complex because the relationship between input and output isn't linear:

- If DEDUCTIBLE increases: allocated proportionally to loss
- If DEDUCTIBLE decreases: loss was reallocated FROM deductible
  - If underlimit > 0: deductible change allocated by underlimit
  - Else: allocated by existing deductible

- If OVERLIMIT increases: more loss exceeded limit, allocated by loss
- If OVERLIMIT decreases: limit was raised, scale down proportionally

- If UNDERLIMIT increases: more loss deducted, allocated by loss
- If UNDERLIMIT decreases: less capacity remains, scale down proportionally

The sign of the factor indicates the allocation direction:
- Positive factor: additive allocation (base + factor * value)
- Negative factor: multiplicative scaling (base * -factor)
"""

from numba import njit
from .common import DEDUCTIBLE, UNDERLIMIT, OVERLIMIT


@njit(cache=True, fastmath=True, error_model="numpy")
def back_alloc_extra_a2(base_children_count, temp_children_queue, nodes_array, profile_i,
                        node_val_count, node_sidx, sidx_indptr, sidx_indexes, sidx_val,
                        loss_in, loss_out, temp_node_loss, loss_indptr, loss_val,
                        extra, temp_node_extras, extras_indptr, extras_val):
    """
    Back-allocate loss AND extras to base children using allocation rule 2 (pro-rata).

    This function distributes the computed insured loss back to individual items,
    along with their share of deductibles and limits. It handles the complex
    relationships between how financial terms affect loss at aggregate vs item level.

    Algorithm:
    1. For single child: Direct assignment (no allocation needed)
    2. For multiple children:
       a. Compute allocation factors for loss and each extra type
       b. Store factors in temp arrays (temp_node_loss, temp_node_extras)
       c. Apply factors to each child's loss and extras

    The factor computation handles four cases for each extra type:
    - Increase: New amount allocated proportionally to loss
    - Decrease to underlimit: Reallocated based on remaining underlimit
    - Decrease to deductible: Reallocated based on existing deductible
    - No change: Factor = 0

    Negative factors indicate multiplicative scaling rather than additive allocation.

    Modifies in-place:
    - loss_in: Updated to loss_out values
    - loss_val: Child losses scaled by allocation factor
    - extras_val: Child extras adjusted by their factors

    Args:
        base_children_count: Number of base children to allocate to
        temp_children_queue: Array of base children node IDs
        nodes_array: Node information array
        profile_i: Current profile/layer index
        node_val_count: Number of sidx values for this node
        node_sidx: Sample indices for this node
        sidx_indptr: CSR pointers for sidx
        sidx_indexes: Node to sidx mapping
        sidx_val: Sample index values
        loss_in: Aggregated loss before profile (input to calc)
        loss_out: Loss after profile application (output of calc)
        temp_node_loss: Dense array for loss allocation factors
        loss_indptr: CSR pointers for loss
        loss_val: Loss values to update
        extra: Extras after profile [val_count, 3]
        temp_node_extras: Dense array with extras BEFORE profile [profile, sidx, 3]
        extras_indptr: CSR pointers for extras
        extras_val: Extras values to update
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
    Back-allocate loss only (no extras) to base children using allocation rule 2.

    Simpler version of back_alloc_extra_a2 when extras tracking is not needed.
    Computes a single loss factor = loss_out / loss_in and applies it to all children.

    For single child: Direct assignment (loss_in = loss_out)
    For multiple children:
    1. Compute factor for each sidx: factor[sidx] = loss_out[sidx] / loss_in[sidx]
    2. For each child: child_loss[sidx] *= factor[sidx]

    Modifies in-place:
    - loss_in: Updated to loss_out values
    - loss_val: Child losses scaled by factor

    Args:
        base_children_count: Number of base children
        temp_children_queue: Base children node IDs
        nodes_array: Node information array
        profile_i: Current profile/layer index
        node_val_count: Number of sidx values
        node_sidx: Sample indices for this node
        sidx_indptr: CSR pointers for sidx
        sidx_indexes: Node to sidx mapping
        sidx_val: Sample index values
        loss_in: Loss before profile
        loss_out: Loss after profile
        temp_node_loss: Dense array for factors [profile, sidx]
        loss_indptr: CSR pointers for loss
        loss_val: Loss values to update
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
    Back-allocate cross-layer profile results to individual layers (loss only).

    When a cross-layer profile is applied, it operates on the sum of all layers.
    This function distributes the result back to each layer proportionally:

    For each sample:
        factor = loss_out[sidx] / loss_in[sidx]  (where loss_in = sum of all layers)
        For each layer:
            layer_loss_after = layer_loss_before * factor

    The results are stored in temp_node_loss_layer_ba for later use in
    per-layer profile application.

    Args:
        layer_count: Number of layers to allocate across
        node_val_count: Number of sidx values
        node_loss_ptr_i: Base index into loss_indptr for this node
        loss_in: Merged loss before profile (sum of all layers)
        loss_out: Loss after cross-layer profile
        loss_indptr: CSR pointers for loss
        loss_val: Loss values (read-only here)
        temp_node_loss_layer_ba: Output array [layer, sidx] for allocated losses
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
    Back-allocate cross-layer profile results to individual layers (loss AND extras).

    Similar to back_alloc_layer but also handles the extras (deductible, overlimit,
    underlimit) allocation. The extras allocation follows the same rules as
    back_alloc_extra_a2 but across layers instead of across children.

    For each sample, computes allocation factors based on how each extra changed:
    - deductible_delta = extra_after - extra_before
    - If delta >= 0: deductible increased, factor = delta / loss_in
    - If delta < 0: deductible decreased, factor based on underlimit or deductible

    Then applies these factors to each layer's extras and computes the layer's
    allocated loss.

    Modifies:
    - temp_node_loss_layer_ba: Stores allocated loss per layer
    - extras_val: Updates each layer's extras in place

    Args:
        layer_count: Number of layers
        node_val_count: Number of sidx values
        node_loss_ptr_i: Base index into loss_indptr for this node
        node_extra_ptr_i: Base index into extras_indptr for this node
        loss_in: Merged loss before profile
        loss_out: Loss after cross-layer profile
        loss_indptr: CSR pointers for loss
        loss_val: Loss values (read-only here)
        temp_node_loss_layer_ba: Output for allocated losses [layer, sidx]
        extras_indptr: CSR pointers for extras
        extras_val: Extras values (modified in place)
        temp_node_extras_layer_merge: Merged extras AFTER profile
        temp_node_extras_layer_merge_save: Merged extras BEFORE profile
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
