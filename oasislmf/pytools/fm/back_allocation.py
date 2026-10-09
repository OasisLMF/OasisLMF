"""Back Allocation Functions for FM Computation
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
from oasislmf.pytools.common.event_stream import MAX_LOSS_IDX, MEAN_IDX, TIV_IDX
from .common import DEDUCTIBLE, UNDERLIMIT, OVERLIMIT, EXTRA_SIDX_COUNT


@njit(cache=True, error_model="numpy")
def is_canonical_sidx(node_sidx, max_sidx_val):
    """True when ``node_sidx`` is exactly MAX_LOSS, TIV, MEAN, 1..max_sidx_val.

    That is what the collapse emits, so it holds for every node at and above the collapse
    level. Ascending and unique make these five checks sufficient: the tail then holds
    max_sidx_val ascending values in [1, max_sidx_val], which only 1..max_sidx_val can be.

    Args:
        node_sidx: the node's ascending sidx values
        max_sidx_val: sample size, the largest unpacked sample index

    Returns:
        bool: whether a position can be derived from a sidx by arithmetic alone
    """
    n = node_sidx.shape[0]
    if max_sidx_val < 1 or n != max_sidx_val + EXTRA_SIDX_COUNT:
        return False
    return (node_sidx[0] == MAX_LOSS_IDX and node_sidx[1] == TIV_IDX and node_sidx[2] == MEAN_IDX
            and node_sidx[3] == 1 and node_sidx[n - 1] == max_sidx_val)


@njit(cache=True, error_model="numpy")
def gallop(node_sidx, lo, target):
    """Index of ``target`` in ascending ``node_sidx``, searching at or after ``lo``.

    Exponential then binary search, which costs O(k log(n/k)) over k lookups into n values
    rather than the O(n + k) of a lockstep walk -- the difference when a leaf carrying one
    building is read against a node carrying many. ``target`` must be present at or after
    ``lo``; the caller guarantees it by only ever looking up a subset.

    Args:
        node_sidx: ascending sidx values to search
        lo: lower bound, the position after the previous hit
        target: the sidx to locate

    Returns:
        int: index into node_sidx
    """
    n = node_sidx.shape[0]
    if node_sidx[lo] == target:
        return lo
    step = 1
    while lo + step < n and node_sidx[lo + step] < target:
        lo += step
        step <<= 1
    hi = lo + step
    if hi > n - 1:
        hi = n - 1
    lo += 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if node_sidx[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, error_model="numpy")
def resolve_positions(node_sidx, child_sidx, canonical, child_pos):
    """Fill ``child_pos`` with the position in ``node_sidx`` of each value in ``child_sidx``.

    Back allocation writes a factor per node value and reads it back per child value, over two
    different sidx sets. Resolving the child's values to positions once lets both sides index
    the factor array positionally, so it is sized by the node rather than by the packed sidx
    range of the whole structure.

    A node's sidx are the union of its children's, so a base child's are a subset of the
    node's; both are ascending. Where the node carries the canonical collapsed set the
    position is arithmetic, otherwise it is searched.

    The search is a gallop rather than a plain binary search because it probes ``lo`` first, so a
    child that is DENSE in its node costs one comparison per value. That is the ordinary case:
    the building count is uniform across a location's items, and what varies is only which sidx
    survive -- the loss threshold, a peril with no hazard, an undamaged coverage type. Measured
    on a 64-building node the cost is near-flat from a full child (9.7us) to a tenth of one
    (5.2us), so it already adapts to how much of the node a child holds.

    An equal-length fast path was tried -- a subset of the node's size IS the node, so the
    position would be the index -- and removed: it needs the sets to be exactly equal, which the
    thinning above makes rare, and on the fullmc benchmark it moved the run by nothing
    (2:06.3/2:04.9 against 2:05.6/2:06.6 without it).

    Args:
        node_sidx: the node's ascending sidx values
        child_sidx: the child's ascending sidx values, a subset of node_sidx
        canonical: result of is_canonical_sidx for node_sidx
        child_pos: output, at least child_sidx.shape[0] long
    """
    if canonical:
        for val_i in range(child_sidx.shape[0]):
            sidx = child_sidx[val_i]
            if sidx < 0:
                # the carried specials are -5, -3, -1: spaced two apart, so halving places them
                child_pos[val_i] = (sidx + 5) >> 1
            else:
                child_pos[val_i] = sidx + EXTRA_SIDX_COUNT - 1
    else:
        pos = 0
        for val_i in range(child_sidx.shape[0]):
            pos = gallop(node_sidx, pos, child_sidx[val_i])
            child_pos[val_i] = pos
            pos += 1


@njit(cache=True, fastmath=True, error_model="numpy")
def back_alloc_extra_a2(base_children_count, storage_is_base_child, temp_children_queue, nodes_array, profile_i,
                        node_val_count, node_sidx, max_sidx_val, child_pos, sidx_indptr, sidx_indexes, sidx_val,
                        loss_in, loss_out, temp_node_loss, loss_indptr, loss_val,
                        extra, temp_node_extras, extras_indptr, extras_val):
    """Back-allocate loss AND extras to base children using allocation rule 2 (pro-rata).

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
        storage_is_base_child: whether the node the loss was computed into IS the single base
            child. Only then can the post-profile loss be assigned straight to it. Building
            packing breaks that assumption: a node above the collapse level with one child is
            forced to aggregate, so the loss sits on the PARENT and the child still has to be
            back-allocated to.
        temp_children_queue: Array of base children node IDs
        nodes_array: Node information array
        profile_i: Current profile/layer index
        node_val_count: Number of sidx values for this node
        node_sidx: Sample indices for this node
        max_sidx_val: Sample size, to recognise the canonical collapsed layout
        child_pos: Scratch for a child's positions in node_sidx
        sidx_indptr: CSR pointers for sidx
        sidx_indexes: Node to sidx mapping
        sidx_val: Sample index values
        loss_in: Aggregated loss before profile (input to calc)
        loss_out: Loss after profile application (output of calc)
        temp_node_loss: Loss allocation factors [profile, position in node_sidx]
        loss_indptr: CSR pointers for loss
        loss_val: Loss values to update
        extra: Extras after profile [val_count, 3]
        temp_node_extras: Extras BEFORE the profile on entry, allocation factors on exit
            [profile, position in node_sidx, 3]
        extras_indptr: CSR pointers for extras
        extras_val: Extras values to update
    """
    if base_children_count == 1 and storage_is_base_child:  # loss_in IS the base child's storage
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
            diff = extra[val_i, DEDUCTIBLE] - temp_node_extras[profile_i, val_i, DEDUCTIBLE]
            if diff >= 0:
                realloc = 0
                if loss_in[val_i] > 0:
                    temp_node_extras[profile_i, val_i, DEDUCTIBLE] = diff / loss_in[val_i]
                    temp_node_loss[profile_i, val_i] = loss_out[val_i] / loss_in[val_i]
                else:
                    temp_node_extras[profile_i, val_i, DEDUCTIBLE] = 0
                    temp_node_loss[profile_i, val_i] = 0
            else:
                realloc = diff  # to loss or to over

                if extra[val_i, UNDERLIMIT] > 0:
                    temp_node_extras[profile_i, val_i, DEDUCTIBLE] = diff / temp_node_extras[profile_i, val_i, UNDERLIMIT]
                else:
                    temp_node_extras[profile_i, val_i, DEDUCTIBLE] = diff / temp_node_extras[profile_i, val_i, DEDUCTIBLE]
                temp_node_loss[profile_i, val_i] = loss_out[val_i] / (loss_in[val_i] - diff)

            diff = extra[val_i, OVERLIMIT] - temp_node_extras[profile_i, val_i, OVERLIMIT]
            if diff > 0:
                temp_node_extras[profile_i, val_i, OVERLIMIT] = diff / (loss_in[val_i] - realloc)
            elif diff == 0:
                temp_node_extras[profile_i, val_i, OVERLIMIT] = 0
            else:  # we set it to <0 to be able to check it later
                temp_node_extras[profile_i, val_i, OVERLIMIT] = - extra[val_i, OVERLIMIT] / temp_node_extras[
                    profile_i, val_i, OVERLIMIT]

            diff = extra[val_i, UNDERLIMIT] - temp_node_extras[profile_i, val_i, UNDERLIMIT]
            if diff > 0:
                temp_node_extras[profile_i, val_i, UNDERLIMIT] = diff / loss_in[val_i]
            elif diff == 0:
                temp_node_extras[profile_i, val_i, UNDERLIMIT] = 0
            else:  # we set it to <0 to be able to check it later
                temp_node_extras[profile_i, val_i, UNDERLIMIT] = - extra[val_i, UNDERLIMIT] / temp_node_extras[
                    profile_i, val_i, UNDERLIMIT]

            loss_in[val_i] = loss_out[val_i]

        canonical = is_canonical_sidx(node_sidx, max_sidx_val)
        for base_child_i in range(base_children_count):
            child = nodes_array[temp_children_queue[base_child_i]]

            child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
            child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
            child_val_count = child_sidx_end - child_sidx_start

            child_sidx = sidx_val[child_sidx_start: child_sidx_end]
            child_loss = loss_val[loss_indptr[child['loss'] + profile_i]: loss_indptr[child['loss'] + profile_i] + child_val_count]
            child_extra = extras_val[
                extras_indptr[child['extra'] + profile_i]: extras_indptr[child['extra'] + profile_i] + child_val_count]
            resolve_positions(node_sidx, child_sidx, canonical, child_pos)

            for val_i in range(child_val_count):
                if temp_node_extras[profile_i, child_pos[val_i], DEDUCTIBLE] < 0:  # realloc loss
                    if temp_node_extras[profile_i, child_pos[val_i], UNDERLIMIT] == 0:
                        realloc = temp_node_extras[profile_i, child_pos[val_i], DEDUCTIBLE] * child_extra[val_i, DEDUCTIBLE]
                    else:
                        realloc = temp_node_extras[profile_i, child_pos[val_i], DEDUCTIBLE] * child_extra[val_i, UNDERLIMIT]
                    if temp_node_extras[profile_i, child_pos[val_i], OVERLIMIT] >= 0:
                        child_extra[val_i, OVERLIMIT] = child_extra[val_i, OVERLIMIT] + temp_node_extras[
                            profile_i, child_pos[val_i], OVERLIMIT] * (child_loss[val_i] - realloc)
                    else:
                        child_extra[val_i, OVERLIMIT] = - temp_node_extras[profile_i, child_pos[val_i], OVERLIMIT] * child_extra[
                            val_i, OVERLIMIT]

                    child_loss[val_i] = (child_loss[val_i] - realloc) * temp_node_loss[profile_i, child_pos[val_i]]
                    child_extra[val_i, DEDUCTIBLE] = child_extra[val_i, DEDUCTIBLE] + realloc
                    child_extra[val_i, UNDERLIMIT] = - temp_node_extras[profile_i, child_pos[val_i], UNDERLIMIT] * child_extra[
                        val_i, UNDERLIMIT]

                else:
                    if temp_node_extras[profile_i, child_pos[val_i], OVERLIMIT] >= 0:
                        child_extra[val_i, OVERLIMIT] = child_extra[val_i, OVERLIMIT] + temp_node_extras[
                            profile_i, child_pos[val_i], OVERLIMIT] * child_loss[val_i]
                    else:
                        child_extra[val_i, OVERLIMIT] = - temp_node_extras[profile_i, child_pos[val_i], OVERLIMIT] * child_extra[
                            val_i, OVERLIMIT]

                    if temp_node_extras[profile_i, child_pos[val_i], UNDERLIMIT] >= 0:
                        child_extra[val_i, UNDERLIMIT] = child_extra[val_i, UNDERLIMIT] + temp_node_extras[
                            profile_i, child_pos[val_i], UNDERLIMIT] * child_loss[val_i]
                    else:
                        child_extra[val_i, UNDERLIMIT] = - temp_node_extras[profile_i, child_pos[val_i], UNDERLIMIT] * child_extra[
                            val_i, UNDERLIMIT]

                    child_extra[val_i, DEDUCTIBLE] = child_extra[val_i, DEDUCTIBLE] + temp_node_extras[
                        profile_i, child_pos[val_i], DEDUCTIBLE] * child_loss[val_i]
                    child_loss[val_i] = child_loss[val_i] * temp_node_loss[profile_i, child_pos[val_i]]
            # print('ba', child['level_id'], child['agg_id'], profile_i, loss_indptr[child['loss'] + profile_i], child_loss[0], temp_node_loss[profile_i, -3],
            # extras_indptr[child['extra'] + profile_i], child_extra[0])


@njit(cache=True, fastmath=True, error_model="numpy")
def back_alloc_a2(base_children_count, storage_is_base_child, temp_children_queue, nodes_array, profile_i,
                  node_val_count, node_sidx, max_sidx_val, child_pos, sidx_indptr, sidx_indexes, sidx_val,
                  loss_in, loss_out, temp_node_loss, loss_indptr, loss_val):
    """Back-allocate loss only (no extras) to base children using allocation rule 2.

    Simpler version of back_alloc_extra_a2 when extras tracking is not needed.
    Computes a single loss factor = loss_out / loss_in and applies it to all children.

    For single child: Direct assignment (loss_in = loss_out). For multiple children:

    1. Compute factor for each sidx: ``factor[sidx] = loss_out[sidx] / loss_in[sidx]``
    2. For each child: ``child_loss[sidx] *= factor[sidx]``

    Modifies in-place:
    - loss_in: Updated to loss_out values
    - loss_val: Child losses scaled by factor

    Args:
        base_children_count: Number of base children
        storage_is_base_child: whether the node the loss was computed into IS the single base
            child. Only then can the post-profile loss be assigned straight to it. Building
            packing breaks that assumption: a node above the collapse level with one child is
            forced to aggregate, so the loss sits on the PARENT and the child still has to be
            back-allocated to.
        temp_children_queue: Base children node IDs
        nodes_array: Node information array
        profile_i: Current profile/layer index
        node_val_count: Number of sidx values
        node_sidx: Sample indices for this node
        max_sidx_val: Sample size, to recognise the canonical collapsed layout
        child_pos: Scratch for a child's positions in node_sidx
        sidx_indptr: CSR pointers for sidx
        sidx_indexes: Node to sidx mapping
        sidx_val: Sample index values
        loss_in: Loss before profile
        loss_out: Loss after profile
        temp_node_loss: Factors [profile, position in node_sidx]
        loss_indptr: CSR pointers for loss
        loss_val: Loss values to update
    """
    if base_children_count == 1 and storage_is_base_child:  # loss_in IS the base child's storage
        loss_in[:] = loss_out
    else:
        for val_i in range(node_val_count):
            if loss_out[val_i]:
                temp_node_loss[profile_i, val_i] = loss_out[val_i] / loss_in[val_i]
            else:
                temp_node_loss[profile_i, val_i] = 0
            loss_in[val_i] = loss_out[val_i]

        canonical = is_canonical_sidx(node_sidx, max_sidx_val)
        for base_child_i in range(base_children_count):
            child = nodes_array[temp_children_queue[base_child_i]]

            child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
            child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
            child_val_count = child_sidx_end - child_sidx_start

            child_sidx = sidx_val[child_sidx_start: child_sidx_end]
            child_loss = loss_val[loss_indptr[child['loss'] + profile_i]: loss_indptr[child['loss'] + profile_i] + child_val_count]
            resolve_positions(node_sidx, child_sidx, canonical, child_pos)

            for val_i in range(child_val_count):
                child_loss[val_i] = child_loss[val_i] * temp_node_loss[profile_i, child_pos[val_i]]
            # print('ba', child['level_id'], child['agg_id'], profile_i, loss_indptr[child['loss'] + profile_i], child_loss[0], temp_node_loss[profile_i, -3])


@njit(cache=True, fastmath=True, error_model="numpy")
def back_alloc_layer(layer_count, node_val_count, node_loss_ptr_i,
                     loss_in, loss_out, loss_indptr, loss_val,
                     temp_node_loss_layer_ba):
    """Back-allocate cross-layer profile results to individual layers (loss only).

    When a cross-layer profile is applied, it operates on the sum of all layers.
    This function distributes the result back to each layer proportionally::

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
    """Back-allocate cross-layer profile results to individual layers (loss AND extras).

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
