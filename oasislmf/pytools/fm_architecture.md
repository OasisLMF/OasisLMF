# Financial Module Architecture

## Manager
Manager is the high level entry point to run an FM computation.
It orchestrates the different modules together in order to process the events coming in.
There are several run mode:
- run_synchronous: processing one event at a time
- run_threaded: threaded mode using queue as synchronization tools.
- run_ray: mode using ray to load balance events

### synchronous mode
In synchronous mode, one event at a time is read from the input stream
then we computed the result and write it to the output stream.

This simple mode is mainly use for debugging as the other modes are more performant in general.
In this mode only one input and one output stream are possible.

### asynchronous modes 

 ![alt text](FM_asynchronous_data_flow.png)

This model can be use as base for a multi-thread, multi-process or cluster architecture
allowing a consistent architecture throw diverse scaling solutions.

By separating the reading and writing of the event from the computation itself and using Queues,
we prepare the work for other processes (gulcalc or other FM) to communicate directly between them.
This could remove the serialization, deserialization steps necessary when using pipes.

The queue_event_reader, event_computer and queue_event_writer are based on the classical
producer-consumer pattern. This model also limits the time spent waiting for IO.
We use a sentinel to indicate to the consumer that all events have been processed.


## Financial Structure
The purpose of this module is to parse the static input financial structure
and consolidate the information into simple object ready to be use by the compute function.
The idea is to factorise all the computation and logic that can be done at this step
and prepare everything possible to have a generic way to handle the computation for each item.

During the computation, computation data (loss, deductible, over limit, not null ...)
will separated into 3 buckets (input, temp, output).
Each computation node will then be assigned a bucket and an index
and the values for this node will then be store (during the computation) into a big matrix where the row
will correspond to their index.

A computation node then consist of a node id (layer, level, agg) and a computation id
that determine what will be done.
In addition to that several dictionaries(map), can provide additional information needed for the
computation step such as dependencies or profile

 ![alt text](FM_computation_data_structure.png)

this structure present a number of advantage:
- the separation into 3 buckets minimize the data transfer at the interface of the computation.
only inputs are needed in only outputs are passed out.
- input and output format are just numpy array that are well supported and could be passed or retrieved
by other processes.
- access to any node result is done with a O(1) complexity
- aggregation that are just intermediary in the program tree structure with no computation
can take the index of their unique child dependency and cost no computation.
- new type of computation can be simply added to the generic compute structure.
A computation node is simply a functional step with some input and some output.
- data are stored as np.arrays that perform well with the most common operation we do 
sums and multiplications
- in case of bug or error, all the intermediary result could easily be stored for investigation. 


### inputs
the necessary static input for the Financial Module are expected to be in the same folder and are:
- fm_programme.bin: the basic hierarchy of nodes organized by level and aggregation id
- fm_policytc.bin: the policy id to apply to each node and layer described in the programme
- fm_profile.bin (or fm_profile_step.bin not implemented yet): the profile (detailed values) of each policy id
- fm_xref.bin: the mapping between result items and the output ID

Inputs are read directly using numpy.fromfile, with named dtype specific to each fine name.
This allow to access each value in a row like a dictionary
and also provide a compatible interface for the two profile options

We make a realistic assumption that the input and output data will fit in memory.

### outputs
The transform static information that will be needed to build and execute the computation for each event
 
- compute_queue: list of all the computation step to execute to compute an event (ex: profile aggregation step, back-allocation)
- node_to_index: node to (bucket, index) mapping for each node needed to perform the computation.
in this context an item in program can correspond to several nodes in the mapping, one node will be for the profile step
others for the back allocation for example.
- node_to_dependencies: map to list of node that are needed for this step computation.
- node_to_profile: node to profile mapping
- output_item_index: output id to np array index mapping. needed to associate output id
to the computation result
- storage_to_len: dictionary of the size needed for each bucket
- options: computation option (is deductible computation needed, do we need to store intermediary sum ...)
- profile: numpy ndarray of fm_profile.bin


## Computation
Computation module implement the actual computation of each event.
It create all the numpy array necessary for the computation and
execute one by one all the step present in the computation queue.
each step is treated generically as a simple step to perform using the node computation id
without pre-supposition of what has been done before.
The correct ordering of the computation step are performed once, before the computation
by the financial structure step.

so far there is far type of step:
- PROFILE: sum all sub-node array if present (loss, deductibles, over_limit, underlimit) then
apply the calcrule inplace.
- IL_PER_GUL: take the final IL and divide it by the sum of all input gul.
- IL_PER_SUB_IL: take the a item back-allocated IL and divide it by the stored sum of IL computed
before calc rule was applied. (this will serve as common basis for the back allocation of all sub-items)
- PROPORTION: multiply the array loss (il) by the factor computed in IL_PER_GUL or IL_PER_SUB_IL
- COPY: use to copy a vector from an index to an other (copy to output bucket in a0 ba rule for example)

event_computer is a callable that takes event input from input queue, compute the output and
put it in the output queue. 


### Extras across layers

Loss and extras behave differently across layers, and getting this wrong has produced real
bugs in both directions (#2209, and the first attempt at fixing it in #2212).

**The rule: every layer holds the whole accumulated extras.** In OED a layer is a policy
(`PolNumber`), and policy terms apply to the whole loss *before* the layer terms carve out that
policy's range. Each layer is therefore offered the whole loss. It applies its own policy terms,
then its attachment, limit and participation, so the layers' results add up to no more than the
input loss. The extras are the
deductible already taken, the loss already cut off by limits, and the room still left under those
limits, by everything *below* the node. That is one quantity, not something split between the
layers, so each layer of the node gets a full copy. The copy is made by the single-child path in
`compute_event`, by `first_time_layer_extra` when the node is its own single base child, and by
`aggregate_children_extras`, whose children's new layers still point at layer 0.

The items *under* a node are different. When `first_time_layer_extra` creates layers for several
base children, those are the items, and their new layers start at zero. They only receive the
changes allocated back to them. No profile reads them, and items are never aggregated again, so
they never act as accumulated extras.

**Why each layer needs a full copy: min and max deductibles act on the total.** A minimum or
maximum deductible is not applied on top of what was taken below. It bounds the *total*
deductible taken so far (`deductible_under_min` / `deductible_over_max` in `policy_extras.py`):

- **minimum**: if the total is under the minimum, it is topped up to the minimum, first from the
  over-limit pool (loss already cut off by a limit below), then from the loss;
- **maximum**: if the total is over the maximum, the excess is given back to the loss, up to the
  underlimit (the room left under the limits below); the rest goes to over-limit.

A per-layer profile carrying one of these therefore reads its layer's extras. If those were zero,
the full minimum would be taken again on top of the deductible already taken.

Per-layer policy terms only exist when the layers' terms differ. Policy terms identical on every
layer are prepared as a single profile, applied once before the layers are created.

Each layer moves its own effective deductible. With 60k taken below and maximum deductibles of
10k on layer 1 and 20k on layer 2, layer 1 gives back 50k and layer 2 gives back 40k. That is not
90k returned out of 60k taken: each policy's deductible is now 10k and 20k respectively, and each
applies to its own copy of the loss.

```
acc105: 1 location, GUL 1.7M, LocDed6All 100k
        layer 2: PolMinDed6All 250k, attachment 200k, participation 0.5

layer 2 enters with loss 1.6M and deductible 100k (a full copy)
  min ded: total 100k < 250k  -> top up by 150k   -> 1.45M
  layer:   (1.45M - 200k) * 0.5                     = 625,000   correct

with the layer's extras zeroed instead (#2212):
  min ded: total 0 < 250k     -> top up by 250k   -> 1.35M
  layer:   (1.35M - 200k) * 0.5                     = 575,000   wrong
```

**The cross-layer merge: loss is summed, extras take the maximum.** A cross-layer profile (account
terms over several layers) merges the layers, applies the profile once, and allocates the result
back. Summing the loss is right because the slices add up. Summing the extras would count the
same deductible once per layer. The merge takes the per-layer **maximum** instead, which is the
node's value whether the layers hold copies or only layer 0 holds it (the base-level path, where
everything is still zero).

```
acc49 (75% damage): 2 locations, LocDed6All 5k each; 2 layers; account deductible
                    10% of loss, min 20k, max 60k

layers after their terms: loss 150,000 + 462,000 = 612,000; each holds deductible 10,000
  merged deductible (max)          10,000
  account ded 10% of 612,000     + 61,200 = 71,200 > max 60k -> total 60,000
  taken at this level              50,000 -> 562,000 -> 281,000 per location   correct

summing the extras instead (#2209):
  merged deductible (sum)          20,000
  account ded                    + 61,200 = 81,200 > max 60k -> total 60,000
  taken at this level              40,000 -> 572,000 -> 286,000 per location   wrong (+5,000)
```

**Back to the layers** (`back_alloc_layer_extra`). Account terms apply after the layer terms, to
the sum of the layers' losses, so the change they make is shared back between the layers' losses.
The layer terms are not re-applied.

- When the profile reduces the loss (deductible taken, limit), the loss is scaled back to each
  layer in proportion, and the added deductible or overlimit is shared in proportion to each
  layer's loss.
- When the profile returns deductible (a maximum deductible below the accumulated deductible), the
  returned amount is shared in proportion to each layer's own deductible or underlimit. The
  denominator is the sum over layers, so the shares add up to what was returned.
- When the profile reduces the overlimit or underlimit, each layer's value is scaled by the ratio
  `after / before`. When the layers hold the same values, that leaves each at the merged value.

#### Which combinations are exact

| Combination | What the FM computes | Status |
|---|---|---|
| Per-layer policy terms (different deductible, min/max deductible or limit per layer), no account terms | Each layer applies its own terms to the whole loss, starting from the full deductible carried forward, then its layer terms. | Exact (acc105, acc106) |
| Account deductible, minimum deductible or limit over layers, with no per-layer terms in between | The layers hold the same extras, so the merged value is the deductible actually taken below. Account terms apply to the summed layer losses, and the change is shared back as above. A maximum deductible is included as long as it does not return deductible taken below. | Exact (acc49 and the `insurance_account` group) |
| Account maximum deductible below the deductible taken below, over layers | The excess over the maximum is added to the summed layer losses and shared between the layers in proportion to their deductible. The exact result would re-run each layer's terms on the larger loss, which the FM cannot do because account terms apply after layer terms. See the example below. | Approximate |
| Account terms over layers whose per-layer terms differ | See below. | Computed, but not defined by OED |

**Account maximum deductible returning deductible over layers.** The exact result would push the
returned deductible through each layer's terms again. The FM adds it to the layers' losses
instead:

```
1 location, GUL 1.7M, LocDed6All 100k
layer 1: attachment 0, unlimited, participation 1       -> 1,600,000
layer 2: attachment 200k, unlimited, participation 0.5  -> 700,000
account: AccMaxDed6All 50k -> 50k of the 100k is returned

FM:    50k shared by deductible (equal)  -> 1,625,000 / 725,000   (+50k)
exact: layers re-run on 1.65M            -> 1,650,000 / 725,000   (+75k)
```

**Account terms over layers whose per-layer terms differ.** After different per-layer terms, the
layers carry different effective deductibles, for example 150k on one and 130k on the other. Each
is that policy's own view of the deductible taken. OED does not define what "the deductible
already taken" means for an account term that spans policies with different deductibles. The FM
uses the **largest** of the layers' values and applies the account terms to the summed layer
losses:

```
1 location, GUL 1M, LocDed6All 100k
layer 1: PolDed6All 50k, attachment 0, limit 500k  -> deductible 150k, loss 500,000
layer 2: PolDed6All 30k, attachment 500k, unlimited -> deductible 130k, loss 370,000
account: AccMinDed6All 200k

merged: loss 870,000, deductible max(150k, 130k) = 150k
min ded tops up by 50k -> 820,000, shared back in proportion: 471,264 / 348,736
```

Other choices would give other results: the sum of the layers' deductibles (280k, which double
counts the 100k taken below) gives 870,000, and the shared 100k plus each layer's own part (180k)
gives 850,000. None of them follows from the OED definition. No error is raised, but results for
this combination should be read with that in mind, and no validation unit covers it.


## Policies
The policy module contains all the function associated to the supported policy
they all take the same numpy array as input and act directly on them (inplace). 

signature:
`calc(policy, loss_out, loss_in, deductible, over_limit, under_limit)`
loss is present in two array because in some case we want to keep the sum value before
the calc rule is applied. (if it is not the case the array are in fact the same object)

## Stream
the stream module is responsible for the parsing and writing of gul and fm stream.
the gul stream in parse and transformed into the numpy input array.
the fm stream is written from the numpy output array.
In the threaded mode reader and writer work in combination with the event_computer using
Queues.


## Queue
In order to prepare for a more distributed approach and also to reduce time spent waiting for io,
the main mode for the FM calc is to use several threads for input, output and computation.
In order to terminate gracefully in case of error, the standard python queue is over-writen to
implement a terminate function that will tell thread to stop if an error is caught.
