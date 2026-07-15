---
orphan: false
---

# Financial Module — architecture

```{note}
This page was migrated from `oasislmf/pytools/fm_architecture.md`, which
previously lived next to the code but was **not wired into any documentation
site**. It is now co-located in the docs so it is published and reviewed
alongside the Financial Module explanation.
```

## Manager

Manager is the high level entry point to run an FM computation. It orchestrates
the different modules together in order to process the events coming in. There
are several run modes:

- `run_synchronous`: processing one event at a time
- `run_threaded`: threaded mode using queues as synchronization tools
- `run_ray`: mode using ray to load balance events

### Synchronous mode

In synchronous mode, one event at a time is read from the input stream, then we
compute the result and write it to the output stream.

This simple mode is mainly used for debugging as the other modes are more
performant in general. In this mode only one input and one output stream are
possible.

### Asynchronous modes

![FM asynchronous data flow](/_static/images/financial_module/FM_asynchronous_data_flow.png)

This model can be used as a base for a multi-thread, multi-process or cluster
architecture, allowing a consistent architecture through diverse scaling
solutions.

By separating the reading and writing of the event from the computation itself
and using Queues, we prepare the work for other processes (gulcalc or other FM)
to communicate directly between them. This could remove the serialization /
deserialization steps necessary when using pipes.

The `queue_event_reader`, `event_computer` and `queue_event_writer` are based on
the classical producer-consumer pattern. This model also limits the time spent
waiting for IO. We use a sentinel to indicate to the consumer that all events
have been processed.

## Financial Structure

The purpose of this module is to parse the static input financial structure and
consolidate the information into simple objects ready to be used by the compute
function. The idea is to factorise all the computation and logic that can be done
at this step and prepare everything possible to have a generic way to handle the
computation for each item.

During the computation, computation data (loss, deductible, over limit, not
null, ...) is separated into 3 buckets (input, temp, output). Each computation
node is then assigned a bucket and an index, and the values for this node are
stored (during the computation) in a big matrix where the row corresponds to
their index.

A computation node then consists of a node id (layer, level, agg) and a
computation id that determines what will be done. In addition, several
dictionaries (maps) can provide additional information needed for the computation
step such as dependencies or profile.

![FM computation data structure](/_static/images/financial_module/FM_computation_data_structure.png)

This structure presents a number of advantages:

- the separation into 3 buckets minimizes the data transfer at the interface of
  the computation: only inputs are needed in and only outputs are passed out.
- input and output formats are just numpy arrays that are well supported and
  could be passed or retrieved by other processes.
- access to any node result is done with O(1) complexity.
- aggregations that are just intermediary in the program tree structure with no
  computation can take the index of their unique child dependency and cost no
  computation.
- new types of computation can be simply added to the generic compute structure.
  A computation node is simply a functional step with some input and some output.
- data are stored as numpy arrays that perform well with the most common
  operations we do: sums and multiplications.
- in case of bug or error, all the intermediary results can easily be stored for
  investigation.

### Inputs

The necessary static inputs for the Financial Module are expected to be in the
same folder and are:

- `fm_programme.bin`: the basic hierarchy of nodes organized by level and
  aggregation id
- `fm_policytc.bin`: the policy id to apply to each node and layer described in
  the programme
- `fm_profile.bin` (or `fm_profile_step.bin`): the profile (detailed values) of
  each policy id
- `fm_xref.bin`: the mapping between result items and the output ID

Inputs are read directly using `numpy.fromfile`, with a named dtype specific to
each file name. This allows access to each value in a row like a dictionary and
also provides a compatible interface for the two profile options.

We make a realistic assumption that the input and output data will fit in memory.

### Outputs

The transformed static information needed to build and execute the computation
for each event:

- `compute_queue`: list of all the computation steps to execute to compute an
  event (e.g. profile aggregation step, back-allocation)
- `node_to_index`: node to (bucket, index) mapping for each node needed to
  perform the computation. In this context an item in the programme can
  correspond to several nodes in the mapping — one node for the profile step,
  others for the back allocation, for example.
- `node_to_dependencies`: map to the list of nodes needed for this step's
  computation.
- `node_to_profile`: node to profile mapping
- `output_item_index`: output id to numpy array index mapping, needed to
  associate an output id to the computation result
- `storage_to_len`: dictionary of the size needed for each bucket
- `options`: computation options (is deductible computation needed, do we need
  to store intermediary sums, ...)
- `profile`: numpy ndarray of `fm_profile.bin`

## Computation

The computation module implements the actual computation of each event. It
creates all the numpy arrays necessary for the computation and executes, one by
one, all the steps present in the computation queue. Each step is treated
generically as a simple step to perform using the node computation id, without
pre-supposition of what has been done before. The correct ordering of the
computation steps is performed once, before the computation, by the financial
structure step.

There are several types of step:

- **PROFILE**: sum all sub-node arrays if present (loss, deductibles, over_limit,
  under_limit) then apply the calcrule in place.
- **IL_PER_GUL**: take the final IL and divide it by the sum of all input GUL.
- **IL_PER_SUB_IL**: take an item's back-allocated IL and divide it by the stored
  sum of IL computed before the calc rule was applied (this serves as a common
  basis for the back allocation of all sub-items).
- **PROPORTION**: multiply the loss array (IL) by the factor computed in
  IL_PER_GUL or IL_PER_SUB_IL.
- **COPY**: used to copy a vector from one index to another (copy to the output
  bucket in the a0 back-allocation rule, for example).

`event_computer` is a callable that takes an event's input from the input queue,
computes the output and puts it in the output queue.

## Policies

The policy module contains all the functions associated with the supported
policies. They all take the same numpy array as input and act directly on them
(in place).

Signature: `calc(policy, loss_out, loss_in, deductible, over_limit, under_limit)`

Loss is present in two arrays because in some cases we want to keep the sum value
before the calc rule is applied (if that is not the case, the arrays are in fact
the same object).

## Stream

The stream module is responsible for the parsing and writing of the GUL and FM
streams. The GUL stream is parsed and transformed into the numpy input array; the
FM stream is written from the numpy output array. In threaded mode the reader and
writer work in combination with the `event_computer` using Queues.

## Queue

In order to prepare for a more distributed approach, and also to reduce time
spent waiting for IO, the main mode for the FM calc is to use several threads for
input, output and computation. In order to terminate gracefully in case of error,
the standard Python queue is overwritten to implement a terminate function that
tells threads to stop if an error is caught.
