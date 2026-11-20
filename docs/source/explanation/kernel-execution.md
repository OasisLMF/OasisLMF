# How the loss kernel runs

The loss kernel is the part of a run that turns events into losses and output tables:
ground-up loss (`gulmc`), the financial module for insured and reinsurance losses (`fmpy`),
summaries (`summarypy`), the per-event tables (`eltpy`, `pltpy`), and the tables built over all
events (`katpy`, `aalpy`, `lecpy`). There are two ways to run it, and both produce the same
files.

## The bash script (default)

`generate-losses` writes a bash script, `run_kernel.sh`, and runs it. The events are split into
`N` partitions, where `N` is `kernel_num_processes`. Each partition gets its own pipeline of
processes connected by named pipes (FIFOs):

```
evepy k N | gulmc ... | tee ... | fmpy ... | fmpy -p input/RI_1 ... > fifo
summarypy < fifo   ->   tee   ->   eltpy / pltpy / work files
```

When every partition is done, `katpy` joins the per-partition tables and `aalpy` and `lecpy`
compute the AAL and the exceedance probability tables. {doc}`../reference/kernel/Workflows`
shows these pipelines tool by tool.

Every tool in every partition is its own Python process. Each one loads its own copy of what it
needs (the footprint, the vulnerabilities, the financial structures, the summary maps), and
passes its losses to the next tool as a byte stream. With many partitions, memory grows with
the number of processes, not with the work: on a 12-core machine a full PiWind kernel (ground
up, insured and reinsurance losses with every output) runs out of 14 GB above 6 partitions.

## The orchestrator (opt-in)

```{versionadded} NEXT
```

With the `orchestrator` option, the same run is made by a pool of worker processes instead of
the script. Each worker builds the whole chain once, as in-process stages (gulmc or `modelpy | gulpy`, the fm steps,
the summaries and the table writers), and then runs whole chunks of events through it. Losses
pass from one stage to the next as in-memory blocks, without pipes:

```
driver: chunk 1, chunk 2, ... chunk N   ->   worker 1: gulmc -> fm -> summary -> elt/plt
                                        ->   worker 2: gulmc -> fm -> summary -> elt/plt
                                        ->   ...
then the same workers run the katpy / aalpy / lecpy jobs
```

- **The events are split as bash splits them.** There are `N` chunks, holding the events
  `evepy k N` would give partition `k`, and the work files keep bash's names. A run made by the
  orchestrator therefore writes the same `work/` and `output/` files, byte for byte, as the
  script with the same settings.
- **Workers take the next chunk when they are free**, so a slow chunk does not hold up the
  others.
- **A resource manager sizes the pool.** It starts two workers, measures the memory each one
  really uses, and adds workers while the machine (or its container's memory limit) has room
  for another, up to one per chunk. Under memory pressure it retires workers between chunks
  instead of letting the run run out of memory. Its decisions are written to the run log.
- **No bash and no FIFOs.** The orchestrator's tests run on Linux, macOS and Windows.

On the same full PiWind kernel (12 logical CPUs, 14 GB limit), the orchestrator ran 12 chunks in
24 s and 2.9 GB at 10 samples, against 68 s and 11.7 GB for the script with 6 partitions; at 100
samples, 67 s and 5.3 GB against 135 s and 12.5 GB.

### What it runs and what falls back to bash

The orchestrator builds its plan from the same parameters the script is written from. When a
run needs something it cannot reproduce, it logs why and runs the script instead, so turning the
option on never stops a run. Supported, unsupported and ignored settings are listed in
{doc}`../how-to/run-kernel-with-orchestrator`.

### Distributed runs

On the platform a run is split into `generate-losses-chunk` steps (one per partition, each
writing `<k>.work/`) and a `generate-losses-output` step. The orchestrator can run any of these
steps, and its chunk files are interchangeable with those of the script, so a run can mix chunks
made each way.

## Where to read more

- {doc}`../how-to/run-kernel-with-orchestrator`: turning it on, checking it ran, logs.
- `docs/orchestrator-design.md` in the repository: the design, the measurements behind it, and
  how its results are tested against the script.
