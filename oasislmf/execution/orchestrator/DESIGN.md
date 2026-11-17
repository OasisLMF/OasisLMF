# Event-worker orchestrator: design plan

Status: phases 1 to 3 built (sections 8 and 9); phase 4 package 13, integration into the
generate-losses steps, built (section 10); packages 14 and 15 open.
Branch: `feature/event-worker-orchestrator`, from `main`. PR #2294 is closed; its branch
`feature/kernel-orchestrator` is kept as a source to port from (section 5, phase 4).

## 1. What this responds to

PR #2294 replaced the generated bash script with a Python supervisor, but kept the same
processes connected by the same pipes. The review feedback was that this is not enough on its
own, and that the orchestrator should change how the kernel works so that it also gets faster:

1. Try Ray: remove the direct piping and make `gulmc` and `fmpy` workers that "just do the next
   event we send them". Proof of concept on eve, gulmc and fm.
2. If Ray does not perform, a custom design: each process is a server, the orchestrator tells
   it over a socket what the next task is and where its data is, event data is passed as a
   shared-memory `byte_mv`, and the orchestrator adds or removes processes according to
   workload and available resources.

## 2. Where the time goes today (measured)

PiWind, 12,598 locations, 1,447 events, 10 samples, GUL + IL + RI with every ORD output, 12
partitions, warm Numba cache. CPU time sampled per process with `psutil`. The machine is an
Intel i7-1355U: 2 performance cores with hyper-threading and 8 efficiency cores, 12 logical
CPUs. Because the cores are not equal, CPU-seconds from a run that fills all twelve are not
directly comparable with those from a run that fits on the fast cores.

| Run | Wall | CPU | Processes |
|---|---|---|---|
| Full kernel script | 73 s | 699 s | 253 during the analysis (192 of them Python), 24 after |
| Same script, 12 events only (fixed cost) | 54 s | 523 s | same |
| `evepy \| gulmc \| fmpy` x 12 | 20.8 s | 192 s | 36 |
| `evepy \| gulmc \| fmpy \| fmpy (RI)` x 12 | 29.1 s | 273 s | 48 |
| `evepy \| gulmc \| fmpy` x 1 (all events) | 61.5 s | 70.7 s | 3 |

CPU per tool for the full script, and how much of it is there with only 12 events to compute:

| Tool | Processes | CPU (s) | Fixed (s) | Per-event work (s) |
|---|---|---|---|---|
| eltpy | 72 | 166 | 166 | ~1 |
| fmpy | 24 | 149 | 63 | 85 |
| gulmc | 12 | 120 | 67 | 53 |
| summarypy | 36 | 104 | 71 | 32 |
| pltpy | 36 | 84 | 83 | ~1 |
| katpy | 18 | 43 | 43 | - |
| evepy | 12 | 12 | 12 | - |
| aalpy + lecpy | 6 | 17 | 17 | - |
| tee | 60 | 5 | 0 | 5 |

What this says:

- **Most of the cost is starting processes.** About 54 of the 73 seconds, and about three quarters
  of the CPU, are spent on work that does not depend on the events: each process is a Python interpreter importing
  Numba and loading its compiled functions (1.5 to 4.8 CPU-seconds per tool when measured
  alone), and 192 of them start at once on 12 CPUs. The progress bar first moves about 30
  seconds in.
- **The same work costs 2.7x the CPU when split over 12 partitions**: one pipeline computes all
  events in 71 CPU-seconds, twelve need 192. Part of this is start-up (36 processes instead
  of 3) and part is the machine (see above), so it is an upper bound on what can be recovered.
- **Pipes themselves are cheap.** System time is about 8% of the total and `tee` is under 1%.
  Replacing pipes with shared memory will not by itself make a large difference. Having far
  fewer processes will.
- **Static partitions leave CPUs idle at the end.** Throughput falls from about 88 to about 40
  events per second over the last third of the run as partitions finish at different times.
- **Stages overlap very little inside one pipeline here**: 61.5 s wall for 69.4 s of gulmc plus
  fmpy CPU. Running the stages of one event one after another in a single process would lose
  about 13% on this portfolio.

Caveat: this is a small run, so fixed cost dominates. On a large model the per-event work
dominates and the gains come from dispatch and from not oversubscribing the CPUs. Section 5
lists the larger benchmarks.

## 3. Goals

- No stream piping between kernel tools. Each tool becomes a worker that computes the events it
  is handed.
- Events are dispatched dynamically, so no CPU idles while events remain.
- The number of worker processes follows the machine and the workload, and can change during
  a run.
- Faster than bash on both small and large runs, measured.
- The same numeric results as a bash run (see 4.3 for how that is tested).
- Nothing that only works on POSIX: no FIFOs, no `fork`-only code, no `select` on pipes.
- Opt-in, bash remains the default, and unsupported configurations fall back to bash with a
  warning, as in PR #2294.

Not in the proof of concept: summaries and ORD outputs, the output stage (kat, aal, lec),
complex-model gulcalc binaries, the load balancer, running across machines.

## 4. Design

Three layers. Layer 1 is needed whichever way events are dispatched, so it is built once and
the alternatives in layer 3 are compared on top of it.

```
layer 3  executor      ray | pool (fused or split)        who runs which stage on which event
layer 2  plan          stage graph from bash_params       gul -> il -> ri, with taps
layer 1  stages        GulmcStage, FmStage, ...           setup once, then one event at a time
```

### 4.1 Layer 1: stages

Each tool's `run()` is split at its event loop into a class that sets up once and then
processes events one at a time.

```python
# oasislmf/pytools/common/stage.py
class EventBlock:
    """Growable byte buffer holding whole events. write() appends; view() is a numpy int8 view."""

class GulmcStage:                      # oasislmf/pytools/gulmc/stage.py
    header: bytes                      # the 8 bytes gulmc writes before the first event
    def __init__(self, run_dir, sample_size, alloc_rule, ...): ...
    def process_event(self, event_id, out): ...          # out.write(bytes) zero or more times

class FmStage:                         # oasislmf/pytools/fm/stage.py
    header: bytes
    def __init__(self, static_path, allocation_rule, net_loss, ...): ...
    def process_block(self, block, gross_out, net_out): ...   # block holds one or more events
```

- **The unit of exchange is the event block**: exactly the bytes a tool writes for an event
  today, without the stream header. A stream is its header followed by its event blocks, so
  any stage boundary can be written to a file and compared byte for byte with what the bash
  pipeline produces at the same point.
- **The Numba kernels do not change.** Writers already call `stream.write(byte_mv[:cursor])`;
  they are given an `EventBlock` instead of stdout. Readers get one new method,
  `EventReader.read_block(byte_mv, n)`, which runs the existing `read_buffer` kernel over an
  in-memory block. This is the same path `read_streams` already takes for the last event of a
  stream.
- **The command-line tools keep working and are rebuilt on the stage**: `gulmc` becomes
  "build `GulmcStage`, read event ids from stdin, write blocks to stdout". One implementation,
  and the existing byte-identical tests for the bash path cover the refactor.
- `evepy` needs no stage. The orchestrator reads `events.bin` and calls the existing partition
  functions.

Later stages follow the same shape: `SummaryStage` (loss block in, one summary block per
summary set out), `EltStage` and `PltStage` (summary block in, rows appended to a file).

### 4.2 Layer 2: plan

A small description of the stage graph built from the same `bash_params` dict that PR #2294's
`build.py` consumes: `gul -> fm(il) -> fm(ri_1) -> ...`, with a tap wherever bash has a `tee`.
The proof of concept only needs the linear chain and a file sink at the end.

### 4.3 Chunks and determinism

The orchestrator splits the events into K chunks with the existing round-robin function, the
same one `evepy k K` uses. Chunks, not workers, name the outputs: chunk k writes what
partition k of a K-partition bash run writes.

The requirement is the same numeric results as bash: the same rows with the same values. The
order of rows in a file, and which partial file a row was written to, are not part of it. So K
is free to choose, independent of the worker count W, and should be a few times W so that
dispatch can balance load.

Chunks still give the strictest and cheapest test of that requirement:

> A run with any number of workers and K chunks is byte-identical to a bash run with K
> processes.

The tests use this with K set to the bash process count. It relies on a tool's output for an
event not depending on events processed in earlier chunks; the tests must prove that rather
than assume it. It does not hold for fmpy on `main` without a reset between chunks, see 8.2.
Where K differs from the bash process count, results are compared as sorted
rows, and aggregated outputs (AAL, EPT) with a floating-point tolerance, since they sum the
same numbers in a different order.

### 4.4 Layer 3: executors

Three candidates. The first two are the ones proposed in the review; the third is an addition.

**A. Ray.** One actor pool per stage. The driver submits `fm.run.remote(gul.run.remote(ids))`
per chunk, so blocks travel between actors as object-store references and are read as
zero-copy numpy arrays. Ray supplies scheduling, back-pressure (`ray.wait` on a bounded number
of in-flight chunks), restarts and the object store.

**B. Pool, split: one stage per worker (the custom design as proposed).** Worker processes are
servers holding one stage each. The orchestrator sends small control messages over a
`multiprocessing.connection` socket: which chunk, and where the input block is. Blocks live in
shared memory: each worker owns a `multiprocessing.shared_memory` arena it writes its output
into, and the message carries (segment, offset, length). The orchestrator sizes each stage's
pool from queue depth and CPU use.

**C. Pool, fused: the whole chain per worker.** The same worker and control protocol as B, but
each worker holds every stage and runs an event through all of them before taking the next:
`gul.process_event(e, block)`, then `fm.process_block(block, ...)`, and so on. Blocks never
leave the process, so there is no shared memory and no data IPC at all; the only messages are
"here is chunk k" and "chunk k done".

B and C are one implementation: a worker hosts a *segment* of one or more consecutive stages.
A block handed between stages inside a segment is local memory; a block crossing between
workers goes through shared memory. C is the configuration with one segment, B the one with a
segment per stage, and anything between (for example `gul` | `il + ri + summaries`) is a
configuration too.

| | A. Ray | B. Split pool | C. Fused pool |
|---|---|---|---|
| Processes for W CPUs | W, plus Ray's own | W | W |
| Data between stages | object store (shared memory) | shared-memory arenas | local memory |
| Cost per event besides compute | task and object bookkeeping | 2 messages per stage, cold cache | none |
| Balancing between stages | pool sizes to tune | pool sizes to tune | automatic |
| Working memory | per stage, per pool member | per stage, per pool member | every stage in every worker (as today) |
| Stages overlap within one event | yes | yes | no (13% on PiWind, see section 2) |
| Add or remove workers mid-run | actor pool resize | spawn or retire, per stage | spawn or retire |
| New dependency | `ray`, 79 MB wheel | none | none |
| Windows | beta | stdlib, supported | stdlib, supported |
| Other platform limits | no macOS x86_64 wheel; Python 3.14 experimental; needs `--shm-size` in Docker | shared-memory cleanup differs by OS | none known |
| Code to write and maintain | least | most | little |

**Decided.** Build layer 1, then A and C side by side, and measure. They share
everything except a few hundred lines each, so doing both costs little and answers the
question the review asks ("does Ray perform?") against a concrete alternative. Build B only
if the numbers call for it: if working memory per worker is the limit on a large portfolio, or
if runs with fewer events than CPUs matter enough that stage overlap is worth having.

The reason to expect C to be the fastest is section 2: the cost today is in process count and
idle CPUs, not in moving bytes. C removes both without adding any transport. Its two
weaknesses are listed in the table and should be measured, not argued.

### 4.5 The pool executor (B and C)

- **Workers** are started with `subprocess`/`spawn`, not `fork`, so the same code runs on
  Windows. On POSIX, importing once and forking W workers would cut start-up further; it is an
  optimisation to measure later, not part of the design.
- **Control protocol**, worker-driven: the worker connects and sends `ready`; the orchestrator
  replies with a chunk (its number and event ids) or `stop`; the worker replies `done` with
  counters and timings. Because workers pull, the fastest ones take the most chunks.
- **Elasticity**: a new worker simply connects and starts pulling; a worker is retired by
  answering its next `ready` with `stop`. The policy starts as "W = min(CPUs, K)". The
  mechanism is in place from the start so that a policy using `psutil` (free memory, idle CPU)
  can be added without touching workers. That policy is the resource manager of section 10.5.
- **Progress**: the orchestrator knows how many events are complete, so it sends the progress
  pings itself and gulmc's `--socket-server` flag is not needed.
- **Failure**: a worker that exits non-zero or reports an exception stops the run. The
  termination, stderr log and `CalledProcessError` handling from PR #2294's executor are
  reused, and the error names the stage and the event id.
- **Output**: for the proof of concept each chunk's final block is written to
  `work/<runtype>_P<k>.bin` with its stream header, to be compared with bash.

Shared memory, needed only for the split configuration: one arena per producing worker, used
as a ring of blocks; the consumer attaches to a segment once and acknowledges each block so
the space is reused; a producer waits when its arena is full; an event larger than the arena
gets its own segment. Known hazard to check: before Python 3.13 the resource tracker can
unlink a segment when a process that merely attached to it exits.

### 4.6 The Ray executor (A)

- `ray.init(num_cpus=W, include_dashboard=False)`, local only.
- Two topologies, to separate the cost of Ray from the cost of the topology: a pool per stage
  (the review's picture), and one actor running the whole chain (C on Ray).
- Blocks are returned as numpy arrays so that large ones go to the object store without a
  copy. Arrays read from the object store are read-only; to verify that the Numba readers
  accept them without a second compilation.
- `ray` becomes an optional extra (`oasislmf[ray]`), never a hard dependency.
- To measure in addition to throughput: `ray.init` time, memory, behaviour with a small
  `/dev/shm` as in a default Docker or Kubernetes container.

## 5. Phases

**Phase 0: harness.** A script that produces the reference per-chunk streams with bash
pipelines, one that compares them with an executor's output, and the per-tool CPU profiler
used for section 2.

Benchmarks. The PiWind variants in OasisModels have the same portfolio, events and sample
count as the run in section 2, so they add coverage of features, not size. Two larger
workloads:

- PiWind at 100 samples: same data, about ten times the per-event work, available now.
- UKWind (OasisModels): 10,000 events and a 20 million location exposure. Its footprint and
  exposure are downloaded by `get_large_data.sh`, and those URLs currently return 404, so it
  needs the files from another source.

**Phase 1: stages.** `stage.py`, `GulmcStage`, `FmStage`, `EventReader.read_block`, the command
line tools rebuilt on them. Done when the existing gulmc and fm test suites pass unchanged.

**Phase 2: proof of concept, eve + gulmc + fm (IL, then RI).** Ray executor and fused pool
executor, in parallel. Done when both are byte-identical to bash per chunk and both are
benchmarked against the numbers to beat:

| Scope | Samples | bash today, 12 partitions | One pipeline, all events |
|---|---|---|---|
| eve, gulmc, fm (IL) | 10 | 20.8 s wall, 192 CPU-s | 61.5 s wall, 71 CPU-s |
| plus fm (RI) | 10 | 29.1 s wall, 273 CPU-s | not measured |
| eve, gulmc, fm (IL) | 100 | 65.4 s wall, 523 CPU-s | 321 s wall, 346 CPU-s |
| plus fm (RI) | 100 | 107.2 s wall, 710 CPU-s | not measured |

At 100 samples the fixed cost is small, and twelve partitions still use 1.5 times the CPU of
one pipeline and finish only 4.9 times sooner. How much of that gap a better dispatch can
close, and how much is the machine's slow cores, is what the benchmark has to show.

Every benchmark needs a discarded warm-up run first: checking out another branch changes file
times, which invalidates the Numba cache, and a run that recompiles is two to three times
slower (168 s against 65 s for the 100-sample IL case).

**Decision point.** Choose the executor from the measurements. Build the split configuration
only if they call for it.

**Phase 3: the rest of the kernel.** Summary, ELT and PLT stages and the work files for aal and
lec, then the output stage. This is where most of the fixed cost in section 2 is (eltpy, pltpy
and summarypy are 144 of the 192 Python processes), so it is where most of the gain on small runs
will come from. A rough estimate for the PiWind run above is 25 to 30 seconds against 73; it
is an estimate from the measured per-event work, not a result.

**Phase 4: integration.** The `--orchestrator` option, fallback rules, chunked platform runs
(`generate-losses-chunk`), a manifest for `rerun`, Windows and macOS CI. Ported from
`feature/kernel-orchestrator`: the option and fallback in `runner.py` and `losses.py`, process
termination and the stderr guard from `executor.py`, and the `NO_SELECT` change to the
pytools. Not ported: the pipe transport, `teepy` and `stream_ref`.

## 6. Work packages for implementation

The stage and block interfaces in 4.1 are fixed first; after that the packages touch different
files and can be done in parallel.

| # | Package | Files | Depends on |
|---|---|---|---|
| 0 | Harness, comparison tool, test fixtures | `tests/orchestrator/` | - |
| 1 | `EventBlock`, chain, chunks, inline executor, CLI | `pytools/common/stage.py`, `execution/orchestrator/` | - |
| 2 | `GulmcStage`, `gulmc` CLI on top of it | `pytools/gulmc/` | - |
| 3 | `FmStage`, `EventReader.read_block`, `fmpy` CLI on top of it | `pytools/fm/`, `pytools/common/event_stream.py` | - |
| 4 | Pool executor and worker (fused) | `execution/orchestrator/{pool,worker}.py` | 1 |
| 5 | Ray executor | `execution/orchestrator/ray_executor.py` | 1 |
| 6 | Benchmarks and report | scripts | 2 to 5 |

Packages 4 and 5 can be developed against fake stages that echo bytes, then joined with 2
and 3.

## 7. Decisions and open questions

Decided:

- Ray and the fused pool are built side by side, not one after the other.
- The requirement is the same numeric results as bash, not the same bytes (4.3).
- PR #2294 is closed. This work is on its own branch from `main` and ports what it needs.
- **The fused pool (C) is the executor** (2026-11-16). Phase 2 benchmarks
  (`~/GitHub/runs/orch-bench-piwind/bench/REPORT.md`): fastest or tied in every scope
  (1.10x to 1.45x over bash x12), least memory, no new dependency. Ray fused does the same
  work but pays about 10 s of fixed cost; Ray staged never wins and runs out of memory with
  whole-chunk blocks. The split configuration (B) is not built. The Ray executor is kept, as
  the record of the comparison, on branch `feature/event-worker-orchestrator-ray` and is not
  part of this branch.
- **Runs with fewer events than CPUs** (formerly open question 2): measured, not a problem. A
  fused worker is about 22% faster than the bash pipeline on a single event (14 s against
  18 s at 1,000 samples) because the pipe boundaries cost more CPU than the overlap saves.

Open:

3. **UKWind data.** Where to get the footprint and exposure now that the download URLs fail.
4. **Complex models.** A supplier's gulcalc binary cannot be loaded as a stage. Either an
   adapter that reads its stream and cuts it into blocks (static partitions for that stage),
   or those runs keep using bash.

## 8. Interface contract for the proof of concept

Fixed before implementation so that the work packages can proceed in parallel. A package may
add to its own module but must not change a signature below without updating this section.

### 8.1 Blocks and sinks

- A **block** is a 1-D numpy `int8` array holding one or more whole events of a loss stream,
  without the stream header. It may be read-only.
- A **sink** is any object with `write(data)`, where `data` supports the buffer protocol
  (`bytes`, `memoryview`, numpy array). `io.BytesIO`, a binary file and `EventBlock` are sinks.
  A sink must copy what it is given before `write` returns: stages hand it a view of a buffer
  they reuse.

```python
# oasislmf/pytools/common/stage.py                                              (package 1)
class EventBlock:
    def __init__(self, capacity=2 * PIPE_CAPACITY): ...
    size: int                      # bytes currently held
    def write(self, data) -> int   # append, growing by doubling; returns bytes appended
    def view(self) -> np.ndarray   # int8 view of the first `size` bytes, no copy;
                                   # invalid after the next write() or clear()
    def clear(self) -> None
    def __len__(self) -> int
```

### 8.2 Stages

```python
# oasislmf/pytools/gulmc/stage.py                                               (package 2)
class GulmcStage:
    def __init__(self, run_dir, sample_size, loss_threshold, alloc_rule, debug, random_generator,
                 ignore_file_type=(), peril_filter=(), data_server=None, ignore_correlation=False,
                 ignore_haz_correlation=False, effective_damageability=False,
                 max_cached_vuln_cdf_size_MB=200, model_df_engine=..., dynamic_footprint=False): ...
    header: bytes                               # what gulmc writes before its first event
    def process_event(self, event_id, out): ... # zero or more out.write() calls for this event
    def close(self): ...                        # also usable as a context manager

# oasislmf/pytools/fm/stage.py                                                  (package 3)
class FmStage:
    def __init__(self, input_header, allocation_rule, static_path, gross=True, net=False,
                 low_memory=False, sort_output=False, storage_method='sparse', **kwargs): ...
    header: bytes                               # what fmpy writes before its first event
    def process_block(self, block, gross_out=None, net_out=None): ...
    def reset(self): ...                        # forget what earlier events left behind
    def close(self): ...                        # also usable as a context manager

# oasislmf/pytools/common/event_stream.py                                       (package 3)
class EventReader:
    def read_block(self, byte_mv, n=None): ...  # generator of event ids, as read_streams yields
```

- `input_header` is the upstream stage's `header`; `FmStage` validates it as fmpy validates
  its input stream and takes the sample count from it.
- `FmStage.process_events(events, gross_out, net_out)` is the same loop driven by any iterator
  of loaded events from `FmStage.reader`; `process_block` is `process_events` over
  `reader.read_block(block)`, and the fmpy command uses it over `reader.read_streams(...)`.
- A read-only block makes Numba compile a second version of each reader kernel (about 7 s for
  fm, once, then cached on disk). Executors should pass writable blocks where they can.
- `process_block` handles the events of the block in order. For each it writes the gross
  bytes to `gross_out` and the net bytes to `net_out`, whichever are not None. `gross` and
  `net` in the constructor say which of the two will ever be asked for.
- `reset()` makes the stage behave like a freshly built one for the events that follow.
  Executors call it at the start of every chunk (`Chain.reset()` calls it on every stage that
  has one). It exists because fmpy on `main` keeps chance-of-loss values (`pass_through`) from
  earlier events and writes them out again for later ones, so the bytes of a bash partition
  depend on it being a fresh process. Resetting per chunk reproduces exactly that. If the
  leak is fixed on `main`, `reset()` has nothing left to do.
- Paths resolve exactly as in the command-line tools, so a stage is used with the run
  directory as the working directory.
- The command-line `run()` functions are rebuilt on the stages. Their output must not change
  by a single byte.

### 8.3 Chain, chunks and executors

```python
# oasislmf/execution/orchestrator/chain.py                                      (package 1)
@dataclass
class ChainSpec:           # JSON-serialisable: sent to worker processes and Ray actors
    run_dir: str
    gul: dict              # GulmcStage keyword arguments
    fm: list               # one dict per fm step, in order: FmStage keyword arguments (without
                           # input_header) plus 'name' ('il', 'ri') and 'feeds_next' ('gross'|'net')

class Chain:
    def __init__(self, spec, factories=None): ...   # factories replace the real stages in tests
    headers: dict                                   # tap name -> stream header bytes
    def run_event(self, event_id, taps): ...        # taps: dict tap name -> sink
    def reset(self): ...                            # reset() every stage that has it; called at each chunk start
    def close(self): ...

# oasislmf/execution/orchestrator/chunks.py                                     (package 1)
def make_chunks(events, n_chunks, event_shuffle=EVE_DEFAULT_SHUFFLE): ...
                                         # list of arrays; chunk k (1-based) holds exactly the
                                         # events `evepy k n_chunks` emits, in the same order
class ChunkWriter: ...                   # writes <out_dir>/<tap>_P<k>.bin = header + blocks
```

- Taps are `gul`, `il` (gross of the IL step) and `ri` (net of the last RI step): the three
  streams the bash pipeline `gulmc | tee gul | fmpy -a2 | tee il | fmpy -a3 -p input/RI_1 -n -`
  produces.
- Every executor is a module exposing
  `run_chunks(spec, chunks, out_dir, taps, n_workers, **options) -> dict` which writes
  `<out_dir>/<tap>_P<k>.bin` for every chunk and requested tap and returns statistics (wall
  time, per-chunk and per-worker timings). `inline.py` (package 1) runs the chunks in the
  calling process; `pool.py` with `worker.py` (package 4) and `ray_executor.py` (package 5)
  are the two candidates.
- `cli.py` (package 1): `python -m oasislmf.execution.orchestrator.cli --executor
  inline|pool|ray --run-dir D --chunks K --workers W --out-dir O --taps gul,il,ri` plus the gul
  and fm options, building the `ChainSpec` from its arguments. Executors are looked up by name
  so that `ray` is imported only when asked for.
- `inline.run_chunks` also takes `chunk_order` (1-based chunk numbers, a permutation of 1..K),
  exposed as `cli.py --chunk-order reverse|shuffle:<seed>|<k,k,...>` for the inline executor
  only. It changes the order the chunks run in through the one chain, never which file a chunk
  writes, and exists to test that `Chain.reset()` makes a chunk's bytes independent of the
  chunks run before it.

### 8.4 Reference data

`~/GitHub/runs/orch-bench-piwind` is a prepared PiWind run directory (12,598 locations, 1,447
events). `ref-bin/` holds wrappers that run `evepy`, `gulmc` and `fmpy` from a pristine checkout
of `main` (`~/GitHub/OasisLMF-ref`), and `make_refs.sh K SAMPLES` writes the reference streams
`ref/K<K>_S<SAMPLES>/{gul,il,ri}_P<k>.bin` with them. `ref/K12_S10` exists. An executor run with
K chunks must reproduce those files byte for byte.

## 9. Phase 3 plan: summaries, ELT, PLT and the output stage (draft, for discussion)

### 9.1 What bash does after fm today

Per partition k and per run type (gul, il, ri), for each summary set n:

```
<rt>_P<k> -> summarypy -t <rt> [-p input/RI_1] -n <fifo>                 one stream per summary set
          -> tee -> pltpy -E bin -s/-q/-m work/kat/<rt>_S<n>_plt_{sample,quantile,moment}_P<k>
                 -> eltpy -E bin -q/-m   work/kat/<rt>_S<n>_elt_{quantile,moment}_P<k>
                 -> eltpy -E bin -s      work/kat/<rt>_S<n>_elt_sample_P<k>
                 -> work/<rt>_S<n>_summary_palt/P<k>.bin                (aalpy input)
                 -> work/<rt>_S<n>_summaryleccalc/P<k>.bin              (lecpy input)
```

After every partition has finished: `katpy` concatenates the `work/kat/*_P<k>` files into
`output/<rt>_S<n>_{splt,qplt,mplt,selt,qelt,melt}.csv`, `aalpy -K` reads `*_summary_palt/` into
`output/<rt>_S<n>_palt.csv`, and `lecpy -K` reads `*_summaryleccalc/` into the EPT/PSEPT files.
Per partition that is 4 Python processes per run type and summary set (summarypy, pltpy,
2 x eltpy) plus a `tee`: 144 of the 192 interpreters in the PiWind run in section 2.

### 9.2 Design

The fused worker's chain grows from a line into a tree, still entirely in-process:

```
gul -> fm(il) -> fm(ri)
 |       |         |
 S(gul)  S(il)     S(ri)          SummaryStage: one summary block per summary set
 |                                each summary block feeds:
 +-> Plt, Elt(q,m), Elt(s)        PltStage / EltStage: rows -> per-chunk work/kat files
 +-> palt file, leccalc file      plain file sinks, bytes exactly as tee writes them
```

- **Same pattern as phase 1.** `SummaryReader`, `ELTReader` and `PLTReader` are already
  `EventReader`s, so `read_block` carries over. Each tool's `run()` is split at its event loop
  into a stage, the command line is rebuilt on it, and its output must not change by a byte.
- **Per-chunk outputs.** ELT/PLT stages write files named by the chunk, so stages that own
  files get `start_chunk(paths)` / `finish_chunk()` (open, write headers, flush, close), called
  by the chain at chunk boundaries next to `reset()`. All other sinks stay as in 8.1.
- **The plan comes from `bash_params`**, the same dict `genbash` consumes (layer 2 in 4.2):
  run types, summary sets, which ORD outputs, `-p input/RI_n`, and the exact work file names
  bash uses. That keeps `katpy`, `aalpy` and `lecpy` unchanged and makes the work directory
  of an orchestrated run interchangeable with a bash one.
- **Output stage.** After the last chunk, the katpy/aalpy/lecpy jobs (one per output file,
  about 25 for PiWind) run as tasks on the same pool: the worker protocol gains a `job`
  message beside `chunk`, and each job calls the tool's `run()` in-process. No new processes.
- **What is not covered** falls back to bash, as decided for phase 4: the legacy ktools C++
  outputs (if `genbash` still emits any for some settings), supplier gulcalc binaries, and
  anything the planner does not recognise. The planner says why it fell back.

### 9.3 Correctness

- Per chunk: every `work/kat/*_P<k>` and `work/*_summary*/P<k>.bin` file byte-identical to a
  bash run with K partitions (`genbash` + `run_kernel.sh` with the pristine tools).
- End to end: `output/` byte-identical to that bash run. With K different from the bash
  partition count, ELT/PLT rows compared sorted, AAL/EPT with a floating-point tolerance.
- summarypy ignores the fm chance-of-loss rows (#2300), so outputs do not depend on
  `reset()`; the work files after summarypy should not either, which the tests check.

### 9.4 Work packages

| # | Package | Files | Depends on |
|---|---|---|---|
| 7 | Harness: full-kernel bash reference (all ORD outputs, PiWind + test fixture), `work/` and `output/` comparison | `tests/orchestrator/`, run dir scripts | - |
| 8 | `SummaryStage`, summarypy CLI on it | `pytools/summary/` | - |
| 9 | `EltStage`, `PltStage`, eltpy/pltpy CLIs on them | `pytools/elt/`, `pytools/plt/` | - |
| 10 | Chain as a tree, per-chunk file stages, planner from `bash_params` | `execution/orchestrator/{chain,plan}.py` | 8, 9 interfaces |
| 11 | Output stage: katpy/aalpy/lecpy jobs on the pool | `execution/orchestrator/{pool,worker,outputs}.py` | 10 |
| 12 | Benchmark: full PiWind kernel (73 s today, estimate 25 to 30 s) | `bench/` | 7 to 11 |

7, 8 and 9 run in parallel once the 9.2 interfaces are written into section 8; 10 starts
against fake stages.

### 9.5 Interface contract for phase 3

Same rules as section 8: a package may add to its own module but must not change a signature
below without updating this section. Blocks and sinks are as in 8.1.

```python
# oasislmf/pytools/summary/stage.py                                             (package 8)
class SummaryStage:
    def __init__(self, input_header, static_path, run_type, summary_set_ids, low_memory=False,
                 output_zeros=False, **kwargs): ...
    headers: dict                                  # summary_set_id -> the 12-byte header summarypy
                                                   # writes to that set (it carries the set id);
                                                   # `header` works only with a single set
    def process_block(self, block, outs, idx_outs=None, row_writers=None): ...
                                                   # outs: dict summary_set_id -> sink; for each
                                                   # event, the bytes summarypy writes to set n.
                                                   # idx_outs: the .idx sinks, required with
                                                   # low_memory (summarypy -m, used by bash when
                                                   # summarypy_low_memory is set). row_writers:
                                                   # csv/parquet SELT rows instead of outs.
    def reset(self): ...                           # restarts the .idx offsets: call per output
                                                   # file. No cross-event state in the losses.
    def close(self): ...

# oasislmf/pytools/elt/stage.py                                                 (package 9)
class EltStage:
    def __init__(self, run_dir, input_header, selt=False, melt=False, qelt=False,
                 output_format='bin', noheader=False): ...
    def start_chunk(self, paths): ...              # paths: dict 'selt'|'melt'|'qelt' -> file for this
                                                   # chunk; opens them and writes headers as eltpy does
    def process_block(self, block): ...            # summary block in, rows appended to the open files
    def finish_chunk(self): ...                    # flush and close: files equal eltpy's, byte for byte
    def close(self): ...

# oasislmf/pytools/plt/stage.py                                                 (package 9)
class PltStage:                                    # as EltStage, with 'splt'|'mplt'|'qplt'
    def __init__(self, run_dir, input_header, splt=False, mplt=False, qplt=False,
                 output_format='bin', noheader=False): ...
    def start_chunk(self, paths): ...
    def process_block(self, block): ...
    def finish_chunk(self): ...
    def close(self): ...

# oasislmf/execution/orchestrator/plan.py                                       (package 10)
def plan_from_bash_params(bash_params) -> 'Plan': ...   # raises Unsupported(reason) when a
                                                        # setting needs the bash path
@dataclass
class Plan:
    spec: ChainSpec        # gains 'summaries': one node per summarypy invocation (run type and
                           # inuring priority), mirroring bash: its source (gul, fm step gross
                           # or net), SummaryStage kwargs, and per summary set the Elt/Plt stage
                           # kwargs, plain work files and .idx files, as path templates
                           # containing '{k}' that equal the names bash uses
    output_jobs: list      # OutputJob(tool, kwargs): katpy/aalpy/lecpy runs, in any order
```

- `Chain` gains `start_chunk(k)` and `finish_chunk()`. `start_chunk(k)` resets every stage that
  has `reset()` and calls `start_chunk(paths)` on every file-owning stage with its templates
  filled in for chunk k; `finish_chunk()` calls theirs. Executors call these two instead of the
  bare `Chain.reset()`, which is the only change the inline and pool executors need.
- The palt and leccalc work files are plain file sinks fed with the summary stream exactly as
  `tee` writes them: the summary header, then the blocks.
- The gul/il/ri taps of 8.3 stay, for testing.
- Output jobs run after every chunk is done (package 11); the pool sends them as `job` messages
  and the worker calls the tool's `run()` with the job's kwargs, the run directory as cwd.

### 9.6 Full-kernel reference data

`~/GitHub/runs/orch-bench-piwind-full` holds the same PiWind portfolio prepared for a full
kernel with every ORD output (summary sets S1 to S3, ELT, PLT, ALT, mean-only ALT, EPT/PSEPT,
parquet for one set). `make_refs.sh K S` writes `ref/K<K>_S<S>/` with pristine `main`; each
reference is a runnable run directory with its `bash_params.json`, `run_kernel.sh`, `work/`
and `output/`. `ref/K4_S10` exists (K = 4 because a bash full kernel starts about 15 Python
processes per partition; K4 peaks at 8.5 GB). bash runs gulmc there with `-a0`, so these
references are not interchangeable with the stream references of 8.4 (`-a1`). A run with
`summarypy_low_memory` needs a low-memory reference: lecpy sums differently when `.idx` files
exist and EPT/PSEPT differ in the last digits. `tests/orchestrator/tools/compare_outputs.py
bytes|rows` compares two run directories.

Result so far: the pool executor, 4 workers, `--plan ref/K4_S10/bash_params.json`, writes all
216 per-partition work files byte-identical to the reference.

Output stage, as built (package 11): executors take `output_jobs`; after the last chunk the
pool answers `ready` with `job` messages (index, tool, kwargs), and a worker closes its chain
before its first job to free the chain's memory. `outputs.run_job` calls the tool's `main()`
in-process with the run directory as cwd and logging isolated per job. `check_independent`
rejects jobs that share outputs, read each other's outputs or share a `work/` subfolder,
before any chunk runs. Stats gain `jobs`, `output_seconds` and per-worker `job_seconds`.
PiWind K4 through the pool, 4 workers: `work/` and `output/` byte-identical to `ref/K4_S10`
(288 files); 54 output jobs take 1.3 s.

### 9.7 Phase 3 result: full PiWind kernel

Package 12 (`~/GitHub/runs/orch-bench-piwind-full/bench/REPORT.md`, 2026-11-17, every run
capped at 14 GB). Full kernel, S10: the pool with 12 workers and 12 chunks takes 24.0 s,
218 CPU-s and 2.9 GB, within the 25 to 30 s estimate above; DESIGN section 2's bash K12 took
73 s and 699 CPU-s. Under the cap bash K8 and K12 run out of memory (about 1.9 GB per
partition, about 15 Python processes each), so the largest bash that fits is K6: 67.7 s,
669 CPU-s, 11.7 GB. At S100: pool 66.9 s / 5.3 GB against bash K6 135.1 s / 12.5 GB. At the
same process count (pool W4 K4 against bash K4) the pool is 1.76x faster at S10 with a
quarter of the CPU and an eighth of the memory, byte-identical. All checked runs are equal to
bash (bytes at K4 and K6, sorted rows at K12 and K48).

Remaining time at S10 (24 s): about 8 s fixed cost (driver, then twelve workers importing
Numba and building chains), 1 s of output jobs, and about 4 s of end-of-run imbalance between
P- and E-cores. K48 did not remove the imbalance; handing out the largest chunks first, or
reusing workers' setup, are the next things to try.

## 10. Phase 4: integration

| # | Package | Files | State |
|---|---|---|---|
| 13 | `orchestrator` option on the generate-losses steps; `runner` builds the plan from the same `bash_params` and runs it with the pool executor (W = K = `max_process_id`), falling back to bash with a logged warning on `Unsupported`; chunked platform runs (`process_number` = one chunk, outputs step = the output jobs); progress pings from the orchestrator; logs and exit codes as bash | `computation/generate/losses.py`, `execution/runner.py`, `execution/bash.py`, `execution/orchestrator/{kernel,progress}.py` | done |
| 14 | CI: the orchestrator suites on Linux, Windows and macOS; stderr guard, chunk output check and resource monitor mirrored | `.github/workflows/`, `tests/`, `execution/orchestrator/{pool,kernel}.py`, `execution/resource_monitor.py` | done, CI unverified (10.4) |
| 15 | Manifest for `rerun`, if still needed once 13 exists | - | not needed so far, see 10.3 |
| 16 | Resource manager: the pool sized from measured memory and CPU (10.5) | `execution/orchestrator/{resources,pool,kernel,cli}.py` | done |

Ported from `feature/kernel-orchestrator`: the option and fallback wiring in `runner.py` and
`losses.py` (`genbash_params`, the warning, `CalledProcessError` on failure) and the shape of its
tests. Not ported: `build.py`, the pipe transport, `teepy`, `stream_ref`, `NO_SELECT`, the
manifest.

### 10.1 What was built (package 13)

- **Option.** `orchestrator` (default False, `--orchestrator` on the command line) on
  `GenerateLosses` (so `oasislmf model run` and `model generate-losses`), `GenerateLossesPartial`
  (`generate-losses-chunk`) and `GenerateLossesOutput` (`generate-losses-output`). It reaches
  `runner.run(orchestrator=True)`, or `bash_params['orchestrator']` for `run_analysis` and
  `run_outputs`, only when set, so supplier runners see nothing new.
- **Same parameters as bash.** `bash.genbash_params` returns the dict `genbash` writes its script
  from (`genbash` now calls it). `runner.run` builds it with exactly the arguments it gives
  `genbash`; `run_analysis` and `run_outputs` already receive bash_params. The planner runs on
  that dict with the script's directory as run directory.
- **Fallback.** `runner.orchestrator_plan` returns None on `Unsupported` after logging
  `The orchestrator cannot run this analysis (<reason>): running the kernel with the bash script
  instead`; the step then runs exactly as without the option. A caller's
  `custom_get_getmodel_cmd`, a custom gulcalc binary (`model_custom_gulcalc` or an inferred
  `<supplier>_<model>_gulcalc`) all arrive as `_get_getmodel_cmd` and fall back with the planner's
  custom gulcalc reason.
- **The script is still written** (`run_kernel.sh`, `<k>.run_analysis.sh`, `run_outputs.sh`),
  but not run: it records what ran and `rerun()` keeps working on it unchanged, so package 15's
  manifest is not needed for that.
- **Execution** (`orchestrator/kernel.py`, `run_plan(plan, bash_params, mode, log_dir, script)`):
  mode `all` (every chunk, then the output jobs), `analysis` (every chunk, or chunk
  `process_number` alone; no jobs) or `outputs` (jobs only, workers are job-only). Pool executor,
  W = K = `max_process_id`; W = 1 for one distributed chunk; W = `max_process_id` job workers for
  the outputs step. Since package 16 these W are the hard maximum of the resource manager (10.5). Side effects as the scripts: the log directory is emptied
  (`bash_wrapper`), the analysis removes earlier output files except summary-info and JSON and
  empties its work directory, the outputs empty `work/` afterwards when
  `remove_working_files` is set (`genbash` always sets it; `GenerateLossesOutput` still passes
  the misspelt `remove_working_file`, which #2292 fixes on main; the bash_params value is what
  counts).
- **Logs.** In the step's log directory (`log`, `log/<k>`, `log/out`):
  `orchestrator_worker_<n>.log` and `orchestrator_worker_<n>.err` (each worker's stdout and
  stderr, kept: pool option `log_dir`), `orchestrator.log` (the run log, also returned as the script's stdout would be:
  per chunk events/seconds/worker, per output job, totals), `stderror.err` (created empty as the
  script's redirections do; a failure is appended to it), and the output jobs' pytools logs
  (`OASIS_PYTOOLS_LOG_DIR` and `OASIS_PYTOOLS_LOG_LEVEL` set for the workers). The
  `ResourceMonitor` is started and stopped around the run as for bash; it reports the pool
  workers as one tool, `orchestrator` (10.3).
- **Errors.** Any executor exception (`WorkerError` naming stage, chunk, event or output job,
  with the worker's stderr tail) becomes `subprocess.CalledProcessError(1,
  ['oasislmf-orchestrator', mode, script], output=<run log>)`, so `GenerateLosses*` print
  `log/.../stderror.err` and raise the same `OasisException` as for a failed script, and
  `runner.run`'s caller calls `rerun()` as before. The pytools-log completeness check runs after
  an orchestrated step too.
- **Distributed chunks.** The planner no longer rejects `process_number`: the plan of
  generate-losses-chunk k uses bash_params' `work_dir` (`<k>.work/`), so its templates name
  exactly bash's files for partition k (`<k>.work/kat/*_P<k>`, `<k>.work/*_summary*/P<k>.bin`),
  and `Plan.chunk_numbers` is `[k]`. The executors take `chunk_numbers` (run only those chunks of
  the K the events are split into). The load-balancer check is skipped there because the chunk
  script disables the load balancer too. The outputs step plans from its own bash_params (no
  `process_number`), so its jobs read `work/`, where the platform merges the chunks' `k.work/`.
- **Progress** (`orchestrator/progress.py`). bash's `gulmc --socket-server` sends
  `{"events_complete": <new events>, "analysis_pk": pk}` through `oasis_ping` at most every
  `SERVER_UPDATE_TIME` s and once at its end; without an `analysis_pk` it pings the local
  `socket-server`, which draws the "Gul events completed" bar. Workers now send `progress`
  messages (events done in the current chunk, at most every `progress_interval` s; the pool
  counts them against the chunk's `done`), and `ProgressReporter` in the orchestrator process
  sends the same pings from a background thread (a failed ping is retried with the next, so
  the platform's sum is exact) when bash_params has an `analysis_pk`, or draws the same bar
  in-process when it has `socket_server_size` (local runs; no socket server is started). The
  initial `events_total` ping stays in `losses.py`, unchanged.

### 10.2 Results

- Tests: `tests/orchestrator` 298 passed (2 strict xfails, #2300), `tests/model_execution` all
  pass (two `CheckConversionTools` tests need a `python` executable on PATH, environment only),
  `tests/computation/test_generate_losses.py` + `test_computation.py` pass except five
  bash-only `test_losses__run_*` runs at K = CPU count that exceed the 8 GB test cap here.
- Fixture (tests/orchestrator, K = 2): `runner.run` with the option on and off, `output/`
  byte-identical (`test_kernel.py`). Chunked: chunk 1 by bash and chunk 2 by the orchestrator,
  and the reverse, each chunk's `k.work/` byte-identical between the two, outputs (one run's by
  the orchestrator, the other's by bash, one keeping and one removing `work/`) byte-identical
  to the whole run; the pings of the orchestrated chunks sum to their event count. The plan of
  chunk k renders exactly the commands of bash's `<k>.run_analysis.sh` (`test_plan.py`).
  Through `OasisManager` (`tests/computation/test_generate_losses.py`): generate-losses on and
  off (RI, RI + RL multi-layer) byte-identical, the 3-chunk platform workflow on and off
  byte-identical, an unsupported setting falls back with the warning, a failed output job
  raises the usual kernel `OasisException`.
- PiWind (`~/GitHub/runs/orch-bench-piwind-full`, full kernel, K = 4, S10): `runner.run` with
  the option on (4 workers, 4 chunks), `output/` byte-identical to `ref/K4_S10` (72 files),
  31.9 s for the whole step including 54 output jobs (1.3 s) and 4.6 s of worker setup. The run
  directory is `phase4/K4_S10`: the reference's `input` and `static`, its summary-info files, and
  `runner.run(settings, number_of_processes=4, num_reinsurance_iterations=1, orchestrator=True)`.

### 10.3 What was built (package 14 and the package 13 gaps)

- **Stderr guard.** bash's guard (`stderr_guard`, off with `kernel_disable_guard`) appends every
  guarded command's stderr to `stderror.err`; `oasis_exec_monitor.sh` checks that file every 2 s
  and kills the script (exit status 143 through the trap) when it holds a line other than a blank
  line, a Python warning header (`<file>:<line>: <Category>Warning:`) or an indented
  `warnings.warn` source line, and a chunk script checks it once more before it exits (exit 1,
  `Error detected in $LOG_DIR/stderror.err`). The orchestrator now does the same with the pool:
  each worker's stderr goes to `orchestrator_worker_<n>.err`, the pool checks those files every
  2 s and once after the last worker exits with the same pattern (`pool.stderr_errors`), and a
  line it keeps fails the run (`WorkerError`, "pool worker n wrote to stderr: <line>", with the
  stderr tail). `run_plan` appends the workers' stderr to `stderror.err` in worker order, before
  the failure line, so `GenerateLosses*` print what bash would have printed there.
- **Chunk output check.** A distributed chunk script fails (exit 1, `Chunk output error: File
  '<path>' is empty`) when a `P<k>.bin` under `<k>.work` is empty. `run_plan` checks the same
  files after an `analysis` step with `process_number` and the guard on. An orchestrated chunk
  writes the summary header into every such file at `start_chunk`, so this only fires if a stage
  never started; it is there to keep the contract.
- **Deviations.** Exit status 1 for every failure (bash: 143 for a monitor kill, 1 for the chunk
  checks), and the message is in `stderror.err` and the run log, not on stdout. The check is also
  made at the end of a whole run (bash only checks at the end of a chunk script, so a whole run
  could finish before the monitor's next poll). The inline executor runs in-process and has no
  stderr of its own: the guard applies to the pool only. Workers' stdout is not checked (bash
  does not check it either).
- **Resource monitor.** `monitored_tool` reports a process whose command line has the module
  `oasislmf.execution.orchestrator.worker` as `orchestrator`; the stages run in-process, so
  there is no per-tool split. `MONITORED_TOOLS` is unchanged, so the pytools-log completeness
  check does not look at `orchestrator_worker_<n>.log`.
- **Tests without bash.** Marker `requires_bash` (`setup.cfg`, `tests/bash_support.py`,
  `tests/conftest.py`): tests that run bash scripts or pytools pipelines over OS pipes (the tools
  `select` on them, which Windows only allows on sockets), applied explicitly or through the
  fixtures `kernel_reference`, `full_kernel_reference` and `bash_whole_run`, and skipped on
  Windows, on macOS (a pytool whose output is a regular file fails there, 10.4) or with
  `OASIS_TEST_NO_BASH` set. 18 of the 324 tests in the CI selection are marked; two more ask for
  `kernel_reference` only when `bash_available()`, so they compare with bash on Linux alone.
  Still run without bash: the runner's orchestrated whole run and a fully orchestrated chunked run
  compared with it, pool against inline streams with real stages, plans from `bash_params`,
  chunking, fake-stage executor tests. The fixture's run directory no longer needs the bash tool
  wrappers, and links `input`/`static` or copies them where symbolic links are not allowed.
- **CI.** Linux: `unittest.yml` already runs every test, the bash comparisons included, on
  Python 3.10 to 3.14 (JIT on) and 3.12 (JIT off). macOS and Windows: an `Orchestrator tests`
  step in `test-macos.yml` and `test-windows.yml` (Python 3.12, against the built wheel):
  `tests/orchestrator`, `test_resource_monitor.py`, `test_runner.py::TestRunWithOrchestrator`
  and `test_generate_losses.py -k orchestrator`, with `-m "not requires_bash"`.
- **Windows by reading.** Workers are started with `subprocess` (no fork), connect over
  `multiprocessing.connection` on 127.0.0.1 (sockets, which `wait` supports on Windows), read
  their configuration from a stdin pipe, and are terminated with their process trees through
  psutil (`TerminateProcess`); every data file is opened in binary mode and is a plain file (no
  FIFO, `/dev/shm` or `select`). Nothing found needed a change in the executor; the test fixture
  did (bash wrappers, symbolic links), as above. ELT/PLT CSVs are written in text mode, so on
  Windows they end lines with CRLF, as the pytools CLIs there do.

### 10.4 What remains

- **First macOS and Windows runs.** Everything that failed was in the tests or the test
  fixture, except the resource monitor on macOS: path separators in path assertions (the cli,
  the planner and genbash join with `os.sep`); the fixture's OED read through
  `oasis_data_manager`'s `file://<path>` URL, which has an empty path on Windows (the fixture now
  passes an `OedExposure` read from Path objects); the leftover-process check seeing
  multiprocessing's resource tracker, which tqdm's lock starts under spawn; and psutil denying
  the USS on macOS, which emptied `resource_monitor.csv` (now the RSS stands in).
- **pytools on macOS (not fixed here).** `write_mv_to_stream` selects on its output for
  exceptional conditions too, and macOS reports one on every regular file, so a pytool whose
  output is a regular file (`fmpy ... > file`, and likely summarypy's `-1 <file>` of a chunk
  script) raises `OSError: error with input stream`. Pipes and FIFOs are not affected, and the
  orchestrator's stages do not use it. The bash comparisons are skipped on macOS until it is
  fixed.
- **Memory on the default path.** The bash comparisons in `unittest.yml` each run the fixture's
  full kernel with bash at K = 2 (about 4.5 GB, 74 processes); fine on the 16 GB Linux runner,
  but they could move behind a dispatch input if that is too much.
- **Package 15, manifest.** Not needed for `rerun()` (the script is written). Left open only if
  a record of what the orchestrator ran (plan JSON, worker count) is wanted in the run directory;
  `orchestrator.log` holds the per-chunk and per-job summary.
- **Platform.** OasisPlatform must pass `orchestrator` in the params of
  `generate_losses_chunk` / `generate_losses_output` (or `run_analysis`) to use it; nothing in
  OasisLMF changes for that. Mixed chunks are interchangeable, so a partial rollout is safe.
- **W for local runs.** Done by package 16 (10.5): W is sized by the resource manager, K stays
  `max_process_id` unless `chunk_factor` is set.

### 10.5 Resource manager (package 16)

Why: this machine (31 GB) ran out of memory twice during the project; a bash full kernel at K12 needs
about 23 GB (9.7). The pool uses about 240 MB per worker at S10 and 440 MB at S100 on PiWind, but a
bigger model or portfolio can need far more per worker, and other things run on the machine. So the
pool is sized from what it measures, not from the CPU count alone.

- **Policy** (`orchestrator/resources.py`, `ResourceManager`, a `WorkerPool` policy). Starts
  `initial_workers` (2) workers. Samples, every `sample_interval` (1 s) and as soon as a worker
  becomes ready or finishes a chunk (not within 0.2 s of the previous sample), the available memory
  and each live worker's private memory (its process tree; see "measurement" below), keeping each
  worker's peak. The **estimate** is the largest worker peak seen, doubled (`unmeasured_factor`) until
  some worker has finished a chunk or job, because a worker grows while it runs its first chunk.
- **Grow** while there is room: n more workers, n the smallest of `(available - margin - reserve) //
  estimate` (reserve = what the live workers may still grow to the estimate, so workers still setting
  up are counted at full size), the hard maximum less the live workers, the CPUs (affinity) less the
  live workers, and the chunks not yet handed out less the workers still setting up (in the job
  phase: the jobs left; job-only workers are then added by the manager, not by `job_workers`).
  Nothing grows before a worker has built its chain, nor for `grow_cooldown` (10 s) after a retirement.
  The margin is `max(1 GB, 10 % of memory)` by default.
- **Back off** when `available < margin`: retire `ceil(deficit / estimate)` of the newest active
  workers, less those already retiring, keeping one. A retired worker gets `stop` at its next `ready`,
  so it always finishes its chunk; nothing is killed for memory. With one worker left and still
  under the margin, or when one worker plus the margin does not fit, the run carries on with one and
  logs a warning (once). A pool with no live worker and work left gets one, whatever the memory.
- **CPU.** Live workers never exceed the CPUs the process may use. `cpu_load=True` (off by default)
  also counts the CPUs other processes keep busy: memory is what crashes a machine, an
  oversubscribed CPU only slows the run, and a noisy load reading would shrink pools for nothing.
- **Measurement** (`SystemProbe`, injectable; tests pass a fake). Available memory is
  `psutil.virtual_memory().available`, tightened by the cgroup limit when there is one (v2: the
  tightest of the process's cgroup and its ancestors, `memory.max - (memory.current - inactive_file)`;
  v1: the memory controller's limit and usage), so a `systemd-run -p MemoryMax=` scope or a container
  limit is seen. A worker's size is `rss - shared` from `memory_info` on Linux (anonymous memory: the
  libraries and memory-mapped model files are page cache shared by every worker and reclaimable, so
  this is what one more worker costs), private bytes on Windows, RSS elsewhere, summed over its
  descendants found from one snapshot of the process table. The first version used USS
  (`memory_full_info`) and a `children(recursive=True)` per worker: about 20 ms per worker, which on
  PiWind slowed the output-job phase from 1.3 to 3.8 s because every finished job triggered a sample.
- **Where it is used.** `kernel.run_plan` (every orchestrated generate-losses step): on by default
  (`resource_manager=False` keeps W fixed), with `n_workers` (`max_process_id` = `ktools_num_processes`,
  the CPU count by default; 1 for a distributed chunk) as the hard maximum. On by default because an
  out-of-memory crash is the one failure the user cannot recover from, and its cost when memory is
  plentiful is small (below). The command line tool: `--resource-manager on|off`, default off, since it
  is the benchmarking tool where a fixed W is the point; `--workers` is then the maximum, with
  `--initial-workers` and `--memory-margin-mb`. `run_chunks(resource_manager=...)` takes an instance or
  True; it is exclusive with `policy`. The pool now calls its policy on every pass of its loop, not only
  after messages, so pressure is seen while every worker is busy in a long chunk.
- **Logs and statistics.** Every decision goes to the logger and to `orchestrator.log` with what it
  was based on, e.g. `resource manager: 2.8 s: spawn 2 -> 4 workers (room for 20 more after reserving
  0.31 GB for the live workers; available 7.58 GB, margin 1.00 GB, estimate 0.31 GB per worker)`,
  then the warnings and a summary (most live workers, margin, estimate, largest worker peak, samples).
  The executor statistics get `resource_manager` (`ResourceManager.summary()`).

**K and correctness.** K stays `max_process_id` by default, so the work and output files are bash's
byte for byte whatever W the manager chooses: chunk k holds `evepy k K`'s events and writes `_P<k>`
files whichever worker runs it, and every chunk resets its stages (the order chunks run in does not
matter, 9.3). `chunk_factor > 1` (a `run_plan` argument, not exposed as a run setting) splits a whole
run (mode `all` only) into `max(max_process_id, min(events, chunk_factor * n_workers))` chunks and gives
katpy the new file list (`outputs.jobs_for_chunks`; aalpy and lecpy read whatever `P*.bin` their
directory holds); outputs are then equal as sorted rows (AAL/EPT sums in another order), not bytes. A
chunked step (`analysis`) always keeps `max_process_id`: the outputs step reads `work/` with its own
count. Decided against making 4x the default: it would break byte equality with bash on every run, the
benchmark (9.7) found K48 no faster than K12 at full W, and the gain (finer retirement, less tail when
the manager holds W below K) only shows on memory-bound runs; those can set it. Largest chunks first was
not done: round-robin chunks have equal event counts, and per-event cost is not known before the run.

**Tests** (`tests/orchestrator/test_resources.py`, 22): fake pool and probe: start and wait for a
measurement, grow to the hard maximum, CPU and chunk limits, stop at the memory limit, reserve for
workers setting up, retire under pressure with the cool-down, never below one with a warning, one worker
does not fit, start with one when memory is already short, `cpu_load`, job-only workers in the job phase,
a worker when none is left, sampling rate; cgroup v2 parsing; the real probe. Real pool with fake stages:
grows and writes every chunk, retires under (faked) pressure and still writes every chunk. `test_kernel.py`:
the run log and options, `chunk_count`, `chunk_factor` (chunk numbers and katpy inputs), and on the
fixture's full kernel through `runner.run`: manager on (the default, also what the bash comparison now
runs) and off byte-identical, `chunk_factor=3` with one initial worker rows-equal. `test_cli.py`: the
options.

**PiWind** (`~/GitHub/runs/orch-bench-piwind-full/phase4/K4_S10_rm`, as `phase4/K4_S10`: `runner.run`,
4 processes, 1 RI iteration, orchestrator on, under an 8 GB cap): `output/` byte-identical to
`ref/K4_S10` (72/72). Decisions: start 2 workers (available 7.89 GB, the cgroup's, margin 1.00 GB);
at 2.8 s, once the first worker had built its chain (0.15 GB, estimate 0.31 GB doubled), spawn 2 -> 4
(room for 20 more). Largest worker 0.25 GB. Wall 40.7 s against 31.9 s for the fixed W4 run: about
2.8 s of the late start of workers 3 and 4, and 2.5 s of the output-job phase slowed by the USS
sampling fixed afterwards (that fix was not re-run on PiWind; the machine was also busier, load 5 to 6).

Open: expose `resource_manager`/`chunk_factor`/margin as run settings (now `run_plan` arguments only);
start more workers blind when the machine obviously has room (each later worker costs the first one's
setup time, about 3 to 5 s); cgroup v2 `cpu.max` quotas are not read; a per-event cost model (footprint
bytes per event) for largest-chunks-first.
