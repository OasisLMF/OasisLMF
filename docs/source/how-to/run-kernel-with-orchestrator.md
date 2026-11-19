# Run the loss kernel with the orchestrator

```{versionadded} NEXT
```

The orchestrator runs the loss kernel in a pool of Python workers instead of the generated bash
script. It is faster and uses less memory, and it writes the same files. See
{doc}`../explanation/kernel-execution` for how it works.

## Turn it on

It is off by default. Turn it on with the `orchestrator` option of the steps that run the
kernel: `model run`, `model generate-losses`, `model generate-losses-chunk` and
`model generate-losses-output`.

On the command line:

```sh
oasislmf model run --config oasislmf.json --orchestrator
```

Or in the JSON configuration file:

```json
{
    "orchestrator": true
}
```

`kernel_num_processes` keeps its meaning: it is the number of partitions the events are split
into, and the most workers the orchestrator will start. The resource manager starts two workers
and adds more while memory allows.

## Check that it ran

The bash script `run_kernel.sh` is still written, as a record of the run (`rerun` uses it), but
it is not run. When the orchestrator made the run, `log/` holds:

- `orchestrator.log`: the run log, with each chunk and output job, its time and worker, and the
  resource manager's decisions;
- `orchestrator_worker_<n>.out` and `.err`: each worker's output streams;
- `orchestrator_worker_<n>_<pid>_<uuid>.log`: each worker's log records, at the run's log level;
- `stderror.err`: as with the script's stderr guard, the workers' stderr is appended there, and a line that is not a
  Python warning fails the run.

When it fell back to the script, the step logs a warning instead:

```
The orchestrator cannot run this analysis (<reason>): running the kernel with the bash script instead
```

and `log/` holds the script's logs only.

## What it supports

| Setting | With the orchestrator |
|---|---|
| gulmc (the default ground-up engine) | Supported. |
| Insured and reinsurance losses, every inuring priority, `ri_output` and `rl_output` | Supported. |
| Every ORD output, CSV or parquet, summarypy low memory, fmpy low memory and sorted output | Supported. |
| Post loss amplification (`pla`, `pla_secondary_factor`, `pla_uniform_factor`) | Supported. |
| `join_summary_info` | Supported. |
| Event shuffling rules (`kernel_event_shuffle`) | Supported. |
| Distributed runs (`generate-losses-chunk`, `generate-losses-output`) | Supported; chunk files are interchangeable with the script's. |
| The load balancer (`kernel_num_gul_per_lb`, `kernel_num_fm_per_lb`) | Ignored: workers already take the next chunk when they are free. The output is that of a run without it. |
| `model_py_server` | Ignored: each worker reads the footprint itself. |
| Legacy non-ORD outputs (`eltcalc`, `summarycalc`, `aalcalc`, `leccalc`...) | Not produced, as with the script, which emits nothing for them. |
| gulpy (`gulmc: false`) | Falls back to the script. |
| A custom gulcalc (`model_custom_gulcalc`, or a `<supplier>_<model>_gulcalc` on the path) | Falls back to the script: an external program computes the ground-up loss. |
| Custom getmodel arguments (`custom_args`) | Falls back to the script. |
| `full_correlation` | Falls back to the script. |

Each ignored setting is noted in the step's log.

## Use it on the platform

The platform runs `generate-losses-chunk` and `generate-losses-output` for each analysis.
OasisPlatform has to pass the `orchestrator` option to those steps for its workers to use it;
OasisLMF needs no other change. A chunk made by the orchestrator can be gathered with chunks made
by the script.
