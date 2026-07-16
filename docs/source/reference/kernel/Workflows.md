# Workflows

The kernel runs as a **stream**: events → ground-up loss → insured loss → summary →
output tables. Because the tools are composable, many output workflows are possible.
This page shows the common ones as **pytools** pipelines.

In a real run, `oasislmf model run` generates `run_kernel.sh`, which runs these pipelines
across several **partitions** in parallel (`evepy p N`), connected by named pipes, and
concatenates the partition outputs with `katpy`. For a worked, stage-by-stage walkthrough
of a single stream see the *step-by-step pipeline* example (in the example-models docs).

Two ground-up engines are available: **`gulmc`** (full Monte-Carlo, reads the model data
directly — the default, used below) and **`gulpy`** (consumes `modelpy`'s CDF stream, i.e.
`evepy | modelpy | gulpy …`). Summary type is selected with `summarypy -t gul|il|ri`.

## Single-output workflows

### 1. Insured-loss event loss table (ELT)

Run ground-up → FM → summary (portfolio summary set 2) → ELT, per partition, then
concatenate:

```bash
evepy 1 2 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -2 - | eltpy -s elt_p1.csv
evepy 2 2 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -2 - | eltpy -s elt_p2.csv
katpy -s -i elt_p1.csv elt_p2.csv -o elt.csv
```

### 2. Insured-loss period loss table (PLT)

As above, through `pltpy` instead:

```bash
evepy 1 2 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -2 - | pltpy -s plt_p1.csv
evepy 2 2 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -2 - | pltpy -s plt_p2.csv
```

### 3. Loss exceedance curves (EPT)

`lecpy` (like `aalpy`) is not a stream stage — it reads all of a summary set's binaries
from `work/`, since EP curves are not valid on an event subset. Write the summary
binaries over multiple partitions, then run `lecpy` once:

```bash
evepy 1 2 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -2 - > work/summary2/p1.bin
evepy 2 2 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -2 - > work/summary2/p2.bin
lecpy -K summary2 -O ept.csv -F -f        # full-uncertainty AEP + OEP
```

### 4. Average annual loss (AAL)

Same pattern; `aalpy` reads the summary binaries from `work/`:

```bash
evepy 1 2 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -2 - > work/summary2/p1.bin
evepy 2 2 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -2 - > work/summary2/p2.bin
aalpy -K summary2 -a aal.csv
```

## Multiple-output workflows

### 5. Ground-up and insured loss together

`tee` the ground-up stream: one copy to a GUL summary, the other on into `fmpy` for the
insured summary — both perspectives from one run:

```bash
evepy 1 2 | gulmc -S100 -a1 | tee >(summarypy -t gul -2 - | eltpy -s gul_elt_p1.csv) \
          | fmpy -a2 | summarypy -t il -2 - | eltpy -s il_elt_p1.csv
```

### 6. Multiple summary levels

`summarypy` can emit several user-defined summary levels at once (up to 10); each can
feed a different output tool:

```bash
evepy 1 2 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -1 s1/p1.bin -2 s2/p1.bin
eltpy -i s1/p1.bin -s elt_s1_p1.csv
eltpy -i s2/p1.bin -s elt_s2_p1.csv
```

## Financial Module (reinsurance) workflows

`fmpy` is recursive: chain calls to apply successive sets of terms (direct insurance,
then reinsurance inuring priorities), each with its own input folder via `-p`, and `-n`
for net losses:

```bash
evepy 1 2 | gulmc -S100 -a1 | fmpy -p direct | fmpy -p ri1 -n > ri1_net_p1.bin
evepy 2 2 | gulmc -S100 -a1 | fmpy -p direct | fmpy -p ri1 -n > ri1_net_p2.bin
```

Each `fmpy` call reads the four `fm_*` input files from its `-p` folder, so a direct +
reinsurance run keeps a `direct/` and `ri1/` (etc.) set of inputs. All perspectives
(gross direct, net of each reinsurance layer) can be summarised and output in one
workflow. See {doc}`../../explanation/financial-module` for the FM concepts.

---

See also: {doc}`CoreComponents` · {doc}`OutputComponents` · {doc}`Specification`.
