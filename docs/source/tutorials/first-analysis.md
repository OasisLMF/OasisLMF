# Run your first analysis

This tutorial takes you end-to-end through your first OasisLMF loss analysis with the
**PiWind** reference model — from installing the toolkit to inspecting the results. From a
user's point of view the whole analysis is a single command; the MDK prepares the model
inputs and runs the loss calculation for you.

```{admonition} Prefer a version that runs itself?
:class: tip

The [**Run a PiWind analysis end-to-end**](https://oasislmf.github.io/models/tutorials/run-piwind-analysis.html)
tutorial in Oasis Models is the *executable* companion to this page: it ships with the PiWind
model data and actually runs the analysis (and the plots) at build time, so every command shown
there is verified. Follow this page to understand the shape of a run; open that one to see it
execute against real result files.
```

## 1. Install OasisLMF

Install the toolkit into a virtual environment (see {doc}`../how-to/installation` for platform
notes and optional extras):

```bash
python -m venv venv && source venv/bin/activate
pip install oasislmf
oasislmf --help
```

## 2. Get an example model

PiWind is Oasis's small reference windstorm model — big enough to be realistic, small enough
to run on a laptop. Clone it and step into a ready-made test configuration:

```bash
git clone https://github.com/OasisLMF/OasisPiWind.git
cd OasisPiWind
```

The repository ships the model data (footprint, vulnerability, damage bins, occurrence),
the keys/lookup configuration, an example OED exposure set, and an `oasislmf.json` that ties
them together.

## 3. Run the analysis

A full ground-up **and** insured-loss run is a single command — point it at the config that
references the PiWind model data, keys/lookup and OED exposure:

```bash
oasislmf model run -C oasislmf.json
```

Under the hood the MDK:

1. **Generates inputs** — runs the keys lookup to find which locations the model covers, then
   builds the ground-up-loss (GUL) and financial-module (FM) input files from your OED exposure.
2. **Generates losses** — runs the loss kernel: samples ground-up losses, then applies the
   policy terms to produce insured losses, and writes the requested outputs.

## 4. Inspect the results

Results land in a run directory under `output/`, as Open Results Data (ORD) tables — for
example `gul_S1_ept.csv` (ground-up exceedance-probability curve) and `il_S1_ept.csv` (insured).
A quick look with pandas:

```python
import pandas as pd
ept = pd.read_csv("output/gul_S1_ept.csv")
# EPCalc 2 = full uncertainty; EPType 1 = OEP, 3 = AEP
oep = ept[(ept.EPCalc == 2) & (ept.EPType == 1)].sort_values("ReturnPeriod")
print(oep[["ReturnPeriod", "Loss"]])
```

## Where to go next

The payoff of the Diátaxis split is that you can now step sideways exactly when you need depth:

- **What the options mean** — the flags used by `model run` are catalogued in
  {doc}`../reference/index`.
- **Why the insured-loss step does what it does** — {doc}`../explanation/financial-module`
  explains the financial module.
- **See it run for real** — the executable
  [PiWind end-to-end tutorial](https://oasislmf.github.io/models/tutorials/run-piwind-analysis.html)
  runs this same analysis at build time and plots the exceedance-probability curve.
- **Explore the model's own data** — {doc}`explore-model-data`.
- **Analyse the ORD outputs in depth** — {doc}`analyse-ord-results`.
