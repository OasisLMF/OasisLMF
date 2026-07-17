# ORD output components

Oasis produces results in the **Open Results Data (ORD)** format. The pytools output
tools ({doc}`OutputComponents`) emit the ORD tables directly; this page maps each ORD
table to the tool that produces it and to the `analysis_settings` flag that requests it.

```{note}
The **column definitions** of each ORD table are part of the ORD *standard*, single-sourced
in the `ODS_OpenResultsData` repository (and pulled into the aggregated Oasis documentation).
This page does not repeat them.
```

## ORD tables, tools and settings

| ORD table | pytools tool | `analysis_settings` → `ord_output` flag |
|-----------|--------------|------------------------------------------|
| **SELT** — Sample Event Loss Table | `eltpy` | `elt_sample` |
| **MELT** — Moment Event Loss Table | `eltpy` | `elt_moment` |
| **QELT** — Quantile Event Loss Table | `eltpy` | `elt_quantile` |
| **SPLT** — Sample Period Loss Table | `pltpy` | `plt_sample` |
| **MPLT** — Moment Period Loss Table | `pltpy` | `plt_moment` |
| **QPLT** — Quantile Period Loss Table | `pltpy` | `plt_quantile` |
| **EPT** — Exceedance Probability Table | `lecpy` | `ept_full_uncertainty_aep` / `_oep`, `ept_mean_sample_aep` / `_oep`, `ept_per_sample_mean_aep` / `_oep` |
| **PSEPT** — Per-Sample EPT | `lecpy` | `psept_aep`, `psept_oep` |
| **ALT** — Average Loss Table (AAL) | `aalpy` | `alt_period` (`alt_meanonly` for mean only) |
| **ALCT** — Average Loss Convergence Table | `aalpy` | `alct_convergence` (+ `alct_confidence`) |

`EPType` in the EPT/PSEPT tables is `1`=OEP, `2`=OEP TVaR, `3`=AEP, `4`=AEP TVaR; `EPCalc`
is the calculation basis (`1` MeanDamage, `2` FullUncertainty, `3` PerSampleMean,
`4` MeanSample). See the ORD standard for full definitions.

## Requesting ORD outputs

ORD tables are requested per summary set in `analysis_settings.json` under `ord_output`.
For example, requesting a sample ELT, an EP table (full-uncertainty AEP + OEP) and the
period AAL for the ground-up perspective:

```json
{
  "gul_output": true,
  "gul_summaries": [
    {
      "id": 1,
      "ord_output": {
        "elt_sample": true,
        "ept_full_uncertainty_aep": true,
        "ept_full_uncertainty_oep": true,
        "alt_period": true,
        "parquet_format": false
      }
    }
  ]
}
```

The same `ord_output` block applies under `il_summaries` / `ri_summaries` for the insured
and reinsurance perspectives. Global options include `parquet_format` (write parquet
instead of CSV) and `return_period_file` (use `input/returnperiods.bin` for the EP curve
return periods).

## How they are produced in a run

`oasislmf model run` generates a `run_kernel.sh` pipeline that streams `summarypy` output
into the relevant output tool per requested table (see {doc}`Workflows`). The per-partition
outputs are concatenated with `katpy`. For a worked, stage-by-stage example see the
step-by-step pipeline notebook (in the example-models docs), and for analysing the
resulting tables see the ORD results notebook.

---

See also: {doc}`OutputComponents` (the tools and their CLI) · {doc}`CoreComponents`.
