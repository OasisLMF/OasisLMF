---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Explore Oasis model data

A hands-on look at the data that drives a ground-up loss calculation — the
**footprint**, **vulnerability**, and **damage bin dictionary** (model static data)
plus the **items** and **coverages** (the exposure). This notebook loads a small
example model shipped with the docs and inspects each file with pandas.

```{note}
This page is an **executable notebook** — every cell below is run when the docs are
built, so the outputs are always produced against the current code and data. See
{doc}`../explanation/index` for the concepts and {doc}`../reference/index` for the
file-format reference.
```

```{code-cell} python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Resolve the example-data directory regardless of the notebook's execution cwd.
_candidates = [
    Path("data/example_model"),
    Path("tutorials/data/example_model"),
    Path("docs/source/tutorials/data/example_model"),
]
DATA = next((c for c in _candidates if c.exists()), None)
assert DATA is not None, "example_model data directory not found"
DATA
```

## Damage bins

Damage is discretised into **bins**. The damage bin dictionary defines the interval
each bin covers (`bin_from`–`bin_to`) — losses are sampled as bin indices and later
mapped back to a mean damage ratio.

```{code-cell} python
damage_bins = pd.read_csv(DATA / "damage_bin_dict.csv")
damage_bins.head()
```

## Footprint

The **footprint** gives, for each event, a probability distribution over hazard
**intensity bins** at each area-peril (grid cell). It is the hazard side of the
calculation.

```{code-cell} python
footprint = pd.read_csv(DATA / "footprint.csv")
print(f"{footprint['event_id'].nunique()} events, "
      f"{footprint['areaperil_id'].nunique()} area-perils")
footprint.head()
```

## Vulnerability

The **vulnerability** functions give, for each vulnerability id, the probability of
each **damage bin** conditional on the hazard **intensity bin** — i.e. how damageable
a coverage is at a given intensity.

```{code-cell} python
vulnerability = pd.read_csv(DATA / "vulnerability.csv")
vulnerability.head()
```

Below is the conditional damage distribution for one vulnerability function at one
hazard intensity — the building block the Monte-Carlo engine samples from:

```{code-cell} python
vid = int(vulnerability["vulnerability_id"].iloc[0])
iid = int(vulnerability.loc[vulnerability.vulnerability_id == vid, "intensity_bin_id"].iloc[0])
sub = vulnerability[(vulnerability.vulnerability_id == vid)
                    & (vulnerability.intensity_bin_id == iid)]

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(sub["damage_bin_id"], sub["probability"])
ax.set_xlabel("damage bin")
ax.set_ylabel("probability")
ax.set_title(f"P(damage bin | intensity) — vulnerability {vid}, intensity bin {iid}")
fig.tight_layout()
```

## Exposure: items and coverages

**Coverages** hold the insured values (TIV); **items** link each coverage to an
area-peril and a vulnerability function (the join between exposure and model data).

```{code-cell} python
items = pd.read_csv(DATA / "items.csv")
coverages = pd.read_csv(DATA / "coverages.csv")
exposure = items.merge(coverages, on="coverage_id")
print(f"{len(items)} items across {len(coverages)} coverages; "
      f"total TIV = {coverages['tiv'].sum():,.0f}")
exposure.head()
```

## Where next

- {doc}`../explanation/modelling-methodology` and
  {doc}`../explanation/sampling-methodology` — how these inputs become sampled losses.
- {doc}`../how-to/ground-up-losses` — run the gulmc engine on data like this.
- {doc}`../reference/Oasis-model-data-formats` — the full file-format reference.
