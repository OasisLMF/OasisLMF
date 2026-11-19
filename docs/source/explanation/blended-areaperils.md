# Blended areaperils

```{versionadded} NEXT
```

Some exposures are known only to an area, such as a postcode, which covers several hazard cells.
A **blended areaperil** lets the model say *where in that area the building may stand*: a set
of candidate positions, each with a weight and each read through a real hazard cell. In every
sample, gulmc stands each building at one position. All its perils and coverages then read the
hazard at that position.

## The idea in one example

A postcode holds 10 buildings. The wind footprint has two cells (11 and 12). The surge
footprint is on its own, coarser grid (cells 101 and 102), and part of the postcode sees no
surge at all.

```{figure} /_static/images/blended_areaperils/postcode_light.svg
:figclass: only-light
:alt: A postcode outline over two wind cells and two surge cells, its buildings grouped into three positions
:width: 100%

Each distinct (wind cell, surge cell) pair that holds buildings is a **position**, weighted by
its building count: P0 (5 buildings), P1 (3) and P2 (2).
```

````{only} html
```{figure} /_static/images/blended_areaperils/postcode_dark.svg
:figclass: only-dark
:alt: A postcode outline over two wind cells and two surge cells, its buildings grouped into three positions
:width: 100%

Each distinct (wind cell, surge cell) pair that holds buildings is a **position**, weighted by
its building count: P0 (5 buildings), P1 (3) and P2 (2).
```
````

| Position | Weight | Wind reads | Surge reads |
|---|---|---|---|
| P0 | 5 | cell 11 | cell 101 |
| P1 | 3 | cell 11 | nothing: not at risk of surge |
| P2 | 2 | cell 12 | cell 102 |

The lookup returns one blended areaperil per peril for this postcode: **9001** for wind and
**9002** for surge. Both describe the same three positions, so a building is always at the same
place for both perils.

## Step 1: pick a position

Each building gets one uniform random number per sample. Read against the running weights, it
selects a position, so a position is chosen with probability proportional to its weight.

```{figure} /_static/images/blended_areaperils/selection_light.svg
:figclass: only-light
:alt: A bar split 0.5 / 0.3 / 0.2 between positions P0, P1 and P2, with five samples' random numbers pointing into it
:width: 100%

Building 1, samples 1 to 5. Samples 1 and 3 land in P0, samples 2 and 5 in P1, and sample 4 in P2.
```

````{only} html
```{figure} /_static/images/blended_areaperils/selection_dark.svg
:figclass: only-dark
:alt: A bar split 0.5 / 0.3 / 0.2 between positions P0, P1 and P2, with five samples' random numbers pointing into it
:width: 100%

Building 1, samples 1 to 5. Samples 1 and 3 land in P0, samples 2 and 5 in P1, and sample 4 in P2.
```
````

The choice is made **once per building and sample**, before any event is processed:

```{figure} /_static/images/blended_areaperils/placements_light.svg
:figclass: only-light
:alt: A grid of two buildings by ten samples, each cell coloured by the position chosen
:width: 100%

Two buildings of the same location over ten samples. Each building is placed on its own: they
can stand at different positions in the same sample. Over many samples each position is chosen
in proportion to its weight.
```

````{only} html
```{figure} /_static/images/blended_areaperils/placements_dark.svg
:figclass: only-dark
:alt: A grid of two buildings by ten samples, each cell coloured by the position chosen
:width: 100%

Two buildings of the same location over ten samples. Each building is placed on its own: they
can stand at different positions in the same sample. Over many samples each position is chosen
in proportion to its weight.
```
````

- **The same in every event.** A building does not move between events: in sample 4 it stands
  at P2 for the whole event set.
- **Reproducible.** The draw depends on the location's identifiers (`PortNumber`, `AccNumber`,
  `LocNumber`), its building number and the sample. With the same number of samples,
  rerunning a portfolio, reordering it, or adding other locations gives each building the same
  positions.

## Step 2: follow the building through

Once placed, the building is modelled exactly as if the lookup had put it at that position:

| Building 1 | Position | Wind | Surge |
|---|---|---|---|
| sample 1 | P0 | hazard of cell 11 | hazard of cell 101 |
| sample 2 | P1 | hazard of cell 11 | no loss |
| sample 4 | P2 | hazard of cell 12 | hazard of cell 102 |

Each peril samples its own hazard intensity and damage at that position, as for any location.
What blending adds is that **wind and surge, building and contents, always agree on where the
building is**. A dependent coverage (see {doc}`coverage-dependency`) follows its source to the
same position.

**Mean and standard deviation.** The analytic outputs are the weighted mixture of the positions.
For surge in the example: 0.5 × (the mean at cell 101) + 0.3 × 0 + 0.2 × (the mean at cell 102).
gulpy receives this same mixture from getmodel. It therefore gets each item's distribution
right, but it does not keep the perils of a building at one position the way gulmc does.

**Dynamic footprints** (see {doc}`dynamic-footprint`). Standard of protection is judged at the position: in the same event a
building can be protected at P0 and flooded at P2.

## What the model needs to provide

### 1. The blended areaperil file

`blended_areaperil_id.csv` (or `.bin`) sits with the rest of the model data, beside the
footprint. Each row is one position of one blended areaperil:

```text
blended_areaperil_id,placement_scheme_id,placement_id,weight,areaperil_id
9001,1,0,5,11
9001,1,1,3,11
9001,1,2,2,12
9002,1,0,5,101
9002,1,1,3,0
9002,1,2,2,102
```

| Column | Meaning |
|---|---|
| `blended_areaperil_id` | The id the lookup returns. It must not also be a real cell id. |
| `placement_scheme_id` | The set of positions. Every blended areaperil of a location (all its perils and coverages) must name the same scheme. |
| `placement_id` | The position, numbered 0, 1, 2… within the blended areaperil. |
| `weight` | How likely the position is (see below). |
| `areaperil_id` | The footprint cell read at that position for this peril; `0` means not at risk. |

The `weight` column takes one of three forms, read from the file:

| Weights | Example | Meaning |
|---|---|---|
| no column | — | All positions equally likely. |
| integers | `5, 3, 2` | Relative weights, such as building or address counts. |
| decimals | `0.5, 0.3, 0.1` | Probabilities. They may add up to less than 1: the rest (here 0.1) is the chance the building is outside every modelled position, where it loses nothing. Adding up to more than 1 is an error. |

To convert between CSV and binary:

```bash
csvtobin blended_areaperil -i blended_areaperil_id.csv -o blended_areaperil_id.bin
bintocsv blended_areaperil -i blended_areaperil_id.bin -o blended_areaperil_id.csv
```

Mistakes the file can show on its own fail the run with a message naming the ids, for example a
position missing from 0, 1, 2…, two perils giving one position different weights, or a blended
id also used as a cell id.

### 2. A lookup that returns blended ids

The lookup returns the blended areaperil as the location's `AreaPerilID`, one per peril or
coverage type, all naming the same placement scheme. Nothing else in the exposure or the run
settings changes.

With a **dynamic footprint**, the lookup must list every section that any of the location's
positions falls in. A missing section gives no error: positions in it never see hazard.

### 3. The zero-loss damage bin

The first bin of the damage bin dictionary (`bin_index` 1) must be the zero-loss bin, which is
the usual Oasis convention. A position with no hazard puts its weight there in the mean and
standard deviation, and in the distribution getmodel sends to gulpy.

## Limits

- **gulpy** gets each item's mixed distribution but not one position shared by a building's
  perils. Use gulmc for that.
- A location's weights cannot differ by building type or coverage type: all its blended
  areaperils share one scheme.

The OasisModels repository has two worked models: **PiWindBlended** (postcode wind and surge on
grids that do not line up) and **DynamicFootprint** (blended locations spanning two flood basins).
