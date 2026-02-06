# Data Transformation Flow: `extract_financial_structure()`

This document describes the data transformation pipeline in `extract_financial_structure()`, which converts raw FM input files into data structures for loss computation.

## Overview

The function transforms 6 input arrays into 6 output arrays, using intermediate CSR (Compressed Sparse Row) structures.

```
INPUTS                          OUTPUTS
─────────────────────────────────────────────────────────────────
fm_programme     ─┐             ┌─► compute_infos
fm_policytc      ─┼──► extract_ ─┼─► nodes_array
fm_profile       ─┤   financial ─┼─► node_parents_array
fm_xref          ─┤   structure ─┼─► node_profiles_array
items            ─┤              ├─► output_array
coverages        ─┘              └─► fm_profile (expanded)
```

---

## Phase 1: Profile Index Mapping (Lines 402-415)

**Input:** `fm_profile`
**Output:** `profile_id_to_profile_index`, `is_tiv_profile`

Maps each `profile_id` to its range `[i_start, i_end)` in `fm_profile` (supports multi-step profiles). Also identifies profiles requiring TIV (Total Insured Value) calculations.

```
fm_profile                          profile_id_to_profile_index
┌────────────┬───────────┐          ┌────────────┬─────────┬───────┐
│ profile_id │ calcrule  │          │ profile_id │ i_start │ i_end │
├────────────┼───────────┤    ──►   ├────────────┼─────────┼───────┤
│     1      │    12     │          │     1      │    0    │   1   │
│     2      │    27     │          │     2      │    1    │   3   │
│     2      │    27     │          │     3      │    3    │   4   │
│     3      │    100    │          └────────────┴─────────┴───────┘
└────────────┴───────────┘

is_tiv_profile[profile_id] = 1 if calcrule requires TIV
```

---

## Phase 2: Level Structure Analysis (Lines 417-429)

**Input:** `fm_programme`
**Output:** `max_level`, `level_node_len`, `multi_peril`

Scans programme to determine:
- Maximum aggregation level
- Number of nodes at each level
- Whether multi-peril structure exists (affects `start_level`)

```
fm_programme                        level_node_len
┌──────────┬─────────────┬──────────┐    ┌───────┬─────┐
│ level_id │ from_agg_id │ to_agg_id│    │ level │ len │
├──────────┼─────────────┼──────────┤    ├───────┼─────┤
│    1     │      1      │     1    │    │   0   │  3  │  (items)
│    1     │      2      │     1    │    │   1   │  2  │  (locations)
│    2     │      1      │     1    │    │   2   │  1  │  (account)
│    2     │      2      │     1    │    └───────┴─────┘
└──────────┴─────────────┴──────────┘
```

---

## Phase 3: TIV Duplicate Pre-counting (Lines 432-454)

**Input:** `fm_policytc`, `is_tiv_profile`
**Output:** `num_tiv_duplicates`, expanded `fm_profile`

TIV profiles must be duplicated per-node because TIV values differ. This pass counts duplicates needed and pre-allocates the expanded profile array.

```
First occurrence:  Uses original profile indices
Subsequent:        Needs duplicate entries in fm_profile

fm_policytc scan:
  profile_id=2 at node A  →  first seen, use original [1,3)
  profile_id=2 at node B  →  duplicate needed, will use [N, N+2)
```

---

## Phase 4: Node Index Computation (Lines 456-475)

**Input:** `level_node_len`, `multi_peril`, `allocation_rule`
**Output:** `start_level`, `out_level`, `node_level_start`, `total_nodes`

Computes the flat indexing scheme for nodes:

```
node_level_start[level] = cumulative sum of nodes at levels < level
node_idx = node_level_start[level] + agg_id

Example (start_level=1):
  level_node_len = [3, 2, 1]
  node_level_start = [0, 0, 2, 3]  (level 0 skipped if single-peril)

  Node (level=1, agg_id=1) → idx = 0 + 1 = 1
  Node (level=1, agg_id=2) → idx = 0 + 2 = 2
  Node (level=2, agg_id=1) → idx = 2 + 1 = 3
```

---

## Phase 5: Profiles CSR Construction (Lines 477-545)

**Input:** `fm_policytc`, `profile_id_to_profile_index`
**Output:** `profiles_indptr`, `profiles_data` (CSR format)

Builds profiles CSR from `fm_policytc` using two-pass approach:

### Pass 1: Count profiles per node
```python
for policytc in fm_policytc:
    node_idx = node_level_start[level] + agg_id
    profiles_count[node_idx] += 1
```

### Build indptr
```python
profiles_indptr[i+1] = profiles_indptr[i] + profiles_count[i]
```

### Pass 2: Fill CSR data with TIV handling
```python
for policytc in fm_policytc:
    if is_tiv_profile[profile_id] and seen_before:
        # Duplicate profile entries in fm_profile
        i_start, i_end = i_new_fm_profile, ...
    profiles_data[pos] = (layer_id, i_start, i_end)
```

### Sort by layer_id
In-place bubble sort within each node's slice.

```
CSR Structure:
profiles_indptr: [0, 2, 5, 7, ...]
profiles_data:   [(layer1, i_s, i_e), (layer2, i_s, i_e), ...]
                  └─── node 1 ───┘    └───── node 2 ─────┘

Lookup: profiles_data[profiles_indptr[idx]:profiles_indptr[idx+1]]
```

---

## Phase 6: Output ID Mapping (Lines 547-560)

**Input:** `fm_xref`
**Output:** `output_id_arr` (2D array)

Creates direct lookup for output IDs at the output level:

```
output_id_arr[agg_id, layer_id] = output_id

fm_xref                              output_id_arr
┌────────┬──────────┬────────┐       ┌─────────┬─────────┬─────────┐
│ agg_id │ layer_id │ output │       │         │ layer 1 │ layer 2 │
├────────┼──────────┼────────┤  ──►  ├─────────┼─────────┼─────────┤
│   1    │    1     │   1    │       │ agg_id 1│    1    │    2    │
│   1    │    2     │   2    │       │ agg_id 2│    3    │    0    │
│   2    │    1     │   3    │       └─────────┴─────────┴─────────┘
└────────┴──────────┴────────┘
```

---

## Phase 7: Node Layer Initialization (Lines 562-581)

**Input:** `fm_programme`, `profiles_indptr`
**Output:** `node_layers_arr`, `node_cross_layers_arr`, `layer_source`

Initializes tracking arrays:
- `node_layers_arr[idx]`: Number of layers for each node
- `node_cross_layers_arr[idx]`: Cross-layer flag (0/1)
- `layer_source[idx]`: Source node index for layer inheritance

```python
for programme in fm_programme:
    parent_idx = node_level_start[level] + to_agg_id
    if node_layers_arr[parent_idx] == 0:
        node_layers_arr[parent_idx] = profiles_indptr[parent_idx+1] - profiles_indptr[parent_idx]
```

---

## Phase 8: Parent/Child CSR Construction (Lines 583-648)

**Input:** `fm_programme`
**Output:** `children_indptr`, `children_data`, `parents_indptr`, `parents_data`

Builds bidirectional parent-child relationships using two-pass approach:

### Pass 1: Count relationships
```python
for programme in fm_programme:
    if level > start_level:
        children_count[parent_idx] += 1
        parents_count[child_idx] += 1
```

### Pass 2: Fill CSR (level-by-level for ordering)
```python
for level in range(max_level, start_level, -1):
    for programme at this level:
        children_data[c_pos] = child_idx
        parents_data[p_pos] = parent_idx  # fill from end for correct order
```

```
Parent→Children CSR:           Child→Parents CSR:
children_indptr: [0, 2, 4, 4]  parents_indptr: [0, 1, 2, 2]
children_data:   [1, 2, 3, 4]  parents_data:   [3, 3]

Node 0 has children [1, 2]     Node 1 has parents [3]
Node 1 has children [3, 4]     Node 2 has parents [3]
```

---

## Phase 9: Layer Propagation & Cross-Layer Detection (Lines 650-677)

**Input:** CSR structures, `node_layers_arr`
**Output:** Updated `node_layers_arr`, `node_cross_layers_arr`, `layer_source`

Propagates layer counts down the tree and detects cross-layer nodes:

```python
for level in range(max_level, start_level, -1):
    for each parent-child relationship:
        if child has fewer/equal layers than parent:
            child inherits parent's layers
            layer_source[child] = layer_source[parent]
        elif child has more layers:  # cross-layer
            mark grandparents as cross-layer nodes
```

---

## Phase 10: Output Array Construction (Lines 679-831)

**Input:** All intermediate structures
**Output:** `nodes_array`, `node_parents_array`, `node_profiles_array`, `output_array`

Final assembly iterating level-by-level, agg_id-by-agg_id:

### nodes_array fields:
| Field | Description |
|-------|-------------|
| `node_id` | Sequential node identifier |
| `level_id`, `agg_id` | Node coordinates |
| `layer_len` | Number of layers |
| `profile_len`, `profiles` | Profile count and index |
| `parent_len`, `parent` | Parent count and index into `node_parents_array` |
| `children` | Index for children lookup |
| `loss`, `net_loss`, `extra` | Indices for loss computation arrays |
| `output_ids` | Index into `output_array` |

### TIV Computation (during profile processing):
```python
if profile requires TIV:
    children = get_all_children_csr(node_idx, ...)
    tiv = get_tiv_csr(children, items, coverages, ...)
    prepare_profile_simple/stepped(profile, tiv)
```

### Extra allocation (for min/max deductible policies):
```python
if profile has need_extras calcrule:
    for each parent at this level:
        for each child of parent:
            allocate extra space if not already allocated
```

---

## Phase 11: Compute Info Assembly (Lines 833-848)

**Output:** `compute_infos` array with metadata for computation engine

| Field | Description |
|-------|-------------|
| `allocation_rule` | Back-allocation method (0, 1, or 2) |
| `max_level` | Highest aggregation level |
| `max_layer` | Maximum layers across all nodes |
| `node_len` | Total nodes + 1 |
| `children_len` | Size of children index array |
| `parents_len` | Size of parents array |
| `profile_len` | Size of profiles array |
| `loss_len` | Size of loss computation array |
| `extra_len` | Size of extras array |
| `compute_len` | Size of compute queue |
| `start_level` | Starting level (0 or 1) |
| `items_len` | Number of items |
| `output_len` | Number of outputs |
| `stepped` | Whether stepped profiles are used |

---

## Data Structures

### CSR (Compressed Sparse Row) Format

Used for sparse relationships:

```
indptr:  [0, 2, 5, 5, 8]     # cumulative counts
data:    [a, b, c, d, e, f, g, h]

Row 0: data[0:2]  = [a, b]
Row 1: data[2:5]  = [c, d, e]
Row 2: data[5:5]  = []        # empty
Row 3: data[5:8]  = [f, g, h]
```

### Intermediate Structures

| Structure | Type | Description |
|-----------|------|-------------|
| `profiles_indptr` + `profiles_data` | CSR | Node → profile entries (layer_id, i_start, i_end) |
| `children_indptr` + `children_data` | CSR | Parent → children relationships |
| `parents_indptr` + `parents_data` | CSR | Child → parents relationships |
| `node_layers_arr` | 1D array | Layer count per node |
| `node_cross_layers_arr` | 1D array | Cross-layer flag per node |
| `layer_source` | 1D array | Layer inheritance source per node |
| `output_id_arr` | 2D array | (agg_id, layer_id) → output_id mapping |
| `is_tiv_profile` | 1D array | TIV requirement flag per profile_id |

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT FILES                                       │
│  fm_programme  fm_policytc  fm_profile  fm_xref  items  coverages       │
└───────┬───────────────┬───────────┬────────┬────────┬────────┬──────────┘
        │               │           │        │        │        │
        ▼               │           ▼        │        │        │
┌───────────────┐       │   ┌──────────────┐ │        │        │
│ Phase 1:      │       │   │ profile_id   │ │        │        │
│ Profile Index │◄──────┼───│ to_profile   │ │        │        │
│ Mapping       │       │   │ _index       │ │        │        │
└───────┬───────┘       │   │ is_tiv_      │ │        │        │
        │               │   │ profile      │ │        │        │
        │               │   └──────────────┘ │        │        │
        ▼               │                    │        │        │
┌───────────────┐       │                    │        │        │
│ Phase 2:      │◄──────┘                    │        │        │
│ Level Struct  │                            │        │        │
│ Analysis      │──► level_node_len          │        │        │
└───────┬───────┘    max_level               │        │        │
        │            multi_peril             │        │        │
        ▼                                    │        │        │
┌───────────────┐                            │        │        │
│ Phase 3:      │◄───────────────────────────┘        │        │
│ TIV Duplicate │──► num_tiv_duplicates               │        │
│ Pre-counting  │    expanded fm_profile              │        │
└───────┬───────┘                                     │        │
        │                                             │        │
        ▼                                             │        │
┌───────────────┐                                     │        │
│ Phase 4:      │──► start_level, out_level           │        │
│ Node Index    │    node_level_start                 │        │
│ Computation   │    total_nodes                      │        │
└───────┬───────┘                                     │        │
        │                                             │        │
        ▼                                             │        │
┌───────────────┐                                     │        │
│ Phase 5:      │──► profiles_indptr                  │        │
│ Profiles CSR  │    profiles_data (CSR)              │        │
│ Construction  │                                     │        │
└───────┬───────┘                                     │        │
        │                                             │        │
        ▼                                             │        │
┌───────────────┐◄────────────────────────────────────┘        │
│ Phase 6:      │──► output_id_arr (2D)                        │
│ Output ID     │                                              │
│ Mapping       │                                              │
└───────┬───────┘                                              │
        │                                                      │
        ▼                                                      │
┌───────────────┐                                              │
│ Phase 7:      │──► node_layers_arr                           │
│ Node Layer    │    node_cross_layers_arr                     │
│ Init          │    layer_source                              │
└───────┬───────┘                                              │
        │                                                      │
        ▼                                                      │
┌───────────────┐                                              │
│ Phase 8:      │──► children_indptr, children_data (CSR)      │
│ Parent/Child  │    parents_indptr, parents_data (CSR)        │
│ CSR           │                                              │
└───────┬───────┘                                              │
        │                                                      │
        ▼                                                      │
┌───────────────┐                                              │
│ Phase 9:      │──► updated node_layers_arr                   │
│ Layer Prop &  │    updated node_cross_layers_arr             │
│ Cross-Layer   │    updated layer_source                      │
└───────┬───────┘                                              │
        │                                                      │
        ▼                                                      │
┌───────────────┐◄─────────────────────────────────────────────┘
│ Phase 10:     │──► nodes_array
│ Output Array  │    node_parents_array
│ Construction  │    node_profiles_array
│               │    output_array
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Phase 11:     │──► compute_infos
│ Compute Info  │
│ Assembly      │
└───────────────┘
```
