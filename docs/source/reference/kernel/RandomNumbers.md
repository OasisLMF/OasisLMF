# Appendix A: Random numbers

Simple uniform random numbers are assigned to each event, group and sample number to sample
ground up loss in the gulmc process. A group is a collection of items which share the same
group_id, and is the method of supporting spatial correlation in ground up loss sampling in
Oasis.

#### Correlation

Items (typically representing, in insurance terms, the underlying risk coverages) that are
assigned the same group_id will use the same random number to sample damage for a given event
and sample number. Items with different group_ids will be assigned independent random numbers.
Therefore sampled damage is fully correlated within groups and fully independent between
groups, where group is an abstract collection of items defined by the user.

The item_id, group_id data is provided by the user in the items input file (items.bin).

`gulmc` samples hazard intensity as well as damage, and the two are correlated independently:
damage draws key off `group_id` (items file) and hazard draws off `hazard_group_id`
(correlations file), each with its own seeding constants.

### Methodology

There is no buffer of pre-generated random numbers and no index into one. For each
`(group_id, event_id)` pair `gulmc` derives a seed and generates the sample draws from it, so a
run is repeatable without a seed parameter, and there is no buffer size to choose.

The damage seed is (`oasislmf/pytools/gul/random.py`):

```
s1   = mod(group_id * 1543270363, 2147483648)
s2   = mod(event_id * 1943272559, 2147483648)
seed = mod(s1 + s2, 2147483648)
```

Hazard sampling uses the same shape with its own constants and modulus, keyed off
`hazard_group_id` rather than `group_id`:

```
s1   = mod(hazard_group_id * 1143271949, 1957483729)
s2   = mod(event_id       * 1243274353, 1957483729)
seed = mod(s1 + s2, 1957483729)
```

Because the seed is a pure function of the group and the event, items in the same group see
identical draws for a given event and sample index, which is what produces the correlation
described above.

#### Choosing the generator

`--random-generator` selects how numbers are drawn from that seed:

| Value | Generator |
|-------|-----------|
| `0` | numpy default (MT19937) |
| `1` | Latin Hypercube |
| `2` | Latin Hypercube on Philox4x32-7 (counter-based, faster) — **default** |

```bash
evepy 1 1 | gulmc -S 100 --random-generator 0 -o gulmc.bin
```

There is no seed option and no random-number file: seeding is derived as above, so repeatability
is inherent rather than something to switch on.

#### Inspecting the numbers used

`-d` writes the random numbers instead of the losses, which is the way to check what a run drew:

```bash
evepy 1 1 | gulmc -S 100 -d 1 -o hazard_rands.bin      # hazard sampling numbers
evepy 1 1 | gulmc -S 100 -d 2 -o damage_rands.bin      # damage sampling numbers
```

`-d 0` (the default) writes the ground up loss stream.

```{note}
**Historical note (ktools).** ktools offered three ways to source random numbers, selected with
`-R{buffer size}`, `-r` (read `random.bin` from the static directory) and the default
auto-seeded mode, with an optional `-s{seed}`. Numbers were drawn from a shared buffer addressed
by a random number index (*ridx*) computed from `group_id`, `event_id` and prime moduli. None of
those flags exist in `gulmc`: only the auto-seeded behaviour survives — it is the scheme
described above — and `--random-generator` now selects the generator instead.
```

[Go to Appendix B FM Profiles](fmprofiles.md)

[Back to Contents](Contents.md)
