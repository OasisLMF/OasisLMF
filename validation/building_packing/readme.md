# building_packing

One aggregate location under a two-layer policy, sized so the three disaggregation modes can be
compared directly. The point of the case is that `items` and `samples` describe the *same* risk --
four buildings each carrying a quarter of the location's TIV, each its own risk with its own site
terms -- and so must produce the same loss. Only where the buildings live differs: `items` gives
each one its own item, `samples` keeps one item and multiplexes them into the sample dimension.

What each part is for:

- **`IsAggregate=1` with `NumberOfBuildings=4`** is the gate on packing. Without `IsAggregate` the
  buildings are summed before any term applies, nothing downstream can tell them apart, and the
  run is not packed however the mode is set.
- **One peril** (`BBF`, not the `AA1` group) keeps the location to a single item, so the site level
  has one node per building rather than one per building per peril.
- **Deductibles with a min and a max** at both Loc and Pol level put the levels on a calcrule that
  uses the financial module's extras array.
- **Two `LayerNumber` rows sharing identical `Pol*6All` terms** make the policy-all level a single
  profile underneath a two-layer level -- the layer storage then aliases onto layer 0, which is
  where packing and row disaggregation can diverge.
- The layer limits are deliberately non-binding, so the result is driven by the deductibles rather
  than saturating at a cap and hiding a difference.

`expected/` is the `items` run, which is what a model run produced before packing existed.
