# TEST ADJUSTMENTS
This is a small model used to test that the custom adjustments to the vulnerability functions are correctly applied.

The model has:
 - 4 events
 - 1 areaperil_id
 - 1 vulnerability function
 - 1 analysis settings json file
 - 1 item
 - 1 coverage
 - no correlation

The footprint has no hazard intensity uncertainty and is designed to deterministically sample the different absolute damage bins defined in the model.

## Generating the binary files
The `static/` and `input/` directories contain a specialised `Makefile` each. 
By running `make` inside those directories, the binary files are created from the `.csv` files.
