# TEST MODEL 1

This is a small model useful for unit testing.

The model has:
 - 5 events
 - 2 areaperil_ids
 - 12 vulnerability functions
 - 10 items
 - 10 coverages
 - 3 peril correlation groups

The footprint has hazard intensity uncertainty, i.e., for each (event, areaperil_id) entry
there is a hazard intensity distribution, e.g.:
```py
event_id,areaperil_id,intensity_bin_id,probability
1,4,1,2.0000000298e-01
1,4,2,6.0000002384e-01
1,4,3,2.0000000298e-01
```

## Generating the binary files
The `static/` and `input/` directories contain a specialised `Makefile` each. 
By running `make` inside those directories, the binary files are created from the `.csv` files.

### Note on `correlations.bin`
At the time of writing, there is no command line tool to
convert `correlations.csv` to binary format, therefore `input/Makefile` does not produce `correlations.bin`. 

`correlations.bin` can be created by executing the following Python code within the `input/` directory:
```py
from oasislmf.pytools.data_layer.conversions.correlations import CorrelationsData

CorrelationsData.from_csv("correlations.csv").to_bin("correlations.bin")
```