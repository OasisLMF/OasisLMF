# TEST MODEL 5

This is a small model useful for unit testing.

The model has:
 - 5 events
 - 2 areaperil_ids
 - 12 vulnerability functions
 - 1 item
 - 1 coverage
 - 2 perils

The footprint has hazard intensity uncertainty, i.e., for each (event, areaperil_id) entry
there is a hazard intensity distribution, e.g.:
```py
event_id,areaperil_id,intensity_bin_id,probability
1,4,1,2.0000000298e-01
1,4,2,6.0000002384e-01
1,4,3,2.0000000298e-01
```
## Purpose
The purpose of this model is to test the weighting calculation for an aggregate vulnerability with weights provided for multiple detailed vulnerability_ids. The resulting blended distribution has been independently calculated and the mean (sidx -1), standard deviation (sidx -2) and maximum loss (sidx -5) computed and compared for a single item with aggregate vulnerability_id. The parity of these metrics are necessary and sufficient to test that the internally calculated damage distribution agrees with the independently calculated distribution.

The modifications to test_model_1 are 

- remove items 2-10
- change the vulnerability_id of item 1 to 10003
- addition of aggregate_vulnerability file to static
- addition of weights file to static

The weights have 10% vulnerabiity_id 6, 20% vulnerability_id 7, 30% vulnerability_id 8 and 40% vulnerability_id 9 assigned to aggregate vulnerability id 10003. 


## Generating the binary files
The `static/` and `input/` directories contain a specialised `Makefile` each. 
By running `make` inside those directories, the binary files are created from the `.csv` files.

### Note on `correlations.bin`,`aggregate_vulnerability.bin` and `weights.bin`
At the time of writing, there is no command line tool to
convert `correlations.csv`, `aggregate_vulnerability` and `weights` to binary format, therefore `input/Makefile` does not produce `correlations.bin`, and `static/Makefile` does not produce `aggregate_vulnerability.bin` and `weights.bin`. 

`correlations.bin` can be created by executing the following Python code within the `input/` directory:
```py
from oasislmf.pytools.data_layer.conversions.correlations import CorrelationsData

CorrelationsData.from_csv("correlations.csv").to_bin("correlations.bin")
```