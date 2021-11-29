# Get Model 
This directory houses the code for the python get model entry point with the following files:

- get_model.py
- vulnerability.py
- footprint.py
- common.py

## Files

### get_model
This file is the entry point for the python get model command. Calling this command will run the ```main``` function 
which gathers the inputs from the user and passes them through to the ```run``` function which in turn loads the 
data and constructs the model.

### vulnerability 
This file houses a series of functions that load, convert, and save vulnerability data with Parquet files.

### footprint
This file houses the classes that load the footprint data from compressed, binary, and CSV files.

### Common
This file defines the data types that are loaded from the data files.


## Isolated development 
When working on the get model code it is not always optimal to be running the full model repeatedly for every code 
iteration. You may also need to run a debugger in your IDE. You can partake in test-driven/isolated development
in the ```tests/pytools/test_getmodel/test_get_model.py``` Binary, compressed, CSV, and parquet data is in the 
```tests/pytools/test_getmodel/``` There is documentation in the  ```tests/pytools/test_getmodel/test_get_model.py```
file and example functions are commented out to help with the development. The example can be seen below:
```python
from unittest import main, TestCase

from oasislmf.pytools.getmodel.get_model import get_items, get_vulns, Footprint

import numpy as np
import numba as nb
import pyarrow.parquet as pq


class GetModelTests(TestCase):

    def test_load_footprint(self):
        with Footprint.load(static_path="./static/") as test:
            outcome = test
        print(outcome)

    def test_get_vulns(self):
        vulns_dict = get_items(input_path="./", file_type="bin")[0]
        first_outcome = get_vulns(static_path="./static/", vuln_dict=vulns_dict, num_intensity_bins=50, file_type="bin")
        vulnerability_array = first_outcome[0]

        second_outcome = get_vulns(static_path="./static/", vuln_dict=vulns_dict, num_intensity_bins=50,
                                   file_type="parquet")


if __name__ == "__main__":
    main()
```
You should be able to use your IDE to run the functions in the class above with a click of the mouse and run debugging 
on individual functions.

## Commandline arguments 
When running the Python get model command, we can parse in the following arguments:

- ```--file-in```: names of the input file_path
- ```--file-out```: names of the output file_path
- ```--run-dir```: path to the run directory


## Writing custom Footprint Objects 
Custom footprint loading objects can be built using the ```FootprintLoadMixin``` class. This mixin will manage the 
context in which the whole get model process is run. It will also check your object to ensure that it has the 
correct methods and attributes to load the footprint data and pass it onto other areas of the getmodel process. Feel 
free to have any attributes and methods that help you obtain your goals however, when 
using this mixin your object needs the following attributes and methods:

### Methods
- **load** => this is a method and loads the footprint data from the file. This load function should populate the 
```self.num_intensity_bins```, ```self.footprint```, and ```self.footprint_index``` attributes of the object calling
the ```load``` function.
- **get_event** => This function has to take in a parameter ```event_id``` and returns an event from the 
```self.footprint```.

### Attributes
- **num_intensity_bins**: (int) => this is the number of intensity bins that the footprint data has 
- **footprint**: (np.array[EventCSV]) => the main data body of the footprint
- **footprint_index**: (dict) => map of footprint IDs with the index in the data

### Structs 
```python
Event = nb.from_dtype(np.dtype([('areaperil_id', np.dtype('u4')),
                                ('intensity_bin_id', np.int32),
                                ('probability', np.dtype('f4')))
                                ]))
```