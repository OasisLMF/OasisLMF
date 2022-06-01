# Introducing `gulpy`

`gulpy` is the new tool for the computation of ground up losses that is set to replace `gulcalc` in the 
Oasis Loss Modelling Framework.

`gulpy` is ready for production usage from oasislmf > v`X.Y.Z`.

This document summarizes the changes introduced with `gulpy` in terms of command line arguments and features.

## Command line arguments
`gulpy` offers the following command line arguments:
```bash
$ gulpy -h
usage: use "gulpy --help" for more information

optional arguments:
  -h, --help            show this help message and exit
  -a ALLOC_RULE         back-allocation rule
  -d                    output random numbers instead of gul (default: False).
  -i FILE_IN, --file-in FILE_IN
                        filename of input stream.
  -o FILE_OUT, --file-out FILE_OUT
                        filename of output stream.
  -L LOSS_THRESHOLD     Loss treshold (default: 1e-6)
  -S SAMPLE_SIZE        Sample size (default: 0).
  -V, --version         show program version number and exit
  --ignore-file-type [IGNORE_FILE_TYPE [IGNORE_FILE_TYPE ...]]
                        the type of file to be loaded
  --random-generator RANDOM_GENERATOR
                        random number generator
                        0: numpy default (MT19937), 1: Latin Hypercube. Default: 1.
  --run-dir RUN_DIR     path to the run directory
  --logging-level LOGGING_LEVEL
                        logging level (debug:10, info:20, warning:30, error:40, critical:50). Default: 30.

```

The following `gulcalc` arguments were ported to `gulpy` with the same meaning and requirements:
```bash
 -a, -d, -h, -L, -S
```
The following `gulcalc` arguments were ported to `gulpy` but were renamed:
```bash
# in gulcalc             # in gulpy
-v                       -V, --version
-i                       -o, --file-out
```
The following `gulcalc` arguments were **not** ported to `gulpy`:
```bash
-r, -R, -c, -j, -s, -A, -l, -b, -v
```

The following arguments were introduced with `gulpy`:
```bash
--file-in, --ignore-file-type, --random-generator, --run-dir, --logging-level
```

## New random number generator: the Latin Hypercube Sampling algorithm
To compute random loss samples, it is necessary to draw random values from the effective damageability probability distribution function (PDF).
Drawing random values from a given PDF is normally achieved by generating a random float value between 0 and 1 and by taking the inverse of the cumulative
distribution function (CDF) for such random value. The collection of random values produced with this approach will be distributed according to the PDF.

To generate random values `gulcalc` uses the [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister) generator. In `gulpy`, instead, we introduce the [Latin Hypercube Sampling (LHS)](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)  as the default algorithm to generate random values.
Compared to the Mersenne Twister, LHS implements a sort of *stratified* random number generation that more evenly probes the range between 0 and 1, which translates in a faster convergence to the desired PDF. 

In other words, in order to probe a given PDF to the same accuracy, the LHS algorithm requires a smaller number of samples than the Mersenne Twister.


## Examples
### Setting the Output
In order to run the ground-up loss calculation and stream the output to stdout in binary format, the following commands are equivalent:
```bash
# with gulcalc                # with gulpy
gulcalc -a0 -S10 -i -         gulpy -a0 -S10
gulcalc -a1 -S20 -i -         gulpy -a1 -S20
gulcalc -a2 -S30 -i -         gulpy -a2 -S30
```
Alternatively, the binary output can be redirected to file with:
```bash
# with gulcalc                          # with gulpy                          # with gulpy [alternative]
gulcalc -a0 -S10 -i gul_out.bin         gulpy -a0 -S10 -o gul_out.bin         gulpy -a0 -S10 --file-out gul_out.bin
gulcalc -a1 -S20 -i gul_out.bin         gulpy -a1 -S20 -o gul_out.bin         gulpy -a1 -S20 --file-out gul_out.bin
gulcalc -a2 -S30 -i gul_out.bin         gulpy -a2 -S30 -o gul_out.bin         gulpy -a2 -S30 --file-out gul_out.bin
```

### Choosing the random number generator
By default, `gulpy` uses the LHS algorithm to draw random numbers samples, which is shown to require less samples than the Mersenne Twister used by `gulcalc` when probing a given probability distribution function. 

If needed, the user can force `gulpy` to use a specific random number generator:
```bash
gulpy --random-generator 0   # uses Mersenne Twister (like gulcalc)
gulpy --random-generator 1   # uses Latin Hypercube Sampling algorithm (new in gulpy)
```


## Performance
As of oasislmf version 1.0.26.rc1 `gulpy` is not used by default in the oasislmf MDK but it can be used by passing the `--gulpy` argument, e.g.:

```bash
# using gulcalc                 # using gulpy
oasislmf model run              oasislmf model run --gulpy
```

On a real windstorm model these are the execution times:

```bash
# command                              # info on this run           # total execution time     # uses                 # speedup
oasislmf model run                     [  10 samples  -a0 rule ]     3634 sec ~ 1h             getmodel + gulcalc     1.0x      [baseline for  10 samples]
oasislmf model run --modelpy           [  10 samples  -a0 rule ]     1544 sec ~ 25 min         modelpy  + gulcalc     2.4x
oasislmf model run --modelpy --gulpy   [  10 samples  -a0 rule ]     1508 sec ~ 25 min         modelpy  + gulpy       2.4x
oasislmf model run                     [ 250 samples  -a0 rule ]    10710 sec ~ 3h             getmodel + gulcalc     1.0x      [baseline for 250 samples]
oasislmf model run --modelpy           [ 250 samples  -a0 rule ]     8617 sec ~ 2h 23 min      modelpy  + gulcalc     1.2x
oasislmf model run --modelpy --gulpy   [ 250 samples  -a0 rule ]     4969 sec ~ 1h 23 min      modelpy  + gulpy       2.2x
```
