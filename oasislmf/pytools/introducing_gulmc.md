# Introducing `gulmc`

`gulmc` is a new tool that uses a "full Monte Carlo" approach for ground up losses calculation that, instead of drawing loss samples from the 'effective damageability' probability distribution (as done by calling `eve | modelpy | gulpy`):
it first draws a sample of the hazard intensity, and then draws an independent sample of the damage from the vulnerability function corresponding to the hazard intensity sample.

`gulmc` was first introduced in oasislmf v1.27.0 and is ready for production usage from oasislmf v`1.28.0` onwards.

This document summarizes the changes introduced with `gulmc` with respect to `gulpy`.
Note: features such as the Latin Hypercube Sampler introduced with `gulpy` are not discussed here as they are described at length in the `gulpy` documentation.


## Command line arguments
`gulmc` offers the following command line arguments:
```bash
$ gulmc -h
usage: use "gulmc --help" for more information

options:
  -h, --help            show this help message and exit
  -a ALLOC_RULE         back-allocation rule. Default: 0
  -d DEBUG              output the ground up loss (0), the random numbers used for hazard sampling (1), the random numbers used for damage sampling (2). Default: 0
  -i FILE_IN, --file-in FILE_IN
                        filename of input stream (list of events from `eve`).
  -o FILE_OUT, --file-out FILE_OUT
                        filename of output stream (ground up losses).
  -L LOSS_THRESHOLD     Loss treshold. Default: 1e-6
  -S SAMPLE_SIZE        Sample size. Default: 0
  -V, --version         show program's version number and exit
  --effective-damageability
                        if passed true, the effective damageability is used to draw loss samples instead of full MC. Default: False
  --ignore-correlation  if passed true, peril correlation groups (if defined) are ignored for the generation of correlated samples. Default: False
  --ignore-haz-correlation
                        if passed true, hazard correlation groups (if defined) are ignored for the generation of correlated samples. Default: False
  --ignore-file-type [IGNORE_FILE_TYPE ...]
                        the type of file to be loaded. Default: set()
  --data-server         =Use tcp/sockets for IPC data sharing.
  --logging-level LOGGING_LEVEL
                        logging level (debug:10, info:20, warning:30, error:40, critical:50). Default: 30
  --vuln-cache-size MAX_CACHED_VULN_CDF_SIZE_MB
                        Size in MB of the in-memory cache to store and reuse vulnerability cdf. Default: 200
  --peril-filter PERIL_FILTER [PERIL_FILTER ...]
                        Id of the peril to keep, if empty take all perils
  --random-generator RANDOM_GENERATOR
                        random number generator
                        0: numpy default (MT19937), 1: Latin Hypercube. Default: 1
  --run-dir RUN_DIR     path to the run directory. Default: "."
```

While all of `gulpy` command line arguments are present in `gulmc` with the same usage and functionality, the following command line arguments have been introduced in `gulmc`:
```bash
 --effective-damageability
 --ignore-correlation
 --ignore-haz-correlation
 --data-server
 --vuln-cache-size
 --peril-filter
```
        

#### Comparing `gulpy` and `gulmc` output
`gulmc` runs the same algorithm of `eve | modelpy | gulpy`, i.e., it runs the 'effective damageability' calculation mode, with the same command line arguments. For example, to run a model with 1000 samples, alloc rule 1, and streaming the binary output to the `output.bin` file, can be done with:
```bash
eve 1 1 | modelpy | gulpy -S1000 -a1 -o output.bin
```
or
```bash
eve 1 1 | gulmc -S1000 -a1 -o output.bin
```
#### Hazard uncertainty treatment
If the hazard intensity in the fooprint has no uncertainty, i.e.:
```csv
event_id,areaperil_id,intensity_bin_id,probability
1,4,1,1
[...]
```
then `gulpy` and `gulmc` produce the same outputs. However, if the hazard intensity has a probability distribution, e.g.:
```csv
event_id,areaperil_id,intensity_bin_id,probability
1,4,1,2.0000000298e-01
1,4,2,6.0000002384e-01
1,4,3,2.0000000298e-01
[...]
```
then, by default, `gulmc` runs the full Monte Carlo sampling of the hazard intensity, and then of damage. In order to reproduce the same results that `gulpy` produces can be achieved by using the `--effective-damageability` flag:
```bash
eve 1 1 | gulmc -S1000 -a1 -o output.bin --effective-damageability
```
#### On the usage of `modelpy` and `eve` with `gulmc`
Due to internal refactoring, `gulmc` now incorporates the functionality performed by `modelpy`, therefore `modelpy` should not be used in a pipe with `gulmc`:
```bash
eve 1 1 | modelpy | gulpy -S1000 -a1 -o output.bin        # wrong usage, won't work
eve 1 1 | gulpy -S1000 -a1 -o output.bin                  # correct usage
```
> **Note:** both `gulpy` and `gulmc` can read the events stream from binary file, i.e., without the need of `eve`, with:
> ```bash
>  gulmc -i input/events.bin -S1000 -a1 -o output.bin
> ```

#### Printing the random values used for sampling
Since we now sample in two dimensions (hazard intensity and damage), the `-d` flag is revamped to output both random values used for sampling. While `gulpy -d` printed the random values used to sample the effective damageability distribution, in `gulmc`:
```bash
gulmc -d1 [...]   # prints the random values used for the hazard intensity sampling
gulmc -d2 [...]   # prints the random values used for the damage sampling
```
> **Note:** if the `--effective-damageability` flag is used, only `-d2` is valid since there is no sampling of the hazard intensity, and the random value printed are those used for the effective damageability sampling.

> **Note:**  if `-d1` or `-d2` are passed, the only valid `alloc_rule` value is `0`. This is because, when printing the random values, back-allocation is not meaningful. `alloc_rule=0` is the default value or it can be set with `-a0`. If a value other than 0 is passed to `-a`, an error will be thrown.