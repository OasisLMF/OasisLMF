![alt text](/_static/images/kernel/banner.jpg "banner")
# 4.1 Core Components <a id="corecomponents"></a>

<a id="eve"></a>
### eve
***
eve takes a list of event ids in binary format as input and generates a partition of event ids as a binary data stream, according to the parameters supplied. Events are "shuffled" by being assigned to processes one by one cyclically, rather than being distributed to processes in blocks, in the order they are input. This helps to even the workload by process, in case all the "big" events are together in the same range of the event list.

##### Output stream
The output stream is a simple list of event_ids (4 byte integers).

##### Parameters
Required parameters are;
* process number - specifies the process number to receive a partition of events
* number of processes - determines the total number of partitions of events to distribute to processes

Optional parameters are;
* -n - No shuffled events. Events are split up and distributed in blocks in respect of the order they are input.
* -r - Randomise events. Events are split up and distributed in blocks randomly using the Fisher-Yates shuffle.
No shuffled events takes priority in cases where both optional parameters are supplied.


##### Usage
```
$ eve [parameters] > [output].bin
$ eve [parameters] | getmodel | gulcalc [parameters] > [stdout].bin
```

##### Example
```
$ eve 1 2 > events1_2_shuffled.bin 
$ eve -n 1 2 > events1_2_unshuffled.bin 
$ eve -r 1 2 > events1_2_random.bin
$ eve 1 2 | getmodel | gulcalc -r -S100 -i - > gulcalc1_2.bin
```

In this example, the events from the file events.bin will be read into memory and the first half (partition 1 of 2) would be streamed out to binary file, or downstream to a single process calculation workflow.

##### Internal data
The program requires an event binary. The file is picked up from the input sub-directory relative to where the program is invoked and has the following filename;
* input/events.bin

The data structure of events.bin is a simple list of event ids (4 byte integers).

[Return to top](#corecomponents)

<a id="getmodel"></a>
### getmodel 
***
getmodel generates a stream of effective damageability distributions (cdfs) from an input list of events. Specifically, it combines the probability distributions from the model files, footprint.bin and vulnerability.bin, to generate effective damageability cdfs for the subset of exposures contained in the items.bin file and converts them into a binary stream. 

This is reference example of the class of programs which generates the damage distributions for an event set and streams them into memory. It is envisaged that model developers who wish to use the toolkit as a back-end calculator of their existing platforms can write their own version of getmodel, reading in their own source data and converting it into the standard output stream. As long as the standard input and output structures are adhered to, each program can be written in any language and read any input data.

##### Output stream

| Byte 1 | Bytes 2-4 |  Description                                   |
|:-------|-----------|:-----------------------------------------------|
|    0   |     1     |  cdf stream                                    |

##### Parameters
None

##### Usage
```
$ [stdin component] | getmodel | [stout component]
$ [stdin component] | getmodel > [stdout].bin
$ getmodel < [stdin].bin > [stdout].bin
```

##### Example
```
$ eve 1 1 | getmodel | gulcalc -r -S100 -i gulcalci.bin
$ eve 1 1 | getmodel > getmodel.bin
$ getmodel < events.bin > getmodel.bin 
```

##### Internal data
The program requires the footprint binary and index file for the model, the vulnerability binary model file, and the items file representing the user's exposures. The files are picked up from sub-directories relative to where the program is invoked, as follows;

* static/footprint.bin
* static/footprint.idx
* static/vulnerability.bin
* static/damage_bin_dict.bin
* input/items.bin

The getmodel output stream is ordered by event and streamed out in blocks for each event. 

##### Calculation
The program filters the footprint binary file for all areaperil_id's which appear in the items file. This selects the event footprints that impact the exposures on the basis on their location.  Similarly the program filters the vulnerability file for vulnerability_id's that appear in the items file. This selects conditional damage distributions which are relevant for the exposures. 

The intensity distributions from the footprint file and conditional damage distributions from the vulnerability file are convolved for every combination of areaperil_id and vulnerability_id in the items file. The effective damage probabilities are calculated, for each damage bin, by summing the product of conditional damage probabilities with intensity probabilities for each event, areaperil, vulnerability combination across the intensity bins.  

The resulting discrete probability distributions are converted into discrete cumulative distribution functions 'cdfs'.  Finally, the damage bin mid-point from the damage bin dictionary ('interpolation' field) is read in as a new field in the cdf stream as 'bin_mean'.  This field is the conditional mean damage for the bin and it is used to choose the interpolation method for random sampling and numerical integration calculations in the gulcalc component. 

[Return to top](#corecomponents)

<a id="gulcalc"></a>
### gulcalc
***
The gulcalc program performs Monte Carlo sampling of ground up loss by randomly sampling the cumulative probability of damage from the uniform distribution and generating damage factors by interpolation of the random numbers against the effective damage cdf. Other loss metrics are computed and assigned to special meaning sample index values as descibed below. 

The sampling methodologies are linear interpolation, quadratic interpolation and point value sampling depending on the damage bin definitions in the input data. 

Gulcalc also performs back-allocation of total coverage losses to the contributing subperil item losses (for multi-subperil models).  This occurs when there are two or more items representing losses from different subperils to the same coverage, such as wind loss and storm surge loss, for example. In these cases, because the subperil losses are generated independently from each other it is possible to result in a total damage ratio greater than 1 for the coverage, or a total loss greated than the Total Insured Value "TIV". Back-allocation ensures that the total loss for a coverage cannot exceed the input TIV.

##### Stream output

| Byte 1 | Bytes 2-4 |  Description                                   |
|:-------|-----------|:-----------------------------------------------|
|    2   |     1     |  loss stream                                   |

##### Parameters
Required parameters are;
* -S{number}. Number of samples
* -a{number} -i{destination}

The destination is either a filename or named pipe, or use - for standard output.

Optional parameters are;
* -L{number} loss threshold (optional) excludes losses below the threshold from the output stream
* -d debug mode - output random numbers rather than losses (optional)
* -A automatically hashed seed driven random number generation (default)
* -R{number} Number of random numbers to generate dynamically 
* -s{number} Manual seed for random numbers (to make them repeatable)
* -r Read random numbers from random.bin file

##### Usage
```
$ [stdin component] | gulcalc [parameters] | [stout component]
$ [stdin component] | gulcalc [parameters]
$ gulcalc [parameters] < [stdin].bin 
```

##### Example
```
$ eve 1 1 | getmodel | gulcalc -R1000000 -S100 -a1 -i - | fmcalc > fmcalc.bin
$ eve 1 1 | getmodel | gulcalc -R1000000 -S100 -a1 -i - | summarycalc -i -1 summarycalc1.bin
$ eve 1 1 | getmodel | gulcalc -r -S100 -a1 -i gulcalci.bin
$ gulcalc -r -S100 -i -a1 gulcalci.bin < getmodel.bin 
```

##### Internal data
The program requires the damage bin dictionary binary for the static folder and the item and coverage binaries from the input folder. The files are found in the following locations relative to the working directory with the filenames;

* static/damage_bin_dictionary.bin
* input/items.bin
* input/coverages.bin

If the user specifies -r as a parameter, then the program also picks up a random number file from the static directory. The filename is;
* static/random.bin

##### Calculation
The stdin stream is a block of cdfs which are ordered by event_id, areaperil_id, vulnerability_id and bin_index ascending, from getmodel. The gulcalc program constructs a cdf for each item, based on matching the areaperil_id and vulnerability_id from the stdin and the item file.

###### Random sampling
Random samples are indexed using positive integers starting from 1, called the 'sidx', or sample index.

For each item cdf and for the number of samples specified, the program draws a uniformly distributed random number and uses it to sample ground up loss from the cdf using one of three methods, as follows;

For a given damage interval corresponding to a cumulative probability interval that each random number falls within;
* If the conditional mean damage (of the cdf) is the mid-point of the damage bin interval (of the damage bin dictionary) then the gulcalc program performs linear interpolation. 
* If the conditional mean damage is equal to the lower and upper damage threshold of the damage bin interval (i.e the bin represents a point value, not a range) then that point value is sampled.
* Else, the gulcalc program performs quadrative interpolation using the bin_mean to calculate the quadratic equation in the damage interval. The bin mean must lie within the middle third of the damage interval to generate a valid quadrative cdf. This requirement comes from the necessary condition that the probability density function must be >= 0 across the entire damage interval.

An example of the three cases and methods is given below;
 
| bin_from | bin_to |  bin_mean | Method used             |
|:---------|--------|-----------| :-----------------------|
|    0.1   |  0.2   |    0.15   | Linear interpolation    |
|    0.1   |  0.1   |    0.1    | Sample bin value        |
|    0.1   |  0.2   |    0.14   | Quadratic interpolation |

If the -R parameter is used along with a specified number of random numbers then random numbers used for sampling are generated on the fly for each event and group of items which have a common group_id using the Mersenne twister psuedo random number generator (the default RNG of the C++ v11 compiler).  These random numbers are not repeatable, unless a seed is also specified (-s{number}).

If the -r parameter is used, gulcalc reads a random number from the provided random number file, which produces repeatable results. 

The default random number behaviour (no additional parameters) is to generate random numbers from a seed determined by a combination of the event_id and group_id, which produces repeatable results. See [Random Numbers](RandomNumbers.md) for more details.

Each sampled damage is multiplied by the item TIV, looked up from the coverage file.

###### Special samples

Samples with negative indexes have special meanings as follows;

| sidx     | description                               |
|:---------| :-----------------------------------------|
|    -1    | Numerical integration mean                |
|    -2    | Numerical integration standard deviation  |
|    -3    | Impacted exposure 						   |
|    -4    | Chance of loss                            |
|    -5    | Maximum loss                              |

* The numerical integration mean loss, sidx=-1, is computed by multiplying the item TIV looked up from the coverage file with the mean damage. The mean damage is computed by a sum across all damage bins of the product of the damage bin mean and the probability density (the difference between consecutive damage bin prob_to values).
* The numerical integration standard deviation of loss, sidx=-2, is computed by multiplying the item TIV looked up from the coverage file with the standard deviation of damage. The standard deviation of damage is the sqrt of the sum across all damage bins of the product of the squared errors between the numerical integration mean damage and each damage bin mean with the probability density (the difference between consecutive damage bin prob_to values).
* The impacted exposure, sidx=-3 represents the 100% damage scenario to all items impacted by (within the footprint of) an event. It is 1 multiplied by the item TIV looked up from the coverage file.
* The chance of loss, sidx -4, is the probability that, conditional on the event occurring, the damage/loss is greater than zero. This value is computed directly from the damage distribution. Its value is 1 if the upper threshold of the first damage bin is non-zero (meaning no chance of zero damage), else it is 1 - prob_to of the first damage bin.
* The maximum loss, sidx -5, represents the maximum possible loss computed from the damage distribution. This is the upper damage threshold of the first damage bin which has prob_to = 1, multiplied by the item TIV from the coverage file. 

###### Allocation method

The allocation method determines how item losses are adjusted when a coverage is subject to losses from multiple perils, because the total loss to a coverage from mutiple perils cannot exceed the input TIV. This situation is identified when multiple item_ids in the item file share the same coverage_id.  The TIV is held in the coverages file against the coverage_id and the item_id TIV is looked up from its relationship to coverage_id in the item file.

The allocation methods are as follows;

|  	a         | description                                                                  |
|:------------| :-----------------------------------------------------------------------------------------------------------------------------|
| 0           | Pass losses through unadjusted (used for single peril models)                                                                 |
| 1           | Sum the losses and cap them to the TIV. Back-allocate TIV to the contributing items in proportion to the unadjusted losses    |
| 2           | Keep the maximum subperil loss and set the others to zero. Back-allocate equally when there are equal maximum losses          |

The mean, impacted exposure and maximum loss special samples are also subject to these allocation rules. 
The impacted exposure value, sidx -3, is always back-allocated equally to the items, for allocation rules 1 and 2, since by definition it is the same value for all items related to the same coverage.


[Return to top](#corecomponents)

<a id="fmcalc"></a>
### fmcalc
***
fmcalc is the reference implementation of the Oasis Financial Module. It applies policy terms and conditions to the ground up losses and produces loss sample output.  It reads in the loss stream from either gulcalc or from another fmcalc and can be called recursively and apply several consecutive sets of policy terms and conditions. 

##### Stream output

| Byte 1 | Bytes 2-4 |  Description                                   |
|:-------|-----------|:-----------------------------------------------|
|    2   |     1     |  loss stream                                   |

##### Parameters
Optional parameters are;
* -p {string} The location of the input files. The default location is the 'input' directory in the present working directory
* -a{integer} The back allocation rule to apply. The options are 0 (no allocation), 1 (ground up loss basis) or 2 (prior level loss basis). The default is 0. 
* -n Output net losses. Net losses are the difference between the input loss and the calculated loss. The default is to output the calculated loss.
* -S Use fm_profile_step input file (default is to use fm_profile)

##### Usage
```
$ [stdin component] | fmcalc [parameters] | [stout component]
$ [stdin component] | fmcalc [parameters] > [stdout].bin
$ fmcalc [parameters] < [stdin].bin > [stdout].bin
```

##### Example
```
$ eve 1 1 | getmodel | gulcalc -r -S100 -a1 -i - | fmcalc -p direct -a2 | summarycalc -f -2 - | eltcalc > elt.csv
$ eve 1 1 | getmodel | gulcalc -r -S100 -a1 -i - | fmcalc -p direct -a1 > fmcalc.bin
$ fmcalc -p ri1 -a2 -S -n < gulcalci.bin > fmcalc.bin
$ fmcalc -p direct | fmcalc -p ri1 -n | fmcalc -p ri2 -n < gulcalci.bin > fm_ri2_net.bin 
```

##### Internal data
For the gulcalc item stream input, the program requires the item, coverage and fm input data files, which are Oasis abstract data objects that describe an insurance or reinsurance programme. This data is picked up from the following files relative to the working directory by default;

* input/items.bin
* input/coverages.bin
* input/fm_programme.bin
* input/fm_policytc.bin
* input/fm_profile.bin or fm_profile_step.bin
* input/fm_xref.bin

For loss stream input from either gulcalc or fmcalc, the program requires only the four fm input data files, 

* input/fm_programme.bin
* input/fm_policytc.bin
* input/fm_profile.bin or fm_profile_step.bin
* input/fm_xref.bin

The location of the files can be changed by using the -p parameter followed by the path location relative to the present working directory. eg -p ri1

##### Calculation

fmcalc passes the loss samples, including the numerical integration mean, sidx -1, and impacted exposure, sidx -3, through a set of financial calculations which are defined by the input files. The special samples -2, -4 and -5 are ignored and dropped in the output. For more information about the calculation see [Financial Module](../../explanation/financial-module.rst)

[Return to top](#corecomponents)

<a id="summarycalc"></a>
### summarycalc 
***
The purpose of summarycalc is firstly to aggregate the samples of loss to a level of interest for reporting, thereby reducing the volume of data in the stream. This is a generic first step which precedes all of the downstream output calculations.  Secondly, it unifies the formats of the gulcalc and fmcalc streams, so that they are transformed into an identical stream type for downstream outputs. Finally, it can generate up to 10 summary level outputs in one go, creating multiple output streams or files.

The output is similar to the gulcalc or fmcalc input which are losses are by sample index and by event, but the ground up or (re)insurance loss input losses are grouped to an abstract level represented by a summary_id.  The relationship between the input identifier and the summary_id are defined in cross reference files called **gulsummaryxref** and **fmsummaryxref**.

##### Stream output

| Byte 1 | Bytes 2-4 |  Description                                   |
|:-------|-----------|:-----------------------------------------------|
|    3   |     1     |  summary stream                                |

##### Parameters

The input stream should be identified explicitly as -i input from gulcalc or -f input from fmcalc.

summarycalc supports up to 10 concurrent outputs.  This is achieved by explictly directing each output to a named pipe, file, or to standard output.  

For each output stream, the following tuple of parameters must be specified for at least one summary set;

* summaryset_id number. valid values are -0 to -9
* destination pipe or file. Use either - for standard output, or {name} for a named pipe or file.

For example the following parameter choices are valid;
```
$ summarycalc -i -1 -                                       
'outputs results for summaryset 1 to standard output
$ summarycalc -i -1 summarycalc1.bin                        
'outputs results for summaryset 1 to a file (or named pipe)
$ summarycalc -i -1 summarycalc1.bin -2 summarycalc2.bin    
'outputs results for summaryset 1 and 2 to a file (or named pipe)
```
Note that the summaryset_id relates to a summaryset_id in the required input data file **gulsummaryxref.bin** or **fmsummaryxref.bin** for a gulcalc input stream or a fmcalc input stream, respectively, and represents a user specified summary reporting level. For example summaryset_id = 1 represents portfolio level, summaryset_id = 2 represents zipcode level and summaryset_id 3 represents site level.

##### Usage
```
$ [stdin component] | summarycalc [parameters] | [stdout component]
$ [stdin component] | summarycalc [parameters]
$ summarycalc [parameters] < [stdin].bin
```

##### Example

```
$ eve 1 1 | getmodel | gulcalc -r -S100 -a1 -i - | summarycalc -i -1 - | eltcalc > eltcalc.csv
$ eve 1 1 | getmodel | gulcalc -r -S100 -a1 -i - | summarycalc -i -1 gulsummarycalc.bin 
$ eve 1 1 | getmodel | gulcalc -r -S100 -a1 -i - | fmcalc | summarycalc -f -1 fmsummarycalc.bin 
$ summarycalc -f -1 fmsummarycalc.bin < fmcalc.bin
```

##### Internal data
The program requires the gulsummaryxref file for gulcalc input (-i option), or the fmsummaryxref file for fmcalc input (-f option). This data is picked up from the following files relative to the working directory;

* input/gulsummaryxref.bin
* input/fmsummaryxref.bin

##### Calculation
summarycalc takes either ground up loss from gulcalc or financial loss samples from fmcalc as input and aggregates them to a user-defined summary reporting level. The output is similar to the input, individual losses by sample index and by event, but the ground up or financial losses are summed to an abstract level represented by a summary_id.  The relationship between the input identifier, item_id for gulcalc or output_id for fmcalc, and the summary_id are defined in the input files.

The special samples are computed as follows;

* The numerical integration mean, sidx -1, impacted exposure, sidx -3, maximum loss value, sidx -5, are treated as normal samples and summed to each summary_id.
* The numerical integration standard deviation, sidx -2, is dropped.
* From gulcalc input only, the chance of loss, sidx -4, at a given summary level is aggregated using the law of total probability, evaluated as  1 - probability all items under a summary_id have zero loss.
1- (1-C1)(1-C2)...(1-CN) where C1, C2, ..CN represents the chance of loss for item 1,2..N under each summary_id, which are present in the gulcalc stream.

[Return to top](#corecomponents)

[Go to 4.2 Output Components section](OutputComponents.md)

[Back to Contents](Contents.md)
