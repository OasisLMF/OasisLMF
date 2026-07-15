# 4.2 Output Components <a id="outputcomponents"></a>

### eltcalc <a id="eltcalc"></a>
***
The program calculates mean and standard deviation of loss by summary_id and by event_id.

##### Parameters

None

##### Usage
```
$ [stdin component] | eltcalc > elt.csv
$ eltcalc < [stdin].bin > elt.csv
```

##### Example
```
$ eve 1 1 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - | eltcalc > elt.csv
$ eltcalc < summarycalc.bin > elt.csv 
```

##### Internal data

No additional data is required, all the information is contained within the input stream. 

##### Calculation

For each summary_id and event_id, the sample mean and standard deviation is calculated from the sampled losses in the summarycalc stream and output to file.  The analytical mean is also output as a seperate record, differentiated by a 'type' field. The exposure_value, which is carried in the event_id, summary_id header of the stream is also output.

##### Output
csv file with the following fields;

| Name              | Type   |  Bytes | Description                                                 | Example     |
|:------------------|--------|--------| :-----------------------------------------------------------|------------:|
| summary_id        | int    |    4   | summary_id representing a grouping of losses                |   10        |
| type              | int    |    4   | 1 for analytical mean, 2 for sample mean                    |  2          |
| event_id          | int    |    4   | Oasis event_id                                              |  45567      |
| mean              | float  |    4   | mean                                                        |   1345.678  |
| standard_deviation| float  |    4   | sample standard deviation                                   |    945.89   | 
| exposure_value    | float  |    4   | exposure value for summary_id affected by the event         |   70000     |

[Return to top](#outputcomponents)

### leccalc <a id="leccalc"></a>
***
Loss exceedance curves, also known as exceedance probability curves, are computed by a rank ordering a set of losses by period and computing the probability of exceedance for each level of loss based on relative frequency. Losses are first assigned to periods of time (typically years) by reference to the **occurrence** file which contains the event occurrences in each period over a timeline of, say, 10,000 periods. Event losses are summed within each period for an aggregate loss exceedance curve, or the maximum of the event losses in each period is taken for an occurrence loss exceedance curve.  From this point, there are a few variants available as follows;

* Wheatsheaf/multiple EP - losses by period are rank ordered for each sample, which produces many loss exceedance curves - one for each sample across the same timeline. The wheatsheaf shows the variation in return period loss due to sampled damage uncertainty, for a given timeline of occurrences.

* Full uncertainty/single EP - all sampled losses by period are rank ordered to produce a single loss exceedance curve. This treats each sample as if it were another period of losses in an extrapolated timeline. Stacking the curves end-to-end rather then viewing side-by-side as in the wheatsheaf is a form of averaging with respect to a particular return period loss and provides stability in the point estimate, for a given timeline of occurrences. 

* Sample mean - the losses by period are first averaged across the samples, and then a single loss exceedance curve is created from the period sample mean losses. 

* Wheatsheaf mean - the loss exceedance curves from the Wheatsheaf are averaged across each return period, which produces a single loss exceedance curve.

The ranked losses represent the first, second, third,  etc.. largest loss periods within the total number of periods of say 10,000 years.  The relative frequency of these periods of loss is interpreted as the probability of loss exceedance, that is to say that the top ranked loss has an exceedance probability of 1 in 10000, or 0.01%, the second largest loss has an exceedance probability of 0.02%, and so on.  In the output file, the exceedance probability is expressed as a return period, which is the reciprocal of the exceedance probability multiplied by the total number of periods.  Only non-zero loss periods are returned.

##### Parameters

* -K{sub-directory}. The subdirectory of /work containing the input summarycalc binary files.
Then the following tuple of parameters must be specified for at least one analysis type;
* Analysis type. Use -F for Full Uncertainty Aggregate, -f for Full Uncertainty Occurrence, -W for Wheatsheaf Aggregate,  -w for Wheatsheaf Occurrence, -S for Sample Mean Aggregate, -s for Sample Mean Occurrence, -M for Mean of Wheatsheaf Aggregate, -m for Mean of Wheatsheaf Occurrence
* Output filename

An optional parameter is; 
* -r. Use return period file - use this parameter if you are providing a file with a specific list of return periods. If this file is not present then all calculated return periods will be returned, for losses greater than zero.

##### Usage

```
$ leccalc [parameters] > lec.csv

```

##### Examples
```
'First generate summarycalc binaries by running the core workflow, for the required summary set
$ eve 1 2 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - > work/summary1/summarycalc1.bin
$ eve 2 2 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - > work/summary1/summarycalc2.bin
'Then run leccalc, pointing to the specified sub-directory of work containing summarycalc binaries.
$ leccalc -Ksummary1 -F lec_full_uncertainty_agg.csv -f lec_full_uncertainty_occ.csv 
' With return period file
$  leccalc -r -Ksummary1 -F lec_full_uncertainty_agg.csv -f lec_full_uncertainty_occ.csv 
```

##### Internal data

leccalc requires the occurrence.bin file

* input/occurrence.bin

and will optionally use the following additional files if present

* input/returnperiods.bin
* input/periods.bin

leccalc does not have a standard input that can be streamed in. Instead, it reads in summarycalc binary data from a file in a fixed location.  The format of the binaries must match summarycalc standard output. The location is in the 'work' subdirectory of the present working directory. For example;

* work/summarycalc1.bin
* work/summarycalc2.bin
* work/summarycalc3.bin

The user must ensure the work subdirectory exists.  The user may also specify a subdirectory of /work to store these files. e.g.

* work/summaryset1/summarycalc1.bin
* work/summaryset1/summarycalc2.bin
* work/summaryset1/summarycalc3.bin

The reason for leccalc not having an input stream is that the calculation is not valid on a subset of events, i.e. within a single process when the calculation has been distributed across multiple processes.  It must bring together all event losses before assigning event losses to periods and ranking losses by period.  The summarycalc losses for all events (all processes) must be written to the /work folder before running leccalc.

##### Calculation

All files with extension .bin from the specified subdirectory are read into memory, as well as the occurrence.bin. The summarycalc losses are grouped together and sampled losses are assigned to period according to which period the events occur in.

If multiple events occur within a period;
* For **aggregate** loss exceedance curves, the sum of losses is calculated.
* For **occurrence** loss exceedance curves, the maximum loss is calculated.

Then the calculation differs by lec type, as follows;

* Full uncertainty - all losses by period are rank ordered to produce a single loss exceedance curve. 
* Wheatsheaf - losses by period are rank ordered for each sample, which produces many loss exceedance curves - one for each sample across the same timeline.
* Sample mean - the losses by period are first averaged across the samples, and then a single loss exceedance curve is created from the period sample mean losses.
* Wheatsheaf mean - the return period losses from the Wheatsheaf are averaged, which produces a single loss exceedance curve.

For all curves, the analytical mean loss (sidx = -1) is output as a separate exceedance probability curve.  If the calculation is run with 0 samples, then leccalc will still return the analytical mean loss exceedance curve.  The 'type' field in the output identifies the type of loss exceedance curve, which is 1 for analytical mean, and 2 for curves calculated from the samples.

##### Output

csv file with the following fields;

**Full uncertainty, Sample mean and Wheatsheaf mean loss exceedance curve**

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| summary_id        | int    |    4   | summary_id representing a grouping of losses                        |   10        |
| type              | int    |    4   | 1 for analytical mean, 2 for sample mean                            |    2        |
| return_period     | float  |    4   | return period interval                                              |    250      |
| loss              | float  |    4   | loss exceedance threshold for return period                         |    546577.8 |

**Wheatsheaf loss exceedance curve**

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| summary_id        | int    |    4   | summary_id representing a grouping of losses                        |   10        |
| sidx              | int    |    4   | Oasis sample index                                                  |    50       |
| return_period     | float  |    4   | return period interval                                              |    250      |
| loss              | float  |    4   | loss exceedance threshold for return period                         |    546577.8 |

##### Period weightings

An additional feature of leccalc is available to vary the relative importance of the period losses by providing a period weightings file to the calculation. In this file, a weight can be assigned to each period make it more or less important than neutral weighting (1 divided by the total number of periods). For example, if the neutral weight for period 1 is 1 in 10000 years, or 0.0001, then doubling the weighting to 0.0002 will mean that period's loss reoccurrence rate would double.  Assuming no other period losses, the return period of the loss of period 1 in this example would be halved.

All period_nos must appear in the file from 1 to P (no gaps). There is no constraint on the sum of weights. Periods with zero weight will not contribute any losses to the loss exceedance curve.

This feature will be invoked automatically if the periods.bin file is present in the input directory.

[Return to top](#outputcomponents)

### pltcalc <a id="pltcalc"></a>

***
The program outputs sample mean and standard deviation by summary_id,  event_id and period_no.  The analytical mean is also output as a seperate record, differentiated by a 'type' field.  It also outputs an event occurrence date.

##### Parameters

None

##### Usage

```
$ [stdin component] | pltcalc > plt.csv
$ pltcalc < [stdin].bin > plt.csv
```

##### Examplea

```
$ eve 1 1 | getmodel | gulcalc -r -S100 -C1 | summarycalc -1 - | pltcalc > plt.csv
$ pltcalc < summarycalc.bin > plt.csv 
```

##### Internal data

pltcalc requires the occurrence.bin file

* input/occurrence.bin

##### Calculation

The occurrence.bin file is read into memory.  For each summary_id,  event_id and period_no, the sample mean and standard deviation is calculated from the sampled losses in the summarycalc stream and output to file.  The exposure_value, which is carried in the event_id, summary_id header of the stream is also output, as well as the date field(s) from the occurrence file.

##### Output

There are two output formats, depending on whether an event occurrence date is an integer offset to some base date that most external programs can interpret as a real date, or a calendar day in a numbered scenario year. The output format will depend on the format of the date fields in the occurrence.bin file.  

In the former case, the output format is; 

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| type              | int    |    4   | 1 for analytical mean, 2 for sample mean                            |  1          |
| summary_id        | int    |    4   | summary_id representing a grouping of losses                        |  10         |
| event_id          | int    |    4   | Oasis event_id                                                      |  45567      |
| period_no         | int    |    4   | identifying an abstract period of time, such as a year              |  56876      |
| mean              | float  |    4   | mean                                                                |   1345.678  |
| standard_deviation| float  |    4   | sample standard deviation                                           |    945.89   | 
| exposure_value    | float  |    4   | exposure value for summary_id affected by the event                 |   70000     |
| date_id           | int    |    4   | the date_id of the event occurrence                                 |   28616     |

Using a base date of 1/1/1900 the integer 28616 is interpreted as 16/5/1978.

In the latter case, the output format is; 

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| type              | int    |    4   | 1 for analytical mean, 2 for sample mean                            |  1          |
| summary_id        | int    |    4   | summary_id representing a grouping of losses                        |  10         |
| event_id          | int    |    4   | Oasis event_id                                                      |  45567      |
| period_no         | int    |    4   | identifying an abstract period of time, such as a year              |  56876      |
| mean              | float  |    4   | mean                                                                |   1345.678  |
| standard_deviation| float  |    4   | sample standard deviation                                           |    945.89   | 
| exposure_value    | float  |    4   | exposure value for summary_id affected by the event                 |   70000     |
| occ_year          | int    |    4   | the year number of the event occurrence                             |   56876     |
| occ_month         | int    |    4   | the month of the event occurrence                                   |   5         |
| occ_year          | int    |    4   | the day of the event occurrence                                     |   16        |

[Return to top](#outputcomponents)

### aalcalc <a id="aalcalc"></a>
***
aalcalc computes the overall average annual loss and standard deviation of annual loss.

Two types of aal and standard deviation of loss are calculated; analytical (type 1) and sample (type 2).  If the analysis is run with zero samples, then only type 1 statistics are returned by aalcalc.

The Average Loss Converence Table 'ALCT' is a second optional output which can be generated from aalcalc. This provides extra statistical output which can be used to estimate the amount of simulation error in the average annual loss estimate from samples (type 2).

##### Internal data

aalcalc requires the occurrence.bin file 

* input/occurrence.bin

aalcalc does not have a standard input that can be streamed in. Instead, it reads in summarycalc binary data from a file in a fixed location.  The format of the binaries must match summarycalc standard output. The location is in the 'work' subdirectory of the present working directory. For example;

* work/summarycalc1.bin
* work/summarycalc2.bin
* work/summarycalc3.bin

The user must ensure the work subdirectory exists.  The user may also specify a subdirectory of /work to store these files. e.g.

* work/summaryset1/summarycalc1.bin
* work/summaryset1/summarycalc2.bin
* work/summaryset1/summarycalc3.bin

The reason for aalcalc not having an input stream is that the calculation is not valid on a subset of events, i.e. within a single process when the calculation has been distributed across multiple processes.  It must bring together all event losses before assigning event losses to periods and finally computing the final statistics.  

##### Parameters

* -K{sub-directory}. The sub-directory of /work containing the input aalcalc binary files.
* -o -l {confidence_level} -c {file path} - Generates the alct report in addition to the aalcalc report (optional)

##### Usage

```
$ aalcalc [parameters] > aal.csv
```

##### Examples

```
First generate summarycalc binaries by running the core workflow, for the required summary set
$ eve 1 2 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - > work/summary1/summarycalc1.bin
$ eve 2 2 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - > work/summary1/summarycalc2.bin
Then run aalcalc, pointing to the specified sub-directory of work containing summarycalc binaries.
$ aalcalc -Ksummary1 > aal.csv  
Add alct output at 95% confidence level
$ aalcalc -Ksummary1 -o -l 0.95 -c alct.csv > aal.csv  
```

##### Output

**AAL:**

csv file containing the following fields;

| Name                | Type   |  Bytes | Description                                                         | Example     |
|:--------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| summary_id          | int    |    4   | summary_id representing a grouping of losses                        |   1         |
| type                | int    |    4   | 1 for analytical statistics, 2 for sample statistics                |    2        |
| mean                | float  |    8   | average annual loss                                                 |    1014.23  |
| standard_deviation  | float  |    8   | standard deviation of annual loss                                   |    11039.78 |

**ALCT:**

csv file containing the following fields;

| Name                | Type   |  Bytes | Description                                                         | Example     |
|:--------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| SummaryId           | int    |    4   | summary_id representing a grouping of losses                        |   1         |
| MeanLoss            | float  |    4   | the average annual loss estimate from samples                       |    1014.23  |
| SDLoss              | float  |    8   | the standard deviation of annual loss from samples                  |    11039.78 |
| SampleSize          | int    |    8   | the number of samples used to produce the statistics                |    100      |
| LowerCI             | float  |    8   | the lower threshold of the confidence interval for the mean estimate|    1004.52  |
| UpperCI             | float  |    8   | the upper threshold of the confidence interval for the mean estimate|    1023.94  |
| StandardError       | float  |    8   | the total standard error of the mean estimate                       |    5.90     |
| RelativeError       | float  |    8   | the StandardError divided by the mean estimate                      |    0.005    |
| VarElementHaz       | float  |    8   | the contribution to variance of the estimate from the hazard        |    8707.40  |
| StandardErrorHaz    | float  |    8   | the square root of VarElementHaz                                    |    93.31    |
| RelativeErrorHaz    | float  |    8   | the StandardErrorHaz divided by the mean estimate                   |    0.092    |
| VarElementVuln      | float  |    8   | the contribution to variance of the estimate from the vulnerability |    34.81    |
| StandardErrorVuln   | float  |    8   | the square root of VarElementVuln                                   |    5.90     |
| RelativeErrorVuln   | float  |    8   | the StandardErrorVuln divided by the mean estimate                  |    0.005    |

##### Calculation

The occurrence file and summarycalc files from the specified subdirectory are read into memory. Event losses are assigned to periods based on when the events occur and summed by period and by sample. These are referred to as 'annual loss samples'.

**AAL calculation:**

For type 1, calculations are performed on the type 1 (numerically integrated)  mean annual losses by period. The AAL is the mean annual losses summed across the periods and divided by the number of periods.  The standard deviation is the square root of the sum of squared errors between each annual mean loss and the AAL mean divided by the degrees of freedom (periods - 1). 

For type 2 the mean and standard deviation of the annual loss samples are calculated across all samples and periods. The mean estimates the average annual loss, calculated as the sum of all annual loss samples divided by the total number of periods times the number of samples. The standard deviation is the square root of the sum of squared errors between each annual loss sample and the type 2 mean, divided by the degrees of freedom (periods × samples - 1).

**ALCT calculation:**

In ALCT, MeanLoss and SDLoss are the same as the type 2 mean and standard deviation from the AAL report. StandardError indicates how much the average annual loss estimate might vary if the simulation were rerun with different random numbers, reflecting the simulation error in the estimate. RelativeError, the StandardError as a percentage of the mean, is convenient for assessing simulation error and acceptable levels are typically expressed in percentage terms. StandardError is derived from the ANOVA metrics described below. 

LowerCI and UpperCI represent the absolute lower and upper thresholds for the confidence interval for the AAL estimate, indicating the range of losses within a specified confidence level. A higher confidence level results in a wider confidence interval.

**Variance components:**

VarElementHaz and VarElementVuln arise from attributing variance in the annual loss to hazard effects (variation due to event intensity across years) and vulnerability effects (variation due to sampling from exposure's damage uncertainty distributions). This is done using a one-factor effects model and standard analysis of variance 'ANOVA' on the annual loss samples.

In the one-factor model, annual loss in year i and sample m, denoted L(i,m), is expressed as:

L(i,m) = AAL + h(i) + v(i,m)

where;

* i = 1,2,.. to I years (periods)
* m = 1,2,.. to M samples
* h(i) = mean loss for year i - AAL
* v(i,m) = L(i,m) - mean loss for year i
* I is the total number of years (periods) in the model
* M is the number of samples specified at runtime by the user

Total variance in annual loss is partitioned into independent hazard and vulnerability effects:

Var(L) = Var(h) + Var(v)

ANOVA is used to estimate the variance components Var(h) and Var(v). For standard Oasis models, since the events are fixed across years, the simulation error in the AAL estimate arises only from the vulnerability component.

The StandardError of the AAL estimate in ALCT follows from the calculation of the Variance of the AAL estimate as follows;

Var(AAL estimate) = VarElementVuln = Var(v) / (I * M)

StandardErrorVuln = sqrt(VarElementVuln)

StandardError = StandardErrorVuln

Finally, ALCT provides statistics for multiple increasing sample subsets, showing the convergence of the AAL estimate with increasing sample sizes. These subsets are non-overlapping and fixed, starting with SampleSize=1 (m=1), SampleSize=2 (m=2 to 3), SampleSize=4 (m=4 to 7), up to the maximum subset size. The final row gives statistics for the total samples M, using all available samples.

##### Period weightings

An additional feature of aalcalc is available to vary the relative importance of the period losses by providing a period weightings file to the calculation. In this file, a weight can be assigned to each period make it more or less important than neutral weighting (1 divided by the total number of periods). For example, if the neutral weight for period 1 is 1 in 10000 years, or 0.0001, then doubling the weighting to 0.0002 will mean that period's loss reoccurrence rate would double and the loss contribution to the average annual loss would double.  

All period_nos must appear in the file from 1 to P (no gaps). There is no constraint on the sum of weights. Periods with zero weight will not contribute any losses to the AAL.

This feature will be invoked automatically if the periods.bin file is present in the input directory.

[Return to top](#outputcomponents)

#### Parameters

Optional parameter for aalcalc;

* -c {file path} - Generate the alct report

### kat <a id="kat"></a>
***
In cases where events have been distributed to multiple processes, the output files can be concatenated to standard output.

#### Parameters

Optional parameters are:

* -d {file path} - The directory containing output files to be concatenated.
* -s - Sort by event ID (currently only supported for eltcalc output).

The sort by event ID option assumes that events have not been distributed to processes randomly and the list of event IDs in events.bin is sequential and contiguous. Should either of these conditions be false, the output will still contain all events but sorting cannot be guaranteed.

#### Usage

```
$ kat [parameters] [file]... > [stdout component]
```

#### Examples

```
$ kat -d pltcalc_output/ > pltcalc.csv
$ kat eltcalc_P1 eltcalc_P2 eltcalc_P3 > eltcalc.csv
$ kat -s eltcalc_P1 eltcalc_P2 eltcalc_P3 > eltcalc.csv
$ kat -s -d eltcalc_output/ > eltcalc.csv
```

Files are concatenated in the order in which they are presented on the command line. Should a file path be specified, files are concatenated in alphabetical order. When asked to sort by event ID, the order of input files is irrelevant.

[Return to top](#outputcomponents)

### katparquet <a id="katparquet"></a>
***
The output parquet files from multiple processes can be concatenated to a single parquet file. The results are automatically sorted by event ID. Unlike [kat](#kat), the [ORD table name](ORDOutputComponents.md) for the input files must be specified on the command line.

#### Parameters

* -d {file path} - The directory containing output files to be concatenated.
* -M - Concatenate MPLT files
* -Q - Concatenate QPLT files
* -S - Concatenate SPLT files
* -m - Concatenate MELT files
* -q - Concatenate QELT files
* -s - Concatenate SELT files
* -o {filename} - Output concatenated file

#### Usage

```
$ katparquet [parameters] -o [filename.parquet] [file]...
```

#### Examples

```
$ katparquet -d mplt_files/ -M -o MPLT.parquet
$ katparquet -q -o QPLT.parquet qplt_P1.parquet qplt_P2.parquet qplt_P3.parquet
```

[Return to top](#outputcomponents)

[Go to 4.3 Data conversion components section](DataConversionComponents.md)

[Back to Contents](Contents.md)
