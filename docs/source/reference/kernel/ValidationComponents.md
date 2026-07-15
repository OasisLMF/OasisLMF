![alt text](/_static/images/kernel/banner.jpg "banner")
# 4.6 Validation components <a id="validationcomponents"></a>

The following components run validity checks on csv format files:

**Model data files**
* **[validatedamagebin](#validatedamagebin)** checks damage bin dictionary for validity.
* **[validatefootprint](#validatefootprint)** checks event footprint for validity.
* **[validatevulnerability](#validatevulnerability)** checks vulnerability data for validity.
* **[crossvalidation](#crossvalidation)** performs validation checks across damage bin dictionary, event footprint and vulnerability data.

**Oasis input files**
* **[validateoasisfiles](#validateoasisfiles)** performs validation checks across coverages, items, fm policytc, fm programme and fm profile data.

## Model data files

<a id="validatedamagebin"></a>
### validatedamagebin
The following checks are performed on the damage bin dictionary:

* Each line contains 4 or 5 values.
* First bin index is 1.
* Bin indices are contiguous.
* Interpolation lies inside range.

In addition, warnings are issued in the following cases:

* Lower limit of first bin is not 0.
* Upper limit of last bin is not 1.
* Deprecated `interval_type` column included.
* Interpolation lies within range but not in the bin centre.

The checks can be performed on `damage_bin_dict.csv` from the command line:

```
$ validatedamagebin < damage_bin_dict.csv
```

The checks are also performed by default when converting damage bin dictionary files from csv to binary format:

```
$ damagebintobin < damage_bin_dict.csv > damage_bin_dict.bin

# Suppress vaidation checks with -N argument
$ damagebintobin -N < damage_bin_dict.csv > damage_bin_dict.bin
```

<a id="validatefootprint"></a>
### validatefootprint
The following checks are performed on the event footprint:

* Each line contains 4 values.
* Total probability for each event-areaperil combination is 1.
* Event IDs listed in ascending order.
* For each event ID, areaperils IDs listed in ascending order.
* No duplicate intensity bin IDs for each event-areaperil combination.

Should all checks pass, the maximum value of `intensity_bin_index` is given, which is a required input for `footprinttobin`.

The checks can be performed on `footprint.csv` from the command line:

```
$ validatefootprint < footprint.csv
```

The checks are also performed by default when converting footprint files from csv to binary format:

```
$ footprinttobin -i {number of intensity bins} < footprint.csv

# Suppress validation checks with -N argument
$ footprinttobin -i {number of intensity bins} -N < footprint.csv
```

<a id="validatevulnerability"></a>
### validatevulnerability
The following checks are performed on the vulnerability data:

* Each line contains 4 values.
* Total probability for each vulnerability-intensity bin combination is 1.
* Vulnerability IDs listed in ascending order.
* For each vulnerability ID, all intensity bin IDs are present and listed in ascending order.
* For each vulnerability-intensity bin combination, damage bin IDs are contiguous and start at bin index 1.

Should all checks pass, the maximum value of `damage_bin_id` is given, which is a required input for `vulnerabilitytobin`.

The checks can be performed on `vulnerability.csv` from the command line:

```
$ validatevulnerability < vulnerability.csv
```

The checks are also performed by default when converting vulnerability files from csv to binary format:

```
$ vulnerabilitytobin -d {number of damage bins} < vulnerability.csv > vulnerability.bin

# Suppress validation checks with -N argument
$ vulnerabilitytobin -d {number of damage bins} -N < vulnerability.csv > vulnerability.bin
```

<a id="crossvalidation"></a>
### crossvalidation
The following checks are performed across the damage bin dictionary, event footprint and vulnerability data:

* Damage bin IDs in the vulnerabilty data are subset of those in the damage bin dictionary.
* Intensity bin IDs in the event footprint are subset of those in the vulnerability data.

The checks can be performed on `damage_bin_dict.csv`, `footprint.csv` and `vulnerability.csv` from the command line:

```
$ crossvalidation -d damage_bin_dict.csv -f footprint.csv -s vulnerability.csv
```

## Input oasis files

<a id="validateoasisfiles"></a>
### validateoasisfiles
The following checks are performed across the coverages, items, fm policytc, fm programme and fm profile data:

* 1-to-1 relationship between `agg_id` in `fm_programme.csv` and `item_id` in `items.csv` when `level_id = 1`.
* `coverage_id` in `items.csv` matches those in `coverages.csv`.
* `policytc_id` in `fm_policytc.csv` matches those in `fm_profile.csv`.
* (`level_id`, `agg_id`) pairs in `fm_policytc.csv` are present as (`level_id`, `to_agg_id`) pairs in `fm_programme.csv`.
* When `level_id = n > 1`, `from_agg_id` corresponds to a `to_agg_id` from `level_id = n - 1`.

The checks can be performed on `coverages.csv`, `items.csv`, `fm_policytc.csv`, `fm_programme.csv` and `fm_profile.csv` from the command line, specifying the directory these files are located in:

```
$ validateoasisfiles -d path/to/output/directory
```

The Ground Up Losses (GUL) flag `g` can be specified to only perform checks on `items.csv` and `coverages.csv`:

```
$ validateoasisfiles -g -d /path/to/output/directory
```

[Return to top](#validationcomponents)

[Go to 5. Financial Module](../../explanation/financial-module.rst)

[Back to Contents](Contents.md)
