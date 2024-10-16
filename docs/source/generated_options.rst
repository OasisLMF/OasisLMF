verbose
=======

Description: Use verbose logging.

Expected type: string

Default value: ``False``

config
======

Description: MDK config. JSON file

Expected type: path

Default value: ``None``

oasis_files_dir
===============

Description: Path to the directory in which to generate the Oasis files

Expected type: path

Default value: ``None``

exposure_pre_analysis_module
============================

Description: Exposure Pre-Analysis lookup module path

Expected type: path

Default value: ``None``

post_analysis_module
====================

Description: Post-Analysis module path

Expected type: path

Default value: ``None``

pre_loss_module
===============

Description: pre-loss hook module path

Expected type: path

Default value: ``None``

post_file_gen_module
====================

Description: post-file gen hook module path

Expected type: path

Default value: ``None``

check_oed
=========

Description: if True check input oed files

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``True``

analysis_settings_json
======================

Description: Analysis settings JSON file path

Expected type: path

Default value: ``None``

model_storage_json
==================

Description: Model data storage settings JSON file path

Expected type: path

Default value: ``None``

model_settings_json
===================

Description: Model settings JSON file path

Expected type: path

Default value: ``None``

user_data_dir
=============

Description: Directory containing additional model data files which varies between analysis runs

Expected type: path

Default value: ``None``

model_data_dir
==============

Description: Model data directory path

Expected type: path

Default value: ``None``

copy_model_data
===============

Description: Copy model data instead of creating symbolic links to it.

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

model_run_dir
=============

Description: Model run directory path

Expected type: path

Default value: ``None``

model_package_dir
=================

Description: Path containing model specific package

Expected type: path

Default value: ``None``

ktools_legacy_stream
====================

Description: Run Ground up losses using the older stream type (Compatibility option)

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

fmpy
====

Description: use fmcalc python version instead of c++ version

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``True``

ktools_alloc_rule_il
====================

Description: Set the fmcalc allocation rule used in direct insured loss

Expected type: integer

Default value: ``2``

ktools_alloc_rule_ri
====================

Description: Set the fmcalc allocation rule used in reinsurance

Expected type: integer

Default value: ``3``

summarypy
=========

Description: use summarycalc python version instead of c++ version

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

check_missing_inputs
====================

Description: Fail an analysis run if IL/RI is requested without the required generated files.

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

ktools_num_processes
====================

Description: Number of ktools calculation processes to use

Expected type: integer

Default value: ``-1``

ktools_event_shuffle
====================

Description: Set rule for event shuffling between eve partions, 0 - No shuffle, 1 - round robin (output elts sorted), 2 - Fisher-Yates shuffle, 3 - std::shuffle (previous default in oasislmf<1.14.0) 

Expected type: integer

Default value: ``1``

ktools_alloc_rule_gul
=====================

Description: Set the allocation used in gulcalc

Expected type: integer

Default value: ``0``

ktools_num_gul_per_lb
=====================

Description: Number of gul per load balancer (0 means no load balancer)

Expected type: integer

Default value: ``0``

ktools_num_fm_per_lb
====================

Description: Number of fm per load balancer (0 means no load balancer)

Expected type: integer

Default value: ``0``

ktools_disable_guard
====================

Description: Disables error handling in the ktools run script (abort on non-zero exitcode or output on stderr)

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

ktools_fifo_relative
====================

Description: Create ktools fifo queues under the ./fifo dir

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

modelpy
=======

Description: use getmodel python version instead of c++ version

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

gulpy
=====

Description: use gulcalc python version instead of c++ version

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

gulpy_random_generator
======================

Description: set the random number generator in gulpy (0: Mersenne-Twister, 1: Latin Hypercube. Default: 1).

Expected type: integer

Default value: ``1``

gulmc
=====

Description: use full Monte Carlo gulcalc python version

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``True``

gulmc_random_generator
======================

Description: set the random number generator in gulmc (0: Mersenne-Twister, 1: Latin Hypercube. Default: 1).

Expected type: integer

Default value: ``1``

gulmc_effective_damageability
=============================

Description: use the effective damageability to draw loss samples instead of the full Monte Carlo method. Default: False

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

gulmc_vuln_cache_size
=====================

Description: Size in MB of the cache for the vulnerability calculations. Default: 200

Expected type: integer

Default value: ``200``

fmpy_low_memory
===============

Description: use memory map instead of RAM to store loss array (may decrease performance but reduce RAM usage drastically)

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

fmpy_sort_output
================

Description: order fmpy output by item_id

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``True``

model_custom_gulcalc
====================

Description: Custom gulcalc binary name to call in the model losses step

Expected type: string

Default value: ``None``

model_py_server
===============

Description: running the data server for modelpy

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

peril_filter
============

Description: Peril specific run

Expected type: string

Default value: ``[]``

model_custom_gulcalc_log_start
==============================

Description: Log message produced when custom gulcalc binary process starts

Expected type: string

Default value: ``None``

model_custom_gulcalc_log_finish
===============================

Description: Log message produced when custom gulcalc binary process ends

Expected type: string

Default value: ``None``

base_df_engine
==============

Description: The engine to use when loading dataframes

Expected type: string

Default value: ``oasis_data_manager.df_reader.reader.OasisPandasReader``

model_df_engine
===============

Description: The engine to use when loading model data dataframes (default: --base-df-engine if not set)

Expected type: string

Default value: ``None``

exposure_df_engine
==================

Description: The engine to use when loading exposure data dataframes (default: --base-df-engine if not set)

Expected type: string

Default value: ``None``

post_file_gen_class_name
========================

Description: Name of the class to use for the pre loss calculation

Expected type: string

Default value: ``PostFileGen``

post_file_gen_setting_json
==========================

Description: post file generation config JSON file path

Expected type: path

Default value: ``None``

oed_schema_info
===============

Description: path to custom oed_schema

Expected type: path

Default value: ``None``

oed_location_csv
================

Description: Source location CSV file path

Expected type: path

Default value: ``None``

oed_accounts_csv
================

Description: Source accounts CSV file path

Expected type: path

Default value: ``None``

oed_info_csv
============

Description: Reinsurance info. CSV file path

Expected type: path

Default value: ``None``

oed_scope_csv
=============

Description: Reinsurance scope CSV file path

Expected type: path

Default value: ``None``

location
========

Description: A set of locations to include in the files

Expected type: <class 'str'>

Default value: ``None``

portfolio
=========

Description: A set of portfolios to include in the files

Expected type: <class 'str'>

Default value: ``None``

account
=======

Description: A set of locations to include in the files

Expected type: <class 'str'>

Default value: ``None``

pre_loss_class_name
===================

Description: Name of the class to use for the pre loss calculation

Expected type: string

Default value: ``PreLoss``

pre_loss_setting_json
=====================

Description: pre loss calculation config JSON file path

Expected type: path

Default value: ``None``

keys_data_csv
=============

Description: Pre-generated keys CSV file path

Expected type: path

Default value: ``None``

keys_errors_csv
===============

Description: Pre-generated keys errors CSV file path

Expected type: path

Default value: ``None``

profile_loc_json
================

Description: Source (OED) exposure profile JSON path

Expected type: path

Default value: ``None``

profile_acc_json
================

Description: Source (OED) accounts profile JSON path

Expected type: path

Default value: ``None``

profile_fm_agg_json
===================

Description: FM (OED) aggregation profile path

Expected type: path

Default value: ``None``

currency_conversion_json
========================

Description: settings to perform currency conversion of oed files

Expected type: path

Default value: ``None``

reporting_currency
==================

Description: currency to use in the results reported

Expected type: string

Default value: ``None``

disable_summarise_exposure
==========================

Description: Disables creation of an exposure summary report

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

damage_group_id_cols
====================

Description: Columns from loc file to set group_id

Expected type: string

Default value: ``['PortNumber', 'AccNumber', 'LocNumber']``

hazard_group_id_cols
====================

Description: Columns from loc file to set hazard_group_id

Expected type: string

Default value: ``['PortNumber', 'AccNumber', 'LocNumber']``

lookup_multiprocessing
======================

Description: Flag to enable/disable lookup multiprocessing

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

do_disaggregation
=================

Description: if True run the oasis disaggregation.

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``True``

keys_format
===========

Description: Keys files output format

Expected type: string

Default value: ``oasis``

lookup_config_json
==================

Description: Lookup config JSON file path

Expected type: path

Default value: ``None``

lookup_data_dir
===============

Description: Model lookup/keys data directory path

Expected type: path

Default value: ``None``

lookup_module_path
==================

Description: Model lookup module path

Expected type: path

Default value: ``None``

lookup_complex_config_json
==========================

Description: Complex lookup config JSON file path

Expected type: path

Default value: ``None``

lookup_num_processes
====================

Description: Number of workers in multiprocess pools

Expected type: integer

Default value: ``-1``

lookup_num_chunks
=================

Description: Number of chunks to split the location file into for multiprocessing

Expected type: integer

Default value: ``-1``

model_version_csv
=================

Description: Model version CSV file path

Expected type: path

Default value: ``None``

disable_oed_version_update
==========================

Description: Flag to enable/disable conversion to latest compatible OED version. Must be present in model settings.

Expected type: boolean (yes/no, true/false t/f, y/n, or 1/0)

Default value: ``False``

exposure_pre_analysis_class_name
================================

Description: Name of the class to use for the exposure_pre_analysis

Expected type: string

Default value: ``ExposurePreAnalysis``

exposure_pre_analysis_setting_json
==================================

Description: Exposure Pre-Analysis config JSON file path

Expected type: path

Default value: ``None``

post_analysis_class_name
========================

Description: Name of the class to use for the post_analysis

Expected type: string

Default value: ``PostAnalysis``

lookup_config
=============

Description: 

Expected type: string

Default value: ``None``

lookup_complex_config
=====================

Description: 

Expected type: string

Default value: ``None``

write_ri_tree
=============

Description: 

Expected type: string

Default value: ``False``

write_chunksize
===============

Description: 

Expected type: integer

Default value: ``200000``

oasis_files_prefixes
====================

Description: 

Expected type: string

Default value: ``OrderedDict({'gul': {'complex_items': 'complex_items', 'items': 'items', 'coverages': 'coverages', 'amplifications': 'amplifications', 'sections': 'sections'}, 'il': {'fm_policytc': 'fm_policytc', 'fm_profile': 'fm_profile', 'fm_programme': 'fm_programme', 'fm_xref': 'fm_xref'}})``

src_dir
=======

Description: 

Expected type: path

Default value: ``None``

run_dir
=======

Description: 

Expected type: path

Default value: ``None``

output_file
===========

Description: 

Expected type: path

Default value: ``None``

loss_factor
===========

Description: 

Expected type: <class 'float'>

Default value: ``[1.0]``

output_level
============

Description: Keys files output format

Expected type: string

Default value: ``item``

extra_summary_cols
==================

Description: extra column to include in the summary

Expected type: string

Default value: ``[]``

coverage_types
==============

Description: Select List of supported coverage_types [1, .. ,4]

Expected type: integer

Default value: ``[1, 2, 3, 4]``

model_perils_covered
====================

Description: List of peril covered by the model

Expected type: string

Default value: ``['AA1']``

stream_type
===========

Description: Set the IL input stream type, 2 = default loss stream, 1 = deprecated cov/item stream

Expected type: integer

Default value: ``2``

net_ri
======

Description: 

Expected type: string

Default value: ``True``

include_loss_factor
===================

Description: 

Expected type: string

Default value: ``True``

print_summary
=============

Description: 

Expected type: string

Default value: ``True``

server_login_json
=================

Description: Source location CSV file path

Expected type: path

Default value: ``None``

server_url
==========

Description: URL to Oasis Platform server, default is localhost

Expected type: string

Default value: ``http://localhost:8000``

server_version
==============

Description: Version prefix for OasisPlatform server, 'v1' = single server run, 'v2' = distributed on cluster

Expected type: string

Default value: ``v2``

model_id
========

Description: API `id` of a model to run an analysis with

Expected type: integer

Default value: ``None``

portfolio_id
============

Description: API `id` of a portfolio to run an analysis with

Expected type: integer

Default value: ``None``

analysis_id
===========

Description: API `id` of an analysis to run

Expected type: integer

Default value: ``None``

output_dir
==========

Description: Output data directory for results data (absolute or relative file path)

Expected type: path

Default value: ``./``
