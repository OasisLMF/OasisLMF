OasisLMF Changelog
==================

`1.9.1`_
--------
.. start_latest_release
* [#630](https://github.com/OasisLMF/OasisLMF/issues/630) - full_correlation gulcalc option creates large output files.
.. end_latest_release


`1.9.0`_
--------
* [#566](https://github.com/OasisLMF/OasisLMF/issues/566) - Handle unlimited LayerLimit without large default value
* [#574](https://github.com/OasisLMF/OasisLMF/issues/574) - Use LayerNumber to identify unique policy layers in gross fm file generation 
* [#578](https://github.com/OasisLMF/OasisLMF/issues/578) - Added missing combination of terms in calcrules
* [#603](https://github.com/OasisLMF/OasisLMF/issues/603) - Add type 2 financial terms tests for multi-peril to regression test
* [PR 600](https://github.com/OasisLMF/OasisLMF/pull/600) - Added Scripts for generated example model data for testing.

`1.8.3`_
--------
* [#601](https://github.com/OasisLMF/OasisLMF/issues/601) - Fix calculations for type 2 deductibles and limits in multi-peril models

`1.8.2`_
--------
* [#599](https://github.com/OasisLMF/OasisLMF/issues/599) - Allow setting 'loc_id' externally
* [#594](https://github.com/OasisLMF/OasisLMF/issues/594) - Pass copy of location df to custom lookup to avoid side effects
* [#593](https://github.com/OasisLMF/OasisLMF/issues/593) - Fail fast on analysis settings formatting problem
* [#591](https://github.com/OasisLMF/OasisLMF/issues/591) - Update pinned pandas package
* [#588](https://github.com/OasisLMF/OasisLMF/issues/588) - AreaCode is defined as "string" in OED, but loaded as a number in the DF
* [#596](https://github.com/OasisLMF/OasisLMF/issues/596) - Incorrect number of locations in Overview
* [3dce18f](https://github.com/OasisLMF/OasisLMF/pull/595/commits/3dce18f5872c2855f29548845212bdde8813f472) - Relax required fields in analysis_settings validation
* [bd052a6](https://github.com/OasisLMF/OasisLMF/pull/595/commits/bd052a641b53db5284fb9609b43d6080df77711c) - Fix issue in gen bash, only enable an output type if summary section exists

`1.8.1`_
--------
* [#589](https://github.com/OasisLMF/OasisLMF/issues/589) - Schema fix to allow for 0 samples
* [#583](https://github.com/OasisLMF/OasisLMF/pull/583) - Reduce memory use in gul_inputs creation (DanielFEvans)
* [#582](https://github.com/OasisLMF/OasisLMF/issues/582) -  Check for calc_type in all sections

`1.8.0`_
--------
* [#579](https://github.com/OasisLMF/OasisLMF/issues/579) - Install complex_itemstobin and complex_itemstocsv
* [#565](https://github.com/OasisLMF/OasisLMF/issues/565) - Non-unicode CSV data is not handled neatly
* [#570](https://github.com/OasisLMF/OasisLMF/issues/570) - Issue with item_id to from_agg_id mapping at level 1
* [#556](https://github.com/OasisLMF/OasisLMF/issues/556) - review calc_rules.csv mapping for duplicates and logical gaps
* [#549](https://github.com/OasisLMF/OasisLMF/issues/549) - Add FM Tests May 2020
* [#555](https://github.com/OasisLMF/OasisLMF/issues/555) - Add JSON schema validation on CLI
* [#577](https://github.com/OasisLMF/OasisLMF/issues/577) - Add api client progressbars for OasisAtScale


`1.7.1`_
--------
* #553 - Fix alc rule mapping error
* #550 - Fix enbash fmcalc and full_correlation
* #548 - Fix UL Alloc Rule 0
* Fix - Assign loc_id by sorted values of (LocNum, AccNum, PortNum) - location file resistant to reordering rows
* Fix - default `run_dir` in `oasislmf model generate-exposure-pre-analysis` cmd

`1.7.0`_
--------
* #497 - Add exception wrapping to OasisException
* #528 - FM validation tests with % damage range
* #531 - item file ordering of item_id
* #533 - Added new FM acceptance tests
* Added - Pre-analysis exposure modification (CLI interface)
* Added - revamped CLI Structure

`1.6.0`_
--------
* Fix #513 - Breaking change in msgpack 1.0.0
* Fix #503 - Change areaperil id datatype to unit64
* Fix #481 - Corrections to fm_profile for type 2 terms
* Fix #512 - Issue in generate rtree index CLI
* Fix #516 - Refactored the `upload_settings` method in API client
* Fix #514 - fix ; issues in LocPerilsCovered
* Fix #515 - Store the `loc_id` of failed location rows
* Added #508 - fm12 acceptance test
* Added #480 - Extend calcrules to cover more combinations of financial terms
* Added #523 - Long description field to `model_settings.json` schema
* Added #524 - Total TIV sums in exposure report
* Added #527 - Group OED fields from model settings
* Added #506 - Improve performance in `write_exposure_summary()`

`1.5.1`_
--------
* Fix - Issue in IL file generation, `fm_programme` file missing agg_id rows

`1.5.0`_
--------

* Added step policy support
* Added #453 - Allow user to select group_id based on columns in loc file
* Added #474 - Option to set gulcalc command - raises an error if not in path
* Update to Model Settings schema, #478 #484 #485
* Update #464 - API client with new Queued Job states
* Update #470 - Model_settings.json schema
* Fix #491 -  in `oasislmf exposure run` command
* Fix #477 - setup.py fails when behind a proxy
* Fix #482 - symlink error when rerunning analysis using existing analysis_folder
* Fix #460 - CLI, remove use of lowercase '-c'
* Fix #493 - generate_model_losses fix for spaces in filepath
* Fix #475 - Prevent copy of model_data directory on OSError
* Fix #486 - Run error using `pandas==1.0.0`
* Fix #459 - File generation issue in fm_programme
* Fix #456 - Remove calls to `get_ids` in favour of pandas groupby
* Fix #492 - ComplexModel error guard run in subshell
* Fix #451 - ComplexModel error guard in bash script
* Fix #415 - RI String matching issue
* Fix #462 - Issue in fm_programmes file generation
* Fix #463 - Incorrect limits in FM file generation
* Fix #468 - Fifo issue in Bash script generation


`1.4.6`_
--------
* Update to model_settings schema
* Fixes #452 - Check columns before grouping output
* Fixes #451 - Error checking ktools runs for complex models

`1.4.5`_
--------
* Fix for fm_programme mapping
* Fix for IL files generation
* Fix issue #439 - il summary groups
* Reduce memory use in GUL inputs generation (#440)
* Fix for api client - handle rotating refresh token
* Feature/setting schemas (#438)
* Update API client - add settings JSON endpoints (#444)
* Add fully correlated option to MDK (#446)
* Add dtype conversion and check for valid OED peril codes (#448)

`1.4.4`_
--------
* Hotfix - Added the run flag `--ktools-disable-guard` option for complex models & custom binaries

`1.4.3`_
--------
* Added support for compressed file extensions
* Fix docker kill error
* Fix in IL inputs
* Fix for multiprocessing lookup
* Fix for summary info data types
* Set IL alloc rule default to 3
* Various fixes for CLI
* Various fixes for ktools scripts

`1.4.2`_
--------
* Added Multi-process keys lookup
* Updated API client
* Added Verifying model files command
* Updated command line flags with backwards compatibility

`1.4.1`_
--------
* Added bash autocomplete #386
* Fix for exposure data types on lookup #387
* Fix for non-OED fields in summary levels #377
* Fix in Reinsurance Layer Logic #381
* Refactor deterministic loss generation #371
* Added bdist package for OSX #372
* Added Allocation rule for Ground up loss #376

`1.4.0`_
--------
* Cookiecutter CLI integration - commands for creating simple and complex Oasis model projects/repositories from project templates
* Extend calc. rules and FM test coverage
* Various fixes in FM and data utils
* Various fixes and updates for the API client module
* Add ktools static binary bdist_wheel to published package
* Fix for Layer_id in file generation
* Performance improvment and fixes for the exposure summary reporting
* Added optional `--summarise-exposure` flag for exposure report output
* Added `exposure_summary_levels.json` file to inputs directory, lists valid OED columns to build summary groups
* Added summary info files to output directory `gul_S1_summary-info.csv` which lists data for grouping summary_ids
* Ktools updated to v3.1.0

`1.3.10`_
---------
* Hotfix release - fix for models using custom lookups

`1.3.9`_
--------
* Updated validation and fixes for FM file generation
* Fixes for exposure-summary reporting
* Fixes for FM calc rule selection

`1.3.8`_
--------
* Add FM support for processing types and codes for deductibles and limits
* Improvements for complex model support and logging
* Update to summary sets for grouping results
* Exposure reporting added
* Fixes for Oasis files generation
* Updates to RI and Acceptance testing
* new sub-command `oasislmf exposure ..` for running and validating deterministic models

`1.3.7`_
--------
* Hotfix - ktools-num-processes not read as int from CLI

`1.3.6`_
--------
* Hotfix - Custom lookup issue in manager

`1.3.5`_
--------
* Issue found in ktools `3.0.7` hotfix downgrade to `3.0.6`

`1.3.4`_
--------
* Optimise FM/IL component (IL input items + input files generation)
* Optimise Oasis files generation (GUL + IL input items + input files generation)
* Upgrade data-related utilities
* Update API client
* Fixes for windows compatibility
* Support for Python 2.7 Ends

`1.3.3`_
--------
* Hotfix for GUL files generation
* Hotfix for lookup index generation
* Hotfix for ktools bash script

`1.3.2`_
--------
* Hotfix fix for analysis_settings custom model worker
* Hotfix tweak for deterministic RI loss calculation

`1.3.1`_
--------
* Hotfix for path issue with the analysis_setttings file
* Downgraded ktools from `3.0.6` to `3.0.5` fix pending in fmcalc

`1.3.0`_
--------
* Remove CSV file transformations from Oasis files generation - use OED source exposure directly
* Integrate backend RI functionality with the CLI model subcommands - full RI losses can now be generated
* Add new CLI subcommand for deterministic loss generation including direct and RI losses
* Optimise FM component (13x speedup achieved)
* Add support for custom complex models, python version of ground up losses `gulcalc.py`

`1.2.8`_
--------
* Hotfix for Ktools, version is now 3.0.5
* Hotfix for API Client Upload timeout issue

`1.2.7`_
--------
* Hotfix in Generate-Losses command

`1.2.6`_
--------
* Added Reinsurance to CLI
* Added Ktools run options to CLI
* Fix for Ktools Memory limits in Genbash

`1.2.5`_
--------
* Fix for setting Alloc Rule in genbash

`1.2.4`_
--------
* Fix for Windows 10 (Linux Sub-system), FIFO queues moved into `/tmp/<random>`
* Fix for Reinsurance, Set RiskLevel = `SEL` as default when value is not set
* Fix, calc rule for all positivedeductibles
* Fixes for new API Client
* Added Deterministic loss generation
* Added FM acceptance tests
* Added Automated testing

`1.2.3`_
--------
* Hotfix for Reinsurance required fields
* Dockerfile and run script for unittests

`1.2.2`_
--------
* Added API client for OED API update
* New MDK commands to run the updated API client
* Improved FM file generation testing
* Fixes to scope filters to correctly handle account, policy and location combinations.
* Added portfolio and location group scope filters.
* Fixes to required fields and default values to match OED
* Fixed binary file writing bug, corrupted tar output files


`1.2.1`_
--------

* Compatibility fix for new API worker
* Fix for Parsing config.json on MDK command line
* Fix for Reinsurance
* Add Reinsurance tests
* Fix GUL item group IDs to index item loc. IDs

`1.2.0`_
--------

* Update concurrency utils - replace multiprocessing.Pool by billiard.Pool in multiprocessing wrapper (oasislmf.utils.concurrency.multiprocess) to fix a problem with Celery tasks unable to run applications which use processes or process pools created using the built-in multiprocessing package (https://github.com/celery/celery/issues/1709)
* Add IL/FM support
* Various optimisations, including to GUL items generation

`1.1.27`_ (beta)
----------------

* Fix for installing ktools on mac OSX (3.0.1)
* Fix for Reinsurance input file validation
* Update Subcommand `oasislmf model generate-oasis-file` to use optional xml validation
* Update for unittest stability on CI/CD

`1.1.26`_ (beta)
----------------

* Merge in reinsurance update from feature/reinsurance
* Fix ktools install using pip instal editable mode `pip install -e ..`

`1.1.25`_ (beta)
----------------

* Fix install issue with utils/keys_data.py - file removed as its no longer used.

`1.1.24`_ (beta)
----------------

* Fix ordering of bulk lookup generation in base generic lookup - records should be generated as (loc. ID, peril ID, coverage type ID) combinations.

`1.1.23`_ (beta)
----------------

* Performace update for exposure transforms `transform-source-to-canonical` and `transform-canonical-to-model`.
* Validation of transform is now optional `--xsd-validation-file-path`, if no value is given this step is skipped.

`1.1.22`_ (beta)
----------------

* Fix bug in coverage type matching of canonical items and keys items in the GUL items generator in the exposure manager

`1.1.21`_ (beta)
----------------

* Enable lookup framework and exposure manager to support multi-peril and multi-coverage type models

`1.1.20`_ (beta)
----------------

* Refactor lookup factory to be compatible with new lookup framework
* Various enhancements to the peril areas index class, file index generation command and peril utils
* Fix for installing pip package without building ktools if binaries exist in system path.

`1.1.19`_ (beta)
----------------

* Fix string lowercasing of lookup config values in new lookup classes
* Fix object pickling to account for Python major version when creating Rtree file index from areas file
* Various fixes to arg parsing and logging in Rtree file index model subcommand class

`1.1.18`_ (beta)
----------------

* Upgrade peril utils, including a custom Rtree index class for peril areas
* Implement MDK model subcommand for writing Rtree file indexes for peril areas from peril area (area peril) files
* Various fixes to the new lookup class framework


`1.1.17`_ (beta)
----------------

* Fix list sorting in exposure manager to use explicit sort key

`1.1.15`_ (beta)
----------------

* Add new lookup class framework in `keys` subpackage

`1.1.14`_ (beta)
----------------

* Add MDK model subcommands for performing source -> canonical and canonical -> model file transformations
* Python 3 compatibility fixes to replace map and filter statements everywhere by list comprehensions

`1.1.13`_ (beta)
----------------

* Add performance improvement for exposure transforms
* Limit exposure validation messages to log level `DEBUG`

`1.1.12`_ (beta)
----------------

* Add concurrency utils (threading + multiprocessing) to `utils` sub. pkg.

`1.1.11`_ (beta)
----------------

* Hotfix for get_analysis_status - fixes issue in client api

`1.1.10`_ (beta)
----------------

* Hotfix for utils INI file loading method - fix parsing of IP
  strings

`1.0.9`_ (beta)
---------------

* Hotfix for JSON keys file writer in keys lookup factory - convert
  JSON string to Unicode literal before writing to file

`1.0.8`_ (beta)
---------------

* Enable custom model execution parameters when running models

`1.0.6`_ (beta)
---------------

* Remove timestamped Oasis files from Oasis files generation pipeline

`1.0.5`_ (beta)
---------------

* Add keys error file generation method to keys lookup factory and make
  exposures manager generate keys error files by default

`1.0.1`_ (beta)
---------------

* Add console logging

.. _`1.9.1`:  https://github.com/OasisLMF/OasisLMF/compare/1.9.0...1.9.1
.. _`1.9.0`:  https://github.com/OasisLMF/OasisLMF/compare/1.8.3...1.9.0
.. _`1.8.3`:  https://github.com/OasisLMF/OasisLMF/compare/1.8.2...1.8.3
.. _`1.8.2`:  https://github.com/OasisLMF/OasisLMF/compare/1.8.1...1.8.2
.. _`1.8.1`:  https://github.com/OasisLMF/OasisLMF/compare/1.8.0...1.8.1
.. _`1.8.0`:  https://github.com/OasisLMF/OasisLMF/compare/1.7.1...1.8.0
.. _`1.7.1`:  https://github.com/OasisLMF/OasisLMF/compare/1.7.0...1.7.1
.. _`1.7.0`:  https://github.com/OasisLMF/OasisLMF/compare/1.6.0...1.7.0
.. _`1.6.0`:  https://github.com/OasisLMF/OasisLMF/compare/1.5.1...1.6.0
.. _`1.5.1`:  https://github.com/OasisLMF/OasisLMF/compare/1.5.0...1.5.1
.. _`1.5.0`:  https://github.com/OasisLMF/OasisLMF/compare/1.4.6...1.5.0
.. _`1.4.6`:  https://github.com/OasisLMF/OasisLMF/compare/1.4.5...1.4.6
.. _`1.4.5`:  https://github.com/OasisLMF/OasisLMF/compare/1.4.4...1.4.5
.. _`1.4.4`:  https://github.com/OasisLMF/OasisLMF/compare/1.4.3...1.4.4
.. _`1.4.3`:  https://github.com/OasisLMF/OasisLMF/compare/1.4.2...1.4.3
.. _`1.4.2`:  https://github.com/OasisLMF/OasisLMF/compare/1.4.1...1.4.2
.. _`1.4.1`:  https://github.com/OasisLMF/OasisLMF/compare/1.4.0...1.4.1
.. _`1.4.0`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.10...1.4.0
.. _`1.3.10`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.9...1.3.10
.. _`1.3.9`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.8...1.3.9
.. _`1.3.8`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.7...1.3.8
.. _`1.3.7`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.6...1.3.7
.. _`1.3.6`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.5...1.3.6
.. _`1.3.5`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.4...1.3.5
.. _`1.3.4`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.3...1.3.4
.. _`1.3.3`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.2...1.3.3
.. _`1.3.2`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.1...1.3.2
.. _`1.3.1`:  https://github.com/OasisLMF/OasisLMF/compare/1.3.0...1.3.1
.. _`1.3.0`:  https://github.com/OasisLMF/OasisLMF/compare/1.2.8...1.3.0
.. _`1.2.8`:  https://github.com/OasisLMF/OasisLMF/compare/1.2.7...1.2.8
.. _`1.2.7`:  https://github.com/OasisLMF/OasisLMF/compare/1.2.6...1.2.7
.. _`1.2.6`:  https://github.com/OasisLMF/OasisLMF/compare/1.2.5...1.2.6
.. _`1.2.5`:  https://github.com/OasisLMF/OasisLMF/compare/1.2.4...1.2.5
.. _`1.2.4`:  https://github.com/OasisLMF/OasisLMF/compare/1.2.3...1.2.4
.. _`1.2.3`:  https://github.com/OasisLMF/OasisLMF/compare/1.2.2...1.2.3
.. _`1.2.2`:  https://github.com/OasisLMF/OasisLMF/compare/d6dbf25...master
.. _`1.2.1`:  https://github.com/OasisLMF/OasisLMF/compare/f4d7390...master
.. _`1.2.0`:  https://github.com/OasisLMF/OasisLMF/compare/ad91e2a...master
.. _`1.1.27`: https://github.com/OasisLMF/OasisLMF/compare/ac4375e...master
.. _`1.1.26`: https://github.com/OasisLMF/OasisLMF/compare/dac703e...master
.. _`1.1.25`: https://github.com/OasisLMF/OasisLMF/compare/3a4b983...master
.. _`1.1.24`: https://github.com/OasisLMF/OasisLMF/compare/8f94cab...master
.. _`1.1.23`: https://github.com/OasisLMF/OasisLMF/compare/0577497...master
.. _`1.1.22`: https://github.com/OasisLMF/OasisLMF/compare/bfeee86...master
.. _`1.1.21`: https://github.com/OasisLMF/OasisLMF/compare/c04dc73...master
.. _`1.1.20`: https://github.com/OasisLMF/OasisLMF/compare/fd31879...master
.. _`1.1.19`: https://github.com/OasisLMF/OasisLMF/compare/5421b91...master
.. _`1.1.18`: https://github.com/OasisLMF/OasisLMF/compare/da8fcba...master
.. _`1.1.17`: https://github.com/OasisLMF/OasisLMF/compare/de90f11...master
.. _`1.1.15`: https://github.com/OasisLMF/OasisLMF/compare/18b34b9...master
.. _`1.1.14`: https://github.com/OasisLMF/OasisLMF/compare/f3e0ee8...master
.. _`1.1.13`: https://github.com/OasisLMF/OasisLMF/compare/33f96fd...master
.. _`1.1.12`: https://github.com/OasisLMF/OasisLMF/compare/5045ca2...master
.. _`1.1.10`: https://github.com/OasisLMF/OasisLMF/compare/a969192...master
.. _`1.0.9`:  https://github.com/OasisLMF/OasisLMF/compare/17c691b...master
.. _`1.0.8`:  https://github.com/OasisLMF/OasisLMF/compare/8eeaeaf...master
.. _`1.0.6`:  https://github.com/OasisLMF/OasisLMF/compare/9578398...master
.. _`1.0.5`:  https://github.com/OasisLMF/OasisLMF/compare/c87c782...master
.. _`1.0.1`:  https://github.com/OasisLMF/OasisLMF/compare/7de227d...master
