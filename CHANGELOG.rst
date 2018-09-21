CHANGELOG
=========

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

* Fix bug in coverage type matching of canonical items and keys items in the GUL items generator
in the exposure manager

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
.. _`1.0.9`: https://github.com/OasisLMF/OasisLMF/compare/17c691b...master
.. _`1.0.8`: https://github.com/OasisLMF/OasisLMF/compare/8eeaeaf...master
.. _`1.0.6`: https://github.com/OasisLMF/OasisLMF/compare/9578398...master
.. _`1.0.5`: https://github.com/OasisLMF/OasisLMF/compare/c87c782...master
.. _`1.0.1`: https://github.com/OasisLMF/OasisLMF/compare/7de227d...master
