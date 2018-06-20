CHANGELOG
=========


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

.. _`1.1.17`: https://github.com/OasisLMF/OasisLMF/compare/de90f11...master
.. _`1.1.15`: https://github.com/OasisLMF/OasisLMF/compare/18b34b9...master
.. _`1.1.14`: https://github.com/OasisLMF/OasisLMF/compare/f3e0ee8...master
.. _`1.1.13`: https://github.com/OasisLMF/OasisLMF/compare/33f96fd...dev
.. _`1.1.12`: https://github.com/OasisLMF/OasisLMF/compare/5045ca2...dev
.. _`1.1.10`: https://github.com/OasisLMF/OasisLMF/compare/a969192...master
.. _`1.0.9`: https://github.com/OasisLMF/OasisLMF/compare/17c691b...master
.. _`1.0.8`: https://github.com/OasisLMF/OasisLMF/compare/8eeaeaf...master
.. _`1.0.6`: https://github.com/OasisLMF/OasisLMF/compare/9578398...master
.. _`1.0.5`: https://github.com/OasisLMF/OasisLMF/compare/c87c782...master
.. _`1.0.1`: https://github.com/OasisLMF/OasisLMF/compare/7de227d...master
