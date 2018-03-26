CHANGELOG
=========

`1.0.4`_ (beta)
---------------

* Add keys error file generation method to keys lookup factory and make
  exposures manager generate keys error files by default

`1.0.1`_ (beta)
---------------

* Add console logging

`1.0.0`_ (beta)
---------------

* Update Oasis files generation methods in exposures manager - explicitly set
  integer column type in pandas dataframes for ``item_id``,
  ``coverage_id``, ``areaperil_id``, ``vulnerability_id``, ``group_id``,
  ``summary_id``, ``summaryset_id`` fields

`0.0.9`_ (beta)
---------------

* Tweak ``read_csv`` method in CSV utils module (``oasislmf/utils/csv.py``) -
  replace ``pandas.DataFrame.loc`` by ``pandas.DataFrame.iloc`` when looping
  over dataframe rows

`0.0.8`_ (beta)
---------------

* Update CSV utils module (``oasislmf/utils/csv.py``) - refactor ``get_csv_rows_as_dicts``
  method to use ``pandas`` instead of ``csv``, and rename to ``read_csv``, and update
  CSV utils tests

.. _`1.0.4`: https://github.com/OasisLMF/OasisLMF/compare/c87c782...master
.. _`1.0.1`: https://github.com/OasisLMF/OasisLMF/compare/7de227d...master^^
.. _`1.0.0`: https://github.com/OasisLMF/OasisLMF/commit/d632528dffcc79098d350402d91738afed676c9c
.. _`0.0.9`: https://github.com/OasisLMF/OasisLMF/commit/de56ffface46ee672e5f0e96c86a77ff7df67dcf
.. _`0.0.8`: https://github.com/OasisLMF/OasisLMF/commit/1b8398a2029dac678cf6708eae04f9c80b9db531
