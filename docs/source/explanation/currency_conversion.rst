Currency Conversion
==================


* :ref:`intro_currency`
* :ref:`dict_based_conversion`
* :ref:`fx_currency_rates`
* :ref:`reporting_currency`
* :ref:`mdk`

|
.. _intro_currency:

Introduction:
*************

----

The oasislmf package can handle the conversion of relevant columns of the oed files to a consistent reporting currency. 
The package will reference exchange rates either form a user supplied exchange rate table (dictionary based) , or from an external API (FX).


.. _dict_based_conversion:

File Based Currency Conversion
***************

----

DictBasedCurrencyRates is a solution where all the rate are provided via files and stored internally in the package as a dictionary.

Oasis LMF supports csv file (compressed or not) or a parquet file where they will be read as DataFrame.
The file is referenced using a JSON configuration file with the ``--currency_conversion_json`` flag :

.. code-block:: json

    {
        "currency_conversion_type": "DictBasedCurrencyRates",
        "source_type": "csv",
        "file_path": "tests/inputs/roe.csv"
    }

|

note, you should use ``"source_type": "parquet"`` if parquet file is used

The expected format is (roe being a float in parquet format):

.. csv-table::
   :header: cur_from,cur_to,roe

    USD,GBP,0.85
    USD,EUR,0.95
    GBP,EUR,1.12

|

Rate can also be passed directly in currency_conversion_json with type ``dict``
ex:


.. code-block:: json

    {
        "currency_conversion_type": "DictBasedCurrencyRates",
        "source_type": "list",
        "currency_rates": [
            ["USD", "GBP", 0.85],
            ["USD", "EUR", 0.95],
            ["GBP", "EUR", 1.12]
            ]
    }

|

When looking for a key pair, DictBasedCurrencyRates checks first for the key pair (cur1, cur2) then for (cur2, cur1).
So if a Currency pairs is only specified one way (ex: GBP=>EUR) then it is automatically assume that
roe EUR=>GBP = 1/(roe GPB=>EUR)

if a currency pair is missing ValueError(f"currency pair {(cur_from, cur_to)} is missing") is thrown


.. _fx_currency_rates:

FX Currency Rates
*****************************

----

OasisLMF also lets you use the external package `forex-python <https://forex-python.readthedocs.io/en/latest/usage.html>`_
to perform the conversion. A date may be specified in ISO 8601 format (YYYY-MM-DD)
currency_conversion_json:

.. code-block:: json

    {
        "currency_conversion_type": "FxCurrencyRates",
        "datetime": "2018-10-10"
    }

|


.. _reporting_currency:

Reporting Currency
*****************************

----

The desired reporting currency will also need to be specified when running the oasislmf package.
To do this, the user should enter the currecny code using the ``--reporting-currency`` flag


.. _mdk:

MDK
*****************************

----

To run the currency conversion as part of the MDK then, the user should use a command as follows:

.. code-block:: sh

    oasislmf model run 
        --config oasislmf.json 
        --currency-conversion-json currency_settings.json 
        --reporting-currency GBP

|

Note that this will create and use a copy of the original OED input files with the currency fields converted.
It will also store the original currency and the rate of exchange used in the new OED file for reference.
