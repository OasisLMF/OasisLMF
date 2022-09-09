OED Currency Support
=======================

# Overview
OasisLMF supports OED files with multiple currency. As the computation itself needs to be in a single currency that we call
repporting currency, a new mini-step is added after the input file are copied in the run directory to convert all terms 
expressed in other currencies. the rate of exchange and the original currency are stored respectively in rateofexchange
and originalcurrency. The rates to use and the reporting currency needs to be provided by the users via a new json
setting file called currency_conversion_json in the json config or --currency-conversion-json directly in the MDK.

OasisLMF will check whether OED file contains multiple currency and throw the exception :
```python
OasisException(f"{file_type} file {oed_path} contains multiple currencies,"
               f"reporting_currency and currency_conversion_json are necessary to perform computation")
```

## Convertion rates
OasisLMF provide several ways to convey the conversion rates either by providing the rates via files, using python-forex
or providing your own module and class.

### DictBasedCurrencyRates
DictBasedCurrencyRates is a solution where all the rate are provided via files and stored internally as a dictionary.

#### Csv and Parquet files
We support csv file (compressed or not) or a parquet file where they will be read as DataFrame.
exemple of currency_conversion_json ("source_type": "parquet" if parquet file is used):
```json
{
    "currency_conversion_type": "DictBasedCurrencyRates",
    "source_type": "csv",
    "file_path": "tests/inputs/roe.csv"
}
```

The expected format is (roe being a float in parquet format):
```
cur_from,cur_to,roe
USD,GBP,0.85
USD,EUR,0.95
GBP,EUR,1.12
```

#### json dict based
Rate can also be passed directly in currency_conversion_json
ex:
```json
{
    "currency_conversion_type": "DictBasedCurrencyRates",
    "source_type": "dict",
    "currency_rates": [["USD", "GBP", 0.85],
                       ["USD", "EUR", 0.95],
                       ["GBP", "EUR", 1.12]
                      ]
}
```
 
#### reversible currency pairs
When looking for a key pair, DictBasedCurrencyRates check 1st for the key pair (cur1, cur2) then for (cur2, cur1).
So if a Currency pairs is only specified one way (ex: GBP=>EUR) then it is automatically assume that 
roe EUR=>GBP = 1/(roe GPB=>EUR)

if a currency pair is missing ValueError(f"currency pair {(cur_from, cur_to)} is missing") is thrown


### FxCurrencyRates
OasisLMF let you use the external package [forex-python](https://forex-python.readthedocs.io/en/latest/usage.html)
to perform the conversion. A date may be specified in ISO 8601 format (YYYY-MM-DD)
currency_conversion_json: 
```json
{
  "currency_conversion_type": "FxCurrencyRates",
  "datetime": "2018-10-10"
}
```

### Custom Currency Module
You can also specify the path to your custom currency rate module.

currency_conversion_json: 
```json
{
  "currency_conversion_type": "custom",
  "module_path": "my_module_path",
  "class_name": "my_class_name",
  "custom_parameters": {"param1": "val1", "param2": 10}
}
```
The class will be instantiated with custom_parameters if present then the method get_rate(cur_from, cur_to) will be
called when a rate is needed. 
