import datetime
import os.path

from .path import get_custom_module
from .exceptions import OasisException

from ods_tools.oed import DictBasedCurrencyRates, get_currency_col, convert_currency
try:
    from forex_python.converter import CurrencyRates as BaseFxCurrencyRates
    from forex_python.converter import RatesNotAvailableError

    class FxCurrencyRates(BaseFxCurrencyRates):
        def __init__(self, *args, date_obj=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.date_obj = date_obj

        def get_rate(self, base_cur, dest_cur, date_obj=None):
            if date_obj is None:
                date_obj = self.date_obj
            try:
                return super().get_rate(base_cur, dest_cur, date_obj)
            except RatesNotAvailableError as e:
                raise OasisException("Issue retrieving rate, most probably the forex api is down") from e

except ImportError:
    FxCurrencyRates = None

import logging
logger = logging.getLogger(__name__)


def create_currency_rates(currency_conversion):
    def get_path(name):
        if os.path.isabs(currency_conversion[name]):
            return currency_conversion[name]
        else:
            return os.path.join(currency_conversion['root_dir'], currency_conversion[name])

    if currency_conversion.get('currency_conversion_type') == 'DictBasedCurrencyRates':
        if currency_conversion.get("source_type") == 'csv':
            return DictBasedCurrencyRates.from_csv(get_path('file_path'), **currency_conversion.get("read_parameters", {}))
        elif currency_conversion.get("source_type") == 'parquet':
            return DictBasedCurrencyRates.from_parquet(get_path('file_path'), **currency_conversion.get("read_parameters", {}))
        elif currency_conversion.get("source_type", '').lower() == 'dict':
            return DictBasedCurrencyRates(currency_conversion['currency_rates'])
        else:
            raise OasisException(f"Unsuported currency_conversion source type : {currency_conversion.get('source_type')}")
    elif currency_conversion.get('currency_conversion_type') == 'FxCurrencyRates':
        if FxCurrencyRates is None:
            raise OasisException("You must install package forex-python to use builtin_currency_conversion_type FxCurrencyRates")

        _datetime = currency_conversion.get('datetime')
        if _datetime is not None:
            _datetime = datetime.datetime.fromisoformat(_datetime)
        return FxCurrencyRates(date_obj=_datetime, **currency_conversion.get("fx_currency_rates_parameters", {}))
    elif currency_conversion.get('currency_conversion_type') == 'custom':
        _module = get_custom_module(get_path('module_path'), 'Currency Rate lookup module path')

        try:
            _class = getattr(_module, currency_conversion['class_name'])
        except AttributeError as e:
            raise OasisException(f"class {currency_conversion['class_name']} "
                                 f"is not defined in module {get_path('module_path')}") from e.__cause__

        return _class(**currency_conversion.get("custom_parameters", {}))

    else:
        raise OasisException(f"unsupported currency_conversion_type {currency_conversion.get('currency_conversion_type')}")


def manage_multiple_currency(oed_path, oed_df, reporting_currency, currency_rate, ods_fields):
    currency_col = get_currency_col(oed_df)
    currencies = oed_df[currency_col].unique()

    if not reporting_currency:
        if len(currencies) > 1:
            logger.warning(f"file {oed_path} contains multiple currencies, but no reporting_currency has been specified")

    elif currency_rate is None:
        if len(currencies) > 1 or currencies[0] != reporting_currency:
            raise OasisException(f"currency_conversion_json is necessary to perform conversion of {oed_path} to {reporting_currency}")

    else:
        logs = []
        for cur in currencies:
            if cur != reporting_currency:
                logs.append(f"{cur} => {reporting_currency}: rate of exchange {currency_rate.get_rate(cur, reporting_currency)}")
        if logs:
            logger.info(f"currency conversion for file {oed_path}:\n\t" + "\n\t".join(logs))
            convert_currency(oed_df, reporting_currency, currency_rate, ods_fields)
            return True
    return False
