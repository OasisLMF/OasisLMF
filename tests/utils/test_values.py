from unittest import TestCase

from datetime import datetime

import pytz
from freezegun import freeze_time
from hypothesis import given
from hypothesis.extra.pytz import timezones
from hypothesis.strategies import datetimes, decimals, floats, integers, one_of, sampled_from

from oasislmf.utils.values import get_timestamp, get_utctimestamp, to_string, NONE_VALUES, to_int, to_float


class GetTimestamp(TestCase):
    def test_time_is_not_supplied___timestamp_of_now_is_returned(self):
        _now = datetime.now()

        with freeze_time(_now):
            res = get_timestamp()

            self.assertEqual(_now.strftime('%Y%m%d%H%M%S'), res)

    @given(datetimes(min_value=datetime(1990, 1, 1)))
    def test_time_is_supplied___timestamp_of_supplied_time_is_returned(self, dt):
        res = get_timestamp(dt)

        self.assertEqual(dt.strftime('%Y%m%d%H%M%S'), res)

    @given(datetimes(min_value=datetime(1990, 1, 1)))
    def test_fmt_is_supplied___timestamp_of_supplied_time_is_returned(self, dt):
        fmt = 'foo %Y-%m-%d %H:%M:%S bar'

        res = get_timestamp(dt, fmt=fmt)

        self.assertEqual(dt.strftime(fmt), res)


class GetUtcTimestamp(TestCase):
    def test_time_is_not_supplied___timestamp_of_utcnow_is_returned(self):
        _now = datetime.utcnow()

        with freeze_time(_now):
            res = get_utctimestamp()

            self.assertEqual(_now.strftime('%Y-%b-%d %H:%M:%S'), res)

    @given(datetimes(timezones=timezones(), min_value=datetime(1990, 1, 1)))
    def test_time_is_supplied___timestamp_of_supplied_time_is_returned(self, dt):
        res = get_utctimestamp(dt)

        self.assertEqual(dt.astimezone(pytz.utc).strftime('%Y-%b-%d %H:%M:%S'), res)

    @given(datetimes(timezones=timezones(), min_value=datetime(1990, 1, 1)))
    def test_fmt_is_supplied___timestamp_of_supplied_time_is_returned(self, dt):
        fmt = 'foo %Y-%m-%d %H:%M:%S bar'

        res = get_utctimestamp(dt, fmt=fmt)

        self.assertEqual(dt.astimezone(pytz.utc).strftime(fmt), res)


class ToString(TestCase):
    def test_value_is_none___result_is_empty(self):
        res = to_string(None)

        self.assertEqual('', res)

    @given(one_of(integers(), floats(allow_infinity=False, allow_nan=False), decimals(allow_infinity=False, allow_nan=False)))
    def test_value_is_not_none___result_is_string_conversion(self, value):
        res = to_string(value)

        self.assertEqual(str(value), res)


class ToInt(TestCase):
    @given(sampled_from(NONE_VALUES))
    def test_value_is_one_of_the_none_values___result_is_none(self, value):
        res = to_int(value)

        self.assertIsNone(res)

    @given(floats(allow_infinity=False, allow_nan=False))
    def test_value_is_float___result_is_converted_to_int(self, value):
        res = to_int(value)

        self.assertEqual(int(value), res)


class ToFloat(TestCase):
    @given(sampled_from(NONE_VALUES))
    def test_value_is_one_of_the_none_values___result_is_none(self, value):
        res = to_float(value)

        self.assertIsNone(res)

    @given(decimals(allow_infinity=False, allow_nan=False))
    def test_value_is_decimal___result_is_converted_to_float(self, value):
        res = to_float(value)

        self.assertEqual(float(value), res)
