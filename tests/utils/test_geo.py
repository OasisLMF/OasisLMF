from unittest import TestCase

from hypothesis import given
from hypothesis.strategies import floats, one_of, sampled_from
from mock import patch

from oasislmf.utils.geo import valid_latitude, valid_longitude, valid_lonlat, get_geodesic_distance


def valid_latitudes():
    return floats(min_value=-90, max_value=90)


def invalid_latitudes():
    return one_of(floats(min_value=90).filter(lambda f: f > 90), floats(max_value=-90).filter(lambda f: f < -90))


def valid_longitudes():
    return floats(min_value=-180, max_value=180)


def invalid_longitudes():
    return one_of(floats(min_value=180).filter(lambda f: f > 180), floats(max_value=-180).filter(lambda f: f < -180))


def geo_distance_methods():
    return sampled_from(['vincenty', 'great_circle'])


def distance_units():
    return sampled_from([
        (None, 'meters'),
        ('m', 'm'),
        ('metres', 'meters'),
        ('meters', 'meters'),
        ('km', 'km'),
        ('kilometres', 'kilometers'),
        ('kilometers', 'kilometers'),
        ('ft', 'ft'),
        ('feet', 'feet'),
        ('mi', 'mi'),
        ('miles', 'miles'),
        ('nm', 'nm'),
        ('nautical', 'nautical'),
    ])


class ValidLatitude(TestCase):
    @given(floats(min_value=90).filter(lambda f: f > 90))
    def test_value_is_greater_than_90___response_is_false(self, value):
        self.assertFalse(valid_latitude(value))

    @given(floats(max_value=-90).filter(lambda f: f < -90))
    def test_value_is_less_than_90___response_is_false(self, value):
        self.assertFalse(valid_latitude(value))

    @given(valid_latitudes())
    def test_value_is_between_plus_or_minus_90___response_is_true(self, value):
        self.assertTrue(valid_latitude(value))


class ValidLongitude(TestCase):
    @given(floats(min_value=180).filter(lambda f: f > 180))
    def test_value_is_greater_than_180___response_is_false(self, value):
        self.assertFalse(valid_longitude(value))

    @given(floats(max_value=-180).filter(lambda f: f < -180))
    def test_value_is_less_than_180___response_is_false(self, value):
        self.assertFalse(valid_longitude(value))

    @given(valid_longitudes())
    def test_value_is_between_plus_or_minus_180___response_is_true(self, value):
        self.assertTrue(valid_longitude(value))


class ValidLonLat(TestCase):
    @given(invalid_longitudes(), valid_latitudes())
    def test_longitude_is_invalid___response_is_false(self, lon, lat):
        self.assertFalse(valid_lonlat(lon, lat))

    @given(valid_longitudes(), invalid_latitudes())
    def test_latitude_is_invalid___response_is_false(self, lon, lat):
        self.assertFalse(valid_lonlat(lon, lat))

    @given(valid_longitudes(), valid_latitudes())
    def test_long_and_lat_are_both_valid___response_is_true(self, lon, lat):
        self.assertTrue(valid_lonlat(lon, lat))


class GetGeodesicDistance(TestCase):
    @given(valid_longitudes(), valid_latitudes(), valid_longitudes(), valid_latitudes(), geo_distance_methods(), distance_units())
    def test_all_parameters_are_valid___geopy_value_is_returned_for_correct_method(self, lon1, lat1, lon2, lat2, method, units):
        class FakeGeodesicResult(object):
            def __init__(self, method, p1, p2):
                self.method = method
                self.p1 = p1
                self.p2 = p2
                self.props = []

            def __getattr__(self, item):
                if item in self.__dict__:
                    return self.__dict__[item]
                else:
                    self.props.append(item)
                    return self

        class FakeGeodesicCalc(object):
            def vincenty(self, p1, p2):
                return FakeGeodesicResult('vincenty', p1, p2)

            def great_circle(self, p1, p2):
                return FakeGeodesicResult('great_circle', p1, p2)

        unit_name, unit_prop = units
        p1 = (lon1, lat1)
        p2 = (lon2, lat2)

        with patch('oasislmf.utils.geo.distance', FakeGeodesicCalc()):
            res = get_geodesic_distance(lon1, lat1, lon2, lat2, method, unit_name)

            self.assertEqual(res.method, method)
            self.assertEqual(res.p1, p1)
            self.assertEqual(res.p2, p2)
            self.assertEqual(res.props, [unit_prop])
