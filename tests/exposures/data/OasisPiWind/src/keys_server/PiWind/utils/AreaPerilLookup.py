# -*- coding: utf-8 -*-

__all__ = ['AreaPerilLookup']

import io
import csv
import logging

from shapely.geometry import (
    Point,
    MultiPoint,
)

from oasislmf.utils.status import (
    KEYS_STATUS_FAIL,
    KEYS_STATUS_NOMATCH,
    KEYS_STATUS_SUCCESS,
)


class AreaPerilLookup(object):
    '''
    Functionality to perform an area peril lookup.
    '''

    _lookup_data = []

    def __init__(self, areas_file=None):
        if areas_file:
            with io.open(areas_file, 'r', encoding='utf-8') as f:
                dr = csv.DictReader(f)
                self.load_lookup_data(dr)


    def load_lookup_data(self, data):
        '''
        Load the lookup data.
        Args:
            data: the lookup data.
        '''
        self._lookup_data = [
            (
                int(r['AREA_PERIL_ID']),
                MultiPoint(
                    tuple((float(r['LAT{}'.format(i)]),float(r['LON{}'.format(i)])) for i in range(1, 5))
                ).convex_hull
            ) for r in data
        ]


    def validate_lat(self, lat):
        '''
        Check that a string or number is a valid latitude.
        Args:
            s (string or number): the latitude
        Returns:
            True if string is a valid latitude, False otherwise
        '''
        try:
            return -90 <= float(lat) <= 90
        except ValueError:
            return False


    def validate_lon(self, lon):
        '''
        Check that a string or number is a valid longitude.
        Args:
            s (string or number): the longitude
        Returns:
            True if string is a valid longitude, False otherwise
        '''
        try:
            return -180 <= float(lon) <= 180
        except ValueError:
            return False


    def do_lookup_location(self, location):
        '''
        Perform a lookup on a specified location.
        Args:
            location: the location to lookup.
        Return:
            Lookup result
        '''
        logging.debug("Looking up location.")
        
        status = KEYS_STATUS_NOMATCH
        area_peril_id = None
        message = ''

        lat = location['lat']
        lon = location['lon']

        if not self.validate_lat(lat) & self.validate_lon(lon):
            status = KEYS_STATUS_FAIL
            area_peril_id = None
            message = "Invalid lat/lon"
        else:
            loc_point = Point(lat, lon)
            for (cell_area_peril_id, cell) in self._lookup_data:
                if cell.intersects(loc_point):
                    area_peril_id = cell_area_peril_id
                    status = KEYS_STATUS_SUCCESS
                    break

        return {
            'status': status,
            'area_peril_id': area_peril_id,
            'message': message
        }
