from geopy import distance


def get_geodesic_distance(lon1, lat1, lon2, lat2, method='vincenty', dist_unit='metres'):
    """
    Calculates the geodesic distance between two points given as
    longitude-latitude pairs using a given method - the default
    method is the Vincenty formula (`vincenty`)

        `<https://en.wikipedia.org/wiki/Vincenty%27s_formulae>`_

    and the other choice is the great-circle method (``great_circle``)

        `<https://en.wikipedia.org/wiki/Great-circle_distance>`_

    The default distance unit is metres (``m`` or ``metres`` or ``meters``),
    but other options include kilometres (``km`` or ``kilometers`` or
    ``kilometers``), feet (``ft`` or ``feet``), miles (``mi`` or ``miles``),
    nautical miles (``nm`` or ``nautical``).

    The longitude-latitude arguments can be passed in separately or as two
    tuples prefixed by ``*``, which is required to unpack the tuples inside the
    method.
    """
    point1, point2 = (lon1, lat1), (lon2, lat2)
    if dist_unit:
        if dist_unit == 'metres':
            dist_unit = 'meters'
        elif dist_unit == 'kilometres':
            dist_unit = 'kilometers'
    else:
        dist_unit = 'meters'

    dist = (
        distance.great_circle(point1, point2) if method == 'great_circle'
        else distance.vincenty(point1, point2)
    )
    return getattr(dist, dist_unit)


def valid_latitude(latitude):
    """
    Validates a latitude value
    """
    return -90 <= latitude <= 90


def valid_longitude(longitude):
    """
    Validates a longitude value
    """
    return -180 <= longitude <= 180


def valid_lonlat(longitude, latitude):
    """
    Validates a longitude-latitude value pair
    """
    return valid_longitude(longitude) and valid_latitude(latitude)
