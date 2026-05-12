from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import pytest

from oasislmf.lookup.builtin import (
    Lookup,
    create_lat_lon_id_functions,
    get_step,
    jit_geo_grid_lookup,
    normal_to_z_index,
    undo_z_index,
    z_index,
    z_index_to_normal,
)
from oasislmf.utils.status import OASIS_UNKNOWN_ID

FILES_DIR = Path(__file__).resolve().parent


@pytest.mark.parametrize("x, y, expected", [
    (0, 0, 0),
    (1, 0, 1),
    (0, 1, 2),
    (1, 1, 3),
    (2, 2, 12),
    (3, 3, 15),
    (5, 10, z_index(5, 10)),
])
def test_z_index(x, y, expected):
    assert z_index(x, y) == expected


@pytest.mark.parametrize("z", [
    0, 1, 2, 3, 12, 15, 99, 255, 1023
])
def test_undo_z_index(z):
    x, y = undo_z_index(z)
    assert z_index(x, y) == z


@pytest.mark.parametrize("z, size_across", [
    (OASIS_UNKNOWN_ID, 2),
    (1, 10),
    (5, 10),
    (15, 5),
    (99, 20),
    (255, 100)
])
def test_z_index_normal_conversion(z, size_across):
    normal = z_index_to_normal(z, size_across)
    assert normal_to_z_index(normal, size_across) == z


@pytest.mark.parametrize("is_lat, value, expected, reverse_lat, reverse_lon", [
    (True, 5, 5, False, False),
    (False, 7, 7, False, False),
    (True, 3, 7, True, False),
    (True, 3, 3, False, True),
    (False, 9, 1, True, True),
    (False, 6, 6, True, False)
])
def test_lat_lon_id_functions(is_lat, value, expected, reverse_lat, reverse_lon):
    lat_id, lon_id = create_lat_lon_id_functions(0, 10, 0, 10, 1, reverse_lat, reverse_lon)
    func = lat_id if is_lat else lon_id
    assert func(value) == expected


@pytest.mark.parametrize("idx, expected", [
    (0, 11),
    (1, 22),
    (2, OASIS_UNKNOWN_ID)
])
def test_jit_geo_grid_lookup(idx, expected):
    lat = np.array([1, 2, 11])
    lon = np.array([1, 2, 3])
    lat_min, lat_max, lon_min, lon_max = 0, 10, 0, 10

    @nb.njit()
    def mock_compute_id(lat, lon, lat_id, lon_id):
        return lat_id(lat) * 10 + lon_id(lon)

    lat_id, lon_id = create_lat_lon_id_functions(
        lat_min, lat_max, lon_min, lon_max, 1, False, False
    )

    result = jit_geo_grid_lookup(
        lat, lon, lat_min, lat_max, lon_min, lon_max, mock_compute_id,
        lat_id, lon_id
    )
    assert result[idx] == expected


@pytest.mark.parametrize("grid, expected", [
    ({"lon_min": 0, "lon_max": 10, "lat_min": 0, "lat_max": 10, "arc_size": 1},
     100
     ),
    ({"lon_min": 0, "lon_max": 10, "lat_min": 0, "lat_max": 10, "arc_size": 2},
     25
     ),
])
def test_get_step(grid, expected):
    assert get_step(grid) == expected


@pytest.mark.parametrize("file_path, file_type, success", [
    ("example_input.csv", None, True),
    ("example_input.csv", "csv", True),
    ("example_input.parquet", "parquet", True),
    ("example_input.parquet", None, False),
    ("example_input.csv", "parquet", False)
])
def test_build_merge_respects_filetype(file_path, file_type, success):
    if success:
        Lookup(config={}).build_merge(file_path=str(FILES_DIR / file_path), file_type=file_type, id_columns=['FIRST_ID', 'SECOND_ID', 'FIFTH_ID'])
    else:
        with pytest.raises(Exception):
            Lookup(config={}).build_merge(file_path=str(FILES_DIR / file_path), file_type=file_type, id_columns=['FIRST_ID', 'SECOND_ID', 'FIFTH_ID'])


@pytest.fixture
def rtree_locations_all_coordinates():
    return pd.DataFrame(columns=["longitude", "latitude", "locname"], data=[
        [0.373700517342545, 46.4691264361466, "inside_1"],
        [0.639522260665994, 46.3538195759967, "inside_2"],
        [0.511892615692815, 46.4703388960666, "close_to_1"],
        [0.400106650785272, 46.3307289492925, "far_away"],
    ])

@pytest.fixture
def rtree_locations_no_coordinates():
    return pd.DataFrame(columns=["longitude", "latitude", "locname"], data=[
        [None, None, "A"],
        [None, None, "B"],
    ])

@pytest.fixture
def rtree_locations_some_coordinates():
    return pd.DataFrame(columns=["longitude", "latitude", "locname"], data=[
        [None, None, "A"],
        [0.373700517342545, 46.4691264361466, "inside_1"],
    ])

@pytest.mark.parametrize(
    ("locations_by_name", "expected_ids"),
    [
        ("rtree_locations_all_coordinates", [1, 2, 1, OASIS_UNKNOWN_ID]),
        ("rtree_locations_no_coordinates", [OASIS_UNKNOWN_ID, OASIS_UNKNOWN_ID]),
        ("rtree_locations_some_coordinates", [OASIS_UNKNOWN_ID, 1]),
    ],
    )
def test_build_rtree_associates_correctly(locations_by_name, expected_ids, request):
    """Test that the rtree builin correctly associates locations to polygons.

    Test polygons have the following centroids:
      - poly1    POINT (0.41289 46.46745)
      - poly2    POINT (0.63856 46.348)
    """
    locations = request.getfixturevalue(locations_by_name)
    rtree = Lookup(config={}).build_rtree(
        file_path=(FILES_DIR / "rtree_areas.parquet").as_posix(),
        file_type="parquet",
        id_columns="poly_id",
        nearest_neighbor_max_distance=12000, # Euclidean distance in metres, not spherical distance.
    )
    output = rtree(locations)
    expected = locations.copy().assign(poly_id=expected_ids)

    # Sort values so order doesn't matter.
    pd.testing.assert_frame_equal(
        output.sort_values("locname"),
        expected.sort_values("locname"),
        check_dtype=False,
    )

def test_build_rtree_accepts_deprecated_parameter(rtree_locations_all_coordinates):
    """Test that the rtree builin correctly associates locations to polygons."""
    locations = rtree_locations_all_coordinates
    with pytest.warns(DeprecationWarning):
        rtree = Lookup(config={}).build_rtree(
            file_path=(FILES_DIR / "rtree_areas.parquet").as_posix(),
            file_type="parquet",
            id_columns="poly_id",
            nearest_neighbor_min_distance=12000, # Deprecated parameter name.
        )
    output = rtree(locations)
    expected = locations.copy().assign(poly_id=[1, 2, 1, OASIS_UNKNOWN_ID])

    # Sort values so order doesn't matter.
    pd.testing.assert_frame_equal(
        output.sort_values("locname"),
        expected.sort_values("locname"),
        check_dtype=False,
    )
