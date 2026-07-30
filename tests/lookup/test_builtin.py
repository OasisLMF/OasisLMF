from collections import OrderedDict
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
from oasislmf.utils.status import (
    OASIS_KEYS_STATUS,
    OASIS_KEYS_STATUS_MODELLED,
    OASIS_UNKNOWN_ID,
)

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
    """Test that the rtree builtin correctly associates locations to polygons.

    Test polygons have the following centroids:
      - poly1    POINT (0.41289 46.46745)
      - poly2    POINT (0.63856 46.348)
    """
    locations = request.getfixturevalue(locations_by_name)
    rtree = Lookup(config={}).build_rtree(
        file_path=(FILES_DIR / "rtree_areas.parquet").as_posix(),
        file_type="parquet",
        id_columns="poly_id",
        nearest_neighbor_max_distance=12000,  # Euclidean distance in metres, not spherical distance.
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
    """Test that the rtree builtin still works with the deprecated parameter."""
    with pytest.warns(DeprecationWarning):
        rtree = Lookup(config={}).build_rtree(
            file_path=(FILES_DIR / "rtree_areas.parquet").as_posix(),
            file_type="parquet",
            id_columns="poly_id",
            nearest_neighbor_min_distance=12000,  # Deprecated parameter name should raise warning.
        )
    output = rtree(rtree_locations_all_coordinates)
    expected = rtree_locations_all_coordinates.copy().assign(poly_id=[1, 2, 1, OASIS_UNKNOWN_ID])

    # Sort values so order doesn't matter.
    pd.testing.assert_frame_equal(
        output.sort_values("locname"),
        expected.sort_values("locname"),
        check_dtype=False,
    )


@pytest.mark.parametrize("preparations, values, expected", [
    ({"min": 5}, [1, 5, 7], [5, 5, 7]),           # values below min are raised to min
    ({"max": 10}, [7, 10, 20], [7, 10, 10]),      # values above max are lowered to max
    ({"min": 5, "max": 10}, [1, 7, 20], [5, 7, 10]),
])
def test_build_prepare_min_max_clamp(preparations, values, expected):
    prepare = Lookup(config={}).build_prepare(my_col=preparations)
    result = prepare(pd.DataFrame({"my_col": values}))
    assert result["my_col"].tolist() == expected


def test_split_loc_perils_covered_marks_not_at_risk():
    """A location whose perils are not modelled is flagged 'not at risk' — the
    status must be the string id, not the whole OASIS_KEYS_STATUS dict, otherwise
    it drops out of the 'modelled' set used by the exposure summary report."""
    fct = Lookup(config={}).build_split_loc_perils_covered(model_perils_covered=["QEQ"])
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "LocPerilsCovered": ["QEQ", "WTC"],   # loc 2's peril is outside the model
    })

    result = fct(locations)

    not_at_risk = result[result["loc_id"] == 2]
    assert len(not_at_risk) == 1
    status = not_at_risk["status"].iloc[0]
    assert status == OASIS_KEYS_STATUS["notatrisk"]["id"]
    assert isinstance(status, str)
    assert not_at_risk["status"].isin(OASIS_KEYS_STATUS_MODELLED).all()


class _FakeGeoTiffDataset:
    """Minimal stand-in for a gdal dataset so build_geotiff can be tested without gdal."""

    def __init__(self, array, geotransform):
        self._array = array
        self._geotransform = geotransform
        self.RasterCount = array.shape[2]

    def GetGeoTransform(self):
        return self._geotransform

    def GetVirtualMemArray(self):
        return self._array


class _FakeGdal:
    GA_ReadOnly = 0

    def __init__(self, dataset, inv_gt):
        self._dataset = dataset
        self._inv_gt = inv_gt

    def Open(self, path, mode):
        return self._dataset

    def InvGeoTransform(self, geotransform):
        return self._inv_gt


def test_build_geotiff_out_of_range_uses_correct_default_per_column(monkeypatch):
    """Out-of-raster locations must get each column's OWN default. The defaults
    array is read by band column order, so it must be written that way too — the
    bug wrote it by band id, swapping defaults across columns."""
    # 1x1 raster, 3 bands with values 100/200/300 at the single pixel.
    raster = np.array([[[100, 200, 300]]], dtype="int64")
    # Identity inverse geo-transform: px = int(lon + 0.5), py = int(lat + 0.5).
    inv_gt = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    fake_gdal = _FakeGdal(_FakeGeoTiffDataset(raster, (0, 1, 0, 0, 0, 1)), inv_gt)
    monkeypatch.setattr("oasislmf.lookup.builtin.gdal", fake_gdal)

    # Columns map to bands in a permuted order so a wrong index is detectable.
    band_info = OrderedDict([
        ("a", {"id": 3, "default": -11}),   # band index 2 -> 300
        ("b", {"id": 1, "default": -22}),   # band index 0 -> 100
        ("c", {"id": 2, "default": -33}),   # band index 1 -> 200
    ])
    geotiff_lookup = Lookup(config={}).build_geotiff(file_path="dummy.tif", band_info=band_info)

    locations = pd.DataFrame({
        "longitude": [0.0, 99.0],   # row 0 in-range, row 1 out-of-range
        "latitude": [0.0, 99.0],
    })
    result = geotiff_lookup(locations)

    # In-range: each column reads its own band.
    assert result.loc[0, "a"] == 300
    assert result.loc[0, "b"] == 100
    assert result.loc[0, "c"] == 200
    # Out-of-range: each column gets its own default (the bug swapped these).
    assert result.loc[1, "a"] == -11
    assert result.loc[1, "b"] == -22
    assert result.loc[1, "c"] == -33
