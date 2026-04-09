import pytest
import numpy as np
import numba as nb
import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path
from mock import patch
from oasislmf.lookup.builtin import (
    Lookup, z_index, undo_z_index,
    z_index_to_normal, normal_to_z_index,
    create_lat_lon_id_functions, jit_geo_grid_lookup,
    get_step
)

from oasislmf.utils.exceptions import OasisException
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


@patch("oasislmf.utils.peril.PERILS")
@patch("oasislmf.utils.peril.PERIL_GROUPS")
def test_build_split_loc_perils_covered(mock_peril_groups, mock_perils):
    mock_perils.values.return_value = [
        {'id': 'PERIL1'},
        {'id': 'PERIL2'}
    ]

    mock_peril_groups.values.return_value = [
        {'id': 'ALL', 'peril_ids': ['PERIL1', 'PERIL2']}
    ]

    lookup_fn = Lookup(config={}).build_split_loc_perils_covered(model_perils_covered=["ALL", "PERIL1", "PERIL2"])

    locations = pd.DataFrame({
        "loc_id": [1, 2, 3],
        "LocPerilsCovered": ["ALL", "PERIL1", "PERIL2"]
    })

    result = lookup_fn(locations)
    expected_result = pd.DataFrame({
        'loc_id': [1, 1, 2, 3],
        'peril_group_id': ['ALL', 'ALL', 'PERIL1', 'PERIL2'],
        'peril_id': ['PERIL1', 'PERIL2', 'PERIL1', 'PERIL2']
    })

    result = result[expected_result.columns].sort_values(by=['loc_id', 'peril_id'],
                                                         ignore_index=True)
    assert_frame_equal(result, expected_result)


# ---------------------------------------------------------------------------
# H3 lookup tests
# ---------------------------------------------------------------------------

h3 = pytest.importorskip("h3", minversion="4", reason="h3>=4 not installed")

# (lat, lon, area_peril_id) triples used as the test mapping
_H3_RESOLUTION = 5
_SAMPLE_COORDS = [
    (51.5074, -0.1278),   # London
    (40.7128, -74.0060),  # New York
    (35.6762, 139.6503),  # Tokyo
]


@pytest.fixture()
def h3_mapping_csv(tmp_path):
    """CSV mapping file: h3_int64 -> area_peril_id for the sample coordinates."""
    rows = [
        {
            "h3_int64": h3.str_to_int(h3.latlng_to_cell(lat, lon, _H3_RESOLUTION)),
            "area_peril_id": idx,
        }
        for idx, (lat, lon) in enumerate(_SAMPLE_COORDS, start=1)
    ]
    path = tmp_path / "h3_to_areaperil.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture()
def h3_mapping_parquet(tmp_path):
    """Parquet mapping file: h3_int64 -> area_peril_id for the sample coordinates."""
    rows = [
        {
            "h3_int64": h3.str_to_int(h3.latlng_to_cell(lat, lon, _H3_RESOLUTION)),
            "area_peril_id": idx,
        }
        for idx, (lat, lon) in enumerate(_SAMPLE_COORDS, start=1)
    ]
    path = tmp_path / "h3_to_areaperil.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_build_h3_maps_known_locations(h3_mapping_csv):
    """Locations whose H3 cell is in the mapping file receive the correct area_peril_id."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    locations = pd.DataFrame({
        "loc_id": [1, 2, 3],
        "latitude": [lat for lat, lon in _SAMPLE_COORDS],
        "longitude": [lon for lat, lon in _SAMPLE_COORDS],
    })
    result = lookup_fn(locations)

    for loc_id, expected_ap_id in [(1, 1), (2, 2), (3, 3)]:
        actual = result.loc[result["loc_id"] == loc_id, "area_peril_id"].iloc[0]
        assert actual == expected_ap_id, f"loc_id={loc_id}: expected {expected_ap_id}, got {actual}"


def test_build_h3_null_coordinates_get_unknown_id(h3_mapping_csv):
    """Locations with null lat/lon receive OASIS_UNKNOWN_ID."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    lat0, lon0 = _SAMPLE_COORDS[0]
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "latitude": [None, lat0],
        "longitude": [None, lon0],
    })
    result = lookup_fn(locations)

    assert result.loc[result["loc_id"] == 1, "area_peril_id"].iloc[0] == OASIS_UNKNOWN_ID
    assert result.loc[result["loc_id"] == 2, "area_peril_id"].iloc[0] == 1


def test_build_h3_unmatched_location_gets_unknown_id(h3_mapping_csv):
    """Locations whose H3 cell is absent from the mapping receive OASIS_UNKNOWN_ID."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    locations = pd.DataFrame({
        "loc_id": [1],
        "latitude": [-89.0],
        "longitude": [179.0],
    })
    result = lookup_fn(locations)

    assert result.loc[result["loc_id"] == 1, "area_peril_id"].iloc[0] == OASIS_UNKNOWN_ID


def test_build_h3_respects_parquet_filetype(h3_mapping_parquet):
    """build_h3 correctly reads a parquet mapping file."""
    lookup_fn = Lookup(config={}).build_h3(
        resolution=_H3_RESOLUTION, file_path=str(h3_mapping_parquet), file_type="parquet"
    )

    lat0, lon0 = _SAMPLE_COORDS[0]
    locations = pd.DataFrame({
        "loc_id": [1],
        "latitude": [lat0],
        "longitude": [lon0],
    })
    result = lookup_fn(locations)

    assert result.loc[result["loc_id"] == 1, "area_peril_id"].iloc[0] == 1


def test_build_h3_missing_h3_int64_column_raises(tmp_path):
    """Mapping file without an h3_int64 column raises OasisException at build time."""
    bad_path = tmp_path / "bad_mapping.csv"
    pd.DataFrame({"wrong_column": [1], "area_peril_id": [1]}).to_csv(bad_path, index=False)

    with pytest.raises(OasisException, match="h3_int64"):
        Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(bad_path))


def test_build_h3_area_peril_id_dtype(h3_mapping_csv):
    """area_peril_id column has Int64 dtype after lookup (nullable integer)."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    lat0, lon0 = _SAMPLE_COORDS[0]
    locations = pd.DataFrame({
        "loc_id": [1],
        "latitude": [lat0],
        "longitude": [lon0],
    })
    result = lookup_fn(locations)

    assert result["area_peril_id"].dtype == pd.Int64Dtype()


def test_build_h3_mixed_valid_invalid_locations(h3_mapping_csv):
    """Mix of matched, unmatched, and null-coordinate rows in one call."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    lat0, lon0 = _SAMPLE_COORDS[0]
    locations = pd.DataFrame({
        "loc_id": [1, 2, 3],
        "latitude": [lat0, -89.0, None],
        "longitude": [lon0, 179.0, None],
    })
    result = lookup_fn(locations.copy())

    assert result.loc[result["loc_id"] == 1, "area_peril_id"].iloc[0] == 1
    assert result.loc[result["loc_id"] == 2, "area_peril_id"].iloc[0] == OASIS_UNKNOWN_ID
    assert result.loc[result["loc_id"] == 3, "area_peril_id"].iloc[0] == OASIS_UNKNOWN_ID
