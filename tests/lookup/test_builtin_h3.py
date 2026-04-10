# ---------------------------------------------------------------------------
# H3 lookup tests
# ---------------------------------------------------------------------------
import pytest
import pandas as pd
from oasislmf.lookup.builtin import (
    Lookup)
from oasislmf.utils.status import OASIS_UNKNOWN_ID
from oasislmf.utils.exceptions import OasisException

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
