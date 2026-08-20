# ---------------------------------------------------------------------------
# H3 lookup tests
# ---------------------------------------------------------------------------
import numpy as np
import pytest
import pandas as pd
from oasislmf.lookup.builtin import (
    Lookup)
from oasislmf.utils.status import OASIS_KEYS_STATUS, OASIS_UNKNOWN_ID
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


@pytest.mark.parametrize("lat,lon", [
    (51.5074, np.inf),
    (51.5074, -np.inf),
    (np.inf, -0.1278),
    (-np.inf, -0.1278),
    (np.inf, np.inf),
])
def test_build_h3_infinite_coordinates_get_unknown_id(h3_mapping_csv, lat, lon):
    """An infinite coordinate is unknown, not the last mapped cell's area_peril_id.

    An isnan-only validity test lets an infinity through, and 1j * inf has a nan real part, so the
    complex key reaches factorize as an NA and takes its -1 sentinel. -1 is a legal numpy index, so
    the lookup would silently return whatever the last distinct coordinate in the chunk resolved
    to -- a wrong value that also depends on row order, and so on multiprocessing chunk boundaries.
    """
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    # the last sample coordinate is ordered last here, so a -1 code would resolve to its id
    lat_last, lon_last = _SAMPLE_COORDS[-1]
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "latitude": [lat, lat_last],
        "longitude": [lon, lon_last],
    })
    result = lookup_fn(locations)

    assert result.loc[result["loc_id"] == 2, "area_peril_id"].iloc[0] == len(_SAMPLE_COORDS)
    assert result.loc[result["loc_id"] == 1, "area_peril_id"].iloc[0] == OASIS_UNKNOWN_ID


def test_build_h3_area_peril_id_above_float64_exact_range(tmp_path):
    """area_peril_id keeps its exact value above 2^53, where a float64 intermediate would round it.

    An h3 model can use the cell index itself as the area_peril_id; cells are ~6e17, well past the
    range float64 represents exactly, so every id would come back off by one.
    """
    lat0, lon0 = _SAMPLE_COORDS[0]
    cell0 = h3.str_to_int(h3.latlng_to_cell(lat0, lon0, _H3_RESOLUTION))
    big_id = 2 ** 53 + 1

    path = tmp_path / "big_id_mapping.csv"
    pd.DataFrame({"h3_int64": [cell0], "area_peril_id": [big_id]}).to_csv(path, index=False)

    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(path))
    result = lookup_fn(pd.DataFrame({"loc_id": [1], "latitude": [lat0], "longitude": [lon0]}))

    assert result["area_peril_id"].iloc[0] == big_id


def test_build_h3_cell_index_as_area_peril_id(tmp_path):
    """The identity mapping case: every cell index maps to itself, exactly."""
    cells = [h3.str_to_int(h3.latlng_to_cell(lat, lon, _H3_RESOLUTION)) for lat, lon in _SAMPLE_COORDS]
    path = tmp_path / "identity_mapping.csv"
    pd.DataFrame({"h3_int64": cells, "area_peril_id": cells}).to_csv(path, index=False)

    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(path))
    result = lookup_fn(pd.DataFrame({
        "loc_id": np.arange(1, len(_SAMPLE_COORDS) + 1),
        "latitude": [lat for lat, lon in _SAMPLE_COORDS],
        "longitude": [lon for lat, lon in _SAMPLE_COORDS],
    }))

    assert result["area_peril_id"].to_numpy().tolist() == cells


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


def test_build_h3_missing_area_peril_id_column_raises(tmp_path):
    """Mapping file without an area_peril_id column raises OasisException at build time."""
    bad_path = tmp_path / "no_area_peril.csv"
    lat0, lon0 = _SAMPLE_COORDS[0]
    pd.DataFrame({
        "h3_int64": [h3.str_to_int(h3.latlng_to_cell(lat0, lon0, _H3_RESOLUTION))],
    }).to_csv(bad_path, index=False)

    with pytest.raises(OasisException, match="area_peril_id"):
        Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(bad_path))


def test_build_h3_duplicate_mapping_cell_raises(tmp_path):
    """A cell index repeated in the mapping file is a data error and raises at build time."""
    dup_path = tmp_path / "duplicate_mapping.csv"
    lat0, lon0 = _SAMPLE_COORDS[0]
    lat1, lon1 = _SAMPLE_COORDS[1]
    cell0 = h3.str_to_int(h3.latlng_to_cell(lat0, lon0, _H3_RESOLUTION))
    cell1 = h3.str_to_int(h3.latlng_to_cell(lat1, lon1, _H3_RESOLUTION))
    pd.DataFrame({"h3_int64": [cell0, cell1, cell0], "area_peril_id": [1, 2, 99]}).to_csv(dup_path, index=False)

    with pytest.raises(OasisException, match="unique 'h3_int64'"):
        Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(dup_path))


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


def _row_wise_h3_int64(locations, resolution):
    """The original row-wise cell index construction, kept as the reference implementation."""
    valid = locations['latitude'].notna() & locations['longitude'].notna()
    h3_int64 = pd.Series(0, index=locations.index, dtype='int64')
    if valid.any():
        h3_int64.loc[valid] = [
            h3.str_to_int(h3.latlng_to_cell(lat, lon, resolution))
            for lat, lon in zip(locations.loc[valid, 'latitude'], locations.loc[valid, 'longitude'])
        ]
    return h3_int64.to_numpy()


def test_build_h3_matches_row_wise_reference(tmp_path):
    """The vectorized cell indices match the row-wise implementation over a random sample."""
    rng = np.random.default_rng(20260810)
    n = 5000
    # draw from a small pool of coordinates so that duplicates exercise the de-duplication path
    pool_lat = rng.uniform(-89.0, 89.0, 250)
    pool_lon = rng.uniform(-180.0, 180.0, 250)
    picked = rng.integers(0, pool_lat.size, n)
    latitude = pool_lat[picked]
    longitude = pool_lon[picked]
    # rows where one or both coordinates are missing
    latitude[rng.random(n) < 0.05] = np.nan
    longitude[rng.random(n) < 0.05] = np.nan

    locations = pd.DataFrame({"loc_id": np.arange(1, n + 1), "latitude": latitude, "longitude": longitude})
    expected = _row_wise_h3_int64(locations, _H3_RESOLUTION)

    # map two thirds of the pool's cells, so the sample resolves to a mix of ids and unknowns
    pool_cells = sorted({
        h3.str_to_int(h3.latlng_to_cell(lat, lon, _H3_RESOLUTION)) for lat, lon in zip(pool_lat, pool_lon)
    })
    mapped_cells = pool_cells[:len(pool_cells) * 2 // 3]
    mapping_path = tmp_path / "pool_to_areaperil.csv"
    pd.DataFrame({
        "h3_int64": mapped_cells,
        "area_peril_id": np.arange(1, len(mapped_cells) + 1),
    }).to_csv(mapping_path, index=False)

    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(mapping_path))
    result = lookup_fn(locations.copy())

    # the lookup drops h3_int64, so compare through the mapping it feeds
    mapping = pd.read_csv(mapping_path).set_index('h3_int64')['area_peril_id']
    expected_ap_id = pd.Series(expected).map(mapping).fillna(OASIS_UNKNOWN_ID).astype('int64')
    # guard against a vacuous comparison: both outcomes must be well represented
    assert (expected_ap_id != OASIS_UNKNOWN_ID).sum() > n // 2
    assert (expected_ap_id == OASIS_UNKNOWN_ID).sum() > n // 10
    assert result['area_peril_id'].to_numpy().tolist() == expected_ap_id.to_numpy().tolist()


def test_build_h3_duplicate_coordinates_share_a_result(h3_mapping_csv):
    """Repeated coordinates all resolve to the same area_peril_id (de-duplicated conversion)."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    lat0, lon0 = _SAMPLE_COORDS[0]
    lat1, lon1 = _SAMPLE_COORDS[1]
    locations = pd.DataFrame({
        "loc_id": [1, 2, 3, 4, 5],
        "latitude": [lat0, lat1, lat0, None, lat1],
        "longitude": [lon0, lon1, lon0, None, lon1],
    })
    result = lookup_fn(locations)

    assert result["area_peril_id"].to_numpy().tolist() == [1, 2, 1, OASIS_UNKNOWN_ID, 2]


def test_build_h3_distinct_coordinates_in_one_cell(h3_mapping_csv):
    """Different coordinates landing in the same cell both resolve (duplicate cells after de-dup)."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    lat0, lon0 = _SAMPLE_COORDS[0]
    # a tiny offset stays well inside a resolution 5 cell (edge length ~8km) but is a distinct point
    nudged = (lat0 + 1e-5, lon0 + 1e-5)
    assert nudged != (lat0, lon0)
    assert h3.latlng_to_cell(*nudged, _H3_RESOLUTION) == h3.latlng_to_cell(lat0, lon0, _H3_RESOLUTION)

    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "latitude": [lat0, nudged[0]],
        "longitude": [lon0, nudged[1]],
    })
    result = lookup_fn(locations)

    assert result["area_peril_id"].to_numpy().tolist() == [1, 1]


def test_build_h3_non_default_index(h3_mapping_csv):
    """A non-contiguous input index is matched positionally and reset, as the left join used to."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    locations = pd.DataFrame({
        "loc_id": [1, 2, 3],
        "latitude": [lat for lat, lon in _SAMPLE_COORDS],
        "longitude": [lon for lat, lon in _SAMPLE_COORDS],
    }, index=[17, 4, 92])
    result = lookup_fn(locations)

    assert result["area_peril_id"].to_numpy().tolist() == [1, 2, 3]
    assert result.index.tolist() == [0, 1, 2]


def test_build_h3_row_count_and_order_preserved(h3_mapping_csv):
    """One row out per row in, in the input order."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    lat0, lon0 = _SAMPLE_COORDS[0]
    lat2, lon2 = _SAMPLE_COORDS[2]
    locations = pd.DataFrame({
        "loc_id": [5, 4, 3, 2, 1],
        "latitude": [lat2, None, lat0, -89.0, lat0],
        "longitude": [lon2, None, lon0, 179.0, lon0],
    })
    result = lookup_fn(locations)

    assert len(result) == 5
    assert result["loc_id"].to_numpy().tolist() == [5, 4, 3, 2, 1]
    assert result["area_peril_id"].to_numpy().tolist() == [3, OASIS_UNKNOWN_ID, 1, OASIS_UNKNOWN_ID, 1]


def test_build_h3_nullable_float_coordinates(h3_mapping_csv):
    """Nullable Float64 lat/lon columns (pd.NA rather than NaN) are handled."""
    lookup_fn = Lookup(config={}).build_h3(resolution=_H3_RESOLUTION, file_path=str(h3_mapping_csv))

    lat0, lon0 = _SAMPLE_COORDS[0]
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "latitude": pd.array([lat0, None], dtype="Float64"),
        "longitude": pd.array([lon0, None], dtype="Float64"),
    })
    result = lookup_fn(locations)

    assert result.loc[result["loc_id"] == 1, "area_peril_id"].iloc[0] == 1
    assert result.loc[result["loc_id"] == 2, "area_peril_id"].iloc[0] == OASIS_UNKNOWN_ID


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


def h3_lookup_config(mapping_path):
    """A whole Lookup config whose area_peril step is the h3 one.

    The pivot step supplies the peril, coverage and vulnerability ids process_locations needs, so
    the only thing under test is what the h3 step contributes.
    """
    return {
        "step_definition": {
            "pivot": {
                "type": "simple_pivot",
                "parameters": {"pivots": [{"new_cols": {
                    "peril_id": "WTC", "coverage_type": 1, "vulnerability_id": 1}}]},
            },
            "area_peril": {
                "type": "h3",
                "columns": ["latitude", "longitude"],
                "parameters": {"resolution": _H3_RESOLUTION, "file_path": str(mapping_path)},
            },
        },
        "strategy": ["area_peril", "pivot"],
    }


def test_process_locations_reports_a_matched_location_as_success(h3_mapping_csv):
    """The lookup contract, not just the closure: a matched location comes back successful."""
    lat0, lon0 = _SAMPLE_COORDS[0]
    locations = pd.DataFrame({"loc_id": [1], "latitude": [lat0], "longitude": [lon0]})

    keys = Lookup(config=h3_lookup_config(h3_mapping_csv)).process_locations(locations)

    assert keys["status"].to_list() == [OASIS_KEYS_STATUS["success"]["id"]]
    assert keys["area_peril_id"].to_list() == [1]
    assert keys["message"].to_list() == [""]


@pytest.mark.parametrize("latitude,longitude,description", [
    (-89.0, 179.0, "coordinates whose cell is absent from the mapping"),
    (None, None, "null coordinates"),
    (51.5074, np.inf, "an infinite longitude"),
    (np.inf, -0.1278, "an infinite latitude"),
    (-np.inf, -np.inf, "infinite coordinates"),
])
def test_process_locations_reports_an_unresolved_location_as_a_failure(
        h3_mapping_csv, latitude, longitude, description):
    """A location the lookup cannot resolve survives to the output as a per-location failure.

    This is the requirement downstream actually has, and it is what OASIS_UNKNOWN_ID exists to
    signal. It matters most for the infinite coordinates: they used to slip through the validity
    mask and take another location's area peril, so the location came back *successful* carrying a
    wrong id rather than being reported here.
    """
    locations = pd.DataFrame({"loc_id": [1], "latitude": [latitude], "longitude": [longitude]})

    keys = Lookup(config=h3_lookup_config(h3_mapping_csv)).process_locations(locations)

    assert keys["status"].to_list() == [OASIS_KEYS_STATUS["fail"]["id"]], description
    assert keys["message"].to_list() == ["area_peril_id has an unknown id"], description
    assert keys["area_peril_id"].to_list() == [OASIS_UNKNOWN_ID], description


def test_process_locations_keeps_every_location_and_its_order(h3_mapping_csv):
    """Failures are reported alongside the successes, not dropped from the keys output."""
    lat0, lon0 = _SAMPLE_COORDS[0]
    lat1, lon1 = _SAMPLE_COORDS[1]
    locations = pd.DataFrame({
        "loc_id": [1, 2, 3, 4],
        "latitude": [lat0, 51.5074, lat1, None],
        "longitude": [lon0, np.inf, lon1, None],
    })

    keys = Lookup(config=h3_lookup_config(h3_mapping_csv)).process_locations(locations)

    assert keys["loc_id"].to_list() == [1, 2, 3, 4]
    assert keys["status"].to_list() == [
        OASIS_KEYS_STATUS["success"]["id"], OASIS_KEYS_STATUS["fail"]["id"],
        OASIS_KEYS_STATUS["success"]["id"], OASIS_KEYS_STATUS["fail"]["id"],
    ]
    assert keys["area_peril_id"].to_list() == [1, OASIS_UNKNOWN_ID, 2, OASIS_UNKNOWN_ID]
