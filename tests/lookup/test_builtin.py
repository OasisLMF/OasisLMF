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
from oasislmf.utils.exceptions import OasisException
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


@pytest.mark.parametrize("target_crs", ["EPSG:3857", "EPSG:2154"])
def test_build_rtree_reprojects_non_4326_geometries(target_crs, rtree_locations_all_coordinates, tmp_path):
    """Test that geometries in non-EPSG:4326 CRS (e.g. EPSG:3857 or state plane)
    are automatically reprojected to EPSG:4326 and properly joined to produce risk matches."""
    gpd = pytest.importorskip("geopandas")
    gdf_original = gpd.read_parquet(FILES_DIR / "rtree_areas.parquet")
    gdf_reprojected = gdf_original.to_crs(target_crs)
    temp_file = tmp_path / f"rtree_areas_{target_crs.replace(':', '_')}.parquet"
    gdf_reprojected.to_parquet(temp_file)

    rtree = Lookup(config={}).build_rtree(
        file_path=temp_file.as_posix(),
        file_type="parquet",
        id_columns="poly_id",
        nearest_neighbor_max_distance=12000,
    )
    output = rtree(rtree_locations_all_coordinates)
    expected = rtree_locations_all_coordinates.copy().assign(poly_id=[1, 2, 1, OASIS_UNKNOWN_ID])

    # Verify that non-empty risk matches are produced
    assert (output["poly_id"] != OASIS_UNKNOWN_ID).any()
    assert (output["poly_id"] == 1).any()
    assert (output["poly_id"] == 2).any()

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


def test_split_loc_perils_covered_marks_not_modelled():
    """A location whose perils are outside the model's scope is flagged 'not modelled'
    (not 'not at risk' — we don't know whether an unmodelled peril is a risk or not). The
    status must be the string id, not the whole OASIS_KEYS_STATUS dict, and it must be
    excluded from the 'modelled' set used by the exposure summary report."""
    fct = Lookup(config={}).build_split_loc_perils_covered(model_perils_covered=["QEQ"])
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "LocPerilsCovered": ["QEQ", "WTC"],   # loc 2's peril is outside the model
    })

    result = fct(locations)

    not_modelled = result[result["loc_id"] == 2]
    assert len(not_modelled) == 1
    status = not_modelled["status"].iloc[0]
    assert status == OASIS_KEYS_STATUS["notmodelled"]["id"]
    assert isinstance(status, str)
    assert not not_modelled["status"].isin(OASIS_KEYS_STATUS_MODELLED).any()


def test_split_loc_perils_covered_not_covered_status_restores_previous_behaviour():
    """not_covered_status lets a model developer opt back into the pre-'notmodelled' behaviour
    of flagging uncovered perils as 'notatrisk'."""
    fct = Lookup(config={}).build_split_loc_perils_covered(model_perils_covered=["QEQ"], not_covered_status="notatrisk")
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "LocPerilsCovered": ["QEQ", "WTC"],
    })

    result = fct(locations)

    not_covered = result[result["loc_id"] == 2]
    assert len(not_covered) == 1
    assert not_covered["status"].iloc[0] == OASIS_KEYS_STATUS["notatrisk"]["id"]


def test_split_loc_perils_covered_rejects_unknown_not_covered_status():
    with pytest.raises(OasisException):
        Lookup(config={}).build_split_loc_perils_covered(not_covered_status="not_a_real_status")


def test_build_set_status_sets_status_and_message_from_columns():
    fct = Lookup(config={}).build_set_status(status_column="custom_status", message_column="custom_message")
    locations = pd.DataFrame({
        "loc_id": [1, 2, 3],
        "status": ["success", "success", "success"],
        "message": ["", "", ""],
        "custom_status": ["notatrisk", "notmodelled", None],
        "custom_message": ["outside flood zone", "peril not modelled", None],
    })

    result = fct(locations)

    assert result["status"].tolist() == ["notatrisk", "notmodelled", "success"]
    assert result["message"].tolist() == ["outside flood zone", "peril not modelled", ""]


def test_build_set_status_rejects_unknown_status_value():
    fct = Lookup(config={}).build_set_status(status_column="custom_status")
    locations = pd.DataFrame({
        "loc_id": [1],
        "status": ["success"],
        "message": [""],
        "custom_status": ["not_a_real_status"],
    })

    with pytest.raises(OasisException):
        fct(locations)


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


# --- geog_lookup (GeogScheme/GeogName resolution) -----------------------------

@pytest.fixture
def geog_locations():
    """Locations whose 'W3W' value sits in different GeogName slots per row."""
    return pd.DataFrame({
        "loc_id": [1, 2, 3],
        "GeogScheme1": ["W3W", "ISO2", "W3W"],
        "GeogName1": ["filled.count.soap", "US", "index.home.raft"],
        "GeogScheme2": ["ISO2", "W3W", "CRESTA"],
        "GeogName2": ["GB", "table.chair.lamp", "12"],
    })


def test_geog_lookup_resolves_across_slots(geog_locations):
    fct = Lookup(config={}).build_geog_lookup(geog_scheme="W3W", output_column="w3w", slots=2)
    result = fct(geog_locations)
    assert list(result["w3w"]) == ["filled.count.soap", "table.chair.lamp", "index.home.raft"]


def test_geog_lookup_first_match_wins():
    locations = pd.DataFrame({
        "loc_id": [1],
        "GeogScheme1": ["W3W"], "GeogName1": ["from.slot.one"],
        "GeogScheme2": ["W3W"], "GeogName2": ["from.slot.two"],
    })
    fct = Lookup(config={}).build_geog_lookup(geog_scheme="W3W", output_column="w3w", slots=2)
    assert list(fct(locations)["w3w"]) == ["from.slot.one"]


def test_geog_lookup_case_and_whitespace():
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "GeogScheme1": ["w3w", " W3W "], "GeogName1": ["lower.case.match", "padded.match"],
    })
    fct = Lookup(config={}).build_geog_lookup(geog_scheme="W3W", output_column="w3w", slots=1)
    assert list(fct(locations)["w3w"]) == ["lower.case.match", "padded.match"]


@pytest.mark.parametrize("scheme_col,name_col", [
    ("geogscheme1", "geogname1"),
    ("GEOGSCHEME1", "GEOGNAME1"),
])
def test_geog_lookup_column_name_case_insensitive(scheme_col, name_col):
    """process_locations renames columns to the spelling used in the step's columns list."""
    locations = pd.DataFrame({"loc_id": [1], scheme_col: ["W3W"], name_col: ["any.case.match"]})
    fct = Lookup(config={}).build_geog_lookup(geog_scheme="W3W", output_column="w3w", slots=1)
    assert fct(locations)["w3w"].tolist() == ["any.case.match"]


def test_geog_lookup_missing_scheme_null(geog_locations):
    fct = Lookup(config={}).build_geog_lookup(geog_scheme="XYZ", output_column="xyz", slots=2)
    assert fct(geog_locations)["xyz"].isna().all()


def test_geog_lookup_missing_scheme_error(geog_locations):
    fct = Lookup(config={}).build_geog_lookup(
        geog_scheme="W3W", output_column="w3w", slots=2, on_missing="error")
    ok = fct(geog_locations.copy())  # all three rows have a W3W slot -> no raise
    assert ok["w3w"].notna().all()

    no_w3w = pd.DataFrame({"loc_id": [1], "GeogScheme1": ["ISO2"], "GeogName1": ["US"]})
    with pytest.raises(OasisException):
        fct(no_w3w)


def test_geog_lookup_absent_slots_ignored():
    """Config asks for 3 slots but only 1 pair is present -> no KeyError."""
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "GeogScheme1": ["W3W", "ISO2"], "GeogName1": ["present.slot.one", "US"],
    })
    fct = Lookup(config={}).build_geog_lookup(geog_scheme="W3W", output_column="w3w", slots=3)
    assert list(fct(locations)["w3w"]) == ["present.slot.one", pd.NA]


def test_geog_lookup_rejects_bad_on_missing():
    with pytest.raises(OasisException):
        Lookup(config={}).build_geog_lookup(geog_scheme="W3W", output_column="w3w", on_missing="boom")


def test_geog_lookup_then_merge_end_to_end(geog_locations, tmp_path):
    table = tmp_path / "w3w_areaperil.csv"
    pd.DataFrame({
        "w3w": ["filled.count.soap", "table.chair.lamp", "index.home.raft"],
        "area_peril_id": [101, 102, 103],
    }).to_csv(table, index=False)

    lookup = Lookup(config={})
    resolve = lookup.build_geog_lookup(geog_scheme="W3W", output_column="w3w", slots=2)
    merge = lookup.build_merge(file_path=str(table), id_columns=["area_peril_id"])

    result = merge(resolve(geog_locations))
    assert list(result.sort_values("loc_id")["area_peril_id"]) == [101, 102, 103]


def test_merge_empty_join_raises_clear_error(tmp_path):
    """build_merge against a table sharing no column raises a clear OasisException."""
    table = tmp_path / "unrelated.csv"
    pd.DataFrame({"w3w": ["a.b.c"], "area_peril_id": [1]}).to_csv(table, index=False)

    merge = Lookup(config={}).build_merge(file_path=str(table), id_columns=["area_peril_id"])
    locations = pd.DataFrame({"loc_id": [1], "GeogScheme1": ["W3W"], "GeogName1": ["a.b.c"]})
    with pytest.raises(OasisException, match="nothing to join on"):
        merge(locations)


def test_geog_lookup_sparse_and_null_slots():
    """Real OED has mostly-empty GeogScheme slots (NaN); resolution must not raise
    and must pick the filled slot per row regardless of which one it is."""
    locations = pd.DataFrame({
        "loc_id": [1, 2, 3],
        "GeogScheme1": ["W3W", None, "ISO2"],
        "GeogName1": ["a.b.c", None, "US"],
        "GeogScheme2": [None, "W3W", None],
        "GeogName2": [None, "d.e.f", None],
    })
    fct = Lookup(config={}).build_geog_lookup(geog_scheme="W3W", output_column="w3w", slots=2)
    result = fct(locations)
    assert result["w3w"].tolist()[:2] == ["a.b.c", "d.e.f"]
    assert pd.isna(result["w3w"].iloc[2])


@pytest.fixture
def vulnerability_dict_path(tmp_path):
    """A vulnerability dict reused from a French model: it still carries the FR
    'countrycode' it was built with, which German locations won't match on."""
    path = tmp_path / "vulnerability_dict.csv"
    pd.DataFrame({
        "peril_id": ["WTC", "WTC"],
        "coverage_type": [1, 3],
        "occupancycode": [1000, 1000],
        "countrycode": ["FR", "FR"],
        "vulnerability_id": [10, 30],
    }).to_csv(path, index=False)
    return path


def _strict_columns_config(vulnerability_dict_path, strict_columns, strategy=("areaperil", "vulnerability")):
    return {
        "strategy": list(strategy),
        "step_definition": {
            "areaperil": {
                "type": "prepare",
                "columns": ["countrycode"],
                "parameters": {"area_peril_id": {"default": 1}},
            },
            "vulnerability": {
                "type": "merge",
                "columns": ["peril_id", "coverage_type", "occupancycode"],
                "strict_columns": strict_columns,
                "parameters": {
                    "file_path": str(vulnerability_dict_path),
                    "file_type": "csv",
                    "id_columns": ["vulnerability_id"],
                },
            },
        },
    }


@pytest.fixture
def german_locations():
    return pd.DataFrame({
        "loc_id": [1, 2],
        "peril_id": ["WTC", "WTC"],
        "coverage_type": [1, 3],
        "occupancycode": [1000, 1000],
        "countrycode": ["DE", "DE"],
    })


def test_process_locations_without_strict_columns_leaks_columns_across_steps(vulnerability_dict_path, german_locations):
    """Without strict_columns, the merge step also sees 'countrycode' (needed by the
    'areaperil' step), so pandas.merge implicitly joins on it too and nothing matches."""
    config = _strict_columns_config(vulnerability_dict_path, strict_columns=False)
    result = Lookup(config=config).process_locations(german_locations)
    assert (result["vulnerability_id"] == OASIS_UNKNOWN_ID).all()


def test_process_locations_strict_columns_scopes_step_to_its_own_columns(vulnerability_dict_path, german_locations):
    """With strict_columns, the merge step only sees the columns it declared, so the dict
    file's leftover 'countrycode' is never used as an implicit join key and the merge succeeds."""
    config = _strict_columns_config(vulnerability_dict_path, strict_columns=True)
    result = Lookup(config=config).process_locations(german_locations)
    assert result.sort_values("loc_id")["vulnerability_id"].tolist() == [10, 30]
    # the 'areaperil' step still had access to its own declared column
    assert (result["area_peril_id"] == 1).all()
    assert (result["status"] == OASIS_KEYS_STATUS["success"]["id"]).all()


def test_process_locations_strict_columns_restores_hidden_columns_for_later_steps(vulnerability_dict_path, german_locations):
    """The dict file's own 'countrycode' (FR) must not clash with the hidden locations
    'countrycode' (DE) when reassembling, so a later step can still use the original column."""
    config = _strict_columns_config(vulnerability_dict_path, strict_columns=True, strategy=["vulnerability", "areaperil"])
    lookup = Lookup(config=config)
    seen = {}
    areaperil = lookup.set_step_function("areaperil", config["step_definition"]["areaperil"])

    def spy(locations):
        seen["countrycode"] = locations["countrycode"].tolist()
        return areaperil(locations)
    lookup.areaperil = spy

    result = lookup.process_locations(german_locations)
    assert seen["countrycode"] == ["DE", "DE"]
    assert result.sort_values("loc_id")["vulnerability_id"].tolist() == [10, 30]
    assert (result["area_peril_id"] == 1).all()


def test_process_locations_strict_columns_applies_to_combine_children(vulnerability_dict_path, german_locations):
    config = _strict_columns_config(vulnerability_dict_path, strict_columns=True)
    config["step_definition"]["vulnerability_fr_dict"] = config["step_definition"].pop("vulnerability")
    config["step_definition"]["vulnerability"] = {
        "type": "combine",
        "columns": ["peril_id", "coverage_type", "occupancycode"],
        "parameters": {"id_columns": ["vulnerability_id"], "strategy": ["vulnerability_fr_dict"]},
    }
    result = Lookup(config=config).process_locations(german_locations)
    assert result.sort_values("loc_id")["vulnerability_id"].tolist() == [10, 30]
    assert (result["status"] == OASIS_KEYS_STATUS["success"]["id"]).all()


def test_combine_skips_child_whose_own_columns_are_missing():
    """A combine child is skipped when the columns *it* declares are absent from the
    exposure, so the next child can act as a fallback (rather than the child crashing)."""
    class PostcodeLookup(Lookup):
        def build_by_postcode(self):
            return lambda locations: locations.assign(
                vulnerability_id=locations["postalcode"].map({"AB1": 5}).fillna(OASIS_UNKNOWN_ID).astype(int))

    config = {
        "strategy": ["areaperil", "vulnerability"],
        "step_definition": {
            "areaperil": {"type": "prepare", "parameters": {"area_peril_id": {"default": 1}}},
            "vulnerability": {
                "type": "combine",
                "columns": ["peril_id", "coverage_type"],
                "parameters": {"id_columns": ["vulnerability_id"], "strategy": ["by_postcode", "fallback"]},
            },
            "by_postcode": {"type": "by_postcode", "columns": ["postalcode"], "parameters": {}},
            "fallback": {"type": "prepare", "columns": ["peril_id"], "parameters": {"vulnerability_id": {"default": 7}}},
        },
    }
    locations = pd.DataFrame({"loc_id": [1], "peril_id": ["WTC"], "coverage_type": [1]})  # no PostalCode
    result = PostcodeLookup(config=config).process_locations(locations)
    assert result["vulnerability_id"].tolist() == [7]
    assert (result["status"] == OASIS_KEYS_STATUS["success"]["id"]).all()
