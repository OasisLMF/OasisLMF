import re
import sys
from collections import OrderedDict
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import pytest

from oasislmf.lookup.builtin import (
    Lookup,
    geotiff_read_plan,
    geotiff_value_dtype,
    PerilCoveredDeterministicLookup,
    create_lat_lon_id_functions,
    get_step,
    jit_geo_grid_lookup,
    load_sorted_lookup,
    normal_to_z_index,
    undo_z_index,
    z_index,
    z_index_to_normal,
    write_sorted_lookup,
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
    with pytest.warns(FutureWarning):
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


@pytest.fixture
def overlapping_sections_path(tmp_path):
    """Three overlapping squares; id 10 checks the ids are sorted as numbers, not as strings."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    gdf = gpd.GeoDataFrame(
        {"section_id": [2, 1, 10]},
        geometry=[box(1, 0, 3, 2), box(0, 0, 2, 2), box(1, 1, 3, 3)],
        crs="EPSG:4326",
    )
    path = tmp_path / "sections.parquet"
    gdf.to_parquet(path)
    return path.as_posix()


@pytest.fixture
def overlapping_sections_locations():
    return pd.DataFrame(columns=["longitude", "latitude", "locname"], data=[
        [0.5, 0.5, "one"],
        [1.5, 0.5, "two"],
        [1.5, 1.5, "three"],
        [5.0, 5.0, "none"],
        [None, None, "no_coordinates"],
    ])


def test_build_rtree_keeps_one_row_per_match(overlapping_sections_path, overlapping_sections_locations):
    rtree = Lookup(config={}).build_rtree(
        file_path=overlapping_sections_path, file_type="parquet", id_columns="section_id",
    )
    output = rtree(overlapping_sections_locations)
    assert sorted(output.loc[output["locname"] == "three", "section_id"]) == [1, 2, 10]
    assert len(output) == 1 + 2 + 3 + 1 + 1


@pytest.mark.parametrize("id_columns", [["section_id"], []], ids=["section_id an id column", "not listed"])
def test_process_locations_fails_a_location_in_no_section(overlapping_sections_path, overlapping_sections_locations, id_columns):
    """A location the section step cannot place is a keys error, not a success with no section."""
    config = {
        "step_definition": {
            "section": {
                "type": "rtree",
                "columns": ["latitude", "longitude"],
                "parameters": {"file_path": overlapping_sections_path, "file_type": "parquet", "id_columns": id_columns},
            },
            "pivot": {
                "type": "simple_pivot",
                "parameters": {"pivots": [{"new_cols": {
                    "peril_id": "WTC", "coverage_type": 1, "area_peril_id": 1, "vulnerability_id": 1}}]},
            },
        },
        "strategy": ["section", "pivot"],
    }
    locations = overlapping_sections_locations.assign(loc_id=range(1, 6))

    keys = Lookup(config=config).process_locations(locations)

    success = keys[keys["status"] == OASIS_KEYS_STATUS["success"]["id"]]
    assert sorted(success["section_id"]) == [1, 1, 1, 2, 2, 10]
    failed = keys[keys["status"] == OASIS_KEYS_STATUS["fail"]["id"]].set_index("loc_id")
    assert sorted(failed.index) == [4, 5]
    assert set(failed["message"]) == {"section_id has an unknown id"}


@pytest.mark.parametrize("preparations, values, expected", [
    ({"min": 5}, [1, 5, 7], [5, 5, 7]),           # values below min are raised to min
    ({"max": 10}, [7, 10, 20], [7, 10, 10]),      # values above max are lowered to max
    ({"min": 5, "max": 10}, [1, 7, 20], [5, 7, 10]),
])
def test_build_prepare_min_max_clamp(preparations, values, expected):
    prepare = Lookup(config={}).build_prepare(my_col=preparations)
    result = prepare(pd.DataFrame({"my_col": values}))
    assert result["my_col"].tolist() == expected


@pytest.mark.parametrize("preparations, values, expected", [
    ({"strip": True}, ["  ab ", "cd", None], ["ab", "cd", None]),
    ({"upper": True}, ["Ab", "cd", None], ["AB", "CD", None]),
    ({"lower": True}, ["Ab", "CD", None], ["ab", "cd", None]),
    ({"strip": True, "upper": True}, [" sw1a 1aa ", "Sw1A1aa"], ["SW1A 1AA", "SW1A1AA"]),
    ({"upper": True}, ["ab", 12, None], ["AB", 12, None]),   # non-text values in an object column are kept
    ({"strip": True, "type": "int64"}, [" 1", "2 "], [1, 2]),  # strip runs before type
])
def test_build_prepare_text_normalisation(preparations, values, expected):
    prepare = Lookup(config={}).build_prepare(my_col=preparations)
    result = prepare(pd.DataFrame({"my_col": values}, dtype=object))
    assert [None if pd.isna(v) else v for v in result["my_col"]] == expected


def test_build_prepare_text_normalisation_string_dtype_and_numeric():
    prepare = Lookup(config={}).build_prepare(s={"strip": True, "upper": True}, n={"upper": True})
    df = pd.DataFrame({"s": pd.array([" ab", None], dtype="string"), "n": [1, 2]})
    result = prepare(df)
    assert result["s"].tolist()[0] == "AB" and pd.isna(result["s"].tolist()[1])
    assert result["n"].tolist() == [1, 2]


@pytest.mark.parametrize("preparations, values, expected", [
    ({"strip": True, "upper": True}, [" ab", "cd ", None, "ab"], ["AB", "CD", None, "AB"]),
    # categories that collide once normalised ('ab' / ' AB') are merged
    ({"strip": True, "upper": True}, ["ab", " AB", "cd", None], ["AB", "AB", "CD", None]),
    ({"lower": True}, ["Ab", 1, None], ["ab", 1, None]),  # non-text categories are kept
])
def test_build_prepare_text_normalisation_category_dtype(preparations, values, expected):
    """OED string fields reach the lookup as ``category`` (ods_tools default_string_dtype)."""
    prepare = Lookup(config={}).build_prepare(my_col=preparations)
    result = prepare(pd.DataFrame({"my_col": pd.Series(values, dtype="category")}))
    assert isinstance(result["my_col"].dtype, pd.CategoricalDtype)
    assert [None if pd.isna(v) else v for v in result["my_col"]] == expected


def test_build_prepare_rename_and_drop():
    prepare = Lookup(config={}).build_prepare(
        tmp_id={"rename": "area_peril_id"},
        unused={"drop": True},
        not_there={"drop": True},  # dropping a missing column is fine
    )
    df = pd.DataFrame({"loc_id": [1, 2], "tmp_id": [5, 6], "area_peril_id": [0, 0], "unused": ["x", "y"]})
    result = prepare(df)
    assert list(result.columns) == ["loc_id", "area_peril_id"]
    assert result["area_peril_id"].tolist() == [5, 6]  # the existing column is replaced


def test_build_prepare_rename_then_prepare_new_name():
    prepare = Lookup(config={}).build_prepare(PostCode={"rename": "postalcode"}, postalcode={"upper": True})
    result = prepare(pd.DataFrame({"PostCode": ["ab1"]}))
    assert result.to_dict("list") == {"postalcode": ["AB1"]}


@pytest.mark.parametrize("kwargs, match", [
    ({"c": {"uper": True}}, "unknown option"),
    ({"c": {"upper": True, "lower": True}}, "both 'upper' and 'lower'"),
    ({"c": {"rename": ""}}, "non-empty column name"),
    ({"c": "upper"}, "must be a dict"),
])
def test_build_prepare_invalid_config(kwargs, match):
    with pytest.raises(OasisException, match=match):
        Lookup(config={}).build_prepare(**kwargs)


def test_build_prepare_missing_column_message():
    prepare = Lookup(config={}).build_prepare(postalcode={"upper": True})
    with pytest.raises(OasisException, match="'postalcode' is missing"):
        prepare(pd.DataFrame({"loc_id": [1]}))


def test_prepare_upper_strip_lets_merge_match_mixed_case(tmp_path):
    """Issue #2284 item 1: normalising a column with prepare lets merge match it against an upper-case table."""
    table = tmp_path / "postcode_areaperil.csv"
    pd.DataFrame({"postalcode": ["AB1", "CD2"], "area_peril_id": [11, 22]}).to_csv(table, index=False)
    config = {
        "strategy": ["normalise", "areaperil", "pivot"],
        "step_definition": {
            "normalise": {"type": "prepare", "columns": ["postalcode"],
                          "parameters": {"postalcode": {"strip": True, "upper": True}}},
            "areaperil": {"type": "merge", "columns": ["postalcode"],
                          "parameters": {"file_path": str(table), "id_columns": ["area_peril_id"]}},
            "pivot": {"type": "simple_pivot", "parameters": {"pivots": [{"new_cols": {
                "peril_id": "WTC", "coverage_type": 1, "vulnerability_id": 1}}]}},
        },
    }
    locations = pd.DataFrame({"loc_id": [1, 2, 3],
                              "PostalCode": pd.Series([" ab1", "Cd2 ", "zz9"], dtype="category")})
    keys = Lookup(config=config).process_locations(locations).set_index("loc_id")
    assert keys.loc[1, "area_peril_id"] == 11
    assert keys.loc[2, "area_peril_id"] == 22
    assert keys.loc[3, "status"] == OASIS_KEYS_STATUS["fail"]["id"]


@pytest.mark.parametrize("kwargs, status, counted_as_modelled", [
    ({}, "notmodelled", False),  # we don't know whether an unmodelled peril is a risk, so the summary must not count it
    ({"not_covered_status": "notatrisk"}, "notatrisk", True),  # the behaviour before 'notmodelled'
])
def test_split_loc_perils_covered_flags_perils_outside_the_model_with_not_covered_status(kwargs, status, counted_as_modelled):
    fct = Lookup(config={}).build_split_loc_perils_covered(model_perils_covered=["QEQ"], **kwargs)

    result = fct(pd.DataFrame({"loc_id": [1, 2], "LocPerilsCovered": ["QEQ", "WTC"]}))

    assert result[["loc_id", "peril_id", "status"]].fillna("").values.tolist() == [[1, "QEQ", ""], [2, "", status]]
    assert (status in OASIS_KEYS_STATUS_MODELLED) == counted_as_modelled


def test_split_loc_perils_covered_rejects_unknown_not_covered_status():
    with pytest.raises(OasisException):
        Lookup(config={}).build_split_loc_perils_covered(not_covered_status="not_a_real_status")


def test_split_loc_perils_covered_accepts_a_single_peril_as_a_string():
    """A lookup config can give model_perils_covered as one code rather than a list."""
    locations = pd.DataFrame({"loc_id": [1, 2], "LocPerilsCovered": ["WTC", "QEQ"]})

    from_string = Lookup(config={}).build_split_loc_perils_covered("WTC")(locations)
    from_list = Lookup(config={}).build_split_loc_perils_covered(["WTC"])(locations)

    pd.testing.assert_frame_equal(from_string, from_list)
    assert from_list[["loc_id", "peril_id"]].values.tolist() == [[1, "WTC"], [2, np.nan]]


@pytest.mark.parametrize("perils_covered, model_perils_covered, expected", [
    (["OO1;ORF;OSF"], None, [[1, "ORF", "OO1"], [1, "OSF", "OO1"]]),     # a group listed with the perils it contains
    (["ORF;OO1"], None, [[1, "ORF", "ORF"], [1, "OSF", "OO1"]]),         # the peril listed before its group
    (["WTC;XX1;AA1"], ["XHL"], [[1, "XHL", "XX1"]]),                     # two groups containing XHL
    (["XX1;AA1", "AA1;XX1"], ["XHL"], [[1, "XHL", "XX1"], [2, "XHL", "AA1"]]),  # each location keeps its own first group
], ids=["group with its perils", "peril before its group", "two groups", "per location"])
def test_split_loc_perils_covered_gives_each_peril_once_with_the_first_listed_group(perils_covered, model_perils_covered, expected):
    fct = Lookup(config={}).build_split_loc_perils_covered(model_perils_covered)

    result = fct(pd.DataFrame({"loc_id": range(1, len(perils_covered) + 1), "LocPerilsCovered": perils_covered}))

    result = result[result["peril_id"].notna()]  # leave out the keys error rows
    assert sorted(result[["loc_id", "peril_id", "peril_group_id"]].values.tolist()) == expected


def test_split_loc_perils_covered_keeps_location_rows_that_share_a_loc_id():
    """Rows sharing a loc_id but differing elsewhere keep their own perils; identical rows give them once,
    whatever the index."""
    fct = Lookup(config={}).build_split_loc_perils_covered(["ORF", "OSF", "WTC"])
    locations = pd.DataFrame({
        "loc_id": [1, 1, 2, 2],
        "BuildingTIV": [10., 20., 30., 30.],
        "LocPerilsCovered": ["OO1;ORF", "WTC", "ORF", "ORF"],
    }, index=[7, 3, 5, 9])

    result = fct(locations)

    assert result[["loc_id", "BuildingTIV", "peril_id"]].values.tolist() == [
        [1, 10., "ORF"], [2, 30., "ORF"], [1, 10., "OSF"], [1, 20., "WTC"]]  # sorted on peril_id by the model filter
    assert result.index.tolist() == list(range(4))
    assert list(result.columns) == ["loc_id", "BuildingTIV", "LocPerilsCovered", "peril_group_id", "peril_id"]
    assert list(locations.columns) == ["loc_id", "BuildingTIV", "LocPerilsCovered"]


@pytest.mark.parametrize("loc_ids, perils_covered, expected", [
    ([1], ["WW1"], {1: "WEC;WSS not modelled"}),                  # modelled in part: the rest is listed
    ([1, 2], ["QQ1;OO1", "WTC"], {1: "ORF;OSF;QEQ;QFF;QLF;QLS;QSL;QTS not modelled"}),  # nothing modelled, groups expanded
    ([1, 1], ["WTC", "QEQ"], {1: "QEQ not modelled"}),            # a row sharing a modelled location's loc_id
    ([1, 1, 1], ["WTC;QEQ", "OO1", "QEQ"], {1: "ORF;OSF;QEQ not modelled"}),  # rows of one location give one line
    ([1, 1, 2, 2], ["QEQ", "QEQ", "WTC", "WTC"], {1: "QEQ not modelled"}),    # identical rows give it once
    ([1], ["ZZZ"], {1: "ZZZ not modelled"}),                      # no known peril: the codes as listed
    ([1, 2, 3, 4], ["WTC", "", None, "  "], {2: "no perils covered", 3: "no perils covered", 4: "no perils covered"}),
    ([1, 1, 2, 2], ["", "WTC", "", "QEQ"], {2: "QEQ not modelled"}),  # a blank row adds nothing to its loc_id's other rows
], ids=["partial", "none", "shared loc_id", "union", "duplicates", "unknown codes", "blank", "blank sharing a loc_id"])
def test_split_loc_perils_covered_lists_the_perils_not_modelled_once_per_location(loc_ids, perils_covered, expected):
    fct = Lookup(config={}).build_split_loc_perils_covered(["WTC"])

    result = fct(pd.DataFrame({"loc_id": loc_ids, "LocPerilsCovered": perils_covered}))

    not_modelled = result[result["status"] == OASIS_KEYS_STATUS["notmodelled"]["id"]]
    assert not_modelled["peril_id"].isna().all()
    assert dict(zip(not_modelled["loc_id"], not_modelled["message"])) == expected
    assert len(not_modelled) == len(expected)


@pytest.mark.parametrize("dtype", [object, "string", "category"])  # ods_tools reads a blank cell as '' in a string column
def test_lookup_returns_every_location_in_keys_or_keys_errors(dtype):
    """Input file generation refuses a run where a loc_id is in neither keys nor keys-errors.
    The error line is repeated for each coverage, next to the keys of the perils modelled."""
    config = {
        "step_definition": {
            "split": {"type": "split_loc_perils_covered", "columns": ["LocPerilsCovered"],
                      "parameters": {"model_perils_covered": ["WTC", "WSS"]}},
            "pivot": {"type": "simple_pivot", "parameters": {"pivots": [
                {"new_cols": {"coverage_type": 1, "area_peril_id": 1, "vulnerability_id": 1}},
                {"new_cols": {"coverage_type": 3, "area_peril_id": 1, "vulnerability_id": 2}}]}},
        },
        "strategy": ["split", "pivot"],
    }
    perils_covered = ["WTC", "WW1", "QQ1", "ZZZ", "", None, "  ", "WTC", "", "QEQ", ""]
    loc_ids = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9]  # 8 and 9: a blank row sharing a loc_id

    keys = Lookup(config=config).process_locations(
        pd.DataFrame({"loc_id": loc_ids, "LocPerilsCovered": pd.Series(perils_covered, dtype=dtype)}))

    assert set(keys["loc_id"]) == set(loc_ids)
    success = keys["status"] == OASIS_KEYS_STATUS["success"]["id"]
    assert sorted(keys.loc[success & (keys["loc_id"] == 2), ["peril_id", "coverage_type"]].values.tolist()) == [
        ["WSS", 1], ["WSS", 3], ["WTC", 1], ["WTC", 3]]
    for coverage_type in (1, 3):
        errors = keys[~success & (keys["coverage_type"] == coverage_type)]
        assert dict(zip(errors["loc_id"], errors["message"])) == {
            2: "WEC not modelled", 3: "QEQ;QFF;QLF;QLS;QSL;QTS not modelled", 4: "ZZZ not modelled",
            5: "no perils covered", 6: "no perils covered", 7: "no perils covered", 9: "QEQ not modelled"}
        assert len(errors) == 7


def test_peril_covered_deterministic_lookup_gives_each_peril_once():
    """A group listed with the perils it contains (OO1;ORF;OSF) gives them once, like ORF;OSF."""
    lookup = PerilCoveredDeterministicLookup(config={
        "model_perils_covered": ["OO1"],
        "supported_oed_coverage_types": [1, 3],
    })
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "LocPerilsCovered": ["OO1;ORF;OSF", "ORF;OSF"],
    })

    keys_df = lookup.process_locations(locations)

    assert not keys_df.duplicated(["loc_id", "peril_id", "coverage_type"]).any()
    assert (keys_df["loc_id"].value_counts() == 4).all()  # 2 perils x 2 coverage types each
    assert "peril_group_id" not in locations


def test_peril_covered_deterministic_lookup_keeps_location_rows_that_share_a_loc_id():
    lookup = PerilCoveredDeterministicLookup(config={
        "model_perils_covered": ["WTC"],
        "supported_oed_coverage_types": [1],
    })
    locations = pd.DataFrame({
        "loc_id": [1, 1, 2, 2],
        "BuildingTIV": [10., 20., 30., 30.],
        "LocPerilsCovered": ["WTC", "WTC", "WW1;WTC", "WW1;WTC"],
    })

    keys_df = lookup.process_locations(locations)

    # loc 1: two distinct rows, one key each; loc 2: identical rows, WW1 = {WTC, WEC, WSS} given once
    assert keys_df[["loc_id", "peril_id"]].values.tolist() == [[1, "WTC"], [1, "WTC"], [2, "WEC"], [2, "WSS"], [2, "WTC"]]


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


def _affine(*args):
    """An affine transform, skipping the test when rasterio (which provides affine) is not installed."""
    pytest.importorskip("rasterio")
    from rasterio.transform import Affine
    return Affine(*args)


def _write_geotiff(path, data, transform, nodata=None, crs="EPSG:4326", **profile):
    """Write ``data`` (band, row, col) to a GeoTIFF at ``path`` with the given affine transform and nodata value."""
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
                       dtype=data.dtype, transform=transform, crs=crs, nodata=nodata, **profile) as dst:
        dst.write(data)
    return path


def _multi_band_geotiff(tmp_path):
    """3 bands of 2 rows x 3 cols, 1-degree pixels with the top-left corner at (lon -10, lat 50), north up.

    Pixel (row, col) of band b holds b * 100 + row * 10 + col, so every value names its band and pixel.
    """
    bands, rows, cols = np.meshgrid(np.arange(1, 4), np.arange(2), np.arange(3), indexing="ij")
    data = (bands * 100 + rows * 10 + cols).astype("int32")
    return _write_geotiff(tmp_path / "multi.tif", data, _affine(1.0, 0.0, -10.0, 0.0, -1.0, 50.0))


def test_build_geotiff_reads_each_column_from_its_band(tmp_path):
    """Each column reads its own band (permuted ids) on a real multi-band file, and out-of-range
    locations get each column's own default."""
    band_info = OrderedDict([
        ("a", {"id": 3, "default": -11}),
        ("b", {"id": 1, "default": -22}),
        ("c", {"id": 2, "default": -33}),
    ])
    geotiff_lookup = Lookup(config={}).build_geotiff(file_path=str(_multi_band_geotiff(tmp_path)), band_info=band_info)

    locations = pd.DataFrame({
        "longitude": [-7.5, 20.0],  # col 2 / outside
        "latitude": [48.5, 0.0],    # row 1 / outside
    })
    result = geotiff_lookup(locations)

    assert result[["a", "b", "c"]].values.tolist() == [[312, 112, 212], [-11, -22, -33]]


@pytest.mark.parametrize("lon, lat, expected", [
    (-9.9, 49.9, 100),    # just inside the top-left corner
    (-9.3, 49.3, 100),    # 0.7 of a pixel in: still pixel (0, 0)
    (-8.5, 49.5, 101),    # centre of pixel (0, 1)
    (-7.01, 48.01, 112),  # just inside the bottom-right corner
    (-10.3, 49.5, -1),    # 0.3 of a pixel west of the raster
    (-9.5, 50.3, -1),     # 0.3 of a pixel north of the raster
    (-6.9, 49.5, -1),     # just east of the raster
    (np.nan, 49.5, -1),   # missing coordinates
    (-9.5, np.inf, -1),
])
def test_build_geotiff_assigns_the_pixel_containing_each_location(tmp_path, lon, lat, expected):
    geotiff_lookup = Lookup(config={}).build_geotiff(file_path=str(_multi_band_geotiff(tmp_path)),
                                                     band_info={"value": {"id": 1, "default": -1}})

    result = geotiff_lookup(pd.DataFrame({"longitude": [lon], "latitude": [lat]}))

    assert result["value"].tolist() == [expected]


@pytest.mark.parametrize("band_dtypes, defaults, expected", [
    (["uint8"], [0], "uint8"),           # fits: dtype unchanged
    (["int16"], [0.0], "int16"),         # integral float fits an integer band
    (["uint8"], [-1], "int16"),          # negative default on an unsigned band
    (["int16"], [np.nan], "float32"),    # NaN default on an integer band
    (["int16"], [1.5], "float32"),       # fractional default on an integer band
    (["uint8"], [1000], "uint16"),       # default above the band's range
    (["float32"], [np.nan], "float32"),  # NaN already fits a float band
    (["uint8", "int16"], [0, 0], "int16"),
])
def test_geotiff_value_dtype(band_dtypes, defaults, expected):
    defaults = {f"col_{i}": default for i, default in enumerate(defaults)}
    assert geotiff_value_dtype(band_dtypes, defaults) == np.dtype(expected)


@pytest.mark.parametrize("default", ["none", None, 2 ** 70])
def test_geotiff_value_dtype_rejects_defaults_that_are_not_numbers(default):
    with pytest.raises(OasisException, match="geotiff band flood_depth has default"):
        geotiff_value_dtype(["float32"], {"flood_depth": default})


def test_build_geotiff_default_outside_band_dtype_widens_column(tmp_path):
    """A -1 default on a uint8 band used to raise OverflowError; the column now widens to hold it."""
    data = np.array([[[7, 255]]], dtype="uint8")
    path = _write_geotiff(tmp_path / "uint8.tif", data, _affine(1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
    geotiff_lookup = Lookup(config={}).build_geotiff(file_path=str(path), band_info={"value": {"id": 1, "default": -1}})

    result = geotiff_lookup(pd.DataFrame({"longitude": [0.5, 1.5, 5.0], "latitude": [0.5, 0.5, 0.5]}))

    assert result["value"].dtype == np.dtype("int16")
    assert result["value"].tolist() == [7, 255, -1]


def test_build_geotiff_accepts_nullable_coordinates(tmp_path):
    """pd.NA in nullable Float64 coordinates is treated as missing and gets the default."""
    geotiff_lookup = Lookup(config={}).build_geotiff(file_path=str(_multi_band_geotiff(tmp_path)),
                                                     band_info={"value": {"id": 1, "default": -1}})
    locations = pd.DataFrame({
        "longitude": pd.array([-9.5, pd.NA, -8.5], dtype="Float64"),
        "latitude": pd.array([49.5, 49.5, pd.NA], dtype="Float64"),
    })

    result = geotiff_lookup(locations)

    assert result["value"].tolist() == [100, -1, -1]


def test_build_geotiff_budget_bounded_reads_match_single_read(tmp_path, monkeypatch):
    """Reads split to fit a tiny GEOTIFF_READ_BUDGET_BYTES give the same values as one read."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1000, size=(2, 50, 40)).astype("int16")
    path = _write_geotiff(tmp_path / "big.tif", data, _affine(0.1, 0.0, 0.0, 0.0, -0.1, 5.0))
    band_info = {"first": {"id": 1, "default": -1}, "second": {"id": 2, "default": -2}}
    locations = pd.DataFrame({
        "longitude": rng.uniform(-0.5, 4.5, size=500),
        "latitude": rng.uniform(-0.5, 5.5, size=500),
    })

    single = Lookup(config={}).build_geotiff(file_path=str(path), band_info=band_info)(locations.copy())
    # 2 int16 bands cost 12 bytes per pixel (value, value before conversion, two masks): allow 30 pixels per read,
    # less than one 40-column row, so the file's single block is split along both axes
    monkeypatch.setattr("oasislmf.lookup.builtin.GEOTIFF_READ_BUDGET_BYTES", 30 * 12)
    strips = Lookup(config={}).build_geotiff(file_path=str(path), band_info=band_info)(locations.copy())

    pd.testing.assert_frame_equal(single, strips)
    # and both match indexing the raster directly
    col = np.floor(locations["longitude"].to_numpy() / 0.1)
    row = np.floor((5.0 - locations["latitude"].to_numpy()) / 0.1)
    inside = (col >= 0) & (col < 40) & (row >= 0) & (row < 50)
    expected = np.where(inside, data[0, row.clip(0, 49).astype(int), col.clip(0, 39).astype(int)], -1)
    assert single["first"].tolist() == expected.tolist()


@pytest.mark.parametrize("dtype, nodata", [("int16", -9999), ("float32", np.nan), ("float32", -3.4e38)])
def test_build_geotiff_replaces_nodata_with_band_default(tmp_path, dtype, nodata):
    """Pixels holding the raster's nodata value take the column's default; other pixels keep their value,
    and a value equal to another column's default is not mistaken for nodata."""
    # 1 band, 1 row, 3 cols: valid, nodata, valid
    data = np.array([[[7, nodata, 9]]], dtype=dtype)
    path = _write_geotiff(tmp_path / "nodata.tif", data, _affine(1.0, 0.0, 0.0, 0.0, -1.0, 1.0), nodata=nodata)
    band_info = OrderedDict([("first", {"id": 1, "default": -1}), ("again", {"id": 1, "default": -2})])
    geotiff_lookup = Lookup(config={}).build_geotiff(file_path=str(path), band_info=band_info)

    result = geotiff_lookup(pd.DataFrame({"longitude": [0.5, 1.5, 2.5, 5.0], "latitude": [0.5] * 4}))

    assert result["first"].tolist() == [7, -1, 9, -1]
    assert result["again"].tolist() == [7, -2, 9, -2]


def test_build_geotiff_without_nodata_keeps_every_value(tmp_path):
    """A raster with no nodata value has no invalid pixels, even ones holding a typical nodata number."""
    data = np.array([[[7, -9999]]], dtype="int16")
    path = _write_geotiff(tmp_path / "no_nodata.tif", data, _affine(1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
    geotiff_lookup = Lookup(config={}).build_geotiff(file_path=str(path), band_info={"value": {"id": 1, "default": -1}})

    result = geotiff_lookup(pd.DataFrame({"longitude": [0.5, 1.5], "latitude": [0.5, 0.5]}))

    assert result["value"].tolist() == [7, -9999]


def _plan_reads(rows, cols, block_shape=(16, 16), bytes_per_pixel=1, height=64, width=64, budget=10 ** 6):
    """geotiff_read_plan as ((row_off, col_off, height, width), sorted locations) per read, for readable asserts."""
    pytest.importorskip("rasterio")
    rows, cols = np.array(rows), np.array(cols)
    order, reads = geotiff_read_plan(rows, cols, block_shape, bytes_per_pixel, height, width, budget)
    return [((w.row_off, w.col_off, w.height, w.width), list(zip(rows[order][s:e].tolist(), cols[order][s:e].tolist())))
            for s, e, w in reads]


def test_geotiff_read_plan_merges_adjacent_blocks_and_skips_gaps():
    # blocks (0, 0) and (0, 1) are adjacent, (0, 3) is past a gap, (2, 2) is in another block row
    reads = _plan_reads(rows=[40, 3, 5, 1], cols=[40, 20, 1, 50])
    assert reads == [
        ((0, 0, 16, 32), [(5, 1), (3, 20)]),
        ((0, 48, 16, 16), [(1, 50)]),
        ((32, 32, 16, 16), [(40, 40)]),
    ]


def test_geotiff_read_plan_splits_runs_to_fit_the_budget():
    # four adjacent blocks of 16 x 16 pixels at 1 byte each, with room for two blocks per read
    reads = _plan_reads(rows=[0, 0, 0, 0], cols=[0, 16, 32, 48], budget=2 * 256)
    assert [window for window, _ in reads] == [(0, 0, 16, 32), (0, 32, 16, 32)]


def test_geotiff_read_plan_splits_a_block_larger_than_the_budget():
    # a file stored as one 64 x 64 block, with room for 100 pixels per read: tiles shrink to single 64-pixel rows
    reads = _plan_reads(rows=[0, 63], cols=[0, 63], block_shape=(64, 64), budget=100)
    assert [window for window, _ in reads] == [(0, 0, 1, 64), (63, 0, 1, 64)]
    # and a single row too wide for the budget is cut into columns
    reads = _plan_reads(rows=[0, 0], cols=[0, 63], block_shape=(1, 64), budget=10)
    assert [window for window, _ in reads] == [(0, 0, 1, 10), (0, 60, 1, 4)]


def test_geotiff_read_plan_without_locations_reads_nothing():
    assert _plan_reads(rows=[], cols=[]) == []


def test_geotiff_read_plan_clips_edge_blocks_to_the_raster():
    reads = _plan_reads(rows=[59], cols=[59], height=60, width=60)
    assert reads == [((48, 48, 12, 12), [(59, 59)])]


def test_build_geotiff_reads_only_the_blocks_holding_locations(tmp_path, monkeypatch):
    """On a tiled raster each read covers whole blocks that hold locations, never the blocks between them."""
    rasterio = pytest.importorskip("rasterio")
    data = np.arange(64 * 64, dtype="int32").reshape(1, 64, 64)
    path = _write_geotiff(tmp_path / "tiled.tif", data, _affine(1.0, 0.0, 0.0, 0.0, -1.0, 64.0),
                          tiled=True, blockxsize=16, blockysize=16)
    windows = []
    real_read = rasterio.io.DatasetReader.read

    def spy_read(self, *args, **kwargs):
        windows.append(kwargs["window"])
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(rasterio.io.DatasetReader, "read", spy_read)
    geotiff_lookup = Lookup(config={}).build_geotiff(file_path=str(path), band_info={"value": {"id": 1, "default": -1}})
    # pixel (row, col) is at lon = col + 0.5, lat = 64 - row - 0.5; pixels in two far-apart blocks
    rows, cols = np.array([2, 60, 5]), np.array([3, 61, 9])
    result = geotiff_lookup(pd.DataFrame({"longitude": cols + 0.5, "latitude": 64 - rows - 0.5}))

    assert result["value"].tolist() == (rows * 64 + cols).tolist()
    assert sorted((w.row_off, w.col_off, w.height, w.width) for w in windows) == [(0, 0, 16, 16), (48, 48, 16, 16)]


def test_build_geotiff_rejects_projected_crs(tmp_path):
    """A raster in metres (Web Mercator) cannot be indexed by longitude/latitude, so it is rejected at build time."""
    data = np.zeros((1, 2, 2), dtype="int16")
    path = _write_geotiff(tmp_path / "mercator.tif", data, _affine(1000.0, 0.0, 0.0, 0.0, -1000.0, 2000.0),
                          crs="EPSG:3857")

    with pytest.raises(OasisException, match="projected CRS EPSG:3857"):
        Lookup(config={}).build_geotiff(file_path=str(path), band_info={"value": {"id": 1, "default": -1}})


@pytest.mark.parametrize("crs", ["EPSG:4269", None])  # geographic NAD83, and no CRS at all
def test_build_geotiff_accepts_geographic_or_missing_crs(tmp_path, crs):
    data = np.array([[[7]]], dtype="int16")
    path = _write_geotiff(tmp_path / "geographic.tif", data, _affine(1.0, 0.0, 0.0, 0.0, -1.0, 1.0), crs=crs)
    geotiff_lookup = Lookup(config={}).build_geotiff(file_path=str(path), band_info={"value": {"id": 1, "default": -1}})

    result = geotiff_lookup(pd.DataFrame({"longitude": [0.5], "latitude": [0.5]}))

    assert result["value"].tolist() == [7]


def test_build_geotiff_rejects_band_id_outside_file(tmp_path):
    with pytest.raises(OasisException, match=r"outside of \[1-3\]"):
        Lookup(config={}).build_geotiff(file_path=str(_multi_band_geotiff(tmp_path)),
                                        band_info={"value": {"id": 4, "default": -1}})


@pytest.fixture
def builtin_without_spatial_packages(monkeypatch):
    """A fresh copy of the builtin module imported with the spatial packages blocked.

    The copy runs in this process, so coverage counts its import fallbacks; the real module is left untouched.
    """
    import importlib.util
    from oasislmf.lookup import builtin

    for name in ("shapely", "shapely.geometry", "geopandas", "sklearn", "sklearn.neighbors",
                 "rasterio", "rasterio.windows"):
        monkeypatch.setitem(sys.modules, name, None)  # makes any import of it raise ImportError
    spec = importlib.util.spec_from_file_location("builtin_without_spatial_packages", builtin.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builtin_imports_without_spatial_packages(builtin_without_spatial_packages):
    """Without the spatial packages the module still imports and every spatial name is None."""
    module = builtin_without_spatial_packages
    assert module.Point is None and module.gpd is None and module.BallTree is None
    assert module.rasterio is None and module.Window is None


@pytest.mark.parametrize("build_method, kwargs", [
    ("build_rtree", {"file_path": "unused.parquet", "file_type": "parquet", "id_columns": []}),
    ("build_geotiff", {"file_path": "unused.tif", "band_info": {"value": {"id": 1, "default": -1}}}),
])
def test_spatial_lookups_without_spatial_packages_point_at_extra(builtin_without_spatial_packages, build_method, kwargs):
    lookup = builtin_without_spatial_packages.Lookup(config={})

    with pytest.raises(OasisException, match=re.escape("oasislmf[spatial]")):
        getattr(lookup, build_method)(**kwargs)


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


def _geog_process_locations_config(table, step_columns=()):
    return {
        "strategy": ["resolve", "areaperil", "pivot"],
        "step_definition": {
            "resolve": {"type": "geog_lookup", "columns": list(step_columns),
                        "parameters": {"geog_scheme": "W3W", "output_column": "w3w", "slots": 3}},
            "areaperil": {"type": "merge",
                          "parameters": {"file_path": str(table), "id_columns": ["area_peril_id"],
                                         "join_columns": ["w3w"]}},
            "pivot": {"type": "simple_pivot", "parameters": {"pivots": [{"new_cols": {
                "peril_id": "WTC", "coverage_type": 1, "vulnerability_id": 1}}]}},
        },
    }


def test_geog_lookup_slots_kept_without_listing_them(tmp_path):
    """Issue #2284 item 3: geog_lookup sees the GeogScheme/GeogName slots through process_locations
    without them in 'columns', and slots missing from the file don't fail the run."""
    table = tmp_path / "w3w_areaperil.csv"
    pd.DataFrame({"w3w": ["a.b.c", "d.e.f"], "area_peril_id": [101, 102]}).to_csv(table, index=False)
    locations = pd.DataFrame({  # only slots 1 and 2 of the 3 scanned are present
        "loc_id": [1, 2, 3],
        "GeogScheme1": ["W3W", "ISO2", None], "GeogName1": ["a.b.c", "US", None],
        "GeogScheme2": [None, "W3W", None], "GeogName2": [None, "d.e.f", None],
    })
    keys = Lookup(config=_geog_process_locations_config(table)).process_locations(locations).set_index("loc_id")
    assert keys.loc[1, "area_peril_id"] == 101
    assert keys.loc[2, "area_peril_id"] == 102
    assert keys.loc[3, "status"] == OASIS_KEYS_STATUS["fail"]["id"]


def test_geog_lookup_slot_listed_in_another_step_keeps_its_spelling(tmp_path):
    """A slot named by some step's 'columns' keeps that spelling, so that step's required check passes."""
    table = tmp_path / "w3w_areaperil.csv"
    pd.DataFrame({"w3w": ["a.b.c"], "area_peril_id": [101]}).to_csv(table, index=False)
    config = _geog_process_locations_config(table)
    config["step_definition"]["areaperil"]["columns"] = ["GeogName1"]
    locations = pd.DataFrame({"loc_id": [1], "geogscheme1": ["W3W"], "GEOGNAME1": ["a.b.c"]})
    keys = Lookup(config=config).process_locations(locations)
    assert keys["area_peril_id"].tolist() == [101]


def test_listed_geog_slot_is_still_required(tmp_path):
    table = tmp_path / "w3w_areaperil.csv"
    pd.DataFrame({"w3w": ["a.b.c"], "area_peril_id": [101]}).to_csv(table, index=False)
    config = _geog_process_locations_config(table, step_columns=["GeogScheme3", "GeogName3"])
    locations = pd.DataFrame({"loc_id": [1], "GeogScheme1": ["W3W"], "GeogName1": ["a.b.c"]})
    with pytest.raises(OasisException, match="missing columns"):
        Lookup(config=config).process_locations(locations)


@pytest.mark.parametrize("has_coordinates", [True, False])
def test_optional_columns_kept_if_present_and_not_required(has_coordinates):
    """Issue #2284 item 3: optional_columns are kept through the column filter but don't fail the step."""
    config = {
        "strategy": ["coordinates", "pivot"],
        "step_definition": {
            "coordinates": {"type": "prepare", "optional_columns": ["latitude", "longitude"],
                            "parameters": {"latitude": {"default": -999}, "longitude": {"default": -999}}},
            "pivot": {"type": "simple_pivot", "parameters": {"pivots": [{"new_cols": {
                "peril_id": "WTC", "coverage_type": 1, "area_peril_id": 1, "vulnerability_id": 1}}]}},
        },
    }
    seen = {}

    class SpyLookup(Lookup):
        def build_simple_pivot(self, pivots, remove_pivoted_col=True):
            inner = Lookup.build_simple_pivot(pivots, remove_pivoted_col)

            def fct(locations):
                seen["lat"] = locations["latitude"].tolist()
                return inner(locations)
            return fct

    locations = pd.DataFrame({"loc_id": [1], "unused": ["x"]})
    if has_coordinates:
        locations = locations.assign(LATITUDE=[51.5], Longitude=[-0.1])
    SpyLookup(config=config).process_locations(locations)
    assert seen["lat"] == ([51.5] if has_coordinates else [-999])


def _two_step_config(columns_a, columns_b, key_a="columns", key_b="columns"):
    return {
        "strategy": ["a", "b", "pivot"],
        "step_definition": {
            "a": {"type": "prepare", key_a: columns_a, "parameters": {}},
            "b": {"type": "prepare", key_b: columns_b, "parameters": {}},
            "pivot": {"type": "simple_pivot", "parameters": {"pivots": [{"new_cols": {
                "peril_id": "WTC", "coverage_type": 1, "area_peril_id": 1, "vulnerability_id": 1}}]}},
        },
    }


@pytest.mark.parametrize("key_a, key_b", [
    ("optional_columns", "optional_columns"),
    ("columns", "columns"),
    ("columns", "optional_columns"),
])
def test_column_named_with_different_capitalisation_raises(key_a, key_b):
    """Each column is kept under one spelling, so steps must agree on it (review of #2322)."""
    config = _two_step_config(["Latitude"], ["latitude"], key_a, key_b)
    locations = pd.DataFrame({"loc_id": [1], "LATITUDE": [51.5]})
    with pytest.raises(OasisException, match=r"different capitalisation: 'Latitude' in \['a'\], 'latitude' in \['b'\]"):
        Lookup(config=config).process_locations(locations)


def test_column_named_with_same_spelling_in_several_steps():
    config = _two_step_config(["Latitude"], ["Latitude"], "columns", "optional_columns")
    locations = pd.DataFrame({"loc_id": [1], "LATITUDE": [51.5]})
    assert Lookup(config=config).useful_columns() == {"loc_id", "Latitude"}
    Lookup(config=config).process_locations(locations)


def test_geog_lookup_slot_takes_spelling_of_optional_column(tmp_path):
    """The automatic geog_lookup slots never clash: they take the spelling another step lists."""
    table = tmp_path / "w3w_areaperil.csv"
    pd.DataFrame({"w3w": ["a.b.c"], "area_peril_id": [101]}).to_csv(table, index=False)
    config = _geog_process_locations_config(table)
    config["step_definition"]["areaperil"]["optional_columns"] = ["GeogName1"]
    locations = pd.DataFrame({"loc_id": [1], "geogscheme1": ["W3W"], "GEOGNAME1": ["a.b.c"]})
    assert "GeogName1" in Lookup(config=config).useful_columns()
    keys = Lookup(config=config).process_locations(locations)
    assert keys["area_peril_id"].tolist() == [101]


def test_step_optional_columns():
    assert Lookup.step_optional_columns({"type": "merge"}) == []
    assert Lookup.step_optional_columns({"type": "merge", "optional_columns": "latitude"}) == ["latitude"]
    assert Lookup.step_optional_columns({"type": "geog_lookup", "parameters": {"slots": 2}}) == [
        "geogscheme1", "geogname1", "geogscheme2", "geogname2"]


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


def _vulnerability_merge_config(vulnerability_dict_path, join_columns=None):
    parameters = {"file_path": str(vulnerability_dict_path), "file_type": "csv", "id_columns": ["vulnerability_id"]}
    if join_columns is not None:
        parameters["join_columns"] = join_columns
    return {
        "strategy": ["areaperil", "vulnerability"],
        "step_definition": {
            "areaperil": {
                "type": "prepare",
                "columns": ["countrycode"],
                "parameters": {"area_peril_id": {"default": 1}},
            },
            "vulnerability": {
                "type": "merge",
                "columns": ["peril_id", "coverage_type", "occupancycode"],
                "parameters": parameters,
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


def test_process_locations_merge_without_join_columns_joins_on_every_shared_column(vulnerability_dict_path, german_locations):
    """By default the merge also joins on 'countrycode' (needed by the 'areaperil' step),
    so the FR dict matches none of the DE locations."""
    config = _vulnerability_merge_config(vulnerability_dict_path)
    result = Lookup(config=config).process_locations(german_locations)
    assert (result["vulnerability_id"] == OASIS_UNKNOWN_ID).all()


def test_process_locations_merge_join_columns_ignores_other_shared_columns(vulnerability_dict_path, german_locations):
    """With join_columns, the dict file's leftover 'countrycode' is not a join key and the merge succeeds."""
    config = _vulnerability_merge_config(vulnerability_dict_path, join_columns=["peril_id", "coverage_type", "occupancycode"])
    result = Lookup(config=config).process_locations(german_locations)
    assert result.sort_values("loc_id")["vulnerability_id"].tolist() == [10, 30]
    assert (result["area_peril_id"] == 1).all()
    assert (result["status"] == OASIS_KEYS_STATUS["success"]["id"]).all()


def test_merge_join_columns_keeps_locations_value_for_other_shared_columns(vulnerability_dict_path, german_locations):
    """A non-key column in both the file and the locations keeps its locations value (no _x/_y suffixes),
    and join_columns are matched case-insensitively."""
    merge = Lookup(config={}).build_merge(file_path=str(vulnerability_dict_path), id_columns=["vulnerability_id"],
                                          join_columns=["Peril_ID", "Coverage_Type", "OccupancyCode"])
    result = merge(german_locations.rename(columns={"countrycode": "CountryCode"}))
    assert result["CountryCode"].tolist() == ["DE", "DE"]
    assert "countrycode" not in result.columns
    assert result["vulnerability_id"].tolist() == [10, 30]


def test_merge_join_columns_takes_id_columns_from_file(vulnerability_dict_path, german_locations):
    """An id column already in the locations (e.g. reset to -1 by an earlier step) is replaced by the file's value."""
    merge = Lookup(config={}).build_merge(file_path=str(vulnerability_dict_path), id_columns=["vulnerability_id"],
                                          join_columns=["peril_id", "coverage_type", "occupancycode"])
    result = merge(german_locations.assign(vulnerability_id=OASIS_UNKNOWN_ID))
    assert result["vulnerability_id"].tolist() == [10, 30]
    assert not any(col.endswith(("_x", "_y")) for col in result.columns)


def test_merge_join_columns_missing_from_file_raises(vulnerability_dict_path):
    with pytest.raises(OasisException, match="not found in"):
        Lookup(config={}).build_merge(file_path=str(vulnerability_dict_path), join_columns=["postalcode"])


def test_merge_join_columns_missing_from_locations_raises(vulnerability_dict_path, german_locations):
    merge = Lookup(config={}).build_merge(file_path=str(vulnerability_dict_path), join_columns=["peril_id", "countrycode"])
    with pytest.raises(OasisException, match="not found in the locations"):
        merge(german_locations.drop(columns="countrycode"))


def test_combine_skips_child_whose_own_columns_are_missing(caplog):
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
    with caplog.at_level("WARNING", logger="oasislmf.lookup.builtin"):
        result = PostcodeLookup(config=config).process_locations(locations)
    assert result["vulnerability_id"].tolist() == [7]
    assert "skipping step by_postcode" in caplog.text
    assert (result["status"] == OASIS_KEYS_STATUS["success"]["id"]).all()


def test_combine_and_raises_when_child_columns_are_missing():
    """Unlike 'or', every child of an 'and' combine is required, so a child whose columns
    are absent must raise rather than be skipped and let the location succeed without it."""
    class AutoLookup(Lookup):
        def build_by_autocode(self):
            return lambda locations: locations.assign(
                coverage_ok=locations["autocode"].map({1: 1}).fillna(OASIS_UNKNOWN_ID).astype(int))

    config = {
        "strategy": ["areaperil", "vulnerability"],
        "step_definition": {
            "areaperil": {"type": "prepare", "parameters": {"area_peril_id": {"default": 1}}},
            "vulnerability": {
                "type": "combine",
                "columns": ["peril_id", "coverage_type"],
                "parameters": {
                    "logical_type": "and",
                    "id_columns": [["coverage_ok"], ["vulnerability_id"]],
                    "strategy": ["by_autocode", "default_vuln"],
                },
            },
            "by_autocode": {"type": "by_autocode", "columns": ["autocode"], "parameters": {}},
            "default_vuln": {"type": "prepare", "columns": ["peril_id"], "parameters": {"vulnerability_id": {"default": 7}}},
        },
    }
    locations = pd.DataFrame({"loc_id": [1], "peril_id": ["WTC"], "coverage_type": [1]})  # no AutoCode
    with pytest.raises(OasisException, match="by_autocode"):
        AutoLookup(config=config).process_locations(locations)


# --- sorted_lookup (#2284 item 6) ---------------------------------------------

@pytest.fixture
def cell_table():
    # deliberately unsorted, with a key beyond 2**53 to check integer keys stay exact
    return pd.DataFrame({
        "cell_id": np.array([30, 10, 2**60 + 1, 20], dtype="int64"),
        "area_peril_id": np.array([3, 1, 9, 2], dtype="int32"),
        "weight": [0.3, 0.1, 0.9, 0.2],
    })


def test_write_sorted_lookup_writes_sorted_arrays(cell_table, tmp_path):
    out = write_sorted_lookup(cell_table, tmp_path / "cells", key_column="cell_id")
    keys, values, key_column = load_sorted_lookup(str(out))
    assert key_column == "cell_id"
    assert isinstance(keys, np.memmap)
    assert keys.tolist() == [10, 20, 30, 2**60 + 1]
    assert values["area_peril_id"].tolist() == [1, 2, 3, 9]
    assert values["weight"].tolist() == [0.1, 0.2, 0.3, 0.9]


@pytest.mark.parametrize("file_type", ["csv", "parquet"])
def test_write_sorted_lookup_from_file(cell_table, tmp_path, file_type):
    path = tmp_path / f"table.{file_type}"
    getattr(cell_table, f"to_{file_type}")(path, index=False)
    out = write_sorted_lookup(str(path), tmp_path / "cells", key_column="cell_id", value_columns=["area_peril_id"])
    _, values, _ = load_sorted_lookup(str(out))
    assert list(values) == ["area_peril_id"]


@pytest.mark.parametrize("table, match", [
    (pd.DataFrame({"cell_id": [1, 1], "v": [1, 2]}), "duplicated"),
    (pd.DataFrame({"cell_id": [1.0, None], "v": [1, 2]}), "integers with no nulls"),
    (pd.DataFrame({"cell_id": [1, 2], "v": ["a", "b"]}), "must be numeric"),
    (pd.DataFrame({"cell_id": [1, 2], "v": [1.0, None]}), "has nulls"),
])
def test_write_sorted_lookup_rejects_bad_tables(table, match, tmp_path):
    with pytest.raises(OasisException, match=match):
        write_sorted_lookup(table, tmp_path / "out", key_column="cell_id")


def test_sorted_lookup_matches(cell_table, tmp_path):
    write_sorted_lookup(cell_table, tmp_path / "cells", key_column="cell_id")
    fct = Lookup(config={}).build_sorted_lookup(file_path=str(tmp_path / "cells"), id_columns=["area_peril_id"])
    locations = pd.DataFrame({
        "loc_id": [1, 2, 3, 4, 5, 6],
        "cell_id": pd.array([20, 15, None, 2**60 + 1, 99, 10], dtype="Int64"),
    })
    result = fct(locations)
    assert result["area_peril_id"].tolist() == [2, OASIS_UNKNOWN_ID, OASIS_UNKNOWN_ID, 9, OASIS_UNKNOWN_ID, 1]
    assert result["weight"].iloc[[0, 3, 5]].tolist() == [0.2, 0.9, 0.1]
    assert result["weight"].iloc[[1, 2, 4]].isna().all()
    assert result["loc_id"].tolist() == [1, 2, 3, 4, 5, 6]  # order and rows unchanged


@pytest.mark.parametrize("keys", [
    [20.0, 10.5, np.nan, -1.0],          # float keys: non-integral and NaN don't match
    ["20", "x", None, "-1"],             # text keys are parsed
])
def test_sorted_lookup_non_integer_keys(cell_table, tmp_path, keys):
    write_sorted_lookup(cell_table, tmp_path / "cells", key_column="cell_id", value_columns=["area_peril_id"])
    fct = Lookup(config={}).build_sorted_lookup(file_path=str(tmp_path / "cells"), id_columns="area_peril_id")
    result = fct(pd.DataFrame({"cell_id": pd.Series(keys, dtype=object if isinstance(keys[0], str) else float)}))
    assert result["area_peril_id"].tolist() == [2, OASIS_UNKNOWN_ID, OASIS_UNKNOWN_ID, OASIS_UNKNOWN_ID]


def test_sorted_lookup_errors(cell_table, tmp_path):
    with pytest.raises(OasisException, match="not found"):
        Lookup(config={}).build_sorted_lookup(file_path=str(tmp_path / "nothing"))
    write_sorted_lookup(cell_table, tmp_path / "cells", key_column="cell_id")
    with pytest.raises(OasisException, match="not in"):
        Lookup(config={}).build_sorted_lookup(file_path=str(tmp_path / "cells"), value_columns=["nope"])
    with pytest.raises(OasisException, match="id_columns"):
        Lookup(config={}).build_sorted_lookup(file_path=str(tmp_path / "cells"), id_columns=["vulnerability_id"])
    fct = Lookup(config={}).build_sorted_lookup(file_path=str(tmp_path / "cells"))
    with pytest.raises(OasisException, match="key column 'cell_id'"):
        fct(pd.DataFrame({"loc_id": [1]}))


def test_sorted_lookup_end_to_end(cell_table, tmp_path):
    write_sorted_lookup(cell_table, tmp_path / "cells", key_column="cell_id", value_columns=["area_peril_id"])
    config = {
        "keys_data_path": str(tmp_path),
        "strategy": ["areaperil", "pivot"],
        "step_definition": {
            "areaperil": {"type": "sorted_lookup", "columns": ["cell_id"],
                          "parameters": {"file_path": "%%KEYS_DATA_PATH%%/cells", "id_columns": ["area_peril_id"]}},
            "pivot": {"type": "simple_pivot", "parameters": {"pivots": [{"new_cols": {
                "peril_id": "ORF", "coverage_type": 1, "vulnerability_id": 1}}]}},
        },
    }
    locations = pd.DataFrame({"loc_id": [1, 2], "CELL_ID": [30, 31]})
    keys = Lookup(config=config).process_locations(locations).set_index("loc_id")
    assert (keys.loc[1, "status"], keys.loc[1, "area_peril_id"]) == ("success", 3)
    assert keys.loc[2, "status"] == OASIS_KEYS_STATUS["fail"]["id"]


# --- transform_coordinates (#2284 item 5) -------------------------------------


def _bng_locations():
    return pd.DataFrame({
        "loc_id": [1, 2, 3, 4],
        "latitude": [51.5072, None, -33.87, 55.9533],      # London, missing, Sydney, Edinburgh
        "longitude": [-0.1276, -0.1, 151.21, -3.1883],
    })


def test_transform_coordinates_to_british_national_grid():
    pytest.importorskip("pyproj")
    fct = Lookup(config={}).build_transform_coordinates(target_crs="EPSG:27700")
    result = fct(_bng_locations())
    # PROJ uses the OSTN15 grid when it is installed (x = 530041.5) and a Helmert fallback
    # otherwise (x = 530043.2), so allow a few metres either way
    assert result["x"].iloc[0] == pytest.approx(530042, abs=5)
    assert result["y"].iloc[0] == pytest.approx(180358, abs=5)
    assert result["x"].iloc[3] == pytest.approx(326000, abs=1000)
    assert result[["x", "y"]].iloc[[1, 2]].isna().all().all()  # missing input, and Sydney is outside GB


def test_transform_coordinates_without_area_check_and_custom_columns():
    pytest.importorskip("pyproj")
    fct = Lookup(config={}).build_transform_coordinates(
        target_crs=27700, x_column="easting", y_column="northing", area_of_use_only=False)
    result = fct(_bng_locations())
    assert result["easting"].notna().tolist() == [True, False, True, True]  # Sydney projected, however meaningless


def test_transform_coordinates_area_crossing_antimeridian():
    pytest.importorskip("pyproj")
    fct = Lookup(config={}).build_transform_coordinates(target_crs="EPSG:3832")  # PDC Mercator, 98.69E..68W
    result = fct(pd.DataFrame({"latitude": [-17.7, -17.7, 51.5], "longitude": [178.0, -150.0, 0.0]}))
    assert result["x"].notna().tolist() == [True, True, False]


def test_transform_coordinates_failure_messages():
    pytest.importorskip("pyproj")
    fct = Lookup(config={}).build_transform_coordinates(target_crs="EPSG:27700")
    locations = _bng_locations()
    locations["message"] = ["", "", "", "already set"]
    result = fct(locations)
    assert result["message"].tolist() == [
        "",
        "transform_coordinates failure: missing latitude or longitude",
        "transform_coordinates failure: outside the area of use of EPSG:27700",
        "already set",  # an earlier step's message is kept
    ]


def test_transform_coordinates_area_of_use_ignored_is_logged(caplog):
    pytest.importorskip("pyproj")
    with caplog.at_level("INFO", logger="oasislmf.lookup.builtin"):
        Lookup(config={}).build_transform_coordinates(target_crs="EPSG:4326", source_crs="EPSG:27700")
    assert "area_of_use_only ignored, source CRS EPSG:27700 is not geographic" in caplog.text


def test_transform_coordinates_errors():
    pytest.importorskip("pyproj")
    with pytest.raises(OasisException, match="invalid CRS"):
        Lookup(config={}).build_transform_coordinates(target_crs="EPSG:not-a-crs")
    fct = Lookup(config={}).build_transform_coordinates(target_crs="EPSG:27700")
    with pytest.raises(OasisException, match="'latitude' is not in the locations"):
        fct(pd.DataFrame({"lat": [51.5], "longitude": [0.0]}))


def test_transform_coordinates_then_interval_to_index_end_to_end(tmp_path):
    """Issue #2284 item 5: project to a CRS, then index a regular grid in it with interval_to_index."""
    pytest.importorskip("pyproj")
    edges = [500000, 510000, 520000, 530000, 540000]  # 10 km columns of a BNG grid
    config = {
        "strategy": ["project", "column", "areaperil", "pivot"],
        "step_definition": {
            "project": {"type": "transform_coordinates", "columns": ["latitude", "longitude"],
                        "parameters": {"target_crs": "EPSG:27700"}},
            "column": {"type": "interval_to_index",
                       "parameters": {"value_column_name": "x", "sorted_array": edges,
                                      "index_column_name": "area_peril_id"}},
            "areaperil": {"type": "prepare", "parameters": {"area_peril_id": {"default": -1}}},
            "pivot": {"type": "simple_pivot", "parameters": {"pivots": [{"new_cols": {
                "peril_id": "ORF", "coverage_type": 1, "vulnerability_id": 1}}]}},
        },
    }
    locations = pd.DataFrame({"loc_id": [1, 2], "Latitude": [51.5072, -33.87], "Longitude": [-0.1276, 151.21]})
    keys = Lookup(config=config).process_locations(locations).set_index("loc_id")
    assert (keys.loc[1, "status"], keys.loc[1, "area_peril_id"]) == ("success", 4)  # x ~ 530042
    assert keys.loc[2, "status"] == OASIS_KEYS_STATUS["fail"]["id"]
    assert keys.loc[2, "message"] == ("area_peril_id has an unknown id: "
                                      "transform_coordinates failure: outside the area of use of EPSG:27700")
