import copy
from collections import OrderedDict
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import pytest

from oasislmf.lookup.builtin import (
    Lookup,
    PerilCoveredDeterministicLookup,
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


def _overlapping_geometries():
    """Overlapping squares, so some locations are in several geometries, with extra columns."""
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    return gpd.GeoDataFrame(
        {"poly_id": [1, 2, 3, 4], "factor": [0.1, 0.2, 0.3, 0.4], "label": ["a", "b", "c", "d"]},
        geometry=[shapely.box(0, 0, 2, 2), shapely.box(1, 1, 3, 3), shapely.box(0.5, 0.5, 1.5, 1.5),
                  shapely.box(5, 5, 6, 6)],
        crs="EPSG:4326",
    )


def test_build_rtree_matches_sjoin_with_overlapping_geometries(tmp_path):
    """A location in several geometries gets one row per geometry, in the order and with the
    columns geopandas.sjoin(how='left') gives them."""
    gpd = pytest.importorskip("geopandas")
    gdf = _overlapping_geometries()
    gdf.to_parquet(tmp_path / "areas.parquet")
    locations = pd.DataFrame({
        "loc_id": [10, 11, 12, 13, 14],
        "longitude": [1.2, 2.5, 4.0, 5.5, 0.1],
        "latitude": [1.2, 2.5, 4.0, 5.5, 0.1],
    }, index=[7, 3, 9, 1, 5])

    rtree = Lookup(config={}).build_rtree(
        file_path=(tmp_path / "areas.parquet").as_posix(), file_type="parquet", id_columns="poly_id")
    output = rtree(locations)

    gdf_loc = gpd.GeoDataFrame(locations, geometry=gpd.points_from_xy(locations.longitude, locations.latitude),
                               crs="EPSG:4326")
    expected = gpd.sjoin(gdf_loc, gdf, how="left").drop(columns=["geometry", "index_right"])
    expected = Lookup.set_id_columns(pd.DataFrame(expected), "poly_id")

    pd.testing.assert_frame_equal(output, expected)
    assert sorted(output.loc[7, "poly_id"]) == [1, 2, 3]
    assert output.loc[9, "poly_id"] == OASIS_UNKNOWN_ID


def _count_parquet_reads(monkeypatch):
    from oasislmf.lookup import builtin
    reads = []
    read_parquet = builtin.gpd.read_parquet

    def counting_read_parquet(*args, **kwargs):
        reads.append(args[0])
        return read_parquet(*args, **kwargs)
    monkeypatch.setattr(builtin.gpd, "read_parquet", counting_read_parquet)
    return reads


def _rtree_lookup_config(file_path):
    return {"step_definition": {"peril": {
        "type": "rtree",
        "columns": ["longitude", "latitude"],
        "parameters": {"file_path": file_path, "file_type": "parquet", "id_columns": ["poly_id"],
                       "nearest_neighbor_max_distance": 12000},
    }}, "strategy": ["peril"]}


def test_multiproc_shared_data_loads_rtree_file_once(monkeypatch, rtree_locations_all_coordinates):
    """Lookups built while the parent's shared data is loaded (as forked workers are) reuse it
    instead of reading the geometry file again, and it is released when the context exits."""
    from oasislmf.lookup.builtin import _SHARED_STEP_DATA
    reads = _count_parquet_reads(monkeypatch)
    config = _rtree_lookup_config((FILES_DIR / "rtree_areas.parquet").as_posix())
    params = config["step_definition"]["peril"]["parameters"]

    with Lookup(config=config).multiproc_shared_data():
        assert len(reads) == 1
        assert len(_SHARED_STEP_DATA) == 1
        for _ in range(2):  # e.g. two workers
            output = Lookup(config=config).build_rtree(**params)(rtree_locations_all_coordinates)
            assert output["poly_id"].tolist() == [1, 2, 1, OASIS_UNKNOWN_ID]
        assert len(reads) == 1

    assert not _SHARED_STEP_DATA
    Lookup(config=config).build_rtree(**params)
    assert len(reads) == 2


def test_multiproc_shared_data_not_used_for_edited_file(monkeypatch, tmp_path, rtree_locations_all_coordinates):
    """Shared data identifies the file's version, so an edited file is read again."""
    gpd = pytest.importorskip("geopandas")
    file_path = tmp_path / "areas.parquet"
    gdf = gpd.read_parquet(FILES_DIR / "rtree_areas.parquet")
    gdf.to_parquet(file_path)
    reads = _count_parquet_reads(monkeypatch)
    config = _rtree_lookup_config(file_path.as_posix())
    params = config["step_definition"]["peril"]["parameters"]

    with Lookup(config=config).multiproc_shared_data():
        gdf.assign(poly_id=gdf["poly_id"] + 10).to_parquet(file_path)
        output = Lookup(config=config).build_rtree(**params)(rtree_locations_all_coordinates)

    assert len(reads) == 2
    assert output["poly_id"].tolist() == [11, 12, 11, OASIS_UNKNOWN_ID]


def test_multiproc_shared_data_skips_overridden_build_rtree(monkeypatch):
    """A subclass with its own build_rtree may not use the data, so none is loaded for it."""
    reads = _count_parquet_reads(monkeypatch)

    class CustomLookup(Lookup):
        def build_rtree(self, **kwargs):
            return lambda locations: locations

    config = _rtree_lookup_config((FILES_DIR / "rtree_areas.parquet").as_posix())
    with CustomLookup(config=config).multiproc_shared_data():
        pass
    assert not reads


def test_strategy_step_names_follows_combine_steps():
    """The steps run are those in 'strategy' and, recursively, in its 'combine' steps; a combine
    step that is defined but not used doesn't count, and a built combine step (whose strategy
    holds {'name': ..., 'function': ...}) is followed too."""
    config = {
        "step_definition": {
            "a": {"type": "merge"},
            "b": {"type": "rtree"},
            "c": {"type": "rtree"},
            "d": {"type": "rtree"},
            "e": {"type": "merge"},
            "f": {"type": "merge"},
            "comb": {"type": "combine", "parameters": {"strategy": ["b", "comb2"]}},
            "comb2": {"type": "combine", "parameters": {"strategy": [{"name": "c", "function": None}, "comb"]}},
            "unused_comb": {"type": "combine", "parameters": {"strategy": ["e"]}},
        },
        "strategy": ["a", "comb"],
    }
    assert Lookup(config=config)._strategy_step_names() == {"a", "comb", "b", "comb2", "c"}


def test_multiproc_shared_data_skips_steps_not_in_strategy(monkeypatch, rtree_locations_all_coordinates):
    """Only the files of steps the strategy runs, directly or through a 'combine' step, are
    loaded: the unused rtree and merge steps' files don't exist, so loading them would fail."""
    from oasislmf.lookup.builtin import _SHARED_STEP_DATA
    reads = _count_parquet_reads(monkeypatch)
    used_file = (FILES_DIR / "rtree_areas.parquet").as_posix()
    rtree_step = _rtree_lookup_config(used_file)["step_definition"]["peril"]
    unused_step = {**rtree_step, "parameters": {**rtree_step["parameters"], "file_path": "does_not_exist.parquet"}}
    config = {
        "step_definition": {
            "unused_peril": unused_step,
            "unused_vulnerability": {"type": "merge", "columns": ["peril_id"],
                                     "parameters": {"file_path": "does_not_exist.csv", "id_columns": ["vulnerability_id"]}},
            "used_peril": rtree_step,
            "peril": {"type": "combine", "parameters": {"strategy": ["used_peril"], "id_columns": ["poly_id"]}},
            "unused_combine": {"type": "combine", "parameters": {"strategy": ["unused_peril"]}},
        },
        "strategy": ["peril"],
    }

    with Lookup(config=config).multiproc_shared_data():
        assert reads == [used_file]
        assert len(_SHARED_STEP_DATA) == 1


@pytest.fixture
def merge_table_csv(tmp_path):
    """A merge table with a duplicated key (two rows for WTC/1/1050), a NaN key and mixed-case
    column names."""
    path = tmp_path / "vulnerability.csv"
    pd.DataFrame({
        "Peril_ID": ["WTC", "WTC", "WTC", "ORF", "ORF"],
        "coverage_type": [1, 1, 3, 1, 1],
        "OccupancyCode": [1050, 1050, 1050, 1100, None],
        "vulnerability_id": [11, 12, 13, 21, 22],
    }).to_csv(path, index=False)
    return path


def _whole_table_merge(path, locations, id_columns):
    """What build_merge did before it looked up the matching rows: merge with the whole table."""
    table = pd.read_csv(path)
    table.columns = [col.lower() for col in table.columns]
    rename_map = {col.lower(): col for col in locations.columns if col.lower() in table.columns}
    return Lookup.set_id_columns(locations.merge(table.rename(columns=rename_map), how="left"), id_columns)


@pytest.mark.parametrize("peril_dtype", ["str", "category"])
def test_build_merge_matches_whole_table_merge(merge_table_csv, peril_dtype):
    """Merging each chunk with only the rows it can match gives the whole-table merge result:
    one row per match for duplicated keys, NaN keys matched, and the same dtypes, including
    for a chunk that matches nothing (pandas types a categorical key differently when merged
    with an empty table) and an empty chunk."""
    locations = pd.DataFrame({
        "loc_id": [1, 2, 3, 4, 5],
        "peril_id": ["WTC", "ORF", "WTC", "ORF", "XXX"],
        "coverage_type": [1, 1, 3, 1, 1],
        "occupancycode": [1050, 1100, 1050, None, 1050],
    }, index=[40, 10, 30, 20, 50]).astype({"peril_id": peril_dtype})
    merge = Lookup(config={}).build_merge(file_path=str(merge_table_csv), id_columns=["vulnerability_id"])

    for chunk in (locations, locations.iloc[4:], locations.iloc[:0]):
        pd.testing.assert_frame_equal(merge(chunk), _whole_table_merge(merge_table_csv, chunk, ["vulnerability_id"]))
    assert merge(locations)["vulnerability_id"].tolist() == [11, 12, 21, 13, 22, OASIS_UNKNOWN_ID]


def test_build_merge_join_columns_with_matching_rows(merge_table_csv):
    """With join_columns, only those columns are looked up; a shared id column takes the file's value."""
    locations = pd.DataFrame({"peril_id": ["WTC", "ORF"], "coverage_type": [3, 1], "occupancycode": [1050, 1100],
                              "vulnerability_id": [99, 99]})
    merge = Lookup(config={}).build_merge(file_path=str(merge_table_csv), id_columns=["vulnerability_id"],
                                          join_columns=["Peril_ID", "coverage_type"])
    output = merge(locations)
    assert output["vulnerability_id"].tolist() == [13, 21, 22]
    assert output["occupancycode"].tolist() == [1050, 1100, 1100]


def _count_csv_reads(monkeypatch):
    reads = []
    read_csv = pd.read_csv

    def counting_read_csv(*args, **kwargs):
        reads.append(args[0])
        return read_csv(*args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", counting_read_csv)
    return reads


def _merge_lookup_config(file_path):
    return {"step_definition": {"vulnerability": {
        "type": "merge",
        "columns": ["peril_id", "coverage_type", "occupancycode"],
        "parameters": {"file_path": file_path, "id_columns": ["vulnerability_id"]},
    }}, "strategy": ["vulnerability"]}


def _merge_after_prepare_config(file_path):
    """A merge step that joins on peril_id, which an earlier step adds, and that the step's columns don't list
    (as when an rtree step adds area_peril_id), inside a combine step."""
    return {"step_definition": {
        "set_peril": {"type": "prepare", "parameters": {"peril_id": {"default": "WTC"}, "area_peril_id": {"default": 1}}},
        "vulnerability": {"type": "combine", "columns": [],
                          "parameters": {"id_columns": ["vulnerability_id"], "strategy": ["vulnerability_merge"]}},
        "vulnerability_merge": {
            "type": "merge",
            "columns": ["coverage_type", "occupancycode"],
            "parameters": {"file_path": file_path, "id_columns": ["vulnerability_id"]},
        },
    }, "strategy": ["set_peril", "vulnerability"]}


def test_multiproc_shared_data_loads_merge_file_once(monkeypatch, merge_table_csv):
    """Lookups built while the parent's shared data is loaded (as forked workers are) reuse its
    table and its index, built by running the lookup on the sample locations on the columns the
    step joins on, even one added by an earlier step; it is released when the context exits."""
    from oasislmf.lookup.builtin import _SHARED_STEP_DATA
    reads = _count_csv_reads(monkeypatch)
    config = _merge_after_prepare_config(str(merge_table_csv))
    parent = Lookup(config=config)
    config_before = copy.deepcopy(config)
    locations = pd.DataFrame({"loc_id": [1, 2], "coverage_type": [3, 1], "occupancycode": [1050, 1050]})

    with parent.multiproc_shared_data(sample_locations=locations.head(1)):
        assert len(reads) == 1
        (table,) = _SHARED_STEP_DATA.values()
        assert list(table._indexes) == [("coverage_type", "occupancycode", "peril_id")]
        assert config == config_before  # the workers' lookups are made from the config as it was
        for _ in range(2):  # e.g. two workers
            output = Lookup(config=copy.deepcopy(config)).process_locations(locations)
            assert output.sort_values("loc_id", kind="stable")["vulnerability_id"].tolist() == [13, 11, 12]
        assert len(reads) == 1
        assert len(table._indexes) == 1  # the workers used the parent's index

    assert not _SHARED_STEP_DATA
    Lookup(config=copy.deepcopy(config)).process_locations(locations)
    assert len(reads) == 2


def test_multiproc_shared_data_indexes_merge_join_columns(merge_table_csv):
    """With join_columns set, the table is indexed on them without running the lookup on a sample."""
    from oasislmf.lookup.builtin import _SHARED_STEP_DATA
    config = _merge_lookup_config(str(merge_table_csv))
    config["step_definition"]["vulnerability"]["parameters"]["join_columns"] = ["Peril_ID", "coverage_type"]
    with Lookup(config=config).multiproc_shared_data():
        (table,) = _SHARED_STEP_DATA.values()
        assert list(table._indexes) == [("coverage_type", "peril_id")]


def test_multiproc_shared_data_sample_failure_leaves_indexing_to_workers(monkeypatch, merge_table_csv, caplog):
    """If the lookup fails on the sample, the shared data is still loaded and a worker indexes the
    table itself, with a warning that tells how to avoid it."""
    from oasislmf.lookup import builtin
    config = _merge_after_prepare_config(str(merge_table_csv))
    locations = pd.DataFrame({"loc_id": [1], "coverage_type": [3], "occupancycode": [1050]})
    with Lookup(config=config).multiproc_shared_data(sample_locations=locations.drop(columns=["coverage_type"])):
        (table,) = builtin._SHARED_STEP_DATA.values()
        assert not table._indexes
        assert "the lookup failed on the 1 sample locations" in caplog.text
        assert "wasn't indexed in the parent process" in caplog.text
        monkeypatch.setattr(builtin.os, "getpid", lambda: -1)  # as in a forked worker
        output = Lookup(config=copy.deepcopy(config)).process_locations(locations)
        assert output["vulnerability_id"].tolist() == [13]
        assert "in lookup worker process -1" in caplog.text


def test_multiproc_shared_data_skips_overridden_build_merge(monkeypatch, merge_table_csv):
    """A subclass with its own build_merge may not use the data, so none is loaded for it."""
    reads = _count_csv_reads(monkeypatch)

    class CustomLookup(Lookup):
        def build_merge(self, **kwargs):
            return lambda locations: locations

    with CustomLookup(config=_merge_lookup_config(str(merge_table_csv))).multiproc_shared_data():
        pass
    assert not reads


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


def test_split_loc_perils_covered_group_listed_with_its_perils_gives_each_peril_once():
    """OO1 is the group {ORF, OSF}: listing it with ORF and OSF must not give those perils twice."""
    fct = Lookup(config={}).build_split_loc_perils_covered()
    locations = pd.DataFrame({
        "loc_id": [1, 2],
        "LocPerilsCovered": ["OO1;ORF;OSF", "ORF;OSF"],
    })

    result = fct(locations)

    perils = result.groupby("loc_id")["peril_id"].apply(sorted)
    assert perils[1] == perils[2] == ["ORF", "OSF"]
    assert "peril_group_id" not in locations


@pytest.mark.parametrize("perils_covered, expected_group", [("OO1;ORF", "OO1"), ("ORF;OO1", "ORF")])
def test_split_loc_perils_covered_peril_in_several_groups_keeps_the_first_listed(perils_covered, expected_group):
    """A peril reached through several listed groups keeps the first listed group as peril_group_id."""
    fct = Lookup(config={}).build_split_loc_perils_covered()

    result = fct(pd.DataFrame({"loc_id": [1], "LocPerilsCovered": [perils_covered]}))

    orf = result[result["peril_id"] == "ORF"]
    assert len(orf) == 1
    assert orf["peril_group_id"].iloc[0] == expected_group


def test_peril_covered_deterministic_lookup_group_listed_with_its_perils_gives_each_peril_once():
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
