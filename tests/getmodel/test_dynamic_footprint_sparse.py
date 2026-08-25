"""
Tests for sparse and unpartitioned event definition / hazard case files.

The dynamic footprint is generated on the fly from an event definition file and a
hazard case file, either of which may be partitioned by ``section_id``. A section
that is absent from one of those files means "not at risk": no events affect it
(event definition) or the modelled perils leave it unaffected (hazard case).
Neither absence should crash the model.

See https://github.com/OasisLMF/OasisLMF/issues/2090 (sparse files)
and https://github.com/OasisLMF/OasisLMF/issues/2091 (unpartitioned files).
"""
import json
import logging
import shutil
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from oasis_data_manager.filestore.backends.local import LocalStorage

from oasislmf.pytools.getmodel.common import (
    event_defintion_filename, hazard_case_filename, parquetfootprint_meta_filename)
from oasislmf.pytools.getmodel.footprint import FootprintParquetDynamic
from oasislmf.utils.path import setcwd

# the sections a flat file holds are found by pushing a section_id filter down to the reader,
# so the sparse paths are worth running on every engine, not just the default one
DF_ENGINES = ['oasis_data_manager.df_reader.reader.OasisPandasReader',
              'oasis_data_manager.df_reader.reader.OasisPyarrowReader']


# ---------------------------------------------------------------------------
# Fixture data
#
# Two sections, each with its own areaperils. Event 1 spans both sections,
# event 2 touches section 2 only, event 3 touches section 1 only.
# ---------------------------------------------------------------------------

def make_event_definition(rows=None):
    rows = rows if rows is not None else [
        # event_id, section_id, rp_from, rp_to, interpolation, rp
        (1, 1, 10, 20, 0.5, 15),
        (1, 2, 10, 20, 0.5, 15),
        (2, 2, 10, 20, 0.25, 12),
        (3, 1, 10, 20, 0.75, 17),
    ]
    return pd.DataFrame(rows, columns=['event_id', 'section_id', 'rp_from', 'rp_to', 'interpolation', 'rp']).astype(
        {'event_id': 'int32', 'section_id': 'int32', 'rp_from': 'int32', 'rp_to': 'int32',
         'interpolation': 'float64', 'rp': 'int32'})


def make_hazard_case(rows=None):
    rows = rows if rows is not None else [
        # section_id, areaperil_id, return_period, intensity
        (1, 100, 10, 4),
        (1, 100, 20, 8),
        (1, 101, 10, 6),
        (1, 101, 20, 10),
        (2, 200, 10, 2),
        (2, 200, 20, 6),
    ]
    return pd.DataFrame(rows, columns=['section_id', 'areaperil_id', 'return_period', 'intensity']).astype(
        {'section_id': 'int32', 'areaperil_id': 'uint32', 'return_period': 'int32', 'intensity': 'int32'})


def write_parquet(df, path, partitioned):
    if partitioned:
        df.to_parquet(str(path), partition_cols=['section_id'], index=False)
    else:
        df.to_parquet(str(path), index=False)


def build_model(tmp_path, event_definition=None, hazard_case=None,
                partition_event_definition=True, partition_hazard_case=True,
                sections=None, areaperil_ids=None):
    """Lay out a dynamic footprint model on disk.

    Returns: (LocalStorage, Path) the model (static) storage and the run directory.
    """
    event_definition = make_event_definition() if event_definition is None else event_definition
    hazard_case = make_hazard_case() if hazard_case is None else hazard_case

    static_dir = tmp_path / 'static'
    static_dir.mkdir(parents=True, exist_ok=True)
    input_dir = tmp_path / 'run' / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)

    with open(static_dir / parquetfootprint_meta_filename, 'w') as meta_file:
        json.dump({'num_intensity_bins': 10, 'has_intensity_uncertainty': 0}, meta_file)

    write_parquet(event_definition, static_dir / event_defintion_filename, partition_event_definition)
    write_parquet(hazard_case, static_dir / hazard_case_filename, partition_hazard_case)

    if sections is None:
        sections = sorted(set(event_definition['section_id']) | set(hazard_case['section_id']))
    pd.DataFrame({'section_id': np.array(sections, dtype='int32')}).to_csv(input_dir / 'sections.csv', index=False)

    if areaperil_ids is None:
        areaperil_ids = sorted(set(hazard_case['areaperil_id']))
    pd.DataFrame({'AreaPerilID': np.array(areaperil_ids, dtype='uint32')}).to_csv(input_dir / 'keys.csv', index=False)

    return LocalStorage(root_dir=str(static_dir), cache_dir=None), tmp_path / 'run'


@contextmanager
def open_footprint(storage, run_dir, areaperil_ids=None,
                   df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader"):
    """Open a FootprintParquetDynamic with the run directory as cwd, as gulmc does."""
    with setcwd(run_dir):
        with FootprintParquetDynamic(storage, df_engine=df_engine, areaperil_ids=areaperil_ids) as footprint:
            yield footprint


def areaperils_of(event_footprint):
    return set() if event_footprint is None else set(event_footprint['areaperil_id'])


def intensity_by_areaperil(event_footprint):
    return {row['areaperil_id']: row['intensity'] for row in event_footprint}


# ---------------------------------------------------------------------------
# Complete data in every layout: the baseline that must keep working, and the
# unpartitioned layouts of issue 2091.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('df_engine', DF_ENGINES)
@pytest.mark.parametrize('partition_event_definition', [True, False])
@pytest.mark.parametrize('partition_hazard_case', [True, False])
def test_every_partitioning_layout_builds_the_same_footprint(tmp_path, df_engine, partition_event_definition,
                                                             partition_hazard_case):
    """Each file is flat or partitioned independently of the other, on any reader engine."""
    storage, run_dir = build_model(tmp_path, partition_event_definition=partition_event_definition,
                                   partition_hazard_case=partition_hazard_case)

    with open_footprint(storage, run_dir, df_engine=df_engine) as footprint:
        assert intensity_by_areaperil(footprint.get_event(1)) == {100: 6, 101: 8, 200: 4}
        assert areaperils_of(footprint.get_event(2)) == {200}
        assert areaperils_of(footprint.get_event(3)) == {100, 101}
        assert footprint.get_event(999) is None


@pytest.mark.parametrize('partition_event_definition', [True, False])
@pytest.mark.parametrize('partition_hazard_case', [True, False])
def test_hazard_case_is_read_once_per_run(tmp_path, monkeypatch, partition_event_definition,
                                          partition_hazard_case):
    """The hazard case does not depend on the event, so no layout may re-read it per event.

    Reading it per call costs a partition discovery over the whole dataset each time, which
    scales with the sections the model has rather than with the sections the event needs.
    """
    storage, run_dir = build_model(tmp_path, partition_event_definition=partition_event_definition,
                                   partition_hazard_case=partition_hazard_case)

    reads = []
    original_get_df_reader = FootprintParquetDynamic.get_df_reader

    def counting_get_df_reader(self, filepath, **kwargs):
        reads.append(filepath)
        return original_get_df_reader(self, filepath, **kwargs)

    monkeypatch.setattr(FootprintParquetDynamic, 'get_df_reader', counting_get_df_reader)

    with open_footprint(storage, run_dir) as footprint:
        for event_id in (1, 2, 3, 1, 2, 3):
            footprint.get_event(event_id)

    assert reads.count(hazard_case_filename) == 1


# ---------------------------------------------------------------------------
# Issue 2090 — sections missing from the event definition file
# ---------------------------------------------------------------------------

def test_section_missing_from_event_definition_loads(tmp_path):
    """Section 2 has hazard data but no events: sections 1 and 2 both still load."""
    event_definition = make_event_definition([(1, 1, 10, 20, 0.5, 15)])
    storage, run_dir = build_model(tmp_path, event_definition=event_definition, sections=[1, 2])

    with open_footprint(storage, run_dir) as footprint:
        assert areaperils_of(footprint.get_event(1)) == {100, 101}


def test_all_sections_missing_from_event_definition_yields_no_events(tmp_path):
    """A partitioned event definition holding no section of this portfolio: no events."""
    event_definition = make_event_definition([(1, 99, 10, 20, 0.5, 15)])
    storage, run_dir = build_model(tmp_path, event_definition=event_definition, sections=[1, 2])

    with open_footprint(storage, run_dir) as footprint:
        assert footprint.get_event(1) is None


def test_no_location_sections_yields_no_events(tmp_path):
    """An empty portfolio section list must not be mistaken for an unpartitioned file."""
    storage, run_dir = build_model(tmp_path, sections=[])

    with open_footprint(storage, run_dir) as footprint:
        assert footprint.get_event(1) is None


# ---------------------------------------------------------------------------
# Issue 2090 — sections missing from the hazard case file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('df_engine', DF_ENGINES)
def test_section_missing_from_hazard_case_is_not_at_risk(tmp_path, df_engine):
    """Section 2 has events but no hazard: event 1 keeps section 1's areaperils only."""
    hazard_case = make_hazard_case([(1, 100, 10, 4), (1, 100, 20, 8), (1, 101, 10, 6), (1, 101, 20, 10)])
    storage, run_dir = build_model(tmp_path, hazard_case=hazard_case, sections=[1, 2])

    with open_footprint(storage, run_dir, df_engine=df_engine) as footprint:
        assert areaperils_of(footprint.get_event(1)) == {100, 101}


def test_event_only_in_missing_hazard_section_returns_none(tmp_path):
    """Event 2 only touches section 2, which has no hazard data at all."""
    hazard_case = make_hazard_case([(1, 100, 10, 4), (1, 100, 20, 8)])
    storage, run_dir = build_model(tmp_path, hazard_case=hazard_case, sections=[1, 2])

    with open_footprint(storage, run_dir) as footprint:
        assert footprint.get_event(2) is None


def test_all_sections_missing_from_hazard_case_yields_no_events(tmp_path):
    hazard_case = make_hazard_case([(99, 900, 10, 4), (99, 900, 20, 8)])
    storage, run_dir = build_model(tmp_path, hazard_case=hazard_case, sections=[1, 2], areaperil_ids=[100, 101, 200])

    with open_footprint(storage, run_dir) as footprint:
        assert footprint.get_event(1) is None


def test_hazard_section_emptied_by_areaperil_filter_is_not_at_risk(tmp_path):
    """Section 2 has hazard rows, but none for an areaperil in this portfolio."""
    hazard_case = make_hazard_case([
        (1, 100, 10, 4),
        (1, 100, 20, 8),
        (2, 999, 10, 2),
        (2, 999, 20, 6),
    ])
    storage, run_dir = build_model(tmp_path, hazard_case=hazard_case, sections=[1, 2], areaperil_ids=[100])

    with open_footprint(storage, run_dir) as footprint:
        assert areaperils_of(footprint.get_event(1)) == {100}
        assert footprint.get_event(2) is None


def test_areaperils_outside_portfolio_are_filtered_out(tmp_path):
    """Only the portfolio's areaperils are returned, sparse data or not."""
    storage, run_dir = build_model(tmp_path, areaperil_ids=[100, 200])

    with open_footprint(storage, run_dir) as footprint:
        assert areaperils_of(footprint.get_event(1)) == {100, 200}


# ---------------------------------------------------------------------------
# Issue 2090 and 2091 together — sparse data in unpartitioned files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('df_engine', DF_ENGINES)
def test_flat_files_tolerate_missing_hazard_sections(tmp_path, df_engine):
    """Flat hazard case with no rows for section 2, which event 2 needs."""
    hazard_case = make_hazard_case([(1, 100, 10, 4), (1, 100, 20, 8)])
    storage, run_dir = build_model(tmp_path, hazard_case=hazard_case, sections=[1, 2],
                                   partition_event_definition=False, partition_hazard_case=False)

    with open_footprint(storage, run_dir, df_engine=df_engine) as footprint:
        assert areaperils_of(footprint.get_event(1)) == {100}
        assert footprint.get_event(2) is None


# ---------------------------------------------------------------------------
# Issue 2090 — sparsity within a section: a missing return period bracket
# ---------------------------------------------------------------------------

def test_missing_rp_to_interpolates_toward_zero(tmp_path):
    """Areaperil 101 has no rp 20 row: the absent intensity reads as 0, not as a crash."""
    hazard_case = make_hazard_case([
        (1, 100, 10, 4),
        (1, 100, 20, 8),
        (1, 101, 10, 6),
    ])
    event_definition = make_event_definition([(1, 1, 10, 20, 0.5, 15)])
    storage, run_dir = build_model(tmp_path, event_definition=event_definition, hazard_case=hazard_case,
                                   sections=[1], areaperil_ids=[100, 101])

    with open_footprint(storage, run_dir) as footprint:
        # areaperil 101: floor(6 + (0 - 6) * 0.5) == 3
        assert intensity_by_areaperil(footprint.get_event(1)) == {100: 6, 101: 3}


def test_missing_rp_from_interpolates_from_zero(tmp_path):
    """Already supported: absent rp_from reads as intensity 0."""
    hazard_case = make_hazard_case([
        (1, 100, 10, 4),
        (1, 100, 20, 8),
        (1, 101, 20, 10),
    ])
    event_definition = make_event_definition([(1, 1, 10, 20, 0.5, 15)])
    storage, run_dir = build_model(tmp_path, event_definition=event_definition, hazard_case=hazard_case,
                                   sections=[1], areaperil_ids=[100, 101])

    with open_footprint(storage, run_dir) as footprint:
        # areaperil 101: floor(0 + (10 - 0) * 0.5) == 5
        assert intensity_by_areaperil(footprint.get_event(1)) == {100: 6, 101: 5}


# ---------------------------------------------------------------------------
# Stochastic hazard (probability column) with sparse data
# ---------------------------------------------------------------------------

def test_stochastic_sparse_hazard_probabilities_sum_to_one(tmp_path):
    hazard_case = pd.DataFrame({
        'section_id': np.array([1, 1, 1, 1], dtype='int32'),
        'areaperil_id': np.array([100, 100, 100, 100], dtype='uint32'),
        'return_period': np.array([10, 10, 20, 20], dtype='int32'),
        'intensity': np.array([4, 6, 8, 12], dtype='int32'),
        'probability': [0.4, 0.6, 0.4, 0.6],
    })
    event_definition = make_event_definition([(1, 1, 10, 20, 0.5, 15), (1, 2, 10, 20, 0.5, 15)])
    storage, run_dir = build_model(tmp_path, event_definition=event_definition, hazard_case=hazard_case,
                                   sections=[1, 2], areaperil_ids=[100])

    with open_footprint(storage, run_dir) as footprint:
        event_footprint = footprint.get_event(1)
        assert areaperils_of(event_footprint) == {100}
        assert event_footprint['probability'].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# What the log says about it
#
# Treating absent sections as not at risk means a portfolio whose section ids do not
# match the model data produces zero losses instead of an error, so the log has to be
# what tells the two apart.
# ---------------------------------------------------------------------------

def records_matching(caplog, level, *fragments):
    return [record.message for record in caplog.records
            if record.levelno == level and all(fragment in record.message for fragment in fragments)]


@contextmanager
def capture_footprint_logs(caplog):
    """Capture the footprint logger whatever level an earlier test left it at.

    redirect_logging (oasislmf/pytools/utils.py) puts every oasislmf.* logger at WARNING and
    never puts the level back, so on a full run the ambient root level caplog raises is not
    enough to see an INFO record: the logger itself has to be raised too.
    """
    logger = logging.getLogger(FootprintParquetDynamic.__module__)
    propagate = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            yield
    finally:
        logger.propagate = propagate


def test_absent_section_is_reported(tmp_path, caplog):
    """A section the model data does not cover is named, with the file it is missing from."""
    hazard_case = make_hazard_case([(1, 100, 10, 4), (1, 100, 20, 8), (1, 101, 10, 6), (1, 101, 20, 10)])
    storage, run_dir = build_model(tmp_path, hazard_case=hazard_case, sections=[1, 2])

    with capture_footprint_logs(caplog):
        with open_footprint(storage, run_dir) as footprint:
            footprint.get_event(1)

    assert records_matching(caplog, logging.INFO, '[2]', hazard_case_filename, 'not at risk')


def test_absent_section_is_reported_once_per_file(tmp_path, caplog):
    """The flat path re-reads per event, so the same absence must not be logged per event."""
    hazard_case = make_hazard_case([(1, 100, 10, 4), (1, 100, 20, 8)])
    storage, run_dir = build_model(tmp_path, hazard_case=hazard_case, sections=[1, 2],
                                   partition_event_definition=False, partition_hazard_case=False)

    with capture_footprint_logs(caplog):
        with open_footprint(storage, run_dir) as footprint:
            for event_id in (1, 2, 1, 2):
                footprint.get_event(event_id)

    assert len(records_matching(caplog, logging.INFO, hazard_case_filename, 'not at risk')) == 1


def test_portfolio_absent_from_event_definition_warns(tmp_path, caplog):
    """Zero losses from a section list that matches nothing is a warning, not silence."""
    event_definition = make_event_definition([(1, 1, 10, 20, 0.5, 15)])
    storage, run_dir = build_model(tmp_path, event_definition=event_definition, sections=[7])

    with capture_footprint_logs(caplog):
        with open_footprint(storage, run_dir) as footprint:
            assert footprint.get_event(1) is None

    assert records_matching(caplog, logging.WARNING, event_defintion_filename, 'every loss will be zero')


def test_portfolio_absent_from_hazard_case_warns(tmp_path, caplog):
    """The same, one file further on: events exist for the portfolio but no hazard does."""
    hazard_case = make_hazard_case([(99, 900, 10, 4), (99, 900, 20, 8)])
    storage, run_dir = build_model(tmp_path, hazard_case=hazard_case, sections=[1, 2], areaperil_ids=[100])

    with capture_footprint_logs(caplog):
        with open_footprint(storage, run_dir) as footprint:
            assert footprint.get_event(1) is None

    assert records_matching(caplog, logging.WARNING, hazard_case_filename, 'every loss will be zero')


def test_missing_file_is_reported_as_that_file(tmp_path):
    """A file absent altogether is a broken model, not sparse data: it must still name itself.

    Partition detection has to inspect the file, so it is the first thing to touch a missing
    one; the error the caller sees should be the read failing, not the detection probe.
    """
    storage, run_dir = build_model(tmp_path)
    shutil.rmtree(tmp_path / 'static' / hazard_case_filename)

    with pytest.raises(FileNotFoundError, match=hazard_case_filename):
        with open_footprint(storage, run_dir) as footprint:
            footprint.get_event(1)


def test_complete_model_data_is_quiet(tmp_path, caplog):
    """Nothing absent, nothing said: the reporting must not fire on ordinary models."""
    storage, run_dir = build_model(tmp_path)

    with capture_footprint_logs(caplog):
        with open_footprint(storage, run_dir) as footprint:
            footprint.get_event(1)

    assert records_matching(caplog, logging.INFO, 'not at risk') == []
    assert records_matching(caplog, logging.WARNING, 'every loss will be zero') == []
