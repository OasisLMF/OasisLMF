"""Pin the array generation of the dummy model files against the row by row generation."""

import struct

import numpy as np
import pytest

from oasislmf.computation.data.dummy_model.generate import (AmplificationsFile, CoveragesFile,
                                                            DamageBinDictFile, EventsFile,
                                                            FMPolicyTCFile, FMProfileFile,
                                                            FMProgrammeFile, FMSummaryXrefFile,
                                                            FMXrefFile, FootprintBinFile,
                                                            GULSummaryXrefFile, ItemsFile,
                                                            LossFactorsFile, OccurrenceFile,
                                                            RandomFile, VulnerabilityFile)

RANDOM_SEED = -1


def model_files(directory, **kwargs):
    """Build every dummy model file class, at a size that exercises more than one chunk."""
    settings = {
        'num_vulnerabilities': 5, 'num_intensity_bins': 4, 'num_damage_bins': 6,
        'vulnerability_sparseness': 0.6, 'num_events': 7, 'num_areaperils': 8,
        'areaperils_per_event': 8, 'intensity_sparseness': 0.7, 'no_intensity_uncertainty': False,
        'num_periods': 20, 'num_locations': 9, 'coverages_per_location': 3, 'num_layers': 2,
        'num_amplifications': 4, 'min_pla_factor': 0.875, 'max_pla_factor': 1.5, 'num_randoms': 11,
        **kwargs,
    }
    seed, out = RANDOM_SEED, str(directory)

    return {
        'vulnerability': VulnerabilityFile(
            settings['num_vulnerabilities'], settings['num_intensity_bins'],
            settings['num_damage_bins'], settings['vulnerability_sparseness'], seed, out),
        'events': EventsFile(settings['num_events'], out),
        'footprint': FootprintBinFile(
            settings['num_events'], settings['num_areaperils'], settings['areaperils_per_event'],
            settings['num_intensity_bins'], settings['intensity_sparseness'],
            settings['no_intensity_uncertainty'], seed, out),
        'damage_bin_dict': DamageBinDictFile(settings['num_damage_bins'], out),
        'occurrence': OccurrenceFile(
            settings['num_events'], settings['num_periods'], seed, out, mean=2, stddev=1.0),
        'loss_factors': LossFactorsFile(
            settings['num_events'], settings['num_amplifications'], settings['min_pla_factor'],
            settings['max_pla_factor'], seed, out),
        'random': RandomFile(settings['num_randoms'], seed, out),
        'coverages': CoveragesFile(
            settings['num_locations'], settings['coverages_per_location'], seed, out),
        'items': ItemsFile(
            settings['num_locations'], settings['coverages_per_location'],
            settings['num_areaperils'], settings['num_vulnerabilities'], seed, out),
        'amplifications': AmplificationsFile(
            settings['num_locations'], settings['coverages_per_location'],
            settings['num_amplifications'], seed, out),
        'gulsummaryxref': GULSummaryXrefFile(
            settings['num_locations'], settings['coverages_per_location'], out),
        'fm_programme': FMProgrammeFile(
            settings['num_locations'], settings['coverages_per_location'], out),
        'fm_policytc': FMPolicyTCFile(
            settings['num_locations'], settings['coverages_per_location'], settings['num_layers'], out),
        'fm_profile': FMProfileFile(settings['num_layers'], out),
        'fm_xref': FMXrefFile(
            settings['num_locations'], settings['coverages_per_location'], settings['num_layers'], out),
        'fmsummaryxref': FMSummaryXrefFile(
            settings['num_locations'], settings['coverages_per_location'], settings['num_layers'], out),
    }


def rows_of(model_file):
    """Pack the rows from the row by row generation into the file's dtype."""
    return np.fromiter(model_file.generate_data(), dtype=model_file.array_dtype)


def arrays_of(model_file):
    return np.concatenate(list(model_file.generate_arrays()))


FILE_NAMES = list(model_files('.'))


@pytest.mark.parametrize('file_name', FILE_NAMES)
def test_arrays_match_rows(file_name, tmp_path):
    # a fresh object per path, as generating the data advances the footprint file's offset
    expected = rows_of(model_files(tmp_path)[file_name])
    generated = arrays_of(model_files(tmp_path)[file_name])

    assert generated.dtype == expected.dtype
    np.testing.assert_array_equal(generated, expected)


@pytest.mark.parametrize('file_name', FILE_NAMES)
def test_arrays_match_rows_over_several_chunks(file_name, tmp_path):
    # enough vulnerabilities and events that the chunked generators yield more than once
    sizes = {'num_damage_bins': 400, 'num_events': 300, 'num_amplifications': 5000}
    expected = rows_of(model_files(tmp_path, **sizes)[file_name])
    generated = arrays_of(model_files(tmp_path, **sizes)[file_name])

    np.testing.assert_array_equal(generated, expected)


@pytest.mark.parametrize('sparseness', [0.0, 0.5, 1.0])
def test_vulnerability_matches_rows_at_any_sparseness(sparseness, tmp_path):
    # 0.0 leaves every vulnerability with no impacted bin, taking the zero-loss fallback
    settings = {'vulnerability_sparseness': sparseness}
    expected = rows_of(model_files(tmp_path, **settings)['vulnerability'])
    generated = arrays_of(model_files(tmp_path, **settings)['vulnerability'])

    np.testing.assert_array_equal(generated, expected)


@pytest.mark.parametrize('sparseness', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('no_intensity_uncertainty', [False, True])
@pytest.mark.parametrize('areaperils_per_event', [4, 8])
def test_footprint_matches_rows(sparseness, no_intensity_uncertainty, areaperils_per_event, tmp_path):
    settings = {
        'intensity_sparseness': sparseness,
        'no_intensity_uncertainty': no_intensity_uncertainty,
        'areaperils_per_event': areaperils_per_event,
    }
    expected = rows_of(model_files(tmp_path, **settings)['footprint'])
    generated = arrays_of(model_files(tmp_path, **settings)['footprint'])

    np.testing.assert_array_equal(generated, expected)


def test_footprint_index_matches_the_rows_written_per_event(tmp_path):
    footprint = model_files(tmp_path)['footprint']
    events = list(footprint.generate_arrays())

    index = footprint.index
    assert len(index) == footprint.num_events
    np.testing.assert_array_equal(index['event_id'], np.arange(1, footprint.num_events + 1))
    # every event's data sits at the offset the index gives, and is as long as it claims
    assert index['offset'][0] == footprint.initial_offset
    np.testing.assert_array_equal(
        index['size'], [event.nbytes for event in events]
    )
    np.testing.assert_array_equal(
        index['offset'], footprint.initial_offset + np.cumsum([0] + [event.nbytes for event in events[:-1]])
    )


def test_footprint_index_written_to_file(tmp_path):
    footprint = model_files(tmp_path)['footprint']
    footprint.write_file()

    written = np.fromfile(footprint.idx_file.file_name, dtype=footprint.idx_file.array_dtype)
    np.testing.assert_array_equal(written, footprint.index)


@pytest.mark.parametrize('file_name', FILE_NAMES)
def test_written_file_holds_the_generated_rows(file_name, tmp_path):
    model_file = model_files(tmp_path)[file_name]
    model_file.write_file()

    expected = arrays_of(model_files(tmp_path)[file_name])
    start_stats_size = sum(struct.calcsize(stat['dtype']) for stat in model_file.start_stats or [])
    written = np.fromfile(model_file.file_name, dtype=model_file.array_dtype,
                          offset=start_stats_size)

    np.testing.assert_array_equal(written, expected)
