__all__ = [
    'VulnerabilityFile',
    'Eventsfile',
    'FootprintBinFile',
    'FootprintIdxFile',
    'DamageBinDictFile',
    'OccurrenceFile',
    'RandomFile',
    'CoveragesFile',
    'ItemsFile',
    'FMProgrammeFile',
    'FMPolicyTCFile',
    'FMProfileFile',
    'FMXrefFile',
    'GULSummaryXrefFile',
    'FMSummaryXrefFile'
]

from collections import OrderedDict
import numpy as np
import os
import struct


class ModelFile:
    """
    Base class for all dummy model files.
    """
    def __init__(self):
        pass

    def seed_rng(self):
        if self.random_seed == 0:
            np.random.seed()
        elif self.random_seed == -1:
            # Add salt to random seed using name of child class
            salt = int.from_bytes(type(self).__name__.encode(), 'little')
            np.random.seed((1234 + salt) % 0xFFFFFFFF)
        else:
            # Add salt to random seed using name of child class
            salt = int.from_bytes(type(self).__name__.encode(), 'little')
            np.random.seed((self.random_seed + salt) % 0xFFFFFFFF)

    def write_file(self):
        """
        Convert data to binary format and writing to file.
        """
        with open(self.file_name, 'wb') as f:
            if self.start_stats:
                for stat in self.start_stats:
                    f.write(struct.pack(stat['dtype'], stat['value']))
            dtypes_list = ''.join(self.dtypes.values())
            f.write(struct.pack(
                '=' + dtypes_list * self.data_length,
                *(x for y in self.generate_data() for x in y)
            ))

    def debug_write_file(self):
        """
        Print csv output to screen. Used for debugging file output.
        """
        if self.start_stats:
            for stat in self.start_stats:
                print('{} = {}'.format(stat['desc'], stat['value']))
        line_format = '{}' + ',{}'*(len(self.dtypes)-1)
        print(line_format.format(*self.dtypes.keys()))
        for line in self.generate_data():
            print(line_format.format(*line))


class VulnerabilityFile(ModelFile):
    def __init__(
        self, num_vulnerabilities, num_intensity_bins, num_damage_bins,
        vulnerability_sparseness, random_seed, directory
    ):
        self.num_vulnerabilities = num_vulnerabilities
        self.num_intensity_bins = num_intensity_bins
        self.num_damage_bins = num_damage_bins
        self.vulnerability_sparseness = vulnerability_sparseness
        self.dtypes = OrderedDict([
            ('vulnerability_id', 'i'), ('intensity_bin_index', 'i'),
            ('damage_bin_index', 'i'), ('prob', 'f')
        ])
        self.start_stats = [
            {
                'desc': 'Number of damage bins', 'value': num_damage_bins,
                'dtype': 'i'
            }
        ]
        self.random_seed = random_seed
        self.data_length = num_vulnerabilities * num_intensity_bins * num_damage_bins
        self.file_name = os.path.join(directory, 'vulnerability.bin')

    def generate_data(self):
        super().seed_rng()
        for vulnerability in range(self.num_vulnerabilities):
            for intensity_bin in range(self.num_intensity_bins):

                # Generate probabalities according to vulnerability sparseness
                # and normalise
                triggers = np.random.uniform(size=self.num_damage_bins)
                probabilities = np.apply_along_axis(
                    lambda x: np.where(
                        x < self.vulnerability_sparseness,
                        np.random.uniform(size=x.shape), 0.0
                    ), 0, triggers
                )
                probabilities /= np.sum(probabilities)

                for damage_bin, probability in enumerate(probabilities):
                    yield vulnerability+1, intensity_bin+1, damage_bin+1, probability


class EventsFile(ModelFile):
    def __init__(self, num_events, directory):
        self.num_events = num_events
        self.dtypes = {'event_id': 'i'}
        self.start_stats = None
        self.data_length = num_events
        self.file_name = os.path.join(directory, 'events.bin')

    def generate_data(self):
        return (tuple([event]) for event in range(1, self.num_events+1))


class FootprintFiles(ModelFile):
    bin_dtypes = OrderedDict([
        ('areaperil_id', 'i'), ('intensity_bin_id', 'i'), ('probability', 'f')
    ])

    def __init__(
        self, num_events, num_areaperils, areaperils_per_event,
        num_intensity_bins, intensity_sparseness, no_intensity_uncertainty
    ):
        self.num_events = num_events
        self.num_areaperils = num_areaperils
        self.areaperils_per_event = areaperils_per_event
        self.num_intensity_bins = num_intensity_bins
        self.intensity_sparseness = intensity_sparseness
        self.no_intensity_uncertainty = no_intensity_uncertainty

    def get_bin_start_stats(self):
        return [
            {
                'desc': 'Number of intensity bins',
                'value': self.num_intensity_bins, 'dtype': 'i'
            },
            {
                'desc': 'Has Intensity Uncertainty',
                'value': not self.no_intensity_uncertainty, 'dtype': 'i'
            }
        ]


class FootprintBinFile(FootprintFiles):
    def __init__(
        self, num_events, num_areaperils, areaperils_per_event,
        num_intensity_bins, intensity_sparseness, no_intensity_uncertainty,
        random_seed, directory
    ):
        super().__init__(
            num_events, num_areaperils, areaperils_per_event,
            num_intensity_bins, intensity_sparseness, no_intensity_uncertainty
        )
        self.start_stats = self.get_bin_start_stats()
        self.dtypes = FootprintFiles.bin_dtypes
        self.random_seed = random_seed
        if no_intensity_uncertainty:
            self.data_length = num_events * areaperils_per_event
        else:
            self.data_length = num_events * areaperils_per_event * num_intensity_bins
        self.file_name = os.path.join(directory, 'footprint.bin')

    def generate_data(self):
        super().seed_rng()
        for event in range(self.num_events):

            if self.areaperils_per_event == self.num_areaperils:
                selected_areaperils = np.arange(1, self.num_areaperils+1)
            else:
                selected_areaperils = np.random.choice(
                    self.num_areaperils, self.areaperils_per_event,
                    replace=False
                )
                selected_areaperils += 1
                selected_areaperils = np.sort(selected_areaperils)

            for areaperil in selected_areaperils:
                if self.no_intensity_uncertainty:
                    intensity_bin = np.random.randint(
                        1, self.num_intensity_bins+1
                    )
                    probability = 1.0
                    yield areaperil, intensity_bin, probability
                else:
                    # Generate probabalities according to intensity sparseness
                    # and normalise
                    triggers = np.random.uniform(size=self.num_intensity_bins)
                    probabilities = np.apply_along_axis(
                        lambda x: np.where(
                            x < self.intensity_sparseness,
                            np.random.uniform(size=x.shape), 0.0
                        ), 0, triggers
                    )
                    probabilities /= np.sum(probabilities)

                    for intensity_bin, probability in enumerate(probabilities):
                        yield areaperil, intensity_bin+1, probability


class FootprintIdxFile(FootprintFiles):
    def __init__(
        self, num_events, num_areaperils, areaperils_per_event,
        num_intensity_bins, intensity_sparseness, no_intensity_uncertainty,
        directory
    ):
        super().__init__(
            num_events, num_areaperils, areaperils_per_event,
            num_intensity_bins, intensity_sparseness, no_intensity_uncertainty
        )
        self.start_stats = None
        self.dtypes = OrderedDict([
            ('event_id', 'i'), ('offset', 'q'), ('size', 'q')
        ])
        self.data_length = num_events
        self.file_name = os.path.join(directory, 'footprint.idx')

    def generate_data(self):
        # Size is the same for all events
        size = 0
        for dtype in FootprintFiles.bin_dtypes.values():
            size += struct.calcsize(dtype)
        size *= self.areaperils_per_event
        if not self.no_intensity_uncertainty:
            size *= self.num_intensity_bins
        # Set initial offset
        offset = 0
        for stat in self.get_bin_start_stats():
            offset += struct.calcsize(stat['dtype'])

        for event in range(self.num_events):
            yield event+1, offset, size
            offset += size


class DamageBinDictFile(ModelFile):
    def __init__(self, num_damage_bins, directory):
        self.num_damage_bins = num_damage_bins
        self.dtypes = OrderedDict([
            ('bin_index', 'i'), ('bin_from', 'f'), ('bin_to', 'f'),
            ('interpolation', 'f'), ('interval_type', 'i')
        ])
        self.start_stats = None
        self.data_length = num_damage_bins
        self.file_name = os.path.join(directory, 'damage_bin_dict.bin')

    def generate_data(self):
        # Exclude first and last bins for now
        bin_indexes = np.arange(self.num_damage_bins-2)
        bin_from_values = bin_indexes / (self.num_damage_bins-2)
        bin_to_values = (bin_indexes + 1) / (self.num_damage_bins-2)
        # Set interpolation in middle of bin
        interpolations = (0.5 + bin_indexes) / (self.num_damage_bins-2)
        # Insert first and last bins
        bin_indexes += 2
        bin_indexes = np.insert(bin_indexes, 0, 1)
        bin_indexes = np.append(bin_indexes, self.num_damage_bins)
        fields = [bin_from_values, bin_to_values, interpolations]
        for i, field in enumerate(fields):
            fields[i] = np.insert(field, 0, 0)
            fields[i] = np.append(fields[i], 1)
        bin_from_values, bin_to_values, interpolations = fields
        # Set interval type for all bins to 0 (unused)
        interval_type = 0

        for bin_id, bin_from, bin_to, interpolation in zip(
            bin_indexes, bin_from_values, bin_to_values, interpolations
        ):
            yield bin_id, bin_from, bin_to, interpolation, interval_type


class OccurrenceFile(ModelFile):
    def __init__(self, num_events, num_periods, random_seed, directory):
        self.num_events = num_events
        self.num_periods = num_periods
        self.dtypes = OrderedDict([
            ('event_id', 'i'), ('period_no', 'i'), ('occ_date_id', 'i')
        ])
        self.date_algorithm = 1
        self.start_stats = [
            {
                'desc': 'Date algorithm', 'value': self.date_algorithm,
                'dtype': 'i'
            },
            {
                'desc': 'Number of periods', 'value': self.num_periods,
                'dtype': 'i'
            }
        ]
        self.random_seed = random_seed
        self.data_length = num_events
        self.file_name = os.path.join(directory, 'occurrence.bin')

    def set_occ_date_id(self, year, month, day):
        # Set date relative to epoch
        month = (month + 9) % 12
        year = year - month // 10
        return 365 * year + year // 4 - year // 100 + year // 400 + (306 * month + 5) // 10 + (day - 1)

    def generate_data(self):
        super().seed_rng()
        months = np.arange(1, 13)
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        months_weights = np.array(days_per_month, dtype=float)
        months_weights /= months_weights.sum()   # Normalise
        for event in range(self.num_events):
            period_no = np.random.randint(1, self.num_periods+1)
            occ_year = period_no   # Assume one period represents one year
            occ_month = np.random.choice(months, p=months_weights)
            occ_day = np.random.randint(1, days_per_month[occ_month-1])
            occ_date = self.set_occ_date_id(occ_year, occ_month, occ_day)
            yield event+1, period_no, occ_date


class RandomFile(ModelFile):
    def __init__(self, num_randoms, random_seed, directory):
        self.num_randoms = num_randoms
        self.dtypes = {'random_no': 'f'}
        self.start_stats = None
        self.random_seed = random_seed
        self.data_length = num_randoms
        self.file_name = os.path.join(directory, 'random.bin')

    def generate_data(self):
        super().seed_rng()
        # First random number is 0
        return (tuple([np.random.uniform()]) if i != 0 else (0,) for i in range(self.num_randoms))


class CoveragesFile(ModelFile):
    def __init__(
        self, num_locations, coverages_per_location, random_seed, directory
    ):
        self.num_locations = num_locations
        self.coverages_per_location = coverages_per_location
        self.dtypes = {'tiv': 'f'}
        self.start_stats = None
        self.random_seed = random_seed
        self.data_length = num_locations * coverages_per_location
        self.file_name = os.path.join(directory, 'coverages.bin')

    def generate_data(self):
        super().seed_rng()
        # Assume 1-1 mapping between item and coverage IDs
        return (
            tuple([np.random.uniform(1, 1000000)]) for _ in range(
                self.num_locations * self.coverages_per_location
            )
        )


class ItemsFile(ModelFile):
    def __init__(
        self, num_locations, coverages_per_location, num_areaperils,
        num_vulnerabilities, random_seed, directory
    ):
        self.num_locations = num_locations
        self.coverages_per_location = coverages_per_location
        self.num_areaperils = num_areaperils
        self.num_vulnerabilities = num_vulnerabilities
        self.dtypes = OrderedDict([
            ('item_id', 'i'), ('coverage_id', 'i'), ('areaperil_id', 'i'),
            ('vulnerability_id', 'i'), ('group_id', 'i')
        ])
        self.start_stats = None
        self.random_seed = random_seed
        self.data_length = num_locations * coverages_per_location
        self.file_name = os.path.join(directory, 'items.bin')

    def generate_data(self):
        super().seed_rng()
        for location in range(self.num_locations):
            areaperils = np.random.randint(
                1, self.num_areaperils+1, size=self.coverages_per_location
            )
            vulnerabilities = np.random.randint(
                1, self.num_vulnerabilities+1, size=self.coverages_per_location
            )
            for coverage in range(self.coverages_per_location):
                item = self.coverages_per_location * location + coverage + 1
                # Assume 1-1 mapping between item and coverage IDs
                # Assume group ID mapped to location
                yield item, item, areaperils[coverage], vulnerabilities[coverage], location+1


class FMFile(ModelFile):
    def __init__(self, num_locations, coverages_per_location):
        self.num_locations = num_locations
        self.coverages_per_location = coverages_per_location
        self.start_stats = None


class FMProgrammeFile(FMFile):
    def __init__(self, num_locations, coverages_per_location, directory):
        super().__init__(num_locations, coverages_per_location)
        self.dtypes = OrderedDict([
            ('from_agg_id', 'i'), ('level_id', 'i'), ('to_agg_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location * 2   # 2 from number of levels
        self.file_name = os.path.join(directory, 'fm_programme.bin')

    def generate_data(self):
        levels = [1, 10]
        levels = range(1, len(levels)+1)
        for level in levels:
            for agg_id in range(
                1, self.num_locations * self.coverages_per_location + 1
            ):
                # Site coverage FM level
                if level == 1:
                    yield agg_id, level, agg_id
                # Policy layer FM level
                elif level == len(levels):
                    yield agg_id, level, 1


class FMPolicyTCFile(FMFile):
    def __init__(
        self, num_locations, coverages_per_location, num_layers, directory
    ):
        super().__init__(num_locations, coverages_per_location)
        self.num_layers = num_layers
        self.dtypes = OrderedDict([
            ('layer_id', 'i'), ('level_id', 'i'), ('agg_id', 'i'),
            ('policytc_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location + num_layers
        self.file_name = os.path.join(directory, 'fm_policytc.bin')

    def generate_data(self):
        # Site coverage #1 & policy layer #10 FM levels
        levels = [1, 10]
        levels = range(1, len(levels)+1)
        policytc_id = 1
        for level in levels:
            # Site coverage FM level
            if level == 1:
                for agg_id in range(
                    1, self.num_locations * self.coverages_per_location + 1
                ):
                    # One layer in site coverage FM level
                    yield level, agg_id, 1, policytc_id
                policytc_id += 1   # Next policytc_id
            # Policy layer FM level
            elif level == len(levels):
                for layer in range(self.num_layers):
                    yield level, 1, layer+1, policytc_id
                    policytc_id += 1   # Next policytc_id


class FMProfileFile(ModelFile):
    def __init__(self, num_layers, directory):
        self.num_layers = num_layers
        self.dtypes = OrderedDict([
            ('policytc_id', 'i'), ('calcrule_id', 'i'), ('deductible1', 'f'),
            ('deductible2', 'f'), ('deductible3', 'f'), ('attachment1', 'f'),
            ('limit1', 'f'), ('share1', 'f'), ('share2', 'f'), ('share3', 'f')
        ])
        self.start_stats = None
        self.data_length = 1 + num_layers   # 1 from pass through at level 1
        self.file_name = os.path.join(directory, 'fm_profile.bin')

    def generate_data(self):
        # Pass through for level 1
        profile_rows = [(1, 100, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        # First policy
        init_policytc_id = 2
        init_attachment1 = 500000.0
        attachment1_offset = 5000000.0
        max_limit1 = 100000000.0
        for layer in range(self.num_layers):
            policytc_id = init_policytc_id + layer
            attachment1 = init_attachment1 + attachment1_offset * layer
            # Set limit1 at maximum for last layer
            if (layer+1) == self.num_layers:
                limit1 = max_limit1
            else:
                limit1 = attachment1_offset * (layer+1)
            profile_rows.append(
                (policytc_id, 2, 0.0, 0.0, 0.0, attachment1, limit1, 0.3, 0.0, 0.0)
            )
        for row in profile_rows:
            yield row


class FMXrefFile(FMFile):
    def __init__(
        self, num_locations, coverages_per_location, num_layers, directory
    ):
        super().__init__(num_locations, coverages_per_location)
        self.num_layers = num_layers
        self.dtypes = OrderedDict([
            ('output', 'i'), ('agg_id', 'i'), ('layer_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location * num_layers
        self.file_name = os.path.join(directory, 'fm_xref.bin')

    def generate_data(self):
        layers = range(1, self.num_layers+1)
        output_count = 1
        for agg_id in range(
            1, self.num_locations * self.coverages_per_location + 1
        ):
            for layer in layers:
                yield output_count, agg_id, layer
                output_count += 1


class GULSummaryXrefFile(FMFile):
    def __init__(self, num_locations, coverages_per_location, directory):
        super().__init__(num_locations, coverages_per_location)
        self.dtypes = OrderedDict([
            ('item_id', 'i'), ('summary_id', 'i'), ('summaryset_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location
        self.file_name = os.path.join(directory, 'gulsummaryxref.bin')

    def generate_data(self):
        summary_id = 1
        summaryset_id = 1
        for item in range(self.num_locations * self.coverages_per_location):
            yield item+1, summary_id, summaryset_id


class FMSummaryXrefFile(FMFile):
    def __init__(
        self, num_locations, coverages_per_location, num_layers, directory
    ):
        super().__init__(num_locations, coverages_per_location)
        self.num_layers = num_layers
        self.dtypes = OrderedDict([
            ('output_id', 'i'), ('summary_id', 'i'), ('summaryset_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location * num_layers
        self.file_name = os.path.join(directory, 'fmsummaryxref.bin')

    def generate_data(self):
        summary_id = 1
        summaryset_id = 1
        for output_id in range(
            self.num_locations * self.coverages_per_location * self.num_layers
        ):
            yield output_id+1, summary_id, summaryset_id
