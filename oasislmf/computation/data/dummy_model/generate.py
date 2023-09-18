__all__ = [
    'VulnerabilityFile',
    'EventsFile',
    'LossFactorsFile',
    'FootprintBinFile',
    'FootprintIdxFile',
    'DamageBinDictFile',
    'OccurrenceFile',
    'RandomFile',
    'CoveragesFile',
    'ItemsFile',
    'AmplificationsFile',
    'FMProgrammeFile',
    'FMPolicyTCFile',
    'FMProfileFile',
    'FMXrefFile',
    'GULSummaryXrefFile',
    'FMSummaryXrefFile'
]

import os
import struct
from collections import OrderedDict
from math import erf

import numpy as np


class ModelFile:
    """
    Base class for all dummy model files.

    Each dummy model file is a class that inherits from this base class. The
    typical order of execution is as follows:
        1. Initialise class attributes and methods (__init__).
        2. Set random seed (seed_rng).
        3. Generate random data (generate_data).
        4. Convert random data to binary format and write to file. This step is
        done as each line of data is generated to minimise memory use
        (write_file).

    Attributes:
        seed_rng: Seed random number generator.
        write_file: Write data to output file in binary format.
        debug_write_file: Write data to screen in csv format.
        generate_data: Generate dummy model data.
    """

    def __init__(self):
        pass

    def seed_rng(self):
        """
        Seed random number generator.

        Assign different random number generator seed to generate each
        randomised dummy model file data. Pollute seeds with salt to prevent
        all random number generators starting with the same seed.
        """
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
        Write data to output file in binary format.

        General method to convert generated data to binary format and write to
        file. Calls chlid class-specific generate_data method.
        """
        with open(self.file_name, 'wb') as f:
            if self.start_stats:
                for stat in self.start_stats:
                    f.write(struct.pack(stat['dtype'], stat['value']))
            dtypes_list = ''.join(self.dtypes.values())
            for line in self.generate_data():
                f.write(struct.pack('=' + dtypes_list, *(line)))

    def debug_write_file(self):
        """
        Write data to screen in csv format.

        Used for debugging file output.
        """
        if self.start_stats:
            for stat in self.start_stats:
                print('{} = {}'.format(stat['desc'], stat['value']))
        line_format = '{}' + ',{}' * (len(self.dtypes) - 1)
        print(line_format.format(*self.dtypes.keys()))
        for line in self.generate_data():
            print(line_format.format(*line))

    def generate_data(self):
        """
        Generate dummy model data.

        Class specific method to generate randomised data. Is called by
        write_file method.
        """
        pass


class VulnerabilityFile(ModelFile):
    """
    Generate random data for Vulnerability dummy model file.

    This file shows the conditional distributions of damage for each intensity
    bin and for each vulnerability ID.

    Attributes:
        generate_data: Generate Vulnerability dummy model file data.
    """

    def __init__(
        self, num_vulnerabilities, num_intensity_bins, num_damage_bins,
        vulnerability_sparseness, random_seed, directory
    ):
        """
        Initialise VulnerabilityFile class.

        Args:
            num_vulnerabilities (int): number of vulnerabilities.
            num_intensity_bins (int): number of intensity bins.
            num_damage_bins(int): number of damage bins.
            vulnerability_sparseness (float): percentage of bins normalised to
                range [0,1] impacted for a vulnerability at an intensity level.
            random_seed (float): random seed for random number generator.
            directory (str): dummy model file destination.
        """
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
        """
        Generate Vulnerability dummy model file data.

        Yields:
            vulnerability (int): vulnerability ID.
            intensity_bin (int): intensity bin ID.
            damage_bin (int): damage bin ID.
            probability (float): impact probability.
        """
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
                total_probability = np.sum(probabilities)
                if (total_probability == 0):
                    probabilities[0] = 1.0   # First damage bin is always zero-loss
                else:
                    probabilities /= total_probability

                for damage_bin, probability in enumerate(probabilities):
                    yield vulnerability + 1, intensity_bin + 1, damage_bin + 1, probability


class EventsFile(ModelFile):
    """
    Generate random data for Events dummy model file.

    This file lists event IDs to be run.

    Attributes:
        generate_data: Generate Events dummy model file data.
    """

    def __init__(self, num_events, directory):
        """
        Initialise VulnerabilityFile class.

        Args:
            num_events (int): number of events.
            directory (str): dummy model file destination.
        """
        self.num_events = num_events
        self.dtypes = {'event_id': 'i'}
        self.start_stats = None
        self.data_length = num_events
        self.file_name = os.path.join(directory, 'events.bin')

    def generate_data(self):
        """
        Generate Events dummy model file data.

        Yields:
            event (int): event ID.
        """
        return (tuple([event]) for event in range(1, self.num_events + 1))


class LossFactorsFile(ModelFile):
    """
    Generate data for Loss Factors dummy model file.

    This file maps post loss amplification/reduction loss factors to
    event ID-amplification ID pairs.

    Attributes:
        generate_data: Geenrate Loss Factors dummy model file data.
        write_file: Write data to Loss Factors dummy model file in binary
          format.
    """

    def __init__(
        self, num_events, num_amplifications, min_pla_factor, max_pla_factor,
        random_seed, directory
    ):
        """
        Initialise LossFactorsFile class.

        Args:
            num_events (int): number of events.
            num_amplifications (int): number of amplification IDs.
            min_pla_factor (float): minimum post loss amplification/reduction
              factor.
            max_pla_factor (float): maximum post loss amplification/reduction
              factor.
            random_seed (float): random seed for random number generator.
            directory (str): dummy model file destination.
        """
        self.num_events = num_events
        self.num_amplifications = num_amplifications
        self.min_pla_factor = min_pla_factor
        self.delta_pla_factor = max_pla_factor - min_pla_factor
        self.random_seed = random_seed
        self.file_name = os.path.join(directory, 'lossfactors.bin')
        self.start_stats = [
            {
                'desc': 'Reserved for future use', 'value': 0, 'dtype': 'i'
            }
        ]
        self.dtypes = OrderedDict([
            ('event_id', 'i'), ('amplification_id', 'i'), ('factor', 'f')
        ])

    def generate_data(self):
        """
        Generate Loss Factors dummy model file data.

        Yields:
            event (int): event ID
            amplification (int): amplification ID
            factor (float): post loss amplification/reduction factor
        """
        super().seed_rng()
        for event in range(self.num_events):
            for amplification in range(self.num_amplifications):
                factor = np.random.random() * self.delta_pla_factor + self.min_pla_factor
                factor = np.round(factor, decimals=2)
                if factor == 1.0:
                    continue   # Default loss factor = 1.0
                yield event + 1, amplification + 1, factor

    def write_file(self):
        """
        Write data to output Loss Factors file in binary format.

        Checks number of amplifications are greater than 0 before calling base
        class method.
        """
        if not self.num_amplifications:
            return
        super().write_file()


class FootprintIdxFile(ModelFile):
    """
    Generate data for Footprint index dummy model file.

    The binary footprint file footprint.bin requires the index file
    footprint.idx.

    Attributes:
        write_file: Write data to Footprint index file in binary format.
    """

    def __init__(self, directory):
        """
        Initialise Footprint index file class.

        Args:
            directory (str): dummy model file destination.
        """
        self.dtypes = OrderedDict([
            ('event_id', 'i'), ('offset', 'q'), ('size', 'q')
        ])
        self.dtypes_list = ''.join(self.dtypes.values())
        self.file_name = os.path.join(directory, 'footprint.idx')

    def write_file(self, event_id, offset, event_size):
        """
        Write data to output Footprint index file in binary format.

        Overrides method in base class. Converts data to arguments to binary and
        writes to file. Called by FootprintBinFile.generate_data().

        Args:
            event_id (int): event ID.
            offset (long long): position of data for event ID in generated
                Footprint binary file relative to beginning of that file.
            size (long long): size of data corresponding to event ID in
                generated Footprint binary file, long long.
        """
        with open(self.file_name, 'ab') as f:
            f.write(struct.pack(
                '=' + self.dtypes_list, event_id, offset, event_size)
            )


class FootprintBinFile(ModelFile):
    """
    Generate data for Footprint binary dummy model file.

    This file shows the intensity of a given event-areaperil combination. The
    binary footprint file footprint.bin requires the index file footprint.idx.

    Attributes:
        generate_data: Generate Footprint binary dummy model file data.
    """

    def __init__(
        self, num_events, num_areaperils, areaperils_per_event,
        num_intensity_bins, intensity_sparseness, no_intensity_uncertainty,
        random_seed, directory
    ):
        """
        Initialise Footprint binary file class.

        Args:
            num_events (int): number of events.
            num_areaperils (int): number of areaperils.
            areaperils_per_event (int): number of areaperils impacted per event.
            num_intensity_bins (int): number of intensity bins.
            intensity_sparseness (float): percentage of bins normalised to
                range [0,1] impacted for an event and areaperil.
            no_intensity_uncertainty (bool): flag to indicate whether more than
                one intensity bin can be impacted, bool.
            random_seed (float): random seed for random number generator.
            directory (str): dummy model file destination.
        """
        self.num_events = num_events
        self.num_areaperils = num_areaperils
        self.areaperils_per_event = areaperils_per_event
        self.num_intensity_bins = num_intensity_bins
        self.intensity_sparseness = intensity_sparseness
        self.no_intensity_uncertainty = no_intensity_uncertainty
        self.random_seed = random_seed
        self.file_name = os.path.join(directory, 'footprint.bin')
        self.event_id = 0
        self.start_stats = [
            {
                'desc': 'Number of intensity bins',
                'value': self.num_intensity_bins, 'dtype': 'i'
            },
            {
                'desc': 'Has Intensity Uncertainty',
                'value': not self.no_intensity_uncertainty, 'dtype': 'i'
            }
        ]
        self.dtypes = OrderedDict(
            [
                ('areaperil_id', 'i'),
                ('intensity_bin_id', 'i'),
                ('probability', 'f')
            ]
        )
        self.idx_file = FootprintIdxFile(directory)
        # Size of data is the same for all events
        self.size = 0
        for dtype in self.dtypes.values():
            self.size += struct.calcsize(dtype)
        if not self.no_intensity_uncertainty:
            self.size *= self.num_intensity_bins
        # Set initial offset
        self.offset = 0
        for stat in self.start_stats:
            self.offset += struct.calcsize(stat['dtype'])

    def generate_data(self):
        """
        Generate Footprint binary dummy model file data.

        Yields:
            areaperil (int): areaperil ID.
            intensity_bin (int): intensity bin ID.
            probability (float): impact probability.
        """
        super().seed_rng()
        for event in range(self.num_events):
            event_size = 0

            if self.areaperils_per_event == self.num_areaperils:
                selected_areaperils = np.arange(1, self.num_areaperils + 1)
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
                        1, self.num_intensity_bins + 1
                    )
                    probability = 1.0
                    event_size += self.size
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
                    total_probability = np.sum(probabilities)
                    if total_probability == 0:
                        continue   # No impacted intensity bins
                    event_size += self.size
                    probabilities /= total_probability

                    for intensity_bin, probability in enumerate(probabilities):
                        yield areaperil, intensity_bin + 1, probability

            self.idx_file.write_file(event + 1, self.offset, event_size)
            self.offset += event_size


class DamageBinDictFile(ModelFile):
    """
    Generate data for Damage Bin Dictionary dummy model file.

    This file shows the discretisation of the effective damageability cumulative
    distribution function.

    Attributes:
        generate_data: Generate Damage Bin Dictionary dummy model file data.
    """

    def __init__(self, num_damage_bins, directory):
        """
        Initialise Damage Bin Dictionary file class.

        Args:
            num_damage_bins (int): number of damage bins.
            directory (str): dummy model file destination.
        """
        self.num_damage_bins = num_damage_bins
        self.dtypes = OrderedDict([
            ('bin_index', 'i'), ('bin_from', 'f'), ('bin_to', 'f'),
            ('interpolation', 'f'), ('interval_type', 'i')
        ])
        self.start_stats = None
        self.data_length = num_damage_bins
        self.file_name = os.path.join(directory, 'damage_bin_dict.bin')

    def generate_data(self):
        """
        Generate Damage Bin Dictionary dummy model file data.

        First bin always runs from 0 to 0, i.e. has a midpoint (interpolation)
        of 0. Last bin always runs from 0 to 0, i.e. has a midpoint
        (interpolation) of 1.

        Yields:
            bin_id (int): damage bin ID.
            bin_from (float): damage bin lower limit.
            bin_to (float): damage bin upper limit.
            interpolation (float): damage bin midpoint.
            interval_type (int): interval_type (deprecated).
        """
        # Exclude first and last bins for now
        bin_indexes = np.arange(self.num_damage_bins - 2)
        bin_from_values = bin_indexes / (self.num_damage_bins - 2)
        bin_to_values = (bin_indexes + 1) / (self.num_damage_bins - 2)
        # Set interpolation in middle of bin
        interpolations = (0.5 + bin_indexes) / (self.num_damage_bins - 2)
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
    """
    Generate data for Occurrence dummy model file.

    This file maps events to periods, which can represent any length of time.

    Attributes:
        get_num_periods_from_truncated_normal_cdf: Get number of periods on
            event-by-event basis.
        get_num_periods_from_truncated_normal_cdf: Get number of periods from
            truncated normal cumulative distribution function.
        set_occ_date_id: Set date of occurrence in ktools format.
        generate_data: Generate Occurrence dummy model file data.
    """

    def __init__(
            self, num_events, num_periods, random_seed, directory, mean, stddev
    ):
        """
        Initialise Occurrence file class.

        Args:
            num_events (int): number of events.
            num_periods (int): total number of periods.
            random_seed (float): random seed for random number generator.
            directory (str): dummy model file destination.
            mean (float): mean of truncated normal distribution sampled to
                determine number of periods per event.
            stddev (float): standard deviation of truncated normal distribution
                sampled to determine number of periods per event.
        """
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
        self.mean = mean
        self.stddev = stddev
        self.file_name = os.path.join(directory, 'occurrence.bin')

    def get_num_periods_from_truncated_normal_cdf(self):
        """
        Get number of periods from truncated normal cumulative distribution
        function.

        Events can occur mupltiple times over multiple periods in the occurrence
        file. The number of periods per event is modelled by sampling from a
        truncated normal distribution with mean self.mean and standard deviation
        self.stddev. The lower tail of the distribution is truncated at 0.5 and
        the cumulative distribution function is given by:

        F(x) = [Phi(g(x)) - Phi(g(a))] / [Phi(g(b)) - Phi(g(a))]
        g(y) = (y - mean) / standard_deviation
        Phi(g(y)) = 1/2 * (1 + erf(g(y) / sqrt(2)))
        a = lower boundary = 0.5, b = upper boundary = infinity
          therefore g(b) -> infinity ===> Phi(g(b)) -> 1

        Returns:
            bound_a (int): lower boundary, when converted to an integer gives
                number of periods for this event.
        """
        alpha = (0.5 - self.mean) / self.stddev
        phi_alpha = 0.5 * (1 + erf(alpha / np.sqrt(2)))
        rand_no = np.random.random()
        bound_a = 0.5
        while True:
            xi = (bound_a - self.mean) / self.stddev
            phi_xi = 0.5 * (1 + erf(xi / np.sqrt(2)))
            if rand_no < ((phi_xi - phi_alpha) / (1 - phi_alpha)):
                return int(bound_a)
            bound_a += 1

    def get_num_periods_per_event(self):
        """
        Get number of periods on event-by-event basis.

        Determines whether sampling of truncated normal cumulative distribution
        function is required to obtain number of periods for this event.

        Returns:
            mean|bound_a (int): Number of periods for this event.
        """
        # Return mean if standard deviation is 0
        if self.stddev == 0:
            return self.mean
        else:
            return self.get_num_periods_from_truncated_normal_cdf()

    def set_occ_date_id(self, year, month, day):
        """
        Set date of occurrence in ktools format.

        Reduce year, month and day information to a single integer.

        Args:
            year (int): year.
            month (int): month.
            day (int): day.

        Returns:
            date (int): date in ktools format.
        """
        # Set date relative to epoch
        month = (month + 9) % 12
        year = year - month // 10
        return 365 * year + year // 4 - year // 100 + year // 400 + (306 * month + 5) // 10 + (day - 1)

    def generate_data(self):
        """
        Generate Occurrence dummy model file data.

        Yields:
            event (int): event ID.
            period_no (int): period number.
            date (int): date in ktools format.
        """
        super().seed_rng()
        months = np.arange(1, 13)
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        months_weights = np.array(days_per_month, dtype=float)
        months_weights /= months_weights.sum()   # Normalise
        for event in range(self.num_events):
            for _ in range(self.get_num_periods_per_event()):
                period_no = np.random.randint(1, self.num_periods + 1)
                occ_year = period_no   # Assume one period represents one year
                occ_month = np.random.choice(months, p=months_weights)
                occ_day = np.random.randint(1, days_per_month[occ_month - 1])
                occ_date = self.set_occ_date_id(occ_year, occ_month, occ_day)
                yield event + 1, period_no, occ_date


class RandomFile(ModelFile):
    """
    Generate data for Random Numbers dummy model file.

    This optional file contains random numbers for ground up loss sampling.

    Attributes:
        generate_data: Generate Random Numbers dummy model file data.
    """

    def __init__(self, num_randoms, random_seed, directory):
        """
        Initialise Random Numbers file class.

        Args:
            num_randoms (int): number of random numbers.
            random_seed (float): random seed for random number generator.
            directory (str): dummy model file destination.
        """
        self.num_randoms = num_randoms
        self.dtypes = {'random_no': 'f'}
        self.start_stats = None
        self.random_seed = random_seed
        self.data_length = num_randoms
        self.file_name = os.path.join(directory, 'random.bin')

    def generate_data(self):
        """
        Generate Random Numbers dummy model file data.

        Yields:
            random number (float): random number.
        """
        super().seed_rng()
        # First random number is 0
        return (tuple([np.random.uniform()]) if i != 0 else (0,) for i in range(self.num_randoms))


class CoveragesFile(ModelFile):
    """
    Generate data for Coverages dummy model Oasis file.

    This file maps coverage IDs to Total Insured Values.

    Attributes:
        generate_data: Generate Coverages dummy model Oasis file data.
    """

    def __init__(
        self, num_locations, coverages_per_location, random_seed, directory
    ):
        """
        Initialise Coverages file class.

        Args:
            num_locations (int): number of locations.
            coverages_per_location (int): number of coverage types per location.
            random_seed (float): random seed for random number generator.
            directory (str): dummy model file destination.
        """
        self.num_locations = num_locations
        self.coverages_per_location = coverages_per_location
        self.dtypes = {'tiv': 'f'}
        self.start_stats = None
        self.random_seed = random_seed
        self.data_length = num_locations * coverages_per_location
        self.file_name = os.path.join(directory, 'coverages.bin')

    def generate_data(self):
        """
        Generate Coverages dummy model file data.

        Yields:
            total insured value (float): Total Insured Value (TIV).
        """
        super().seed_rng()
        # Assume 1-1 mapping between item and coverage IDs
        return (
            tuple([np.random.uniform(1, 1000000)]) for _ in range(
                self.num_locations * self.coverages_per_location
            )
        )


class ItemsFile(ModelFile):
    """
    Generate data for Items dummy model Oasis file.

    This file lists the exposure items for which ground up loss will be sampled.

    Attributes:
        generate_data: Generate Items dummy model Oasis file data.
    """

    def __init__(
        self, num_locations, coverages_per_location, num_areaperils,
        num_vulnerabilities, random_seed, directory
    ):
        """
        Initialise Items file class.

        Args:
            num_locations (int): number of locations.
            coverages_per_location (int): number of coverage types per location.
            num_areaperils (int): number of areaperils.
            num_vulnerabilities (int): number of vulnerabilities.
            random_seed (float): random seed for random number generator.
            directory (str): dummy model file destination.
        """
        self.num_locations = num_locations
        self.coverages_per_location = coverages_per_location
        self.num_areaperils = num_areaperils
        self.num_vulnerabilities = num_vulnerabilities
        self.dtypes = OrderedDict([
            ('item_id', 'i'), ('coverage_id', 'i'), ('areaperil_id', 'i'),
            ('vulnerability_id', 'i'), ('group_id', 'i'),
        ])
        self.start_stats = None
        self.random_seed = random_seed
        self.data_length = num_locations * coverages_per_location
        self.file_name = os.path.join(directory, 'items.bin')

    def generate_data(self):
        """
        Generate Items dummy model file data.

        Yields:
            item (int): item ID.
            item (int): coverage ID = item ID (1-1 mapping).
            areaperils[coverage] (int): areaperil ID corresponding to
                coverage ID.
            vulnerabilities[coverage] (int): vulnerability ID corresponding to
                coverage ID.
            location (int): group ID mapped to location ID.
        """
        super().seed_rng()
        for location in range(self.num_locations):
            areaperils = np.random.randint(
                1, self.num_areaperils + 1, size=self.coverages_per_location
            )
            vulnerabilities = np.random.randint(
                1, self.num_vulnerabilities + 1, size=self.coverages_per_location
            )
            for coverage in range(self.coverages_per_location):
                item = self.coverages_per_location * location + coverage + 1
                # Assume 1-1 mapping between item and coverage IDs
                # Assume group ID mapped to location
                yield item, item, areaperils[coverage], vulnerabilities[coverage], location + 1


class AmplificationsFile(ModelFile):
    """
    Generate data for Amplifications dummy model Oasis file.

    This file maps exposure items to amplification IDs.

    Attributes:
        generate_data: Generate Amplifications dummy model Oasis file data.
        write_file: Write data to Amplifications dummy model Oasis file in
          binary format.
    """

    def __init__(
        self, num_locations, coverages_per_location, num_amplifications,
        random_seed, directory
    ):
        """
        Initialise AmplificationsFile class.

        Args:
            num_locations (int): number of locations.
            coverages_per_location (int): number of coverage types per location.
            num_amplifications (int): number of amplifications.
            random_ssed (float): random seed for random number generator.
            directory (str): dummy model file destination.
        """
        self.num_items = coverages_per_location * num_locations
        self.num_amplifications = num_amplifications
        self.random_seed = random_seed
        self.file_name = os.path.join(directory, 'amplifications.bin')
        self.start_stats = [
            {
                'desc': 'Reserved for fuiure use', 'value': 0, 'dtype': 'i'
            }
        ]
        self.dtypes = OrderedDict([('item_id', 'i'), ('amplification_id', 'i')])

    def generate_data(self):
        """
        Generate Amplifications dummy model Oasis file data.

        Yields:
            item (int): item ID
            amplification (int): amplification ID
        """
        super().seed_rng()
        for item in range(self.num_items):
            amplification = np.random.randint(1, self.num_amplifications + 1)
            yield item + 1, amplification

    def write_file(self):
        """
        Write data to output Amplifications file in binary format.

        Checks number of amplifications are greater than 0 before calling base
        class method.
        """
        if not self.num_amplifications:
            return
        super().write_file()


class FMFile(ModelFile):
    """
    Parent class for generating random data for Financial Model files.
    """

    def __init__(self, num_locations, coverages_per_location):
        """
        Initialise Financial Model files classes.

        Args:
            num_locations (int): number of locations.
            coverages_per_location (int): number of coverage types per location.
        """
        self.num_locations = num_locations
        self.coverages_per_location = coverages_per_location
        self.start_stats = None


class FMProgrammeFile(FMFile):
    """
    Generate data for Financial Model Programme dummy model Oasis file.

    This file shows the level hierarchy.

    Attributes:
        generate_data: Generate Financial Model Programme dummy model Oasis file
            data.
    """

    def __init__(self, num_locations, coverages_per_location, directory):
        """
        Initialise Financial Model Programme file class.

        Args:
            num_locations (int): number of locations.
            coverages_per_location (int): number of coverage types per location.
            directory (str): dummy model file destination.
        """
        super().__init__(num_locations, coverages_per_location)
        self.dtypes = OrderedDict([
            ('from_agg_id', 'i'), ('level_id', 'i'), ('to_agg_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location * 2   # 2 from number of levels
        self.file_name = os.path.join(directory, 'fm_programme.bin')

    def generate_data(self):
        """
        Generate Financial Model Programme dummy model file data.

        Yields:
            agg_id (int): from aggregate ID.
            level (int): level ID.
            agg_id (int): to aggregate ID.
        """
        levels = [1, 10]
        levels = range(1, len(levels) + 1)
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
    """
    Generate data for Financial Model Policy dummy model Oasis file.

    This file shows the calculation rule (from the Financial Model Policy file)
    that should be applied to aggregations of loss at a particular level.

    Attributes:
        generate_data: Generate Financial Model Policy dummy model Oasis file
            data.
    """

    def __init__(
        self, num_locations, coverages_per_location, num_layers, directory
    ):
        """
        Initialise Financial Model Policy file class.

        Args:
            num_locations (int): number of locations.
            coverages_per_location (int): number of coverage types per location.
            num_layers (int): number of layers.
            directory (str): dummy model file destination.
        """
        super().__init__(num_locations, coverages_per_location)
        self.num_layers = num_layers
        self.dtypes = OrderedDict([
            ('layer_id', 'i'), ('level_id', 'i'), ('agg_id', 'i'),
            ('policytc_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location + num_layers
        self.file_name = os.path.join(directory, 'fm_policytc.bin')

    def generate_data(self):
        """
        Generate Financial Model Policy dummy model file data.

        Yields:
            level (int): level ID.
            agg_id (int): aggregate ID.
            layer (int): layer ID.
            policytc_id (int): profile/policyTC ID.
        """
        # Site coverage #1 & policy layer #10 FM levels
        levels = [1, 10]
        levels = range(1, len(levels) + 1)
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
                    yield level, 1, layer + 1, policytc_id
                    policytc_id += 1   # Next policytc_id


class FMProfileFile(ModelFile):
    """
    Generate data for Financial Model Profile dummy model Oasis file.

    This file contains the list of calculation rules with profile values used
    to generate insurance losses.

    Attributes:
        generate_data: Generate Financial Model Profile dummy model Oasis file
            data.
    """

    def __init__(self, num_layers, directory):
        """
        Initialise Financial Model Profile file class.

        Args:
            num_layers (int): number of layers.
            directory (str): dummy model file destination.
        """
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
        """
        Generate Financial Model Profile dummy model file data.

        Yields:
            policytc_id (int): profile/policyTC ID.
            calculation rule ID (int): calculation rule ID (2 or 100).
            first deductible (float): first deductible (fixed at 0.0).
            second deductible (float): second deductible (fixed at 0.0).
            third deductible (float): third deductible (fixed at 0.0).
            attachment1 (float): attachment point/excess.
            limit1 (float): limit.
            first proportional share (float): first proportional
                share (0.0 or 0.3).
            second proportional share (float): second proportional
                share (fixed at 0.0).
            third proportional share (float): third proportional
                share (fixed at 0.0).
        """
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
            if (layer + 1) == self.num_layers:
                limit1 = max_limit1
            else:
                limit1 = attachment1_offset * (layer + 1)
            profile_rows.append(
                (policytc_id, 2, 0.0, 0.0, 0.0, attachment1, limit1, 0.3, 0.0, 0.0)
            )
        for row in profile_rows:
            yield row


class FMXrefFile(FMFile):
    """
    Generate data for Financial Model Cross Reference dummy model Oasis file.

    This file shows the mapping between the financial model output ID, and
    aggregate and layer IDs.

    Attributes:
        generate_data: Generate Financial Model Cross Reference dummy model
            Oasis file data.
    """

    def __init__(
        self, num_locations, coverages_per_location, num_layers, directory
    ):
        """
        Initialise Financial Model Cross Reference file class.

        Args:
            num_locations (int): number of locations.
            coverages_per_location (int): number of coverage types per location.
            num_layers (int): number of layers.
            directory (str): dummy model file destination.
        """
        super().__init__(num_locations, coverages_per_location)
        self.num_layers = num_layers
        self.dtypes = OrderedDict([
            ('output', 'i'), ('agg_id', 'i'), ('layer_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location * num_layers
        self.file_name = os.path.join(directory, 'fm_xref.bin')

    def generate_data(self):
        """
        Generate Financial Model Cross Reference dummy model file data.

        Yields:
            output_count (int): output ID.
            agg_id (int): aggregate ID.
            layer (int): layer ID.
        """
        layers = range(1, self.num_layers + 1)
        output_count = 1
        for agg_id in range(
            1, self.num_locations * self.coverages_per_location + 1
        ):
            for layer in layers:
                yield output_count, agg_id, layer
                output_count += 1


class GULSummaryXrefFile(FMFile):
    """
    Generate data for Ground Up Losses Summary Cross Reference dummy model Oasis
    file.

    This file shows how item ground up losses are summed together at various
    summary levels in summarycalc.

    Attributes:
        generate_data: Generate Ground Up Losses Summary Cross Reference dummy
            model Oasis file data.
    """

    def __init__(self, num_locations, coverages_per_location, directory):
        """
        Initialise Ground Up Losses Summary Cross Reference file class.

        Args:
            num_locations (int): number of locations.
            coverages_per_location (int): number of coverage types per location.
            directory (str): dummy model file destination.
        """
        super().__init__(num_locations, coverages_per_location)
        self.dtypes = OrderedDict([
            ('item_id', 'i'), ('summary_id', 'i'), ('summaryset_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location
        self.file_name = os.path.join(directory, 'gulsummaryxref.bin')

    def generate_data(self):
        """
        Generate Ground Up Losses Summary Cross Reference dummy model file data.

        Yields:
            item (int): item ID.
            summary_id (int): summary ID.
            summaryset_id (int): summary set ID.
        """
        summary_id = 1
        summaryset_id = 1
        for item in range(self.num_locations * self.coverages_per_location):
            yield item + 1, summary_id, summaryset_id


class FMSummaryXrefFile(FMFile):
    """
    Generate data for Financial Model Summary Cross Reference dummy model Oasis
    file.

    This file shows how insurance losses are summed together at various levels
    by summarycalc.

    Attributes:
        generate_data: Generate Financial Model Summary Cross Reference dummy
            model Oasis file data.
    """

    def __init__(
        self, num_locations, coverages_per_location, num_layers, directory
    ):
        """
        Initialise Financial Model Summary Cross Reference file class.

        Args:
            num_locations (int): number of locations.
            coverages_per_location (int): number of coverage types per location.
            num_layers (int): number of layers.
            directory (str): dummy model file destination.
        """
        super().__init__(num_locations, coverages_per_location)
        self.num_layers = num_layers
        self.dtypes = OrderedDict([
            ('output_id', 'i'), ('summary_id', 'i'), ('summaryset_id', 'i')
        ])
        self.data_length = num_locations * coverages_per_location * num_layers
        self.file_name = os.path.join(directory, 'fmsummaryxref.bin')

    def generate_data(self):
        """
        Generate Financial Model Summary Cross Reference dummy model file data.

        Yields:
            output_id (int): output ID.
            summary_id (int): summary ID.
            summaryset_id (int): summary set ID.
        """
        summary_id = 1
        summaryset_id = 1
        for output_id in range(
            self.num_locations * self.coverages_per_location * self.num_layers
        ):
            yield output_id + 1, summary_id, summaryset_id
