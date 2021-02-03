"""Funtionality for outputting keys results to file."""

import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
from csv import DictWriter
from functools import partial

from ..utils.status import OASIS_KEYS_STATUS


class BaseKeysOutputStrategy(ABC):
    """A base class for writing yeys results out to file.

    Inheriting classes will implement the detail of individual file formats.
    """

    def __init__(self, keys_file, keys_errors_file=None):
        """Initialise BaseKeysOutputStrategy class.

        Args:
            keys_file (file-like): File object to write successful results to.
            keys_errors_file (file-like, optional): File object to write non-successful results to.
                If None or not specified, non-successful results are not written.

        Raises:
            TypeError: A file path is passed, rather than a file object.

        """
        self.keys_file = keys_file
        self.keys_errors_file = keys_errors_file

        if isinstance(self.keys_file, str):
            raise TypeError("keys_file must be a file object")
        if self.keys_errors_file is not None and isinstance(self.keys_errors_file, str):
            raise TypeError("keys_errors_file must be a file object or None")

    @property
    def do_nonsuccess_output(self):
        """Whether to write non-successful keys results to file."""
        return bool(self.keys_errors_file)

    @abstractmethod
    def write(self, results):
        """Write keys results to file.

        If both a keys filepath and a keys error filepath are provided to the initialiser, then
        results are written to one file or the other. If no keys error filepath is provided, then
        those results are discarded, although they are still counted.

        Args:
            results (iterable of dict): Iterable of results dictionary instances.

        Returns:
            int, int: Number of successes and non-successes encountered.
        """
        pass


class JSONKeysOutputStrategy(BaseKeysOutputStrategy):
    """Class for outputting keys results in JSON format."""

    def write(self, results):
        """Write keys results to file in JSON format.

        If both a keys filepath and a keys error filepath are provided to the initialiser, then
        results are written to one file or the other. If no keys error filepath is provided, then
        those results are discarded, although they are still counted.

        Args:
            results (iterable of dict): Iterable of results dictionary instances.

        Returns:
            int, int: Number of successes and non-successes encountered.
        """
        successes = []
        nonsuccesses = []
        success_status = OASIS_KEYS_STATUS['success']['id']
        # Convert keys results to lists, to allow output of JSON.
        # Note that this will collect all keys results in memory, which will fail for sufficiently
        # large portfolios.
        for r in results:
            successes.append(r) if r['status'] == success_status else nonsuccesses.append(r)

        n_successes = len(successes)
        n_nonsuccesses = len(nonsuccesses)

        self._write_json_records(self.keys_file, successes)
        if self.do_nonsuccess_output:
            self._write_json_records(self.keys_errors_file, nonsuccesses)

        return n_successes, n_nonsuccesses

    def _write_json_records(self, file_handle, results):
        file_handle.write(json.dumps(results, sort_keys=True, indent=4, ensure_ascii=False))


class CSVKeysOutputStrategy(BaseKeysOutputStrategy):
    """Class for outputting keys results in CSV format."""

    KEYS_FIELD_NAMES__STANDARD_MODEL = (
        'LocID',
        'PerilID',
        'CoverageTypeID',
        'AreaPerilID',
        'VulnerabilityID',
    )
    KEYS_FIELD_NAMES__CUSTOM_MODEL = (
        'LocID',
        'PerilID',
        'CoverageTypeID',
        'ModelData',
    )
    KEYS_ERRORS_FIELD_NAMES = (
        'LocID',
        'PerilID',
        'CoverageTypeID',
        'Status',
        'Message',
    )

    KEYS_NAME_MAPPING = OrderedDict([
        ('loc_id', 'LocID'),
        ('peril_id', 'PerilID'),
        ('coverage_type', 'CoverageTypeID'),
        ('area_peril_id', 'AreaPerilID'),
        ('vulnerability_id', 'VulnerabilityID'),
        ('model_data', 'ModelData'),
        ('status', 'Status'),
        ('message', 'Message'),
    ])

    def __init__(self, keys_file, keys_errors_file=None):
        super().__init__(keys_file, keys_errors_file=keys_errors_file)
        self.keys_success_mapping = OrderedDict()
        self.keys_nonsuccess_mapping = OrderedDict()

    def write(self, results):
        """Write keys results to file in JSON format.

        If both a keys filepath and a keys error filepath are provided to the initialiser, then
        results are written to one file or the other. If no keys error filepath is provided, then
        those results are discarded, although they are still counted.

        Args:
            results (iterable of dict): Iterable of results dictionary instances.

        Returns:
            int, int: Number of successes and non-successes encountered.
        """
        n_successes = 0
        n_nonsuccesses = 0

        success_status = OASIS_KEYS_STATUS['success']['id']
        is_first_row = True
        for r in results:
            if is_first_row:
                write_success, write_nonsuccess = self._set_up_output(r)
                is_first_row = False

            if r['status'] == success_status:
                n_successes += 1
                write_success(r)
            else:
                n_nonsuccesses += 1
                write_nonsuccess(r)

        return n_successes, n_nonsuccesses

    def _write_csv_row(self, csv_writer, key_mapping, row):
        """Write a keys result to file.

        If csv_writer is None, no action is taken. This is used when a keys-errors file is not
        written, and hence all nonsuccessful results are discarded.

        Args:
            csv_writer (csv.DictWriter): CSV output writer.
            key_mapping (dict): Dictionary mapping row keys to the keys being written to the CSV
                file.
            row (dict): Keys result row to write.

        """
        if not csv_writer:
            return

        row_mapped = {
            new_key: row[old_key] for old_key, new_key in key_mapping.items()
            if old_key in row
        }

        csv_writer.writerow(row_mapped)

    def _set_up_output(self, example_record):
        """Set up CSV output for keys file, dependent on the example record format.

        Args:
            example_record (dict): Example keys record, from which the format is determined.

        Returns:
            callable, callable: Functions to call to write successful and nonsuccessful results,
                respectively.

        """
        if 'model_data' in example_record:
            success_header_row = self.KEYS_FIELD_NAMES__CUSTOM_MODEL
        else:
            success_header_row = self.KEYS_FIELD_NAMES__STANDARD_MODEL

        self._success_writer = DictWriter(self.keys_file, fieldnames=success_header_row)
        self._success_writer.writeheader()
        self._populate_keys_mapping_dict(
            self.keys_success_mapping,
            success_header_row,
        )
        write_success = partial(
            self._write_csv_row,
            self._success_writer,
            self.keys_success_mapping,
        )

        if self.do_nonsuccess_output:
            self._nonsuccess_writer = DictWriter(
                self.keys_errors_file,
                fieldnames=self.KEYS_ERRORS_FIELD_NAMES,
            )
            self._nonsuccess_writer.writeheader()
            self._populate_keys_mapping_dict(
                self.keys_nonsuccess_mapping,
                self.KEYS_ERRORS_FIELD_NAMES,
            )
            write_nonsuccess = partial(
                self._write_csv_row,
                self._nonsuccess_writer,
                self.keys_nonsuccess_mapping,
            )

        else:
            write_nonsuccess = None

        return write_success, write_nonsuccess

    def _populate_keys_mapping_dict(self, dict_to_populate, fields_to_use):
        for old_key, new_key in self.KEYS_NAME_MAPPING.items():
            if new_key not in fields_to_use:
                continue
            dict_to_populate[old_key] = new_key
