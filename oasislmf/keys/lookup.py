# -*- coding: utf-8 -*-

from __future__ import unicode_literals, absolute_import

__all__ = [
    'OasisKeysLookupFactory',
    'OasisBaseKeysLookup',
]

import csv
import importlib
import io
import json
import os
import re
import sys

from collections import OrderedDict

import pandas as pd

from six import StringIO

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.status import KEYS_STATUS_NOMATCH, KEYS_STATUS_SUCCESS

UNKNOWN_ID = -1


class OasisBaseKeysLookup(object):  # pragma: no cover
    """
    A base class / interface that serves a template for model-specific keys
    lookup classes.
    """
    @oasis_log()
    def __init__(
        self,
        keys_data_directory=None,
        supplier=None,
        model_name=None,
        model_version=None
    ):
        """
        Class constructor
        """
        if keys_data_directory is not None:
            self.keys_data_directory = keys_data_directory
        else:
            self.keys_data_directory = os.path.join(os.sep, 'var', 'oasis', 'keys_data')

        self.supplier = supplier
        self.model_name = model_name
        self.model_version = model_version

    @oasis_log()
    def process_locations(self, loc_df):
        """
        Process location rows - passed in as a pandas dataframe.
        """
        pass

    def _get_location_record(self, raw_loc_item):
        """
        Returns a dict of standard location keys and values based on
        a raw location item, which is a row in a Pandas dataframe.
        """
        pass

    def _get_area_peril_id(self, record):
        """
        Get the area peril ID for a particular location record.
        """
        return UNKNOWN_ID, "Not implemented"

    def _get_vulnerability_id(self, record):
        """
        Get the vulnerability ID for a particular location record.
        """
        return UNKNOWN_ID, "Not implemented"

    @oasis_log()
    def _get_area_peril_ids(self, loc_data, include_context=True):
        """
        Generates area peril IDs in two modes - if include_context is
        True (default) it will generate location records/rows including
        the area peril IDs, otherwise it will generate pairs of location
        IDs and the corresponding area peril IDs.
        """
        pass

    @oasis_log()
    def _get_vulnerability_ids(self, loc_data, include_context=True):
        """
        Generates vulnerability IDs in two modes - if include_context is
        True (default) it will generate location records/rows including
        the area peril IDs, otherwise it will generate pairs of location
        IDs and the corresponding vulnerability IDs.
        """
        pass

    def _get_lookup_success(self, ap_id, vul_id):
        """
        Determine the status of the keys lookup.
        """
        if ap_id == UNKNOWN_ID or vul_id == UNKNOWN_ID:
            return KEYS_STATUS_NOMATCH
        return KEYS_STATUS_SUCCESS


class OasisKeysLookupFactory(object):
    """
    A factory class to load and run keys lookup services for different
    models/suppliers.
    """
    @classmethod
    def get_model_info(cls, model_version_file_path):
        """
        Get model information from the model version file.
        """
        with io.open(model_version_file_path, 'r', encoding='utf-8') as f:
            return next(csv.DictReader(
                f, fieldnames=['supplier_id', 'model_id', 'model_version_id']
            ))

    @classmethod
    def get_lookup_package(cls, lookup_package_path):
        """
        Returns the lookup service parent package (called `keys_server` and
        located in `src` in the model keys server Git repository or in
        `var/www/oasis` in the keys server Docker container) from the given
        path.
        """
        parent_dir = os.path.abspath(os.path.dirname(lookup_package_path))
        package_name = re.sub(r'\.py$', '', os.path.basename(lookup_package_path))

        sys.path.insert(0, parent_dir)
        lookup_package = importlib.import_module(package_name)
        sys.path.pop(0)

        return lookup_package

    @classmethod
    def get_lookup_class_instance(cls, lookup_package, keys_data_path, model_info):
        """
        Get the keys lookup class instance.
        """
        klc = getattr(lookup_package, '{}KeysLookup'.format(model_info['model_id']))

        return klc(
            keys_data_directory=keys_data_path,
            supplier=model_info['supplier_id'],
            model_name=model_info['model_id'],
            model_version=model_info['model_version_id']
        )

    @classmethod
    def get_model_exposures(cls, model_exposures=None, model_exposures_file_path=None):
        """
        Get the model exposures/location file data as a pandas dataframe given
        either the path of the model exposures file or the string contents of
        such a file.
        """
        if model_exposures_file_path:
            loc_df = pd.read_csv(os.path.abspath(model_exposures_file_path), float_precision='high')
        elif model_exposures:
            loc_df = pd.read_csv(StringIO(model_exposures), float_precision='high')
        else:
            raise OasisException('Either model_exposures_file_path or model_exposures must be specified')

        loc_df = loc_df.where(loc_df.notnull(), None)
        loc_df.columns = loc_df.columns.str.lower()

        return loc_df

    @classmethod
    def write_oasis_keys_file(cls, records, output_file_path):
        """
        Writes an Oasis keys file from an iterable of keys records.
        """
        heading_row = OrderedDict([
            ('id', 'LocID'),
            ('peril_id', 'PerilID'),
            ('coverage', 'CoverageID'),
            ('area_peril_id', 'AreaPerilID'),
            ('vulnerability_id', 'VulnerabilityID'),
        ])

        pd.DataFrame(
            columns=heading_row.keys(),
            data=[heading_row] + records,
        ).to_csv(
            output_file_path,
            index=False,
            encoding='utf-8',
            header=False,
        )

        return output_file_path, len(records)

    @classmethod
    def write_list_keys_file(cls, records, output_file_path):
        """
        Writes the keys records as a simple list to file.
        """
        with io.open(output_file_path, 'w', encoding='utf-8') as f:
            count = 0
            for count, record in enumerate(records, start=1):
                f.write('{},\n'.format(json.dumps(record, sort_keys=True, indent=4)))

            return f.name, count

    @classmethod
    def create(
        cls,
        model_keys_data_path=None,
        model_version_file_path=None,
        lookup_package_path=None,
    ):
        """
        Creates a keys lookup class instance for the given model and supplier -
        local file paths are required for the model keys data folder, the model
        version file and the Git repository for the model keys server. Returns a
        pair ``(model_info, klc)``, where ``model_info`` is a dictionary holding
        model information from the model version file and `klc` is the lookup
        service class instance for the model.
        """
        if model_keys_data_path:
            model_keys_data_path = os.path.abspath(model_keys_data_path)

        if model_version_file_path:
            model_version_file_path = os.path.abspath(model_version_file_path)

        if lookup_package_path:
            lookup_package_path = os.path.abspath(lookup_package_path)

        model_info = cls.get_model_info(model_version_file_path)
        lookup_package = cls.get_lookup_package(lookup_package_path)
        klc = cls.get_lookup_class_instance(lookup_package, model_keys_data_path, model_info)
        return model_info, klc

    @classmethod
    def get_keys(
        cls,
        lookup=None,
        model_exposures=None,
        model_exposures_file_path=None,
        success_only=True
    ):
        """
        Generates keys keys records (JSON) for the given model and supplier -
        requires an instance of the lookup service (which can be created using
        the `create` method in this factory class), and either the model
        location file path or the string contents of such a file.

        The optional keyword argument ``success_only`` indicates whether only
        records with successful lookups should be returned (default), or all
        records.
        """
        if not (model_exposures or model_exposures_file_path):
            raise OasisException('No model exposures provided')

        model_loc_df = cls.get_model_exposures(
            model_exposures_file_path=model_exposures_file_path,
            model_exposures=model_exposures,
        )

        for record in lookup.process_locations(model_loc_df):
            if success_only:
                if record['status'].lower() == 'success':
                    yield record
            else:
                yield record

    @classmethod
    def save_keys(
        cls,
        lookup=None,
        output_file_path=None,
        model_exposures=None,
        model_exposures_file_path=None,
        success_only=True
    ):
        """
        Writes the keys keys records generated by the lookup service for the
        given model and supplier to a local file - requires a lookup service
        instance (which can be created using the `create` method in this
        factory class), the path of the model location file, the path of
        output file, and the format of the output file which can be an
        Oasis keys file (``oasis_keys``) or a simple listing of the records
        to file (``list_keys``).

        The optional keyword argument ``success_only`` indicates whether only
        records with successful lookups should be returned (default), or all
        records.

        Returns a pair ``(f, n)`` where ``f`` is the output file object
        and ``n`` is the number of records written to the file.
        """
        if not (model_exposures or model_exposures_file_path):
            raise OasisException('No model exposures or model exposures file path provided')

        output_file_path = os.path.abspath(output_file_path)

        keys = cls.get_keys(
            lookup=lookup,
            model_exposures=model_exposures,
            model_exposures_file_path=model_exposures_file_path,
            success_only=success_only
        )

        return cls.write_oasis_keys_file(list(keys), output_file_path)
