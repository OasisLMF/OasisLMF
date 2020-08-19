__all__ = [
    'OasisLookupInterface',
]

# 'OasisBaseKeysLookup' -> OasisLookupInterface


import os

from ..utils.log import oasis_log
from ..utils.status import OASIS_KEYS_STATUS

''' Interface class for developing custom lookup code
'''


class OasisLookupInterface(object):  # pragma: no cover
    """
    Old Oasis base class -deprecated
    """
    @oasis_log()
    def __init__(
        self,
        keys_data_directory=None,
        supplier=None,
        model_name=None,
        model_version=None,
        complex_lookup_config_fp=None,
        output_directory=None
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
        self.complex_lookup_config_fp = complex_lookup_config_fp
        self.output_directory = output_directory
        self.UNKNOWN_ID = -1

    @oasis_log()
    def process_locations(self, loc_df):
        """
        Process location rows - passed in as a pandas dataframe.
        """
        pass

    def _get_area_peril_id(self, record):
        """
        Get the area peril ID for a particular location record.
        """
        return self.UNKNOWN_ID, "Not implemented"

    def _get_vulnerability_id(self, record):
        """
        Get the vulnerability ID for a particular location record.
        """
        return self.UNKNOWN_ID, "Not implemented"

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

    def _get_custom_lookup_success(self, ap_id, vul_id):
        """
        Determine the status of the keys lookup.
        """
        if ap_id == self.UNKNOWN_ID or vul_id == self.UNKNOWN_ID:
            return OASIS_KEYS_STATUS['nomatch']['id']
        return OASIS_KEYS_STATUS['success']['id']
