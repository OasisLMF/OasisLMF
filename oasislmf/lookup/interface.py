__all__ = [
    'KeyServerInterface',
    'OasisLookupInterface',
]

# 'OasisBaseKeysLookup' -> OasisLookupInterface


import os
import abc

from ..utils.log import oasis_log
from ..utils.status import OASIS_KEYS_STATUS


''' Interface class for developing custom key server or key lookup code
'''


class KeyServerInterface(metaclass=abc.ABCMeta):
    """
    Interface to implement to create a KeyServer
    It define the method to be implemented to be used correctly in lookup.factory.KeyServerFactory
    all classes must:
     - specify the version of the interface they use
     - implement the init method
     - implement the generate_key_files method

    """
    interface_version = "1"

    @abc.abstractmethod
    def __init__(self, config, config_dir, user_data_dir, output_dir):
        """
        During the key generation step, the generic factory will call the constructor of the lookup class with the
        following parameters.

        :param config: contains all the information necessary to run the model
        :type config: dict

        :param config_dir: path to the model directory, can be used to locate relative path to all the files
                           that serve as base for the model
        :type config_dir: str

        :param user_data_dir: Path to additional data necessary for the model that can vary from analysis to analysis
        :type user_data_dir: str

        :param output_dir: Path to the analysis output directory, can be use to write additional files that are produce
                           during the keys file generation

        """

        raise NotImplementedError

    @abc.abstractmethod
    def generate_key_files(self,
                           location_fp,
                           successes_fp,
                           errors_fp=None,
                           output_format='oasis',
                           keys_success_msg=False,
                           multiproc_enabled=True,
                           multiproc_num_cores=-1,
                           multiproc_num_partitions=-1,
                           **kwargs):
        """
        Writes a keys file, and optionally a keys error file.

        :param location_fp: path to the locations file
        :type location_fp: str

        :param successes_fp: path to the success keys file
        :type successes_fp: str

        :param errors_fp: path to the error keys file (optional)
        :type errors_fp: str

        :param output_format: format of the keys files (oasis or json)
        :type output_format: str

        :param keys_success_msg: option to write msg for success key
        :type keys_success_msg: bool

        :param multiproc_enabled: option to run with multiple processor
        :type multiproc_enabled: bool

        :param multiproc_num_cores: number of cores to use in multiproc mode
        :type multiproc_num_cores: int

        :param multiproc_num_partitions: number of partition to create in multiproc mode
        :type multiproc_num_partitions: int

        If ``keys_errors_file_path`` is not present then the method returns a
        pair ``(p, n)`` where ``p`` is the keys file path and ``n`` is the
        number of "successful" keys records written to the keys file, otherwise
        it returns a quadruple ``(p1, n1, p2, n2)`` where ``p1`` is the keys
        file path, ``n1`` is the number of "successful" keys records written to
        the keys file, ``p2`` is the keys errors file path and ``n2`` is the
        number of "unsuccessful" keys records written to keys errors file.
        """
        raise NotImplementedError


class KeyLookupInterface(metaclass=abc.ABCMeta):
    """Interface for KeyLookup
    it define the interface to be used correctly by lookup.factory.BasicKeyServer
    all classes must:
     - specify the version of the interface they use
     - implement the init method
     - implement the process_location method
    """

    interface_version = "1"

    @abc.abstractmethod
    def __init__(self, config, config_dir, user_data_dir, output_dir):
        raise NotImplementedError

    @abc.abstractmethod
    def process_locations(self, loc_df):
        """
        Process location rows - passed in as a pandas dataframe.
        """
        raise NotImplementedError


class OasisLookupInterface:  # pragma: no cover
    """
    Old Oasis base class -deprecated
    If you were using this interface, you can make you class inherit from the new abstract class AbstractBasicKeyServer
    or implement the KeyServerInterface interface
    """
    interface_version = "0"

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
