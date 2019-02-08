# -*- coding: utf-8 -*-

from itertools import chain

__all__ = [
    'OasisFilesPipeline'
]


class OasisFilesPipeline(object):
    def __init__(
        self,
        model_key=None,
        source_exposure_file_path=None,
        source_accounts_file_path=None,
        keys_file_path=None,
        keys_errors_file_path=None
    ):
        self._model_key = model_key
        self._source_exposure_file_path = source_exposure_file_path
        self._source_accounts_file_path = source_accounts_file_path
        self._keys_file_path = keys_file_path
        self._keys_errors_file_path = keys_errors_file_path

        self._items_file_path = None
        self._coverages_file_path = None
        self._gulsummaryxref_file_path = None

        self._fm_policytc_file_path = None
        self._fm_profile_file_path = None
        self._fm_programme_file_path = None
        self._fm_xref_file_path = None
        self._fmsummaryxref_file_path = None

        self._source_files = {
            'source_exposure': self._source_exposure_file_path,
            'source_accounts': self._source_accounts_file_path 
        }

        self._keys_files = {
            'keys': self._keys_file_path,
            'keys_errors': self._keys_errors_file_path
        }

        self._gul_input_files = {
            'items': self._items_file_path,
            'coverages': self._coverages_file_path,
            'gulsummaryxref': self._gulsummaryxref_file_path
        }

        self._fm_input_files = {
            'fm_policytc': self._fm_policytc_file_path,
            'fm_profile': self._fm_profile_file_path,
            'fm_programme': self._fm_programme_file_path,
            'fm_xref': self._fm_xref_file_path,
            'fmsummaryxref': self._fmsummaryxref_file_path
        }

        self._oasis_files = {k:v for k, v in chain(self._gul_input_files.items(), self._fm_input_files.items())}

    def __str__(self):
        return '{}: {}'.format(self.__repr__(), self.model_key)

    def __repr__(self):
        return '{}: {}'.format(self.__class__, self.__dict__)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')

    @property
    def model_key(self):
        """
        Model key property - getter only.

            :getter: Gets the key of model to which the pipeline is attached.
        """
        return self._model_key

    @property
    def source_exposure_file_path(self):
        """
        Source exposure file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._source_exposure_file_path

    @source_exposure_file_path.setter
    def source_exposure_file_path(self, p):
        self._source_exposure_file_path = self.source_files['source_exposure'] = p

    @property
    def source_accounts_file_path(self):
        """
        Source accounts file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._source_accounts_file_path

    @source_accounts_file_path.setter
    def source_accounts_file_path(self, p):
        self._source_accounts_file_path = self.source_files['source_accounts'] = p

    @property
    def keys_file_path(self):
        """
        Oasis keys file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._keys_file_path

    @keys_file_path.setter
    def keys_file_path(self, p):
        self._keys_file_path = self.keys_files['keys'] = p

    @property
    def keys_errors_file_path(self):
        """
        Oasis keys errors file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._keys_errors_file_path

    @keys_errors_file_path.setter
    def keys_errors_file_path(self, p):
        self._keys_errors_file_path = self.keys_files['keys_errors'] = p

    @property
    def items_file_path(self):
        """
        Oasis items file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._items_file_path

    @items_file_path.setter
    def items_file_path(self, p):
        self._items_file_path = self.gul_input_files['items'] = self.oasis_files['items'] = p

    @property
    def coverages_file_path(self):
        """
        Oasis coverages file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._coverages_file_path

    @coverages_file_path.setter
    def coverages_file_path(self, p):
        self._coverages_file_path = self.gul_input_files['coverages'] = self.oasis_files['coverages'] = p

    @property
    def gulsummaryxref_file_path(self):
        """
        GUL summary file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._gulsummaryxref_file_path

    @gulsummaryxref_file_path.setter
    def gulsummaryxref_file_path(self, p):
        self._gulsummaryxref_file_path = self.gul_input_files['gulsummaryxref'] = self.oasis_files['gulsummaryxref'] = p

    @property
    def fm_policytc_file_path(self):
        """
        FM policy T & C file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fm_policytc_file_path

    @fm_policytc_file_path.setter
    def fm_policytc_file_path(self, p):
        self._fm_policytc_file_path = self.fm_input_files['fm_policytc'] = self.oasis_files['fm_policytc'] = p

    @property
    def fm_profile_file_path(self):
        """
        FM profile file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fm_profile_file_path

    @fm_profile_file_path.setter
    def fm_profile_file_path(self, p):
        self._fm_profile_file_path = self.fm_input_files['fm_profile'] = self.oasis_files['fm_profile'] = p

    @property
    def fm_programme_file_path(self):
        """
        FM programme file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fm_programme_file_path

    @fm_programme_file_path.setter
    def fm_programme_file_path(self, p):
        self._fm_programme_file_path = self.fm_input_files['fm_programme'] = self.oasis_files['fm_programme'] = p

    @property
    def fm_xref_file_path(self):
        """
        FM xref file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fm_xref_file_path

    @fm_xref_file_path.setter
    def fm_xref_file_path(self, p):
        self._fm_xref_file_path = self.fm_input_files['fm_xref'] = self.oasis_files['fm_xref'] = p

    @property
    def fmsummaryxref_file_path(self):
        """
        FM fmsummaryxref file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fmsummaryxref_file_path

    @fmsummaryxref_file_path.setter
    def fmsummaryxref_file_path(self, p):
        self._fmsummaryxref_file_path = self.fm_input_files['fmsummaryxref'] = self.oasis_files['fmsummaryxref'] = p

    @property
    def source_files(self):
        """
        Oasis source files set property - getter only.

            :getter: Gets the complete set of paths of the Oasis source files,
            including source loc. and acc. files.
        """
        return self._source_files

    @property
    def keys_files(self):
        """
        Oasis keys files set property - getter only.

            :getter: Gets the complete set of paths of the Oasis keys
            files, including the main keys file and the keys errors file.
        """
        return self._keys_files

    @property
    def gul_input_files(self):
        """
        Oasis GUL files set property - getter only.

            :getter: Gets the complete set of paths of the generated Oasis
            GUL files, including  ``items.csv``, ``coverages.csv``, `gulsummaryxref.csv`.
        """
        return self._gul_input_files

    @property
    def fm_input_files(self):
        """
        Oasis FM files set property - getter only.

            :getter: Gets the complete set of generated FM files, including
                     ``fm_policytc.csv``, ``fm_profile.csv``, `fm_programme.csv`,
                     ``fm_xref.csv``, ``fmsummaryxref.csv``.
        """
        return self._fm_input_files

    @property
    def oasis_files(self):
        """
        Oasis GUL + FM files set property - getter only.

            :getter: Gets the complete set of generated GUL + FM files,
                     including ``items.csv`, ``coverages.csv``,
                     ``gulsummaryxref.csv``, ``fm_policytc.csv``,
                     ``fm_profile.csv``, `fm_programme.csv`,
                     ``fm_xref.csv``, ``fmsummaryxref.csv``.
        """
        return self._oasis_files


    def clear(self, files_subsets=None):
        """
        Clears file path attributes in the pipeline.

            :param files_subsets: List of files subset labels (if empty)
                                  then all files subsets are cleared)::

                'source': source exposure file path + source accounts
                                file path
                'keys':   keys file path, keys errors file path
                'gul':    GUL input files
                'fm':     FM input files
                `oasis`:  GUL + FM input files

            :type files_subsets: list
        """
        if not files_subsets:
            filenames = chain(self.source_files, self.keys_files, self.oasis_files)
        else:
            filenames = chain(fn for fsb in files_subsets for fn in getattr(self, '{}_files'.format(fsb)))

        for fn in filenames:
            setattr(self, '{}_file_path'.format(fn), None)
