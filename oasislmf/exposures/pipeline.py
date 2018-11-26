# -*- coding: utf-8 -*-

from itertools import chain

__all__ = [
    'OasisFilesPipeline'
]


class OasisFilesPipeline(object):
    def __init__(
        self,
        model_key=None,
        source_exposures_file_path=None,
        source_accounts_file_path=None,
        canonical_exposures_file_path=None,
        canonical_accounts_file_path=None,
        model_exposures_file_path=None,
        keys_file_path=None,
        keys_errors_file_path=None
    ):
        self._model_key = model_key
        self._source_exposures_file_path = source_exposures_file_path
        self._source_accounts_file_path = source_accounts_file_path
        self._canonical_exposures_file_path = canonical_exposures_file_path
        self._canonical_accounts_file_path = canonical_accounts_file_path
        self._model_exposures_file_path = model_exposures_file_path
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
            'source_exposures': self._source_exposures_file_path,
            'source_accounts': self._source_accounts_file_path 
        }

        self._intermediate_files = {
            'canonical_exposures': self._canonical_exposures_file_path,
            'canonical_accounts': self._canonical_accounts_file_path,
            'model_exposures': self._model_exposures_file_path,
            'keys': self._keys_file_path,
            'keys_errors': self._keys_errors_file_path
        }

        self._gul_files = {
            'items': self._items_file_path,
            'coverages': self._coverages_file_path,
            'gulsummaryxref': self._gulsummaryxref_file_path
        }

        self._fm_files = {
            'fm_policytc': self._fm_policytc_file_path,
            'fm_profile': self._fm_profile_file_path,
            'fm_programme': self._fm_programme_file_path,
            'fm_xref': self._fm_xref_file_path,
            'fmsummaryxref': self._fmsummaryxref_file_path
        }

        self._oasis_files = {k:v for k, v in chain(self._gul_files.items(), self._fm_files.items())}

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
    def source_exposures_file_path(self):
        """
        Source exposures file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._source_exposures_file_path

    @source_exposures_file_path.setter
    def source_exposures_file_path(self, p):
        self._source_exposures_file_path = self.source_files['source_exposures'] = p

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
    def canonical_exposures_file_path(self):
        """
        Canonical exposures file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._canonical_exposures_file_path

    @canonical_exposures_file_path.setter
    def canonical_exposures_file_path(self, p):
        self._canonical_exposures_file_path = self.intermediate_files['canonical_exposures'] = p

    @property
    def canonical_accounts_file_path(self):
        """
        Canonical accounts file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._canonical_accounts_file_path

    @canonical_accounts_file_path.setter
    def canonical_accounts_file_path(self, p):
        self._canonical_accounts_file_path = self.intermediate_files['canonical_accounts'] = p

    @property
    def model_exposures_file_path(self):
        """
        Model exposures file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._model_exposures_file_path

    @model_exposures_file_path.setter
    def model_exposures_file_path(self, p):
        self._model_exposures_file_path = self.intermediate_files['model_exposures'] = p

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
        self._keys_file_path = self.intermediate_files['keys'] = p

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
        self._keys_errors_file_path = self.intermediate_files['keys_errors'] = p

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
        self._items_file_path = self.gul_files['items'] = self.oasis_files['items'] = p

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
        self._coverages_file_path = self.gul_files['coverages'] = self.oasis_files['coverages'] = p

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
        self._gulsummaryxref_file_path = self.gul_files['gulsummaryxref'] = self.oasis_files['gulsummaryxref'] = p

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
        self._fm_policytc_file_path = self.fm_files['fm_policytc'] = self.oasis_files['fm_policytc'] = p

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
        self._fm_profile_file_path = self.fm_files['fm_profile'] = self.oasis_files['fm_profile'] = p

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
        self._fm_programme_file_path = self.fm_files['fm_programme'] = self.oasis_files['fm_programme'] = p

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
        self._fm_xref_file_path = self.fm_files['fm_xref'] = self.oasis_files['fm_xref'] = p

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
        self._fmsummaryxref_file_path = self.fm_files['fmsummaryxref'] = self.oasis_files['fmsummaryxref'] = p

    @property
    def source_files(self):
        """
        Oasis source files set property - getter only.

            :getter: Gets the complete set of paths of the Oasis source files,
            including source loc. and acc. files.
        """
        return self._source_files

    @property
    def intermediate_files(self):
        """
        Oasis intermediate files set property - getter only.

            :getter: Gets the complete set of paths of the Oasis intermediate
            files, including canonical and model exposures files, and keys
            and keys errors files.
        """
        return self._intermediate_files

    @property
    def gul_files(self):
        """
        Oasis GUL files set property - getter only.

            :getter: Gets the complete set of paths of the generated Oasis
            GUL files, including  ``items.csv``, ``coverages.csv``, `gulsummaryxref.csv`.
        """
        return self._gul_files

    @property
    def fm_files(self):
        """
        Oasis FM files set property - getter only.

            :getter: Gets the complete set of generated FM files, including
                     ``fm_policytc.csv``, ``fm_profile.csv``, `fm_programme.csv`,
                     ``fm_xref.csv``, ``fmsummaryxref.csv``.
        """
        return self._fm_files

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

                'source': source exposures file path + source accounts
                                file path
                'intermediate': canonical exposures file path, model
                                      exposures file path, keys file path,
                                      keys errors file path
                'gul': GUL files
                'fm': FM files
                `oasis`: GUL files + GM files

            :type files_subsets: list
        """
        if not files_subsets:
            filenames = chain(self.source_files, self.intermediate_files, self.oasis_files)
        else:
            filenames = chain(fn for fsb in files_subsets for fn in getattr(self, '{}_files'.format(fsb)))

        for fn in filenames:
            setattr(self, '{}_file_path'.format(fn), None)
