# -*- coding: utf-8 -*-

__all__ = [
    'OasisFilesPipeline'
]


class OasisFilesPipeline(object):
    def __init__(
        self,
        model_key=None,
        source_exposures_file_path=None,
        canonical_exposures_file_path=None,
        model_exposures_file_path=None,
        keys_file_path=None,
        keys_error_file_path=None
    ):
        self._model_key = model_key
        self._source_exposures_file_path = source_exposures_file_path
        self._canonical_exposures_file_path = canonical_exposures_file_path
        self._model_exposures_file_path = model_exposures_file_path
        self._keys_file_path = keys_file_path
        self._keys_error_file_path = keys_error_file_path

        self._items_file_path = None
        self._coverages_file_path = None
        self._gulsummaryxref_file_path = None

        self._fm_policytc_file_path = None
        self._fm_profile_file_path = None
        self._fm_programme_file_path = None
        self._fm_xref_file_path = None
        self._fmsummaryxref_file_path = None

        self._exposure_files = {
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

        self._oasis_files = {k:v for k, v in self._exposure_files.items() + self._fm_files.items()}

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
        self._source_exposures_file_path = p

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
        self._canonical_exposures_file_path = p

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
        self._model_exposures_file_path = p

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
        self._keys_file_path = p

    @property
    def keys_error_file_path(self):
        """
        Oasis keys error file path property.

            :getter: Gets the file path
            :setter: Sets the current file path to the specified file path
        """
        return self._keys_error_file_path

    @keys_error_file_path.setter
    def keys_error_file_path(self, p):
        self._keys_error_file_path = p

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
        self._items_file_path = self.exposure_files['items'] = self.oasis_files['items'] = p

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
        self._coverages_file_path = self.exposure_files['coverages'] = self.oasis_files['items'] = p

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
        self._gulsummaryxref_file_path = self.exposure_files['gulsummaryxref'] = self.oasis_files['items'] = p

    @property
    def fm_policytc_file_path(self):
        """
        FM policy T & C file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fm_policytc_file_path

    @fm_policytc_file_path.setter
    def fm_policytc_file_path(self, f):
        self._fm_policytc_file_path = self.fm_files['fm_policytc'] = self.oasis_files['items'] = f

    @property
    def fm_profile_file_path(self):
        """
        FM profile file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fm_profile_file_path

    @fm_profile_file_path.setter
    def fm_profile_file_path(self, f):
        self._fm_profile_file_path = self.fm_files['fm_profile'] = self.oasis_files['items'] = f

    @property
    def fm_programme_file_path(self):
        """
        FM programme file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fm_programme_file_path

    @fm_programme_file_path.setter
    def fm_programme_file_path(self, f):
        self._fm_programme_file_path = self.fm_files['fm_programme'] = self.oasis_files['items'] = f

    @property
    def fm_xref_file_path(self):
        """
        FM xref file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fm_xref_file_path

    @fm_xref_file_path.setter
    def fm_xref_file_path(self, f):
        self._fm_xref_file_path = self.fm_files['fm_xref'] = self.oasis_files['items'] = f

    @property
    def fmsummaryxref_file_path(self):
        """
        FM fmsummaryxref file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._fmfmsummaryxref_file_path

    @fmsummaryxref_file_path.setter
    def fmsummaryxref_file_path(self, f):
        self._fmsummaryxref_file_path = self.fm_files['fmsummaryxref'] = self.oasis_files['items'] = f


    @property
    def exposure_files(self):
        """
        Oasis exposure files set property - getter only.

            :getter: Gets the complete set of paths of the generated Oasis
            exposure files, including  ``items.csv``, ``coverages.csv``, `gulsummaryxref.csv`.
        """
        return self._exposure_files

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
        Oasis exposure + FM files set property - getter only.

            :getter: Gets the complete set of generated exposure + FM files,
                     including ``items.csv`, ``coverages.csv``,
                     ``gulsummaryxref.csv``, ``fm_policytc.csv``,
                     ``fm_profile.csv``, `fm_programme.csv`,
                     ``fm_xref.csv``, ``fmsummaryxref.csv``.
        """
        return self._oasis_files


    def clear(self, exposure_only=False, fm_only=False):
        """
        Clears all file path attributes in the pipeline.
        """
        if not (exposure_only or fm_only):
            filenames = self.oasis_files.keys()
        else:
            filenames = self.exposure_files.keys() if exposure_only else self.fm_files.keys()

        map(
            lambda p: setattr(self, '{}_file_path'.format(p), None),
            filenames
        )
