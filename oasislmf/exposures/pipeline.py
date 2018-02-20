# -*- coding: utf-8 -*-

__all__ = [
    'OasisFilesPipeline'
]


class OasisFilesPipeline(object):
    def __init__(
        self,
        model_key=None,
        source_exposures_path=None,
        canonical_exposures_path=None,
        model_exposures_path=None,
        keys_file_path=None
    ):
        self._model_key = model_key
        self._source_exposures_path = source_exposures_path
        self._canonical_exposures_path = canonical_exposures_path
        self._model_exposures_path = model_exposures_path
        self._keys_file_path = keys_file_path

        self._items_path = None
        self._coverages_path = None
        self._gulsummaryxref_path = None

        self._oasis_files = {
            'items': self._items_path,
            'coverages': self._coverages_path,
            'gulsummaryxref': self._gulsummaryxref_path
        }

        self._file_attrib_names = (
            'source_exposures_file',
            'canonical_exposures_file',
            'model_exposures_file',
            'keys_file',
            'items_file',
            'coverages_file',
            'gulsummaryxref_file',
        )

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
    def source_exposures_path(self):
        """
        Source exposures file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._source_exposures_path

    @source_exposures_path.setter
    def source_exposures_path(self, f):
        self._source_exposures_path = f

    @property
    def canonical_exposures_path(self):
        """
        Canonical exposures file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._canonical_exposures_path

    @canonical_exposures_path.setter
    def canonical_exposures_path(self, f):
        self._canonical_exposures_path = f

    @property
    def model_exposures_path(self):
        """
        Model exposures file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._model_exposures_path

    @model_exposures_path.setter
    def model_exposures_path(self, f):
        self._model_exposures_path = f

    @property
    def keys_file_path(self):
        """
        Oasis keys file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._keys_file_path

    @keys_file_path.setter
    def keys_file_path(self, f):
        self._keys_file_path = f

    @property
    def items_file_path(self):
        """
        Oasis items file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._items_path

    @items_file_path.setter
    def items_file_path(self, f):
        self._items_path = self.oasis_files['items'] = f

    @property
    def coverages_path(self):
        """
        Oasis coverages file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._coverages_path

    @coverages_path.setter
    def coverages_path(self, f):
        self._coverages_path = self.oasis_files['coverages'] = f

    @property
    def gulsummaryxref_path(self):
        """
        GUL summary file property.

            :getter: Gets the actual file object
            :setter: Sets the file to the specified file object
        """
        return self._gulsummaryxref_path

    @gulsummaryxref_path.setter
    def gulsummaryxref_path(self, f):
        self._gulsummaryxref_path = self.oasis_files['gulsummaryxref'] = f

    @property
    def oasis_files(self):
        """
        Oasis files set property - getter only.

            :getter: Gets the complete set of generated Oasis files, including
                     ``items.csv``, ``coverages.csv``, `gulsummaryxref.csv`.
        """
        return self._oasis_files

    def clear(self):
        """
        Clears all file attributes in the pipeline.
        """
        map(
            lambda f: setattr(self, f, None),
            self._file_attrib_names
        )
