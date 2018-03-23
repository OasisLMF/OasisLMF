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

        self._oasis_files = {
            'items': self._items_file_path,
            'coverages': self._coverages_file_path,
            'gulsummaryxref': self._gulsummaryxref_file_path
        }

        self._file_paths = (
            'source_exposures_file_path',
            'canonical_exposures_file_path',
            'model_exposures_file_path',
            'keys_file_path',
            'keys_error_file_path',
            'items_file_path',
            'coverages_file_path',
            'gulsummaryxref_file_path',
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
        self._items_file_path = self.oasis_files['items'] = p

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
        self._coverages_file_path = self.oasis_files['coverages'] = p

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
        self._gulsummaryxref_file_path = self.oasis_files['gulsummaryxref'] = p

    @property
    def oasis_files(self):
        """
        Oasis files set property - getter only.

            :getter: Gets the complete set of paths of the generated Oasis
            files, including  ``items.csv``, ``coverages.csv``, `gulsummaryxref.csv`.
        """
        return self._oasis_files

    def clear(self):
        """
        Clears all file path attributes in the pipeline.
        """
        map(
            lambda p: setattr(self, p, None),
            self._file_paths
        )
