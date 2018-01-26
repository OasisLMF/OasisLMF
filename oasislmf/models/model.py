#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

__all__ = [
    'OasisModel',
    'OasisModelFactory'
]

import os
import io

from ..utils.exceptions import OasisException
from ..exposures.pipeline import OasisFilesPipeline


__author__ = "Sandeep Murthy"
__copyright__ = "2017, Oasis Loss Modelling Framework"


class OasisModel(object):
    """
    A simple object representation of Oasis models and their resources - an
    Oasis model is identified by a triple: a specific supplier ID, model ID
    and model version. The constructor requires these three arguments
    for creating a new Oasis model object. Each model object also has a
    resources dictionary that can be used to "attach" any resources by clients,
    e.g. a lookup service instance, a transforms files pipeline, validation
    and transformation files for the source -> canonical and canonical
    -> model exposure transforms, etc.
    """

    def __init__(
        self,
        model_supplier_id,
        model_id,
        model_version_id,
        resources=None
    ):
        """
        Constructor - requires supplier ID, model ID and model version ID.
        """
        self._supplier_id = model_supplier_id
        self._model_id = model_id
        self._model_version_id = model_version_id
        self._key = '{}/{}/{}'.format(model_supplier_id, model_id, model_version_id)
        self._resources = resources if resources else {}

        # set default resources
        self._resources.setdefault('oasis_files_path', os.path.join('Files', self._key.replace('/', '-')))
        self._resources['oasis_files_path'] = os.path.abspath(self._resources['oasis_files_path'])

        self._resources.setdefault('oasis_files_pipeline', OasisFilesPipeline(model_key=self._key))
        if not isinstance(self._resources['oasis_files_pipeline'], OasisFilesPipeline):
            raise OasisException('Oasis files pipeline object for model {} is not of type {}'.format(self, OasisFilesPipeline))

        if 'source_exposures_file_path' in self._resources:
            with io.open(self._resources['source_exposures_file_path'], 'r', encoding='utf-8') as f:
                self._resources['oasis_files_pipeline'].source_exposures_file = f

        self._resources['canonical_exposures_profile'] = self.load_canonical_profile(**self._resources)

    @classmethod
    def load_canonical_profile(cls, canonical_exposures_profile_json=None, canonical_exposures_profile_json_path=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the canonical
        exposures profile for a given ``oasis_model``, stores this in the
        model object's resources dict, and returns the object.
        """
        if canonical_exposures_profile_json:
            return json.loads(canonical_exposures_profile_json)
        elif canonical_exposures_profile_json_path:
            with io.open(canonical_exposures_profile_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        return None

    @property
    def key(self):
        """
        Model key - getter only. Format is

            :getter: Returns <model supplier ID>/<model ID>/<model version ID>
            :type: string
        """
        return self._key

    @property
    def supplier_id(self):
        """
        Model supplier ID property - getter only.

            :getter: Gets the model supplier ID
            :type: string
        """
        return self._supplier_id

    @property
    def model_id(self):
        """
        Model ID property - getter only.

            :getter: Gets the model ID
            :type: string
        """
        return self._model_id

    @property
    def model_version_id(self):
        """
        Model version ID property - getter only.

            :getter: Gets the model version ID
            :type: string
        """
        return self._model_version_id

    @property
    def resources(self, key=None):
        """
        Model resources dictionary property.

            :getter: Gets the attached resource in the model resources dict
                     using the optional resource ``key`` argument. If ``key``
                     is not given then the entire resources dict is returned.

            :setter: Sets the value of the optional resource ``key`` in the
                     resources dict to ``val``. If no ``key`` is given then
                     ``val`` is assumed to be a new resources dict and is
                     used to replace the existing dict.

            :deleter: Deletes the value of the optional resource ``key`` in
                      the resources dict. If no ``key`` is given then the
                      entire existing dict is cleared.
        """
        return self._resources[key] if key else self._resources

    @resources.setter
    def resources(self, key=None, val=None):
        if key:
            self._resources.update({key: val})
        else:
            self._resources.clear()
            self._resources.update(val)

    @resources.deleter
    def resources(self, key=None):
        if key:
            del self._resources[key]
        else:
            self._resources.clear()

    def __str__(self):
        return '{}: {}'.format(self.__repr__(), self.key)

    def __repr__(self):
        return '{}: {}'.format(self.__class__, self.__dict__)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')


class OasisModelFactory(object):
    """
    Factory class for creating Oasis model objects.
    """
    @classmethod
    def create(
        cls,
        model_supplier_id,
        model_id,
        model_version_id,
        resources=None
    ):
        """
        Service method to instantiate Oasis model objects with attached
        resource dicts.
        """
        return OasisModel(
            model_supplier_id=model_supplier_id,
            model_id=model_id,
            model_version_id=model_version_id,
            resources=resources
        )
