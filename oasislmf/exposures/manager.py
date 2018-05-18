# -*- coding: utf-8 -*-

__all__ = [
    'OasisExposuresManagerInterface',
    'OasisExposuresManager'
]

import copy
import io
import itertools
import json
import logging
import os
import shutil
import sys
import time

import pandas as pd

from interface import (
    Interface,
    implements,
)

from ..keys.lookup import OasisKeysLookupFactory
from ..utils.concurrency import (
    multiprocess,
    multithread,
    Task,
)
from ..utils.exceptions import OasisException
from ..utils.fm import (
    canonical_profiles_fm_terms_grouped_by_level_and_term_type,
    get_fm_terms_by_level_as_list,
    get_policytc_id,
    get_policytc_ids,
)
from ..utils.values import get_utctimestamp
from ..models import OasisModel
from .pipeline import OasisFilesPipeline
from .csv_trans import Translator


class OasisExposuresManagerInterface(Interface):  # pragma: no cover
    """
    Interface class form managing a collection of exposures.

    :param oasis_models: A list of Oasis model objects with resources provided in the model objects'
        resources dictionaries.
    :type oasis_models: ``list(OasisModel)``
    """

    def __init__(self, oasis_models=None):
        """
        Class constructor.

        :param oasis_models: An optional list of Oasis model objects
        :type oasis_models: list
        """
        pass

    def add_model(self, oasis_model):
        """
        Adds Oasis model object to the manager and sets up its resources.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel
        """
        pass

    def delete_model(self, oasis_model):
        """
        Deletes an existing Oasis model object in the manager.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel
        """
        pass

    def transform_source_to_canonical(self, oasis_model=None, **kwargs):
        """
        Transforms a source exposures/locations for a given ``oasis_model``
        or set of keyword arguments to a canonical/standard Oasis format.

        All the required resources must be provided either in the model object
        resources dict or the keyword arguments.

        It is up to the specific implementation of this class of how these
        resources will be named and how they will be used to
        effect the transformation.

        The transform is generic by default, but could be supplier specific if
        required.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def transform_canonical_to_model(self, oasis_model=None, **kwargs):
        """
        Transforms the canonical exposures/locations for a given ``oasis_model``
        or set of keyword arguments object to a format suitable for an Oasis
        model keys lookup service.

        All the required resources must be provided either in the model object
        resources dict or the keyword arguments.

        It is up to the specific implementation of this class of how these
        resources will be named and how they will be used to
        effect the transformation.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def get_keys(self, oasis_model=None, **kwargs):
        """
        Generates the Oasis keys and keys error files for a given
        ``oasis_model`` or set of keyword arguments.

        The keys file is a CSV file containing keys lookup information for
        locations with successful lookups, and has the following headers::

            LocID,PerilID,CoverageID,AreaPerilID,VulnerabilityID

        while the keys error file is a CSV file containing keys lookup
        information for locations with unsuccessful lookups (failures,
        no matches) and has the following headers::

            LocID,PerilID,CoverageID,Message

        All the required resources must be provided either in the model object
        resources dict or the keyword arguments.

        It is up to the specific implementation of this class of how these
        resources will be named and how they will be used to
        effect the transformation.

        A "standard" implementation should use the lookup service factory
        class in ``oasis_utils`` (a submodule of `omdk`) namely

            ``oasis_utils.oasis_keys_lookup_service_utils.KeysLookupServiceFactory``

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def load_canonical_exposures_profile(self, oasis_model=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the canonical
        exposures profile for a given ``oasis_model`` or set of keyword
        arguments.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def load_canonical_accounts_profile(self, oasis_model=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the canonical
        accounts profile for a given ``oasis_model`` or set of keyword
        arguments.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def generate_gul_items(self, canonical_exposures_profile, canonical_exposures_file_path, keys_file_path, **kwargs):
        """
        Generates GUL items using the given canonical exposures profile and
        exposures file path + keys file path.

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_exposures_file_path: Canonical exposures file path
        :type canonical_exposures_file_path: str

        :param keys_file_path: Keys file path
        :type keys_file_path: str
        """
        pass

    def generate_fm_items(self, canonical_exposures, gul_items, canonical_exposures_profile, canonical_accounts_profile, canonical_accounts_file_path, **kwargs):
        """
        Generates GUL items using the given canonical exposures profile and
        exposures file path + keys file path.

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_exposures_file_path: Canonical exposures file path
        :type canonical_exposures_file_path: str

        :param keys_file_path: Keys file path
        :type keys_file_path: str
        """
        pass

    def write_gul_files(self, oasis_model=None, **kwargs):
        """
        Writes Oasis GUL files for a given ``oasis_model`` or set of keyword
        arguments.

        The required resources must be provided either via the model object
        resources dict or the keyword arguments.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def write_fm_files(self, oasis_model=None, **kwargs):
        """
        Writes Oasis FM files for a given ``oasis_model`` or set of keyword
        arguments.

        The required resources must be provided either via the model object
        resources dict or the keyword arguments.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def write_oasis_files(self, oasis_model=None, fm=False, **kwargs):
        """
        Writes the full set of Oasis files, which includes GUL files and
        possibly also the FM files (if ``fm`` is ``True``) for a given
        ``oasis_model`` or set of keyword arguments.

        The required resources must be provided either via the model object
        resources dict or the keyword arguments.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def create_model(self, model_supplier_id, model_id, model_version_id, resources=None):
        """
        Creates an Oasis model object, with attached resources if a resources
        dict was provided.

        :param model_supplier_id: The model supplier ID
        :type model_supplier_id: str

        :param model_id: The model ID
        :type model_id: str

        :param model_version_id: The model version ID or string
        :type model_version_id: str

        :param resources: Optional dictionary of model resources
        :type resources: dict
        """
        pass


class OasisExposuresManager(implements(OasisExposuresManagerInterface)):

    def __init__(self, oasis_models=None):
        self.logger = logging.getLogger()

        self.logger.debug('Exposures manager {} initialising'.format(self))

        self.logger.debug('Adding models')
        self._models = {}

        self.add_models(oasis_models)

        self.logger.debug('Exposures manager {} finished initialising'.format(self))

    def add_model(self, oasis_model):
        """
        Adds model to the manager and sets up its resources.
        """
        self._models[oasis_model.key] = oasis_model

        return oasis_model

    def add_models(self, oasis_models):
        """
        Adds a list of Oasis model objects to the manager.
        """
        for model in oasis_models or []:
            self.add_model(model)

    def delete_model(self, oasis_model):
        """
        Deletes an existing Oasis model object in the manager.
        """
        if oasis_model.key in self._models:
            oasis_model.resources['oasis_files_pipeline'].clear()

            del self._models[oasis_model.key]

    def delete_models(self, oasis_models):
        """
        Deletes a list of existing Oasis model objects in the manager.
        """
        for model in oasis_models:
            self.delete_model(model)

    @property
    def keys_lookup_factory(self):
        """
        Keys lookup service factory property - getter only.

            :getter: Gets the current keys lookup service factory instance
        """
        return self._keys_lookup_factory

    @property
    def models(self):
        """
        Model objects dictionary property.

            :getter: Gets the model in the models dict using the optional
                     ``key`` argument. If ``key`` is not given then the dict
                     is returned.

            :setter: Sets the value of the optional ``key`` in the models dict
                     to ``val`` where ``val`` is assumed to be an Oasis model
                     object (``omdk.OasisModel.OasisModel``).

                     If no ``key`` is given then ``val`` is assumed to be a new
                     models dict and is used to replace the existing dict.

            :deleter: Deletes the value of the optional ``key`` in the models
                      dict. If no ``key`` is given then the entire existing
                      dict is cleared.
        """
        return self._models

    @models.setter
    def models(self, val):
        self._models.clear()
        self._models.update(val)

    @models.deleter
    def models(self):
        self._models.clear()

    def transform_source_to_canonical(self, oasis_model=None, source_type='exposures', **kwargs):
        """
        Transforms a canonical exposures/locations file for a given
        ``oasis_model`` object to a canonical/standard Oasis format.

        It can also transform a source accounts file to a canonical accounts
        file, if the optional argument ``source_type`` has the value of ``accounts``.
        The default ``source_type`` is ``exposures``.

        By default parameters supplied to this function fill be used if present
        otherwise they will be taken from the `oasis_model` resources dictionary
        if the model is supplied.

        :param oasis_model: An optional Oasis model object
        :type oasis_model: ``oasislmf.models.model.OasisModel``

        :param source_exposures_file_path: Source exposures file path (if ``source_type`` is ``exposures``)
        :type source_exposures_file_path: str

        :param source_exposures_validation_file_path: Source exposures validation file (if ``source_type`` is ``exposures``)
        :type source_exposures_validation_file_path: str

        :param source_to_canonical_exposures_transformation_file_path: Source exposures transformation file (if ``source_type`` is ``exposures``)
        :type source_to_canonical_exposures_transformation_file_path: str

        :param canonical_exposures_file_path: Path to the output canonical exposure file (if ``source_type`` is ``exposures``)
        :type canonical_exposures_file_path: str

        :param source_accounts_file_path: Source accounts file path (if ``source_type`` is ``accounts``)
        :type source_exposures_file_path: str

        :param source_accounts_validation_file_path: Source accounts validation file (if ``source_type`` is ``accounts``)
        :type source_exposures_validation_file_path: str

        :param source_to_canonical_accounts_transformation_file_path: Source accounts transformation file (if ``source_type`` is ``accounts``)
        :type source_to_canonical_accounts_transformation_file_path: str

        :param canonical_accounts_file_path: Path to the output canonical accounts file (if ``source_type`` is ``accounts``)
        :type canonical_accounts_file_path: str

        :return: The path to the output canonical file
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        input_file_path = os.path.abspath(kwargs['source_accounts_file_path']) if source_type == 'accounts' else os.path.abspath(kwargs['source_exposures_file_path'])
        validation_file_path = os.path.abspath(kwargs['source_accounts_validation_file_path']) if source_type == 'accounts' else os.path.abspath(kwargs['source_exposures_validation_file_path'])
        transformation_file_path = os.path.abspath(kwargs['source_to_canonical_accounts_transformation_file_path']) if source_type == 'accounts' else os.path.abspath(kwargs['source_to_canonical_exposures_transformation_file_path'])
        output_file_path = os.path.abspath(kwargs['canonical_accounts_file_path']) if source_type == 'accounts' else os.path.abspath(kwargs['canonical_exposures_file_path'])

        translator = Translator(input_file_path, output_file_path, validation_file_path, transformation_file_path, append_row_nums=True)
        translator()

        if oasis_model:
            if source_type == 'accounts':
                oasis_model.resources['oasis_files_pipeline'].canonical_accounts_file_path = output_file_path
            else:
                oasis_model.resources['oasis_files_pipeline'].canonical_exposures_file_path = output_file_path

        return output_file_path

    def transform_canonical_to_model(self, oasis_model=None, **kwargs):
        """
        Transforms the canonical exposures/locations file for a given
        ``oasis_model`` object to a format suitable for an Oasis model keys
        lookup service.

        By default parameters supplied to this function fill be used if present
        otherwise they will be taken from the `oasis_model` resources dictionary
        if the model is supplied.

        :param oasis_model: The model to get keys for
        :type oasis_model: ``oasislmf.models.model.OasisModel``

        :param canonical_exposures_file_path: Path to the canonical exposures file
        :type canonical_exposures_file_path: str

        :param canonical_exposures_validation_file_path: Path to the exposure validation file
        :type canonical_exposures_validation_file_path: str

        :param canonical_to_model_exposures_transformation_file_path: Path to the exposure transformation file
        :type canonical_to_model_exposures_transformation_file_path: str

        :param model_exposures_file_path: Path to the output model exposure file
        :type model_exposures_file_path: str

        :return: The path to the output model exposure file
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        input_file_path = os.path.abspath(kwargs['canonical_exposures_file_path'])
        validation_file_path = os.path.abspath(kwargs['canonical_exposures_validation_file_path'])
        transformation_file_path = os.path.abspath(kwargs['canonical_to_model_exposures_transformation_file_path'])
        output_file_path = os.path.abspath(kwargs['model_exposures_file_path'])

        translator = Translator(input_file_path, output_file_path, validation_file_path, transformation_file_path, append_row_nums=False)
        translator()

        if oasis_model:
            oasis_model.resources['oasis_files_pipeline'].model_exposures_file_path = output_file_path

        return output_file_path

    def load_canonical_exposures_profile(
            self,
            oasis_model=None,
            canonical_exposures_profile_json=None,
            canonical_exposures_profile_json_path=None,
            **kwargs
        ):
        """
        Loads a JSON string or JSON file representation of the canonical
        exposures profile for a given ``oasis_model``, stores this in the
        model object's resources dict, and returns the object.
        """
        if oasis_model:
            canonical_exposures_profile_json = canonical_exposures_profile_json or oasis_model.resources.get('canonical_exposures_profile_json')
            canonical_exposures_profile_json_path = canonical_exposures_profile_json_path or oasis_model.resources.get('canonical_exposures_profile_json_path')

        profile = {}
        if canonical_exposures_profile_json:
            profile = json.loads(canonical_exposures_profile_json)
        elif canonical_exposures_profile_json_path:
            with io.open(canonical_exposures_profile_json_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)

        if oasis_model:
            oasis_model.resources['canonical_exposures_profile'] = profile

        return profile

    def load_canonical_accounts_profile(
            self,
            oasis_model=None,
            canonical_accounts_profile_json=None,
            canonical_accounts_profile_json_path=None,
            **kwargs
        ):
        """
        Loads a JSON string or JSON file representation of the canonical
        exposures profile for a given ``oasis_model``, stores this in the
        model object's resources dict, and returns the object.
        """
        if oasis_model:
            canonical_accounts_profile_json = canonical_accounts_profile_json or oasis_model.resources.get('canonical_accounts_profile_json')
            canonical_accounts_profile_json_path = canonical_accounts_profile_json_path or oasis_model.resources.get('canonical_accounts_profile_json_path')

        profile = {}
        if canonical_accounts_profile_json:
            profile = json.loads(canonical_accounts_profile_json)
        elif canonical_accounts_profile_json_path:
            with io.open(canonical_accounts_profile_json_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)

        if oasis_model:
            oasis_model.resources['canonical_accounts_profile'] = profile

        return profile

    def get_keys(self, oasis_model=None, model_exposures_file_path=None, lookup=None, keys_file_path=None, keys_errors_file_path=None, **kwargs):
        """
        Generates the Oasis keys and keys error files for a given model object.
        The keys file is a CSV file containing keys lookup information for
        locations with successful lookups, and has the following headers::

            LocID,PerilID,CoverageID,AreaPerilID,VulnerabilityID

        while the keys error file is a CSV file containing keys lookup
        information for locations with unsuccessful lookups (failures,
        no matches) and has the following headers::

            LocID,PerilID,CoverageID,Message

        By default it is assumed that all the resources required for the
        transformation are present in the model object's resources dict,
        if the model is supplied. These can be overridden by providing the
        relevant optional parameters.

        If no model is supplied then the optional paramenters must be
        supplied.

        If the model is supplied the result key file path is stored in the
        models ``file_pipeline.keyfile_path`` property.

        :param oasis_model: The model to get keys for
        :type oasis_model: ``OasisModel``

        :param keys_file_path: Path to the keys file, required if ``oasis_model`` is ``None``
        :type keys_file_path: str

        :param keys_errors_file_path: Path to the keys error file, required if ``oasis_model`` is ``None``
        :type keys_errors_file_path: str

        :param lookup: Path to the keys lookup service to use, required if ``oasis_model`` is ``None``
        :type lookup: str

        :param model_exposures_file_path: Path to the exposures file, required if ``oasis_model`` is ``None``
        :type model_exposures_file_path: str

        :return: The path to the generated keys file
        """
        if oasis_model:
            model_exposures_file_path = model_exposures_file_path or oasis_model.resources['oasis_files_pipeline'].model_exposures_file_path
            lookup = lookup or oasis_model.resources.get('lookup')
            keys_file_path = keys_file_path or oasis_model.resources['oasis_files_pipeline'].keys_file_path
            keys_errors_file_path = keys_errors_file_path or oasis_model.resources['oasis_files_pipeline'].keys_errors_file_path

        model_exposures_file_path, keys_file_path, keys_errors_file_path = tuple(
            os.path.abspath(p) if p and not os.path.isabs(p) else p for p in [model_exposures_file_path, keys_file_path, keys_errors_file_path]
        )

        keys_file_path, _, keys_errors_file_path, _ = OasisKeysLookupFactory().save_keys(
            keys_file_path=keys_file_path,
            keys_errors_file_path=keys_errors_file_path,
            lookup=lookup,
            model_exposures_file_path=model_exposures_file_path,
        )

        if oasis_model:
            oasis_model.resources['oasis_files_pipeline'].keys_file_path = keys_file_path
            oasis_model.resources['oasis_files_pipeline'].keys_errors_file_path = keys_errors_file_path

        return keys_file_path, keys_errors_file_path

    def _process_default_kwargs(self, oasis_model=None, fm=False, **kwargs):
        if oasis_model:
            omr = oasis_model.resources
            ofp = omr['oasis_files_pipeline']

            kwargs.setdefault('source_exposures_file_path', omr.get('source_exposures_file_path'))
            kwargs.setdefault('source_accounts_file_path', omr.get('source_accounts_file_path'))

            kwargs.setdefault('source_exposures_validation_file_path', omr.get('source_exposures_validation_file_path'))
            kwargs.setdefault('source_accounts_validation_file_path', omr.get('source_accounts_validation_file_path'))

            kwargs.setdefault('source_to_canonical_exposures_transformation_file_path', omr.get('source_to_canonical_exposures_transformation_file_path'))
            kwargs.setdefault('source_to_canonical_accounts_transformation_file_path', omr.get('source_to_canonical_accounts_transformation_file_path'))

            kwargs.setdefault('canonical_exposures_profile', omr.get('canonical_exposures_profile'))
            kwargs.setdefault('canonical_accounts_profile', omr.get('canonical_accounts_profile'))

            kwargs.setdefault('canonical_exposures_profile_json', omr.get('canonical_exposures_profile_json'))
            kwargs.setdefault('canonical_accounts_profile_json', omr.get('canonical_accounts_profile_json'))

            kwargs.setdefault('canonical_exposures_profile_json_path', omr.get('canonical_exposures_profile_json_path'))
            kwargs.setdefault('canonical_accounts_profile_json_path', omr.get('canonical_accounts_profile_json_path'))

            kwargs.setdefault('canonical_exposures_file_path', ofp.canonical_exposures_file_path)
            kwargs.setdefault('canonical_accounts_file_path', ofp.canonical_accounts_file_path)

            kwargs.setdefault('canonical_exposures_validation_file_path', omr.get('canonical_exposures_validation_file_path'))
            kwargs.setdefault('canonical_to_model_exposures_transformation_file_path', omr.get('canonical_to_model_exposures_transformation_file_path'))

            kwargs.setdefault('model_exposures_file_path', ofp.model_exposures_file_path)

            kwargs.setdefault('keys_file_path', ofp.keys_file_path)
            kwargs.setdefault('keys_errors_file_path', ofp.keys_errors_file_path)

            kwargs.setdefault('canonical_exposures_data_frame', omr.get('canonical_exposures_data_frame'))
            kwargs.setdefault('gul_master_data_frame', omr.get('gul_master_data_frame'))

            kwargs.setdefault('items_file_path', ofp.items_file_path)
            kwargs.setdefault('coverages_file_path', ofp.coverages_file_path)
            kwargs.setdefault('gulsummaryxref_file_path', ofp.gulsummaryxref_file_path)

            kwargs.setdefault('fm_master_data_frame', omr.get('fm_master_data_frame'))

            kwargs.setdefault('fm_policytc_file_path', ofp.fm_policytc_file_path)
            kwargs.setdefault('fm_profile_file_path', ofp.fm_profile_file_path)
            kwargs.setdefault('fm_policytc_file_path', ofp.fm_programme_file_path)
            kwargs.setdefault('fm_xref_file_path', ofp.fm_xref_file_path)
            kwargs.setdefault('fmsummaryxref_file_path', ofp.fmsummaryxref_file_path)

        if not kwargs.get('canonical_exposures_profile'):
            kwargs['canonical_exposures_profile'] = self.load_canonical_exposures_profile(
                oasis_model=oasis_model,
                canonical_exposures_profile_json=kwargs.get('canonical_exposures_profile_json'),
                canonical_exposures_profile_json_path=kwargs.get('canonical_exposures_profile_json_path'),
            )

        if fm and not kwargs.get('canonical_accounts_profile'):
            kwargs['canonical_accounts_profile'] = self.load_canonical_accounts_profile(
                oasis_model=oasis_model,
                canonical_accounts_profile_json=kwargs.get('canonical_accounts_profile_json'),
                canonical_accounts_profile_json_path=kwargs.get('canonical_accounts_profile_json_path'),
            )

        return kwargs

    def generate_gul_items(
        self,
        canonical_exposures_profile,
        canonical_exposures_file_path,
        keys_file_path
    ):
        pass

    def generate_fm_items(
        self,
        canonical_exposures,
        gul_items,
        canonical_exposures_profile,
        canonical_accounts_profile,
        canonical_accounts_file_path,
        preset_only=False
    ):
        pass

    def load_gul_master_data_frame(
        self,
        canonical_exposures_profile,
        canonical_exposures_file_path,
        keys_file_path
    ):
        """
        Generates a pandas dataframe containing all the columns required to
        write the GUL files.
        """
        try:
            with io.open(canonical_exposures_file_path, 'r', encoding='utf-8') as cf, io.open(keys_file_path, 'r', encoding='utf-8') as kf:
                canexp_df, keys_df = pd.read_csv(cf, float_precision='high'), pd.read_csv(kf, float_precision='high')

            if len(canexp_df) == 0:
                raise OasisException('No canonical exposure items found - please check the canonical exposures (loc) file')

            if len(keys_df) == 0:
                raise OasisException('No keys items found - please check the model exposures (loc) file')
            
            canexp_df = canexp_df.where(canexp_df.notnull(), None)
            canexp_df.columns = canexp_df.columns.str.lower()
            canexp_df['index'] = pd.Series(data=list(canexp_df.index), dtype=object)

            keys_df = keys_df.rename(columns={'CoverageID': 'CoverageType'})
            keys_df = keys_df.where(keys_df.notnull(), None)
            keys_df.columns = keys_df.columns.str.lower()
            keys_df['index'] = pd.Series(data=list(keys_df.index), dtype=object)

            merged_df = pd.merge(canexp_df, keys_df, left_on='row_id', right_on='locid')

            if len(merged_df) == 0:
                raise OasisException('No matching keys items found for canonical exposure items - please check the model exposures (loc) file')

            merged_df['index'] = pd.Series(data=list(merged_df.index), dtype=object)

            cep = canonical_exposures_profile

            tiv_elements = tuple(sorted(
                [v for v in cep.itervalues() if v.get('FieldName') and v['FieldName'] == 'TIV'],
                key=lambda v: v['FMTermGroupID']
            ))

            if not tiv_elements:
                raise OasisException('No TIV elements found in the canonical exposures profile - please check the canonical exposures (loc) profile')

            columns = (
                'item_id',
                'canexp_id',
                'coverage_id',
                'tiv',
                'areaperil_id',
                'vulnerability_id',
                'group_id',
                'summary_id',
                'summaryset_id',
            )
            gulm_df = pd.DataFrame(columns=list(columns), dtype=object)

            item_id = 0
            zero_tiv_items = 0
            for _, item in merged_df.iterrows():
                positive_tiv_elements = [t for t in tiv_elements if item.get(t['ProfileElementName'].lower()) and item[t['ProfileElementName'].lower()] > 0 and t['CoverageTypeID'] == item['coveragetype']]

                if not positive_tiv_elements:
                    zero_tiv_items += 1
                    continue

                for _, t in enumerate(positive_tiv_elements):
                    item_id += 1
                    tiv = item[t['ProfileElementName'].lower()]
                    gulm_df = gulm_df.append([{
                        'item_id': item_id,
                        'canexp_id': item['row_id'] - 1,
                        'coverage_id': item_id,
                        'tiv': tiv,
                        'areaperil_id': item['areaperilid'],
                        'vulnerability_id': item['vulnerabilityid'],
                        'group_id': item_id,
                        'summary_id': 1,
                        'summaryset_id': 1
                    }])
        except (KeyError, IndexError, IOError, OasisException, OSError, TypeError, ValueError) as e:
            raise OasisException(e)
        else:
            if zero_tiv_items == len(merged_df):
                raise OasisException('All canonical exposure items have zero TIVs - please check the canonical exposures (loc) file')

            for col in gulm_df.columns:
                gulm_df[col] = gulm_df[col].astype(float) if col == 'tiv' else gulm_df[col].astype(int)

            gulm_df.index = range(len(gulm_df))

            return gulm_df, canexp_df
    
    def load_fm_master_data_frame(
        self,
        canonical_exposures_data_frame,
        gul_master_data_frame,
        canonical_exposures_profile,
        canonical_accounts_profile,
        canonical_accounts_file_path,
        preset_only=False
    ):
        """
        Generates a pandas dataframe containing all the columns required to
        write the FM files.
        """
        try:
            canexp_df = canonical_exposures_data_frame
            gulm_df = gul_master_data_frame

            with io.open(canonical_accounts_file_path, 'r', encoding='utf-8') as f:
                canacc_df = pd.read_csv(f, float_precision='high')

            if len(canacc_df) == 0:
                raise OasisException('No canonical accounts items')
            
            canacc_df = canacc_df.where(canacc_df.notnull(), None)
            canacc_df.columns = canacc_df.columns.str.lower()
            canacc_df['index'] = pd.Series(data=list(canacc_df.index), dtype=object)
            
            columns = (
                'item_id', 'canexp_id', 'canacc_id', 'level_id', 'layer_id', 'agg_id', 'policytc_id', 'deductible',
                'limit', 'share', 'deductible_type', 'calcrule_id', 'tiv',
            )

            cep = canonical_exposures_profile
            cap = canonical_accounts_profile

            gfmt = canonical_profiles_fm_terms_grouped_by_level_and_term_type(canonical_profiles=(cep, cap,))

            fm_levels = tuple(sorted(gfmt.keys()))

            preset_data = [
                p for p in itertools.product(
                    fm_levels,
                    zip(tuple(gulm_df.item_id.values), tuple(gulm_df.canexp_id.values), (-1,)*len(gulm_df), (1,)*len(gulm_df), tuple(gulm_df.tiv.values))
                )
            ]

            layer_ids = tuple(range(1, len(canacc_df.policynum.values) + 1))

            if max(layer_ids) > 1:
                preset_data.extend([
                    (t[1][0], (t[1][1][0], t[1][1][1], t[1][1][2], t[0], t[1][1][4]))
                    for t in itertools.product(layer_ids[1:], preset_data[-len(gulm_df):])
                ])
            
            preset_data = tuple(preset_data)

            data = tuple(
                {
                    k:v for k, v in zip(
                        columns,
                        [item_id,canexp_id,canacc_id,fm_levels.index([fml for fml in fm_levels if fml == level_id][0]) + 1,layer_id,1,0,0.0,0.0,0.0,u'B',2,tiv])
                } for level_id, (item_id, canexp_id, canacc_id, layer_id, tiv) in preset_data
            )

            fm_df = pd.DataFrame(columns=list(columns), data=list(data), dtype=object)

            fm_df['index'] = pd.Series(data=list(fm_df.index), dtype=object)

            get_canexp_item = lambda i: canexp_df.iloc[fm_df.iloc[i]['canexp_id']]

            get_item_layer = lambda i: list(canacc_df[canacc_df['accntnum'] == get_canexp_item(i)['accntnum']]['policynum'].values)[int(fm_df.iloc[i]['layer_id']) - 1]

            get_canacc_item = lambda i: canacc_df[canacc_df['accntnum'] == get_canexp_item(i)['accntnum']][canacc_df['policynum'] == get_item_layer(i)]

            fm_df['canacc_id'] = pd.Series(data=fm_df['index'].apply(lambda i: int(get_canacc_item(i)['index'])), dtype=object)

            if preset_only:
                return fm_df, canacc_df

            concurrent_tasks = (
                Task(get_fm_terms_by_level_as_list, args=(level_id, gfmt[level_id], canexp_df.copy(deep=True), canacc_df.copy(deep=True), pd.DataFrame(fm_df.where(fm_df['level_id'] == level_id).drop_duplicates().dropna(), dtype=object),), key=level_id)
                for level_id in fm_levels
            )

            fm_terms = {
                result['index']:result for result in multiprocess(concurrent_tasks, pool_size=len(fm_levels))
            }

            def set_col(col):
                fm_df[col] = fm_df['index'].apply(lambda i: fm_terms[i][col])

            fm_terms_cols = ('limit', 'deductible', 'deductible_type', 'share', 'calcrule_id',)

            for col in fm_terms_cols:
                set_col(col)

            policytc_ids = get_policytc_ids(fm_df)

            fm_df['policytc_id'] = fm_df['index'].apply(lambda i: get_policytc_id(fm_df.iloc[i], policytc_ids))
        except (KeyError, IndexError, IOError, OSError, TypeError, ValueError) as e:
            raise OasisException(e)
        else:
            return fm_df, canacc_df

    def write_items_file(self, gul_master_data_frame, items_file_path):
        """
        Writes an items file.
        """
        gulm_df = gul_master_data_frame

        try:
            gulm_df.to_csv(
                columns=['item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id'],
                path_or_buf=items_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return items_file_path

    def write_coverages_file(self, gul_master_data_frame, coverages_file_path):
        """
        Writes a coverages file.
        """
        gulm_df = gul_master_data_frame

        try:
            gulm_df.to_csv(
                columns=['coverage_id', 'tiv'],
                path_or_buf=coverages_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return coverages_file_path

    def write_gulsummaryxref_file(self, gul_master_data_frame, gulsummaryxref_file_path):
        """
        Writes a gulsummaryxref file.
        """
        gulm_df = gul_master_data_frame

        try:
            gulm_df.to_csv(
                columns=['coverage_id', 'summary_id', 'summaryset_id'],
                path_or_buf=gulsummaryxref_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return gulsummaryxref_file_path

    def write_fm_policytc_file(self, fm_master_data_frame, fm_policytc_file_path):
        """
        Writes an FM policy T & C file.
        """
        fm_df = fm_master_data_frame

        try:
            fm_df.to_csv(
                columns=['layer_id', 'level_id', 'agg_id', 'policytc_id'],
                path_or_buf=fm_policytc_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return fm_policytc_file_path

    def write_fm_profile_file(self, fm_master_data_frame, fm_profile_file_path):
        """
        Writes an FM profile file.
        """
        pass

    def write_fm_programme_file(self, fm_master_data_frame, fm_programme_file_path):
        """
        Writes a FM programme file.
        """
        pass

    def write_fm_xref_file(self, fm_master_data_frame, fm_xref_file_path):
        """
        Writes a FM xref file.
        """
        pass

    def write_fmsummaryxref_file(self, fm_master_data_frame, fmsummaryxref_file_path):
        """
        Writes an FM summaryxref file.
        """
        pass

    def write_gul_files(self, oasis_model=None, **kwargs):
        """
        Writes the standard Oasis GUL files, namely::

            items.csv
            coverages.csv
            gulsummaryxref.csv
        """
        if oasis_model:
            omr = oasis_model.resources
            ofp = omr['oasis_files_pipeline']

        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        canonical_exposures_profile = kwargs.get('canonical_exposures_profile')
        canonical_exposures_file_path = kwargs.get('canonical_exposures_file_path')
        keys_file_path = kwargs.get('keys_file_path')
        
        gulm_df, canexp_df = self.load_gul_master_data_frame(canonical_exposures_profile, canonical_exposures_file_path, keys_file_path)

        if oasis_model:
            omr['canonical_exposures_data_frame'] = canexp_df
            omr['gul_master_data_frame'] = gulm_df

        gul_files = (
            ofp.gul_files if oasis_model
            else {
                 'items': kwargs.get('items_file_path'),
                 'coverages': kwargs.get('coverages_file_path'),
                 'gulsummaryxref': kwargs.get('gulsummaryxref_file_path')
            }
        )

        concurrent_tasks = (
            Task(getattr(self, 'write_{}_file'.format(f)), args=(gulm_df, gul_files[f],), key=f)
            for f in gul_files
        )

        for _, _ in multithread(concurrent_tasks, pool_size=len(gul_files)):
            pass

        return gul_files

    def write_fm_files(self, oasis_model=None, **kwargs):
        """
        Generate Oasis FM files, namely::

            fm_policytc.csv
            fm_profile.csv
            fm_programm.ecsv
            fm_xref.csv
            fm_summaryxref.csv
        """
        if oasis_model:
            omr = oasis_model.resources
            ofp = omr['oasis_files_pipeline']

        if oasis_model:
            canexp_df, gulm_df = omr.get('canonical_exposures_data_frame'), omr.get('gul_master_data_frame')
        else:
            canexp_df, gulm_df = kwargs.get('canonical_exposures_data_frame'), kwargs.get('gul_master_data_frame')

        canonical_exposures_profile = kwargs.get('canonical_exposures_profile')
        canonical_accounts_profile = kwargs.get('canonical_accounts_profile')
        canonical_accounts_file_path = kwargs.get('canonical_accounts_file_path')

        fm_df, canacc_df = self.load_fm_master_data_frame(canexp_df, gulm_df, canonical_exposures_profile, canonical_accounts_profile, canonical_accounts_file_path)

        if oasis_model:
            omr['canonical_accounts_data_frame'] = canacc_df
            omr['fm_master_data_frame'] = fm_df
        else:
            kwargs['canonical_accounts_data_frame'] = canacc_df
            kwargs['fm_master_data_frame'] = fm_df

        fm_files = (
            ofp.fm_files if oasis_model
            else  {
                'fm_policytc': kwargs.get('fm_policytc_file_path'),
                'fm_profile': kwargs.get('fm_profile_file_path'),
                'fm_programme': kwargs.get('fm_programme_file_path'),
                'fm_xref': kwargs.get('fm_xref_file_path'),
                'fmsummaryxref': kwargs.get('fmsummaryxref_file_path')
            }
        )

        concurrent_tasks = (
            Task(getattr(self, 'write_{}_file'.format(f)), args=(fm_df, fm_files[f],), key=f)
            for f in fm_files
        )

        for _, _ in multithread(concurrent_tasks, pool_size=len(fm_files)):
            pass

        return fm_files

    def write_oasis_files(self, oasis_model=None, fm=False, **kwargs):
        """
        Writes the Oasis files - GUL + FM (if ``fm`` is ``True``).
        """
        gul_files = self.write_gul_files(oasis_model=oasis_model, **kwargs)

        if not fm:
            return gul_files

        fm_files = self.write_fm_files(oasis_model=oasis_model, **kwargs)

        oasis_files = {k:v for k, v in itertools.chain(gul_files.items(), fm_files.items())}

        return oasis_files

    def clear_oasis_files_pipeline(self, oasis_model, **kwargs):
        """
        Clears the files pipeline for the given Oasis model object.
        """
        oasis_model.resources.get('oasis_files_pipeline').clear()

        return oasis_model

    def start_oasis_files_pipeline(
        self,
        oasis_model=None,
        oasis_files_path=None, 
        fm=False,
        source_exposures_file_path=None,
        source_accounts_file_path=None,
        logger=None
    ):
        """
        Starts the files pipeline for the given Oasis model object,
        which is the generation of the Oasis items, coverages and GUL summary
        files, and possibly the FM files, from the source exposures file,
        source accounts file, canonical exposures profile, and associated
        validation files and transformation files for the source and
        intermediate files (canonical exposures, model exposures).

        :param oasis_model: The Oasis model object
        :type oasis_model: oasislmf.models.model.OasisModel

        :param oasis_files_path: Path where generated Oasis files should be
                                 written
        :type oasis_files_path: str

        :param fm: Boolean indicating whether FM files should be generated
        :param fm: bool

        :param source_exposures_file_path: Path to the source exposures file
        :type source_exposures_file_path: str

        :param source_accounts_file_path: Path to the source accounts file
        :type source_accounts_file_path: str

        :param logger: Logger object
        :type logger: logging.Logger

        :return: A dictionary of Oasis files (GUL + FM (if FM option indicated))
        """
        if oasis_model:
            omr = oasis_model.resources
            ofp = omr['oasis_files_pipeline']

            ofp.clear()

        logger = logger or logging.getLogger()

        logger.info('\nChecking Oasis files directory exists for model')
        if oasis_model and not oasis_files_path:
            oasis_files_path = omr.get('oasis_files_path')

        if not oasis_files_path:
            raise OasisException('No Oasis files directory provided.'.format(oasis_model))
        elif not os.path.exists(oasis_files_path):
            raise OasisException('Oasis files directory {} does not exist on the filesystem.'.format(oasis_files_path))

        logger.info('Oasis files directory is {}'.format(oasis_files_path))

        logger.info('\nChecking for source exposures file')
        if oasis_model and not source_exposures_file_path:
            source_exposures_file_path = omr.get('source_exposures_file_path') or ofp.source_exposures_file_path
        if not source_exposures_file_path:
            raise OasisException('No source exposures file path provided in arguments or model resources')
        elif not os.path.exists(source_exposures_file_path):
            raise OasisException("Source exposures file path {} does not exist on the filesysem.".format(source_exposures_file_path))

        if fm:
            logger.info('\nChecking for source accounts file')
            if oasis_model and not source_accounts_file_path:
                source_accounts_file_path = omr.get('source_accounts_file_path') or ofp.source_accounts_file_path
            if not source_accounts_file_path:
                raise OasisException('FM option indicated but no source accounts file path provided in arguments or model resources')
            elif not os.path.exists(source_accounts_file_path):
                raise OasisException("Source accounts file path {} does not exist on the filesysem.".format(source_accounts_file_path))

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')

        canonical_exposures_file_path = os.path.join(oasis_files_path, 'canexp-{}.csv'.format(utcnow))
        canonical_accounts_file_path = os.path.join(oasis_files_path, 'canacc-{}.csv'.format(utcnow))

        model_exposures_file_path = os.path.join(oasis_files_path, 'modexp-{}.csv'.format(utcnow))

        keys_file_path = os.path.join(oasis_files_path, 'oasiskeys-{}.csv'.format(utcnow))
        keys_errors_file_path = os.path.join(oasis_files_path, 'oasiskeys-errors-{}.csv'.format(utcnow))

        items_file_path = os.path.join(oasis_files_path, 'items.csv')
        coverages_file_path = os.path.join(oasis_files_path, 'coverages.csv')
        gulsummaryxref_file_path = os.path.join(oasis_files_path, 'gulsummaryxref.csv')

        fm_policytc_file_path = os.path.join(oasis_files_path, 'fm_policytc.csv')
        fm_profile_file_path = os.path.join(oasis_files_path, 'fm_profile.csv')
        fm_programme_file_path=os.path.join(oasis_files_path, 'fm_programme.csv')
        fm_xref_file_path = os.path.join(oasis_files_path, 'fm_xref.csv')
        fmsummaryxref_file_path = os.path.join(oasis_files_path, 'fmsummaryxref.csv')

        if oasis_model:
            ofp.source_exposures_file_path = source_exposures_file_path
            ofp.source_accounts_file_path = source_accounts_file_path

            ofp.canonical_exposures_file_path = canonical_exposures_file_path
            ofp.canonical_accounts_file_path = canonical_exposures_file_path

            ofp.model_exposures_file_path = model_exposures_file_path

            ofp.keys_file_path = keys_file_path
            ofp.keys_errors_file_path = keys_errors_file_path

            ofp.items_file_path = items_file_path
            ofp.coverages_file_path = coverages_file_path
            ofp.gulsummaryxref_file_path = gulsummaryxref_file_path

            ofp.fm_policytc_file_path = fm_policytc_file_path
            ofp.fm_profile_file_path = fm_profile_file_path
            ofp.fm_programme_file_path = fm_programme_file_path
            ofp.fm_xref_file_path = fm_xref_file_path
            ofp.fmsummaryxref_file_path = fmsummaryxref_file_path

        kwargs = self._process_default_kwargs(
            oasis_model=oasis_model,
            fm=fm,
            source_exposures_file_path=source_exposures_file_path,
            source_accounts_file_path=source_accounts_file_path,
            canonical_exposures_file_path=canonical_exposures_file_path,
            canonical_accounts_file_path=canonical_accounts_file_path,
            model_exposures_file_path=model_exposures_file_path,
            keys_file_path=keys_file_path,
            keys_errors_file_path=keys_errors_file_path,
            items_file_path=items_file_path,
            coverages_file_path=coverages_file_path,
            gulsummaryxref_file_path=gulsummaryxref_file_path,
            fm_policytc_file_path=fm_policytc_file_path,
            fm_profile_file_path=fm_profile_file_path,
            fm_programme_file_path=fm_programme_file_path,
            fm_xref_file_path=fm_xref_file_path,
            fmsummaryxref_file_path=fmsummaryxref_file_path
        )

        source_exposures_file_path = kwargs.get('source_exposures_file_path')
        self.logger.info('\nCopying source exposures file {source_exposures_file_path} to Oasis files directory'.format(**kwargs))
        shutil.copy2(source_exposures_file_path, oasis_files_path)

        if fm:
            source_accounts_file_path = kwargs.get('source_accounts_file_path')
            self.logger.info('\nCopying source accounts file {source_accounts_file_path} to Oasis files directory'.format(**kwargs))
            shutil.copy2(source_accounts_file_path, oasis_files_path)

        logger.info('\nWriting canonical exposures file {canonical_exposures_file_path}'.format(**kwargs))
        self.transform_source_to_canonical(oasis_model=oasis_model, **kwargs)

        if fm:
            logger.info('\nWriting canonical accounts file {canonical_accounts_file_path}'.format(**kwargs))
            self.transform_source_to_canonical(oasis_model=oasis_model, source_type='accounts', **kwargs)

        logger.info('\nWriting model exposures file {model_exposures_file_path}'.format(**kwargs))
        self.transform_canonical_to_model(oasis_model=oasis_model, **kwargs)

        logger.info('\nWriting keys file {keys_file_path} and keys errors file {keys_errors_file_path}'.format(**kwargs))
        self.get_keys(oasis_model=oasis_model, **kwargs)

        logger.info('\nWriting GUL files')
        gul_files = self.write_gul_files(oasis_model=oasis_model, **kwargs)

        if not fm:
            return gul_files

        logger.info('\nWriting FM files')
        fm_files = self.write_fm_files(oasis_model=oasis_model, **kwargs)

        oasis_files = ofp.oasis_files if oasis_model else {k:v for k, v in itertools.chain(gul_files.items(), fm_files.items())}

        return oasis_files

    def create_model(self, model_supplier_id, model_id, model_version_id, resources=None):
        model = OasisModel(
            model_supplier_id,
            model_id,
            model_version_id,
            resources=resources
        )

        # set default resources
        model.resources.setdefault('oasis_files_path', os.path.abspath(os.path.join('Files', model.key.replace('/', '-'))))
        if not os.path.isabs(model.resources.get('oasis_files_path')):
            model.resources['oasis_files_path'] = os.path.abspath(model.resources['oasis_files_path'])

        model.resources.setdefault('oasis_files_pipeline', OasisFilesPipeline(model_key=model.key))
        if not isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline):
            raise OasisException(
                'Oasis files pipeline object for model {} is not of type {}'.format(model, OasisFilesPipeline))

        if model.resources.get('canonical_exposures_profile') is None:
            self.load_canonical_exposures_profile(oasis_model=model)

        if (
            model.resources.get('canonical_accounts_profile_json_path') or
            model.resources.get('canonical_accounts_profile_json') or
            model.resources.get('canonical_accounts_profile')
        ) and model.resources.get('source_accounts_file_path'):
            if model.resources.get('canonical_accounts_profile') is None:
                self.load_canonical_accounts_profile(oasis_model=model)

        self.add_model(model)

        return model
