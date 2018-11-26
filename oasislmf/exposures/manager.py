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
import multiprocessing
import os
import shutil
import six
import sys
import time

import pandas as pd

from interface import (
    Interface,
    implements,
)


from ..keys.lookup import OasisLookupFactory
from ..utils.concurrency import (
    multiprocess,
    multithread,
    Task,
)
from ..utils.exceptions import OasisException
from ..utils.fm import (
    unified_canonical_fm_profile_by_level_and_term_group,
    get_fm_terms_by_level_as_list,
    get_policytc_ids,
)
from ..utils.metadata import (
    OASIS_COVERAGE_TYPES,
    OASIS_FM_LEVELS,
    OASIS_PERILS,
    OED_COVERAGE_TYPES,
    OED_FM_LEVELS,
    OED_PERILS,
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

            LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID

        while the keys error file is a CSV file containing keys lookup
        information for locations with unsuccessful lookups (failures,
        no matches) and has the following headers::

            LocID,PerilID,CoverageTypeID,Message

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
        exposures profile for a given model.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def load_canonical_accounts_profile(self, oasis_model=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the canonical
        accounts profile for a given model.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def load_fm_aggregation_profile(self, oasis_model=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the FM
        aggregation profile for a given model.

        :param oasis_model: An Oasis model object 
        :type oasis_model: oasislmf.models.model.OasisModel

        :param kwargs: Optional keyword arguments
        """
        pass

    def generate_gul_items(self, canonical_exposures_profile, canonical_exposures_df, keys_df, **kwargs):
        """
        Generates GUL items.

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_exposures_df: Canonical exposures
        :type canonical_exposures_df: pandas.DataFrame

        :param keys_df: Keys
        :type keys_df: pandas.DataFrame
        """
        pass

    def generate_fm_items(self, canonical_exposures_df, gul_items_df, canonical_exposures_profile, canonical_accounts_profile, canonical_accounts_df, fm_agg_profile, **kwargs):
        """
        Generates FM items.

        :param canonical_exposures_df: Canonical exposures
        :type canonical_exposures_df: pandas.DataFrame

        :param gul_items_df: GUL items
        :type gul_items_df: pandas.DataFrame

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_accounts_profile: Canonical accounts profile
        :type canonical_accounts_profile: dict

        :param canonical_accounts_df: Canonical accounts
        :param canonical_accounts_df: pandas.DataFrame

        :param fm_agg_profile: FM aggregation profile
        :param fm_agg_profile: dict
        """
        pass

    def load_gul_items(self, canonical_exposures_profile, canonical_exposures_file_path, keys_file_path, **kwargs):
        """
        Loads GUL items generated by ``generate_gul_items`` into a static
        structure such as a pandas dataframe.

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_exposures_file_path: Canonical exposures file path
        :type canonical_exposures_file_path: str

        :param keys_file_path: Keys file path
        :type keys_file_path: str
        """
        pass

    def load_fm_items(self, canonical_exposures_df, gul_items_df, canonical_exposures_profile, canonical_accounts_profile, canonical_accounts_file_path, fm_agg_profile, **kwargs):
        """
        Loads FM items generated by ``generate_fm_items`` into a static
        structure such as a pandas dataframe.

        :param canonical_exposures_df: Canonical exposures
        :type canonical_exposures_df: pandas.DataFrame

        :param gul_items_df: GUL items
        :type gul_items_df: pandas.DataFrame

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_accounts_profile: Canonical accounts profile
        :type canonical_accounts_profile: dict

        :param canonical_accounts_file_path: Canonical accounts file path
        :param canonical_accounts_file_path: str

        :param fm_agg_profile: FM aggregation profile
        :param fm_agg_profile: dict
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

    def create_model(self, model_supplier_id, model_id, model_version, resources=None):
        """
        Creates an Oasis model object, with attached resources if a resources
        dict was provided.

        :param model_supplier_id: The model supplier ID
        :type model_supplier_id: str

        :param model_id: The model ID
        :type model_id: str

        :param model_version: The model version string
        :type model_version: str

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
    def models(self, key=None):
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
        return self._models[key] if key else self._models

    @models.setter
    def models(self, val, key=None):
        if key:
            self._models.update({key:val})
        else:
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
        transformation_file_path = os.path.abspath(kwargs['source_to_canonical_accounts_transformation_file_path']) if source_type == 'accounts' else os.path.abspath(kwargs['source_to_canonical_exposures_transformation_file_path'])

        try:
            validation_file_path = kwargs['source_accounts_validation_file_path'] if source_type == 'accounts' else kwargs.get['source_exposures_validation_file_path']
        except (KeyError, TypeError) as e:
            validation_file_path = None

        output_file_path = os.path.abspath(kwargs['canonical_accounts_file_path']) if source_type == 'accounts' else os.path.abspath(kwargs['canonical_exposures_file_path'])

        translator = Translator(input_file_path, output_file_path, transformation_file_path, xsd_path=validation_file_path, append_row_nums=True)

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

        input_file_path = os.path.abspath(kwargs.get('canonical_exposures_file_path'))
        transformation_file_path = os.path.abspath(kwargs.get('canonical_to_model_exposures_transformation_file_path'))

        try:
            validation_file_path = kwargs['canonical_exposures_validation_file_path']
        except (KeyError, TypeError) as e:
            validation_file_path = None

        output_file_path = os.path.abspath(kwargs.get('model_exposures_file_path'))

        translator = Translator(input_file_path, output_file_path, transformation_file_path, xsd_path=validation_file_path, append_row_nums=False)

        translator()

        if oasis_model:
            oasis_model.resources['oasis_files_pipeline'].model_exposures_file_path = output_file_path

        return output_file_path

    def load_canonical_exposures_profile(self, oasis_model=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the canonical
        exposures profile for a given ``oasis_model``, stores this in the
        model object's resources dict, and returns the object.
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        profile_json = kwargs.get('canonical_exposures_profile_json')
        profile_json_path = kwargs.get('canonical_exposures_profile_json_path')

        profile = None
        if profile_json:
            profile = json.loads(profile_json)
        elif profile_json_path:
            with io.open(profile_json_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)

        if oasis_model:
            oasis_model.resources['canonical_exposures_profile'] = profile

        return profile

    def load_canonical_accounts_profile(self, oasis_model=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the canonical
        exposures profile for a given ``oasis_model``, stores this in the
        model object's resources dict, and returns the object.
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        profile_json = kwargs.get('canonical_accounts_profile_json')
        profile_json_path = kwargs.get('canonical_accounts_profile_json_path')

        profile = None
        if profile_json:
            profile = json.loads(profile_json)
        elif profile_json_path:
            with io.open(profile_json_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)

        if oasis_model:
            oasis_model.resources['canonical_accounts_profile'] = profile

        return profile

    def load_fm_aggregation_profile(self, oasis_model=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the FM
        aggregation profile for a given ``oasis_model``, stores this in the
        model object's resources dict, and returns the object.
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        profile_path = kwargs.get('fm_agg_profile_path')
        profile_json = kwargs.get('fm_agg_profile_json')

        profile = None
        if profile_json:
            profile = {int(k):v for k, v in six.iteritems(json.loads(profile_json))}
        elif profile_path:
            with io.open(profile_path, 'r', encoding='utf-8') as f:
                profile = {int(k):v for k, v in six.iteritems(json.load(f))}

        if oasis_model:
            oasis_model.resources['fm_agg_profile'] = profile

        return profile

    def load_lookup_config(self, oasis_model=None, **kwargs):
        """
        Loads a lookup config JSON string or file.
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        lookup = kwargs.get('lookup')

        if not lookup:
            return

        try:
            config = lookup.config
        except AttributeError:
            pass
        else:
            if oasis_model:
                oasis_model.resources['lookup_config'] = config

            return config

        config_json = kwargs.get('lookup_config_json')
        config_fp = kwargs.get('lookup_config_fp')

        config = None
        if config_json:
            config = json.loads(config_json)
        elif config_fp:
            with io.open(config_fp, 'r', encoding='utf-8') as f:
                config = json.load(f)

        if oasis_model:
            oasis_model.resources['lookup_config'] = config

        return config

    def get_keys(self, oasis_model=None, **kwargs):
        """
        Generates the Oasis keys and keys error files for a given model object.
        The keys file is a CSV file containing keys lookup information for
        locations with successful lookups, and has the following headers::

            LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID

        while the keys error file is a CSV file containing keys lookup
        information for locations with unsuccessful lookups (failures,
        no matches) and has the following headers::

            LocID,PerilID,CoverageTypeID,Message

        By default it is assumed that all the resources required for the
        transformation are present in the model object's resources dict,
        if the model is supplied. These can be overridden by providing the
        relevant optional parameters.

        If no model is supplied then the optional paramenters must be
        supplied.

        If the model is supplied the result keGy file path is stored in the
        models ``file_pipeline.keyfile_path`` property.

        :param oasis_model: The model to get keys for
        :type oasis_model: ``OasisModel``

        :return: The path to the generated keys file
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        model_exposures_file_path = kwargs.get('model_exposures_file_path')
        lookup = kwargs.get('lookup')
        keys_file_path = kwargs.get('keys_file_path')
        keys_errors_file_path = kwargs.get('keys_errors_file_path')

        for p in (model_exposures_file_path, keys_file_path, keys_errors_file_path,):
            p = os.path.abspath(p) if p and not os.path.isabs(p) else p

        keys_file_path, _, keys_errors_file_path, _ = OasisLookupFactory().save_results(
            lookup,
            keys_file_path,
            errors_fp=keys_errors_file_path,
            model_exposures_fp=model_exposures_file_path
        )

        if oasis_model:
            oasis_model.resources['oasis_files_pipeline'].keys_file_path = keys_file_path
            oasis_model.resources['oasis_files_pipeline'].keys_errors_file_path = keys_errors_file_path

        return keys_file_path, keys_errors_file_path

    def _process_default_kwargs(self, oasis_model=None, **kwargs):
        if oasis_model:
            omr = oasis_model.resources
            ofp = omr['oasis_files_pipeline']

            kwargs.setdefault('fm', omr.get('fm') or kwargs.get('fm'))

            kwargs.setdefault('logger', omr.get('logger') or logging.getLogger())

            kwargs.setdefault('oasis_files_path', omr.get('oasis_files_path'))

            kwargs.setdefault('source_exposures_file_path', omr.get('source_exposures_file_path') or ofp.source_exposures_file_path)
            kwargs.setdefault('source_accounts_file_path', omr.get('source_accounts_file_path') or ofp.source_accounts_file_path)

            kwargs.setdefault('source_exposures_validation_file_path', omr.get('source_exposures_validation_file_path'))
            kwargs.setdefault('source_accounts_validation_file_path', omr.get('source_accounts_validation_file_path'))

            kwargs.setdefault('source_to_canonical_exposures_transformation_file_path', omr.get('source_to_canonical_exposures_transformation_file_path'))
            kwargs.setdefault('source_to_canonical_accounts_transformation_file_path', omr.get('source_to_canonical_accounts_transformation_file_path'))

            kwargs.setdefault('canonical_exposures_profile', omr.get('canonical_exposures_profile'))
            kwargs.setdefault('canonical_accounts_profile', omr.get('canonical_accounts_profile'))

            kwargs.setdefault('canonical_exposures_profile', omr.get('canonical_exposures_profile'))
            kwargs.setdefault('canonical_accounts_profile', omr.get('canonical_accounts_profile'))

            kwargs.setdefault('canonical_exposures_profile_json', omr.get('canonical_exposures_profile_json'))
            kwargs.setdefault('canonical_accounts_profile_json', omr.get('canonical_accounts_profile_json'))

            kwargs.setdefault('fm_agg_profile', omr.get('fm_agg_profile'))
            kwargs.setdefault('fm_agg_profile_path', omr.get('fm_agg_profile_path'))
            kwargs.setdefault('fm_agg_profile_json', omr.get('fm_agg_profile_json'))

            kwargs.setdefault('canonical_exposures_profile_json_path', omr.get('canonical_exposures_profile_json_path'))
            kwargs.setdefault('canonical_accounts_profile_json_path', omr.get('canonical_accounts_profile_json_path'))

            kwargs.setdefault('canonical_exposures_file_path', ofp.canonical_exposures_file_path)
            kwargs.setdefault('canonical_accounts_file_path', ofp.canonical_accounts_file_path)

            kwargs.setdefault('canonical_exposures_validation_file_path', omr.get('canonical_exposures_validation_file_path'))
            kwargs.setdefault('canonical_to_model_exposures_transformation_file_path', omr.get('canonical_to_model_exposures_transformation_file_path'))

            kwargs.setdefault('lookup_config', omr.get('lookup_config'))
            kwargs.setdefault('lookup_config_json', omr.get('lookup_config_json'))
            kwargs.setdefault('lookup_config_fp', omr.get('lookup_config_fp'))
            kwargs.setdefault('lookup', omr.get('lookup'))

            kwargs.setdefault('model_exposures_file_path', ofp.model_exposures_file_path)

            kwargs.setdefault('keys_file_path', ofp.keys_file_path)
            kwargs.setdefault('keys_errors_file_path', ofp.keys_errors_file_path)

            kwargs.setdefault('canonical_exposures_df', omr.get('canonical_exposures_df'))
            kwargs.setdefault('gul_items_df', omr.get('gul_items_df'))

            kwargs.setdefault('items_file_path', ofp.items_file_path)
            kwargs.setdefault('coverages_file_path', ofp.coverages_file_path)
            kwargs.setdefault('gulsummaryxref_file_path', ofp.gulsummaryxref_file_path)

            kwargs.setdefault('fm_items_df', omr.get('fm_items_df'))
            kwargs.setdefault('canonical_accounts_df', omr.get('canonical_accounts_df'))

            kwargs.setdefault('fm_policytc_file_path', ofp.fm_policytc_file_path)
            kwargs.setdefault('fm_profile_file_path', ofp.fm_profile_file_path)
            kwargs.setdefault('fm_policytc_file_path', ofp.fm_programme_file_path)
            kwargs.setdefault('fm_xref_file_path', ofp.fm_xref_file_path)
            kwargs.setdefault('fmsummaryxref_file_path', ofp.fmsummaryxref_file_path)

        return kwargs

    def generate_gul_items(
        self,
        canonical_exposures_profile,
        canonical_exposures_df,
        keys_df
    ):
        """
        Generates GUL items.

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_exposures_df: Canonical exposures data frame
        :type canonical_exposures_df: pandas.DataFrame

        :param keys_df: Keys file data frame
        :type keys_df: pandas.DataFrame
        """
        canexp_df = canonical_exposures_df

        cep = canonical_exposures_profile

        ufcp = unified_canonical_fm_profile_by_level_and_term_group(profiles=(cep,))

        if not ufcp:
            raise OasisException(
                'Canonical loc. profile is possibly missing FM term information: '
                'FM term definitions for TIV, limit, deductible, attachment and/or share.'
            )

        fm_levels = tuple(ufcp.keys())

        try:
            for df in [canexp_df, keys_df]:
                if not df.columns.contains('index'):
                    df['index'] = pd.Series(data=range(len(df)))

            if not str(canexp_df['row_id'].dtype).startswith('int'):
                canexp_df['row_id'] = canexp_df['row_id'].astype(int)

            if not str(keys_df['locid'].dtype).startswith('int'):
                keys_df['locid'] = keys_df['locid'].astype(int)

            oed_acc_col_repl = [{'accnumber': 'accntnum'}]
            for repl in oed_acc_col_repl:
                canexp_df.rename(columns=repl, inplace=True)

            merged_df = pd.merge(canexp_df, keys_df, left_on='row_id', right_on='locid').drop_duplicates()
            merged_df['index'] = pd.Series(data=range(len(merged_df)), dtype=object)

            cov_level_id = fm_levels[0]

            cov_tivs = tuple(t for t in [ufcp[cov_level_id][gid].get('tiv') for gid in ufcp[cov_level_id]] if t)

            if not cov_tivs:
                raise OasisException('No coverage fields found in the canonical exposures profile - please check the canonical exposures (loc) profile')

            fm_terms = {
                tiv_tgid: {
                    term_type: (
                        ufcp[cov_level_id][tiv_tgid][term_type]['ProfileElementName'].lower() if ufcp[cov_level_id][tiv_tgid].get(term_type) else None
                    ) for term_type in ('deductible', 'deductiblemin', 'deductiblemax', 'limit', 'share',)
                } for tiv_tgid in ufcp[cov_level_id]
            }


            group_id = 0
            prev_it_loc_id = -1
            item_id = 0
            zero_tiv_items = 0

            def positive_tiv_coverages(it): 
                return [t for t in cov_tivs if it.get(t['ProfileElementName'].lower()) and it[t['ProfileElementName'].lower()] > 0 and t['CoverageTypeID'] == it['coveragetypeid']] or [0]
            
            for it, ptiv in itertools.chain((it, ptiv) for _, it in merged_df.iterrows() for it, ptiv in itertools.product([it], positive_tiv_coverages(it))):
                if ptiv == 0:
                    zero_tiv_items += 1
                    continue

                item_id += 1
                if it['row_id'] != prev_it_loc_id:
                    group_id += 1

                tiv_elm = ptiv['ProfileElementName'].lower()
                tiv = it[tiv_elm]
                tiv_tgid = ptiv['FMTermGroupID']

                yield {
                    'item_id': item_id,
                    'canexp_id': it['row_id'] - 1,
                    'peril_id': it['perilid'],
                    'coverage_type_id': it['coveragetypeid'],
                    'coverage_id': item_id,
                    'is_bi_coverage': it['coveragetypeid'] in [OASIS_COVERAGE_TYPES['time']['id'], OED_COVERAGE_TYPES['bi']['id']],
                    'tiv_elm': tiv_elm,
                    'tiv': tiv,
                    'tiv_tgid': tiv_tgid,
                    'ded_elm': fm_terms[tiv_tgid].get('deductible'),
                    'ded_min_elm': fm_terms[tiv_tgid].get('deductiblemin'),
                    'ded_max_elm': fm_terms[tiv_tgid].get('deductiblemax'),
                    'lim_elm': fm_terms[tiv_tgid].get('limit'),
                    'shr_elm': fm_terms[tiv_tgid].get('share'),
                    'areaperil_id': it['areaperilid'],
                    'vulnerability_id': it['vulnerabilityid'],
                    'group_id': group_id,
                    'summary_id': 1,
                    'summaryset_id': 1
                }
                prev_it_loc_id = it['row_id']


        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
            raise OasisException(e)
        else:
            if zero_tiv_items == len(merged_df):
                raise OasisException('All canonical exposure items have zero TIVs - please check the canonical exposures (loc.) file')

    def generate_fm_items(
        self,
        canonical_exposures_df,
        gul_items_df,
        canonical_exposures_profile,
        canonical_accounts_profile,
        canonical_accounts_df,
        fm_agg_profile
    ):
        """
        Generates FM items.

        :param canonical_exposures_df: Canonical exposures
        :type canonical_exposures_df: pandas.DataFrame

        :param gul_items_df: GUL items
        :type gul_items_df: pandas.DataFrame

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_accounts_profile: Canonical accounts profile
        :type canonical_accounts_profile: dict

        :param canonical_accounts_df: Canonical accounts
        :param canonical_accounts_df: pandas.DataFrame

        :param fm_agg_profile: FM aggregation profile
        :param fm_agg_profile: dict
        """
        cep = canonical_exposures_profile
        cap = canonical_accounts_profile
        
        canexp_df = canonical_exposures_df

        canacc_df = canonical_accounts_df

        for df in [canexp_df, gul_items_df, canacc_df]:
            if not df.columns.contains('index'):
                df['index'] = pd.Series(data=range(len(df)))

        oed_acc_col_repl = [{'accnumber': 'accntnum'}, {'polnumber': 'policynum'}]
        for repl in oed_acc_col_repl:
                canacc_df.rename(columns=repl, inplace=True)

        cangul_df = pd.merge(canexp_df, gul_items_df, left_on='index', right_on='canexp_id')
        cangul_df['index'] = pd.Series(data=cangul_df.index)

        keys = (
            'item_id', 'gul_item_id', 'peril_id', 'coverage_type_id', 'coverage_id',
            'is_bi_coverage', 'canexp_id', 'canacc_id', 'policy_num', 'level_id', 'layer_id',
            'agg_id', 'policytc_id', 'deductible', 'deductible_min',
            'deductible_max', 'attachment', 'limit', 'share', 'calcrule_id', 'tiv_elm',
            'tiv', 'tiv_tgid', 'ded_elm', 'ded_min_elm', 'ded_max_elm',
            'lim_elm', 'shr_elm',
        )

        try:
            ufcp = unified_canonical_fm_profile_by_level_and_term_group(profiles=(cep, cap,))

            if not ufcp:
                raise OasisException(
                    'Canonical loc. and/or acc. profiles are possibly missing FM term information: '
                    'FM term definitions for TIV, limit, deductible and/or share.'
                )

            fmap = fm_agg_profile

            if not fmap:
                raise OasisException(
                    'FM aggregation profile is empty - this is required to perform aggregation'
                )

            fm_levels = tuple(ufcp.keys())

            cov_level_id = fm_levels[0]

            coverage_level_preset_data = list(zip(
                tuple(cangul_df['item_id'].values),          # 1 - FM item ID
                tuple(cangul_df['item_id'].values),          # 2 - GUL item ID
                tuple(cangul_df['peril_id'].values),         # 3 - peril ID
                tuple(cangul_df['coverage_type_id'].values), # 4 - coverage type ID
                tuple(cangul_df['coverage_id'].values),      # 5 - coverage ID
                tuple(cangul_df['is_bi_coverage'].values),   # 6 - is BI coverage?
                tuple(cangul_df['canexp_id'].values),        # 7 - can. exp. DF index
                (-1,)*len(cangul_df),                        # 8 - can. acc. DF index
                (-1,)*len(cangul_df),                        # 9 - can. acc. policy num.
                (cov_level_id,)*len(cangul_df),              # 10 - coverage level ID
                (1,)*len(cangul_df),                         # 11 - layer ID
                (-1,)*len(cangul_df),                        # 12 - agg. ID
                tuple(cangul_df['tiv_elm'].values),          # 13 - TIV element
                tuple(cangul_df['tiv'].values),              # 14 -TIV value
                tuple(cangul_df['tiv_tgid'].values),         # 15 -coverage element/term group ID
                tuple(cangul_df['ded_elm'].values),          # 16 -deductible element
                tuple(cangul_df['ded_min_elm'].values),      # 17 -deductible min. element
                tuple(cangul_df['ded_max_elm'].values),      # 18 -deductible max. element
                tuple(cangul_df['lim_elm'].values),          # 19 -limit element
                tuple(cangul_df['shr_elm'].values)           # 20 -share element
            ))

            def get_canacc_item(i):
                return canacc_df[(canacc_df['accntnum'] == cangul_df[cangul_df['canexp_id']==coverage_level_preset_data[i][6]].iloc[0]['accntnum'])].iloc[0]

            def get_canacc_id(i):
                return int(get_canacc_item(i)['index'])

            coverage_level_preset_items = {
                i: {
                    k:v for k, v in zip(
                        keys,
                        [i + 1, gul_item_id, peril_id, coverage_type_id, coverage_id, is_bi_coverage, canexp_id, get_canacc_id(i), policy_num, level_id, layer_id, agg_id, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12, tiv_elm, tiv, tiv_tgid, ded_elm, ded_min_elm, ded_max_elm, lim_elm, shr_elm]
                    )
                } for i, (item_id, gul_item_id, peril_id, coverage_type_id, coverage_id, is_bi_coverage, canexp_id, _, policy_num, level_id, layer_id, agg_id, tiv_elm, tiv, tiv_tgid, ded_elm, ded_min_elm, ded_max_elm, lim_elm, shr_elm) in enumerate(coverage_level_preset_data)
            }

            num_cov_items = len(coverage_level_preset_items)

            preset_items = {
                level_id: (coverage_level_preset_items if level_id == cov_level_id else copy.deepcopy(coverage_level_preset_items)) for level_id in fm_levels
            }

            for i, (level_id, item_id, it) in enumerate(itertools.chain((level_id, k, v) for level_id in fm_levels[1:] for k, v in preset_items[level_id].items())):
                it['level_id'] = level_id
                it['item_id'] = num_cov_items + i + 1
                it['ded_elm'] = it['ded_min_elm'] = it['ded_max_elm'] = it['lim_elm'] = it['shr_elm'] = None

            num_sub_layer_level_items = sum(len(preset_items[level_id]) for level_id in fm_levels[:-1])
            layer_level = max(fm_levels)
            layer_level_items = copy.deepcopy(preset_items[layer_level])
            layer_level_min_idx = min(layer_level_items)

            def layer_id(i):
                return list(
                    canacc_df[canacc_df['accntnum'] == canacc_df.iloc[i]['accntnum']]['policynum'].values
                ).index(canacc_df.iloc[i]['policynum']) + 1

            for i, (canexp_id, canacc_id) in enumerate(
                itertools.chain((canexp_id, canacc_id) for canexp_id in layer_level_items for canexp_id, canacc_id in itertools.product(
                    [canexp_id],
                    canacc_df[canacc_df['accntnum'] == canacc_df.iloc[layer_level_items[canexp_id]['canacc_id']]['accntnum']]['index'].values)
                )
            ):
                it = copy.deepcopy(layer_level_items[canexp_id])
                it['item_id'] = num_sub_layer_level_items + i + 1
                it['layer_id'] = layer_id(canacc_id)
                it['canacc_id'] = canacc_id
                preset_items[layer_level][layer_level_min_idx + i] = it

            for it in (it for c in itertools.chain(six.itervalues(preset_items[k]) for k in preset_items) for it in c):
                it['policy_num'] = canacc_df.iloc[it['canacc_id']]['policynum']
                lfmaggkey = fmap[it['level_id']]['FMAggKey']
                for v in six.itervalues(lfmaggkey):
                    src = v['src'].lower()
                    if src in ['canexp', 'canacc']:
                        f = v['field'].lower()
                        it[f] = canexp_df.iloc[it['canexp_id']][f] if src == 'canexp' else canacc_df.iloc[it['canacc_id']][f]

            oed = True if max(fm_levels) > 6 else False

            concurrent_tasks = (
                Task(get_fm_terms_by_level_as_list, args=(ufcp[level_id], fmap[level_id], preset_items[level_id], canexp_df.copy(deep=True), canacc_df.copy(deep=True), oed,), key=level_id)
                for level_id in fm_levels
            )
            num_ps = min(len(fm_levels), multiprocessing.cpu_count())
            for it in multiprocess(concurrent_tasks, pool_size=num_ps):
                yield it
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
            raise OasisException(e)

    def load_gul_items(self, canonical_exposures_profile, canonical_exposures_file_path, keys_file_path):
        """
        Loads GUL items generated by ``generate_gul_items`` into a static
        structure such as a pandas dataframe.

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_exposures_file_path: Canonical exposures file path
        :type canonical_exposures_file_path: str

        :param keys_file_path: Keys file path
        :type keys_file_path: str
        """
        cep = canonical_exposures_profile

        try:
            with io.open(canonical_exposures_file_path, 'r', encoding='utf-8') as cf, io.open(keys_file_path, 'r', encoding='utf-8') as kf:
                canexp_df, keys_df = pd.read_csv(cf, float_precision='high'), pd.read_csv(kf, float_precision='high')

            if len(canexp_df) == 0:
                raise OasisException('No canonical exposure items found - please check the canonical exposures (loc) file')

            if len(keys_df) == 0:
                raise OasisException('No keys items found - please check the model exposures (loc) file')

            canexp_df = canexp_df.where(canexp_df.notnull(), None)
            canexp_df.columns = canexp_df.columns.str.lower()
            canexp_df['index'] = pd.Series(data=canexp_df.index, dtype=int)

            keys_df = keys_df.where(keys_df.notnull(), None)
            keys_df.columns = keys_df.columns.str.lower()
            keys_df['index'] = pd.Series(data=keys_df.index, dtype=int)

            gul_items_df = pd.DataFrame(data=[it for it in self.generate_gul_items(cep, canexp_df, keys_df)], dtype=object)
            gul_items_df['index'] = pd.Series(data=gul_items_df.index, dtype=int)

            for col in gul_items_df.columns:
                if col.endswith('id'):
                    gul_items_df[col] = gul_items_df[col].astype(int)
                elif col == 'tiv':
                    gul_items_df[col] = gul_items_df[col].astype(float)
        except (IOError, MemoryError, OasisException, OSError, TypeError, ValueError) as e:
            raise OasisException(e)
            
        return gul_items_df, canexp_df


    def load_fm_items(
        self,
        canonical_exposures_df,
        gul_items_df,
        canonical_exposures_profile,
        canonical_accounts_profile,
        canonical_accounts_file_path,
        fm_agg_profile,
        reduced=True
    ):
        """
        Loads FM items generated by ``generate_fm_items`` into a static
        structure such as a pandas dataframe.

        :param canonical_exposures_df: Canonical exposures
        :type canonical_exposures_df: pandas.DataFrame

        :param gul_items_df: GUL items
        :type gul_items_df: pandas.DataFrame

        :param canonical_exposures_profile: Canonical exposures profile
        :type canonical_exposures_profile: dict

        :param canonical_accounts_profile: Canonical accounts profile
        :type canonical_accounts_profile: dict

        :param canonical_accounts_file_path: Canonical accounts file path
        :param canonical_accounts_file_path: str

        :param fm_agg_profile: FM aggregation profile
        :param fm_agg_profile: dict

        :param reduced: Whether to generate only FM items with not all zero
                        values for limit, deductible and share. By default ``True``
        :param reduced: bool
        """
        canexp_df = canonical_exposures_df

        cep = canonical_exposures_profile
        cap = canonical_accounts_profile

        fmap = fm_agg_profile

        try:
            with io.open(canonical_accounts_file_path, 'r', encoding='utf-8') as f:
                canacc_df = pd.read_csv(f, float_precision='high')

            if len(canacc_df) == 0:
                raise OasisException('No canonical accounts items')
            
            canacc_df = canacc_df.where(canacc_df.notnull(), None)
            canacc_df.columns = canacc_df.columns.str.lower()
            canacc_df['index'] = pd.Series(data=canacc_df.index, dtype=int)

            fm_items = [it for it in self.generate_fm_items(canexp_df, gul_items_df, cep, cap, canacc_df, fmap)]
            fm_items.sort(key=lambda it: it['item_id'])

            fm_items_df = pd.DataFrame(data=fm_items, dtype=object)
            fm_items_df['index'] = pd.Series(data=fm_items_df.index, dtype=int)

            if reduced:
                def is_zero_terms_level(level_id):
                    return not any(
                        it['deductible']!=0 or
                        it['deductible_min']!=0 or
                        it['deductible_max']!=0 or
                        it['limit']!=0 or
                        it['share']!=0
                        for _, it in fm_items_df[fm_items_df['level_id']==level_id].iterrows()
                    )

                fm_levels = list(set(fm_items_df['level_id']))
                non_zero_terms_levels = [lid for lid in fm_levels if lid in [fm_levels[0], fm_levels[-1]] or not is_zero_terms_level(lid)]

                fm_items_df = fm_items_df[(fm_items_df['level_id'].isin(non_zero_terms_levels))]

                fm_items_df['index'] = range(len(fm_items_df))

                fm_items_df['item_id'] = range(1, len(fm_items_df) + 1)

                level_ids = [l for l in set(fm_items_df['level_id'])]

                level_id = lambda i: level_ids.index(fm_items_df.iloc[i]['level_id']) + 1

                fm_items_df['level_id'] = fm_items_df['index'].apply(level_id)

            layer_level_id = fm_items_df['level_id'].max()

            policytc_ids = get_policytc_ids(fm_items_df)
            get_policytc_id = lambda i: [k for k in six.iterkeys(policytc_ids) if policytc_ids[k] == {k:fm_items_df.iloc[i][k] for k in ('limit', 'deductible', 'attachment', 'deductible_min', 'deductible_max', 'share', 'calcrule_id',)}][0]
            fm_items_df['policytc_id'] = fm_items_df['index'].apply(lambda i: get_policytc_id(i))

            for col in fm_items_df.columns:
                if col.endswith('id'):
                    fm_items_df[col] = fm_items_df[col].astype(int)
                elif col in ('tiv', 'limit', 'deductible', 'deductible_min', 'deductible_max', 'share',):
                    fm_items_df[col] = fm_items_df[col].astype(float)
        except (IOError, MemoryError, OasisException, OSError, TypeError, ValueError) as e:
            raise OasisException(e)

        return fm_items_df, canacc_df


    def write_items_file(self, gul_items_df, items_file_path):
        """
        Writes an items file.
        """
        try:
            gul_items_df.to_csv(
                columns=['item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id'],
                path_or_buf=items_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return items_file_path

    def write_coverages_file(self, gul_items_df, coverages_file_path):
        """
        Writes a coverages file.
        """
        try:
            gul_items_df.to_csv(
                columns=['coverage_id', 'tiv'],
                path_or_buf=coverages_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return coverages_file_path

    def write_gulsummaryxref_file(self, gul_items_df, gulsummaryxref_file_path):
        """
        Writes a gulsummaryxref file.
        """
        try:
            gul_items_df.to_csv(
                columns=['coverage_id', 'summary_id', 'summaryset_id'],
                path_or_buf=gulsummaryxref_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return gulsummaryxref_file_path

    def write_fm_policytc_file(self, fm_items_df, fm_policytc_file_path):
        """
        Writes an FM policy T & C file.
        """
        try:
            fm_policytc_df = pd.DataFrame(
                columns=['layer_id', 'level_id', 'agg_id', 'policytc_id'],
                data=[key[:4] for key, _ in fm_items_df.groupby(['layer_id', 'level_id', 'agg_id', 'policytc_id', 'limit', 'deductible', 'share'])],
                dtype=object
            )
            fm_policytc_df.to_csv(
                path_or_buf=fm_policytc_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return fm_policytc_file_path

    def write_fm_profile_file(self, fm_items_df, fm_profile_file_path):
        """
        Writes an FM profile file.
        """
        try:
            cols = ['policytc_id', 'calcrule_id', 'limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share']

            fm_profile_df = fm_items_df[cols]

            fm_profile_df = pd.DataFrame(
                columns=cols,
                data=[key for key, _ in fm_profile_df.groupby(cols)]
            )

            col_repl = [
                {'deductible': 'deductible1'},
                {'deductible_min': 'deductible2'},
                {'deductible_max': 'deductible3'},
                {'attachment': 'attachment1'},
                {'limit': 'limit1'},
                {'share': 'share1'}
            ]
            for repl in col_repl:
                fm_profile_df.rename(columns=repl, inplace=True)

            n = len(fm_profile_df)

            fm_profile_df['index'] = range(n)

            fm_profile_df['share2'] = fm_profile_df['share3'] = [0]*n

            fm_profile_df.to_csv(
                columns=['policytc_id','calcrule_id','deductible1', 'deductible2', 'deductible3', 'attachment1', 'limit1', 'share1', 'share2', 'share3'],
                path_or_buf=fm_profile_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return fm_profile_file_path

    def write_fm_programme_file(self, fm_items_df, fm_programme_file_path):
        """
        Writes a FM programme file.
        """
        try:
            cov_level = fm_items_df['level_id'].min()
            fm_programme_df = pd.DataFrame(
                pd.concat([fm_items_df[fm_items_df['level_id']==cov_level], fm_items_df])[['level_id', 'agg_id']],
                dtype=int
            ).reset_index(drop=True)

            num_cov_items = len(fm_items_df[fm_items_df['level_id']==cov_level])

            for i in range(num_cov_items):
                fm_programme_df.at[i, 'level_id'] = 0

            def from_agg_id_to_agg_id(from_level_id, to_level_id):
                iterator = (
                    (from_level_it, to_level_it)
                    for (_,from_level_it), (_, to_level_it) in zip(
                        fm_programme_df[fm_programme_df['level_id']==from_level_id].iterrows(),
                        fm_programme_df[fm_programme_df['level_id']==to_level_id].iterrows()
                    )
                )
                for from_level_it, to_level_it in iterator:
                    yield from_level_it['agg_id'], to_level_id, to_level_it['agg_id']

            levels = list(set(fm_programme_df['level_id']))

            data = [
                (from_agg_id, level_id, to_agg_id) for from_level_id, to_level_id in zip(levels, levels[1:]) for from_agg_id, level_id, to_agg_id in from_agg_id_to_agg_id(from_level_id, to_level_id)
            ]

            fm_programme_df = pd.DataFrame(columns=['from_agg_id', 'level_id', 'to_agg_id'], data=data, dtype=int).drop_duplicates()

            fm_programme_df.to_csv(
                path_or_buf=fm_programme_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return fm_programme_file_path

    def write_fm_xref_file(self, fm_items_df, fm_xref_file_path):
        """
        Writes a FM xref file.
        """
        try:
            data = [
                (i + 1, agg_id, layer_id) for i, (agg_id, layer_id) in enumerate(itertools.product(set(fm_items_df['agg_id']), set(fm_items_df['layer_id'])))
            ]

            fm_xref_df = pd.DataFrame(columns=['output', 'agg_id', 'layer_id'], data=data, dtype=int)

            fm_xref_df.to_csv(
                path_or_buf=fm_xref_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return fm_xref_file_path

    def write_fmsummaryxref_file(self, fm_items_df, fmsummaryxref_file_path):
        """
        Writes an FM summaryxref file.
        """
        try:
            data = [
                (i + 1, 1, 1) for i, _ in enumerate(itertools.product(set(fm_items_df['agg_id']), set(fm_items_df['layer_id'])))
            ]

            fmsummaryxref_df = pd.DataFrame(columns=['output', 'summary_id', 'summaryset_id'], data=data, dtype=int)

            fmsummaryxref_df.to_csv(
                path_or_buf=fmsummaryxref_file_path,
                encoding='utf-8',
                chunksize=1000,
                index=False
            )
        except (IOError, OSError) as e:
            raise OasisException(e)

        return fmsummaryxref_file_path

    def write_gul_files(self, oasis_model=None, **kwargs):
        """
        Writes the standard Oasis GUL files, namely::

            items.csv
            coverages.csv
            gulsummaryxref.csv
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        if oasis_model:
            omr = oasis_model.resources
            ofp = omr['oasis_files_pipeline']

        canonical_exposures_profile = kwargs.get('canonical_exposures_profile')
        canonical_exposures_file_path = kwargs.get('canonical_exposures_file_path')
        keys_file_path = kwargs.get('keys_file_path')
        
        gul_items_df, canexp_df = self.load_gul_items(canonical_exposures_profile, canonical_exposures_file_path, keys_file_path)

        if oasis_model:
            omr['canonical_exposures_df'] = canexp_df
            omr['gul_items_df'] = gul_items_df

        gul_files = (
            ofp.gul_files if oasis_model
            else {
                 'items': kwargs.get('items_file_path'),
                 'coverages': kwargs.get('coverages_file_path'),
                 'gulsummaryxref': kwargs.get('gulsummaryxref_file_path')
            }
        )

        concurrent_tasks = (
            Task(getattr(self, 'write_{}_file'.format(f)), args=(gul_items_df.copy(deep=True), gul_files[f],), key=f)
            for f in gul_files
        )
        num_ps = min(len(gul_files), multiprocessing.cpu_count())
        for _, _ in multithread(concurrent_tasks, pool_size=num_ps):
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
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        if oasis_model:
            omr = oasis_model.resources
            ofp = omr['oasis_files_pipeline']

            canexp_df, gul_items_df = omr['canonical_exposures_df'], omr['gul_items_df']
            canonical_exposures_profile = omr['canonical_exposures_profile']
            canonical_accounts_profile = omr['canonical_accounts_profile']
            canonical_accounts_file_path = ofp.canonical_accounts_file_path
            fm_agg_profile = omr['fm_agg_profile']
        else:
            canexp_df, gul_items_df = kwargs.get('canonical_exposures_df'), kwargs.get('gul_items_df')
            canonical_exposures_profile = kwargs.get('canonical_exposures_profile')
            canonical_accounts_profile = kwargs.get('canonical_accounts_profile')
            canonical_accounts_file_path = kwargs.get('canonical_accounts_file_path')
            fm_agg_profile = kwargs.get('fm_agg_profile')

        fm_items_df, canacc_df = self.load_fm_items(canexp_df, gul_items_df, canonical_exposures_profile, canonical_accounts_profile, canonical_accounts_file_path, fm_agg_profile)

        if oasis_model:
            omr['canonical_accounts_df'] = canacc_df
            omr['fm_items_df'] = fm_items_df

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
            Task(getattr(self, 'write_{}_file'.format(f)), args=(fm_items_df.copy(deep=True), fm_files[f],), key=f)
            for f in fm_files
        )
        num_ps = min(len(fm_files), multiprocessing.cpu_count())
        n = len(fm_files)
        for _, _ in multithread(concurrent_tasks, pool_size=num_ps):
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
        oasis_model.resources['oasis_files_pipeline'].clear()

        return oasis_model

    def start_oasis_files_pipeline(
        self,
        oasis_model=None,
        **kwargs
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

        :param kwargs: Keyword arguments
        :type kwargs: dict

        :return: A dictionary of Oasis files (GUL + FM (if FM option indicated))
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        if oasis_model:
            omr = oasis_model.resources
            ofp = omr.get('oasis_files_pipeline') or OasisFilesPipeline(model_key=oasis_model.key)

        logger = kwargs.get('logger') or logging.getLogger()

        logger.info('\nChecking Oasis files directory exists for model')
        oasis_files_path = kwargs.get('oasis_files_path')

        if not oasis_files_path:
            raise OasisException('No Oasis files directory provided')
        elif not os.path.exists(oasis_files_path):
            raise OasisException('Oasis files directory {} does not exist on the filesystem.'.format(oasis_files_path))
        
        logger.info('\nOasis files directory: {}'.format(oasis_files_path))

        logger.info('\nChecking for source exposures file')
        source_exposures_file_path = kwargs.get('source_exposures_file_path')
        
        if not source_exposures_file_path:
            raise OasisException('No source exposures file path provided')
        elif not os.path.exists(source_exposures_file_path):
            raise OasisException("Source exposures file path {} does not exist on the filesysem".format(source_exposures_file_path))

        logger.info('\nFound source exposures file {source_exposures_file_path} - copying to Oasis files directory'.format(**kwargs))
        shutil.copy2(source_exposures_file_path, oasis_files_path)

        logger.info('\nLoading canonical exposures profile')
        canonical_exposures_profile = kwargs.get('canonical_exposures_profile') or self.load_canonical_exposures_profile(oasis_model=oasis_model, **kwargs)

        if canonical_exposures_profile is None:
            raise OasisException('No canonical exposures profile provided')

        logger.info('\nFound canonical exposures profile: {}'.format(canonical_exposures_profile))

        fm = kwargs.get('fm')

        source_accounts_file_path = None
        canonical_accounts_profile = None
        fm_agg_profile = None

        if fm:
            logger.info('\nChecking for source accounts file')
            source_accounts_file_path = kwargs.get('source_accounts_file_path')
            
            if not source_accounts_file_path:
                raise OasisException('FM option indicated but no source accounts file path provided in arguments or model resources')
            elif not os.path.exists(source_accounts_file_path):
                raise OasisException("Source accounts file path {} does not exist on the filesysem.".format(source_accounts_file_path))

            logger.info('\nFound source accounts file {source_accounts_file_path} - copying to Oasis files directory'.format(**kwargs))
            shutil.copy2(source_accounts_file_path, oasis_files_path)

            logger.info('\nLoading canonical accounts profile')
            canonical_accounts_profile = kwargs.get('canonical_accounts_profile') or self.load_canonical_accounts_profile(oasis_model=oasis_model, **kwargs)

            if canonical_accounts_profile is None:
                raise OasisException('FM option indicated but no canonical accounts profile provided')

            logger.info('\nLoaded canonical accounts profile: {}'.format(canonical_accounts_profile))

            logger.info('\nLoading FM aggregation profile')
            fm_agg_profile = kwargs.get('fm_agg_profile') or self.load_fm_aggregation_profile(oasis_model=oasis_model, **kwargs)

            if fm_agg_profile is None:
                raise OasisException('FM option indicated but no FM aggregation profile provided')

            logger.info('\nLoaded FM aggregation profile: {}'.format(fm_agg_profile))

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
            ofp.canonical_accounts_file_path = canonical_accounts_file_path

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

    def create_model(self, model_supplier_id, model_id, model_version, resources=None):
        model = OasisModel(
            model_supplier_id,
            model_id,
            model_version,
            resources=resources
        )

        # set default resources
        omr = model.resources

        omr.setdefault('oasis_files_path', os.path.abspath(os.path.join('Files', model.key.replace('/', '-'))))
        if not os.path.isabs(omr['oasis_files_path']):
            omr['oasis_files_path'] = os.path.abspath(omr['oasis_files_path'])

        ofp = OasisFilesPipeline(model_key=model.key)
        omr['oasis_files_pipeline'] = ofp

        if omr.get('canonical_exposures_profile') is None:
            self.load_canonical_exposures_profile(oasis_model=model)

        if omr.get('canonical_accounts_profile') is None:
            self.load_canonical_accounts_profile(oasis_model=model)

        if omr.get('fm_agg_profile') is None:
            self.load_fm_aggregation_profile(oasis_model=model)

        if omr.get('lookup') and omr.get('lookup_config') is None:
            self.load_lookup_config(oasis_model=model)

        self.add_model(model)

        return model
