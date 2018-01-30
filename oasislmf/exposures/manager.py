#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import os
import io

import shutil
import json

import six

__all__ = [
    'OasisExposuresManagerInterface',
    'OasisExposuresManager'
]

from interface import Interface, implements
import pandas as pd

from ..keys.lookup import OasisKeysLookupFactory
from ..utils.exceptions import OasisException
from ..utils.mono import run_mono_executable
from ..utils.values import get_utctimestamp

__author__ = "Sandeep Murthy"
__copyright__ = "2017, Oasis Loss Modelling Framework"


class OasisExposuresManagerInterface(Interface):  # pragma: no cover
    """
    An interface for defining the behaviour of an Oasis exposures manager.
    """

    def __init__(self, oasis_models=None):
        """
        Class constructor.
        """
        pass

    def add_model(self, oasis_model):
        """
        Adds Oasis model object to the manager and sets up its resources.
        """
        pass

    def delete_model(self, oasis_model):
        """
        Deletes an existing Oasis model object in the manager.
        """
        pass

    def start_files_pipeline(self, oasis_model, **kwargs):
        """
        Starts the exposure transforms pipeline for the given ``oasis_model``,
        i.e. the generation of the canonical exposures files, keys file
        and finally the Oasis files.

        All the required resources must be provided either in the model object
        resources dict or the ``kwargs`` dict.

        It is up to the specific implementation of a manager of whether to
        use the model object resources dict or additional optional arguments
        in ``kwargs`` for this process.

        In a standard implementation of the manager the call to
        `start_files_pipeline` should trigger calls to the individual methods for
        performing file transformations in a normal sequence, e.g.

            `transform_source_to_canonical`
            `transform_canonical_to_model`
            `transform_model_to_keys`
            `load_canonical_profile`
            `generate_oasis_files`

        and the generated files should be stored as attributes in the given
        model object's transforms files pipeline.
        """
        pass

    def clear_files_pipeline(self, oasis_model, **kwargs):
        """
        Clears the exposure transforms files pipeline for the given
        ``oasis_model`` optionally using additional arguments in the ``kwargs``
        dict.

        All the required resources must be provided either in the model object
        resources dict or the ``kwargs`` dict.

        In the design of the exposure transform framework a model's files
        pipeline is an object value in its resources dict with the key
        ``transforms_files_pipeline`` and is thereby accessible with

            `oasis_model.resources['transforms_files_pipeline']`

        This returns an object of type

            `exposure_transforms.OasisExposureTransformsFilesPipeline`

        which stores the different files in the transformation stages for the
        model as property attributes, e.g.

            `oasis_model.resources['transforms_files_pipeline'].source_exposures_file`

        A standard implementation could either assign a new object of this
        type in the call to ``clear_files_pipeline``, or set some subset of the
        file attributes of this pipelines object to null.
        """
        pass

    def transform_source_to_canonical(self, oasis_model, **kwargs):
        """
        Transforms the source exposures/locations file for a given
        ``oasis_model`` object to a canonical/standard Oasis format.

        All the required resources must be provided either in the model object
        resources dict or the ``kwargs`` dict.

        It is up to the specific implementation of this class of how these
        resources will be named in ``kwargs`` and how they will be used to
        effect the transformation.

        The transform is generic by default, but could be supplier specific if
        required.
        """
        pass

    def transform_canonical_to_model(self, oasis_model, **kwargs):
        """
        Transforms the canonical exposures/locations file for a given
        ``oasis_model`` object to a format understood by Oasis keys lookup
        services.

        All the required resources must be provided either in the model object
        resources dict or the ``kwargs`` dict.

        It is up to the specific implementation of this class of how these
        resources will be named in ``kwargs`` and how they will be used to
        effect the transformation.
        """
        pass

    @classmethod
    def get_keys(cls, oasis_model=None, **kwargs):
        """
        Generates the Oasis keys CSV file for a given model object, with
        headers

            ``LocID,PerilID,CoverageID,AreaPerilID,VulnerabilityID``


        All the required resources must be provided either in the model object
        resources dict or the ``kwargs`` dict.

        It is up to the specific implementation of this class of how these
        resources will be named in ``kwargs`` and how they will be used to
        effect the transformation.

        A "standard" implementation should use the lookup service factory
        class in ``oasis_utils`` (a submodule of `omdk`) namely

            ``oasis_utils.oasis_keys_lookup_service_utils.KeysLookupServiceFactory``
        """
        pass

    @classmethod
    def load_canonical_profile(cls, oasis_model=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the canonical
        exposures profile for a given ``oasis_model``, stores this in the
        model object's resources dict, and returns the object.
        """
        pass

    @classmethod
    def generate_items_file(cls, oasis_model=None, **kwargs):
        """
        Generates an items file for the given ``oasis_model``.
        """
        pass

    @classmethod
    def generate_coverages_file(cls, oasis_model=None, **kwargs):
        """
        Generates a coverages file for the given ``oasis_model``.
        """
        pass

    @classmethod
    def generate_gulsummaryxref_file(cls, oasis_model=None, **kwargs):
        """
        Generates a gulsummaryxref file for the given ``oasis_model``.
        """
        pass

    @classmethod
    def generate_oasis_files(cls, oasis_model=None, **kwargs):
        """
        For a given ``oasis_model`` generates the standard Oasis files, namely

            ``items.csv``
            ``coverages.csv``
            ``gulsummaryxref.csv``

        All the required resources must be provided either in the model object
        resources dict or the ``kwargs`` dict.

        It is up to the specific implementation of this class of how these
        resources will be named in ``kwargs`` and how they will be used to
        effect the transformation.
        """
        pass


class OasisExposuresManager(implements(OasisExposuresManagerInterface)):
    def __init__(self, oasis_models=None):
        """
        Class constructor.

        :param oasis_models: A list of Oasis model objects with resources provided in the model objects'
            resources dictionaries.
        :type oasis_models: ``list(OasisModel)``
        """
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

    def transform_source_to_canonical(self, oasis_model, with_model_resources=True, **kwargs):
        """
        Transforms the source exposures/locations file for a given
        ``oasis_model`` object to a canonical/standard Oasis format.

        By default it is assumed that all the resources required for the
        transformation are present in the model object's resources dict,
        specifically its transforms files pipeline  - this is indicated by the
        optional ``with_model_resources`` variable which is ``True`` by
        default. In this case the generated file is stored in the appropriate
        attribute of the model object's oasis files pipeline, which is in
        turn a key in the manager's models dict. The models dict is updated,
        and the model object returned.

        If not then ``with_model_resources`` should be set to ``False``, in
        which case all the resources required for the transformation should be
        present in the optional ``kwargs`` dict as named arguments. In this
        case only the generated canonical exposures file is returned.
        """
        omr = oasis_model.resources
        tfp = omr['oasis_files_pipeline']

        if not with_model_resources:
            xtrans_path = kwargs['xtrans_path']
            input_file_path = kwargs['source_exposures_file_path']
            validation_file_path = kwargs['source_exposures_validation_file_path']
            transformation_file_path = kwargs['source_to_canonical_exposures_transformation_file_path']
            output_file_path = kwargs['canonical_exposures_file_path']
        else:
            xtrans_path = omr['xtrans_path']
            input_file_path = tfp.source_exposures_file.name
            validation_file_path = omr['source_exposures_validation_file_path']
            transformation_file_path = omr['source_to_canonical_exposures_transformation_file_path']
            output_file_path = tfp.canonical_exposures_file.name

        (
            xtrans_path,
            input_file_path,
            validation_file_path,
            transformation_file_path,
            output_file_path
        ) = map(
            os.path.abspath,
            [
                xtrans_path,
                input_file_path,
                validation_file_path,
                transformation_file_path,
                output_file_path
            ]
        )

        xtrans_args = {
            'd': validation_file_path,
            'c': input_file_path,
            't': transformation_file_path,
            'o': output_file_path,
            's': ''
        }

        try:
            run_mono_executable(xtrans_path, xtrans_args)
        except OasisException as e:
            raise e

        with io.open(output_file_path, 'r', encoding='utf-8') as f:
            if not with_model_resources:
                return f

            tfp.canonical_exposures_file = f

        return oasis_model

    def transform_canonical_to_model(self, oasis_model, with_model_resources=True, **kwargs):
        """
        Transforms the canonical exposures/locations file for a given
        ``oasis_model`` object to a format understood by Oasis keys lookup
        services.

        By default it is assumed that all the resources required for the
        transformation are present in the model object's resources dict,
        specifically its transforms files pipeline  - this is indicated by the
        optional ``with_model_resources`` variable which is ``True`` by
        default. In this case the generated file is stored in the appropriate
        attribute of the model object's transforms files pipeline, the
        manager's model dict is updated, and the model object returned.

        If not then ``with_model_resources`` should be set to ``False``, in
        which case all the resources required for the transformation should be
        present in the optional ``kwargs`` dict as named arguments. In this
        case only the generated canonical file is returned.
        """
        omr = oasis_model.resources
        tfp = omr['oasis_files_pipeline']

        if not with_model_resources:
            xtrans_path = kwargs['xtrans_path']
            input_file_path = kwargs['canonical_exposures_file_path']
            validation_file_path = kwargs['canonical_exposures_validation_file_path']
            transformation_file_path = kwargs['canonical_to_model_exposures_transformation_file_path']
            output_file_path = kwargs['model_exposures_file_path']
        else:
            xtrans_path = omr['xtrans_path']
            input_file_path = tfp.canonical_exposures_file.name
            validation_file_path = omr['canonical_exposures_validation_file_path']
            transformation_file_path = omr['canonical_to_model_exposures_transformation_file_path']
            output_file_path = tfp.model_exposures_file.name

        (
            xtrans_path,
            input_file_path,
            validation_file_path,
            transformation_file_path,
            output_file_path
        ) = map(
            os.path.abspath,
            [
                xtrans_path,
                input_file_path,
                validation_file_path,
                transformation_file_path,
                output_file_path
            ]
        )

        xtrans_args = {
            'd': validation_file_path,
            'c': input_file_path,
            't': transformation_file_path,
            'o': output_file_path
        }

        try:
            run_mono_executable(xtrans_path, xtrans_args)
        except OasisException as e:
            raise e

        with io.open(output_file_path, 'r', encoding='utf-8') as f:
            if not with_model_resources:
                return f

            tfp.model_exposures_file = f

        return oasis_model

    @classmethod
    def load_canonical_profile(
            cls,
            oasis_model=None,
            canonical_exposures_profile_json=None,
            canonical_exposures_profile_json_path=None,
            **kwargs):
        """
        Loads a JSON string or JSON file representation of the canonical
        exposures profile for a given ``oasis_model``, stores this in the
        model object's resources dict, and returns the object.
        """
        if oasis_model:
            canonical_exposures_profile_json = canonical_exposures_profile_json or oasis_model.resources.get(
                'canonical_exposures_profile_json')
            canonical_exposures_profile_json_path = canonical_exposures_profile_json_path or oasis_model.resources.get(
                'canonical_exposures_profile_json_path')

        profile = {}
        if canonical_exposures_profile_json:
            profile = json.loads(canonical_exposures_profile_json)
        elif canonical_exposures_profile_json_path:
            with io.open(canonical_exposures_profile_json_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)

        if oasis_model:
            oasis_model.resources['canonical_exposures_profile'] = profile

        return profile

    @classmethod
    def get_keys(cls, oasis_model=None, model_exposures_file_path=None, lookup=None, keys_file_path=None, **kwargs):
        """
        Generates the Oasis keys CSV file for a given model object, with
        headers

            ``LocID,PerilID,CoverageID,AreaPerilID,VulnerabilityID``

        using the lookup service defined for this model.

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

        :param lookup: Path to the keys lookup service to use, required if ``oasis_model`` is ``None``
        :type lookup: str

        :param model_exposures_file_path: Path to the exposures file, required if ``oasis_model`` is ``None``
        :type model_exposures_file_path: str

        :return: The path to the generated keys file
        """
        if oasis_model:
            model_exposures_file_path = model_exposures_file_path or oasis_model.files_pipeline.model_exposures_path
            lookup = lookup or oasis_model.resources.get('lookup')
            keys_file_path = keys_file_path or oasis_model.files_pipeline.keys_file_path

        model_exposures_file_path = os.path.abspath(model_exposures_file_path)
        keys_file_path = os.path.abspath(keys_file_path)

        oasis_keys_path, _ = OasisKeysLookupFactory().save_keys(
            lookup=lookup,
            model_exposures_file_path=model_exposures_file_path,
            output_file_path=keys_file_path,
        )

        if oasis_model:
            oasis_model.files_pipeline.keys_file_path = oasis_keys_path

        return oasis_keys_path

    @classmethod
    def _process_default_kwargs(cls, oasis_model=None, **kwargs):
        if oasis_model:
            kwargs.setdefault('canonical_exposures_file_path', oasis_model.files_pipeline.canonical_exposures_path)
            kwargs.setdefault('keys_file_path', oasis_model.files_pipeline.keys_file_path)
            kwargs.setdefault('canonical_exposures_profile', oasis_model.resources.get('canonical_exposures_profile'))
            kwargs.setdefault('canonical_exposures_profile_json', oasis_model.resources.get('canonical_exposures_profile_json'))
            kwargs.setdefault('canonical_exposures_profile_json_path', oasis_model.resources.get('canonical_exposures_profile_json_path'))
            kwargs.setdefault('items_file_path', oasis_model.files_pipeline.items_file_path)
            kwargs.setdefault('items_timestamped_file_path', oasis_model.resources.get('items_timestamped_file_path'))
            kwargs.setdefault('coverages_file_path', oasis_model.resources.get('coverages_file_path'))
            kwargs.setdefault('coverages_timestamped_file_path', oasis_model.resources.get('coverages_timestamped_file_path'))
            kwargs.setdefault('gulsummaryxref_file_path', oasis_model.resources.get('gulsummaryxref_file_path'))
            kwargs.setdefault('gulsummaryxref_timestamped_file_path', oasis_model.resources.get('gulsummaryxref_timestamped_file_path'))

        if not kwargs.get('canonical_exposures_profile'):
            kwargs['canonical_exposures_profile'] = cls.load_canonical_profile(
                oasis_model=oasis_model,
                canonical_exposures_profile_json=kwargs.get('canonical_exposures_profile_json'),
                canonical_exposures_profile_json_path=kwargs.get('canonical_exposures_profile_json_path'),
            )

        return kwargs

    @classmethod
    def load_item_records(cls, canonical_exposures_file_path, keys_file_path, canonical_exposures_profile, **kwargs):
        with io.open(canonical_exposures_file_path, 'r', encoding='utf-8') as cf:
            canexp_df = pd.read_csv(cf)
            canexp_df = canexp_df.where(canexp_df.notnull(), None)
            canexp_df.columns = map(str.lower, canexp_df.columns)

        with io.open(keys_file_path, 'r', encoding='utf-8') as kf:
            keys_df = pd.read_csv(kf)
            keys_df = keys_df.rename(columns={'CoverageID': 'CoverageType'})
            keys_df = keys_df.where(keys_df.notnull(), None)
            keys_df.columns = map(str.lower, keys_df.columns)

        tiv_fields = sorted(
            filter(lambda v: v.get('FieldName') == 'TIV', six.itervalues(canonical_exposures_profile))
        )

        ii = 0
        for i in range(len(keys_df)):
            ki = keys_df.iloc[i]

            ci_df = canexp_df[canexp_df['row_id'] == ki['locid']]

            if ci_df.empty:
                raise OasisException(
                    "No matching canonical exposure item found in canonical exposures data frame for keys item {}.".format(ki)
                )
            elif len(ci_df) > 1:
                raise OasisException(
                    "Duplicate canonical exposure items found in canonical exposures data frame for keys item {}.".format(ki)
                )

            ci = ci_df.iloc[0]

            tiv_field = next(f for f in tiv_fields if f['CoverageTypeID'] == ki['coveragetype'])

            if ci[tiv_field['ProfileElementName'].lower()] > 0:
                ii += 1
                yield ii, ki, ci[tiv_field['ProfileElementName'].lower()]

    @classmethod
    def generate_items_file(cls, oasis_model=None, **kwargs):
        """
        Generates an items file for the given ``oasis_model``.
        """
        kwargs = cls._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        columns = ['item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id']
        items_df = pd.DataFrame(columns=columns)

        items_df = items_df.append([
            {
                'item_id': item_id,
                'coverage_id': item_id,
                'areaperil_id': item['areaperilid'],
                'vulnerability_id': item['vulnerabilityid'],
                'group_id': item_id
            } for item_id, item, tiv in cls.load_item_records(**kwargs)
        ])
        items_df = items_df.astype(int)

        items_df.to_csv(
            columns=columns,
            path_or_buf=kwargs['items_file_path'],
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
        items_df.to_csv(
            columns=columns,
            path_or_buf=kwargs['items_timestamped_file_path'],
            encoding='utf-8',
            chunksize=1000,
            index=False
        )

        if oasis_model:
            oasis_model.files_pipeline.items_file_path = kwargs['items_file_path']

        return kwargs['items_file_path']

    @classmethod
    def generate_coverages_file(cls, oasis_model=None, **kwargs):
        """
        Generates a coverages file for the given ``oasis_model``.
        """
        kwargs = cls._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        columns = ['coverage_id', 'tiv']
        coverages_df = pd.DataFrame(columns=columns)

        coverages_df = coverages_df.append([
            {
                'coverage_id': item_id,
                'tiv': item_tiv,
            } for item_id, item, item_tiv in cls.load_item_records(**kwargs)
        ])

        coverages_df = coverages_df.astype(int)

        coverages_df.to_csv(
            columns=columns,
            path_or_buf=kwargs.get('coverages_file_path'),
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
        coverages_df.to_csv(
            columns=columns,
            path_or_buf=kwargs.get('coverages_timestamped_file_path'),
            encoding='utf-8',
            chunksize=1000,
            index=False
        )

        if oasis_model:
            oasis_model.files_pipeline.coverages_file = kwargs.get('coverages_file_path')

        return kwargs.get('coverages_file_path')

    @classmethod
    def generate_gulsummaryxref_file(cls, oasis_model=None, **kwargs):
        """
        Generates a gulsummaryxref file for the given ``oasis_model``.
        """
        kwargs = cls._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        columns = ['coverage_id', 'summary_id', 'summaryset_id']
        gulsummaryxref_df = pd.DataFrame(columns=columns)
        gulsummaryxref_df = gulsummaryxref_df.append([
            {
                'coverage_id': item_id,
                'summary_id': 1,
                'summaryset_id': 1
            } for item_id, item, item_tiv in cls.load_item_records(**kwargs)
        ])

        gulsummaryxref_df = gulsummaryxref_df.astype(int)

        gulsummaryxref_df.to_csv(
            columns=columns,
            path_or_buf=kwargs.get('gulsummaryxref_file_path'),
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
        gulsummaryxref_df.to_csv(
            columns=columns,
            path_or_buf=kwargs.get('gulsummaryxref_timestamped_file_path'),
            encoding='utf-8',
            chunksize=1000,
            index=False
        )

        if oasis_model:
            oasis_model.files_pipeline.gulsummaryxref_path = kwargs.get('gulsummaryxref_timestamped_file_path')

        return kwargs.get('gulsummaryxref_timestamped_file_path')

    @classmethod
    def generate_oasis_files(cls, oasis_model=None, **kwargs):
        """
        For a given ``oasis_model`` generates the standard Oasis files, namely

            ``items.csv``
            ``coverages.csv``
            ``gulsummaryxref.csv``

        """
        kwargs = cls._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        columns = [
            'item_id',
            'coverage_id',
            'tiv',
            'areaperil_id',
            'vulnerability_id',
            'group_id',
            'summary_id',
            'summaryset_id'
        ]
        master_df = pd.DataFrame(columns=columns)
        master_df = master_df.append(
            {
                'item_id': item_id,
                'coverage_id': item_id,
                'tiv': item_tiv,
                'areaperil_id': item['areaperilid'],
                'vulnerability_id': item['vulnerabilityid'],
                'group_id': item_id,
                'summary_id': 1,
                'summaryset_id': 1
            } for item_id, item, item_tiv in cls.load_item_records(
                kwargs.get('canonical_exposures_file_path'),
                kwargs.get('keys_file_path'),
                kwargs.get('canonical_exposures_profile')
            )
        )
        columns = ['item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id']
        for col in columns:
            master_df[col] = master_df[col].astype(int)

        master_df.to_csv(
            columns=columns,
            path_or_buf=kwargs.get('items_file_path'),
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
        master_df.to_csv(
            columns=columns,
            path_or_buf=kwargs.get('items_timestamped_file_path'),
            encoding='utf-8',
            chunksize=1000,
            index=False
        )

        columns = ['coverage_id', 'tiv']
        master_df['coverage_id'] = master_df['coverage_id'].astype(int)

        master_df.to_csv(
            columns=columns,
            float_format='%.5f',
            path_or_buf=kwargs['coverages_file_path'],
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
        master_df.to_csv(
            columns=columns,
            float_format='%.5f',
            path_or_buf=kwargs['coverages_timestamped_file_path'],
            encoding='utf-8',
            chunksize=1000,
            index=False
        )

        columns = ['coverage_id', 'summary_id', 'summaryset_id']
        for col in columns:
            master_df[col] = master_df[col].astype(int)

        master_df.to_csv(
            columns=columns,
            path_or_buf=kwargs['gulsummaryxref_file_path'],
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
        master_df.to_csv(
            columns=columns,
            path_or_buf=kwargs['gulsummaryxref_timestamped_file_path'],
            encoding='utf-8',
            chunksize=1000,
            index=False
        )

        if oasis_model:
            oasis_model.files_pipeline.items_file_path = kwargs['items_file_path']
            oasis_model.files_pipeline.coverages_path = kwargs['coverages_file_path']
            oasis_model.files_pipeline.gulsummaryxref_path = kwargs['gulsummaryxref_file_path']

        return kwargs['items_file_path'], kwargs['coverages_file_path'], kwargs['gulsummaryxref_file_path']

    def clear_files_pipeline(self, oasis_model, **kwargs):
        """
        Clears the oasis files pipeline for the given Oasis model object.

        Args:
            ``oasis_model`` (``omdk.models.OasisModel.OasisModel``): The model object.

            ``**kwargs`` (arbitary keyword arguments):

        Returns:
            ``oasis_model`` (``omdk.models.OasisModel.OasisModel``): The model object with its
            Oasis files pipeline cleared.
        """
        oasis_model.resources['oasis_files_pipeline'].clear()

        return oasis_model

    def start_files_pipeline(self, oasis_model, with_model_resources=True, **kwargs):
        """
        Starts the oasis files pipeline for the given Oasis model object,
        which is the generation of the Oasis items, coverages and GUL summary
        files from the source exposures file, canonical exposures profile,
        and associated validation files and transformation files for the
        source and intermediate files (canonical exposures, model exposures).

        Args:
            ``oasis_model`` (``omdk.models.OasisModel.OasisModel``): The model object.

            ``with_model_resources``: (``bool``): Indicates whether to look for
            resources in the model object's resources dictionary or in ``**kwargs``.

            ``**kwargs`` (arbitrary keyword arguments): If ``with_model_resources``
            is ``False`` then the filesystem paths (including filename and extension)
            of the canonical exposures file, the model exposures
        """
        try:
            if not with_model_resources:
                self.logger.info('Checking output files directory exists for model')

                oasis_files_path = kwargs['oasis_files_path'] if 'oasis_files_path' in kwargs else None

                if not oasis_files_path:
                    raise OasisException(
                        "No output directory provided for {} in '**kwargs'.".format(oasis_model)
                    )
                elif not os.path.exists(oasis_files_path):
                    raise OasisException(
                        "Output directory {} for {} provided in '**kwargs' "
                        "does not exist on the filesystem.".format(oasis_files_path, oasis_model)
                    )

                self.logger.info('Output files directory {} exists'.format(oasis_files_path))

                self.logger.info('Checking for source exposures file')

                source_exposures_file_path = kwargs['source_exposures_file_path']

                if not source_exposures_file_path:
                    raise OasisException(
                        "No source exposures file path provided for {} in '**kwargs'.".format(oasis_model))
                elif not os.path.exists(source_exposures_file_path):
                    raise OasisException(
                        "Source exposures file path {} provided for {} in '**kwargs' is invalid.".format(
                            source_exposures_file_path, oasis_model))

                self.logger.info('Source exposures file {} exists'.format(source_exposures_file_path))

                self.logger.info('Copying source exposures file to model output files directory')

                source_exposures_file_name = source_exposures_file_path.split(os.path.sep)[-1]
                target_file_path = os.path.join(oasis_files_path, source_exposures_file_name)

                if not os.path.exists(target_file_path):
                    shutil.copy(source_exposures_file_path, target_file_path)

                kwargs['source_exposures_file_path'] = target_file_path

                utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')

                target_file_path = os.path.join(oasis_files_path, 'canexp-{}.csv'.format(utcnow))
                kwargs['canonical_exposures_file_path'] = target_file_path

                self.logger.info('Generating canonical exposures file {}'.format(target_file_path))

                self.transform_source_to_canonical(oasis_model, with_model_resources=False, **kwargs)

                target_file_path = os.path.join(oasis_files_path, 'modexp-{}.csv'.format(utcnow))
                kwargs['model_exposures_file_path'] = target_file_path

                self.logger.info('Generating model exposures file {}'.format(target_file_path))

                self.transform_canonical_to_model(oasis_model, with_model_resources=False, **kwargs)

                target_file_path = os.path.join(oasis_files_path, 'oasiskeys-{}.csv'.format(utcnow))
                kwargs['keys_file_path'] = target_file_path

                self.logger.info('Generating keys file {}'.format(target_file_path))

                self.get_keys(oasis_model, with_model_resources=False, **kwargs)

                self.logger.info('Checking for canonical exposures profile source for model')

                canonical_exposures_profile_json_path = kwargs['canonical_exposures_profile_json_path']

                if not canonical_exposures_profile_json_path:
                    raise OasisException(
                        "No canonical exposures profile JSON file path provided for {} in '**kwargs'.".format(
                            oasis_model))
                elif not os.path.exists(canonical_exposures_profile_json_path):
                    raise OasisException(
                        "Canonical exposures profile JSON file path {} provided for {} in '**kwargs' is invalid.".format(
                            canonical_exposures_profile_json_path, oasis_model))

                self.logger.info(
                    'Canonical exposures profile source {} exists'.format(canonical_exposures_profile_json_path))

                self.logger.info('Loading canonical exposures profile into model resources')
                canonical_exposures_profile = self.load_canonical_profile(oasis_model, with_model_resources=False,
                                                                          **kwargs)
                kwargs['canonical_exposures_profile'] = canonical_exposures_profile
                self.logger.info(
                    'Loaded canonical exposures profile {} into model resources'.format(canonical_exposures_profile))

                kwargs['items_file_path'] = os.path.join(oasis_files_path, 'items.csv')
                kwargs['items_timestamped_file_path'] = os.path.join(oasis_files_path, 'items-{}.csv'.format(utcnow))
                kwargs['coverages_file_path'] = os.path.join(oasis_files_path, 'coverages.csv')
                kwargs['coverages_timestamped_file_path'] = os.path.join(oasis_files_path,
                                                                         'coverages-{}.csv'.format(utcnow))
                kwargs['gulsummaryxref_file_path'] = os.path.join(oasis_files_path, 'gulsummaryxref.csv')
                kwargs['gulsummaryxref_timestamped_file_path'] = os.path.join(oasis_files_path,
                                                                              'gulsummaryxref-{}.csv'.format(utcnow))

                self.logger.info('Generating Oasis files for model')
                items_file, coverages_file, gulsummaryxref_file = self.generate_oasis_files(
                    oasis_model, with_model_resources=False, **kwargs
                )

                return {
                    'items': items_file,
                    'coverages': coverages_file,
                    'gulsummaryxref': gulsummaryxref_file
                }
            else:
                omr = oasis_model.resources
                tfp = omr['oasis_files_pipeline']

                self.logger.info('Checking output files directory exists for model')

                oasis_files_path = omr['oasis_files_path'] if 'oasis_files_path' in omr else None
                if not oasis_files_path:
                    raise OasisException(
                        "No output directory provided for {} in resources dict.".format(oasis_model)
                    )
                elif not os.path.exists(oasis_files_path):
                    raise OasisException(
                        "Output directory {} for {} provided in resources dict "
                        "does not exist on the filesystem.".format(oasis_files_path, oasis_model)
                    )

                self.logger.info('Output files directory {} exists'.format(oasis_files_path))

                self.logger.info('Checking for source exposures file in the model files pipeline')

                source_exposures_file = tfp.source_exposures_file if tfp.source_exposures_file else None

                if not source_exposures_file:
                    raise OasisException(
                        "No source exposures file in the Oasis files pipeline for {}.".format(oasis_model))
                elif not os.path.exists(source_exposures_file.name):
                    raise OasisException(
                        "Source exposures file path {} provided for {} is invalid.".format(source_exposures_file.name,
                                                                                           oasis_model))

                self.logger.info('Source exposures file {} exists'.format(source_exposures_file))

                utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')

                target_file_path = os.path.join(oasis_files_path, 'canexp-{}.csv'.format(utcnow))
                with io.open(target_file_path, 'w', encoding='utf-8') as f:
                    tfp.canonical_exposures_file = f

                self.logger.info('Generating canonical exposures file')

                self.transform_source_to_canonical(oasis_model)

                self.logger.info('Generated canonical exposures file {}'.format(target_file_path))

                target_file_path = os.path.join(oasis_files_path, 'modexp-{}.csv'.format(utcnow))
                with io.open(target_file_path, 'w', encoding='utf-8') as f:
                    tfp.model_exposures_file = f

                self.logger.info('Generating model exposures file')

                self.transform_canonical_to_model(oasis_model)

                self.logger.info('Generated model exposures file {}'.format(target_file_path))

                target_file_path = os.path.join(oasis_files_path, 'oasiskeys-{}.csv'.format(utcnow))
                with io.open(target_file_path, 'w', encoding='utf-8') as f:
                    tfp.keys_file = f

                self.logger.info('Generating keys file')

                self.get_keys(oasis_model)

                self.logger.info('Generated keys file {}'.format(target_file_path))

                self.logger.info('Checking for canonical exposures profile for model')

                if not oasis_model.resources['canonical_exposures_profile']:
                    self.logger.info(
                        'Canonical exposures profile not found in model resources - attempting to load from source')
                    self.load_canonical_profile(oasis_model)
                    self.logger.info('Loaded canonical exposures profile {} into model resources'.format(
                        oasis_model.resources['canonical_exposures_profile']))
                else:
                    self.logger.info('Canonical exposures profile exists for model')

                omr['items_file_path'] = os.path.join(oasis_files_path, 'items.csv')
                omr['items_timestamped_file_path'] = os.path.join(oasis_files_path, 'items-{}.csv'.format(utcnow))
                omr['coverages_file_path'] = os.path.join(oasis_files_path, 'coverages.csv')
                omr['coverages_timestamped_file_path'] = os.path.join(oasis_files_path,
                                                                      'coverages-{}.csv'.format(utcnow))
                omr['gulsummaryxref_file_path'] = os.path.join(oasis_files_path, 'gulsummaryxref.csv')
                omr['gulsummaryxref_timestamped_file_path'] = os.path.join(oasis_files_path,
                                                                           'gulsummaryxref-{}.csv'.format(utcnow))

                self.logger.info('Generating Oasis files for model')

                self.generate_oasis_files(oasis_model)

                self.logger.info('Generated Oasis files {}'.format(tfp.oasis_files))
        except OasisException as e:
            raise e

        return oasis_model
