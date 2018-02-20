# -*- coding: utf-8 -*-

__all__ = [
    'OasisExposuresManagerInterface',
    'OasisExposuresManager'
]

import io
import json
import logging
import os
import shutil

import pandas as pd
import six

from interface import Interface, implements

from ..keys.lookup import OasisKeysLookupFactory
from ..utils.exceptions import OasisException
from ..utils.values import get_utctimestamp


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

    def transform_source_to_canonical(self, oasis_model, **kwargs):
        """
        Transforms the source exposures/locations for a given ``oasis_model``
        object to a canonical/standard Oasis format.

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
        Transforms the canonical exposures/locations for a given ``oasis_model``
        object to a format understood by Oasis keys lookup
        services.

        All the required resources must be provided either in the model object
        resources dict or the ``kwargs`` dict.

        It is up to the specific implementation of this class of how these
        resources will be named in ``kwargs`` and how they will be used to
        effect the transformation.
        """
        pass

    def get_keys(self, oasis_model=None, **kwargs):
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

    def load_canonical_profile(self, oasis_model=None, **kwargs):
        """
        Loads a JSON string or JSON file representation of the canonical
        exposures profile for a given ``oasis_model``, stores this in the
        model object's resources dict, and returns the object.
        """
        pass

    def generate_items_file(self, oasis_model=None, **kwargs):
        """
        Generates an items file for the given ``oasis_model``.
        """
        pass

    def generate_coverages_file(self, oasis_model=None, **kwargs):
        """
        Generates a coverages file for the given ``oasis_model``.
        """
        pass

    def generate_gulsummaryxref_file(self, oasis_model=None, **kwargs):
        """
        Generates a gulsummaryxref file for the given ``oasis_model``.
        """
        pass

    def generate_oasis_files(self, oasis_model=None, **kwargs):
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

        :todo: replace this with the python version
        """
        # omr = oasis_model.resources
        # tfp = omr['oasis_files_pipeline']
        #
        # if not with_model_resources:
        #     xtrans_path = kwargs['xtrans_path']
        #     input_file_path = kwargs['source_exposures_file_path']
        #     validation_file_path = kwargs['source_exposures_validation_file_path']
        #     transformation_file_path = kwargs['source_to_canonical_exposures_transformation_file_path']
        #     output_file_path = kwargs['canonical_exposures_file_path']
        # else:
        #     xtrans_path = omr['xtrans_path']
        #     input_file_path = tfp.source_exposures_file.name
        #     validation_file_path = omr['source_exposures_validation_file_path']
        #     transformation_file_path = omr['source_to_canonical_exposures_transformation_file_path']
        #     output_file_path = tfp.canonical_exposures_file.name
        #
        # (
        #     xtrans_path,
        #     input_file_path,
        #     validation_file_path,
        #     transformation_file_path,
        #     output_file_path
        # ) = map(
        #     os.path.abspath,
        #     [
        #         xtrans_path,
        #         input_file_path,
        #         validation_file_path,
        #         transformation_file_path,
        #         output_file_path
        #     ]
        # )
        #
        # xtrans_args = {
        #     'd': validation_file_path,
        #     'c': input_file_path,
        #     't': transformation_file_path,
        #     'o': output_file_path,
        #     's': ''
        # }
        #
        # try:
        #     run_mono_executable(xtrans_path, xtrans_args)
        # except OasisException as e:
        #     raise e
        #
        # with io.open(output_file_path, 'r', encoding='utf-8') as f:
        #     if not with_model_resources:
        #         return f
        #
        #     tfp.canonical_exposures_file = f
        #
        # return oasis_model

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

        :todo: replace this with the python version
        """
        # omr = oasis_model.resources
        # tfp = omr['oasis_files_pipeline']
        #
        # if not with_model_resources:
        #     xtrans_path = kwargs['xtrans_path']
        #     input_file_path = kwargs['canonical_exposures_file_path']
        #     validation_file_path = kwargs['canonical_exposures_validation_file_path']
        #     transformation_file_path = kwargs['canonical_to_model_exposures_transformation_file_path']
        #     output_file_path = kwargs['model_exposures_file_path']
        # else:
        #     xtrans_path = omr['xtrans_path']
        #     input_file_path = tfp.canonical_exposures_file.name
        #     validation_file_path = omr['canonical_exposures_validation_file_path']
        #     transformation_file_path = omr['canonical_to_model_exposures_transformation_file_path']
        #     output_file_path = tfp.model_exposures_file.name
        #
        # (
        #     xtrans_path,
        #     input_file_path,
        #     validation_file_path,
        #     transformation_file_path,
        #     output_file_path
        # ) = map(
        #     os.path.abspath,
        #     [
        #         xtrans_path,
        #         input_file_path,
        #         validation_file_path,
        #         transformation_file_path,
        #         output_file_path
        #     ]
        # )
        #
        # xtrans_args = {
        #     'd': validation_file_path,
        #     'c': input_file_path,
        #     't': transformation_file_path,
        #     'o': output_file_path
        # }
        #
        # try:
        #     run_mono_executable(xtrans_path, xtrans_args)
        # except OasisException as e:
        #     raise e
        #
        # with io.open(output_file_path, 'r', encoding='utf-8') as f:
        #     if not with_model_resources:
        #         return f
        #
        #     tfp.model_exposures_file = f
        #
        # return oasis_model

    def load_canonical_profile(
            self,
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

    def get_keys(self, oasis_model=None, model_exposures_file_path=None, lookup=None, keys_file_path=None, **kwargs):
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

    def _process_default_kwargs(self, oasis_model=None, **kwargs):
        if oasis_model:
            kwargs.setdefault('canonical_exposures_file_path', oasis_model.files_pipeline.canonical_exposures_path)
            kwargs.setdefault('keys_file_path', oasis_model.files_pipeline.keys_file_path)
            kwargs.setdefault('canonical_exposures_profile', oasis_model.resources.get('canonical_exposures_profile'))
            kwargs.setdefault('canonical_exposures_profile_json', oasis_model.resources.get('canonical_exposures_profile_json'))
            kwargs.setdefault('canonical_exposures_profile_json_path', oasis_model.resources.get('canonical_exposures_profile_json_path'))
            kwargs.setdefault('items_file_path', oasis_model.resources.get('items_file_path'))
            kwargs.setdefault('items_timestamped_file_path', oasis_model.resources.get('items_timestamped_file_path'))
            kwargs.setdefault('coverages_file_path', oasis_model.resources.get('coverages_file_path'))
            kwargs.setdefault('coverages_timestamped_file_path', oasis_model.resources.get('coverages_timestamped_file_path'))
            kwargs.setdefault('gulsummaryxref_file_path', oasis_model.resources.get('gulsummaryxref_file_path'))
            kwargs.setdefault('gulsummaryxref_timestamped_file_path', oasis_model.resources.get('gulsummaryxref_timestamped_file_path'))
            kwargs.setdefault('source_exposures_file_path', oasis_model.resources.get('source_exposures_file_path'))
            kwargs.setdefault('source_exposures_validation_file_path', oasis_model.resources.get('source_exposures_validation_file_path'))
            kwargs.setdefault('source_to_canonical_exposures_transformation_file_path', oasis_model.resources.get('source_to_canonical_exposures_transformation_file_path'))
            kwargs.setdefault('canonical_exposures_file_path', oasis_model.resources.get('canonical_exposures_file_path'))
            kwargs.setdefault('canonical_exposures_validation_file_path', oasis_model.resources.get('canonical_exposures_validation_file_path'))
            kwargs.setdefault('canonical_to_model_exposures_transformation_file_path', oasis_model.resources.get('canonical_to_model_exposures_transformation_file_path'))

        if not kwargs.get('canonical_exposures_profile'):
            kwargs['canonical_exposures_profile'] = self.load_canonical_profile(
                oasis_model=oasis_model,
                canonical_exposures_profile_json=kwargs.get('canonical_exposures_profile_json'),
                canonical_exposures_profile_json_path=kwargs.get('canonical_exposures_profile_json_path'),
            )

        return kwargs

    def load_master_data_frame(self, canonical_exposures_file_path, keys_file_path, canonical_exposures_profile, **kwargs):
        with io.open(canonical_exposures_file_path, 'r', encoding='utf-8') as cf:
            canexp_df = pd.read_csv(cf, float_precision='high')
            canexp_df = canexp_df.where(canexp_df.notnull(), None)
            canexp_df.columns = canexp_df.columns.str.lower()

        with io.open(keys_file_path, 'r', encoding='utf-8') as kf:
            keys_df = pd.read_csv(kf, float_precision='high')
            keys_df = keys_df.rename(columns={'CoverageID': 'CoverageType'})
            keys_df = keys_df.where(keys_df.notnull(), None)
            keys_df.columns = keys_df.columns.str.lower()

        tiv_fields = sorted(
            filter(lambda v: v.get('FieldName') == 'TIV', six.itervalues(canonical_exposures_profile))
        )

        result = pd.DataFrame(columns=[
            'item_id',
            'coverage_id',
            'tiv',
            'areaperil_id',
            'vulnerability_id',
            'group_id',
            'summary_id',
            'summaryset_id'
        ])

        item_id = 0
        for i in range(len(keys_df)):
            keys_item = keys_df.iloc[i]

            canexp_item = canexp_df[canexp_df['row_id'] == keys_item['locid']]

            if canexp_item.empty:
                raise OasisException(
                    "No matching canonical exposure item found in canonical exposures data frame for keys item {}.".format(keys_item)
                )
            elif len(canexp_item) > 1:
                raise OasisException(
                    "Duplicate canonical exposure items found in canonical exposures data frame for keys item {}.".format(keys_item)
                )

            canexp_item = canexp_item.iloc[0]

            tiv_field = next(f for f in tiv_fields if f['CoverageTypeID'] == keys_item['coveragetype'])
            tiv_lookup = tiv_field['ProfileElementName'].lower()
            tiv_value = canexp_item[tiv_lookup]
            if tiv_value > 0:
                item_id += 1
                result = result.append([{
                    'item_id': item_id,
                    'coverage_id': item_id,
                    'tiv': tiv_value,
                    'areaperil_id': keys_item['areaperilid'],
                    'vulnerability_id': keys_item['vulnerabilityid'],
                    'group_id': item_id,
                    'summary_id': 1,
                    'summaryset_id': 1,
                }])

        return result

    def _write_csvs(self, data_frame, file_path, timestamped_file_path, columns):
        data_frame.to_csv(
            columns=columns,
            path_or_buf=file_path,
            encoding='utf-8',
            chunksize=1000,
            index=False
        )

        data_frame.to_csv(
            columns=columns,
            path_or_buf=timestamped_file_path,
            encoding='utf-8',
            chunksize=1000,
            index=False
        )

    def generate_items_file(self, oasis_model=None, data_frame=None, **kwargs):
        """
        Generates an items file for the given ``oasis_model``.
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        if data_frame is None:
            data_frame = self.load_master_data_frame(**kwargs)

        self._write_csvs(
            data_frame,
            kwargs['items_file_path'],
            kwargs['items_timestamped_file_path'],
            ['item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id']
        )

        if oasis_model:
            oasis_model.files_pipeline.items_file_path = kwargs['items_file_path']

        return kwargs['items_file_path']

    def generate_coverages_file(self, oasis_model=None, data_frame=None, **kwargs):
        """
        Generates a coverages file for the given ``oasis_model``.
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        if data_frame is None:
            data_frame = self.load_master_data_frame(**kwargs)

        self._write_csvs(
            data_frame,
            kwargs['coverages_file_path'],
            kwargs['coverages_timestamped_file_path'],
            ['coverage_id', 'tiv']
        )

        if oasis_model:
            oasis_model.files_pipeline.coverages_file = kwargs.get('coverages_file_path')

        return kwargs.get('coverages_file_path')

    def generate_gulsummaryxref_file(self, oasis_model=None, data_frame=None, **kwargs):
        """
        Generates a gulsummaryxref file for the given ``oasis_model``.
        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)

        if data_frame is None:
            data_frame = self.load_master_data_frame(**kwargs)

        self._write_csvs(
            data_frame,
            kwargs['gulsummaryxref_file_path'],
            kwargs['gulsummaryxref_timestamped_file_path'],
            ['coverage_id', 'summary_id', 'summaryset_id']
        )

        if oasis_model:
            oasis_model.files_pipeline.gulsummaryxref_path = kwargs['gulsummaryxref_timestamped_file_path']

        return kwargs['gulsummaryxref_timestamped_file_path']

    def generate_oasis_files(self, oasis_model=None, **kwargs):
        """
        For a given ``oasis_model`` generates the standard Oasis files, namely

            ``items.csv``
            ``coverages.csv``
            ``gulsummaryxref.csv``

        """
        kwargs = self._process_default_kwargs(oasis_model=oasis_model, **kwargs)
        data_frame = self.load_master_data_frame(**kwargs)

        self.generate_items_file(oasis_model=oasis_model, data_frame=data_frame, **kwargs)
        self.generate_coverages_file(oasis_model=oasis_model, data_frame=data_frame, **kwargs)
        self.generate_gulsummaryxref_file(oasis_model=oasis_model, data_frame=data_frame, **kwargs)

        return {
            'items_file_path': kwargs['items_file_path'],
            'coverages_file_path': kwargs['coverages_file_path'],
            'gulsummaryxref_file_path': kwargs['gulsummaryxref_file_path']
        }

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
        oasis_model.resources.get('oasis_files_pipeline').clear()

        return oasis_model

    def start_files_pipeline(self, oasis_model=None, oasis_files_path=None, source_exposures_path=None, logger=None):
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
        logger = logger or logging.getLogger()
        logger.info('Checking output files directory exists for model')

        if oasis_model and not oasis_files_path:
            oasis_files_path = oasis_model.resources.get('oasis_files_path')

        if not oasis_files_path:
            raise OasisException('No output directory provided.'.format(oasis_model))
        elif not os.path.exists(oasis_files_path):
            raise OasisException('Output directory {} does not exist on the filesystem.'.format(oasis_files_path))

        logger.info('Checking for source exposures file')
        if oasis_model and not source_exposures_path:
            source_exposures_path = oasis_model.files_pipeline.source_exposures_path or None

        if not source_exposures_path:
            raise OasisException("No source exposures file provided.")
        elif not os.path.exists(source_exposures_path):
            raise OasisException("Source exposures file {} does not exist on the filesysem.".format(source_exposures_path))

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        kwargs = self._process_default_kwargs(
            oasis_model=oasis_model,
            items_file_path=os.path.join(oasis_files_path, 'items.csv'),
            items_timestamped_file_path=os.path.join(oasis_files_path, 'items-{}.csv'.format(utcnow)),
            coverages_file_path=os.path.join(oasis_files_path, 'coverages.csv'),
            coverages_timestamped_file_path=os.path.join(oasis_files_path, 'coverages-{}.csv'.format(utcnow)),
            gulsummaryxref_file_path=os.path.join(oasis_files_path, 'gulsummaryxref.csv'),
            gulsummaryxref_timestamped_file_path=os.path.join(oasis_files_path, 'gulsummaryxref-{}.csv'.format(utcnow)),
            canonical_exposures_file_path=os.path.join(oasis_files_path, 'canexp-{}.csv'.format(utcnow)),
            model_exposures_file_path=os.path.join(oasis_files_path, 'modexp-{}.csv'.format(utcnow)),
            keys_file_path=os.path.join(oasis_files_path, 'oasiskeys-{}.csv'.format(utcnow)),
            source_exposures_file_path=os.path.join(oasis_files_path, os.path.basename(source_exposures_path))
        )

        if not os.path.exists(kwargs['source_exposures_file_path']):
            self.logger.info('Copying source exposures file to model output files directory')
            shutil.copy(source_exposures_path, kwargs['source_exposures_file_path'])

        logger.info('Generating canonical exposures file {canonical_exposures_file_path}'.format(**kwargs))
        self.transform_source_to_canonical(**kwargs)

        logger.info('Generating model exposures file {model_exposures_file_path}'.format(**kwargs))
        self.transform_canonical_to_model(**kwargs)

        logger.info('Generating keys file {keys_file_path}'.format(**kwargs))
        self.get_keys(oasis_model=oasis_model, **kwargs)

        logger.info('Generating Oasis files for model')
        return self.generate_oasis_files(oasis_model=oasis_model, **kwargs)
