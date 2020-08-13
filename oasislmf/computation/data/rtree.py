__all__ = [
    'GenerateRtreeIndexData'
]

import os

from ..base import ComputationStep
from ...utils.data import get_json
from ...utils.path import as_path
from ...utils.exceptions import OasisException
from ...lookup.rtree import PerilAreasIndex


class GenerateRtreeIndexData(ComputationStep):
    """
    Generates and writes an Rtree file index of peril area IDs (area peril IDs)
    and area polygon bounds from a peril areas (area peril) file.
    """
    step_params = [
        {'name': 'index_output_file', 'flag': '-f', 'is_path': True, 'pre_exist': False,
         'help': 'Index file path (no file extension required)'},
        {'name': 'lookup_config_json', 'flag': '-m', 'is_path': True, 'pre_exist': True, 'required': True,
         'help': 'Lookup config JSON file path'},
        {'name': 'lookup_data_dir', 'flag': '-d', 'is_path': True, 'pre_exist': True,
         'help': 'Model lookup/keys data directory path'},
    ]

    def run(self):
        config = get_json(src_fp=self.lookup_config_json)
        config_dir = os.path.dirname(self.lookup_config_json) if self.lookup_config_json else self.lookup_data_dir

        peril_config = config.get('peril')

        if not peril_config:
            raise OasisException(
                'The lookup config must contain a peril-related subdictionary with a key named '
                '`peril` defining area-peril-related model information'
            )

        areas_fp = peril_config.get('file_path')

        if not areas_fp:
            raise OasisException(
                'The lookup peril config must define the path of a peril areas '
                '(or area peril) file with the key name `file_path`'
            )

        if areas_fp.startswith('%%KEYS_DATA_PATH%%'):
            areas_fp = areas_fp.replace('%%KEYS_DATA_PATH%%', self.lookup_data_dir)

        if not os.path.isabs(areas_fp):
            areas_fp = os.path.join(config_dir, areas_fp)
            areas_fp = as_path(areas_fp, 'areas_fp')

        src_type = str.lower(str(peril_config.get('file_type')) or '') or 'csv'

        peril_id_col = str.lower(str(peril_config.get('peril_id_col')) or '') or 'peril_id'

        coverage_config = config.get('coverage')

        if not coverage_config:
            raise OasisException(
                'The lookup config must contain a coverage-related subdictionary with a key named '
                '`coverage` defining coverage related model information'
            )

        coverage_type_col = str.lower(str(coverage_config.get('coverage_type_col')) or '') or 'coverage_type'

        peril_area_id_col = str.lower(str(peril_config.get('peril_area_id_col')) or '') or 'area_peril_id'

        area_poly_coords_cols = peril_config.get('area_poly_coords_cols')

        if not area_poly_coords_cols:
            raise OasisException(
                'The lookup peril config must define the column names of '
                'the coordinates used to define areas in the peril areas '
                '(area peril) file using the key `area_poly_coords_cols`'
            )

        non_na_cols = (
            tuple(col.lower() for col in peril_config['non_na_cols']) if peril_config.get('non_na_cols')
            else tuple(col.lower() for col in [peril_area_id_col] + area_poly_coords_cols.values())
        )

        col_dtypes = peril_config.get('col_dtypes') or {peril_area_id_col: int}
        sort_cols = peril_config.get('sort_cols') or peril_area_id_col
        area_poly_coords_seq_start_idx = peril_config.get('area_poly_coords_seq_start_idx') or 1
        area_reg_poly_radius = peril_config.get('area_reg_poly_radius') or 0.00166

        index_props = peril_config.get('rtree_index')
        index_props.pop('filename')

        PerilAreasIndex.create_from_peril_areas_file(
            src_fp=areas_fp,
            src_type=src_type,
            peril_id_col=peril_id_col,
            coverage_type_col=coverage_type_col,
            peril_area_id_col=peril_area_id_col,
            non_na_cols=non_na_cols,
            col_dtypes=col_dtypes,
            sort_cols=sort_cols,
            area_poly_coords_cols=area_poly_coords_cols,
            area_poly_coords_seq_start_idx=area_poly_coords_seq_start_idx,
            area_reg_poly_radius=area_reg_poly_radius,
            index_fp=self.index_output_file,
            index_props=index_props
        )
        self.logger.info('\nGenerated peril areas Rtree index file {}.idx'.format(self.index_output_file))
        self.logger.info('Generated peril areas Rtree data file {}.dat\n'.format(self.index_output_file))
