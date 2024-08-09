import os
import numba as nb
import numpy as np
import pandas as pd

oasis_int = np.dtype(os.environ.get('OASIS_INT', 'i4'))
nb_oasis_int = nb.from_dtype(oasis_int)
oasis_int_size = oasis_int.itemsize

oasis_float = np.dtype(os.environ.get('OASIS_FLOAT', 'f4'))
nb_oasis_float = nb.from_dtype(oasis_float)
oasis_float_size = oasis_float.itemsize

areaperil_int = np.dtype(os.environ.get('AREAPERIL_TYPE', 'u4'))
nb_areaperil_int = nb.from_dtype(areaperil_int)
areaperil_int_size = areaperil_int.itemsize

null_index = oasis_int.type(-1)


summary_xref_dtype = np.dtype([('item_id', 'i4'), ('summary_id', 'i4'), ('summary_set_id', 'i4')])

# financial structure static input dtypes
fm_programme_dtype = np.dtype([('from_agg_id', 'i4'), ('level_id', 'i4'), ('to_agg_id', 'i4')])
fm_policytc_dtype = np.dtype([('level_id', 'i4'), ('agg_id', 'i4'), ('layer_id', 'i4'), ('profile_id', 'i4')])
fm_profile_dtype = np.dtype([('profile_id', 'i4'),
                             ('calcrule_id', 'i4'),
                             ('deductible_1', 'f4'),
                             ('deductible_2', 'f4'),
                             ('deductible_3', 'f4'),
                             ('attachment_1', 'f4'),
                             ('limit_1', 'f4'),
                             ('share_1', 'f4'),
                             ('share_2', 'f4'),
                             ('share_3', 'f4'),
                             ])
fm_profile_step_dtype = np.dtype([('profile_id', 'i4'),
                                  ('calcrule_id', 'i4'),
                                  ('deductible_1', 'f4'),
                                  ('deductible_2', 'f4'),
                                  ('deductible_3', 'f4'),
                                  ('attachment_1', 'f4'),
                                  ('limit_1', 'f4'),
                                  ('share_1', 'f4'),
                                  ('share_2', 'f4'),
                                  ('share_3', 'f4'),
                                  ('step_id', 'i4'),
                                  ('trigger_start', 'f4'),
                                  ('trigger_end', 'f4'),
                                  ('payout_start', 'f4'),
                                  ('payout_end', 'f4'),
                                  ('limit_2', 'f4'),
                                  ('scale_1', 'f4'),
                                  ('scale_2', 'f4'),
                                  ])
fm_profile_csv_col_map = {
    'deductible_1': 'deductible1',
    'deductible_2': 'deductible2',
    'deductible_3': 'deductible3',
    'attachment_1': 'attachment1',
    'limit_1': 'limit1',
    'share_1': 'share1',
    'share_2': 'share2',
    'share_3': 'share3',
    'limit_2': ' limit2',
    'scale_1': 'scale1',
    'scale_2': 'scale2',
}
fm_xref_dtype = np.dtype([('output_id', 'i4'), ('agg_id', 'i4'), ('layer_id', 'i4')])
fm_xref_csv_col_map = {'output_id': 'output'}

coverages_dtype = np.dtype([('coverage_id', 'i4'), ('tiv', 'f4')])

items_dtype = np.dtype([('item_id', 'i4'),
                        ('coverage_id', 'i4'),
                        ('areaperil_id', areaperil_int),
                        ('vulnerability_id', 'i4'),
                        ('group_id', 'i4')])


def load_as_ndarray(dir_path, name, _dtype, must_exist=True, col_map=None):
    """
    load a file as a numpy ndarray
    useful for multi-columns files
    Args:
        dir_path: path to the directory where the binary or csv file is stored
        name: name of the file
        _dtype: np.dtype
        must_exist: raise FileNotFoundError if no file is present
        col_map: name re-mapping to change name of csv columns
    Returns:
        numpy ndarray
    """

    if os.path.isfile(os.path.join(dir_path, name + '.bin')):
        return np.fromfile(os.path.join(dir_path, name + '.bin'), dtype=_dtype)
    elif must_exist or os.path.isfile(os.path.join(dir_path, name + '.csv')):
        # in csv column cam be out of order and have different name,
        # we load with pandas and write each column to the ndarray
        if col_map is None:
            col_map = {}
        with open(os.path.join(dir_path, name + '.csv')) as file_in:
            cvs_dtype = {col_map.get(key, key): col_dtype for key, (col_dtype, _) in _dtype.fields.items()}
            df = pd.read_csv(file_in, delimiter=',', dtype=cvs_dtype, usecols=list(cvs_dtype.keys()))
            res = np.empty(df.shape[0], dtype=_dtype)
            for name in _dtype.names:
                res[name] = df[col_map.get(name, name)]
            return res
    else:
        return np.empty(0, dtype=_dtype)


def load_as_array(dir_path, name, _dtype, must_exist=True):
    """
    load file as a single numpy array,
     useful for files with a binary version with only one type of value where their index correspond to an id.
     For example coverage.bin only contains tiv value for each coverage id
     coverage_id n correspond to index n-1
    Args:
        dir_path: path to the directory where the binary or csv file is stored
        name: name of the file
        _dtype: numpy dtype of the required array
        must_exist: raise FileNotFoundError if no file is present
    Returns:
        numpy array of dtype type
    """
    fp = os.path.join(dir_path, name + '.bin')
    if os.path.isfile(fp):
        return np.fromfile(fp, dtype=_dtype)
    elif must_exist or os.path.isfile(os.path.join(dir_path, name + '.csv')):
        fp = os.path.join(dir_path, name + '.csv')
        with open(fp) as file_in:
            return np.loadtxt(file_in, dtype=_dtype, delimiter=',', skiprows=1, usecols=1)
    else:
        return np.empty(0, dtype=_dtype)


float_equal_precision = np.finfo(oasis_float).eps


@nb.njit(cache=True)
def almost_equal(a, b):
    return abs(a - b) < float_equal_precision
