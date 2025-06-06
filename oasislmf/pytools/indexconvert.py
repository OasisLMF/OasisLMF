import pandas as pd
import numpy as np
from oasislmf.lookup.builtin import normal_to_z_index
from oasislmf.utils.status import OASIS_UNKNOWN_ID


def change_footprint_apid(path, size_lat):
    """Reads CSV, applies Z-index conversion, and writes back."""
    change_footprint_apid_multi_peril(path, size_lat, None, 1)


def change_footprint_apid_multi_peril(path, size_lat, size_lon, num_perils):
    """
    Reads CSV, applies Z-index conversion for multiple perils and writes
    back to file
    """
    df = pd.read_csv(path)

    areaperil_ids = df['areaperil_id'].values - 1

    mask = df['areaperil_id'] != OASIS_UNKNOWN_ID
    filtered_df = df[mask].copy()

    areaperil_ids = filtered_df['areaperil_id'].values - 1

    if size_lon is not None:
        grid_size = size_lat * size_lon
        peril_values = areaperil_ids // grid_size
        index_values = areaperil_ids % grid_size + 1
    else:
        peril_values = areaperil_ids // 1
        index_values = areaperil_ids + 1

    z_indices = np.array(
        [normal_to_z_index(id, size_lat) for id in index_values]
    )

    filtered_df['areaperil_id'] = z_indices * num_perils + peril_values
    df.loc[mask, 'areaperil_id'] = filtered_df['areaperil_id']

    df = df.sort_values(by=df.columns.tolist(), kind='stable')
    df.to_csv(path, index=False)
