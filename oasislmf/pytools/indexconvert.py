import pandas as pd
import numpy as np 
from lookup.builtin import normal_to_z_index

def change_footprint_apid(path, size_lat):
    """Reads CSV, applies Z-index conversion, and writes back."""
    df = pd.read_csv(path)
    
    areaperil_ids = df['areaperil_id'].values
    df['areaperil_id'] = np.array([normal_to_z_index(id, size_lat) for id in areaperil_ids])
    df.to_csv(path, index=False)

def change_footprint_apid_multi_peril(path, size_lat, size_lon, num_perils): 
    """Reads CSV, applies Z-index conversion for multiple perils and writes back to file"""
    df = pd.read_csv(path)

    areaperil_ids = df['areaperil_id'].values - 1
    grid_size = size_lat * size_lon
    
    peril_values = areaperil_ids // grid_size 
    index_values = areaperil_ids % grid_size + 1
    z_indices = np.array([normal_to_z_index(id, size_lat) for id in index_values])
    df['areaperil_id'] = z_indices * num_perils + peril_values  

    df.to_csv("footprint.csv", index=False)
    