# Note that only the currently used fields are shown unless show_all is set to True. 
import os 
import pandas as pd

def load_df(path, required_file=None):
    if path:
        return  pd.read_csv(path)
    else:
        if required_file:
            raise FileNotFoundError(f"Required File does not exist: {required_file}")
        else:
            return None
