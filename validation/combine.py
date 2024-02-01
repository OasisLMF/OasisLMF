#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
import os
import json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-dir", "--subdirectory", help="enter subdirectory name")
parser.add_argument("--suffix", help="suffix to add to the name of the files")

params = parser.parse_args()

subdir = params.subdirectory
suffix = params.suffix

if suffix is None:
    suffix = '_concat'

os.chdir(subdir)

cwd = os.getcwd()
units_dir = 'units'

# get directories
with open(os.path.join(units_dir, 'units.txt'), "r") as txt_file:
    fms = txt_file.read().split('\n')

dirs = [fm for fm in fms if fm]
print(dirs)

# combine dataframes
df_loc = []
df_acc = []
df_ri_info = []
df_ri_scope = []

# add each fm files
for fm_next in dirs:
    loc_filepath = os.path.join(units_dir, fm_next, 'location.csv')
    df_loc_tmp = pd.read_csv(loc_filepath, dtype=str, keep_default_na=False)
    df_loc_tmp['FlexiLocUnit'] = fm_next
    df_loc.append(df_loc_tmp)

    acc_filepath = os.path.join(units_dir, fm_next, 'account.csv')
    df_acc_tmp = pd.read_csv(acc_filepath, dtype=str, keep_default_na=False)
    df_acc_tmp['FlexiAccUnit'] = fm_next
    # concat files
    df_acc.append(df_acc_tmp)

    ri_info_filepath = os.path.join(units_dir, fm_next, 'ri_info.csv')
    ri_scope_filepath = os.path.join(units_dir, fm_next, 'ri_scope.csv')
    if os.path.isfile(ri_info_filepath) and os.path.isfile(ri_scope_filepath):
        df_ri_info_tmp = pd.read_csv(ri_info_filepath, dtype=str, keep_default_na=False)
        df_ri_scope_tmp = pd.read_csv(ri_scope_filepath, dtype=str, keep_default_na=False)
        df_ri_info.append(df_ri_info_tmp)
        df_ri_scope.append(df_ri_scope_tmp)

pd.concat(df_loc).to_csv(f'location{suffix}.csv', index=False)
pd.concat(df_acc).to_csv(f'account{suffix}.csv', index=False)

if df_ri_info and df_ri_scope:
    pd.concat(df_ri_info).to_csv(f'ri_info{suffix}.csv', index=False)
    pd.concat(df_ri_scope).to_csv(f'ri_scope{suffix}.csv', index=False)
