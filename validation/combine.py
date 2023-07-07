#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
import os
import json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-dir", "--subdirectory", help="enter subdirectory name")

params = parser.parse_args()

subdir = params.subdirectory


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

pd.concat(df_loc).to_csv('location_concat.csv', index=False)
pd.concat(df_acc).to_csv('account_concat.csv', index=False)
