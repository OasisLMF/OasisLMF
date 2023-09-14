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

location_file = pd.read_csv('location.csv', dtype=str, keep_default_na=False)
account_file = pd.read_csv('account.csv', dtype=str, keep_default_na=False)
if os.path.isfile('ri_info.csv') and os.path.isfile('ri_scope.csv'):
    ri_info_file = pd.read_csv('ri_info.csv', dtype=str, keep_default_na=False)
    ri_scope_file = pd.read_csv('ri_scope.csv', dtype=str, keep_default_na=False)
else:
    ri_scope_file = None

split_location = location_file.groupby('FlexiLocUnit')
split_account = account_file.groupby('FlexiAccUnit')

newpath = 'units'
if not os.path.exists(newpath):
    os.makedirs(newpath)

cwd = os.getcwd()

# loop through the groups and save to directories based on unique values
for name, group in split_location:
    sub_dir = os.path.join(newpath, name)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    group = group.drop(['FlexiLocUnit'], axis=1)

    group.to_csv(sub_dir + "/location.csv", index=0)

    if ri_scope_file is not None:
        ri_scope = ri_scope_file[(ri_scope_file['PortNumber'].isin(np.unique(group['PortNumber'])))
                                 & (ri_scope_file['AccNumber'].isin(list(np.unique(group['AccNumber'])) + ['']))]
        ri_info = ri_info_file[ri_info_file['ReinsNumber'].isin(np.unique(ri_scope['ReinsNumber']))]
        if not ri_scope.empty:
            ri_scope.to_csv(sub_dir + "/ri_scope.csv", index=0)
            ri_info.to_csv(sub_dir + "/ri_info.csv", index=0)


for name, group in split_account:
    sub_dir = os.path.join(newpath, name)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    group = group.drop(['FlexiAccUnit'], axis=1)
    group.to_csv(sub_dir + "/account.csv", index=0)

names = sorted([str(item[0]) for item in split_location])

# Function to sort fm string in Ascedning order


def ascedning(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):

    return [ascedning(c) for c in re.split(r'(\d+)', text)]


names.sort(key=natural_keys)
# print(names)

units_dir = os.path.join(cwd, 'units')

if not os.path.exists(units_dir):
    os.mkdir(units_dir)

with open(os.path.join(units_dir, 'units.txt'), "w") as txt_file:
    names, groups = map(list, zip(*split_location))
    names.sort(key=natural_keys)
    for name in names:
        txt_file.write(str(name) + '\n')
