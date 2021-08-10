#!/usr/bin/env python3

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

locationfile = pd.read_csv('location.csv')
accountfile = pd.read_csv('account.csv')

split_location = locationfile.groupby('FlexiLocUnit')
split_account = accountfile.groupby('FlexiAccUnit')

newpath = 'units'
if not os.path.exists(newpath):
    os.makedirs(newpath)

cwd = os.getcwd()
for name, group in split_location:
    sub_dir = os.path.join(newpath,(str)(name))

#loop through the groups and save to directories based on unique values
for name, group in split_location:
    sub_dir = os.path.join(newpath,(str)(name))
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    group = group.drop(['FlexiLocUnit'], axis=1)
    group.to_csv(sub_dir + "/location.csv", index=0)

for name, group in split_account:
    sub_dir = os.path.join(newpath,(str)(name))
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    group = group.drop(['FlexiAccUnit'], axis=1)
    group.to_csv(sub_dir + "/account.csv", index=0)

names = sorted([str(item[0]) for item in split_location])

#Function to sort fm string in Ascedning order
import re

def ascedning(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [ ascedning(c) for c in re.split(r'(\d+)', text) ]


names.sort(key=natural_keys)
#print(names)

units_dir=os.path.join(cwd,'units')

if not os.path.exists(units_dir):
    os.mkdir(units_dir)

with open(os.path.join(units_dir,'units.txt'), "w") as txt_file:
    names, groups = map(list, zip(*split_location))
    for name in names:
        txt_file.write(str(name) + '\n')
