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
    sub_dir = os.path.join(newpath,name)

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
    names.sort(key=natural_keys)
    for name in names:
        txt_file.write(str(name) + '\n')


# get directories
with open(os.path.join(units_dir,'units.txt'), "r") as txt_file:
    fms = txt_file.read().split('\n')

dirs = []
for fm in fms:
    if fm!='':
        dirs.append(fm)

# combine dataframes

# start with first one
fm_first = dirs[0]
fm_first_filepath = os.path.join(newpath,fm_first,'location.csv')

df_loc = pd.read_csv(fm_first_filepath)
df_loc['FlexiLocUnit']=fm_first

# add in remaining fm files, iterating through remainder
for i in range(1,len(dirs)):
    fm_next = dirs[i]
    fm_next_filepath = os.path.join(newpath,fm_next,'location.csv')
    df_loc_tmp = pd.read_csv(fm_next_filepath)
    df_loc_tmp['FlexiLocUnit']=fm_next
    # concat files
    df_loc = pd.concat([df_loc,df_loc_tmp])

df_loc.to_csv('location_concat.csv',index=False)


#Account concat

fm2_first = dirs[0]
fm2_first_filepath = os.path.join(newpath,fm2_first,'account.csv')

df_loc2 = pd.read_csv(fm2_first_filepath)
df_loc2['FlexiAccUnit']=fm2_first

# add in remaining fm files, iterating through remainder
for i in range(1,len(dirs)):
    fm2_next = dirs[i]
    fm2_next_filepath = os.path.join(newpath,fm2_next,'account.csv')
    df_loc_tmp2 = pd.read_csv(fm2_next_filepath)
    df_loc_tmp2['FlexiAccUnit']=fm2_next
    # concat files
    df_loc2 = pd.concat([df_loc2,df_loc_tmp2])

df_loc2.to_csv('account_concat.csv',index=False)
