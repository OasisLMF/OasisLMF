#!/bin/bash
# Script to concatenate loc_summary.csv across subdirectories listed in list.txt
filename='list.txt'
n=1
while read d; do
# reading each line
echo "$d"
sed -i '/SS01_04,/d' "units/$d/location.csv"
sed -i '/SS01_04,/d' "units/$d/account.csv"
n=$((n+1))
done < $filename
n=$((n-1))
echo "Removed SS01_04 from $n test cases"
