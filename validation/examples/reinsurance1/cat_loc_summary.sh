#!/bin/bash
# Script to update the expected results of a list of test cases
# Removes /expected and copies the contents of /run into /expected for those directories listed in list.txt
filename='list.txt'
rm loc_summary.csv
touch loc_summary.csv
n=1
while read d; do
# reading each line
echo "$d"
#sed "-n 1p tests/$d/expected/loc_summary.csv > tests/loc_summary.csv" #header
cat loc_summary.csv "tests/$d/expected/loc_summary.csv" > loc_summary1.csv
cp loc_summary1.csv loc_summary.csv
n=$((n+1))
done < $filename
rm loc_summary1.csv
n=$((n-1))
echo "$n test case loc_summary concantenated"
