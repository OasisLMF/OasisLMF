#!/bin/bash
# Script to concatenate loc_summary.csv across subdirectories listed in list.txt
filename='list.txt'
rm loc_summary.csv
touch loc_summary.csv
n=1
while read d; do
# reading each line
echo "$d"
tail -n +2 "units/$d/expected/loc_summary.csv" > loc_summary1.csv
cat loc_summary.csv loc_summary1.csv > loc_summary2.csv
cp loc_summary2.csv loc_summary.csv
n=$((n+1))
if n=2
then
	head -n 1 "units/$d/expected/loc_summary.csv" > loc_summary_header.csv
fi
done < $filename
cat loc_summary_header.csv loc_summary2.csv > loc_summary.csv
rm loc_summary1.csv
rm loc_summary2.csv
rm loc_summary_header.csv
n=$((n-1))
echo "$n test case loc_summary concantenated"
