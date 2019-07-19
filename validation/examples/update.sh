#!/bin/bash
# Script to update the expected results of a list of test cases
# Removes /expected and copies the contents of /run into /expected for those directories listed in list.txt
filename='list.txt'
n=1
while read d; do
# reading each line
echo "Updating expected for $d"
rm -r -f "$d/expected/"
mkdir "$d/expected"
cp -r "$d/run/"* "$d/expected/"
n=$((n+1))
done < $filename
echo "$n test case expected results updated"