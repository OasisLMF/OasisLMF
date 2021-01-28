#!/bin/bash
# Script to concatenate oed files within subdirectories listed in list.txt
filename='include.txt'
rm location.csv
rm account.csv
rm ri_info.csv
rm ri_scope.csv
touch location.csv
touch account.csv
touch ri_info.csv
touch ri_scope.csv
n=1
while read d; do
# reading each line
echo "$d"
#sed "-n 1p tests/$d/expected/loc_summary.csv > tests/loc_summary.csv" #header
tail -n +2 "units/$d/location.csv" > location1.csv
sed -i s/^/"$d",/ location1.csv
cat location.csv location1.csv > location2.csv
cp location2.csv location.csv
tail -n +2 "units/$d/account.csv" > account1.csv
sed -i s/^/"$d",/ account1.csv
cat account.csv account1.csv > account2.csv
cp account2.csv account.csv
tail -n +2 "units/$d/ri_info.csv" > ri_info1.csv
cat ri_info.csv ri_info1.csv > ri_info2.csv
cp ri_info2.csv ri_info.csv
tail -n +2 "units/$d/ri_scope.csv" > ri_scope1.csv
cat ri_scope.csv ri_scope1.csv > ri_scope2.csv
cp ri_scope2.csv ri_scope.csv
n=$((n+1))
if n=2
then
	head -n 1 "units/$d/location.csv" > location_header.csv
	sed -i s/^/'FlexiLocUnit,'/ location_header.csv
	head -n 1 "units/$d/account.csv" > account_header.csv
	sed -i s/^/'FlexiAccUnit,'/ account_header.csv
	head -n 1 "units/$d/ri_info.csv" > ri_info_header.csv
	head -n 1 "units/$d/ri_scope.csv" > ri_scope_header.csv
	echo "head $d"
	echo "$n"
fi
done < $filename
cat location_header.csv location2.csv > location.csv
cat account_header.csv account2.csv > account.csv
cat ri_info_header.csv ri_info2.csv > ri_info.csv
cat ri_scope_header.csv ri_scope2.csv > ri_scope.csv
rm location1.csv
rm account1.csv
rm ri_info1.csv
rm ri_scope1.csv
rm location2.csv
rm account2.csv
rm ri_info2.csv
rm ri_scope2.csv
rm location_header.csv
rm account_header.csv
rm ri_info_header.csv
rm ri_scope_header.csv
n=$((n-1))
echo "$n test case concantenated"
