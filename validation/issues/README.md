*README*

**Process for adding a new issue to fm validation test**

**Creating and testing the new test case***

1) Create unit test case folder oasislmf/validation/issues/units/[issue_number]
2) Use the OED file templates account_new.csv (etc) in oasislmf/validation/issues/units and enter test case input data. Save in new unit folder
3) Run new test case against fixed branch and generate correct results with loc_summary file;
oasislmf exposure run -s validation/issues/units/[issue_number] -r validation/issues/units/[issue_number]/runs -o loc -f validation/issues/units/[issue_number]/runs/loc_summary.csv

***Add the finished test case to the automated tests/fm***

4) In oasislmf/validation/issues/units edit units.txt and add new issue number to the bottom.
5) In oasislmf/validation run combine.py as follows;
python combine.py -d issues
This will recombine the location and accounts file from all of the unit subfolders specified in units.txt and place them in the issues/ directory as location_concat.csv and account_concat.csv. Note that ri_info and ri_scope must be concatenated manually.
6) Rename location_concat.csv as location.csv and account_concat.csv as account.csv
7) Test the combined test files;
oasislmf exposure run -s validation/issues/ -r validation/issues/runs -o loc -f validation/issues/runs/loc_summary.csv
8) Compare with issues/expected and check that they are the same except the new issue results are present and correct in the combined results.
9) Go into oasislmf/tests/fm and run
pytest
10) Either manually update the expected results in validation/issues to include the results for the new test case, or change the self.update_expected flag to True and rerun the pytest until all tests pass.



