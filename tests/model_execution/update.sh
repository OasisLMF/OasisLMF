cp cov_kparse_output/* cov_kparse_reference
cp itm_kparse_output/* itm_kparse_reference
cp err_kparse_output/* err_kparse_reference
cp tmp_kparse_output/* tmp_kparse_reference

cd tmp_kparse_reference; ./update-tmp-tests.sh
