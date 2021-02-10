#!/bin/bash

cp cov_kparse_output/* cov_kparse_reference
cp itm_kparse_output/* itm_kparse_reference
cp err_kparse_output/* err_kparse_reference
cp lb_kparse_output/* lb_kparse_reference
cp tmp_kparse_output/* tmp_kparse_reference
cp eve_kparse_output/* eve_kparse_reference


cd tmp_kparse_reference; ./update-tmp-tests.sh
cd ../

cp itm_fc_kparse_output/* itm_fc_kparse_reference
cp err_fc_kparse_output/* err_fc_kparse_reference
cp tmp_fc_kparse_output/* tmp_fc_kparse_reference


cd tmp_fc_kparse_reference; ./update-tmp-fc-tests.sh
