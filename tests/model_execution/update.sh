#!/bin/bash


cp output_bash_base/* reference_bash_base
cp output_bash_lb/* reference_bash_lb 
cp output_bash_err/* reference_bash_err 
cp output_bash_csm/* reference_bash_csm 

cd reference_bash_err; ./update-tmp-tests.sh


