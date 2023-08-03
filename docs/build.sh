#!/bin/bash

DIR_BASE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR_ENV="$(which python | xargs dirname)"

echo Building documentation in "$(pwd)"
echo
# Update python env
pip install -r requirements.txt

# Update JSON specs 
    # ./update-redoc.py

# Build docs
cd $DIR_BASE
make html SPHINXBUILD="python ${DIR_ENV}/sphinx-build"

# Create TAR
# if [[ ! -d "$DIR_BASE/output/" ]]; then 
#     mkdir $DIR_BASE/output/
# fi 
# tar -czvf OasisLMF_docs.tar.gz -C build/html/ .
# mv OasisLMF_docs.tar.gz $DIR_BASE/output/