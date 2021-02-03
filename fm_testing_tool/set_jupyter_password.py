#!/usr/bin/env python3
import io
import os 
import json
from notebook.auth import passwd

try:
    pass_text = os.environ['JUPYTER_PASS']
    pass_hash = passwd(pass_text)

    store_dirpath  = '/root/.jupyter/'
    store_filename = 'jupyter_notebook_config.json'
    store_contents = {"NotebookApp": {"password": pass_hash}}

    with io.open(os.path.join(store_dirpath, store_filename), 'w') as nb_conf:
        json.dump(store_contents, nb_conf, indent=4)
    print('jupyter password set from Env variable')    
except Exception as e:
    print('failed to set password:')
    print(e)
