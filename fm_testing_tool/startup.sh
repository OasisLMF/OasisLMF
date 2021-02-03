#!/bin/bash

set_jupyter_password.py
jupyter notebook --port=8888 --allow-root --no-browser --ip=0.0.0.0
