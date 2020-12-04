#!/bin/bash

docker build -f Dockerfile.FmTestingNotebook -t fm_testing_tool_fm_testing_notebook ..
docker-compose up -d
