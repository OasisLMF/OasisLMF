#!/bin/bash

docker rmi fm_testing_tool_fm_testing_notebook:latest
docker build -f Dockerfile.FmTestingNotebook -t fm_testing_tool_fm_testing_notebook ..
docker-compose up -d
