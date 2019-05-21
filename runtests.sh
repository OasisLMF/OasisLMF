#!/usr/bin/env bash

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_OUTPUT="$SCRIPT_DIR"/reports
TAR_OUTPUT="$SCRIPT_DIR"/dist

if [ ! -d "$LOG_OUTPUT" ]; then
  mkdir "$LOG_OUTPUT"
else
  rm -fr "$LOG_OUTPUT"/*
fi

if [ ! -d "$TAR_OUTPUT" ]; then
  mkdir "$TAR_OUTPUT"
else
  rm -fr "$TAR_OUTPUT"/*
fi

if [ -z "$1" ]; then
    DOCKER_TAG='latest'
else
    DOCKER_TAG=$1
fi

docker build --no-cache -f docker/Dockerfile.oasislmf_tester -t oasislmf-tester .
docker run  --ulimit nofile=8192:8192 -v "$LOG_OUTPUT":/var/log/oasis -v "$SCRIPT_DIR":/home  oasislmf-tester:"$DOCKER_TAG"
