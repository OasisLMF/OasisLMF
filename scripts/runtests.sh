#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$( realpath $SCRIPT_DIR | sed 's/scripts.*//g' )
LOG_OUTPUT="$ROOT_DIR"reports
TAR_OUTPUT="$ROOT_DIR"dist

cd $ROOT_DIR
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

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
docker run  --ulimit nofile=8192:8192 -v "$LOG_OUTPUT":/var/log/oasis -v "$ROOT_DIR":/home  oasislmf-tester:"$DOCKER_TAG"
