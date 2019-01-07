#!/usr/bin/env bash

CWD=$(pwd)
LOG_OUTPUT=$CWD/reports
TAR_OUTPUT=$CWD/dist

mkdir $LOG_OUTPUT $TAR_OUTPUT

if [ -z "$1" ]; then
    DOCKER_TAG='latest'
else
    DOCKER_TAG=$1
fi

docker build -f docker/Dockerfile.oasislmf_tester -t oasislmf-tester .
docker run  --ulimit nofile=8192:8192 -v $LOG_OUTPUT:/var/log/oasis -v $TAR_OUTPUT:/tmp/output oasislmf-tester:$DOCKER_TAG
