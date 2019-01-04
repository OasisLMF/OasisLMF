#!/usr/bin/env bash

CWD=$(pwd)
LOG_OUTPUT=$CWD/reports
TAR_OUTPUT=$CWD/dist

mkdir $LOG_OUTPUT $TAR_OUTPUT

docker build -f docker/Dockerfile.oasislmf_tester -t oasislmf-tester .
docker run -v $LOG_OUTPUT:/var/log/oasis -v $TAR_OUTPUT:/tmp/output oasislmf-tester:latest
