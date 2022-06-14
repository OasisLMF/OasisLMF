#!/bin/bash

SCRIPT_PID=$1
LOG_DIR=$2

ERR_FILE=$LOG_DIR'/stderror.err'
STATUS_LINES=15
POLL_RATE=2

run_ktools_kill(){
     set -e
     FMCALC=`ps -C fmcalc -o pmem | grep -v MEM | sort -n -r | head -1`
     GULCALC=`ps -C gulcalc -o pmem | grep -v MEM | sort -n -r | head -1`
     GETMODEL=`ps -C getmodel -o pmem | grep -v MEM | sort -n -r | head -1`
     echo "TOTALS:  $FMCALC $GULCALC $GETMODEL" >> $LOG_DIR/mem-free
     free -h >> $LOG_DIR/mem-free
     if ! kill $1 > /dev/null 2>&1; then
         kill $1
        echo "Could not send SIGTERM to process $1" >&2
     else
         echo "kill signal send PID ($1)"
     fi
}

# trigger kill on stderr output
while :
do
    sleep $POLL_RATE
    # check monitored process is running
    if [ $(ps xao pid | grep -c $SCRIPT_PID) -eq 0 ]; then
        echo "PID ($SCRIPT_PID) is not running, exiting"
        exit 0
    fi
    if [ -s $ERR_FILE ]; then
        run_ktools_kill $SCRIPT_PID
    fi
done
