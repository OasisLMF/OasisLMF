#!/bin/bash

LOG_FILE='log/killout.txt'
ERR_FILE='log/stderror.err'
STATUS_LINES=15
POLL_RATE=2
script_pid=$1

run_ktools_kill(){
     set -e
     FMCALC=`ps -C fmcalc -o pmem | grep -v MEM | sort -n -r | head -1`
     GULCALC=`ps -C gulcalc -o pmem | grep -v MEM | sort -n -r | head -1`
     GETMODEL=`ps -C getmodel -o pmem | grep -v MEM | sort -n -r | head -1`
     echo "TOTALS:  $FMCALC $GULCALC $GETMODEL" >> log/mem-free
     free -h >> log/mem-free
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
    if [ $(ps xao pid | grep -c $script_pid) -eq 0 ]; then
        echo "PID ($script_pid) is not running, exiting"
        exit 0
    fi    
    if [ -s $ERR_FILE ]; then
        run_ktools_kill $script_pid
    fi
done
