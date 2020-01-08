#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

error_handler(){
   echo 'Run Error - terminating'
   exit_code=$?
   set +x
   group_pid=$(ps -p $$ -o pgid --no-headers)
   sess_pid=$(ps -p $$ -o sess --no-headers)
   printf "Script PID:%d, GPID:%s, SPID:%d" $$ $group_pid $sess_pid >> log/killout.txt

   if hash pstree 2>/dev/null; then
       pstree -pn $$ >> log/killout.txt
       PIDS_KILL=$(pstree -pn $$ | grep -o "([[:digit:]]*)" | grep -o "[[:digit:]]*")
       kill -9 $(echo "$PIDS_KILL" | grep -v $group_pid | grep -v $$) 2>/dev/null
   else
       ps f -g $sess_pid > log/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $group_pid | grep -v celery | grep -v $group_pid | grep -v $$)
       echo "$PIDS_KILL" >> log/killout.txt
       kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
   fi
   exit $(( 1 > $exit_code ? 1 : $exit_code ))
}
trap error_handler QUIT HUP INT KILL TERM ERR

touch log/stderror.err
ktools_monitor.sh $$ & pid0=$!

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/


mkfifo fifo/gul_P1

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryeltcalc_P1
mkfifo fifo/il_S1_eltcalc_P1



# --- Do insured loss computes ---

eltcalc < fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryeltcalc_P1 > /dev/null & pid2=$!

( summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> log/stderror.err  &

( eve 1 1 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > fifo/il_P1  ) 2>> log/stderror.err &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 > output/il_S1_eltcalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f fifo/*

# Stop ktools watcher
kill -9 $pid0
