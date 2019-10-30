#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail

error_handler(){
   echo 'Run Error - terminating'
   proc_group_id=$(ps -p $$ -o pgid --no-headers)
   sess_id=$(ps -p $$ -o sess --no-headers)
   echo "script pid: $$" > log/killout.txt
   echo "group pid: $proc_group_id" >> log/killout.txt
   echo "session pid: $sess_id" >> log/killout.txt
   echo "----------------"  >> log/killout.txt

   if hash pstree 2>/dev/null; then
       pstree -pn $$ >> log/killout.txt
       PIDS_KILL=$(pstree -pn $$ | grep -o "([[:digit:]]*)" | grep -o "[[:digit:]]*")
       kill -9 $(echo "$PIDS_KILL" | grep -v $proc_group_id | grep -v $$) 2>/dev/null
   else
       ps f -g $sess_id > log/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $proc_group_id | grep -v celery | grep -v $proc_group_id | grep -v $$)
       echo "$PIDS_KILL" >> log/killout.txt
       kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
   fi
   exit 1
}
trap error_handler QUIT HUP INT KILL TERM ERR

mkdir -p log
rm -R -f log/*
touch log/stderror.err
ktools_monitor.sh $$ & pid0=$!

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat
mkfifo fifo/il_P1
mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summarypltcalc_P1
mkfifo fifo/il_S1_pltcalc_P1


# --- Do insured loss computes ---

pltcalc < fifo/il_S1_summarypltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid1=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summarypltcalc_P1 > /dev/null & pid2=$!

( summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> log/stderror.err  &

( eve 1 1 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > fifo/il_P1  ) 2>> log/stderror.err &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P1 > output/il_S1_pltcalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f fifo/*

# Stop ktools watcher
kill -9 $pid0
