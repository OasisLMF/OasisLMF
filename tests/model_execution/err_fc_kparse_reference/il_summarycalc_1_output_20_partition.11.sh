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
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/


mkfifo fifo/il_P12

mkfifo fifo/il_S1_summary_P12
mkfifo fifo/il_S1_summarysummarycalc_P12
mkfifo fifo/il_S1_summarycalc_P12

mkfifo il_S1_summary_P12
mkfifo il_S1_summarysummarycalc_P12
mkfifo il_S1_summarycalc_P12



# --- Do insured loss computes ---
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P12 > work/kat/il_S1_summarycalc_P12 & pid1=$!
tee < fifo/il_S1_summary_P12 fifo/il_S1_summarysummarycalc_P12 > /dev/null & pid2=$!
( summarycalc -f  -1 fifo/il_S1_summary_P12 < fifo/il_P12 ) 2>> log/stderror.err  &

( eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P12 -a1 -i - | fmcalc -a2 > fifo/il_P12  ) 2>> log/stderror.err &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---

( fmcalc-a2 < gul_P12 > il_P12 ) 2>> log/stderror.err & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
summarycalctocsv -s < il_S1_summarysummarycalc_P12 > work/full_correlation/kat/il_S1_summarycalc_P12 & pid1=$!
tee < il_S1_summary_P12 il_S1_summarysummarycalc_P12 > /dev/null & pid2=$!
( summarycalc -f  -1 il_S1_summary_P12 < il_P12 ) 2>> log/stderror.err  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_summarycalc_P12 > output/il_S1_summarycalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_summarycalc_P12 > output/full_correlation/il_S1_summarycalc.csv & kpid2=$!
wait $kpid1 $kpid2


# Stop ktools watcher
kill -9 $pid0
