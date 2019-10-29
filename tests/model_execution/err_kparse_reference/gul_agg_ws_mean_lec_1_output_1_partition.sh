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

   ps f -g $sess_id > log/subprocess_list
   pgrep -a --pgroup $proc_group_id | grep -v $proc_group_id | grep -v $$ >> log/killout.txt
   kill -9 $(pgrep --pgroup $proc_group_id | grep -x -v $proc_group_id | grep -x -v $$) 2>/dev/null
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
mkfifo fifo/gul_P1
mkfifo fifo/gul_S1_summary_P1

mkdir work/gul_S1_summaryleccalc

# --- Do ground up loss computes ---


tee < fifo/gul_S1_summary_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!

( summarycalc -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> log/stderror.err  &

( eve 1 1 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P1  ) 2>> log/stderror.err &

wait $pid1


# --- Do ground up loss kats ---


leccalc -r -Kgul_S1_summaryleccalc -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f fifo/*

# Stop ktools watcher
kill -9 $pid0
