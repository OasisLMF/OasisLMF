#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*


touch log/stderror.err
ktools_monitor.sh $$ & pid0=$!

exit_handler(){
   exit_code=$?
   kill -9 $pid0 2> /dev/null
   if [ "$exit_code" -gt 0 ]; then
       echo 'Ktools Run Error - exitcode='$exit_code
   else
       echo 'Run Completed'
   fi

   set +x
   group_pid=$(ps -p $$ -o pgid --no-headers)
   sess_pid=$(ps -p $$ -o sess --no-headers)
   script_pid=$$
   printf "Script PID:%d, GPID:%s, SPID:%d
" $script_pid $group_pid $sess_pid >> log/killout.txt

   ps f -g $sess_pid > log/subprocess_list
   PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk 'BEGIN { FS = "[ \t\n]+" }{ if ($1 >= '$script_pid') print}' | grep -v celery | grep -v *.sh)
   echo "$PIDS_KILL" >> log/killout.txt
   kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
   exit $exit_code
}
trap exit_handler QUIT HUP INT KILL TERM ERR EXIT

check_complete(){
    set +e
    proc_list="eve getmodel gulcalc fmcalc summarycalc eltcalc aalcalc leccalc pltcalc"
    has_error=0
    for p in $proc_list; do
        started=$(find log -name "$p*.log" | wc -l)
        finished=$(find log -name "$p*.log" -exec grep -l "finish" {} + | wc -l)
        if [ "$finished" -lt "$started" ]; then
            echo "[ERROR] $p - $((started-finished)) processes lost"
            has_error=1
        elif [ "$started" -gt 0 ]; then
            echo "[OK] $p"
        fi
    done
    if [ "$has_error" -ne 0 ]; then
        false # raise non-zero exit code
    fi
}
# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

mkdir work/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryaalcalc

mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1

mkfifo fifo/full_correlation/gul_P1

mkfifo fifo/full_correlation/gul_S1_summary_P1



# --- Do ground up loss computes ---


tee < fifo/gul_S1_summary_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!

( summarycalc -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---


tee < fifo/full_correlation/gul_S1_summary_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid2=$!

( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P1 < fifo/full_correlation/gul_P1 ) 2>> log/stderror.err  &

( eve 1 1 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P1 -a1 -i - > fifo/gul_P1  ) 2>> log/stderror.err &

wait $pid1 $pid2


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---


( aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv ) 2>> log/stderror.err & lpid1=$!
( aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv ) 2>> log/stderror.err & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*

check_complete
exit_handler
