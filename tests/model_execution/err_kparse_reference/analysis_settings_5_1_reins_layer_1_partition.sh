#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


touch $LOG_DIR/stderror.err
ktools_monitor.sh $$ $LOG_DIR & pid0=$!

exit_handler(){
   exit_code=$?

   # disable handler
   trap - QUIT HUP INT KILL TERM ERR EXIT

   kill -9 $pid0 2> /dev/null
   if [ "$exit_code" -gt 0 ]; then
       # Error - run process clean up
       echo 'Ktools Run Error - exitcode='$exit_code

       set +x
       group_pid=$(ps -p $$ -o pgid --no-headers)
       sess_pid=$(ps -p $$ -o sess --no-headers)
       script_pid=$$
       printf "Script PID:%d, GPID:%s, SPID:%d
" $script_pid $group_pid $sess_pid >> $LOG_DIR/killout.txt

       ps -jf f -g $sess_pid > $LOG_DIR/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk 'BEGIN { FS = "[ \t\n]+" }{ if ($1 >= '$script_pid') print}' | grep -v celery | egrep -v *\\.log$  | egrep -v *\\.sh$ | sort -n -r)
       echo "$PIDS_KILL" >> $LOG_DIR/killout.txt
       kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
       exit $exit_code
   else
       # script successful
       exit 0
   fi
}
trap exit_handler QUIT HUP INT KILL TERM ERR EXIT

check_complete(){
    set +e
    proc_list="eve getmodel gulcalc fmcalc summarycalc eltcalc aalcalc leccalc pltcalc ordleccalc"
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
    else
        echo 'Run Completed'
    fi
}
# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/ri_S1_summaryleccalc

mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P1.idx

mkfifo fifo/ri_P1

mkfifo fifo/ri_S1_summary_P1
mkfifo fifo/ri_S1_summary_P1.idx



# --- Do reinsurance loss computes ---



tee < fifo/ri_S1_summary_P1 work/ri_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < fifo/ri_S1_summary_P1.idx work/ri_S1_summaryleccalc/P1.idx > /dev/null & pid2=$!

( summarycalc -m -f -p RI_1 -1 fifo/ri_S1_summary_P1 < fifo/ri_P1 ) 2>> $LOG_DIR/stderror.err  &

# --- Do insured loss computes ---



tee < fifo/il_S1_summary_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid3=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid4=$!

( summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---

( eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid5=$!
( summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid6=$!
( pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid7=$!


tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid8=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx > /dev/null & pid9=$!

( summarycalc -m -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &

( ( eve 1 1 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 | tee fifo/il_P1 | fmcalc -a3 -n -p RI_1 > fifo/ri_P1 ) 2>> $LOG_DIR/stderror.err ) & pid10=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10


# --- Do reinsurance loss kats ---


# --- Do insured loss kats ---


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 > output/gul_S1_eltcalc.csv & kpid1=$!
kat work/kat/gul_S1_pltcalc_P1 > output/gul_S1_pltcalc.csv & kpid2=$!
kat work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid3=$!
wait $kpid1 $kpid2 $kpid3


( ordleccalc -r -Kri_S1_summaryleccalc -F -f -S -s -M -m -W -w -O output/ri_S1_ept.csv -o output/ri_S1_psept.csv ) 2>> $LOG_DIR/stderror.err & lpid1=$!
( ordleccalc  -Kil_S1_summaryleccalc -F -S -s -M -m -O output/il_S1_ept.csv ) 2>> $LOG_DIR/stderror.err & lpid2=$!
( aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid3=$!
wait $lpid1 $lpid2 $lpid3

rm -R -f work/*
rm -R -f fifo/*

check_complete
