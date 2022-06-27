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

find fifo/ \( -name '*P27[^0-9]*' -o -name '*P27' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc

mkfifo fifo/gul_P27

mkfifo fifo/gul_S1_summary_P27
mkfifo fifo/gul_S1_summary_P27.idx
mkfifo fifo/gul_S1_eltcalc_P27
mkfifo fifo/gul_S1_summarycalc_P27
mkfifo fifo/gul_S1_pltcalc_P27

mkfifo fifo/il_P27

mkfifo fifo/il_S1_summary_P27
mkfifo fifo/il_S1_summary_P27.idx
mkfifo fifo/il_S1_eltcalc_P27
mkfifo fifo/il_S1_summarycalc_P27
mkfifo fifo/il_S1_pltcalc_P27



# --- Do insured loss computes ---
( eltcalc -s < fifo/il_S1_eltcalc_P27 > work/kat/il_S1_eltcalc_P27 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P27 > work/kat/il_S1_summarycalc_P27 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( pltcalc -H < fifo/il_S1_pltcalc_P27 > work/kat/il_S1_pltcalc_P27 ) 2>> $LOG_DIR/stderror.err & pid3=$!
tee < fifo/il_S1_summary_P27 fifo/il_S1_eltcalc_P27 fifo/il_S1_summarycalc_P27 fifo/il_S1_pltcalc_P27 work/il_S1_summaryaalcalc/P27.bin work/il_S1_summaryleccalc/P27.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P27.idx work/il_S1_summaryaalcalc/P27.idx work/il_S1_summaryleccalc/P27.idx > /dev/null & pid5=$!
( summarycalc -m -f  -1 fifo/il_S1_summary_P27 < fifo/il_P27 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---
( eltcalc -s < fifo/gul_S1_eltcalc_P27 > work/kat/gul_S1_eltcalc_P27 ) 2>> $LOG_DIR/stderror.err & pid6=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P27 > work/kat/gul_S1_summarycalc_P27 ) 2>> $LOG_DIR/stderror.err & pid7=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P27 > work/kat/gul_S1_pltcalc_P27 ) 2>> $LOG_DIR/stderror.err & pid8=$!
tee < fifo/gul_S1_summary_P27 fifo/gul_S1_eltcalc_P27 fifo/gul_S1_summarycalc_P27 fifo/gul_S1_pltcalc_P27 work/gul_S1_summaryaalcalc/P27.bin work/gul_S1_summaryleccalc/P27.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P27.idx work/gul_S1_summaryaalcalc/P27.idx work/gul_S1_summaryleccalc/P27.idx > /dev/null & pid10=$!
( summarycalc -m -i  -1 fifo/gul_S1_summary_P27 < fifo/gul_P27 ) 2>> $LOG_DIR/stderror.err  &

( eve 27 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P27 | fmcalc -a2 > fifo/il_P27  ) 2>> $LOG_DIR/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10


check_complete
