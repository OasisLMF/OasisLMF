#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


touch $LOG_DIR/stderror.err
oasis_exec_monitor.sh $$ $LOG_DIR & pid0=$!

exit_handler(){
   exit_code=$?

   # disable handler
   trap - QUIT HUP INT KILL TERM ERR EXIT

   kill -9 $pid0 2> /dev/null
   if [ "$exit_code" -gt 0 ]; then
       # Error - run process clean up
       echo 'Kernel execution error - exitcode='$exit_code

       set +x
       group_pid=$(ps -p $$ -o pgid --no-headers)
       sess_pid=$(ps -p $$ -o sess --no-headers)
       script_pid=$$
       printf "Script PID:%d, GPID:%s, SPID:%d
" $script_pid $group_pid $sess_pid >> $LOG_DIR/killout.txt

       ps -jf f -g $sess_pid > $LOG_DIR/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk 'BEGIN { FS = "[ \t\n]+" }{ if ($1 >= '$script_pid') print}' | grep -v celery | egrep -v *\\.log$  | egrep -v *startup.sh$ | sort -n -r)
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
    proc_list="evepy modelpy gulpy fmpy gulmc summarypy plapy katpy eltpy pltpy aalpy lecpy"
    has_error=0
    for p in $proc_list; do
        started=$(find log -name "${p}_[0-9]*.log" | wc -l)
        finished=$(find log -name "${p}_[0-9]*.log" -exec grep -l "finish" {} + | wc -l)
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

rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S2_summaryleccalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S2_summaryleccalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P8

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_P8

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8.idx



# --- Do insured loss computes ---


tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid2=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1 work/il_S2_summaryleccalc/P1.bin > /dev/null & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1.idx work/il_S2_summaryleccalc/P1.idx > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 work/il_S1_summaryleccalc/P2.bin > /dev/null & pid5=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid6=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2 work/il_S2_summaryleccalc/P2.bin > /dev/null & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2.idx work/il_S2_summaryleccalc/P2.idx > /dev/null & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 work/il_S1_summaryleccalc/P3.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid10=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3 work/il_S2_summaryleccalc/P3.bin > /dev/null & pid11=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3.idx work/il_S2_summaryleccalc/P3.idx > /dev/null & pid12=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 work/il_S1_summaryleccalc/P4.bin > /dev/null & pid13=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4 work/il_S2_summaryleccalc/P4.bin > /dev/null & pid15=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4.idx work/il_S2_summaryleccalc/P4.idx > /dev/null & pid16=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 work/il_S1_summaryleccalc/P5.bin > /dev/null & pid17=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid18=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5 work/il_S2_summaryleccalc/P5.bin > /dev/null & pid19=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5.idx work/il_S2_summaryleccalc/P5.idx > /dev/null & pid20=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 work/il_S1_summaryleccalc/P6.bin > /dev/null & pid21=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid22=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6 work/il_S2_summaryleccalc/P6.bin > /dev/null & pid23=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6.idx work/il_S2_summaryleccalc/P6.idx > /dev/null & pid24=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 work/il_S1_summaryleccalc/P7.bin > /dev/null & pid25=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid26=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7 work/il_S2_summaryleccalc/P7.bin > /dev/null & pid27=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7.idx work/il_S2_summaryleccalc/P7.idx > /dev/null & pid28=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 work/il_S1_summaryleccalc/P8.bin > /dev/null & pid29=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid30=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8 work/il_S2_summaryleccalc/P8.bin > /dev/null & pid31=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8.idx work/il_S2_summaryleccalc/P8.idx > /dev/null & pid32=$!

( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1 < /tmp/%FIFO_DIR%/fifo/il_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2 < /tmp/%FIFO_DIR%/fifo/il_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3 < /tmp/%FIFO_DIR%/fifo/il_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4 < /tmp/%FIFO_DIR%/fifo/il_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5 < /tmp/%FIFO_DIR%/fifo/il_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6 < /tmp/%FIFO_DIR%/fifo/il_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/il_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8 < /tmp/%FIFO_DIR%/fifo/il_P8 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---


tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid33=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid34=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1 work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid35=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1.idx work/gul_S2_summaryleccalc/P1.idx > /dev/null & pid36=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid37=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid38=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2 work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid39=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2.idx work/gul_S2_summaryleccalc/P2.idx > /dev/null & pid40=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid41=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid42=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3 work/gul_S2_summaryleccalc/P3.bin > /dev/null & pid43=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3.idx work/gul_S2_summaryleccalc/P3.idx > /dev/null & pid44=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid45=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid46=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4 work/gul_S2_summaryleccalc/P4.bin > /dev/null & pid47=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4.idx work/gul_S2_summaryleccalc/P4.idx > /dev/null & pid48=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid49=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid50=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5 work/gul_S2_summaryleccalc/P5.bin > /dev/null & pid51=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5.idx work/gul_S2_summaryleccalc/P5.idx > /dev/null & pid52=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid53=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6.idx work/gul_S1_summaryleccalc/P6.idx > /dev/null & pid54=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6 work/gul_S2_summaryleccalc/P6.bin > /dev/null & pid55=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6.idx work/gul_S2_summaryleccalc/P6.idx > /dev/null & pid56=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid57=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid58=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7 work/gul_S2_summaryleccalc/P7.bin > /dev/null & pid59=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7.idx work/gul_S2_summaryleccalc/P7.idx > /dev/null & pid60=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid61=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid62=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8 work/gul_S2_summaryleccalc/P8.bin > /dev/null & pid63=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8.idx work/gul_S2_summaryleccalc/P8.idx > /dev/null & pid64=$!

( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2 < /tmp/%FIFO_DIR%/fifo/gul_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3 < /tmp/%FIFO_DIR%/fifo/gul_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4 < /tmp/%FIFO_DIR%/fifo/gul_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5 < /tmp/%FIFO_DIR%/fifo/gul_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6 < /tmp/%FIFO_DIR%/fifo/gul_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/gul_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8 < /tmp/%FIFO_DIR%/fifo/gul_P8 ) 2>> $LOG_DIR/stderror.err  &

( ( evepy 1 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P1 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  ) 2>> $LOG_DIR/stderror.err ) & pid65=$!
( ( evepy 2 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P2 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P2  ) 2>> $LOG_DIR/stderror.err ) & pid66=$!
( ( evepy 3 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P3 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P3  ) 2>> $LOG_DIR/stderror.err ) & pid67=$!
( ( evepy 4 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P4 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P4  ) 2>> $LOG_DIR/stderror.err ) & pid68=$!
( ( evepy 5 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P5 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P5  ) 2>> $LOG_DIR/stderror.err ) & pid69=$!
( ( evepy 6 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P6 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P6  ) 2>> $LOG_DIR/stderror.err ) & pid70=$!
( ( evepy 7 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P7 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  ) 2>> $LOG_DIR/stderror.err ) & pid71=$!
( ( evepy 8 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P8 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P8  ) 2>> $LOG_DIR/stderror.err ) & pid72=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72


# --- Do insured loss kats ---


# --- Do ground up loss kats ---


( lecpy -r -Kil_S1_summaryleccalc -F -S -s -M -m -W -w -O output/il_S1_ept.csv -o output/il_S1_psept.csv ) 2>> $LOG_DIR/stderror.err & lpid1=$!
( lecpy -r -Kil_S2_summaryleccalc -F -S -s -M -m -W -w -O output/il_S2_ept.csv -o output/il_S2_psept.csv ) 2>> $LOG_DIR/stderror.err & lpid2=$!
( lecpy -r -Kgul_S1_summaryleccalc -F -S -s -M -m -W -w -O output/gul_S1_ept.csv -o output/gul_S1_psept.csv ) 2>> $LOG_DIR/stderror.err & lpid3=$!
( lecpy -r -Kgul_S2_summaryleccalc -F -S -s -M -m -W -w -O output/gul_S2_ept.csv -o output/gul_S2_psept.csv ) 2>> $LOG_DIR/stderror.err & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/

check_complete
