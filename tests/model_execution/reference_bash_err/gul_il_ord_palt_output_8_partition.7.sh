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

find /tmp/%FIFO_DIR%/fifo/ \( -name '*P8[^0-9]*' -o -name '*P8' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summary_palt
mkdir -p work/gul_S2_summaryleccalc
mkdir -p work/gul_S2_summary_palt
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S2_summaryleccalc
mkdir -p work/il_S2_summary_palt

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P8

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_P8

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8.idx



# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 work/il_S1_summaryleccalc/P8.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid2=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8 work/il_S2_summary_palt/P8.bin work/il_S2_summaryleccalc/P8.bin > /dev/null & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8.idx work/il_S2_summary_palt/P8.idx work/il_S2_summaryleccalc/P8.idx > /dev/null & pid4=$!
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8 < /tmp/%FIFO_DIR%/fifo/il_P8 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 work/gul_S1_summary_palt/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid5=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx work/gul_S1_summary_palt/P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid6=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8 work/gul_S2_summary_palt/P8.bin work/gul_S2_summaryleccalc/P8.bin > /dev/null & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8.idx work/gul_S2_summary_palt/P8.idx work/gul_S2_summaryleccalc/P8.idx > /dev/null & pid8=$!
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8 < /tmp/%FIFO_DIR%/fifo/gul_P8 ) 2>> $LOG_DIR/stderror.err  &

( ( evepy 8 8 | gulmc --socket-server='False' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P8 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P8  ) 2>> $LOG_DIR/stderror.err ) & pid9=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9


check_complete
