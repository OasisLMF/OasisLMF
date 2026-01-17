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
    proc_list="eve evepy getmodel gulcalc fmcalc summarycalc eltcalc aalcalc aalcalcmeanonly leccalc pltcalc ordleccalc modelpy gulpy fmpy gulmc summarypy eltpy pltpy aalpy lecpy"
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

    started=$( grep "Starting custom gulcalc command" log/gul_stderror.err | wc -l)
    finished=$( grep "Custom gulcalc command finished" log/gul_stderror.err | wc -l)
    if [ "$finished" -lt "$started" ]; then
        echo "[ERROR] gulcalc - $((started-finished)) processes lost"
        has_error=1
    elif [ "$started" -gt 0 ]; then
        echo "[OK] gulcalc"
    fi
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

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P8

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P2

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P3

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P4

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P5

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P6

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P7

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P8



# --- Do ground up loss computes ---


( eltpy -E bin  -s work/kat/gul_S1_elt_sample_P1 < /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P2 < /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P3 < /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P3 ) 2>> $LOG_DIR/stderror.err & pid3=$!
( eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P4 < /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P4 ) 2>> $LOG_DIR/stderror.err & pid4=$!
( eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P5 < /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P5 ) 2>> $LOG_DIR/stderror.err & pid5=$!
( eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P6 < /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P6 ) 2>> $LOG_DIR/stderror.err & pid6=$!
( eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P7 < /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P7 ) 2>> $LOG_DIR/stderror.err & pid7=$!
( eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P8 < /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P8 ) 2>> $LOG_DIR/stderror.err & pid8=$!

tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P1 > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P2 > /dev/null & pid10=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P3 > /dev/null & pid11=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P4 > /dev/null & pid12=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P5 > /dev/null & pid13=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P6 > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P7 > /dev/null & pid15=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P8 > /dev/null & pid16=$!

( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/gul_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/gul_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/gul_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/gul_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/gul_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/gul_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/gul_P8 ) 2>> $LOG_DIR/stderror.err  &

( ( evepy 1 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  > /tmp/%FIFO_DIR%/fifo/gul_P1  ) 2>> $LOG_DIR/stderror.err ) &  pid17=$!
( ( evepy 2 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  > /tmp/%FIFO_DIR%/fifo/gul_P2  ) 2>> $LOG_DIR/stderror.err ) &  pid18=$!
( ( evepy 3 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  > /tmp/%FIFO_DIR%/fifo/gul_P3  ) 2>> $LOG_DIR/stderror.err ) &  pid19=$!
( ( evepy 4 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  > /tmp/%FIFO_DIR%/fifo/gul_P4  ) 2>> $LOG_DIR/stderror.err ) &  pid20=$!
( ( evepy 5 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  > /tmp/%FIFO_DIR%/fifo/gul_P5  ) 2>> $LOG_DIR/stderror.err ) &  pid21=$!
( ( evepy 6 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  > /tmp/%FIFO_DIR%/fifo/gul_P6  ) 2>> $LOG_DIR/stderror.err ) &  pid22=$!
( ( evepy 7 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  > /tmp/%FIFO_DIR%/fifo/gul_P7  ) 2>> $LOG_DIR/stderror.err ) &  pid23=$!
( ( evepy 8 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  > /tmp/%FIFO_DIR%/fifo/gul_P8  ) 2>> $LOG_DIR/stderror.err ) &  pid24=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24


# --- Do ground up loss kats ---

katpy -s -f bin -i work/kat/gul_S1_elt_sample_P1 work/kat/gul_S1_elt_sample_P2 work/kat/gul_S1_elt_sample_P3 work/kat/gul_S1_elt_sample_P4 work/kat/gul_S1_elt_sample_P5 work/kat/gul_S1_elt_sample_P6 work/kat/gul_S1_elt_sample_P7 work/kat/gul_S1_elt_sample_P8 -o output/gul_S1_selt.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/

check_complete
