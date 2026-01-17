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
mkdir -p work/il_S3_summary_palt
mkdir -p work/il_S3_summary_altmeanonly

mkfifo /tmp/%FIFO_DIR%/fifo/il_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_P8

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P1.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P2.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P3.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P4.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P5.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P6.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P7.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S3_summary_P8.idx



# --- Do insured loss computes ---


( eltpy -E bin  -q work/kat/il_S1_elt_quantile_P1 -m work/kat/il_S1_elt_moment_P1 < /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( eltpy -E bin  -s work/kat/il_S1_elt_sample_P1 < /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( pltpy -E bin  -s work/kat/il_S2_plt_sample_P1 -q work/kat/il_S2_plt_quantile_P1 -m work/kat/il_S2_plt_moment_P1 < /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid3=$!
( eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P2 -m work/kat/il_S1_elt_moment_P2 < /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid4=$!
( eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P2 < /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid5=$!
( pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P2 -q work/kat/il_S2_plt_quantile_P2 -m work/kat/il_S2_plt_moment_P2 < /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid6=$!
( eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P3 -m work/kat/il_S1_elt_moment_P3 < /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P3 ) 2>> $LOG_DIR/stderror.err & pid7=$!
( eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P3 < /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P3 ) 2>> $LOG_DIR/stderror.err & pid8=$!
( pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P3 -q work/kat/il_S2_plt_quantile_P3 -m work/kat/il_S2_plt_moment_P3 < /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P3 ) 2>> $LOG_DIR/stderror.err & pid9=$!
( eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P4 -m work/kat/il_S1_elt_moment_P4 < /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P4 ) 2>> $LOG_DIR/stderror.err & pid10=$!
( eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P4 < /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P4 ) 2>> $LOG_DIR/stderror.err & pid11=$!
( pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P4 -q work/kat/il_S2_plt_quantile_P4 -m work/kat/il_S2_plt_moment_P4 < /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P4 ) 2>> $LOG_DIR/stderror.err & pid12=$!
( eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P5 -m work/kat/il_S1_elt_moment_P5 < /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P5 ) 2>> $LOG_DIR/stderror.err & pid13=$!
( eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P5 < /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P5 ) 2>> $LOG_DIR/stderror.err & pid14=$!
( pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P5 -q work/kat/il_S2_plt_quantile_P5 -m work/kat/il_S2_plt_moment_P5 < /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P5 ) 2>> $LOG_DIR/stderror.err & pid15=$!
( eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P6 -m work/kat/il_S1_elt_moment_P6 < /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P6 ) 2>> $LOG_DIR/stderror.err & pid16=$!
( eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P6 < /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P6 ) 2>> $LOG_DIR/stderror.err & pid17=$!
( pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P6 -q work/kat/il_S2_plt_quantile_P6 -m work/kat/il_S2_plt_moment_P6 < /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P6 ) 2>> $LOG_DIR/stderror.err & pid18=$!
( eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P7 -m work/kat/il_S1_elt_moment_P7 < /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P7 ) 2>> $LOG_DIR/stderror.err & pid19=$!
( eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P7 < /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P7 ) 2>> $LOG_DIR/stderror.err & pid20=$!
( pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P7 -q work/kat/il_S2_plt_quantile_P7 -m work/kat/il_S2_plt_moment_P7 < /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P7 ) 2>> $LOG_DIR/stderror.err & pid21=$!
( eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P8 -m work/kat/il_S1_elt_moment_P8 < /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P8 ) 2>> $LOG_DIR/stderror.err & pid22=$!
( eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P8 < /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P8 ) 2>> $LOG_DIR/stderror.err & pid23=$!
( pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P8 -q work/kat/il_S2_plt_quantile_P8 -m work/kat/il_S2_plt_moment_P8 < /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P8 ) 2>> $LOG_DIR/stderror.err & pid24=$!

tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P1 /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P1 > /dev/null & pid25=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1 /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P1 > /dev/null & pid26=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P1 work/il_S3_summary_palt/P1.bin work/il_S3_summary_altmeanonly/P1.bin > /dev/null & pid27=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P1.idx work/il_S3_summary_palt/P1.idx > /dev/null & pid28=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P2 /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P2 > /dev/null & pid29=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2 /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P2 > /dev/null & pid30=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P2 work/il_S3_summary_palt/P2.bin work/il_S3_summary_altmeanonly/P2.bin > /dev/null & pid31=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P2.idx work/il_S3_summary_palt/P2.idx > /dev/null & pid32=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P3 /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P3 > /dev/null & pid33=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3 /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P3 > /dev/null & pid34=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P3 work/il_S3_summary_palt/P3.bin work/il_S3_summary_altmeanonly/P3.bin > /dev/null & pid35=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P3.idx work/il_S3_summary_palt/P3.idx > /dev/null & pid36=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P4 /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P4 > /dev/null & pid37=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4 /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P4 > /dev/null & pid38=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P4 work/il_S3_summary_palt/P4.bin work/il_S3_summary_altmeanonly/P4.bin > /dev/null & pid39=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P4.idx work/il_S3_summary_palt/P4.idx > /dev/null & pid40=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P5 /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P5 > /dev/null & pid41=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5 /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P5 > /dev/null & pid42=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P5 work/il_S3_summary_palt/P5.bin work/il_S3_summary_altmeanonly/P5.bin > /dev/null & pid43=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P5.idx work/il_S3_summary_palt/P5.idx > /dev/null & pid44=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P6 /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P6 > /dev/null & pid45=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6 /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P6 > /dev/null & pid46=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P6 work/il_S3_summary_palt/P6.bin work/il_S3_summary_altmeanonly/P6.bin > /dev/null & pid47=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P6.idx work/il_S3_summary_palt/P6.idx > /dev/null & pid48=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P7 /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P7 > /dev/null & pid49=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7 /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P7 > /dev/null & pid50=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P7 work/il_S3_summary_palt/P7.bin work/il_S3_summary_altmeanonly/P7.bin > /dev/null & pid51=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P7.idx work/il_S3_summary_palt/P7.idx > /dev/null & pid52=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P8 /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P8 > /dev/null & pid53=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8 /tmp/%FIFO_DIR%/fifo/il_S2_plt_ord_P8 > /dev/null & pid54=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P8 work/il_S3_summary_palt/P8.bin work/il_S3_summary_altmeanonly/P8.bin > /dev/null & pid55=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S3_summary_P8.idx work/il_S3_summary_palt/P8.idx > /dev/null & pid56=$!

( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1 -3 /tmp/%FIFO_DIR%/fifo/il_S3_summary_P1 < /tmp/%FIFO_DIR%/fifo/il_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2 -3 /tmp/%FIFO_DIR%/fifo/il_S3_summary_P2 < /tmp/%FIFO_DIR%/fifo/il_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3 -3 /tmp/%FIFO_DIR%/fifo/il_S3_summary_P3 < /tmp/%FIFO_DIR%/fifo/il_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4 -3 /tmp/%FIFO_DIR%/fifo/il_S3_summary_P4 < /tmp/%FIFO_DIR%/fifo/il_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5 -3 /tmp/%FIFO_DIR%/fifo/il_S3_summary_P5 < /tmp/%FIFO_DIR%/fifo/il_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6 -3 /tmp/%FIFO_DIR%/fifo/il_S3_summary_P6 < /tmp/%FIFO_DIR%/fifo/il_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7 -3 /tmp/%FIFO_DIR%/fifo/il_S3_summary_P7 < /tmp/%FIFO_DIR%/fifo/il_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8 -3 /tmp/%FIFO_DIR%/fifo/il_S3_summary_P8 < /tmp/%FIFO_DIR%/fifo/il_P8 ) 2>> $LOG_DIR/stderror.err  &

( ( evepy 1 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  ) 2>> $LOG_DIR/stderror.err ) & pid57=$!
( ( evepy 2 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P2  ) 2>> $LOG_DIR/stderror.err ) & pid58=$!
( ( evepy 3 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P3  ) 2>> $LOG_DIR/stderror.err ) & pid59=$!
( ( evepy 4 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P4  ) 2>> $LOG_DIR/stderror.err ) & pid60=$!
( ( evepy 5 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P5  ) 2>> $LOG_DIR/stderror.err ) & pid61=$!
( ( evepy 6 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P6  ) 2>> $LOG_DIR/stderror.err ) & pid62=$!
( ( evepy 7 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  ) 2>> $LOG_DIR/stderror.err ) & pid63=$!
( ( evepy 8 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P8  ) 2>> $LOG_DIR/stderror.err ) & pid64=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64


# --- Do insured loss kats ---

katpy -q -f bin -i work/kat/il_S1_elt_quantile_P1 work/kat/il_S1_elt_quantile_P2 work/kat/il_S1_elt_quantile_P3 work/kat/il_S1_elt_quantile_P4 work/kat/il_S1_elt_quantile_P5 work/kat/il_S1_elt_quantile_P6 work/kat/il_S1_elt_quantile_P7 work/kat/il_S1_elt_quantile_P8 -o output/il_S1_qelt.csv & kpid1=$!
katpy -m -f bin -i work/kat/il_S1_elt_moment_P1 work/kat/il_S1_elt_moment_P2 work/kat/il_S1_elt_moment_P3 work/kat/il_S1_elt_moment_P4 work/kat/il_S1_elt_moment_P5 work/kat/il_S1_elt_moment_P6 work/kat/il_S1_elt_moment_P7 work/kat/il_S1_elt_moment_P8 -o output/il_S1_melt.csv & kpid2=$!
katpy -s -f bin -i work/kat/il_S1_elt_sample_P1 work/kat/il_S1_elt_sample_P2 work/kat/il_S1_elt_sample_P3 work/kat/il_S1_elt_sample_P4 work/kat/il_S1_elt_sample_P5 work/kat/il_S1_elt_sample_P6 work/kat/il_S1_elt_sample_P7 work/kat/il_S1_elt_sample_P8 -o output/il_S1_selt.csv & kpid3=$!
katpy -S -f bin -i work/kat/il_S2_plt_sample_P1 work/kat/il_S2_plt_sample_P2 work/kat/il_S2_plt_sample_P3 work/kat/il_S2_plt_sample_P4 work/kat/il_S2_plt_sample_P5 work/kat/il_S2_plt_sample_P6 work/kat/il_S2_plt_sample_P7 work/kat/il_S2_plt_sample_P8 -o output/il_S2_splt.csv & kpid4=$!
katpy -Q -f bin -i work/kat/il_S2_plt_quantile_P1 work/kat/il_S2_plt_quantile_P2 work/kat/il_S2_plt_quantile_P3 work/kat/il_S2_plt_quantile_P4 work/kat/il_S2_plt_quantile_P5 work/kat/il_S2_plt_quantile_P6 work/kat/il_S2_plt_quantile_P7 work/kat/il_S2_plt_quantile_P8 -o output/il_S2_qplt.csv & kpid5=$!
katpy -M -f bin -i work/kat/il_S2_plt_moment_P1 work/kat/il_S2_plt_moment_P2 work/kat/il_S2_plt_moment_P3 work/kat/il_S2_plt_moment_P4 work/kat/il_S2_plt_moment_P5 work/kat/il_S2_plt_moment_P6 work/kat/il_S2_plt_moment_P7 work/kat/il_S2_plt_moment_P8 -o output/il_S2_mplt.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6


( aalpy -Kil_S3_summary_palt -c output/il_S3_alct.csv -l 0.95 -a output/il_S3_palt.csv ) 2>> $LOG_DIR/stderror.err & lpid1=$!
( aalpy -Kil_S3_summary_altmeanonly -a output/il_S3_altmeanonly.csv ) 2>> $LOG_DIR/stderror.err & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/

check_complete
