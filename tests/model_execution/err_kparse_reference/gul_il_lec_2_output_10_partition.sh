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
    proc_list="eve getmodel gulcalc fmcalc summarycalc eltcalc aalcalc aalcalcmeanonly leccalc pltcalc ordleccalc modelpy gulpy fmpy gulmc summarypy eltpy pltpy aalpy lecpy"
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

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/gul_S2_summaryleccalc
mkdir -p work/gul_S2_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/il_S2_summaryleccalc
mkdir -p work/il_S2_summaryaalcalc

mkfifo fifo/gul_P1
mkfifo fifo/gul_P2
mkfifo fifo/gul_P3
mkfifo fifo/gul_P4
mkfifo fifo/gul_P5
mkfifo fifo/gul_P6
mkfifo fifo/gul_P7
mkfifo fifo/gul_P8
mkfifo fifo/gul_P9
mkfifo fifo/gul_P10

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1
mkfifo fifo/gul_S2_summary_P1
mkfifo fifo/gul_S2_summary_P1.idx
mkfifo fifo/gul_S2_eltcalc_P1
mkfifo fifo/gul_S2_summarycalc_P1
mkfifo fifo/gul_S2_pltcalc_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2
mkfifo fifo/gul_S2_summary_P2
mkfifo fifo/gul_S2_summary_P2.idx
mkfifo fifo/gul_S2_eltcalc_P2
mkfifo fifo/gul_S2_summarycalc_P2
mkfifo fifo/gul_S2_pltcalc_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summary_P3.idx
mkfifo fifo/gul_S1_eltcalc_P3
mkfifo fifo/gul_S1_summarycalc_P3
mkfifo fifo/gul_S1_pltcalc_P3
mkfifo fifo/gul_S2_summary_P3
mkfifo fifo/gul_S2_summary_P3.idx
mkfifo fifo/gul_S2_eltcalc_P3
mkfifo fifo/gul_S2_summarycalc_P3
mkfifo fifo/gul_S2_pltcalc_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summary_P4.idx
mkfifo fifo/gul_S1_eltcalc_P4
mkfifo fifo/gul_S1_summarycalc_P4
mkfifo fifo/gul_S1_pltcalc_P4
mkfifo fifo/gul_S2_summary_P4
mkfifo fifo/gul_S2_summary_P4.idx
mkfifo fifo/gul_S2_eltcalc_P4
mkfifo fifo/gul_S2_summarycalc_P4
mkfifo fifo/gul_S2_pltcalc_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summary_P5.idx
mkfifo fifo/gul_S1_eltcalc_P5
mkfifo fifo/gul_S1_summarycalc_P5
mkfifo fifo/gul_S1_pltcalc_P5
mkfifo fifo/gul_S2_summary_P5
mkfifo fifo/gul_S2_summary_P5.idx
mkfifo fifo/gul_S2_eltcalc_P5
mkfifo fifo/gul_S2_summarycalc_P5
mkfifo fifo/gul_S2_pltcalc_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summary_P6.idx
mkfifo fifo/gul_S1_eltcalc_P6
mkfifo fifo/gul_S1_summarycalc_P6
mkfifo fifo/gul_S1_pltcalc_P6
mkfifo fifo/gul_S2_summary_P6
mkfifo fifo/gul_S2_summary_P6.idx
mkfifo fifo/gul_S2_eltcalc_P6
mkfifo fifo/gul_S2_summarycalc_P6
mkfifo fifo/gul_S2_pltcalc_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summary_P7.idx
mkfifo fifo/gul_S1_eltcalc_P7
mkfifo fifo/gul_S1_summarycalc_P7
mkfifo fifo/gul_S1_pltcalc_P7
mkfifo fifo/gul_S2_summary_P7
mkfifo fifo/gul_S2_summary_P7.idx
mkfifo fifo/gul_S2_eltcalc_P7
mkfifo fifo/gul_S2_summarycalc_P7
mkfifo fifo/gul_S2_pltcalc_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_summary_P8.idx
mkfifo fifo/gul_S1_eltcalc_P8
mkfifo fifo/gul_S1_summarycalc_P8
mkfifo fifo/gul_S1_pltcalc_P8
mkfifo fifo/gul_S2_summary_P8
mkfifo fifo/gul_S2_summary_P8.idx
mkfifo fifo/gul_S2_eltcalc_P8
mkfifo fifo/gul_S2_summarycalc_P8
mkfifo fifo/gul_S2_pltcalc_P8

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_summary_P9.idx
mkfifo fifo/gul_S1_eltcalc_P9
mkfifo fifo/gul_S1_summarycalc_P9
mkfifo fifo/gul_S1_pltcalc_P9
mkfifo fifo/gul_S2_summary_P9
mkfifo fifo/gul_S2_summary_P9.idx
mkfifo fifo/gul_S2_eltcalc_P9
mkfifo fifo/gul_S2_summarycalc_P9
mkfifo fifo/gul_S2_pltcalc_P9

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_summary_P10.idx
mkfifo fifo/gul_S1_eltcalc_P10
mkfifo fifo/gul_S1_summarycalc_P10
mkfifo fifo/gul_S1_pltcalc_P10
mkfifo fifo/gul_S2_summary_P10
mkfifo fifo/gul_S2_summary_P10.idx
mkfifo fifo/gul_S2_eltcalc_P10
mkfifo fifo/gul_S2_summarycalc_P10
mkfifo fifo/gul_S2_pltcalc_P10

mkfifo fifo/il_P1
mkfifo fifo/il_P2
mkfifo fifo/il_P3
mkfifo fifo/il_P4
mkfifo fifo/il_P5
mkfifo fifo/il_P6
mkfifo fifo/il_P7
mkfifo fifo/il_P8
mkfifo fifo/il_P9
mkfifo fifo/il_P10

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P1.idx
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_pltcalc_P1
mkfifo fifo/il_S2_summary_P1
mkfifo fifo/il_S2_summary_P1.idx
mkfifo fifo/il_S2_eltcalc_P1
mkfifo fifo/il_S2_summarycalc_P1
mkfifo fifo/il_S2_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_pltcalc_P2
mkfifo fifo/il_S2_summary_P2
mkfifo fifo/il_S2_summary_P2.idx
mkfifo fifo/il_S2_eltcalc_P2
mkfifo fifo/il_S2_summarycalc_P2
mkfifo fifo/il_S2_pltcalc_P2

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summary_P3.idx
mkfifo fifo/il_S1_eltcalc_P3
mkfifo fifo/il_S1_summarycalc_P3
mkfifo fifo/il_S1_pltcalc_P3
mkfifo fifo/il_S2_summary_P3
mkfifo fifo/il_S2_summary_P3.idx
mkfifo fifo/il_S2_eltcalc_P3
mkfifo fifo/il_S2_summarycalc_P3
mkfifo fifo/il_S2_pltcalc_P3

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summary_P4.idx
mkfifo fifo/il_S1_eltcalc_P4
mkfifo fifo/il_S1_summarycalc_P4
mkfifo fifo/il_S1_pltcalc_P4
mkfifo fifo/il_S2_summary_P4
mkfifo fifo/il_S2_summary_P4.idx
mkfifo fifo/il_S2_eltcalc_P4
mkfifo fifo/il_S2_summarycalc_P4
mkfifo fifo/il_S2_pltcalc_P4

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summary_P5.idx
mkfifo fifo/il_S1_eltcalc_P5
mkfifo fifo/il_S1_summarycalc_P5
mkfifo fifo/il_S1_pltcalc_P5
mkfifo fifo/il_S2_summary_P5
mkfifo fifo/il_S2_summary_P5.idx
mkfifo fifo/il_S2_eltcalc_P5
mkfifo fifo/il_S2_summarycalc_P5
mkfifo fifo/il_S2_pltcalc_P5

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summary_P6.idx
mkfifo fifo/il_S1_eltcalc_P6
mkfifo fifo/il_S1_summarycalc_P6
mkfifo fifo/il_S1_pltcalc_P6
mkfifo fifo/il_S2_summary_P6
mkfifo fifo/il_S2_summary_P6.idx
mkfifo fifo/il_S2_eltcalc_P6
mkfifo fifo/il_S2_summarycalc_P6
mkfifo fifo/il_S2_pltcalc_P6

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summary_P7.idx
mkfifo fifo/il_S1_eltcalc_P7
mkfifo fifo/il_S1_summarycalc_P7
mkfifo fifo/il_S1_pltcalc_P7
mkfifo fifo/il_S2_summary_P7
mkfifo fifo/il_S2_summary_P7.idx
mkfifo fifo/il_S2_eltcalc_P7
mkfifo fifo/il_S2_summarycalc_P7
mkfifo fifo/il_S2_pltcalc_P7

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_summary_P8.idx
mkfifo fifo/il_S1_eltcalc_P8
mkfifo fifo/il_S1_summarycalc_P8
mkfifo fifo/il_S1_pltcalc_P8
mkfifo fifo/il_S2_summary_P8
mkfifo fifo/il_S2_summary_P8.idx
mkfifo fifo/il_S2_eltcalc_P8
mkfifo fifo/il_S2_summarycalc_P8
mkfifo fifo/il_S2_pltcalc_P8

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_summary_P9.idx
mkfifo fifo/il_S1_eltcalc_P9
mkfifo fifo/il_S1_summarycalc_P9
mkfifo fifo/il_S1_pltcalc_P9
mkfifo fifo/il_S2_summary_P9
mkfifo fifo/il_S2_summary_P9.idx
mkfifo fifo/il_S2_eltcalc_P9
mkfifo fifo/il_S2_summarycalc_P9
mkfifo fifo/il_S2_pltcalc_P9

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_summary_P10.idx
mkfifo fifo/il_S1_eltcalc_P10
mkfifo fifo/il_S1_summarycalc_P10
mkfifo fifo/il_S1_pltcalc_P10
mkfifo fifo/il_S2_summary_P10
mkfifo fifo/il_S2_summary_P10.idx
mkfifo fifo/il_S2_eltcalc_P10
mkfifo fifo/il_S2_summarycalc_P10
mkfifo fifo/il_S2_pltcalc_P10



# --- Do insured loss computes ---

( eltcalc < fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid3=$!
( eltcalc < fifo/il_S2_eltcalc_P1 > work/kat/il_S2_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid4=$!
( summarycalctocsv < fifo/il_S2_summarycalc_P1 > work/kat/il_S2_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid5=$!
( pltcalc < fifo/il_S2_pltcalc_P1 > work/kat/il_S2_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid6=$!
( eltcalc -s < fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid7=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 ) 2>> $LOG_DIR/stderror.err & pid8=$!
( pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid9=$!
( eltcalc -s < fifo/il_S2_eltcalc_P2 > work/kat/il_S2_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid10=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P2 > work/kat/il_S2_summarycalc_P2 ) 2>> $LOG_DIR/stderror.err & pid11=$!
( pltcalc -H < fifo/il_S2_pltcalc_P2 > work/kat/il_S2_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid12=$!
( eltcalc -s < fifo/il_S1_eltcalc_P3 > work/kat/il_S1_eltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid13=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P3 > work/kat/il_S1_summarycalc_P3 ) 2>> $LOG_DIR/stderror.err & pid14=$!
( pltcalc -H < fifo/il_S1_pltcalc_P3 > work/kat/il_S1_pltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid15=$!
( eltcalc -s < fifo/il_S2_eltcalc_P3 > work/kat/il_S2_eltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid16=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P3 > work/kat/il_S2_summarycalc_P3 ) 2>> $LOG_DIR/stderror.err & pid17=$!
( pltcalc -H < fifo/il_S2_pltcalc_P3 > work/kat/il_S2_pltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid18=$!
( eltcalc -s < fifo/il_S1_eltcalc_P4 > work/kat/il_S1_eltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid19=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P4 > work/kat/il_S1_summarycalc_P4 ) 2>> $LOG_DIR/stderror.err & pid20=$!
( pltcalc -H < fifo/il_S1_pltcalc_P4 > work/kat/il_S1_pltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid21=$!
( eltcalc -s < fifo/il_S2_eltcalc_P4 > work/kat/il_S2_eltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid22=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P4 > work/kat/il_S2_summarycalc_P4 ) 2>> $LOG_DIR/stderror.err & pid23=$!
( pltcalc -H < fifo/il_S2_pltcalc_P4 > work/kat/il_S2_pltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid24=$!
( eltcalc -s < fifo/il_S1_eltcalc_P5 > work/kat/il_S1_eltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid25=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P5 > work/kat/il_S1_summarycalc_P5 ) 2>> $LOG_DIR/stderror.err & pid26=$!
( pltcalc -H < fifo/il_S1_pltcalc_P5 > work/kat/il_S1_pltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid27=$!
( eltcalc -s < fifo/il_S2_eltcalc_P5 > work/kat/il_S2_eltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid28=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P5 > work/kat/il_S2_summarycalc_P5 ) 2>> $LOG_DIR/stderror.err & pid29=$!
( pltcalc -H < fifo/il_S2_pltcalc_P5 > work/kat/il_S2_pltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid30=$!
( eltcalc -s < fifo/il_S1_eltcalc_P6 > work/kat/il_S1_eltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid31=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P6 > work/kat/il_S1_summarycalc_P6 ) 2>> $LOG_DIR/stderror.err & pid32=$!
( pltcalc -H < fifo/il_S1_pltcalc_P6 > work/kat/il_S1_pltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid33=$!
( eltcalc -s < fifo/il_S2_eltcalc_P6 > work/kat/il_S2_eltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid34=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P6 > work/kat/il_S2_summarycalc_P6 ) 2>> $LOG_DIR/stderror.err & pid35=$!
( pltcalc -H < fifo/il_S2_pltcalc_P6 > work/kat/il_S2_pltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid36=$!
( eltcalc -s < fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid37=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P7 > work/kat/il_S1_summarycalc_P7 ) 2>> $LOG_DIR/stderror.err & pid38=$!
( pltcalc -H < fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid39=$!
( eltcalc -s < fifo/il_S2_eltcalc_P7 > work/kat/il_S2_eltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid40=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P7 > work/kat/il_S2_summarycalc_P7 ) 2>> $LOG_DIR/stderror.err & pid41=$!
( pltcalc -H < fifo/il_S2_pltcalc_P7 > work/kat/il_S2_pltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid42=$!
( eltcalc -s < fifo/il_S1_eltcalc_P8 > work/kat/il_S1_eltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid43=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P8 > work/kat/il_S1_summarycalc_P8 ) 2>> $LOG_DIR/stderror.err & pid44=$!
( pltcalc -H < fifo/il_S1_pltcalc_P8 > work/kat/il_S1_pltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid45=$!
( eltcalc -s < fifo/il_S2_eltcalc_P8 > work/kat/il_S2_eltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid46=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P8 > work/kat/il_S2_summarycalc_P8 ) 2>> $LOG_DIR/stderror.err & pid47=$!
( pltcalc -H < fifo/il_S2_pltcalc_P8 > work/kat/il_S2_pltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid48=$!
( eltcalc -s < fifo/il_S1_eltcalc_P9 > work/kat/il_S1_eltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid49=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P9 > work/kat/il_S1_summarycalc_P9 ) 2>> $LOG_DIR/stderror.err & pid50=$!
( pltcalc -H < fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid51=$!
( eltcalc -s < fifo/il_S2_eltcalc_P9 > work/kat/il_S2_eltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid52=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P9 > work/kat/il_S2_summarycalc_P9 ) 2>> $LOG_DIR/stderror.err & pid53=$!
( pltcalc -H < fifo/il_S2_pltcalc_P9 > work/kat/il_S2_pltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid54=$!
( eltcalc -s < fifo/il_S1_eltcalc_P10 > work/kat/il_S1_eltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid55=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P10 > work/kat/il_S1_summarycalc_P10 ) 2>> $LOG_DIR/stderror.err & pid56=$!
( pltcalc -H < fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid57=$!
( eltcalc -s < fifo/il_S2_eltcalc_P10 > work/kat/il_S2_eltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid58=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P10 > work/kat/il_S2_summarycalc_P10 ) 2>> $LOG_DIR/stderror.err & pid59=$!
( pltcalc -H < fifo/il_S2_pltcalc_P10 > work/kat/il_S2_pltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid60=$!


tee < fifo/il_S1_summary_P1 fifo/il_S1_eltcalc_P1 fifo/il_S1_summarycalc_P1 fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid61=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summaryaalcalc/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid62=$!
tee < fifo/il_S2_summary_P1 fifo/il_S2_eltcalc_P1 fifo/il_S2_summarycalc_P1 fifo/il_S2_pltcalc_P1 work/il_S2_summaryaalcalc/P1.bin work/il_S2_summaryleccalc/P1.bin > /dev/null & pid63=$!
tee < fifo/il_S2_summary_P1.idx work/il_S2_summaryaalcalc/P1.idx work/il_S2_summaryleccalc/P1.idx > /dev/null & pid64=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_eltcalc_P2 fifo/il_S1_summarycalc_P2 fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid65=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summaryaalcalc/P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid66=$!
tee < fifo/il_S2_summary_P2 fifo/il_S2_eltcalc_P2 fifo/il_S2_summarycalc_P2 fifo/il_S2_pltcalc_P2 work/il_S2_summaryaalcalc/P2.bin work/il_S2_summaryleccalc/P2.bin > /dev/null & pid67=$!
tee < fifo/il_S2_summary_P2.idx work/il_S2_summaryaalcalc/P2.idx work/il_S2_summaryleccalc/P2.idx > /dev/null & pid68=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_eltcalc_P3 fifo/il_S1_summarycalc_P3 fifo/il_S1_pltcalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid69=$!
tee < fifo/il_S1_summary_P3.idx work/il_S1_summaryaalcalc/P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid70=$!
tee < fifo/il_S2_summary_P3 fifo/il_S2_eltcalc_P3 fifo/il_S2_summarycalc_P3 fifo/il_S2_pltcalc_P3 work/il_S2_summaryaalcalc/P3.bin work/il_S2_summaryleccalc/P3.bin > /dev/null & pid71=$!
tee < fifo/il_S2_summary_P3.idx work/il_S2_summaryaalcalc/P3.idx work/il_S2_summaryleccalc/P3.idx > /dev/null & pid72=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_eltcalc_P4 fifo/il_S1_summarycalc_P4 fifo/il_S1_pltcalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid73=$!
tee < fifo/il_S1_summary_P4.idx work/il_S1_summaryaalcalc/P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid74=$!
tee < fifo/il_S2_summary_P4 fifo/il_S2_eltcalc_P4 fifo/il_S2_summarycalc_P4 fifo/il_S2_pltcalc_P4 work/il_S2_summaryaalcalc/P4.bin work/il_S2_summaryleccalc/P4.bin > /dev/null & pid75=$!
tee < fifo/il_S2_summary_P4.idx work/il_S2_summaryaalcalc/P4.idx work/il_S2_summaryleccalc/P4.idx > /dev/null & pid76=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_eltcalc_P5 fifo/il_S1_summarycalc_P5 fifo/il_S1_pltcalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid77=$!
tee < fifo/il_S1_summary_P5.idx work/il_S1_summaryaalcalc/P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid78=$!
tee < fifo/il_S2_summary_P5 fifo/il_S2_eltcalc_P5 fifo/il_S2_summarycalc_P5 fifo/il_S2_pltcalc_P5 work/il_S2_summaryaalcalc/P5.bin work/il_S2_summaryleccalc/P5.bin > /dev/null & pid79=$!
tee < fifo/il_S2_summary_P5.idx work/il_S2_summaryaalcalc/P5.idx work/il_S2_summaryleccalc/P5.idx > /dev/null & pid80=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_eltcalc_P6 fifo/il_S1_summarycalc_P6 fifo/il_S1_pltcalc_P6 work/il_S1_summaryaalcalc/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid81=$!
tee < fifo/il_S1_summary_P6.idx work/il_S1_summaryaalcalc/P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid82=$!
tee < fifo/il_S2_summary_P6 fifo/il_S2_eltcalc_P6 fifo/il_S2_summarycalc_P6 fifo/il_S2_pltcalc_P6 work/il_S2_summaryaalcalc/P6.bin work/il_S2_summaryleccalc/P6.bin > /dev/null & pid83=$!
tee < fifo/il_S2_summary_P6.idx work/il_S2_summaryaalcalc/P6.idx work/il_S2_summaryleccalc/P6.idx > /dev/null & pid84=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_eltcalc_P7 fifo/il_S1_summarycalc_P7 fifo/il_S1_pltcalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid85=$!
tee < fifo/il_S1_summary_P7.idx work/il_S1_summaryaalcalc/P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid86=$!
tee < fifo/il_S2_summary_P7 fifo/il_S2_eltcalc_P7 fifo/il_S2_summarycalc_P7 fifo/il_S2_pltcalc_P7 work/il_S2_summaryaalcalc/P7.bin work/il_S2_summaryleccalc/P7.bin > /dev/null & pid87=$!
tee < fifo/il_S2_summary_P7.idx work/il_S2_summaryaalcalc/P7.idx work/il_S2_summaryleccalc/P7.idx > /dev/null & pid88=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_eltcalc_P8 fifo/il_S1_summarycalc_P8 fifo/il_S1_pltcalc_P8 work/il_S1_summaryaalcalc/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid89=$!
tee < fifo/il_S1_summary_P8.idx work/il_S1_summaryaalcalc/P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid90=$!
tee < fifo/il_S2_summary_P8 fifo/il_S2_eltcalc_P8 fifo/il_S2_summarycalc_P8 fifo/il_S2_pltcalc_P8 work/il_S2_summaryaalcalc/P8.bin work/il_S2_summaryleccalc/P8.bin > /dev/null & pid91=$!
tee < fifo/il_S2_summary_P8.idx work/il_S2_summaryaalcalc/P8.idx work/il_S2_summaryleccalc/P8.idx > /dev/null & pid92=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_eltcalc_P9 fifo/il_S1_summarycalc_P9 fifo/il_S1_pltcalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid93=$!
tee < fifo/il_S1_summary_P9.idx work/il_S1_summaryaalcalc/P9.idx work/il_S1_summaryleccalc/P9.idx > /dev/null & pid94=$!
tee < fifo/il_S2_summary_P9 fifo/il_S2_eltcalc_P9 fifo/il_S2_summarycalc_P9 fifo/il_S2_pltcalc_P9 work/il_S2_summaryaalcalc/P9.bin work/il_S2_summaryleccalc/P9.bin > /dev/null & pid95=$!
tee < fifo/il_S2_summary_P9.idx work/il_S2_summaryaalcalc/P9.idx work/il_S2_summaryleccalc/P9.idx > /dev/null & pid96=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_eltcalc_P10 fifo/il_S1_summarycalc_P10 fifo/il_S1_pltcalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid97=$!
tee < fifo/il_S1_summary_P10.idx work/il_S1_summaryaalcalc/P10.idx work/il_S1_summaryleccalc/P10.idx > /dev/null & pid98=$!
tee < fifo/il_S2_summary_P10 fifo/il_S2_eltcalc_P10 fifo/il_S2_summarycalc_P10 fifo/il_S2_pltcalc_P10 work/il_S2_summaryaalcalc/P10.bin work/il_S2_summaryleccalc/P10.bin > /dev/null & pid99=$!
tee < fifo/il_S2_summary_P10.idx work/il_S2_summaryaalcalc/P10.idx work/il_S2_summaryleccalc/P10.idx > /dev/null & pid100=$!

( summarycalc -m -f  -1 fifo/il_S1_summary_P1 -2 fifo/il_S2_summary_P1 < fifo/il_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P2 -2 fifo/il_S2_summary_P2 < fifo/il_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P3 -2 fifo/il_S2_summary_P3 < fifo/il_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P4 -2 fifo/il_S2_summary_P4 < fifo/il_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P5 -2 fifo/il_S2_summary_P5 < fifo/il_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P6 -2 fifo/il_S2_summary_P6 < fifo/il_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P7 -2 fifo/il_S2_summary_P7 < fifo/il_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P8 -2 fifo/il_S2_summary_P8 < fifo/il_P8 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P9 -2 fifo/il_S2_summary_P9 < fifo/il_P9 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P10 -2 fifo/il_S2_summary_P10 < fifo/il_P10 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---

( eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid101=$!
( summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid102=$!
( pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid103=$!
( eltcalc < fifo/gul_S2_eltcalc_P1 > work/kat/gul_S2_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid104=$!
( summarycalctocsv < fifo/gul_S2_summarycalc_P1 > work/kat/gul_S2_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid105=$!
( pltcalc < fifo/gul_S2_pltcalc_P1 > work/kat/gul_S2_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid106=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid107=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 ) 2>> $LOG_DIR/stderror.err & pid108=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid109=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P2 > work/kat/gul_S2_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid110=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P2 > work/kat/gul_S2_summarycalc_P2 ) 2>> $LOG_DIR/stderror.err & pid111=$!
( pltcalc -H < fifo/gul_S2_pltcalc_P2 > work/kat/gul_S2_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid112=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P3 > work/kat/gul_S1_eltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid113=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 ) 2>> $LOG_DIR/stderror.err & pid114=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P3 > work/kat/gul_S1_pltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid115=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P3 > work/kat/gul_S2_eltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid116=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P3 > work/kat/gul_S2_summarycalc_P3 ) 2>> $LOG_DIR/stderror.err & pid117=$!
( pltcalc -H < fifo/gul_S2_pltcalc_P3 > work/kat/gul_S2_pltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid118=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid119=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 ) 2>> $LOG_DIR/stderror.err & pid120=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P4 > work/kat/gul_S1_pltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid121=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P4 > work/kat/gul_S2_eltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid122=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P4 > work/kat/gul_S2_summarycalc_P4 ) 2>> $LOG_DIR/stderror.err & pid123=$!
( pltcalc -H < fifo/gul_S2_pltcalc_P4 > work/kat/gul_S2_pltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid124=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P5 > work/kat/gul_S1_eltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid125=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 ) 2>> $LOG_DIR/stderror.err & pid126=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P5 > work/kat/gul_S1_pltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid127=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P5 > work/kat/gul_S2_eltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid128=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P5 > work/kat/gul_S2_summarycalc_P5 ) 2>> $LOG_DIR/stderror.err & pid129=$!
( pltcalc -H < fifo/gul_S2_pltcalc_P5 > work/kat/gul_S2_pltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid130=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P6 > work/kat/gul_S1_eltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid131=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 ) 2>> $LOG_DIR/stderror.err & pid132=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P6 > work/kat/gul_S1_pltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid133=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P6 > work/kat/gul_S2_eltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid134=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P6 > work/kat/gul_S2_summarycalc_P6 ) 2>> $LOG_DIR/stderror.err & pid135=$!
( pltcalc -H < fifo/gul_S2_pltcalc_P6 > work/kat/gul_S2_pltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid136=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid137=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 ) 2>> $LOG_DIR/stderror.err & pid138=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P7 > work/kat/gul_S1_pltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid139=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P7 > work/kat/gul_S2_eltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid140=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P7 > work/kat/gul_S2_summarycalc_P7 ) 2>> $LOG_DIR/stderror.err & pid141=$!
( pltcalc -H < fifo/gul_S2_pltcalc_P7 > work/kat/gul_S2_pltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid142=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid143=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 ) 2>> $LOG_DIR/stderror.err & pid144=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P8 > work/kat/gul_S1_pltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid145=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P8 > work/kat/gul_S2_eltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid146=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P8 > work/kat/gul_S2_summarycalc_P8 ) 2>> $LOG_DIR/stderror.err & pid147=$!
( pltcalc -H < fifo/gul_S2_pltcalc_P8 > work/kat/gul_S2_pltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid148=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid149=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 ) 2>> $LOG_DIR/stderror.err & pid150=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P9 > work/kat/gul_S1_pltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid151=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P9 > work/kat/gul_S2_eltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid152=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P9 > work/kat/gul_S2_summarycalc_P9 ) 2>> $LOG_DIR/stderror.err & pid153=$!
( pltcalc -H < fifo/gul_S2_pltcalc_P9 > work/kat/gul_S2_pltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid154=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid155=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 ) 2>> $LOG_DIR/stderror.err & pid156=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P10 > work/kat/gul_S1_pltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid157=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P10 > work/kat/gul_S2_eltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid158=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P10 > work/kat/gul_S2_summarycalc_P10 ) 2>> $LOG_DIR/stderror.err & pid159=$!
( pltcalc -H < fifo/gul_S2_pltcalc_P10 > work/kat/gul_S2_pltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid160=$!


tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid161=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid162=$!
tee < fifo/gul_S2_summary_P1 fifo/gul_S2_eltcalc_P1 fifo/gul_S2_summarycalc_P1 fifo/gul_S2_pltcalc_P1 work/gul_S2_summaryaalcalc/P1.bin work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid163=$!
tee < fifo/gul_S2_summary_P1.idx work/gul_S2_summaryaalcalc/P1.idx work/gul_S2_summaryleccalc/P1.idx > /dev/null & pid164=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid165=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summaryaalcalc/P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid166=$!
tee < fifo/gul_S2_summary_P2 fifo/gul_S2_eltcalc_P2 fifo/gul_S2_summarycalc_P2 fifo/gul_S2_pltcalc_P2 work/gul_S2_summaryaalcalc/P2.bin work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid167=$!
tee < fifo/gul_S2_summary_P2.idx work/gul_S2_summaryaalcalc/P2.idx work/gul_S2_summaryleccalc/P2.idx > /dev/null & pid168=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_eltcalc_P3 fifo/gul_S1_summarycalc_P3 fifo/gul_S1_pltcalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid169=$!
tee < fifo/gul_S1_summary_P3.idx work/gul_S1_summaryaalcalc/P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid170=$!
tee < fifo/gul_S2_summary_P3 fifo/gul_S2_eltcalc_P3 fifo/gul_S2_summarycalc_P3 fifo/gul_S2_pltcalc_P3 work/gul_S2_summaryaalcalc/P3.bin work/gul_S2_summaryleccalc/P3.bin > /dev/null & pid171=$!
tee < fifo/gul_S2_summary_P3.idx work/gul_S2_summaryaalcalc/P3.idx work/gul_S2_summaryleccalc/P3.idx > /dev/null & pid172=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_eltcalc_P4 fifo/gul_S1_summarycalc_P4 fifo/gul_S1_pltcalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid173=$!
tee < fifo/gul_S1_summary_P4.idx work/gul_S1_summaryaalcalc/P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid174=$!
tee < fifo/gul_S2_summary_P4 fifo/gul_S2_eltcalc_P4 fifo/gul_S2_summarycalc_P4 fifo/gul_S2_pltcalc_P4 work/gul_S2_summaryaalcalc/P4.bin work/gul_S2_summaryleccalc/P4.bin > /dev/null & pid175=$!
tee < fifo/gul_S2_summary_P4.idx work/gul_S2_summaryaalcalc/P4.idx work/gul_S2_summaryleccalc/P4.idx > /dev/null & pid176=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_eltcalc_P5 fifo/gul_S1_summarycalc_P5 fifo/gul_S1_pltcalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid177=$!
tee < fifo/gul_S1_summary_P5.idx work/gul_S1_summaryaalcalc/P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid178=$!
tee < fifo/gul_S2_summary_P5 fifo/gul_S2_eltcalc_P5 fifo/gul_S2_summarycalc_P5 fifo/gul_S2_pltcalc_P5 work/gul_S2_summaryaalcalc/P5.bin work/gul_S2_summaryleccalc/P5.bin > /dev/null & pid179=$!
tee < fifo/gul_S2_summary_P5.idx work/gul_S2_summaryaalcalc/P5.idx work/gul_S2_summaryleccalc/P5.idx > /dev/null & pid180=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_eltcalc_P6 fifo/gul_S1_summarycalc_P6 fifo/gul_S1_pltcalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid181=$!
tee < fifo/gul_S1_summary_P6.idx work/gul_S1_summaryaalcalc/P6.idx work/gul_S1_summaryleccalc/P6.idx > /dev/null & pid182=$!
tee < fifo/gul_S2_summary_P6 fifo/gul_S2_eltcalc_P6 fifo/gul_S2_summarycalc_P6 fifo/gul_S2_pltcalc_P6 work/gul_S2_summaryaalcalc/P6.bin work/gul_S2_summaryleccalc/P6.bin > /dev/null & pid183=$!
tee < fifo/gul_S2_summary_P6.idx work/gul_S2_summaryaalcalc/P6.idx work/gul_S2_summaryleccalc/P6.idx > /dev/null & pid184=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_eltcalc_P7 fifo/gul_S1_summarycalc_P7 fifo/gul_S1_pltcalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid185=$!
tee < fifo/gul_S1_summary_P7.idx work/gul_S1_summaryaalcalc/P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid186=$!
tee < fifo/gul_S2_summary_P7 fifo/gul_S2_eltcalc_P7 fifo/gul_S2_summarycalc_P7 fifo/gul_S2_pltcalc_P7 work/gul_S2_summaryaalcalc/P7.bin work/gul_S2_summaryleccalc/P7.bin > /dev/null & pid187=$!
tee < fifo/gul_S2_summary_P7.idx work/gul_S2_summaryaalcalc/P7.idx work/gul_S2_summaryleccalc/P7.idx > /dev/null & pid188=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_eltcalc_P8 fifo/gul_S1_summarycalc_P8 fifo/gul_S1_pltcalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid189=$!
tee < fifo/gul_S1_summary_P8.idx work/gul_S1_summaryaalcalc/P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid190=$!
tee < fifo/gul_S2_summary_P8 fifo/gul_S2_eltcalc_P8 fifo/gul_S2_summarycalc_P8 fifo/gul_S2_pltcalc_P8 work/gul_S2_summaryaalcalc/P8.bin work/gul_S2_summaryleccalc/P8.bin > /dev/null & pid191=$!
tee < fifo/gul_S2_summary_P8.idx work/gul_S2_summaryaalcalc/P8.idx work/gul_S2_summaryleccalc/P8.idx > /dev/null & pid192=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_eltcalc_P9 fifo/gul_S1_summarycalc_P9 fifo/gul_S1_pltcalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid193=$!
tee < fifo/gul_S1_summary_P9.idx work/gul_S1_summaryaalcalc/P9.idx work/gul_S1_summaryleccalc/P9.idx > /dev/null & pid194=$!
tee < fifo/gul_S2_summary_P9 fifo/gul_S2_eltcalc_P9 fifo/gul_S2_summarycalc_P9 fifo/gul_S2_pltcalc_P9 work/gul_S2_summaryaalcalc/P9.bin work/gul_S2_summaryleccalc/P9.bin > /dev/null & pid195=$!
tee < fifo/gul_S2_summary_P9.idx work/gul_S2_summaryaalcalc/P9.idx work/gul_S2_summaryleccalc/P9.idx > /dev/null & pid196=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_eltcalc_P10 fifo/gul_S1_summarycalc_P10 fifo/gul_S1_pltcalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid197=$!
tee < fifo/gul_S1_summary_P10.idx work/gul_S1_summaryaalcalc/P10.idx work/gul_S1_summaryleccalc/P10.idx > /dev/null & pid198=$!
tee < fifo/gul_S2_summary_P10 fifo/gul_S2_eltcalc_P10 fifo/gul_S2_summarycalc_P10 fifo/gul_S2_pltcalc_P10 work/gul_S2_summaryaalcalc/P10.bin work/gul_S2_summaryleccalc/P10.bin > /dev/null & pid199=$!
tee < fifo/gul_S2_summary_P10.idx work/gul_S2_summaryaalcalc/P10.idx work/gul_S2_summaryleccalc/P10.idx > /dev/null & pid200=$!

( summarycalc -m -i  -1 fifo/gul_S1_summary_P1 -2 fifo/gul_S2_summary_P1 < fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P3 -2 fifo/gul_S2_summary_P3 < fifo/gul_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P4 -2 fifo/gul_S2_summary_P4 < fifo/gul_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P5 -2 fifo/gul_S2_summary_P5 < fifo/gul_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P6 -2 fifo/gul_S2_summary_P6 < fifo/gul_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P7 -2 fifo/gul_S2_summary_P7 < fifo/gul_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P8 -2 fifo/gul_S2_summary_P8 < fifo/gul_P8 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P9 -2 fifo/gul_S2_summary_P9 < fifo/gul_P9 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P10 -2 fifo/gul_S2_summary_P10 < fifo/gul_P10 ) 2>> $LOG_DIR/stderror.err  &

( ( eve 1 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 > fifo/il_P1  ) 2>> $LOG_DIR/stderror.err ) & pid201=$!
( ( eve 2 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P2 | fmcalc -a2 > fifo/il_P2  ) 2>> $LOG_DIR/stderror.err ) & pid202=$!
( ( eve 3 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P3 | fmcalc -a2 > fifo/il_P3  ) 2>> $LOG_DIR/stderror.err ) & pid203=$!
( ( eve 4 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P4 | fmcalc -a2 > fifo/il_P4  ) 2>> $LOG_DIR/stderror.err ) & pid204=$!
( ( eve 5 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P5 | fmcalc -a2 > fifo/il_P5  ) 2>> $LOG_DIR/stderror.err ) & pid205=$!
( ( eve 6 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P6 | fmcalc -a2 > fifo/il_P6  ) 2>> $LOG_DIR/stderror.err ) & pid206=$!
( ( eve 7 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P7 | fmcalc -a2 > fifo/il_P7  ) 2>> $LOG_DIR/stderror.err ) & pid207=$!
( ( eve 8 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P8 | fmcalc -a2 > fifo/il_P8  ) 2>> $LOG_DIR/stderror.err ) & pid208=$!
( ( eve 9 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P9 | fmcalc -a2 > fifo/il_P9  ) 2>> $LOG_DIR/stderror.err ) & pid209=$!
( ( eve 10 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P10 | fmcalc -a2 > fifo/il_P10  ) 2>> $LOG_DIR/stderror.err ) & pid210=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160 $pid161 $pid162 $pid163 $pid164 $pid165 $pid166 $pid167 $pid168 $pid169 $pid170 $pid171 $pid172 $pid173 $pid174 $pid175 $pid176 $pid177 $pid178 $pid179 $pid180 $pid181 $pid182 $pid183 $pid184 $pid185 $pid186 $pid187 $pid188 $pid189 $pid190 $pid191 $pid192 $pid193 $pid194 $pid195 $pid196 $pid197 $pid198 $pid199 $pid200 $pid201 $pid202 $pid203 $pid204 $pid205 $pid206 $pid207 $pid208 $pid209 $pid210


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 > output/il_S1_summarycalc.csv & kpid3=$!
kat work/kat/il_S2_eltcalc_P1 work/kat/il_S2_eltcalc_P2 work/kat/il_S2_eltcalc_P3 work/kat/il_S2_eltcalc_P4 work/kat/il_S2_eltcalc_P5 work/kat/il_S2_eltcalc_P6 work/kat/il_S2_eltcalc_P7 work/kat/il_S2_eltcalc_P8 work/kat/il_S2_eltcalc_P9 work/kat/il_S2_eltcalc_P10 > output/il_S2_eltcalc.csv & kpid4=$!
kat work/kat/il_S2_pltcalc_P1 work/kat/il_S2_pltcalc_P2 work/kat/il_S2_pltcalc_P3 work/kat/il_S2_pltcalc_P4 work/kat/il_S2_pltcalc_P5 work/kat/il_S2_pltcalc_P6 work/kat/il_S2_pltcalc_P7 work/kat/il_S2_pltcalc_P8 work/kat/il_S2_pltcalc_P9 work/kat/il_S2_pltcalc_P10 > output/il_S2_pltcalc.csv & kpid5=$!
kat work/kat/il_S2_summarycalc_P1 work/kat/il_S2_summarycalc_P2 work/kat/il_S2_summarycalc_P3 work/kat/il_S2_summarycalc_P4 work/kat/il_S2_summarycalc_P5 work/kat/il_S2_summarycalc_P6 work/kat/il_S2_summarycalc_P7 work/kat/il_S2_summarycalc_P8 work/kat/il_S2_summarycalc_P9 work/kat/il_S2_summarycalc_P10 > output/il_S2_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 > output/gul_S1_summarycalc.csv & kpid9=$!
kat work/kat/gul_S2_eltcalc_P1 work/kat/gul_S2_eltcalc_P2 work/kat/gul_S2_eltcalc_P3 work/kat/gul_S2_eltcalc_P4 work/kat/gul_S2_eltcalc_P5 work/kat/gul_S2_eltcalc_P6 work/kat/gul_S2_eltcalc_P7 work/kat/gul_S2_eltcalc_P8 work/kat/gul_S2_eltcalc_P9 work/kat/gul_S2_eltcalc_P10 > output/gul_S2_eltcalc.csv & kpid10=$!
kat work/kat/gul_S2_pltcalc_P1 work/kat/gul_S2_pltcalc_P2 work/kat/gul_S2_pltcalc_P3 work/kat/gul_S2_pltcalc_P4 work/kat/gul_S2_pltcalc_P5 work/kat/gul_S2_pltcalc_P6 work/kat/gul_S2_pltcalc_P7 work/kat/gul_S2_pltcalc_P8 work/kat/gul_S2_pltcalc_P9 work/kat/gul_S2_pltcalc_P10 > output/gul_S2_pltcalc.csv & kpid11=$!
kat work/kat/gul_S2_summarycalc_P1 work/kat/gul_S2_summarycalc_P2 work/kat/gul_S2_summarycalc_P3 work/kat/gul_S2_summarycalc_P4 work/kat/gul_S2_summarycalc_P5 work/kat/gul_S2_summarycalc_P6 work/kat/gul_S2_summarycalc_P7 work/kat/gul_S2_summarycalc_P8 work/kat/gul_S2_summarycalc_P9 work/kat/gul_S2_summarycalc_P10 > output/gul_S2_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


( aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid1=$!
( leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid2=$!
( aalcalc -Kil_S2_summaryaalcalc > output/il_S2_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid3=$!
( leccalc -r -Kil_S2_summaryleccalc -F output/il_S2_leccalc_full_uncertainty_aep.csv -f output/il_S2_leccalc_full_uncertainty_oep.csv -S output/il_S2_leccalc_sample_mean_aep.csv -s output/il_S2_leccalc_sample_mean_oep.csv -W output/il_S2_leccalc_wheatsheaf_aep.csv -M output/il_S2_leccalc_wheatsheaf_mean_aep.csv -m output/il_S2_leccalc_wheatsheaf_mean_oep.csv -w output/il_S2_leccalc_wheatsheaf_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid4=$!
( aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid5=$!
( leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid6=$!
( aalcalc -Kgul_S2_summaryaalcalc > output/gul_S2_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid7=$!
( leccalc -r -Kgul_S2_summaryleccalc -F output/gul_S2_leccalc_full_uncertainty_aep.csv -f output/gul_S2_leccalc_full_uncertainty_oep.csv -S output/gul_S2_leccalc_sample_mean_aep.csv -s output/gul_S2_leccalc_sample_mean_oep.csv -W output/gul_S2_leccalc_wheatsheaf_aep.csv -M output/gul_S2_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S2_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S2_leccalc_wheatsheaf_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid8=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8

rm -R -f work/*
rm -R -f fifo/*

check_complete
