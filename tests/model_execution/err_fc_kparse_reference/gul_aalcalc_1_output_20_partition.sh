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
mkdir -p output/full_correlation/

rm -R -f fifo/*
mkdir -p fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/full_correlation/gul_S1_summaryaalcalc

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
mkfifo fifo/gul_P11
mkfifo fifo/gul_P12
mkfifo fifo/gul_P13
mkfifo fifo/gul_P14
mkfifo fifo/gul_P15
mkfifo fifo/gul_P16
mkfifo fifo/gul_P17
mkfifo fifo/gul_P18
mkfifo fifo/gul_P19
mkfifo fifo/gul_P20

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summary_P3.idx

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summary_P4.idx

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summary_P5.idx

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summary_P6.idx

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summary_P7.idx

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_summary_P8.idx

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_summary_P9.idx

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_summary_P10.idx

mkfifo fifo/gul_S1_summary_P11
mkfifo fifo/gul_S1_summary_P11.idx

mkfifo fifo/gul_S1_summary_P12
mkfifo fifo/gul_S1_summary_P12.idx

mkfifo fifo/gul_S1_summary_P13
mkfifo fifo/gul_S1_summary_P13.idx

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_summary_P14.idx

mkfifo fifo/gul_S1_summary_P15
mkfifo fifo/gul_S1_summary_P15.idx

mkfifo fifo/gul_S1_summary_P16
mkfifo fifo/gul_S1_summary_P16.idx

mkfifo fifo/gul_S1_summary_P17
mkfifo fifo/gul_S1_summary_P17.idx

mkfifo fifo/gul_S1_summary_P18
mkfifo fifo/gul_S1_summary_P18.idx

mkfifo fifo/gul_S1_summary_P19
mkfifo fifo/gul_S1_summary_P19.idx

mkfifo fifo/gul_S1_summary_P20
mkfifo fifo/gul_S1_summary_P20.idx

mkfifo fifo/full_correlation/gul_P1
mkfifo fifo/full_correlation/gul_P2
mkfifo fifo/full_correlation/gul_P3
mkfifo fifo/full_correlation/gul_P4
mkfifo fifo/full_correlation/gul_P5
mkfifo fifo/full_correlation/gul_P6
mkfifo fifo/full_correlation/gul_P7
mkfifo fifo/full_correlation/gul_P8
mkfifo fifo/full_correlation/gul_P9
mkfifo fifo/full_correlation/gul_P10
mkfifo fifo/full_correlation/gul_P11
mkfifo fifo/full_correlation/gul_P12
mkfifo fifo/full_correlation/gul_P13
mkfifo fifo/full_correlation/gul_P14
mkfifo fifo/full_correlation/gul_P15
mkfifo fifo/full_correlation/gul_P16
mkfifo fifo/full_correlation/gul_P17
mkfifo fifo/full_correlation/gul_P18
mkfifo fifo/full_correlation/gul_P19
mkfifo fifo/full_correlation/gul_P20

mkfifo fifo/full_correlation/gul_S1_summary_P1
mkfifo fifo/full_correlation/gul_S1_summary_P1.idx

mkfifo fifo/full_correlation/gul_S1_summary_P2
mkfifo fifo/full_correlation/gul_S1_summary_P2.idx

mkfifo fifo/full_correlation/gul_S1_summary_P3
mkfifo fifo/full_correlation/gul_S1_summary_P3.idx

mkfifo fifo/full_correlation/gul_S1_summary_P4
mkfifo fifo/full_correlation/gul_S1_summary_P4.idx

mkfifo fifo/full_correlation/gul_S1_summary_P5
mkfifo fifo/full_correlation/gul_S1_summary_P5.idx

mkfifo fifo/full_correlation/gul_S1_summary_P6
mkfifo fifo/full_correlation/gul_S1_summary_P6.idx

mkfifo fifo/full_correlation/gul_S1_summary_P7
mkfifo fifo/full_correlation/gul_S1_summary_P7.idx

mkfifo fifo/full_correlation/gul_S1_summary_P8
mkfifo fifo/full_correlation/gul_S1_summary_P8.idx

mkfifo fifo/full_correlation/gul_S1_summary_P9
mkfifo fifo/full_correlation/gul_S1_summary_P9.idx

mkfifo fifo/full_correlation/gul_S1_summary_P10
mkfifo fifo/full_correlation/gul_S1_summary_P10.idx

mkfifo fifo/full_correlation/gul_S1_summary_P11
mkfifo fifo/full_correlation/gul_S1_summary_P11.idx

mkfifo fifo/full_correlation/gul_S1_summary_P12
mkfifo fifo/full_correlation/gul_S1_summary_P12.idx

mkfifo fifo/full_correlation/gul_S1_summary_P13
mkfifo fifo/full_correlation/gul_S1_summary_P13.idx

mkfifo fifo/full_correlation/gul_S1_summary_P14
mkfifo fifo/full_correlation/gul_S1_summary_P14.idx

mkfifo fifo/full_correlation/gul_S1_summary_P15
mkfifo fifo/full_correlation/gul_S1_summary_P15.idx

mkfifo fifo/full_correlation/gul_S1_summary_P16
mkfifo fifo/full_correlation/gul_S1_summary_P16.idx

mkfifo fifo/full_correlation/gul_S1_summary_P17
mkfifo fifo/full_correlation/gul_S1_summary_P17.idx

mkfifo fifo/full_correlation/gul_S1_summary_P18
mkfifo fifo/full_correlation/gul_S1_summary_P18.idx

mkfifo fifo/full_correlation/gul_S1_summary_P19
mkfifo fifo/full_correlation/gul_S1_summary_P19.idx

mkfifo fifo/full_correlation/gul_S1_summary_P20
mkfifo fifo/full_correlation/gul_S1_summary_P20.idx



# --- Do ground up loss computes ---



tee < fifo/gul_S1_summary_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx > /dev/null & pid2=$!
tee < fifo/gul_S1_summary_P2 work/gul_S1_summaryaalcalc/P2.bin > /dev/null & pid3=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summaryaalcalc/P2.idx > /dev/null & pid4=$!
tee < fifo/gul_S1_summary_P3 work/gul_S1_summaryaalcalc/P3.bin > /dev/null & pid5=$!
tee < fifo/gul_S1_summary_P3.idx work/gul_S1_summaryaalcalc/P3.idx > /dev/null & pid6=$!
tee < fifo/gul_S1_summary_P4 work/gul_S1_summaryaalcalc/P4.bin > /dev/null & pid7=$!
tee < fifo/gul_S1_summary_P4.idx work/gul_S1_summaryaalcalc/P4.idx > /dev/null & pid8=$!
tee < fifo/gul_S1_summary_P5 work/gul_S1_summaryaalcalc/P5.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P5.idx work/gul_S1_summaryaalcalc/P5.idx > /dev/null & pid10=$!
tee < fifo/gul_S1_summary_P6 work/gul_S1_summaryaalcalc/P6.bin > /dev/null & pid11=$!
tee < fifo/gul_S1_summary_P6.idx work/gul_S1_summaryaalcalc/P6.idx > /dev/null & pid12=$!
tee < fifo/gul_S1_summary_P7 work/gul_S1_summaryaalcalc/P7.bin > /dev/null & pid13=$!
tee < fifo/gul_S1_summary_P7.idx work/gul_S1_summaryaalcalc/P7.idx > /dev/null & pid14=$!
tee < fifo/gul_S1_summary_P8 work/gul_S1_summaryaalcalc/P8.bin > /dev/null & pid15=$!
tee < fifo/gul_S1_summary_P8.idx work/gul_S1_summaryaalcalc/P8.idx > /dev/null & pid16=$!
tee < fifo/gul_S1_summary_P9 work/gul_S1_summaryaalcalc/P9.bin > /dev/null & pid17=$!
tee < fifo/gul_S1_summary_P9.idx work/gul_S1_summaryaalcalc/P9.idx > /dev/null & pid18=$!
tee < fifo/gul_S1_summary_P10 work/gul_S1_summaryaalcalc/P10.bin > /dev/null & pid19=$!
tee < fifo/gul_S1_summary_P10.idx work/gul_S1_summaryaalcalc/P10.idx > /dev/null & pid20=$!
tee < fifo/gul_S1_summary_P11 work/gul_S1_summaryaalcalc/P11.bin > /dev/null & pid21=$!
tee < fifo/gul_S1_summary_P11.idx work/gul_S1_summaryaalcalc/P11.idx > /dev/null & pid22=$!
tee < fifo/gul_S1_summary_P12 work/gul_S1_summaryaalcalc/P12.bin > /dev/null & pid23=$!
tee < fifo/gul_S1_summary_P12.idx work/gul_S1_summaryaalcalc/P12.idx > /dev/null & pid24=$!
tee < fifo/gul_S1_summary_P13 work/gul_S1_summaryaalcalc/P13.bin > /dev/null & pid25=$!
tee < fifo/gul_S1_summary_P13.idx work/gul_S1_summaryaalcalc/P13.idx > /dev/null & pid26=$!
tee < fifo/gul_S1_summary_P14 work/gul_S1_summaryaalcalc/P14.bin > /dev/null & pid27=$!
tee < fifo/gul_S1_summary_P14.idx work/gul_S1_summaryaalcalc/P14.idx > /dev/null & pid28=$!
tee < fifo/gul_S1_summary_P15 work/gul_S1_summaryaalcalc/P15.bin > /dev/null & pid29=$!
tee < fifo/gul_S1_summary_P15.idx work/gul_S1_summaryaalcalc/P15.idx > /dev/null & pid30=$!
tee < fifo/gul_S1_summary_P16 work/gul_S1_summaryaalcalc/P16.bin > /dev/null & pid31=$!
tee < fifo/gul_S1_summary_P16.idx work/gul_S1_summaryaalcalc/P16.idx > /dev/null & pid32=$!
tee < fifo/gul_S1_summary_P17 work/gul_S1_summaryaalcalc/P17.bin > /dev/null & pid33=$!
tee < fifo/gul_S1_summary_P17.idx work/gul_S1_summaryaalcalc/P17.idx > /dev/null & pid34=$!
tee < fifo/gul_S1_summary_P18 work/gul_S1_summaryaalcalc/P18.bin > /dev/null & pid35=$!
tee < fifo/gul_S1_summary_P18.idx work/gul_S1_summaryaalcalc/P18.idx > /dev/null & pid36=$!
tee < fifo/gul_S1_summary_P19 work/gul_S1_summaryaalcalc/P19.bin > /dev/null & pid37=$!
tee < fifo/gul_S1_summary_P19.idx work/gul_S1_summaryaalcalc/P19.idx > /dev/null & pid38=$!
tee < fifo/gul_S1_summary_P20 work/gul_S1_summaryaalcalc/P20.bin > /dev/null & pid39=$!
tee < fifo/gul_S1_summary_P20.idx work/gul_S1_summaryaalcalc/P20.idx > /dev/null & pid40=$!

( summarycalc -m -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P4 < fifo/gul_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P5 < fifo/gul_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P7 < fifo/gul_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P8 < fifo/gul_P8 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P9 < fifo/gul_P9 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P10 < fifo/gul_P10 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P11 < fifo/gul_P11 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P12 < fifo/gul_P12 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P13 < fifo/gul_P13 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P14 < fifo/gul_P14 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P15 < fifo/gul_P15 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P16 < fifo/gul_P16 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P17 < fifo/gul_P17 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P18 < fifo/gul_P18 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P19 < fifo/gul_P19 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P20 < fifo/gul_P20 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---



tee < fifo/full_correlation/gul_S1_summary_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid41=$!
tee < fifo/full_correlation/gul_S1_summary_P1.idx work/full_correlation/gul_S1_summaryaalcalc/P1.idx > /dev/null & pid42=$!
tee < fifo/full_correlation/gul_S1_summary_P2 work/full_correlation/gul_S1_summaryaalcalc/P2.bin > /dev/null & pid43=$!
tee < fifo/full_correlation/gul_S1_summary_P2.idx work/full_correlation/gul_S1_summaryaalcalc/P2.idx > /dev/null & pid44=$!
tee < fifo/full_correlation/gul_S1_summary_P3 work/full_correlation/gul_S1_summaryaalcalc/P3.bin > /dev/null & pid45=$!
tee < fifo/full_correlation/gul_S1_summary_P3.idx work/full_correlation/gul_S1_summaryaalcalc/P3.idx > /dev/null & pid46=$!
tee < fifo/full_correlation/gul_S1_summary_P4 work/full_correlation/gul_S1_summaryaalcalc/P4.bin > /dev/null & pid47=$!
tee < fifo/full_correlation/gul_S1_summary_P4.idx work/full_correlation/gul_S1_summaryaalcalc/P4.idx > /dev/null & pid48=$!
tee < fifo/full_correlation/gul_S1_summary_P5 work/full_correlation/gul_S1_summaryaalcalc/P5.bin > /dev/null & pid49=$!
tee < fifo/full_correlation/gul_S1_summary_P5.idx work/full_correlation/gul_S1_summaryaalcalc/P5.idx > /dev/null & pid50=$!
tee < fifo/full_correlation/gul_S1_summary_P6 work/full_correlation/gul_S1_summaryaalcalc/P6.bin > /dev/null & pid51=$!
tee < fifo/full_correlation/gul_S1_summary_P6.idx work/full_correlation/gul_S1_summaryaalcalc/P6.idx > /dev/null & pid52=$!
tee < fifo/full_correlation/gul_S1_summary_P7 work/full_correlation/gul_S1_summaryaalcalc/P7.bin > /dev/null & pid53=$!
tee < fifo/full_correlation/gul_S1_summary_P7.idx work/full_correlation/gul_S1_summaryaalcalc/P7.idx > /dev/null & pid54=$!
tee < fifo/full_correlation/gul_S1_summary_P8 work/full_correlation/gul_S1_summaryaalcalc/P8.bin > /dev/null & pid55=$!
tee < fifo/full_correlation/gul_S1_summary_P8.idx work/full_correlation/gul_S1_summaryaalcalc/P8.idx > /dev/null & pid56=$!
tee < fifo/full_correlation/gul_S1_summary_P9 work/full_correlation/gul_S1_summaryaalcalc/P9.bin > /dev/null & pid57=$!
tee < fifo/full_correlation/gul_S1_summary_P9.idx work/full_correlation/gul_S1_summaryaalcalc/P9.idx > /dev/null & pid58=$!
tee < fifo/full_correlation/gul_S1_summary_P10 work/full_correlation/gul_S1_summaryaalcalc/P10.bin > /dev/null & pid59=$!
tee < fifo/full_correlation/gul_S1_summary_P10.idx work/full_correlation/gul_S1_summaryaalcalc/P10.idx > /dev/null & pid60=$!
tee < fifo/full_correlation/gul_S1_summary_P11 work/full_correlation/gul_S1_summaryaalcalc/P11.bin > /dev/null & pid61=$!
tee < fifo/full_correlation/gul_S1_summary_P11.idx work/full_correlation/gul_S1_summaryaalcalc/P11.idx > /dev/null & pid62=$!
tee < fifo/full_correlation/gul_S1_summary_P12 work/full_correlation/gul_S1_summaryaalcalc/P12.bin > /dev/null & pid63=$!
tee < fifo/full_correlation/gul_S1_summary_P12.idx work/full_correlation/gul_S1_summaryaalcalc/P12.idx > /dev/null & pid64=$!
tee < fifo/full_correlation/gul_S1_summary_P13 work/full_correlation/gul_S1_summaryaalcalc/P13.bin > /dev/null & pid65=$!
tee < fifo/full_correlation/gul_S1_summary_P13.idx work/full_correlation/gul_S1_summaryaalcalc/P13.idx > /dev/null & pid66=$!
tee < fifo/full_correlation/gul_S1_summary_P14 work/full_correlation/gul_S1_summaryaalcalc/P14.bin > /dev/null & pid67=$!
tee < fifo/full_correlation/gul_S1_summary_P14.idx work/full_correlation/gul_S1_summaryaalcalc/P14.idx > /dev/null & pid68=$!
tee < fifo/full_correlation/gul_S1_summary_P15 work/full_correlation/gul_S1_summaryaalcalc/P15.bin > /dev/null & pid69=$!
tee < fifo/full_correlation/gul_S1_summary_P15.idx work/full_correlation/gul_S1_summaryaalcalc/P15.idx > /dev/null & pid70=$!
tee < fifo/full_correlation/gul_S1_summary_P16 work/full_correlation/gul_S1_summaryaalcalc/P16.bin > /dev/null & pid71=$!
tee < fifo/full_correlation/gul_S1_summary_P16.idx work/full_correlation/gul_S1_summaryaalcalc/P16.idx > /dev/null & pid72=$!
tee < fifo/full_correlation/gul_S1_summary_P17 work/full_correlation/gul_S1_summaryaalcalc/P17.bin > /dev/null & pid73=$!
tee < fifo/full_correlation/gul_S1_summary_P17.idx work/full_correlation/gul_S1_summaryaalcalc/P17.idx > /dev/null & pid74=$!
tee < fifo/full_correlation/gul_S1_summary_P18 work/full_correlation/gul_S1_summaryaalcalc/P18.bin > /dev/null & pid75=$!
tee < fifo/full_correlation/gul_S1_summary_P18.idx work/full_correlation/gul_S1_summaryaalcalc/P18.idx > /dev/null & pid76=$!
tee < fifo/full_correlation/gul_S1_summary_P19 work/full_correlation/gul_S1_summaryaalcalc/P19.bin > /dev/null & pid77=$!
tee < fifo/full_correlation/gul_S1_summary_P19.idx work/full_correlation/gul_S1_summaryaalcalc/P19.idx > /dev/null & pid78=$!
tee < fifo/full_correlation/gul_S1_summary_P20 work/full_correlation/gul_S1_summaryaalcalc/P20.bin > /dev/null & pid79=$!
tee < fifo/full_correlation/gul_S1_summary_P20.idx work/full_correlation/gul_S1_summaryaalcalc/P20.idx > /dev/null & pid80=$!

( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P1 < fifo/full_correlation/gul_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P2 < fifo/full_correlation/gul_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P3 < fifo/full_correlation/gul_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P4 < fifo/full_correlation/gul_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P5 < fifo/full_correlation/gul_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P6 < fifo/full_correlation/gul_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P7 < fifo/full_correlation/gul_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P8 < fifo/full_correlation/gul_P8 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P9 < fifo/full_correlation/gul_P9 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P10 < fifo/full_correlation/gul_P10 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P11 < fifo/full_correlation/gul_P11 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P12 < fifo/full_correlation/gul_P12 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P13 < fifo/full_correlation/gul_P13 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P14 < fifo/full_correlation/gul_P14 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P15 < fifo/full_correlation/gul_P15 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P16 < fifo/full_correlation/gul_P16 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P17 < fifo/full_correlation/gul_P17 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P18 < fifo/full_correlation/gul_P18 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P19 < fifo/full_correlation/gul_P19 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P20 < fifo/full_correlation/gul_P20 ) 2>> $LOG_DIR/stderror.err  &

( eve 1 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P1 -a1 -i - > fifo/gul_P1  ) 2>> $LOG_DIR/stderror.err &
( eve 2 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P2 -a1 -i - > fifo/gul_P2  ) 2>> $LOG_DIR/stderror.err &
( eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P3 -a1 -i - > fifo/gul_P3  ) 2>> $LOG_DIR/stderror.err &
( eve 4 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P4 -a1 -i - > fifo/gul_P4  ) 2>> $LOG_DIR/stderror.err &
( eve 5 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P5 -a1 -i - > fifo/gul_P5  ) 2>> $LOG_DIR/stderror.err &
( eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P6 -a1 -i - > fifo/gul_P6  ) 2>> $LOG_DIR/stderror.err &
( eve 7 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P7 -a1 -i - > fifo/gul_P7  ) 2>> $LOG_DIR/stderror.err &
( eve 8 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P8 -a1 -i - > fifo/gul_P8  ) 2>> $LOG_DIR/stderror.err &
( eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P9 -a1 -i - > fifo/gul_P9  ) 2>> $LOG_DIR/stderror.err &
( eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P10 -a1 -i - > fifo/gul_P10  ) 2>> $LOG_DIR/stderror.err &
( eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P11 -a1 -i - > fifo/gul_P11  ) 2>> $LOG_DIR/stderror.err &
( eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P12 -a1 -i - > fifo/gul_P12  ) 2>> $LOG_DIR/stderror.err &
( eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P13 -a1 -i - > fifo/gul_P13  ) 2>> $LOG_DIR/stderror.err &
( eve 14 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P14 -a1 -i - > fifo/gul_P14  ) 2>> $LOG_DIR/stderror.err &
( eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P15 -a1 -i - > fifo/gul_P15  ) 2>> $LOG_DIR/stderror.err &
( eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P16 -a1 -i - > fifo/gul_P16  ) 2>> $LOG_DIR/stderror.err &
( eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P17 -a1 -i - > fifo/gul_P17  ) 2>> $LOG_DIR/stderror.err &
( eve 18 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P18 -a1 -i - > fifo/gul_P18  ) 2>> $LOG_DIR/stderror.err &
( eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P19 -a1 -i - > fifo/gul_P19  ) 2>> $LOG_DIR/stderror.err &
( eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P20 -a1 -i - > fifo/gul_P20  ) 2>> $LOG_DIR/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

