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


mkfifo fifo/full_correlation/gul_fc_P1
mkfifo fifo/full_correlation/gul_fc_P2
mkfifo fifo/full_correlation/gul_fc_P3
mkfifo fifo/full_correlation/gul_fc_P4
mkfifo fifo/full_correlation/gul_fc_P5
mkfifo fifo/full_correlation/gul_fc_P6
mkfifo fifo/full_correlation/gul_fc_P7
mkfifo fifo/full_correlation/gul_fc_P8
mkfifo fifo/full_correlation/gul_fc_P9
mkfifo fifo/full_correlation/gul_fc_P10
mkfifo fifo/full_correlation/gul_fc_P11
mkfifo fifo/full_correlation/gul_fc_P12
mkfifo fifo/full_correlation/gul_fc_P13
mkfifo fifo/full_correlation/gul_fc_P14
mkfifo fifo/full_correlation/gul_fc_P15
mkfifo fifo/full_correlation/gul_fc_P16
mkfifo fifo/full_correlation/gul_fc_P17
mkfifo fifo/full_correlation/gul_fc_P18
mkfifo fifo/full_correlation/gul_fc_P19
mkfifo fifo/full_correlation/gul_fc_P20

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
mkfifo fifo/il_P11
mkfifo fifo/il_P12
mkfifo fifo/il_P13
mkfifo fifo/il_P14
mkfifo fifo/il_P15
mkfifo fifo/il_P16
mkfifo fifo/il_P17
mkfifo fifo/il_P18
mkfifo fifo/il_P19
mkfifo fifo/il_P20

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_pltcalc_P2

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_pltcalc_P3

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_pltcalc_P4

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_pltcalc_P5

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_pltcalc_P6

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_pltcalc_P7

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_pltcalc_P8

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_pltcalc_P9

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_pltcalc_P10

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_pltcalc_P11

mkfifo fifo/il_S1_summary_P12
mkfifo fifo/il_S1_pltcalc_P12

mkfifo fifo/il_S1_summary_P13
mkfifo fifo/il_S1_pltcalc_P13

mkfifo fifo/il_S1_summary_P14
mkfifo fifo/il_S1_pltcalc_P14

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_pltcalc_P15

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_pltcalc_P16

mkfifo fifo/il_S1_summary_P17
mkfifo fifo/il_S1_pltcalc_P17

mkfifo fifo/il_S1_summary_P18
mkfifo fifo/il_S1_pltcalc_P18

mkfifo fifo/il_S1_summary_P19
mkfifo fifo/il_S1_pltcalc_P19

mkfifo fifo/il_S1_summary_P20
mkfifo fifo/il_S1_pltcalc_P20

mkfifo fifo/full_correlation/il_P1
mkfifo fifo/full_correlation/il_P2
mkfifo fifo/full_correlation/il_P3
mkfifo fifo/full_correlation/il_P4
mkfifo fifo/full_correlation/il_P5
mkfifo fifo/full_correlation/il_P6
mkfifo fifo/full_correlation/il_P7
mkfifo fifo/full_correlation/il_P8
mkfifo fifo/full_correlation/il_P9
mkfifo fifo/full_correlation/il_P10
mkfifo fifo/full_correlation/il_P11
mkfifo fifo/full_correlation/il_P12
mkfifo fifo/full_correlation/il_P13
mkfifo fifo/full_correlation/il_P14
mkfifo fifo/full_correlation/il_P15
mkfifo fifo/full_correlation/il_P16
mkfifo fifo/full_correlation/il_P17
mkfifo fifo/full_correlation/il_P18
mkfifo fifo/full_correlation/il_P19
mkfifo fifo/full_correlation/il_P20

mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_pltcalc_P1

mkfifo fifo/full_correlation/il_S1_summary_P2
mkfifo fifo/full_correlation/il_S1_pltcalc_P2

mkfifo fifo/full_correlation/il_S1_summary_P3
mkfifo fifo/full_correlation/il_S1_pltcalc_P3

mkfifo fifo/full_correlation/il_S1_summary_P4
mkfifo fifo/full_correlation/il_S1_pltcalc_P4

mkfifo fifo/full_correlation/il_S1_summary_P5
mkfifo fifo/full_correlation/il_S1_pltcalc_P5

mkfifo fifo/full_correlation/il_S1_summary_P6
mkfifo fifo/full_correlation/il_S1_pltcalc_P6

mkfifo fifo/full_correlation/il_S1_summary_P7
mkfifo fifo/full_correlation/il_S1_pltcalc_P7

mkfifo fifo/full_correlation/il_S1_summary_P8
mkfifo fifo/full_correlation/il_S1_pltcalc_P8

mkfifo fifo/full_correlation/il_S1_summary_P9
mkfifo fifo/full_correlation/il_S1_pltcalc_P9

mkfifo fifo/full_correlation/il_S1_summary_P10
mkfifo fifo/full_correlation/il_S1_pltcalc_P10

mkfifo fifo/full_correlation/il_S1_summary_P11
mkfifo fifo/full_correlation/il_S1_pltcalc_P11

mkfifo fifo/full_correlation/il_S1_summary_P12
mkfifo fifo/full_correlation/il_S1_pltcalc_P12

mkfifo fifo/full_correlation/il_S1_summary_P13
mkfifo fifo/full_correlation/il_S1_pltcalc_P13

mkfifo fifo/full_correlation/il_S1_summary_P14
mkfifo fifo/full_correlation/il_S1_pltcalc_P14

mkfifo fifo/full_correlation/il_S1_summary_P15
mkfifo fifo/full_correlation/il_S1_pltcalc_P15

mkfifo fifo/full_correlation/il_S1_summary_P16
mkfifo fifo/full_correlation/il_S1_pltcalc_P16

mkfifo fifo/full_correlation/il_S1_summary_P17
mkfifo fifo/full_correlation/il_S1_pltcalc_P17

mkfifo fifo/full_correlation/il_S1_summary_P18
mkfifo fifo/full_correlation/il_S1_pltcalc_P18

mkfifo fifo/full_correlation/il_S1_summary_P19
mkfifo fifo/full_correlation/il_S1_pltcalc_P19

mkfifo fifo/full_correlation/il_S1_summary_P20
mkfifo fifo/full_correlation/il_S1_pltcalc_P20



# --- Do insured loss computes ---

( pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( pltcalc -H < fifo/il_S1_pltcalc_P3 > work/kat/il_S1_pltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid3=$!
( pltcalc -H < fifo/il_S1_pltcalc_P4 > work/kat/il_S1_pltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid4=$!
( pltcalc -H < fifo/il_S1_pltcalc_P5 > work/kat/il_S1_pltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid5=$!
( pltcalc -H < fifo/il_S1_pltcalc_P6 > work/kat/il_S1_pltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid6=$!
( pltcalc -H < fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid7=$!
( pltcalc -H < fifo/il_S1_pltcalc_P8 > work/kat/il_S1_pltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid8=$!
( pltcalc -H < fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid9=$!
( pltcalc -H < fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid10=$!
( pltcalc -H < fifo/il_S1_pltcalc_P11 > work/kat/il_S1_pltcalc_P11 ) 2>> $LOG_DIR/stderror.err & pid11=$!
( pltcalc -H < fifo/il_S1_pltcalc_P12 > work/kat/il_S1_pltcalc_P12 ) 2>> $LOG_DIR/stderror.err & pid12=$!
( pltcalc -H < fifo/il_S1_pltcalc_P13 > work/kat/il_S1_pltcalc_P13 ) 2>> $LOG_DIR/stderror.err & pid13=$!
( pltcalc -H < fifo/il_S1_pltcalc_P14 > work/kat/il_S1_pltcalc_P14 ) 2>> $LOG_DIR/stderror.err & pid14=$!
( pltcalc -H < fifo/il_S1_pltcalc_P15 > work/kat/il_S1_pltcalc_P15 ) 2>> $LOG_DIR/stderror.err & pid15=$!
( pltcalc -H < fifo/il_S1_pltcalc_P16 > work/kat/il_S1_pltcalc_P16 ) 2>> $LOG_DIR/stderror.err & pid16=$!
( pltcalc -H < fifo/il_S1_pltcalc_P17 > work/kat/il_S1_pltcalc_P17 ) 2>> $LOG_DIR/stderror.err & pid17=$!
( pltcalc -H < fifo/il_S1_pltcalc_P18 > work/kat/il_S1_pltcalc_P18 ) 2>> $LOG_DIR/stderror.err & pid18=$!
( pltcalc -H < fifo/il_S1_pltcalc_P19 > work/kat/il_S1_pltcalc_P19 ) 2>> $LOG_DIR/stderror.err & pid19=$!
( pltcalc -H < fifo/il_S1_pltcalc_P20 > work/kat/il_S1_pltcalc_P20 ) 2>> $LOG_DIR/stderror.err & pid20=$!


tee < fifo/il_S1_summary_P1 fifo/il_S1_pltcalc_P1 > /dev/null & pid21=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_pltcalc_P2 > /dev/null & pid22=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_pltcalc_P3 > /dev/null & pid23=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_pltcalc_P4 > /dev/null & pid24=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_pltcalc_P5 > /dev/null & pid25=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_pltcalc_P6 > /dev/null & pid26=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_pltcalc_P7 > /dev/null & pid27=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_pltcalc_P8 > /dev/null & pid28=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_pltcalc_P9 > /dev/null & pid29=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_pltcalc_P10 > /dev/null & pid30=$!
tee < fifo/il_S1_summary_P11 fifo/il_S1_pltcalc_P11 > /dev/null & pid31=$!
tee < fifo/il_S1_summary_P12 fifo/il_S1_pltcalc_P12 > /dev/null & pid32=$!
tee < fifo/il_S1_summary_P13 fifo/il_S1_pltcalc_P13 > /dev/null & pid33=$!
tee < fifo/il_S1_summary_P14 fifo/il_S1_pltcalc_P14 > /dev/null & pid34=$!
tee < fifo/il_S1_summary_P15 fifo/il_S1_pltcalc_P15 > /dev/null & pid35=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_pltcalc_P16 > /dev/null & pid36=$!
tee < fifo/il_S1_summary_P17 fifo/il_S1_pltcalc_P17 > /dev/null & pid37=$!
tee < fifo/il_S1_summary_P18 fifo/il_S1_pltcalc_P18 > /dev/null & pid38=$!
tee < fifo/il_S1_summary_P19 fifo/il_S1_pltcalc_P19 > /dev/null & pid39=$!
tee < fifo/il_S1_summary_P20 fifo/il_S1_pltcalc_P20 > /dev/null & pid40=$!

( summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P3 < fifo/il_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P4 < fifo/il_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P5 < fifo/il_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P6 < fifo/il_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P8 < fifo/il_P8 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P9 < fifo/il_P9 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P10 < fifo/il_P10 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P11 < fifo/il_P11 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P12 < fifo/il_P12 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P13 < fifo/il_P13 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P14 < fifo/il_P14 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P15 < fifo/il_P15 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P17 < fifo/il_P17 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P18 < fifo/il_P18 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P19 < fifo/il_P19 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P20 < fifo/il_P20 ) 2>> $LOG_DIR/stderror.err  &

# --- Do insured loss computes ---

( pltcalc < fifo/full_correlation/il_S1_pltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid41=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P2 > work/full_correlation/kat/il_S1_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid42=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P3 > work/full_correlation/kat/il_S1_pltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid43=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P4 > work/full_correlation/kat/il_S1_pltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid44=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P5 > work/full_correlation/kat/il_S1_pltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid45=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P6 > work/full_correlation/kat/il_S1_pltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid46=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P7 > work/full_correlation/kat/il_S1_pltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid47=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P8 > work/full_correlation/kat/il_S1_pltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid48=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P9 > work/full_correlation/kat/il_S1_pltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid49=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P10 > work/full_correlation/kat/il_S1_pltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid50=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P11 > work/full_correlation/kat/il_S1_pltcalc_P11 ) 2>> $LOG_DIR/stderror.err & pid51=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P12 > work/full_correlation/kat/il_S1_pltcalc_P12 ) 2>> $LOG_DIR/stderror.err & pid52=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P13 > work/full_correlation/kat/il_S1_pltcalc_P13 ) 2>> $LOG_DIR/stderror.err & pid53=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P14 > work/full_correlation/kat/il_S1_pltcalc_P14 ) 2>> $LOG_DIR/stderror.err & pid54=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P15 > work/full_correlation/kat/il_S1_pltcalc_P15 ) 2>> $LOG_DIR/stderror.err & pid55=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P16 > work/full_correlation/kat/il_S1_pltcalc_P16 ) 2>> $LOG_DIR/stderror.err & pid56=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P17 > work/full_correlation/kat/il_S1_pltcalc_P17 ) 2>> $LOG_DIR/stderror.err & pid57=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P18 > work/full_correlation/kat/il_S1_pltcalc_P18 ) 2>> $LOG_DIR/stderror.err & pid58=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P19 > work/full_correlation/kat/il_S1_pltcalc_P19 ) 2>> $LOG_DIR/stderror.err & pid59=$!
( pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P20 > work/full_correlation/kat/il_S1_pltcalc_P20 ) 2>> $LOG_DIR/stderror.err & pid60=$!


tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_pltcalc_P1 > /dev/null & pid61=$!
tee < fifo/full_correlation/il_S1_summary_P2 fifo/full_correlation/il_S1_pltcalc_P2 > /dev/null & pid62=$!
tee < fifo/full_correlation/il_S1_summary_P3 fifo/full_correlation/il_S1_pltcalc_P3 > /dev/null & pid63=$!
tee < fifo/full_correlation/il_S1_summary_P4 fifo/full_correlation/il_S1_pltcalc_P4 > /dev/null & pid64=$!
tee < fifo/full_correlation/il_S1_summary_P5 fifo/full_correlation/il_S1_pltcalc_P5 > /dev/null & pid65=$!
tee < fifo/full_correlation/il_S1_summary_P6 fifo/full_correlation/il_S1_pltcalc_P6 > /dev/null & pid66=$!
tee < fifo/full_correlation/il_S1_summary_P7 fifo/full_correlation/il_S1_pltcalc_P7 > /dev/null & pid67=$!
tee < fifo/full_correlation/il_S1_summary_P8 fifo/full_correlation/il_S1_pltcalc_P8 > /dev/null & pid68=$!
tee < fifo/full_correlation/il_S1_summary_P9 fifo/full_correlation/il_S1_pltcalc_P9 > /dev/null & pid69=$!
tee < fifo/full_correlation/il_S1_summary_P10 fifo/full_correlation/il_S1_pltcalc_P10 > /dev/null & pid70=$!
tee < fifo/full_correlation/il_S1_summary_P11 fifo/full_correlation/il_S1_pltcalc_P11 > /dev/null & pid71=$!
tee < fifo/full_correlation/il_S1_summary_P12 fifo/full_correlation/il_S1_pltcalc_P12 > /dev/null & pid72=$!
tee < fifo/full_correlation/il_S1_summary_P13 fifo/full_correlation/il_S1_pltcalc_P13 > /dev/null & pid73=$!
tee < fifo/full_correlation/il_S1_summary_P14 fifo/full_correlation/il_S1_pltcalc_P14 > /dev/null & pid74=$!
tee < fifo/full_correlation/il_S1_summary_P15 fifo/full_correlation/il_S1_pltcalc_P15 > /dev/null & pid75=$!
tee < fifo/full_correlation/il_S1_summary_P16 fifo/full_correlation/il_S1_pltcalc_P16 > /dev/null & pid76=$!
tee < fifo/full_correlation/il_S1_summary_P17 fifo/full_correlation/il_S1_pltcalc_P17 > /dev/null & pid77=$!
tee < fifo/full_correlation/il_S1_summary_P18 fifo/full_correlation/il_S1_pltcalc_P18 > /dev/null & pid78=$!
tee < fifo/full_correlation/il_S1_summary_P19 fifo/full_correlation/il_S1_pltcalc_P19 > /dev/null & pid79=$!
tee < fifo/full_correlation/il_S1_summary_P20 fifo/full_correlation/il_S1_pltcalc_P20 > /dev/null & pid80=$!

( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P1 < fifo/full_correlation/il_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P2 < fifo/full_correlation/il_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P3 < fifo/full_correlation/il_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P4 < fifo/full_correlation/il_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P5 < fifo/full_correlation/il_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P6 < fifo/full_correlation/il_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P7 < fifo/full_correlation/il_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P8 < fifo/full_correlation/il_P8 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P9 < fifo/full_correlation/il_P9 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P10 < fifo/full_correlation/il_P10 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P11 < fifo/full_correlation/il_P11 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P12 < fifo/full_correlation/il_P12 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P13 < fifo/full_correlation/il_P13 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P14 < fifo/full_correlation/il_P14 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P15 < fifo/full_correlation/il_P15 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P16 < fifo/full_correlation/il_P16 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P17 < fifo/full_correlation/il_P17 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P18 < fifo/full_correlation/il_P18 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P19 < fifo/full_correlation/il_P19 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P20 < fifo/full_correlation/il_P20 ) 2>> $LOG_DIR/stderror.err  &

( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P1 > fifo/full_correlation/il_P1 ) 2>> $LOG_DIR/stderror.err ) & pid81=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P2 > fifo/full_correlation/il_P2 ) 2>> $LOG_DIR/stderror.err ) & pid82=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P3 > fifo/full_correlation/il_P3 ) 2>> $LOG_DIR/stderror.err ) & pid83=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P4 > fifo/full_correlation/il_P4 ) 2>> $LOG_DIR/stderror.err ) & pid84=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P5 > fifo/full_correlation/il_P5 ) 2>> $LOG_DIR/stderror.err ) & pid85=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P6 > fifo/full_correlation/il_P6 ) 2>> $LOG_DIR/stderror.err ) & pid86=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P7 > fifo/full_correlation/il_P7 ) 2>> $LOG_DIR/stderror.err ) & pid87=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P8 > fifo/full_correlation/il_P8 ) 2>> $LOG_DIR/stderror.err ) & pid88=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P9 > fifo/full_correlation/il_P9 ) 2>> $LOG_DIR/stderror.err ) & pid89=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P10 > fifo/full_correlation/il_P10 ) 2>> $LOG_DIR/stderror.err ) & pid90=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P11 > fifo/full_correlation/il_P11 ) 2>> $LOG_DIR/stderror.err ) & pid91=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P12 > fifo/full_correlation/il_P12 ) 2>> $LOG_DIR/stderror.err ) & pid92=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P13 > fifo/full_correlation/il_P13 ) 2>> $LOG_DIR/stderror.err ) & pid93=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P14 > fifo/full_correlation/il_P14 ) 2>> $LOG_DIR/stderror.err ) & pid94=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P15 > fifo/full_correlation/il_P15 ) 2>> $LOG_DIR/stderror.err ) & pid95=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P16 > fifo/full_correlation/il_P16 ) 2>> $LOG_DIR/stderror.err ) & pid96=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P17 > fifo/full_correlation/il_P17 ) 2>> $LOG_DIR/stderror.err ) & pid97=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P18 > fifo/full_correlation/il_P18 ) 2>> $LOG_DIR/stderror.err ) & pid98=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P19 > fifo/full_correlation/il_P19 ) 2>> $LOG_DIR/stderror.err ) & pid99=$!
( ( fmcalc -a2 < fifo/full_correlation/gul_fc_P20 > fifo/full_correlation/il_P20 ) 2>> $LOG_DIR/stderror.err ) & pid100=$!
( ( eve 1 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P1 -a1 -i - | fmcalc -a2 > fifo/il_P1  ) 2>> $LOG_DIR/stderror.err ) & pid101=$!
( ( eve 2 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P2 -a1 -i - | fmcalc -a2 > fifo/il_P2  ) 2>> $LOG_DIR/stderror.err ) & pid102=$!
( ( eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P3 -a1 -i - | fmcalc -a2 > fifo/il_P3  ) 2>> $LOG_DIR/stderror.err ) & pid103=$!
( ( eve 4 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P4 -a1 -i - | fmcalc -a2 > fifo/il_P4  ) 2>> $LOG_DIR/stderror.err ) & pid104=$!
( ( eve 5 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P5 -a1 -i - | fmcalc -a2 > fifo/il_P5  ) 2>> $LOG_DIR/stderror.err ) & pid105=$!
( ( eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P6 -a1 -i - | fmcalc -a2 > fifo/il_P6  ) 2>> $LOG_DIR/stderror.err ) & pid106=$!
( ( eve 7 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P7 -a1 -i - | fmcalc -a2 > fifo/il_P7  ) 2>> $LOG_DIR/stderror.err ) & pid107=$!
( ( eve 8 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P8 -a1 -i - | fmcalc -a2 > fifo/il_P8  ) 2>> $LOG_DIR/stderror.err ) & pid108=$!
( ( eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P9 -a1 -i - | fmcalc -a2 > fifo/il_P9  ) 2>> $LOG_DIR/stderror.err ) & pid109=$!
( ( eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P10 -a1 -i - | fmcalc -a2 > fifo/il_P10  ) 2>> $LOG_DIR/stderror.err ) & pid110=$!
( ( eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P11 -a1 -i - | fmcalc -a2 > fifo/il_P11  ) 2>> $LOG_DIR/stderror.err ) & pid111=$!
( ( eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P12 -a1 -i - | fmcalc -a2 > fifo/il_P12  ) 2>> $LOG_DIR/stderror.err ) & pid112=$!
( ( eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P13 -a1 -i - | fmcalc -a2 > fifo/il_P13  ) 2>> $LOG_DIR/stderror.err ) & pid113=$!
( ( eve 14 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P14 -a1 -i - | fmcalc -a2 > fifo/il_P14  ) 2>> $LOG_DIR/stderror.err ) & pid114=$!
( ( eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P15 -a1 -i - | fmcalc -a2 > fifo/il_P15  ) 2>> $LOG_DIR/stderror.err ) & pid115=$!
( ( eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P16 -a1 -i - | fmcalc -a2 > fifo/il_P16  ) 2>> $LOG_DIR/stderror.err ) & pid116=$!
( ( eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P17 -a1 -i - | fmcalc -a2 > fifo/il_P17  ) 2>> $LOG_DIR/stderror.err ) & pid117=$!
( ( eve 18 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P18 -a1 -i - | fmcalc -a2 > fifo/il_P18  ) 2>> $LOG_DIR/stderror.err ) & pid118=$!
( ( eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P19 -a1 -i - | fmcalc -a2 > fifo/il_P19  ) 2>> $LOG_DIR/stderror.err ) & pid119=$!
( ( eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P20 -a1 -i - | fmcalc -a2 > fifo/il_P20  ) 2>> $LOG_DIR/stderror.err ) & pid120=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 work/kat/il_S1_pltcalc_P11 work/kat/il_S1_pltcalc_P12 work/kat/il_S1_pltcalc_P13 work/kat/il_S1_pltcalc_P14 work/kat/il_S1_pltcalc_P15 work/kat/il_S1_pltcalc_P16 work/kat/il_S1_pltcalc_P17 work/kat/il_S1_pltcalc_P18 work/kat/il_S1_pltcalc_P19 work/kat/il_S1_pltcalc_P20 > output/il_S1_pltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_pltcalc_P1 work/full_correlation/kat/il_S1_pltcalc_P2 work/full_correlation/kat/il_S1_pltcalc_P3 work/full_correlation/kat/il_S1_pltcalc_P4 work/full_correlation/kat/il_S1_pltcalc_P5 work/full_correlation/kat/il_S1_pltcalc_P6 work/full_correlation/kat/il_S1_pltcalc_P7 work/full_correlation/kat/il_S1_pltcalc_P8 work/full_correlation/kat/il_S1_pltcalc_P9 work/full_correlation/kat/il_S1_pltcalc_P10 work/full_correlation/kat/il_S1_pltcalc_P11 work/full_correlation/kat/il_S1_pltcalc_P12 work/full_correlation/kat/il_S1_pltcalc_P13 work/full_correlation/kat/il_S1_pltcalc_P14 work/full_correlation/kat/il_S1_pltcalc_P15 work/full_correlation/kat/il_S1_pltcalc_P16 work/full_correlation/kat/il_S1_pltcalc_P17 work/full_correlation/kat/il_S1_pltcalc_P18 work/full_correlation/kat/il_S1_pltcalc_P19 work/full_correlation/kat/il_S1_pltcalc_P20 > output/full_correlation/il_S1_pltcalc.csv & kpid2=$!
wait $kpid1 $kpid2


rm -R -f work/*
rm -R -f fifo/*

check_complete
