#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

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

   ps -jf f -g $sess_pid > log/subprocess_list
   PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk 'BEGIN { FS = "[ \t\n]+" }{ if ($1 >= '$script_pid') print}' | grep -v celery | egrep -v *\\.log$  | egrep -v *\\.sh$ | sort -n -r)
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

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc

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
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_eltcalc_P3
mkfifo fifo/gul_S1_summarycalc_P3
mkfifo fifo/gul_S1_pltcalc_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_eltcalc_P4
mkfifo fifo/gul_S1_summarycalc_P4
mkfifo fifo/gul_S1_pltcalc_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_eltcalc_P5
mkfifo fifo/gul_S1_summarycalc_P5
mkfifo fifo/gul_S1_pltcalc_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_eltcalc_P6
mkfifo fifo/gul_S1_summarycalc_P6
mkfifo fifo/gul_S1_pltcalc_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_eltcalc_P7
mkfifo fifo/gul_S1_summarycalc_P7
mkfifo fifo/gul_S1_pltcalc_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_eltcalc_P8
mkfifo fifo/gul_S1_summarycalc_P8
mkfifo fifo/gul_S1_pltcalc_P8

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_eltcalc_P9
mkfifo fifo/gul_S1_summarycalc_P9
mkfifo fifo/gul_S1_pltcalc_P9

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_eltcalc_P10
mkfifo fifo/gul_S1_summarycalc_P10
mkfifo fifo/gul_S1_pltcalc_P10

mkfifo fifo/gul_S1_summary_P11
mkfifo fifo/gul_S1_eltcalc_P11
mkfifo fifo/gul_S1_summarycalc_P11
mkfifo fifo/gul_S1_pltcalc_P11

mkfifo fifo/gul_S1_summary_P12
mkfifo fifo/gul_S1_eltcalc_P12
mkfifo fifo/gul_S1_summarycalc_P12
mkfifo fifo/gul_S1_pltcalc_P12

mkfifo fifo/gul_S1_summary_P13
mkfifo fifo/gul_S1_eltcalc_P13
mkfifo fifo/gul_S1_summarycalc_P13
mkfifo fifo/gul_S1_pltcalc_P13

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_eltcalc_P14
mkfifo fifo/gul_S1_summarycalc_P14
mkfifo fifo/gul_S1_pltcalc_P14

mkfifo fifo/gul_S1_summary_P15
mkfifo fifo/gul_S1_eltcalc_P15
mkfifo fifo/gul_S1_summarycalc_P15
mkfifo fifo/gul_S1_pltcalc_P15

mkfifo fifo/gul_S1_summary_P16
mkfifo fifo/gul_S1_eltcalc_P16
mkfifo fifo/gul_S1_summarycalc_P16
mkfifo fifo/gul_S1_pltcalc_P16

mkfifo fifo/gul_S1_summary_P17
mkfifo fifo/gul_S1_eltcalc_P17
mkfifo fifo/gul_S1_summarycalc_P17
mkfifo fifo/gul_S1_pltcalc_P17

mkfifo fifo/gul_S1_summary_P18
mkfifo fifo/gul_S1_eltcalc_P18
mkfifo fifo/gul_S1_summarycalc_P18
mkfifo fifo/gul_S1_pltcalc_P18

mkfifo fifo/gul_S1_summary_P19
mkfifo fifo/gul_S1_eltcalc_P19
mkfifo fifo/gul_S1_summarycalc_P19
mkfifo fifo/gul_S1_pltcalc_P19

mkfifo fifo/gul_S1_summary_P20
mkfifo fifo/gul_S1_eltcalc_P20
mkfifo fifo/gul_S1_summarycalc_P20
mkfifo fifo/gul_S1_pltcalc_P20

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
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_pltcalc_P2

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_eltcalc_P3
mkfifo fifo/il_S1_summarycalc_P3
mkfifo fifo/il_S1_pltcalc_P3

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_eltcalc_P4
mkfifo fifo/il_S1_summarycalc_P4
mkfifo fifo/il_S1_pltcalc_P4

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_eltcalc_P5
mkfifo fifo/il_S1_summarycalc_P5
mkfifo fifo/il_S1_pltcalc_P5

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_eltcalc_P6
mkfifo fifo/il_S1_summarycalc_P6
mkfifo fifo/il_S1_pltcalc_P6

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_eltcalc_P7
mkfifo fifo/il_S1_summarycalc_P7
mkfifo fifo/il_S1_pltcalc_P7

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_eltcalc_P8
mkfifo fifo/il_S1_summarycalc_P8
mkfifo fifo/il_S1_pltcalc_P8

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_eltcalc_P9
mkfifo fifo/il_S1_summarycalc_P9
mkfifo fifo/il_S1_pltcalc_P9

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_eltcalc_P10
mkfifo fifo/il_S1_summarycalc_P10
mkfifo fifo/il_S1_pltcalc_P10

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_eltcalc_P11
mkfifo fifo/il_S1_summarycalc_P11
mkfifo fifo/il_S1_pltcalc_P11

mkfifo fifo/il_S1_summary_P12
mkfifo fifo/il_S1_eltcalc_P12
mkfifo fifo/il_S1_summarycalc_P12
mkfifo fifo/il_S1_pltcalc_P12

mkfifo fifo/il_S1_summary_P13
mkfifo fifo/il_S1_eltcalc_P13
mkfifo fifo/il_S1_summarycalc_P13
mkfifo fifo/il_S1_pltcalc_P13

mkfifo fifo/il_S1_summary_P14
mkfifo fifo/il_S1_eltcalc_P14
mkfifo fifo/il_S1_summarycalc_P14
mkfifo fifo/il_S1_pltcalc_P14

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_eltcalc_P15
mkfifo fifo/il_S1_summarycalc_P15
mkfifo fifo/il_S1_pltcalc_P15

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_eltcalc_P16
mkfifo fifo/il_S1_summarycalc_P16
mkfifo fifo/il_S1_pltcalc_P16

mkfifo fifo/il_S1_summary_P17
mkfifo fifo/il_S1_eltcalc_P17
mkfifo fifo/il_S1_summarycalc_P17
mkfifo fifo/il_S1_pltcalc_P17

mkfifo fifo/il_S1_summary_P18
mkfifo fifo/il_S1_eltcalc_P18
mkfifo fifo/il_S1_summarycalc_P18
mkfifo fifo/il_S1_pltcalc_P18

mkfifo fifo/il_S1_summary_P19
mkfifo fifo/il_S1_eltcalc_P19
mkfifo fifo/il_S1_summarycalc_P19
mkfifo fifo/il_S1_pltcalc_P19

mkfifo fifo/il_S1_summary_P20
mkfifo fifo/il_S1_eltcalc_P20
mkfifo fifo/il_S1_summarycalc_P20
mkfifo fifo/il_S1_pltcalc_P20



# --- Do insured loss computes ---

( eltcalc < fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 ) 2>> log/stderror.err & pid1=$!
( summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 ) 2>> log/stderror.err & pid2=$!
( pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 ) 2>> log/stderror.err & pid3=$!
( eltcalc -s < fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 ) 2>> log/stderror.err & pid4=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 ) 2>> log/stderror.err & pid5=$!
( pltcalc -s < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 ) 2>> log/stderror.err & pid6=$!
( eltcalc -s < fifo/il_S1_eltcalc_P3 > work/kat/il_S1_eltcalc_P3 ) 2>> log/stderror.err & pid7=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P3 > work/kat/il_S1_summarycalc_P3 ) 2>> log/stderror.err & pid8=$!
( pltcalc -s < fifo/il_S1_pltcalc_P3 > work/kat/il_S1_pltcalc_P3 ) 2>> log/stderror.err & pid9=$!
( eltcalc -s < fifo/il_S1_eltcalc_P4 > work/kat/il_S1_eltcalc_P4 ) 2>> log/stderror.err & pid10=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P4 > work/kat/il_S1_summarycalc_P4 ) 2>> log/stderror.err & pid11=$!
( pltcalc -s < fifo/il_S1_pltcalc_P4 > work/kat/il_S1_pltcalc_P4 ) 2>> log/stderror.err & pid12=$!
( eltcalc -s < fifo/il_S1_eltcalc_P5 > work/kat/il_S1_eltcalc_P5 ) 2>> log/stderror.err & pid13=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P5 > work/kat/il_S1_summarycalc_P5 ) 2>> log/stderror.err & pid14=$!
( pltcalc -s < fifo/il_S1_pltcalc_P5 > work/kat/il_S1_pltcalc_P5 ) 2>> log/stderror.err & pid15=$!
( eltcalc -s < fifo/il_S1_eltcalc_P6 > work/kat/il_S1_eltcalc_P6 ) 2>> log/stderror.err & pid16=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P6 > work/kat/il_S1_summarycalc_P6 ) 2>> log/stderror.err & pid17=$!
( pltcalc -s < fifo/il_S1_pltcalc_P6 > work/kat/il_S1_pltcalc_P6 ) 2>> log/stderror.err & pid18=$!
( eltcalc -s < fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 ) 2>> log/stderror.err & pid19=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P7 > work/kat/il_S1_summarycalc_P7 ) 2>> log/stderror.err & pid20=$!
( pltcalc -s < fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 ) 2>> log/stderror.err & pid21=$!
( eltcalc -s < fifo/il_S1_eltcalc_P8 > work/kat/il_S1_eltcalc_P8 ) 2>> log/stderror.err & pid22=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P8 > work/kat/il_S1_summarycalc_P8 ) 2>> log/stderror.err & pid23=$!
( pltcalc -s < fifo/il_S1_pltcalc_P8 > work/kat/il_S1_pltcalc_P8 ) 2>> log/stderror.err & pid24=$!
( eltcalc -s < fifo/il_S1_eltcalc_P9 > work/kat/il_S1_eltcalc_P9 ) 2>> log/stderror.err & pid25=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P9 > work/kat/il_S1_summarycalc_P9 ) 2>> log/stderror.err & pid26=$!
( pltcalc -s < fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 ) 2>> log/stderror.err & pid27=$!
( eltcalc -s < fifo/il_S1_eltcalc_P10 > work/kat/il_S1_eltcalc_P10 ) 2>> log/stderror.err & pid28=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P10 > work/kat/il_S1_summarycalc_P10 ) 2>> log/stderror.err & pid29=$!
( pltcalc -s < fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 ) 2>> log/stderror.err & pid30=$!
( eltcalc -s < fifo/il_S1_eltcalc_P11 > work/kat/il_S1_eltcalc_P11 ) 2>> log/stderror.err & pid31=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P11 > work/kat/il_S1_summarycalc_P11 ) 2>> log/stderror.err & pid32=$!
( pltcalc -s < fifo/il_S1_pltcalc_P11 > work/kat/il_S1_pltcalc_P11 ) 2>> log/stderror.err & pid33=$!
( eltcalc -s < fifo/il_S1_eltcalc_P12 > work/kat/il_S1_eltcalc_P12 ) 2>> log/stderror.err & pid34=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P12 > work/kat/il_S1_summarycalc_P12 ) 2>> log/stderror.err & pid35=$!
( pltcalc -s < fifo/il_S1_pltcalc_P12 > work/kat/il_S1_pltcalc_P12 ) 2>> log/stderror.err & pid36=$!
( eltcalc -s < fifo/il_S1_eltcalc_P13 > work/kat/il_S1_eltcalc_P13 ) 2>> log/stderror.err & pid37=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P13 > work/kat/il_S1_summarycalc_P13 ) 2>> log/stderror.err & pid38=$!
( pltcalc -s < fifo/il_S1_pltcalc_P13 > work/kat/il_S1_pltcalc_P13 ) 2>> log/stderror.err & pid39=$!
( eltcalc -s < fifo/il_S1_eltcalc_P14 > work/kat/il_S1_eltcalc_P14 ) 2>> log/stderror.err & pid40=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P14 > work/kat/il_S1_summarycalc_P14 ) 2>> log/stderror.err & pid41=$!
( pltcalc -s < fifo/il_S1_pltcalc_P14 > work/kat/il_S1_pltcalc_P14 ) 2>> log/stderror.err & pid42=$!
( eltcalc -s < fifo/il_S1_eltcalc_P15 > work/kat/il_S1_eltcalc_P15 ) 2>> log/stderror.err & pid43=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P15 > work/kat/il_S1_summarycalc_P15 ) 2>> log/stderror.err & pid44=$!
( pltcalc -s < fifo/il_S1_pltcalc_P15 > work/kat/il_S1_pltcalc_P15 ) 2>> log/stderror.err & pid45=$!
( eltcalc -s < fifo/il_S1_eltcalc_P16 > work/kat/il_S1_eltcalc_P16 ) 2>> log/stderror.err & pid46=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P16 > work/kat/il_S1_summarycalc_P16 ) 2>> log/stderror.err & pid47=$!
( pltcalc -s < fifo/il_S1_pltcalc_P16 > work/kat/il_S1_pltcalc_P16 ) 2>> log/stderror.err & pid48=$!
( eltcalc -s < fifo/il_S1_eltcalc_P17 > work/kat/il_S1_eltcalc_P17 ) 2>> log/stderror.err & pid49=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P17 > work/kat/il_S1_summarycalc_P17 ) 2>> log/stderror.err & pid50=$!
( pltcalc -s < fifo/il_S1_pltcalc_P17 > work/kat/il_S1_pltcalc_P17 ) 2>> log/stderror.err & pid51=$!
( eltcalc -s < fifo/il_S1_eltcalc_P18 > work/kat/il_S1_eltcalc_P18 ) 2>> log/stderror.err & pid52=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P18 > work/kat/il_S1_summarycalc_P18 ) 2>> log/stderror.err & pid53=$!
( pltcalc -s < fifo/il_S1_pltcalc_P18 > work/kat/il_S1_pltcalc_P18 ) 2>> log/stderror.err & pid54=$!
( eltcalc -s < fifo/il_S1_eltcalc_P19 > work/kat/il_S1_eltcalc_P19 ) 2>> log/stderror.err & pid55=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P19 > work/kat/il_S1_summarycalc_P19 ) 2>> log/stderror.err & pid56=$!
( pltcalc -s < fifo/il_S1_pltcalc_P19 > work/kat/il_S1_pltcalc_P19 ) 2>> log/stderror.err & pid57=$!
( eltcalc -s < fifo/il_S1_eltcalc_P20 > work/kat/il_S1_eltcalc_P20 ) 2>> log/stderror.err & pid58=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P20 > work/kat/il_S1_summarycalc_P20 ) 2>> log/stderror.err & pid59=$!
( pltcalc -s < fifo/il_S1_pltcalc_P20 > work/kat/il_S1_pltcalc_P20 ) 2>> log/stderror.err & pid60=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_eltcalc_P1 fifo/il_S1_summarycalc_P1 fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid61=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_eltcalc_P2 fifo/il_S1_summarycalc_P2 fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid62=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_eltcalc_P3 fifo/il_S1_summarycalc_P3 fifo/il_S1_pltcalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid63=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_eltcalc_P4 fifo/il_S1_summarycalc_P4 fifo/il_S1_pltcalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid64=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_eltcalc_P5 fifo/il_S1_summarycalc_P5 fifo/il_S1_pltcalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid65=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_eltcalc_P6 fifo/il_S1_summarycalc_P6 fifo/il_S1_pltcalc_P6 work/il_S1_summaryaalcalc/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid66=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_eltcalc_P7 fifo/il_S1_summarycalc_P7 fifo/il_S1_pltcalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid67=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_eltcalc_P8 fifo/il_S1_summarycalc_P8 fifo/il_S1_pltcalc_P8 work/il_S1_summaryaalcalc/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid68=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_eltcalc_P9 fifo/il_S1_summarycalc_P9 fifo/il_S1_pltcalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid69=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_eltcalc_P10 fifo/il_S1_summarycalc_P10 fifo/il_S1_pltcalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid70=$!
tee < fifo/il_S1_summary_P11 fifo/il_S1_eltcalc_P11 fifo/il_S1_summarycalc_P11 fifo/il_S1_pltcalc_P11 work/il_S1_summaryaalcalc/P11.bin work/il_S1_summaryleccalc/P11.bin > /dev/null & pid71=$!
tee < fifo/il_S1_summary_P12 fifo/il_S1_eltcalc_P12 fifo/il_S1_summarycalc_P12 fifo/il_S1_pltcalc_P12 work/il_S1_summaryaalcalc/P12.bin work/il_S1_summaryleccalc/P12.bin > /dev/null & pid72=$!
tee < fifo/il_S1_summary_P13 fifo/il_S1_eltcalc_P13 fifo/il_S1_summarycalc_P13 fifo/il_S1_pltcalc_P13 work/il_S1_summaryaalcalc/P13.bin work/il_S1_summaryleccalc/P13.bin > /dev/null & pid73=$!
tee < fifo/il_S1_summary_P14 fifo/il_S1_eltcalc_P14 fifo/il_S1_summarycalc_P14 fifo/il_S1_pltcalc_P14 work/il_S1_summaryaalcalc/P14.bin work/il_S1_summaryleccalc/P14.bin > /dev/null & pid74=$!
tee < fifo/il_S1_summary_P15 fifo/il_S1_eltcalc_P15 fifo/il_S1_summarycalc_P15 fifo/il_S1_pltcalc_P15 work/il_S1_summaryaalcalc/P15.bin work/il_S1_summaryleccalc/P15.bin > /dev/null & pid75=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_eltcalc_P16 fifo/il_S1_summarycalc_P16 fifo/il_S1_pltcalc_P16 work/il_S1_summaryaalcalc/P16.bin work/il_S1_summaryleccalc/P16.bin > /dev/null & pid76=$!
tee < fifo/il_S1_summary_P17 fifo/il_S1_eltcalc_P17 fifo/il_S1_summarycalc_P17 fifo/il_S1_pltcalc_P17 work/il_S1_summaryaalcalc/P17.bin work/il_S1_summaryleccalc/P17.bin > /dev/null & pid77=$!
tee < fifo/il_S1_summary_P18 fifo/il_S1_eltcalc_P18 fifo/il_S1_summarycalc_P18 fifo/il_S1_pltcalc_P18 work/il_S1_summaryaalcalc/P18.bin work/il_S1_summaryleccalc/P18.bin > /dev/null & pid78=$!
tee < fifo/il_S1_summary_P19 fifo/il_S1_eltcalc_P19 fifo/il_S1_summarycalc_P19 fifo/il_S1_pltcalc_P19 work/il_S1_summaryaalcalc/P19.bin work/il_S1_summaryleccalc/P19.bin > /dev/null & pid79=$!
tee < fifo/il_S1_summary_P20 fifo/il_S1_eltcalc_P20 fifo/il_S1_summarycalc_P20 fifo/il_S1_pltcalc_P20 work/il_S1_summaryaalcalc/P20.bin work/il_S1_summaryleccalc/P20.bin > /dev/null & pid80=$!

( summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P3 < fifo/il_P3 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P4 < fifo/il_P4 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P5 < fifo/il_P5 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P6 < fifo/il_P6 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P8 < fifo/il_P8 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P9 < fifo/il_P9 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P10 < fifo/il_P10 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P11 < fifo/il_P11 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P12 < fifo/il_P12 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P13 < fifo/il_P13 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P14 < fifo/il_P14 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P15 < fifo/il_P15 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P17 < fifo/il_P17 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P18 < fifo/il_P18 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P19 < fifo/il_P19 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P20 < fifo/il_P20 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---

( eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 ) 2>> log/stderror.err & pid81=$!
( summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 ) 2>> log/stderror.err & pid82=$!
( pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 ) 2>> log/stderror.err & pid83=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 ) 2>> log/stderror.err & pid84=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 ) 2>> log/stderror.err & pid85=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 ) 2>> log/stderror.err & pid86=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P3 > work/kat/gul_S1_eltcalc_P3 ) 2>> log/stderror.err & pid87=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 ) 2>> log/stderror.err & pid88=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P3 > work/kat/gul_S1_pltcalc_P3 ) 2>> log/stderror.err & pid89=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 ) 2>> log/stderror.err & pid90=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 ) 2>> log/stderror.err & pid91=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P4 > work/kat/gul_S1_pltcalc_P4 ) 2>> log/stderror.err & pid92=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P5 > work/kat/gul_S1_eltcalc_P5 ) 2>> log/stderror.err & pid93=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 ) 2>> log/stderror.err & pid94=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P5 > work/kat/gul_S1_pltcalc_P5 ) 2>> log/stderror.err & pid95=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P6 > work/kat/gul_S1_eltcalc_P6 ) 2>> log/stderror.err & pid96=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 ) 2>> log/stderror.err & pid97=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P6 > work/kat/gul_S1_pltcalc_P6 ) 2>> log/stderror.err & pid98=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 ) 2>> log/stderror.err & pid99=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 ) 2>> log/stderror.err & pid100=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P7 > work/kat/gul_S1_pltcalc_P7 ) 2>> log/stderror.err & pid101=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 ) 2>> log/stderror.err & pid102=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 ) 2>> log/stderror.err & pid103=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P8 > work/kat/gul_S1_pltcalc_P8 ) 2>> log/stderror.err & pid104=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 ) 2>> log/stderror.err & pid105=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 ) 2>> log/stderror.err & pid106=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P9 > work/kat/gul_S1_pltcalc_P9 ) 2>> log/stderror.err & pid107=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 ) 2>> log/stderror.err & pid108=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 ) 2>> log/stderror.err & pid109=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P10 > work/kat/gul_S1_pltcalc_P10 ) 2>> log/stderror.err & pid110=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P11 > work/kat/gul_S1_eltcalc_P11 ) 2>> log/stderror.err & pid111=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P11 > work/kat/gul_S1_summarycalc_P11 ) 2>> log/stderror.err & pid112=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P11 > work/kat/gul_S1_pltcalc_P11 ) 2>> log/stderror.err & pid113=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P12 > work/kat/gul_S1_eltcalc_P12 ) 2>> log/stderror.err & pid114=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P12 > work/kat/gul_S1_summarycalc_P12 ) 2>> log/stderror.err & pid115=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P12 > work/kat/gul_S1_pltcalc_P12 ) 2>> log/stderror.err & pid116=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P13 > work/kat/gul_S1_eltcalc_P13 ) 2>> log/stderror.err & pid117=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P13 > work/kat/gul_S1_summarycalc_P13 ) 2>> log/stderror.err & pid118=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P13 > work/kat/gul_S1_pltcalc_P13 ) 2>> log/stderror.err & pid119=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P14 > work/kat/gul_S1_eltcalc_P14 ) 2>> log/stderror.err & pid120=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P14 > work/kat/gul_S1_summarycalc_P14 ) 2>> log/stderror.err & pid121=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P14 > work/kat/gul_S1_pltcalc_P14 ) 2>> log/stderror.err & pid122=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P15 > work/kat/gul_S1_eltcalc_P15 ) 2>> log/stderror.err & pid123=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P15 > work/kat/gul_S1_summarycalc_P15 ) 2>> log/stderror.err & pid124=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P15 > work/kat/gul_S1_pltcalc_P15 ) 2>> log/stderror.err & pid125=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P16 > work/kat/gul_S1_eltcalc_P16 ) 2>> log/stderror.err & pid126=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P16 > work/kat/gul_S1_summarycalc_P16 ) 2>> log/stderror.err & pid127=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P16 > work/kat/gul_S1_pltcalc_P16 ) 2>> log/stderror.err & pid128=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P17 > work/kat/gul_S1_eltcalc_P17 ) 2>> log/stderror.err & pid129=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P17 > work/kat/gul_S1_summarycalc_P17 ) 2>> log/stderror.err & pid130=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P17 > work/kat/gul_S1_pltcalc_P17 ) 2>> log/stderror.err & pid131=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P18 > work/kat/gul_S1_eltcalc_P18 ) 2>> log/stderror.err & pid132=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P18 > work/kat/gul_S1_summarycalc_P18 ) 2>> log/stderror.err & pid133=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P18 > work/kat/gul_S1_pltcalc_P18 ) 2>> log/stderror.err & pid134=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P19 > work/kat/gul_S1_eltcalc_P19 ) 2>> log/stderror.err & pid135=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P19 > work/kat/gul_S1_summarycalc_P19 ) 2>> log/stderror.err & pid136=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P19 > work/kat/gul_S1_pltcalc_P19 ) 2>> log/stderror.err & pid137=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P20 > work/kat/gul_S1_eltcalc_P20 ) 2>> log/stderror.err & pid138=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P20 > work/kat/gul_S1_summarycalc_P20 ) 2>> log/stderror.err & pid139=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P20 > work/kat/gul_S1_pltcalc_P20 ) 2>> log/stderror.err & pid140=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid141=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid142=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_eltcalc_P3 fifo/gul_S1_summarycalc_P3 fifo/gul_S1_pltcalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid143=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_eltcalc_P4 fifo/gul_S1_summarycalc_P4 fifo/gul_S1_pltcalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid144=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_eltcalc_P5 fifo/gul_S1_summarycalc_P5 fifo/gul_S1_pltcalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid145=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_eltcalc_P6 fifo/gul_S1_summarycalc_P6 fifo/gul_S1_pltcalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid146=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_eltcalc_P7 fifo/gul_S1_summarycalc_P7 fifo/gul_S1_pltcalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid147=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_eltcalc_P8 fifo/gul_S1_summarycalc_P8 fifo/gul_S1_pltcalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid148=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_eltcalc_P9 fifo/gul_S1_summarycalc_P9 fifo/gul_S1_pltcalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid149=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_eltcalc_P10 fifo/gul_S1_summarycalc_P10 fifo/gul_S1_pltcalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid150=$!
tee < fifo/gul_S1_summary_P11 fifo/gul_S1_eltcalc_P11 fifo/gul_S1_summarycalc_P11 fifo/gul_S1_pltcalc_P11 work/gul_S1_summaryaalcalc/P11.bin work/gul_S1_summaryleccalc/P11.bin > /dev/null & pid151=$!
tee < fifo/gul_S1_summary_P12 fifo/gul_S1_eltcalc_P12 fifo/gul_S1_summarycalc_P12 fifo/gul_S1_pltcalc_P12 work/gul_S1_summaryaalcalc/P12.bin work/gul_S1_summaryleccalc/P12.bin > /dev/null & pid152=$!
tee < fifo/gul_S1_summary_P13 fifo/gul_S1_eltcalc_P13 fifo/gul_S1_summarycalc_P13 fifo/gul_S1_pltcalc_P13 work/gul_S1_summaryaalcalc/P13.bin work/gul_S1_summaryleccalc/P13.bin > /dev/null & pid153=$!
tee < fifo/gul_S1_summary_P14 fifo/gul_S1_eltcalc_P14 fifo/gul_S1_summarycalc_P14 fifo/gul_S1_pltcalc_P14 work/gul_S1_summaryaalcalc/P14.bin work/gul_S1_summaryleccalc/P14.bin > /dev/null & pid154=$!
tee < fifo/gul_S1_summary_P15 fifo/gul_S1_eltcalc_P15 fifo/gul_S1_summarycalc_P15 fifo/gul_S1_pltcalc_P15 work/gul_S1_summaryaalcalc/P15.bin work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid155=$!
tee < fifo/gul_S1_summary_P16 fifo/gul_S1_eltcalc_P16 fifo/gul_S1_summarycalc_P16 fifo/gul_S1_pltcalc_P16 work/gul_S1_summaryaalcalc/P16.bin work/gul_S1_summaryleccalc/P16.bin > /dev/null & pid156=$!
tee < fifo/gul_S1_summary_P17 fifo/gul_S1_eltcalc_P17 fifo/gul_S1_summarycalc_P17 fifo/gul_S1_pltcalc_P17 work/gul_S1_summaryaalcalc/P17.bin work/gul_S1_summaryleccalc/P17.bin > /dev/null & pid157=$!
tee < fifo/gul_S1_summary_P18 fifo/gul_S1_eltcalc_P18 fifo/gul_S1_summarycalc_P18 fifo/gul_S1_pltcalc_P18 work/gul_S1_summaryaalcalc/P18.bin work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid158=$!
tee < fifo/gul_S1_summary_P19 fifo/gul_S1_eltcalc_P19 fifo/gul_S1_summarycalc_P19 fifo/gul_S1_pltcalc_P19 work/gul_S1_summaryaalcalc/P19.bin work/gul_S1_summaryleccalc/P19.bin > /dev/null & pid159=$!
tee < fifo/gul_S1_summary_P20 fifo/gul_S1_eltcalc_P20 fifo/gul_S1_summarycalc_P20 fifo/gul_S1_pltcalc_P20 work/gul_S1_summaryaalcalc/P20.bin work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid160=$!

( summarycalc -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P4 < fifo/gul_P4 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P5 < fifo/gul_P5 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P7 < fifo/gul_P7 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P8 < fifo/gul_P8 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P9 < fifo/gul_P9 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P10 < fifo/gul_P10 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P11 < fifo/gul_P11 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P12 < fifo/gul_P12 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P13 < fifo/gul_P13 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P14 < fifo/gul_P14 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P15 < fifo/gul_P15 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P16 < fifo/gul_P16 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P17 < fifo/gul_P17 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P18 < fifo/gul_P18 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P19 < fifo/gul_P19 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P20 < fifo/gul_P20 ) 2>> log/stderror.err  &

( eve 1 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 > fifo/il_P1  ) 2>> log/stderror.err &
( eve 2 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P2 | fmcalc -a2 > fifo/il_P2  ) 2>> log/stderror.err &
( eve 3 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P3 | fmcalc -a2 > fifo/il_P3  ) 2>> log/stderror.err &
( eve 4 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P4 | fmcalc -a2 > fifo/il_P4  ) 2>> log/stderror.err &
( eve 5 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P5 | fmcalc -a2 > fifo/il_P5  ) 2>> log/stderror.err &
( eve 6 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P6 | fmcalc -a2 > fifo/il_P6  ) 2>> log/stderror.err &
( eve 7 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P7 | fmcalc -a2 > fifo/il_P7  ) 2>> log/stderror.err &
( eve 8 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P8 | fmcalc -a2 > fifo/il_P8  ) 2>> log/stderror.err &
( eve 9 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P9 | fmcalc -a2 > fifo/il_P9  ) 2>> log/stderror.err &
( eve 10 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P10 | fmcalc -a2 > fifo/il_P10  ) 2>> log/stderror.err &
( eve 11 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P11 | fmcalc -a2 > fifo/il_P11  ) 2>> log/stderror.err &
( eve 12 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P12 | fmcalc -a2 > fifo/il_P12  ) 2>> log/stderror.err &
( eve 13 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P13 | fmcalc -a2 > fifo/il_P13  ) 2>> log/stderror.err &
( eve 14 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P14 | fmcalc -a2 > fifo/il_P14  ) 2>> log/stderror.err &
( eve 15 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P15 | fmcalc -a2 > fifo/il_P15  ) 2>> log/stderror.err &
( eve 16 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P16 | fmcalc -a2 > fifo/il_P16  ) 2>> log/stderror.err &
( eve 17 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P17 | fmcalc -a2 > fifo/il_P17  ) 2>> log/stderror.err &
( eve 18 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P18 | fmcalc -a2 > fifo/il_P18  ) 2>> log/stderror.err &
( eve 19 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P19 | fmcalc -a2 > fifo/il_P19  ) 2>> log/stderror.err &
( eve 20 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P20 | fmcalc -a2 > fifo/il_P20  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 work/kat/il_S1_eltcalc_P11 work/kat/il_S1_eltcalc_P12 work/kat/il_S1_eltcalc_P13 work/kat/il_S1_eltcalc_P14 work/kat/il_S1_eltcalc_P15 work/kat/il_S1_eltcalc_P16 work/kat/il_S1_eltcalc_P17 work/kat/il_S1_eltcalc_P18 work/kat/il_S1_eltcalc_P19 work/kat/il_S1_eltcalc_P20 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 work/kat/il_S1_pltcalc_P11 work/kat/il_S1_pltcalc_P12 work/kat/il_S1_pltcalc_P13 work/kat/il_S1_pltcalc_P14 work/kat/il_S1_pltcalc_P15 work/kat/il_S1_pltcalc_P16 work/kat/il_S1_pltcalc_P17 work/kat/il_S1_pltcalc_P18 work/kat/il_S1_pltcalc_P19 work/kat/il_S1_pltcalc_P20 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 work/kat/il_S1_summarycalc_P11 work/kat/il_S1_summarycalc_P12 work/kat/il_S1_summarycalc_P13 work/kat/il_S1_summarycalc_P14 work/kat/il_S1_summarycalc_P15 work/kat/il_S1_summarycalc_P16 work/kat/il_S1_summarycalc_P17 work/kat/il_S1_summarycalc_P18 work/kat/il_S1_summarycalc_P19 work/kat/il_S1_summarycalc_P20 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 work/kat/gul_S1_eltcalc_P11 work/kat/gul_S1_eltcalc_P12 work/kat/gul_S1_eltcalc_P13 work/kat/gul_S1_eltcalc_P14 work/kat/gul_S1_eltcalc_P15 work/kat/gul_S1_eltcalc_P16 work/kat/gul_S1_eltcalc_P17 work/kat/gul_S1_eltcalc_P18 work/kat/gul_S1_eltcalc_P19 work/kat/gul_S1_eltcalc_P20 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 work/kat/gul_S1_pltcalc_P11 work/kat/gul_S1_pltcalc_P12 work/kat/gul_S1_pltcalc_P13 work/kat/gul_S1_pltcalc_P14 work/kat/gul_S1_pltcalc_P15 work/kat/gul_S1_pltcalc_P16 work/kat/gul_S1_pltcalc_P17 work/kat/gul_S1_pltcalc_P18 work/kat/gul_S1_pltcalc_P19 work/kat/gul_S1_pltcalc_P20 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 work/kat/gul_S1_summarycalc_P11 work/kat/gul_S1_summarycalc_P12 work/kat/gul_S1_summarycalc_P13 work/kat/gul_S1_summarycalc_P14 work/kat/gul_S1_summarycalc_P15 work/kat/gul_S1_summarycalc_P16 work/kat/gul_S1_summarycalc_P17 work/kat/gul_S1_summarycalc_P18 work/kat/gul_S1_summarycalc_P19 work/kat/gul_S1_summarycalc_P20 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6


( aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv ) 2>> log/stderror.err & lpid1=$!
( leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv ) 2>> log/stderror.err & lpid2=$!
( aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv ) 2>> log/stderror.err & lpid3=$!
( leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv ) 2>> log/stderror.err & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f fifo/*

check_complete
exit_handler
