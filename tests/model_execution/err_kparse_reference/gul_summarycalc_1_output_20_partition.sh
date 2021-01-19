#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
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

   ps f -g $sess_pid > log/subprocess_list
   PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk -F: '$1>$script_pid' | grep -v celery | grep -v python | grep -v $group_pid | grep -v run_ktools)
   echo "$PIDS_KILL" >> log/killout.txt
   kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
   exit $exit_code
}
trap exit_handler QUIT HUP INT KILL TERM ERR

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
mkfifo fifo/gul_S1_summarycalc_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summarycalc_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summarycalc_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summarycalc_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summarycalc_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summarycalc_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summarycalc_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_summarycalc_P8

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_summarycalc_P9

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_summarycalc_P10

mkfifo fifo/gul_S1_summary_P11
mkfifo fifo/gul_S1_summarycalc_P11

mkfifo fifo/gul_S1_summary_P12
mkfifo fifo/gul_S1_summarycalc_P12

mkfifo fifo/gul_S1_summary_P13
mkfifo fifo/gul_S1_summarycalc_P13

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_summarycalc_P14

mkfifo fifo/gul_S1_summary_P15
mkfifo fifo/gul_S1_summarycalc_P15

mkfifo fifo/gul_S1_summary_P16
mkfifo fifo/gul_S1_summarycalc_P16

mkfifo fifo/gul_S1_summary_P17
mkfifo fifo/gul_S1_summarycalc_P17

mkfifo fifo/gul_S1_summary_P18
mkfifo fifo/gul_S1_summarycalc_P18

mkfifo fifo/gul_S1_summary_P19
mkfifo fifo/gul_S1_summarycalc_P19

mkfifo fifo/gul_S1_summary_P20
mkfifo fifo/gul_S1_summarycalc_P20



# --- Do ground up loss computes ---

( summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 ) 2>> log/stderror.err & pid1=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 ) 2>> log/stderror.err & pid2=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 ) 2>> log/stderror.err & pid3=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 ) 2>> log/stderror.err & pid4=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 ) 2>> log/stderror.err & pid5=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 ) 2>> log/stderror.err & pid6=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 ) 2>> log/stderror.err & pid7=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 ) 2>> log/stderror.err & pid8=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 ) 2>> log/stderror.err & pid9=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 ) 2>> log/stderror.err & pid10=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P11 > work/kat/gul_S1_summarycalc_P11 ) 2>> log/stderror.err & pid11=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P12 > work/kat/gul_S1_summarycalc_P12 ) 2>> log/stderror.err & pid12=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P13 > work/kat/gul_S1_summarycalc_P13 ) 2>> log/stderror.err & pid13=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P14 > work/kat/gul_S1_summarycalc_P14 ) 2>> log/stderror.err & pid14=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P15 > work/kat/gul_S1_summarycalc_P15 ) 2>> log/stderror.err & pid15=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P16 > work/kat/gul_S1_summarycalc_P16 ) 2>> log/stderror.err & pid16=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P17 > work/kat/gul_S1_summarycalc_P17 ) 2>> log/stderror.err & pid17=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P18 > work/kat/gul_S1_summarycalc_P18 ) 2>> log/stderror.err & pid18=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P19 > work/kat/gul_S1_summarycalc_P19 ) 2>> log/stderror.err & pid19=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P20 > work/kat/gul_S1_summarycalc_P20 ) 2>> log/stderror.err & pid20=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_summarycalc_P1 > /dev/null & pid21=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_summarycalc_P2 > /dev/null & pid22=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_summarycalc_P3 > /dev/null & pid23=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_summarycalc_P4 > /dev/null & pid24=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_summarycalc_P5 > /dev/null & pid25=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_summarycalc_P6 > /dev/null & pid26=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_summarycalc_P7 > /dev/null & pid27=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_summarycalc_P8 > /dev/null & pid28=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_summarycalc_P9 > /dev/null & pid29=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_summarycalc_P10 > /dev/null & pid30=$!
tee < fifo/gul_S1_summary_P11 fifo/gul_S1_summarycalc_P11 > /dev/null & pid31=$!
tee < fifo/gul_S1_summary_P12 fifo/gul_S1_summarycalc_P12 > /dev/null & pid32=$!
tee < fifo/gul_S1_summary_P13 fifo/gul_S1_summarycalc_P13 > /dev/null & pid33=$!
tee < fifo/gul_S1_summary_P14 fifo/gul_S1_summarycalc_P14 > /dev/null & pid34=$!
tee < fifo/gul_S1_summary_P15 fifo/gul_S1_summarycalc_P15 > /dev/null & pid35=$!
tee < fifo/gul_S1_summary_P16 fifo/gul_S1_summarycalc_P16 > /dev/null & pid36=$!
tee < fifo/gul_S1_summary_P17 fifo/gul_S1_summarycalc_P17 > /dev/null & pid37=$!
tee < fifo/gul_S1_summary_P18 fifo/gul_S1_summarycalc_P18 > /dev/null & pid38=$!
tee < fifo/gul_S1_summary_P19 fifo/gul_S1_summarycalc_P19 > /dev/null & pid39=$!
tee < fifo/gul_S1_summary_P20 fifo/gul_S1_summarycalc_P20 > /dev/null & pid40=$!

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

( eve 1 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P1  ) 2>> log/stderror.err &
( eve 2 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P2  ) 2>> log/stderror.err &
( eve 3 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P3  ) 2>> log/stderror.err &
( eve 4 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P4  ) 2>> log/stderror.err &
( eve 5 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P5  ) 2>> log/stderror.err &
( eve 6 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P6  ) 2>> log/stderror.err &
( eve 7 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P7  ) 2>> log/stderror.err &
( eve 8 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P8  ) 2>> log/stderror.err &
( eve 9 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P9  ) 2>> log/stderror.err &
( eve 10 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P10  ) 2>> log/stderror.err &
( eve 11 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P11  ) 2>> log/stderror.err &
( eve 12 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P12  ) 2>> log/stderror.err &
( eve 13 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P13  ) 2>> log/stderror.err &
( eve 14 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P14  ) 2>> log/stderror.err &
( eve 15 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P15  ) 2>> log/stderror.err &
( eve 16 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P16  ) 2>> log/stderror.err &
( eve 17 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P17  ) 2>> log/stderror.err &
( eve 18 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P18  ) 2>> log/stderror.err &
( eve 19 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P19  ) 2>> log/stderror.err &
( eve 20 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P20  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40


# --- Do ground up loss kats ---

kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 work/kat/gul_S1_summarycalc_P11 work/kat/gul_S1_summarycalc_P12 work/kat/gul_S1_summarycalc_P13 work/kat/gul_S1_summarycalc_P14 work/kat/gul_S1_summarycalc_P15 work/kat/gul_S1_summarycalc_P16 work/kat/gul_S1_summarycalc_P17 work/kat/gul_S1_summarycalc_P18 work/kat/gul_S1_summarycalc_P19 work/kat/gul_S1_summarycalc_P20 > output/gul_S1_summarycalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f fifo/*

check_complete
exit_handler
